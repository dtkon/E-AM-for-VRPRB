from typing import Any, Callable, List, Optional, Tuple, Union, cast
import math
import abc
import torch
from torch import nn

from route_solver import cal_distance

from . import layers


class AM_Actor(abc.ABC, nn.Module):
    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor, Any]]

    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


class AM_preActor(AM_Actor):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_hidden: int,
        n_heads: int,
        n_blocks_graph: int,
        normalization: str,
        problem: str = 'cvrp',
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // n_heads

        if problem == 'cvrp':
            self.graph_encoder: Union[
                layers.GraphEncoder_CVRP, layers.GraphEncoder_TSP
            ] = layers.GraphEncoder_CVRP(
                embedding_dim,
                feed_forward_hidden,
                n_heads,
                n_blocks_graph,
                normalization,
            )
        elif problem == 'tsp':
            self.graph_encoder = layers.GraphEncoder_TSP(
                embedding_dim,
                feed_forward_hidden,
                n_heads,
                n_blocks_graph,
                normalization,
            )
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(
        self, problems: torch.Tensor, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d) or 2(x,y)]
        '''

        # encoder
        graph_embed = self.graph_encoder(
            problems
        )  # (batch_size, problem_size, embedding_dim)
        avg_graph_embed = graph_embed.mean(1)  # (batch_size, embedding_dim)

        return graph_embed, avg_graph_embed


class AM_CVRP_Actor(AM_Actor):
    def __init__(self, n_heads: int, embedding_dim: int) -> None:
        super().__init__()

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        self.decoder = layers.AM_Decoder(n_heads, embedding_dim, 2 * embedding_dim + 1)

        self.init_parameters()

    def forward(
        self,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        decoder_type: str = 'sample',
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d)]

        decoder_type: sample or greedy

        return: [actions(batch_size, node_indexes), log_prob(batch_size,)]
        '''
        batch_size, problem_size, _ = problems.size()

        batch_arange = torch.arange(batch_size, device=problems.device)

        proj_key, proj_val, proj_key_for_glimpse = self.precomputing(graph_embed)

        # track solution
        actions = torch.zeros((batch_size, 1), device=problems.device, dtype=torch.long)
        log_prob_list = []

        remain_capacity = torch.ones((batch_size, 1), device=problems.device)
        assign_count = torch.ones_like(actions)
        current_node_embed = graph_embed[:, 0, :]

        # prepare mask
        mask, selected_mask, ref_mask = AM_CVRP_Actor.prepare_mask(
            batch_size, problem_size, problems.device
        )

        while True:
            ### prepare context
            context = torch.cat(
                [avg_graph_embed, current_node_embed, remain_capacity],
                1,
            )
            ###

            ### decoder forward
            next_node, log_prob = self.decoder(
                context.unsqueeze(1),
                proj_key,
                proj_val,
                proj_key_for_glimpse,
                mask,
                select_type=decoder_type,
            )
            next_node = next_node.view(-1)
            ###

            ### process return
            actions = torch.cat((actions, next_node.view(-1, 1)), dim=1)
            log_prob_list.append(log_prob.reshape(-1))
            ###

            ### if all done
            assign_count += (next_node != 0).view(-1, 1)
            if torch.all(assign_count == problem_size):
                actions = torch.cat(
                    (
                        actions,
                        torch.zeros(
                            (batch_size, 1), device=problems.device, dtype=torch.long
                        ),
                    ),
                    dim=1,
                )
                # self.decoder.clear_cache()
                break
            ###

            ### calculate new context
            remain_capacity -= problems[batch_arange, next_node, 2].view(batch_size, -1)
            remain_capacity[next_node == 0] = 1

            current_node_embed = graph_embed[batch_arange, next_node, :]
            ###

            ### update mask
            demand_too_large = (
                remain_capacity.repeat(1, problem_size) < problems[:, :, 2]
            )  # (batch_size, problem_size)

            done = (assign_count == problem_size).view(-1)  # (batch,)

            mask = AM_CVRP_Actor.update_mask(
                batch_arange,
                next_node,
                selected_mask,
                ref_mask,
                demand_too_large,
                done,
            )
            ###

        log_probs = torch.stack(log_prob_list, 1).sum(1)  # (batch_size,)

        return actions, log_probs, None

    @staticmethod
    def prepare_mask(
        batch_size: int,
        problem_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        return: mask(used in decoder), selected_mask(used to be updated), done_mask(fixed for reference)
        '''
        selected_mask = torch.zeros(
            (batch_size, problem_size), device=device, dtype=torch.bool
        )
        selected_mask[:, 0] = True
        mask = selected_mask.clone()
        done_mask = torch.ones((1, problem_size), device=device, dtype=torch.bool)
        done_mask[:, 0] = False

        return mask, selected_mask, done_mask

    @staticmethod
    def update_mask(
        batch_arange: torch.Tensor,
        next_node: torch.Tensor,
        selected_mask: torch.Tensor,
        done_mask: torch.Tensor,
        demand_too_large: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        '''
        selected_mask will be changed in-place
        '''
        selected_mask[batch_arange, next_node] = True  # (batch_size, problem_size)

        return_to_depot = next_node == 0  # (batch,)

        mask = selected_mask.clone()
        mask[demand_too_large] = True

        mask[:, 0] = False
        mask[:, 0][return_to_depot] = True

        mask[done, :] = done_mask

        return mask


class AM_CVRP_1D_LimitVeh_Actor(AM_Actor):
    def __init__(self, n_heads_dec: int, embedding_dim: int) -> None:
        super().__init__()

        self.n_heads_dec = n_heads_dec
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // n_heads_dec

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        self.decoder = layers.AM_Decoder(
            n_heads_dec, embedding_dim, 2 * embedding_dim + 1
        )

        self.init_parameters()

    def forward(
        self,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        enc_key_proj = self.precomputing.enc_key_proj
        enc_val_proj = self.precomputing.enc_val_proj
        enc_key_for_glimpse_proj = self.precomputing.enc_key_for_glimpse_proj
        first_MHA_Wq = cast(torch.Tensor, self.decoder.first_MHA.W_query)
        first_MHA_Wo = cast(torch.Tensor, self.decoder.first_MHA.W_out)

        return self.compute(
            enc_key_proj,
            enc_val_proj,
            enc_key_for_glimpse_proj,
            first_MHA_Wq,
            first_MHA_Wo,
            problems,
            graph_embed,
            avg_graph_embed,
            vehicle_num,
            decoder_type,
        )

    def compute(
        self,
        enc_key_proj: torch.Tensor,
        enc_val_proj: torch.Tensor,
        enc_key_for_glimpse_proj: torch.Tensor,
        first_MHA_Wq: torch.Tensor,
        first_MHA_Wo: torch.Tensor,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d)]

        decoder_type: sample or greedy

        return: [actions(batch_size, node_indexes), log_prob(batch_size,)]
        '''
        batch_size, problem_size, _ = problems.size()

        batch_arange = torch.arange(batch_size, device=problems.device)

        proj_key, proj_val, proj_key_for_glimpse = layers.AM_Decoder_Precompute.compute(
            graph_embed, enc_key_proj, enc_val_proj, enc_key_for_glimpse_proj
        )

        first_MHA = (
            lambda context, proj_key, proj_val, mask: layers.MultiHeadAttention.compute(
                self.n_heads_dec,
                self.hidden_dim,
                self.embedding_dim,
                context,
                proj_key,
                proj_val,
                mask=mask,
                W_query=first_MHA_Wq,
                W_out=first_MHA_Wo,
            )
        )

        second_SHA_score = (
            lambda glimpse, proj_key_for_glimpse: layers.MultiHeadAttention.compute(
                1,
                self.embedding_dim,
                self.embedding_dim,
                glimpse,
                proj_key_for_glimpse,
                only_score=True,
            )
        )

        # track solution
        actions = torch.zeros((batch_size, 1), device=problems.device, dtype=torch.long)
        log_prob_list = []

        remain_capacity = torch.ones((batch_size, 1), device=problems.device)
        assign_count = torch.ones_like(actions)
        current_node_embed = graph_embed[:, 0, :]

        used_vehicle_num = torch.zeros_like(actions)

        # prepare mask
        mask, selected_mask, ref_mask = AM_CVRP_Actor.prepare_mask(
            batch_size, problem_size, problems.device
        )

        while True:
            ### prepare context
            context = torch.cat(
                [avg_graph_embed, current_node_embed, remain_capacity],
                1,
            )
            ###

            ### decoder forward
            next_node, log_prob = layers.AM_Decoder.compute(
                context.unsqueeze(1),
                proj_key,
                proj_val,
                proj_key_for_glimpse,
                first_MHA,
                second_SHA_score,
                mask,
                select_type=decoder_type,
            )
            next_node = next_node.view(-1)
            ###

            ### process return
            actions = torch.cat((actions, next_node.view(-1, 1)), dim=1)
            log_prob_list.append(log_prob.reshape(-1))
            ###

            new_vehicle_go = (actions[:, -2] == 0) & (actions[:, -1] != 0)
            used_vehicle_num[new_vehicle_go] += 1

            ### if all done
            assign_count += (next_node != 0).view(-1, 1)
            if torch.all(assign_count == problem_size):
                actions = torch.cat(
                    (
                        actions,
                        torch.zeros(
                            (batch_size, 1), device=problems.device, dtype=torch.long
                        ),
                    ),
                    dim=1,
                )
                break
            ###

            ### calculate new context
            remain_capacity -= problems[batch_arange, next_node, 2].view(batch_size, -1)
            remain_capacity[next_node == 0] = 1

            current_node_embed = graph_embed[batch_arange, next_node, :]
            ###

            ### update mask
            demand_too_large = (
                remain_capacity.repeat(1, problem_size) < problems[:, :, 2]
            )  # (batch_size, problem_size)

            done = (assign_count == problem_size).view(-1)  # (batch,)

            mask = AM_CVRP_1D_LimitVeh_Actor.update_mask(
                batch_arange,
                next_node,
                selected_mask,
                ref_mask,
                demand_too_large,
                done,
                problems,
                vehicle_num,
                remain_capacity,
                used_vehicle_num,
            )
            ###

        log_probs = torch.stack(log_prob_list, 1).sum(1)  # (batch_size,)

        infeasible = (used_vehicle_num > vehicle_num).squeeze(1)

        return actions, log_probs, infeasible

    @staticmethod
    def update_mask(
        batch_arange: torch.Tensor,
        next_node: torch.Tensor,
        selected_mask: torch.Tensor,
        done_mask: torch.Tensor,
        demand_too_large: torch.Tensor,
        done: torch.Tensor,
        problems: torch.Tensor,
        vehicle_num: int,
        remain_capacity: torch.Tensor,
        used_vehicle_num: torch.Tensor,
    ) -> torch.Tensor:
        '''
        selected_mask will be changed in-place
        '''
        selected_mask[batch_arange, next_node] = True  # (batch_size, problem_size)

        return_to_depot = next_node == 0  # (batch,)

        mask = selected_mask.clone()
        mask[demand_too_large] = True

        mask[:, 0] = False
        mask[:, 0][return_to_depot] = True

        # vehicle num control
        rest_demand = torch.where(
            selected_mask,
            torch.tensor(0, dtype=problems.dtype, device=problems.device),
            problems[:, :, 2],
        ).sum(
            1
        )  # (batch_size,)
        remain_vehicle_num = vehicle_num - used_vehicle_num
        other_rest_capacity = torch.where(
            remain_vehicle_num < 0,
            torch.tensor(
                0, dtype=remain_vehicle_num.dtype, device=remain_vehicle_num.device
            ),
            remain_vehicle_num,
        )

        should_keep_work = (
            ~return_to_depot
            & ~(demand_too_large[:, 1:].all(-1))
            & (other_rest_capacity.squeeze(1) < rest_demand)
        )

        mask[:, 0][should_keep_work] = True
        ###

        mask[done, :] = done_mask

        return mask


class AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor(AM_Actor):
    def __init__(
        self,
        n_heads_dec: int,
        n_heads_veh: int,
        embedding_dim: int,
        with_vehicle_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.n_heads_dec = n_heads_dec
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // n_heads_dec

        self.with_vehicle_encoder = with_vehicle_encoder

        if with_vehicle_encoder:
            self.vehicle_encoder = layers.VehicleContextEncoder_CVRP(
                n_heads_veh, embedding_dim
            )

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        self.decoder = layers.AM_Decoder(
            n_heads_dec,
            embedding_dim,
            embedding_dim * 2 + 2,
        )

        self.init_parameters()

    def forward(
        self,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        unfinish_complete: bool = False,
        will_re_tsp: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        enc_key_proj = self.precomputing.enc_key_proj
        enc_val_proj = self.precomputing.enc_val_proj
        enc_key_for_glimpse_proj = self.precomputing.enc_key_for_glimpse_proj
        first_MHA_Wq = cast(torch.Tensor, self.decoder.first_MHA.W_query)
        first_MHA_Wo = cast(torch.Tensor, self.decoder.first_MHA.W_out)

        return self.compute(
            enc_key_proj,
            enc_val_proj,
            enc_key_for_glimpse_proj,
            first_MHA_Wq,
            first_MHA_Wo,
            problems,
            graph_embed,
            avg_graph_embed,
            vehicle_num,
            decoder_type,
            unfinish_complete,
            will_re_tsp,
        )

    def compute(
        self,
        enc_key_proj: torch.Tensor,
        enc_val_proj: torch.Tensor,
        enc_key_for_glimpse_proj: torch.Tensor,
        first_MHA_Wq: torch.Tensor,
        first_MHA_Wo: torch.Tensor,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        unfinish_complete: bool = False,
        will_re_tsp: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, problem_size, _ = problems.size()

        problems_aggregate = problems.repeat_interleave(vehicle_num, 0)
        # problems_vehicle_dim = problems.unsqueeze(1).repeat(1, vehicle_num, 1, 1)

        proj_key, proj_val, proj_key_for_glimpse = layers.AM_Decoder_Precompute.compute(
            graph_embed, enc_key_proj, enc_val_proj, enc_key_for_glimpse_proj
        )

        first_MHA = (
            lambda context, proj_key, proj_val, mask: layers.MultiHeadAttention.compute(
                self.n_heads_dec,
                self.hidden_dim,
                self.embedding_dim,
                context,
                proj_key,
                proj_val,
                mask=mask,
                W_query=first_MHA_Wq,
                W_out=first_MHA_Wo,
            )
        )

        second_SHA_score = (
            lambda glimpse, proj_key_for_glimpse: layers.MultiHeadAttention.compute(
                1,
                self.embedding_dim,
                self.embedding_dim,
                glimpse,
                proj_key_for_glimpse,
                only_score=True,
            )
        )

        batch_arange = torch.arange(batch_size, device=problems.device)

        # track solution
        actions = torch.zeros(
            (batch_size, vehicle_num, 1), device=problems.device, dtype=torch.long
        )
        log_prob_list: List[torch.Tensor] = []

        avg_graph_embed = avg_graph_embed.unsqueeze(1).repeat(
            1, vehicle_num, 1
        )  # (batch_size, vehicle_num, embedding_dim)
        current_node_embed = graph_embed[:, 0:1, :].repeat(
            1, vehicle_num, 1
        )  # (batch_size, vehicle_num, embedding_dim)
        current_customers = actions[:, :, -1].clone()  # (batch_size, vehicle_num)

        remain_capacity = torch.ones(
            (batch_size, vehicle_num), dtype=torch.float, device=problems.device
        )
        current_distance_diff = torch.zeros_like(remain_capacity)

        assign_count = torch.ones(
            (batch_size), dtype=torch.long, device=problems.device
        )

        # prepare mask
        mask, selected_mask, done_mask = (
            AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor.prepare_mask(
                batch_size, problem_size, vehicle_num, problems.device
            )
        )

        unable_finish_flag = torch.zeros(
            (batch_size), dtype=torch.bool, device=problems.device
        )

        vehicle_encounter_unable_serve = torch.zeros(
            (batch_size, vehicle_num), dtype=torch.bool, device=problems.device
        )

        while True:
            ### prepare context
            vehicle_mask = mask.all(-1)  # (batch_size, vehicle_num)

            if self.with_vehicle_encoder:
                emb_context = self.vehicle_encoder(
                    current_node_embed,
                    remain_capacity,
                    current_distance_diff,
                    mask=vehicle_mask,
                )
                context = torch.cat((avg_graph_embed, emb_context), 2)
            else:
                context = torch.cat(
                    (
                        avg_graph_embed,
                        current_node_embed,
                        remain_capacity.unsqueeze(-1),
                        current_distance_diff.unsqueeze(-1),
                    ),
                    2,
                )
            ###

            ### decoder forward
            next_node, log_prob = layers.AM_Decoder.compute(
                context,
                proj_key,
                proj_val,
                proj_key_for_glimpse,
                first_MHA,
                second_SHA_score,
                mask,
                select_type=decoder_type,
                cross_prob=True,
            )
            ###

            ### process return
            vehicle_sel = next_node[:, 0]
            customer_sel = next_node[:, 1]

            current_customers = actions[:, :, -1].clone()
            current_customers[batch_arange, vehicle_sel] = customer_sel

            actions = torch.cat((actions, current_customers.unsqueeze(2)), dim=2)

            if unfinish_complete:
                log_prob[unable_finish_flag.clone()] = (
                    0  # cut prob backward. clone to avoid gradient inplace modify.
                )
            log_prob_list.append(log_prob.view(-1))
            ###

            ### if all done
            assign_count += customer_sel != 0
            if torch.all(assign_count == problem_size):
                actions = torch.cat(
                    (
                        actions,
                        torch.zeros(
                            (batch_size, vehicle_num, 1),
                            device=problems.device,
                            dtype=torch.long,
                        ),
                    ),
                    dim=2,
                )
                break
            ###

            ### calculate new context
            remain_capacity[batch_arange, vehicle_sel] -= problems[
                batch_arange, customer_sel, 2
            ]

            # update selected_mask first
            selected_mask[batch_arange, :, customer_sel] = (
                True  # (batch_size, vehicle_num, problem_size)
            )

            demand_too_large_or_have_visited = (
                remain_capacity.unsqueeze(2).repeat(1, 1, problem_size) + 1e-6
                < problems[:, :, 2].unsqueeze(1)
            ) | selected_mask  # (batch_size, vehicle_num, problem_size)

            vehicle_encounter_unable_serve = torch.all(
                demand_too_large_or_have_visited, 2
            ) & (
                current_customers != 0
            )  # (batch_size, vehicle_num). Notice there may be multiple vehicles encounter at same time.
            if torch.any(vehicle_encounter_unable_serve):
                current_customers[vehicle_encounter_unable_serve] = 0

                actions = torch.cat((actions, current_customers.unsqueeze(2)), dim=2)

                # disallow multi-trip
                remain_capacity[vehicle_encounter_unable_serve] = 0

                vehicle_mask[vehicle_encounter_unable_serve] = True

            current_node_embed = graph_embed.gather(
                1, current_customers.unsqueeze(2).repeat(1, 1, graph_embed.size(2))
            )

            if not will_re_tsp:
                actions_aggregate = actions.view(batch_size * vehicle_num, -1)
                current_distance = cal_distance(
                    problems_aggregate, actions_aggregate
                ).view(batch_size, vehicle_num)
                masked_distance = torch.where(
                    vehicle_mask,
                    torch.tensor(
                        float('inf'),
                        dtype=current_distance.dtype,
                        device=current_distance.device,
                    ),
                    current_distance,
                )

                current_distance_diff = (
                    current_distance - masked_distance.min(1, keepdim=True)[0]
                )
                current_distance_diff = torch.where(
                    vehicle_mask,
                    torch.tensor(
                        0,
                        dtype=current_distance_diff.dtype,
                        device=current_distance_diff.device,
                    ),
                    current_distance_diff,
                )
                # current_distance_diff = current_distance / current_distance.sum(
                #    1, keepdim=True
                # )
            else:
                # each vehicle use max distance to depot as diff

                actions_aggregate = actions.view(batch_size * vehicle_num, -1)
                # problems_aggregate: (batch_size*vehicle_num, problem_size, 3)
                actions_aggregate = actions_aggregate.unsqueeze(-1).repeat(1, 1, 2)
                seq_nodes = torch.gather(
                    problems_aggregate[:, :, :2], 1, actions_aggregate
                )  # (batch_size*vehicle_num, node_indexes, 2)
                depot_aggr = problems_aggregate[:, 0, :2].unsqueeze(1)
                diff_nodes = seq_nodes - depot_aggr
                nodes_dis = (
                    diff_nodes.norm(dim=-1).max(-1)[0].view(batch_size, vehicle_num)
                )
                current_distance_diff = nodes_dis

            ###

            ### update mask
            done = (assign_count == problem_size).view(-1)  # (batch,)

            mask = AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor.update_mask(
                batch_arange,
                customer_sel,
                selected_mask,
                done_mask,
                demand_too_large_or_have_visited,
                done,
            )  # (batch_size, vehicle_num, problem_size)

            unable_finish = mask.all(-1).all(-1) & ~done  # (batch_size,)
            if torch.any(unable_finish):
                if unfinish_complete:
                    # replenishment
                    remain_capacity[unable_finish, :] = 1
                    unable_finish_flag[unable_finish] = True

                    # renew mask
                    demand_too_large = remain_capacity.unsqueeze(2).repeat(
                        1, 1, problem_size
                    ) < problems[:, :, 2].unsqueeze(
                        1
                    )  # (batch_size, vehicle_num, problem_size)
                    mask = AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor.update_mask(
                        batch_arange,
                        customer_sel,
                        selected_mask,
                        done_mask,
                        demand_too_large,
                        done,
                    )  # (batch_size, vehicle_num, problem_size)
                else:
                    unable_finish_flag[unable_finish] = True
                    assign_count[unable_finish] = problem_size
                    done[unable_finish] = True
                    mask[unable_finish] = done_mask
            ###

        log_probs = torch.stack(log_prob_list, 1).sum(1)  # (batch_size,)

        return actions, log_probs, unable_finish_flag

    @staticmethod
    def prepare_mask(
        batch_size: int,
        problem_size: int,
        vehicle_num: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        return: mask(used in decoder), selected_mask(used to be updated), done_mask(fixed for reference)
        '''
        selected_mask = torch.zeros(
            (batch_size, vehicle_num, problem_size), device=device, dtype=torch.bool
        )
        selected_mask[:, :, 0] = True
        mask = selected_mask.clone()
        done_mask = torch.ones(
            (1, vehicle_num, problem_size), device=device, dtype=torch.bool
        )
        done_mask[:, 0, 0] = False

        return mask, selected_mask, done_mask

    @staticmethod
    def update_mask(
        batch_arange: torch.Tensor,
        customer_sel: torch.Tensor,
        selected_mask: torch.Tensor,
        done_mask: torch.Tensor,
        demand_too_large: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        '''
        selected_mask will be changed in-place
        '''
        selected_mask[batch_arange, :, customer_sel] = (
            True  # (batch_size, vehicle_num, problem_size)
        )

        mask = selected_mask.clone()
        mask[demand_too_large] = True

        mask[done, :] = done_mask

        return mask

    @staticmethod
    def should_keepWork_earlyBack(
        remain_capacity: torch.Tensor,
        problems: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        '''

        remain_capacity: (batch_size, vehicle_num)

        problems: (batch_size, problem_size, 3)

        selected_mask: (batch_size, vehicle_num, problem_size)

        return: bool(batch_size, vehicle_num)
        '''

        rest_demand = torch.where(selected_mask[:, 0, :], 0, problems[:, :, 2]).sum(
            1
        )  # (batch_size,)
        other_rest_capacity = (
            remain_capacity.sum(1, keepdim=True) - remain_capacity
        )  # (batch_size, vehicle_num)

        return other_rest_capacity < rest_demand.unsqueeze(1)


class AM_CVRP_2D_LimitVeh_MultiTrip_Actor(AM_Actor):
    def __init__(
        self,
        n_heads_dec: int,
        n_heads_veh: int,
        embedding_dim: int,
        with_vehicle_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.n_heads_dec = n_heads_dec
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // n_heads_dec

        self.with_vehicle_encoder = with_vehicle_encoder

        if with_vehicle_encoder:
            self.vehicle_encoder = layers.VehicleContextEncoder_CVRP(
                n_heads_veh, embedding_dim
            )

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        self.decoder = layers.AM_Decoder(
            n_heads_dec,
            embedding_dim,
            embedding_dim * 2 + 2,
        )

        self.init_parameters()

    def forward(
        self,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:

        enc_key_proj = self.precomputing.enc_key_proj
        enc_val_proj = self.precomputing.enc_val_proj
        enc_key_for_glimpse_proj = self.precomputing.enc_key_for_glimpse_proj
        first_MHA_Wq = cast(torch.Tensor, self.decoder.first_MHA.W_query)
        first_MHA_Wo = cast(torch.Tensor, self.decoder.first_MHA.W_out)

        return self.compute(
            enc_key_proj,
            enc_val_proj,
            enc_key_for_glimpse_proj,
            first_MHA_Wq,
            first_MHA_Wo,
            problems,
            graph_embed,
            avg_graph_embed,
            vehicle_num,
            decoder_type,
        )

    def compute(
        self,
        enc_key_proj: torch.Tensor,
        enc_val_proj: torch.Tensor,
        enc_key_for_glimpse_proj: torch.Tensor,
        first_MHA_Wq: torch.Tensor,
        first_MHA_Wo: torch.Tensor,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:

        batch_size, problem_size, _ = problems.size()

        problems_aggregate = problems.repeat_interleave(vehicle_num, 0)
        # problems_vehicle_dim = problems.unsqueeze(1).repeat(1, vehicle_num, 1, 1)

        proj_key, proj_val, proj_key_for_glimpse = layers.AM_Decoder_Precompute.compute(
            graph_embed, enc_key_proj, enc_val_proj, enc_key_for_glimpse_proj
        )

        first_MHA = (
            lambda context, proj_key, proj_val, mask: layers.MultiHeadAttention.compute(
                self.n_heads_dec,
                self.hidden_dim,
                self.embedding_dim,
                context,
                proj_key,
                proj_val,
                mask=mask,
                W_query=first_MHA_Wq,
                W_out=first_MHA_Wo,
            )
        )

        second_SHA_score = (
            lambda glimpse, proj_key_for_glimpse: layers.MultiHeadAttention.compute(
                1,
                self.embedding_dim,
                self.embedding_dim,
                glimpse,
                proj_key_for_glimpse,
                only_score=True,
            )
        )

        batch_arange = torch.arange(batch_size, device=problems.device)

        # track solution
        actions = torch.zeros(
            (batch_size, vehicle_num, 1), device=problems.device, dtype=torch.long
        )
        log_prob_list: List[torch.Tensor] = []

        avg_graph_embed = avg_graph_embed.unsqueeze(1).repeat(
            1, vehicle_num, 1
        )  # (batch_size, vehicle_num, embedding_dim)
        current_node_embed = graph_embed[:, 0:1, :].repeat(
            1, vehicle_num, 1
        )  # (batch_size, vehicle_num, embedding_dim)
        current_customers = actions[:, :, -1].clone()

        remain_capacity = torch.ones(
            (batch_size, vehicle_num), dtype=torch.float, device=problems.device
        )
        current_distance_diff = torch.zeros_like(remain_capacity)

        assign_count = torch.ones(
            (batch_size), dtype=torch.long, device=problems.device
        )

        # prepare mask
        mask, selected_mask, done_mask = (
            AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor.prepare_mask(
                batch_size, problem_size, vehicle_num, problems.device
            )
        )

        while True:
            ### prepare context
            if self.with_vehicle_encoder:
                emb_context = self.vehicle_encoder(
                    current_node_embed,
                    remain_capacity,
                    current_distance_diff,
                )
                context = torch.cat((avg_graph_embed, emb_context), 2)
            else:
                context = torch.cat(
                    (
                        avg_graph_embed,
                        current_node_embed,
                        remain_capacity.unsqueeze(-1),
                        current_distance_diff.unsqueeze(-1),
                    ),
                    2,
                )

            ###

            ### decoder forward
            next_node, log_prob = layers.AM_Decoder.compute(
                context,
                proj_key,
                proj_val,
                proj_key_for_glimpse,
                first_MHA,
                second_SHA_score,
                mask,
                select_type=decoder_type,
                cross_prob=True,
            )
            ###

            ### process return
            vehicle_sel = next_node[:, 0]
            customer_sel = next_node[:, 1]

            current_customers = actions[:, :, -1].clone()
            current_customers[batch_arange, vehicle_sel] = customer_sel

            actions = torch.cat((actions, current_customers.unsqueeze(2)), dim=2)
            log_prob_list.append(log_prob.view(-1))
            ###

            ### if all done
            assign_count += customer_sel != 0
            if torch.all(assign_count == problem_size):
                actions = torch.cat(
                    (
                        actions,
                        torch.zeros(
                            (batch_size, vehicle_num, 1),
                            device=problems.device,
                            dtype=torch.long,
                        ),
                    ),
                    dim=2,
                )
                break
            ###

            ### calculate new context
            remain_capacity[batch_arange, vehicle_sel] -= problems[
                batch_arange, customer_sel, 2
            ]

            # allow multi-trip replenishment
            remain_capacity[current_customers == 0] = 1
            # remain_capacity[batch_arange, vehicle_sel][customer_sel == 0] = 1 # fail to modify

            current_node_embed[batch_arange, vehicle_sel, :] = graph_embed[
                batch_arange, customer_sel, :
            ]

            actions_aggregate = actions.view(batch_size * vehicle_num, -1)
            current_distance = cal_distance(problems_aggregate, actions_aggregate).view(
                batch_size, vehicle_num
            )

            current_distance_diff = (
                current_distance - current_distance.min(1, keepdim=True)[0]
            )
            # current_distance_diff = current_distance / current_distance.sum(
            #    1, keepdim=True
            # )

            ###

            ### update mask
            demand_too_large = remain_capacity.unsqueeze(2).repeat(
                1, 1, problem_size
            ) < problems[:, :, 2].unsqueeze(
                1
            )  # (batch_size, vehicle_num, problem_size)

            done = (assign_count == problem_size).view(-1)  # (batch,)

            mask = AM_CVRP_2D_LimitVeh_MultiTrip_Actor.update_mask(
                batch_arange,
                customer_sel,
                current_customers,
                selected_mask,
                done_mask,
                demand_too_large,
                done,
            )  # (batch_size, vehicle_num, problem_size)
            ###

        log_probs = torch.stack(log_prob_list, 1).sum(1)  # (batch_size,)

        return actions, log_probs, None

    @staticmethod
    def update_mask(
        batch_arange: torch.Tensor,
        customer_sel: torch.Tensor,
        current_customers: torch.Tensor,
        selected_mask: torch.Tensor,
        done_mask: torch.Tensor,
        demand_too_large: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        '''
        selected_mask will be changed in-place
        '''
        selected_mask[batch_arange, :, customer_sel] = (
            True  # (batch_size, vehicle_num, problem_size)
        )

        return_to_depot = current_customers == 0  # (batch_size, vehicle_num)

        mask = selected_mask.clone()
        mask[demand_too_large] = True

        mask[:, :, 0] = False
        mask[:, :, 0][return_to_depot] = True

        mask[done, :] = done_mask

        return mask


class AM_TSP_2D_LimitVeh_Actor(AM_Actor):
    def __init__(
        self,
        n_heads_dec: int,
        n_heads_veh: int,
        embedding_dim: int,
        with_vehicle_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.n_heads_dec = n_heads_dec
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // n_heads_dec

        self.with_vehicle_encoder = with_vehicle_encoder

        if with_vehicle_encoder:
            self.vehicle_encoder = layers.VehicleContextEncoder_TSP(
                n_heads_veh, embedding_dim
            )

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        self.decoder = layers.AM_Decoder(
            n_heads_dec,
            embedding_dim,
            embedding_dim * 2 + 1,
        )

        self.init_parameters()

    def forward(
        self,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:

        enc_key_proj = self.precomputing.enc_key_proj
        enc_val_proj = self.precomputing.enc_val_proj
        enc_key_for_glimpse_proj = self.precomputing.enc_key_for_glimpse_proj
        first_MHA_Wq = cast(torch.Tensor, self.decoder.first_MHA.W_query)
        first_MHA_Wo = cast(torch.Tensor, self.decoder.first_MHA.W_out)

        return self.compute(
            enc_key_proj,
            enc_val_proj,
            enc_key_for_glimpse_proj,
            first_MHA_Wq,
            first_MHA_Wo,
            problems,
            graph_embed,
            avg_graph_embed,
            vehicle_num,
            decoder_type,
        )

    def compute(
        self,
        enc_key_proj: torch.Tensor,
        enc_val_proj: torch.Tensor,
        enc_key_for_glimpse_proj: torch.Tensor,
        first_MHA_Wq: torch.Tensor,
        first_MHA_Wo: torch.Tensor,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:

        batch_size, problem_size, _ = problems.size()

        problems_aggregate = problems.repeat_interleave(vehicle_num, 0)
        # problems_vehicle_dim = problems.unsqueeze(1).repeat(1, vehicle_num, 1, 1)

        proj_key, proj_val, proj_key_for_glimpse = layers.AM_Decoder_Precompute.compute(
            graph_embed, enc_key_proj, enc_val_proj, enc_key_for_glimpse_proj
        )

        first_MHA = (
            lambda context, proj_key, proj_val, mask: layers.MultiHeadAttention.compute(
                self.n_heads_dec,
                self.hidden_dim,
                self.embedding_dim,
                context,
                proj_key,
                proj_val,
                mask=mask,
                W_query=first_MHA_Wq,
                W_out=first_MHA_Wo,
            )
        )

        second_SHA_score = (
            lambda glimpse, proj_key_for_glimpse: layers.MultiHeadAttention.compute(
                1,
                self.embedding_dim,
                self.embedding_dim,
                glimpse,
                proj_key_for_glimpse,
                only_score=True,
            )
        )

        batch_arange = torch.arange(batch_size, device=problems.device)

        # track solution
        actions = torch.zeros(
            (batch_size, vehicle_num, 1), device=problems.device, dtype=torch.long
        )
        log_prob_list: List[torch.Tensor] = []

        avg_graph_embed = avg_graph_embed.unsqueeze(1).repeat(
            1, vehicle_num, 1
        )  # (batch_size, vehicle_num, embedding_dim)
        current_node_embed = graph_embed[:, 0:1, :].repeat(
            1, vehicle_num, 1
        )  # (batch_size, vehicle_num, embedding_dim)
        current_customers = actions[:, :, -1].clone()

        current_distance_diff = torch.zeros(
            (batch_size, vehicle_num), device=problems.device
        )

        assign_count = torch.ones((batch_size), device=problems.device)

        # prepare mask
        mask, selected_mask, done_mask = (
            AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor.prepare_mask(
                batch_size, problem_size, vehicle_num, problems.device
            )
        )  # (batch_size, vehicle_num, problem_size)

        while True:
            ### prepare context
            if self.with_vehicle_encoder:
                emb_context = self.vehicle_encoder(
                    current_node_embed, current_distance_diff
                )
                context = torch.cat((avg_graph_embed, emb_context), 2)
            else:
                context = torch.cat(
                    (
                        avg_graph_embed,
                        current_node_embed,
                        current_distance_diff.unsqueeze(-1),
                    ),
                    2,
                )

            ###

            ### decoder forward
            next_node, log_prob = layers.AM_Decoder.compute(
                context,
                proj_key,
                proj_val,
                proj_key_for_glimpse,
                first_MHA,
                second_SHA_score,
                mask,
                select_type=decoder_type,
                cross_prob=True,
            )
            ###

            ### process return
            vehicle_sel = next_node[:, 0]
            customer_sel = next_node[:, 1]

            current_customers = actions[:, :, -1].clone()
            current_customers[batch_arange, vehicle_sel] = customer_sel

            actions = torch.cat((actions, current_customers.unsqueeze(2)), dim=2)
            log_prob_list.append(log_prob.view(-1))
            ###

            ### if all done
            assign_count += customer_sel != 0
            if torch.all(assign_count == problem_size):
                actions = torch.cat(
                    (
                        actions,
                        torch.zeros(
                            (batch_size, vehicle_num, 1),
                            device=problems.device,
                            dtype=torch.long,
                        ),
                    ),
                    dim=2,
                )
                break
            ###

            current_node_embed[batch_arange, vehicle_sel, :] = graph_embed[
                batch_arange, customer_sel, :
            ]

            actions_aggregate = actions.view(batch_size * vehicle_num, -1)
            current_distance = cal_distance(problems_aggregate, actions_aggregate).view(
                batch_size, vehicle_num
            )

            current_distance_diff = (
                current_distance - current_distance.min(1, keepdim=True)[0]
            )
            # current_distance_diff = current_distance / current_distance.sum(
            #    1, keepdim=True
            # )

            ###

            done = (assign_count == problem_size).view(-1)  # (batch,)

            mask = AM_TSP_2D_LimitVeh_Actor.update_mask(
                batch_arange,
                customer_sel,
                selected_mask,
                done_mask,
                done,
            )
            ###

        log_probs = torch.stack(log_prob_list, 1).sum(1)  # (batch_size,)

        return actions, log_probs, None

    @staticmethod
    def update_mask(
        batch_arange: torch.Tensor,
        customer_sel: torch.Tensor,
        selected_mask: torch.Tensor,
        done_mask: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        '''
        selected_mask will be changed in-place
        '''
        selected_mask[batch_arange, :, customer_sel] = (
            True  # (batch_size, vehicle_num, problem_size)
        )

        mask = selected_mask.clone()

        mask[done, :] = done_mask

        return mask


class AM_MO_Actor(AM_Actor):
    def __init__(
        self,
        pref_dim: int,
        n_heads_dec: int,
        n_heads_veh: int,
        embedding_dim: int,
        with_vehicle_encoder: bool = True,
        problem: str = 'cvrp',
        use_2D: bool = True,
    ) -> None:
        super().__init__()

        self.n_heads_dec = n_heads_dec
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // n_heads_dec

        self.with_vehicle_encoder = with_vehicle_encoder

        if use_2D:
            if problem == 'cvrp':
                self.vehicle_encoder: Optional[
                    Union[
                        layers.VehicleContextEncoder_CVRP,
                        layers.VehicleContextEncoder_TSP,
                    ]
                ] = (
                    layers.VehicleContextEncoder_CVRP(n_heads_veh, embedding_dim)
                    if with_vehicle_encoder
                    else None
                )
                context_dim = embedding_dim * 2 + 2
                self.compute = AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor.compute
            elif problem == 'tsp':
                self.vehicle_encoder = (
                    layers.VehicleContextEncoder_TSP(n_heads_veh, embedding_dim)
                    if with_vehicle_encoder
                    else None
                )
                context_dim = embedding_dim * 2 + 1
                self.compute = AM_TSP_2D_LimitVeh_Actor.compute  # type: ignore
            else:
                raise NotImplementedError
        else:
            context_dim = embedding_dim * 2 + 1
            self.compute = AM_CVRP_1D_LimitVeh_Actor.compute  # type: ignore

        self.hyper_decoder = layers.AM_Decoder_HyperNetwork_PrefSetter(
            pref_dim, n_heads_dec, embedding_dim, context_dim
        )

        self.init_parameters()

    def forward(
        self,
        pref: torch.Tensor,
        problems: torch.Tensor,
        graph_embed: torch.Tensor,
        avg_graph_embed: torch.Tensor,
        vehicle_num: int,
        decoder_type: str = 'sample',
        will_re_tsp: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        '''
        graph_embed: (batch_size, problem_size, embedding_dim)

        decoder_type: sample or greedy

        return: [actions(batch_size, vehicle_num, node_indexes), log_prob(batch_size,)]
        actions have duplicate, no wait.
        '''

        # hyper parameters
        hyper_para = self.hyper_decoder(pref)
        enc_key_proj = hyper_para['enc_key_proj']
        enc_val_proj = hyper_para['enc_val_proj']
        enc_key_for_glimpse_proj = hyper_para['enc_key_for_glimpse_proj']
        first_MHA_Wq = hyper_para['first_MHA_Wq']
        first_MHA_Wo = hyper_para['first_MHA_Wo']

        return self.compute(
            self,  # type: ignore
            enc_key_proj,
            enc_val_proj,
            enc_key_for_glimpse_proj,
            first_MHA_Wq,
            first_MHA_Wo,
            problems,
            graph_embed,
            avg_graph_embed,
            vehicle_num,
            decoder_type,
            will_re_tsp=will_re_tsp,
        )
