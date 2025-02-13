from typing import Callable, Mapping, Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F


class SkipConnection(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    __call__: Callable[['SkipConnection', torch.Tensor], torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        in_query_dim: Optional[int],
        in_key_dim: Optional[int],
        in_val_dim: Optional[int],
        out_dim: int,
        only_score: bool = False,
    ) -> None:
        '''
        in_query_dim: None means q won't be linear projected.

        only_score: if only compute attention score.
        '''
        super().__init__()

        hidden_dim = out_dim // n_heads

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.in_query_dim = in_query_dim
        self.in_key_dim = in_key_dim
        self.in_val_dim = in_val_dim

        self.only_score = only_score

        # self.norm_factor = 1 / math.sqrt(hidden_dim)  # See Attention is all you need

        self.W_query = None
        self.W_key = None
        self.W_val = None
        self.W_out = None

        if in_query_dim is not None:
            self.W_query = nn.Parameter(torch.zeros(n_heads, in_query_dim, hidden_dim))
        if in_key_dim is not None:
            self.W_key = nn.Parameter(torch.zeros(n_heads, in_key_dim, hidden_dim))
        if in_val_dim is not None and not only_score:
            self.W_val = nn.Parameter(torch.zeros(n_heads, in_val_dim, hidden_dim))
        if not only_score:
            self.W_out = nn.Parameter(torch.zeros(n_heads, hidden_dim, out_dim))

    @staticmethod
    def compute(
        n_heads: int,
        hidden_dim: int,
        out_dim: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        W_query: Optional[torch.Tensor] = None,
        W_key: Optional[torch.Tensor] = None,
        W_val: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        with_norm: bool = True,
        only_score: bool = False,
    ) -> torch.Tensor:
        batch_size, n_query, in_que_dim = q.size()
        _, n_key, in_key_dim = k.size()

        if v is not None:
            _, n_val, in_val_dim = v.size()
            assert n_key == n_val

        if W_query is not None:
            qflat = q.contiguous().view(
                -1, in_que_dim
            )  # (batch_size * n_query, in_que_dim)
            shp_Q = (n_heads, batch_size, n_query, hidden_dim)

            # Calculate queries, (n_heads, batch_size, n_query, hidden_dim)
            Q = torch.matmul(qflat, W_query).view(shp_Q)
            # self.W_query: (n_heads, in_que_dim, hidden_dim)
            # Q_before_view: (n_heads, batch_size * n_query, hidden_dim)
        else:
            assert in_que_dim == out_dim
            Q = q.view(batch_size, n_query, hidden_dim, n_heads).permute(3, 0, 1, 2)

        # Calculate keys and values (n_heads, batch_size, n_key, hidden_dim)
        shp_KV = (n_heads, batch_size, n_key, hidden_dim)

        if W_key is not None:
            kflat = k.contiguous().view(
                -1, in_key_dim
            )  # (batch_size * n_key, in_key_dim)
            K = torch.matmul(kflat, W_key).view(shp_KV)
        else:
            assert in_key_dim == out_dim
            K = k.view(batch_size, n_key, hidden_dim, n_heads).permute(3, 0, 1, 2)

        if v is not None:
            if W_val is not None:
                vflat = v.contiguous().view(-1, in_val_dim)
                V = torch.matmul(vflat, W_val).view(shp_KV)
            else:
                assert in_val_dim == out_dim
                V = v.view(batch_size, n_val, hidden_dim, n_heads).permute(3, 0, 1, 2)

        # Calculate compatibility (n_heads, batch_size, n_query, n_key)
        compatibility = torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None, :, None, :]  # (batch_size, n_key)
                mask = mask.repeat(
                    n_heads, 1, n_query, 1
                )  # (n_heads, batch_size, n_query, n_key)
            elif mask.dim() == 3:
                mask = mask[None, :, :, :]  # (batch_size, n_query, n_key)
                mask = mask.repeat(
                    n_heads, 1, 1, 1
                )  # (n_heads, batch_size, n_query, n_key)
            else:
                raise NotImplementedError
            compatibility[mask] = -1e20

        if only_score and not with_norm:
            return compatibility

        norm_factor = 1 / math.sqrt(hidden_dim)
        compatibility = norm_factor * compatibility

        if only_score and with_norm:
            return compatibility

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)  # (n_heads, batch_size, n_query, hidden_dim)

        assert W_out is not None

        out = torch.mm(
            heads.permute(1, 2, 0, 3)  # (batch_size, n_query, n_heads, hidden_dim)
            .contiguous()
            .view(
                -1, n_heads * hidden_dim
            ),  # (batch_size * n_query, n_heads * hidden_dim)
            W_out.view(-1, out_dim),  # (n_heads * hidden_dim, out_dim)
        ).view(batch_size, n_query, out_dim)

        return out

    __call__: Callable[..., torch.Tensor]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        with_norm: bool = True,
    ) -> torch.Tensor:
        '''
        q: (batch_size, n_query, in_que_dim)

        k: (batch_size, n_key, in_key_dim)

        v: (batch_size, n_key, in_val_dim)

        mask: (batch_size, n_key)
        '''
        if self.only_score:  # calculate attention score
            assert v is None

        return self.compute(
            self.n_heads,
            self.hidden_dim,
            self.out_dim,
            q,
            k,
            v,
            self.W_query,
            self.W_key,
            self.W_val,
            self.W_out,
            mask,
            with_norm,
            self.only_score,
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()
        self.MHA = MultiHeadAttention(
            n_heads, input_dim, input_dim, input_dim, input_dim
        )

    __call__: Callable[..., torch.Tensor]

    def forward(self, q: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.MHA(q, q, q, **kwargs)


class MultiHeadSelfAttentionScore(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()
        self.MHA = MultiHeadAttention(n_heads, input_dim, input_dim, None, input_dim)

    __call__: Callable[..., torch.Tensor]

    def forward(self, q: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.MHA(q, q, **kwargs)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        feed_forward_dim: int = 64,
        embedding_dim: int = 64,
        output_dim: int = 1,
        p_dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        self.ReLU = nn.ReLU(inplace=True)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.ReLU(self.fc1(input))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result)
        return result


class Normalization(nn.Module):
    def __init__(self, input_dim: int, normalization: str) -> None:
        super().__init__()

        self.normalization = normalization

        if self.normalization != 'layer':
            normalizer_class = {'batch': nn.BatchNorm1d, 'instance': nn.InstanceNorm1d}[
                normalization
            ]
            self.normalizer = normalizer_class(input_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    # def init_parameters(self) -> None:
    #    for param in self.parameters():
    #        stdv = 1.0 / math.sqrt(param.size(-1))
    #        param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                input.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif self.normalization == 'batch':
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif self.normalization == 'instance':
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert False, "Unknown normalizer type"


class FFNormSubLayer(nn.Module):
    def __init__(
        self, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.FF = (
            nn.Sequential(
                nn.Linear(input_dim, feed_forward_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, input_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(input_dim, input_dim)
        )

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class EncodingBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        feed_forward_hidden: int,
        normalization: str,
    ) -> None:
        super().__init__()
        self.skip_MHA = SkipConnection(MultiHeadSelfAttention(n_heads, input_dim))
        self.norm = Normalization(input_dim, normalization)
        self.FFnorm = FFNormSubLayer(input_dim, feed_forward_hidden, normalization)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.FFnorm(self.norm(self.skip_MHA(input)))


class GraphEncoder_CVRP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_hidden: int,
        n_heads: int,
        n_blocks: int,
        normalization: str,
    ) -> None:
        super().__init__()

        self.customer_embedder = nn.Linear(3, embedding_dim)
        self.depot_embedder = nn.Linear(2, embedding_dim)

        if n_blocks > 0:
            self.encoding_blocks: Optional[nn.Module] = nn.Sequential(
                *(
                    EncodingBlock(
                        n_heads, embedding_dim, feed_forward_hidden, normalization
                    )
                    for _ in range(n_blocks)
                )
            )
        else:
            self.encoding_blocks = None

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        input: graph[batch_size, problem_size, 3(x,y,d)], depot(first node)'s demand=0

        return: (batch_size, problem_size, embedding_dim)
        '''
        customers = input[:, 1:, :]
        depot = input[:, :1, :2]

        cus_emb = self.customer_embedder(customers)
        dep_emb = self.depot_embedder(depot)

        init_embedding = torch.cat((dep_emb, cus_emb), 1)

        if self.encoding_blocks is not None:
            embedding = self.encoding_blocks(init_embedding)
        else:
            embedding = init_embedding

        return embedding


class GraphEncoder_TSP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_hidden: int,
        n_heads: int,
        n_blocks: int,
        normalization: str,
    ) -> None:
        super().__init__()

        self.customer_embedder = nn.Linear(2, embedding_dim)

        if n_blocks > 0:
            self.encoding_blocks: Optional[nn.Module] = nn.Sequential(
                *(
                    EncodingBlock(
                        n_heads, embedding_dim, feed_forward_hidden, normalization
                    )
                    for _ in range(n_blocks)
                )
            )
        else:
            self.encoding_blocks = None

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        input: graph[batch_size, problem_size, 2(x,y)]

        return: (batch_size, problem_size, embedding_dim)
        '''
        init_embedding = self.customer_embedder(input)

        if self.encoding_blocks is not None:
            embedding = self.encoding_blocks(init_embedding)
        else:
            embedding = init_embedding

        return embedding


# https://github.com/kaist-silab/equity-transformer/blob/main/nets/positional_encoding.py
class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    __call__: Callable[['PostionalEncoding', int, int], torch.Tensor]

    def forward(self, batch_size: int, seq_len: int) -> torch.Tensor:
        # self.encoding
        # [max_len = 512, d_model = 512]

        ### batch_size, seq_len = batch_size, seq_len
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class VehicleContextEncoder_CVRP(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int) -> None:
        super().__init__()

        # self.vehicle_proj = nn.Linear(embedding_dim + 2, embedding_dim + 2)
        self.selfMHA = MultiHeadSelfAttention(n_heads, embedding_dim + 2)

    __call__: Callable[..., torch.Tensor]

    def forward(
        self,
        current_node_embed: torch.Tensor,
        remain_capacity: torch.Tensor,
        current_distance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        current_node_embed: (batch_size, vehicle_num, embedding_dim)

        remain_capacity: (batch_size, vehicle_num)

        current_distance: (batch_size, vehicle_num)

        mask: (batch_size, vehicle_num)
        '''

        context = torch.cat(
            (
                current_node_embed,
                remain_capacity.unsqueeze(-1),
                current_distance.unsqueeze(-1),
            ),
            2,
        )

        return self.selfMHA(context, mask=mask)


class VehicleContextEncoder_TSP(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int) -> None:
        super().__init__()

        # self.vehicle_proj = nn.Linear(embedding_dim + 1, embedding_dim + 1)
        self.selfMHA = MultiHeadSelfAttention(n_heads, embedding_dim + 1)

    __call__: Callable[..., torch.Tensor]

    def forward(
        self,
        current_node_embed: torch.Tensor,
        current_distance: torch.Tensor,
    ) -> torch.Tensor:
        '''
        current_node_embed: (batch_size, vehicle_num, embedding_dim)

        remain_capacity: (batch_size, vehicle_num)

        current_distance: (batch_size, vehicle_num)
        '''
        context = torch.cat(
            (current_node_embed, current_distance.unsqueeze(-1)),
            2,
        )
        return self.selfMHA(context)


class AM_Decoder_Precompute(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.enc_key_proj = nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.enc_val_proj = nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.enc_key_for_glimpse_proj = nn.Parameter(
            torch.zeros(embedding_dim, embedding_dim)
        )

    @staticmethod
    def compute(
        graph_embedding: torch.Tensor,
        enc_key_proj: torch.Tensor,
        enc_val_proj: torch.Tensor,
        enc_key_for_glimpse_proj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj_key = F.linear(graph_embedding, enc_key_proj)
        proj_val = F.linear(graph_embedding, enc_val_proj)
        proj_key_for_glimpse = F.linear(graph_embedding, enc_key_for_glimpse_proj)

        return proj_key, proj_val, proj_key_for_glimpse

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def forward(
        self, graph_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute(
            graph_embedding,
            self.enc_key_proj,
            self.enc_val_proj,
            self.enc_key_for_glimpse_proj,
        )


class AM_Decoder(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int, context_dim: int) -> None:
        super().__init__()

        self.first_MHA = MultiHeadAttention(
            n_heads, context_dim, None, None, embedding_dim
        )

        self.second_SHA_score = MultiHeadAttention(
            1, None, None, None, embedding_dim, only_score=True
        )

    @staticmethod
    def compute(
        context: torch.Tensor,
        proj_key: torch.Tensor,
        proj_val: torch.Tensor,
        proj_key_for_glimpse: torch.Tensor,
        first_MHA: Callable[..., torch.Tensor],
        second_SHA_score: Callable[..., torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        C: float = 10,
        select_type: str = 'sample',
        fixed_next_node: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        cross_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, ref_num, _ = proj_key.size()
        query_num = context.size(1)

        glimpse = first_MHA(context, proj_key, proj_val, mask=mask)

        compatibility = (
            torch.tanh(second_SHA_score(glimpse, proj_key_for_glimpse)) * C
        ).squeeze(0) / temperature
        # (batch_size, query_num, ref_num)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).repeat(1, query_num, 1)
            compatibility[mask] = -1e20

        if not cross_prob:
            prob = F.softmax(compatibility, dim=-1)
            log_p = F.log_softmax(
                compatibility, dim=-1
            )  # (batch_size, query_num, ref_num)

            if fixed_next_node is None:
                if select_type == 'greedy':
                    next_node = (
                        prob.view(-1, ref_num).max(1)[1].reshape(batch_size, query_num)
                    )  # (batch_size, query_num)
                elif select_type == 'sample':
                    next_node = (
                        prob.view(-1, ref_num)
                        .multinomial(1)
                        .reshape(batch_size, query_num)
                    )  # (batch_size, query_num)
                elif select_type == 'distrib':
                    return prob, compatibility
                else:
                    raise NotImplementedError
            else:
                next_node = fixed_next_node

            arange = torch.arange(batch_size * query_num)
            sel_log_p = log_p.view(-1, ref_num)[arange, next_node.view(-1)].reshape(
                batch_size, query_num
            )  # (batch_size, query_num)
        else:
            prob = F.softmax(compatibility.view(batch_size, -1), dim=-1)
            log_p = F.log_softmax(
                compatibility.view(batch_size, -1), dim=-1
            )  # (batch_size, query_num*ref_num)

            if fixed_next_node is None:
                if select_type == 'greedy':
                    next_node_sel = prob.max(1)[1]  # (batch_size,)
                elif select_type == 'sample':
                    next_node_sel = prob.multinomial(1).view(-1)  # (batch_size,)
                elif select_type == 'distrib':
                    return prob, compatibility
                else:
                    raise NotImplementedError

                vehicle_sel = torch.div(
                    next_node_sel, ref_num, rounding_mode='trunc'
                )  # (batch_size,)
                customer_sel = next_node_sel % ref_num

                next_node = torch.stack((vehicle_sel, customer_sel), dim=1)
            else:
                next_node = fixed_next_node  # (batch_size, 2)

            arange = torch.arange(batch_size)
            sel_log_p = log_p[arange, next_node_sel]  # (batch_size,)

        return next_node, sel_log_p

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self,
        context: torch.Tensor,
        proj_key: torch.Tensor,
        proj_val: torch.Tensor,
        proj_key_for_glimpse: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        C: float = 10,
        select_type: str = 'sample',
        fixed_next_node: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        cross_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        proj_key: projected key, (batch_size, key_num/val_num, embedding_dim)

        context: query, (batch_size, query_num, context_dim)

        mask: bool(batch_size, problem_size) or (batch_size, query_num, problem_size)

        return: [next_nodes(batch_size, query_num or 2), log_prob(batch_size, query_num or None)]
                or [prob_distrib(batch_size, query_num(, or *)ref_num),
                    compatibility(batch_size, query_num, ref_num)]
        '''
        return self.compute(
            context,
            proj_key,
            proj_val,
            proj_key_for_glimpse,
            self.first_MHA,
            self.second_SHA_score,
            mask,
            C,
            select_type,
            fixed_next_node,
            temperature,
            cross_prob,
        )


class AM_Decoder_HyperNetwork_PrefSetter(nn.Module):
    def __init__(
        self,
        pref_dim: int,
        n_heads: int,
        embedding_dim: int,
        context_dim: int,
    ) -> None:
        super().__init__()

        self.pref_dim = pref_dim
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.hidden_dim = embedding_dim // n_heads

        hyper_hidden_dim = 256
        hyper_output_dim = 5 * pref_dim

        para_name_list = [
            'first_MHA_Wq',
            'enc_key_proj',
            'enc_val_proj',
            'first_MHA_Wo',
            'enc_key_for_glimpse_proj',
        ]

        self.hyper_fc = nn.Sequential(
            *(
                nn.Linear(pref_dim, hyper_hidden_dim),
                nn.Linear(hyper_hidden_dim, hyper_hidden_dim),
                nn.Linear(hyper_hidden_dim, hyper_output_dim),
            )
        )

        self.hyper_proj_to_para = nn.ModuleDict()
        for para_name in para_name_list:
            self.hyper_proj_to_para[para_name] = nn.Linear(
                self.pref_dim,
                (
                    embedding_dim * context_dim
                    if para_name == 'first_MHA_Wq'
                    else embedding_dim * embedding_dim
                ),
                bias=False,
            )

    __call__: Callable[..., Mapping[str, torch.Tensor]]

    def forward(self, pref: torch.Tensor) -> Mapping[str, torch.Tensor]:
        '''
        pref: (pref_dim,)
        '''
        mid_embd = self.hyper_fc(pref)

        ret_paras = {}

        for i, para_name in enumerate(self.hyper_proj_to_para):
            if para_name == 'first_MHA_Wq':
                reshape_to: Tuple[int, ...] = (
                    self.n_heads,
                    self.context_dim,
                    self.hidden_dim,
                )

            elif para_name == 'first_MHA_Wo':
                reshape_to = (self.n_heads, self.hidden_dim, self.embedding_dim)

            elif para_name in (
                'enc_key_proj',
                'enc_val_proj',
                'enc_key_for_glimpse_proj',
            ):
                reshape_to = (self.embedding_dim, self.embedding_dim)

            else:
                raise NotImplementedError

            ret_paras[para_name] = self.hyper_proj_to_para[para_name](
                mid_embd[i * self.pref_dim : (i + 1) * self.pref_dim]
            ).reshape(reshape_to)

        return ret_paras
