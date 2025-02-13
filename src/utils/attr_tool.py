from functools import reduce


class NoDefaultProvided(object):
    pass


def getattrd(obj, name: str, default=NoDefaultProvided):
    """
    Same as getattr(), but allows dot notation lookup
    Discussed in:
    http://stackoverflow.com/questions/11975781
    """

    try:
        return reduce(getattr, name.split("."), obj)
    except AttributeError:
        if default != NoDefaultProvided:
            return default
        raise


def setattrd(obj, name: str, value):
    index = name.rfind('.')
    if index != -1:
        attr_name = name[index + 1 :]
        belong = name[:index]
        setattr(getattrd(obj, belong), attr_name, value)
    else:
        setattr(obj, name, value)
