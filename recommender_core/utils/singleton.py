from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectKey:
    obj_type: object
    args: tuple
    kwargs: frozenset


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        obj_key = ObjectKey(obj_type=cls, args=args, kwargs=frozenset(kwargs.items()))
        if obj_key not in cls._instances:
            cls._instances[obj_key] = super().__new__(cls)
        return cls._instances[obj_key]
