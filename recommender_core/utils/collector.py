from abc import ABCMeta
from functools import wraps
from types import FunctionType
from recommender_core.utils.singleton import Singleton


class DataCollector(Singleton):
    data: dict = {}


class ClassTracer(ABCMeta):
    collector = DataCollector()
    active = False

    def __new__(cls, class_name, bases, class_dict):
        if not cls.active:
            cls.active = True
            return type.__new__(cls, class_name, bases, class_dict)

        new_class_dict = {}
        for attr_name, attribute in class_dict.items():
            if isinstance(attribute, FunctionType) and not getattr(attribute, "_exclude_from_tracing", False):
                # replace it with a wrapped version
                attribute = ClassTracer.wrapper(attribute)
            new_class_dict[attr_name] = attribute

        for base in bases:
            if issubclass(base, ClassTracer):
                for attr_name in dir(base):
                    if attr_name in new_class_dict: # already overridden, skip
                        continue
                    attr = getattr(base, attr_name)
                    if isinstance(attr, FunctionType) and not getattr(attr, "_exclude_from_tracing", False):
                        new_class_dict[attr_name] = cls.wrapper(attr)

        return type.__new__(cls, class_name, bases, new_class_dict)

    @staticmethod
    def wrapper(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            # first arg of class methods is class instance -- aka "self"
            res = method(*args, **kwargs)
            if "traces" not in ClassTracer.collector.data:
                ClassTracer.collector.data["traces"] = []

            ClassTracer.collector.data["traces"].append({
                "class_name": args[0].__class__.__name__,
                "func_name": method.__name__,
                "args": ClassTracer.serialize_data(args[1:]),
                "kwargs": ClassTracer.serialize_data(kwargs),
                "result": ClassTracer.serialize_data(res)
            })
            return res
        return wrapped

    @staticmethod
    def exclude(method):
        """Decorator to mark functions that should not be traced."""
        method._exclude_from_tracing = True
        return method

    @staticmethod
    def serialize_data(data):
        def transform_value(item):
            if isinstance(item, type):
                return item.__name__
            elif isinstance(item, object):
                if hasattr(item, "__str__"):
                    return item.__str__()
                else:
                    return f"{item.__class__.__name__} instance"
            return item

        if isinstance(data, list) or isinstance(data, tuple):
            new_data = []
            for item in data:
                new_data.append(transform_value(item))

        elif isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                new_data[key] = transform_value(value)
        else:
            new_data = transform_value(data)
        return new_data