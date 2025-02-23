from abc import ABCMeta
from functools import wraps
from types import FunctionType
from recommender_core.utils.singleton import Singleton


class DataCollector(Singleton):
    data: dict = {}


class ClassTracer(ABCMeta):
    collector = DataCollector()

    def __new__(cls, class_name, bases, class_dict):
        new_class_dict = {}
        for attributeName, attribute in class_dict.items():
            if isinstance(attribute, FunctionType):
                # replace it with a wrapped version
                attribute = ClassTracer.wrapper(attribute)
            new_class_dict[attributeName] = attribute
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
                "args": args[1:],
                "kwargs": kwargs,
                "result": res
            })
            return res
        return wrapped

    @staticmethod
    def exclude(method):
        """Decorator to mark functions that should not be traced."""
        method._exclude_from_tracing = True
        return method