class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        elif cls.__init__:
            # call __init__ function just the first time
            cls.__init__ = lambda self, *args, **kwargs: None   # __init__ function is "<lambda>" now.
        return cls._instance
