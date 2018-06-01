import inspect


def private_method(func):
    """decorator for making an instance method private"""

    def func_wrapper(*args, **kwargs):
        """decorator wrapper function"""
        outer_frame = inspect.stack()[1][0]
        if 'self' not in outer_frame.f_locals or outer_frame.f_locals['self'] is not args[0]:
            raise RuntimeError('%s.%s is a private method' % (args[0].__class__.__name__, func.__name__))
        func(*args, **kwargs)

    return func_wrapper
