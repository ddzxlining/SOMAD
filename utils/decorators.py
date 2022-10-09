from functools import wraps
from time import time


def timeit(alternative_title=None):
    def wrap(f):
        @wraps(f)  # keeps the f.__name__ outside the wrapper
        def wrapped_f(*args, **kwargs):
            t0 = time()
            result = f(*args, **kwargs)
            ts = round(time() - t0, 3)
            title = alternative_title if alternative_title is not None else f.__name__
            print(" %s took: %f seconds" % (title, ts))
            return result
        return wrapped_f
    return wrap
