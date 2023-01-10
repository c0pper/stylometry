import time
from typing import Tuple, Callable
from functools import wraps


def timeit(func) -> Callable:
    """
    * We decorate the function with timeit decorator
    * decorator makes note of start time
    * then executes the function
    * decorator marks end time
    * calculates time difference and prints the time taken for the function

    :param func: function to be timed
    :return: Tuple with result and total_time
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs) -> Tuple:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = round((end_time - start_time), 2)
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result, total_time

    return timeit_wrapper
