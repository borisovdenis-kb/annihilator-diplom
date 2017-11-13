import time


def get_execution_time(func):
    start = time.time()
    func()
    end = time.time()
    return end - start
