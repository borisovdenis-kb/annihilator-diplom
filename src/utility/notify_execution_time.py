import time


def notify_execution_time(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        return end - start
    return wrapper
