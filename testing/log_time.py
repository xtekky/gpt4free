from time import time


async def log_time_async(method: callable, **kwargs):
    start = time()
    result = await method(**kwargs)
    secs = f"{round(time() - start, 2)} secs"
    if result:
        return " ".join([result, secs])
    return secs


def log_time_yield(method: callable, **kwargs):
    start = time()
    result = yield from method(**kwargs)
    yield f" {round(time() - start, 2)} secs"


def log_time(method: callable, **kwargs):
    start = time()
    result = method(**kwargs)
    secs = f"{round(time() - start, 2)} secs"
    if result:
        return " ".join([result, secs])
    return secs
