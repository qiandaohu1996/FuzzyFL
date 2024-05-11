import time
import functools
import torch.autograd.profiler as profiler
from contextlib import contextmanager
import torch
from tqdm import tqdm

Profile_Memory = False




def memory_profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if Profile_Memory:
            torch.cuda.reset_peak_memory_stats()

            with profiler.profile(with_stack=True) as prof:                    # Update the arguments
                result = func(*args, **kwargs)
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
            print(f"GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


@contextmanager
def segment_timing(message):
    start_time = time.time()                    # 记录进入上下文时的时间戳
    try:
        yield                    # 运行代码块
    finally:
        end_time = time.time()                    # 记录离开上下文时的时间戳
        elapsed_time = end_time - start_time                    # 计算运行时间
        tqdm.write(f"{message} : {elapsed_time*1000:.3f} ms")                    # 打印运行时间和额外的消息
Profile_Memory = False


def calc_exec_time(calc_time=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if calc_time:
                start_time = time.time()
            result = func(*args, **kwargs)
            if calc_time:
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                tqdm.write(
                    f"function {func.__name__} execution time : {execution_time:.3f} ms"
                )
            return result
        return wrapper
    return decorator


def memory_profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if Profile_Memory:
            torch.cuda.reset_peak_memory_stats()

            with profiler.profile(with_stack=True) as prof:                    # Update the arguments
                result = func(*args, **kwargs)
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
            print(f"GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


@contextmanager
def segment_timing(message):
    start_time = time.time()                    # 记录进入上下文时的时间戳

    try:
        yield                    # 运行代码块
    finally:
        end_time = time.time()                    # 记录离开上下文时的时间戳
        elapsed_time = end_time - start_time                    # 计算运行时间
        print(f"{message} : {elapsed_time*1000:.3f} ms")                    # 打印运行时间和额外的消息


# 使用timing上下文管理器计算代码片段的运行时间，并传入额外的消息参数
