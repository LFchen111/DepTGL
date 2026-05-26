# distributed/dist_utils.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Union, Callable, Any
import datetime

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def init_distributed(backend: Union[str, None] = None, rank: int = None, world_size: int = None, 
                     master_addr: str = "127.0.0.1", master_port: str = "29500"):
    """
    初始化分布式训练环境
    支持两种模式：
    1. 环境变量模式（用于 torchrun）：从环境变量读取 RANK, WORLD_SIZE 等
    2. 参数模式（用于 spawn）：直接传入 rank, world_size 等参数
    """
    if is_distributed():
        return

    # 优先使用环境变量（兼容 torchrun）
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
    elif rank is not None and world_size is not None:
        # 使用传入的参数（spawn 模式）
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["LOCAL_RANK"] = str(rank)  # 单机多卡时，LOCAL_RANK = RANK
    else:
        raise RuntimeError(
            "未检测到分布式环境变量（RANK/WORLD_SIZE）且未提供参数。"
            "请用 torchrun 启动，或使用 spawn_distributed 函数，或关闭 --distributed。"
        )

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # 设置 init_method
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60000)
    )

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    # 只有在使用 nccl 后端时才强制设置 CUDA 设备
    # gloo 后端让训练代码自己处理设备选择
    if backend == "nccl" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception as e:
            print(f"Warning: Failed to set CUDA device {local_rank}: {e}")
            print(f"Available GPUs: {torch.cuda.device_count()}")

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def barrier():
    if is_distributed():
        dist.barrier()

def destroy_process_group():
    """清理分布式进程组"""
    if is_distributed():
        dist.destroy_process_group()

def all_gather_object(obj):
    """
    简化实现：用 all_gather_object 走 pickle
    """
    if not is_distributed():
        return [obj]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def _spawn_worker(rank: int, world_size: int, fn: Callable, fn_args: tuple, 
                  backend: str = None, master_addr: str = "127.0.0.1", 
                  master_port: str = "29500"):
    """
    每个 spawn 进程的入口函数
    """
    # 初始化分布式环境
    init_distributed(backend=backend, rank=rank, world_size=world_size, 
                    master_addr=master_addr, master_port=master_port)
    
    try:
        # 调用训练函数，传入 rank 作为第一个参数
        fn(rank, *fn_args)
    finally:
        # 清理
        destroy_process_group()


def spawn_distributed(fn: Callable, args: tuple = (), nprocs: int = 1,
                     backend: str = None, master_addr: str = "127.0.0.1",
                     master_port: str = "29500", join: bool = True,
                     daemon: bool = False, start_method: str = "spawn"):
    """
    使用 torch.multiprocessing.spawn 启动分布式训练
    
    Args:
        fn: 训练函数，第一个参数必须是 rank (int)
        args: 传递给训练函数的额外参数（tuple）
        nprocs: 进程数量（world_size）
        backend: 分布式后端（None 时自动选择：CUDA 用 nccl，CPU 用 gloo）
        master_addr: master 地址
        master_port: master 端口
        join: 是否等待所有进程完成
        daemon: 是否设置为守护进程
        start_method: 启动方法（'spawn', 'fork', 'forkserver'），默认 'spawn'
    
    Example:
        def train(rank, args):
            # 初始化分布式
            # ... 训练代码 ...
        
        spawn_distributed(train, args=(args,), nprocs=4)
    """
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # 设置启动方法
    try:
        current_method = mp.get_start_method()
        if current_method != start_method:
            mp.set_start_method(start_method, force=True)
    except RuntimeError:
        # 如果已经设置过，尝试强制设置
        try:
            mp.set_start_method(start_method, force=True)
        except RuntimeError:
            # 如果还是失败，使用当前方法
            pass
    
    # 使用 spawn 启动多进程
    # _spawn_worker 的签名是 (rank, world_size, fn, fn_args, backend, master_addr, master_port)
    # mp.spawn 会传递 rank 和 args 中的参数
    mp.spawn(
        _spawn_worker,
        args=(nprocs, fn, args, backend, master_addr, master_port),
        nprocs=nprocs,
        join=join,
        daemon=daemon,
        start_method=start_method
    )


def run_with_spawn_if_needed(train_fn: Callable, args: Any, world_size: int = 1,
                            use_distributed: bool = False, **spawn_kwargs):
    """
    智能启动函数：如果需要分布式训练，使用 spawn 启动；否则直接运行
    
    这个函数可以在训练脚本的主入口使用，支持两种模式：
    1. 非分布式：直接调用 train_fn(args)
    2. 分布式：使用 spawn 启动多个进程，每个进程调用 train_fn(rank, args)
    
    Args:
        train_fn: 训练函数。如果是分布式模式，第一个参数必须是 rank (int)，第二个参数是 args
                  如果是非分布式模式，第一个参数是 args
        args: 训练参数对象（通常是 argparse.Namespace）
        world_size: 进程数量（仅在分布式模式下使用）
        use_distributed: 是否使用分布式训练
        **spawn_kwargs: 传递给 spawn_distributed 的其他参数（backend, master_addr, master_port 等）
    
    Example:
        # 在 train_self_supervised.py 的主入口：
        if __name__ == "__main__":
            args = parser.parse_args()
            
            def train_main(rank_or_args, parsed_args):
                if isinstance(rank_or_args, int):
                    # 分布式模式：rank_or_args 是 rank
                    rank = rank_or_args
                    args = parsed_args
                    # ... 初始化分布式 ...
                else:
                    # 非分布式模式：rank_or_args 是 args
                    args = rank_or_args
                    # ... 正常训练 ...
            
            run_with_spawn_if_needed(train_main, args, 
                                    world_size=4, 
                                    use_distributed=args.distributed)
    """
    if use_distributed and world_size > 1:
        # 分布式模式：使用 spawn 启动
        def wrapped_train_fn(rank: int):
            train_fn(rank, args)
        
        spawn_distributed(
            wrapped_train_fn,
            args=(),
            nprocs=world_size,
            **spawn_kwargs
        )
    else:
        # 非分布式模式：直接运行
        train_fn(args)
