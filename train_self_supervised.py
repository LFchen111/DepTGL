import math
import logging
import time
import sys
import argparse
import random
import torch
import torch.distributed as dist
import numpy as np
import pickle
import os
from pathlib import Path


from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, check_memory_threshold, TimeProfiler, BreakdownProfiler
from utils.data_processing import get_data, compute_time_statistics, Data
from distributed.dist_utils import init_distributed, get_rank, get_world_size, is_distributed, barrier, destroy_process_group, all_gather_object
from distributed.dist_memory_wrapper import DistributedMemoryWrapper
from distributed.data_partitioner import get_partitioned_data
from distributed.range_partitioner import RangePartitioner, HashPartitioner

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=17, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--distributed', action='store_true',
                    help='Whether to use distributed training')
parser.add_argument('--sync_every', type=int, default=1,
                    help='Number of batches between memory synchronization')
parser.add_argument('--num_nodes', type=int, default=None,
                    help='Number of nodes in the graph (if None, inferred from data)')
parser.add_argument('--partition_strategy', type=str, default='hash', choices=['range', 'hash'],
                    help='Node partition strategy: range (contiguous IDs) or hash (recommended when node_id correlates with time)')
parser.add_argument('--world_size', type=int, default=4,
                    help='Number of processes for distributed training (for spawn mode)')
parser.add_argument('--backend', type=str, default='gloo', choices=['nccl', 'gloo'],
                    help='Distributed backend: gloo (recommended for mixed CPU/GPU tensors) or nccl (GPU only)')
parser.add_argument('--start_gpu', type=int, default=0,
                    help='Starting GPU index for distributed training (useful when GPU 0 is occupied)')
parser.add_argument('--quick_test_batches', type=int, default=None,
                    help='Number of batches to run for quick memory test (for auto-tuning sync_every). If set, training exits early after N batches.')
parser.add_argument('--memory_check_threshold', type=float, default=0.90,
                    help='Memory usage threshold for soft OOM detection (0-1, default 0.90)')
parser.add_argument('--profile_time', action='store_true',
                    help='Enable detailed time profiling for each operation')
parser.add_argument('--profile_interval', type=int, default=100,
                    help='Print profiling stats every N batches (default: 10, 0=only at epoch end)')
parser.add_argument('--profile_forward_breakdown', action='store_true',
                    help='Enable fine-grained breakdown inside forward pass (neighbor sampling, aggregation, memory, etc.)')
parser.add_argument('--profile_forward_breakdown_interval', type=int, default=100,
                    help='Print forward breakdown every N batches (default: 100, 0=only at epoch end)')
parser.add_argument('--profile_forward_breakdown_sync_cuda', action='store_true',
                    help='Synchronize CUDA for more accurate breakdown timings (SLOWER).')
parser.add_argument('--disable_dynamic_sync', action='store_true', help='关闭基于梯度的动态同步跳过 (用于消融实验)')
parser.add_argument('--disable_load_aware_drop', action='store_true', help='关闭基于负载的计算丢弃 (用于消融实验)')
parser.add_argument('--disable_prefetch', action='store_true', help='关闭数据预拉取 (退化为彻底无通信)')
parser.add_argument('--grad_ema_threshold', type=float, default=0, help='梯度平稳判定阈值 (Exp 6)')
parser.add_argument('--time_ema_threshold', type=float, default=3, help='掉队者判定阈值 (Exp 6)')
parser.add_argument('--random_ablation', action='store_true', help='使用随机跳过和随机丢弃代替自适应策略 (Random-DepTGL 基线)')
parser.add_argument('--print_sync_details', action='store_true', help='是否打印每次同步的详细耗时信息 (默认不打印以减少刷屏)')
parser.add_argument('--use_distgl_partition', action='store_true',
                    help='Use DistGL source-based edge partitioning strategy instead of default edge replication')
# ================= [新增：全缓存 Baseline 参数] =================
parser.add_argument('--full_cache', action='store_true',
                    help='Enable Full Cache baseline: all nodes are treated as local, causing massive memory usage but high accuracy')
# =====================================================================
# ================= [新增：网络模拟参数] =================
parser.add_argument('--emulate_network', action='store_true',
                    help='是否开启分布式集群网络延迟模拟')
parser.add_argument('--emulate_bandwidth', type=float, default=10.0,
                    help='模拟的集群网络带宽 (Gbps)，默认 10 Gbps (万兆网)')
parser.add_argument('--emulate_latency', type=float, default=2.0,
                    help='模拟的跨机器物理延迟 (ms)，默认 2 ms')
# ========================================================
# ================= [新增：NeutronStream Baseline 参数] =================
parser.add_argument('--use_neutronstream', action='store_true',
                    help='Enable NeutronStream baseline policy (Fixed window sync + Prefetch, NO adaptive strategies)')
# ========================================================
# 将 default 设为 0.0，强制默认走原本的 Baseline 流程
parser.add_argument('--top_k_ratio', type=float, default=0.0,
                    help='[MemShare] Ratio of top degree nodes to share (eg. 0.1). 0.0 means completely disabled.')

def simulate_network_delay(num_nodes_communicated, memory_dim, bandwidth_gbps, base_latency_ms):
    """
    根据通信数据量和设定的带宽，强制进程休眠，模拟真实的物理网络耗时
    """
    if num_nodes_communicated == 0:
        return 0.0

    # 1. 计算本次通信的数据量 (Bytes)，每个 memory 向量为 float32 (4 bytes)
    data_size_bytes = num_nodes_communicated * memory_dim * 4

    # 2. 将带宽转换为 Bytes/second；Gbps 使用 SI 单位 (1 Gbps = 10^9 bits / 8)
    bandwidth_Bps = (bandwidth_gbps * 1e9) / 8.0

    # 3. 计算理论传输耗时
    transfer_time_s = data_size_bytes / bandwidth_Bps

    # 4. 总延迟 = 物理延迟 + 传输耗时
    total_delay_s = (base_latency_ms / 1000.0) + transfer_time_s

    # 5. 强制休眠，模拟网络阻塞
    time.sleep(total_delay_s)
    return total_delay_s


def train_main(rank, args):
    """
    训练主函数 - 支持 spawn 模式
    Args:
        rank: 进程 rank（spawn 模式下自动传入，非分布式时传入 0）
        args: 命令行参数
    """
    # 初始化分布式训练（如果需要）
    if args.distributed:
        init_distributed()  # spawn 模式下会自动使用参数初始化
        rank = get_rank()
        world_size = get_world_size()
    else:
        rank = 0
        world_size = 1

    # 设置全局变量
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim

    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
    get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}_{}.log'.format(str(time.time()), rank))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # logger.info(f'Rank {rank}: Starting training with args: {args}')

    # if args.distributed:
    #     logger.info(f'Initialized distributed training: rank={rank}, world_size={world_size}')

    # ================= [新增：NeutronStream 策略覆盖] =================
    if args.use_neutronstream:
        # if rank == 0:
        #     logger.info("====================================================")
        #     logger.info("[Baseline] Enabling NeutronStream-Py Policy!")
        #     logger.info(f" - Fixed Window Sync (Bounded Staleness): Enabled (sync_every={args.sync_every})")
        #     logger.info(" - Dependency Prefetching: Enabled")
        #     logger.info(" - Hash Partitioning: Enabled")
        #     logger.info(" - Adaptive Dynamic Sync (DepTGL): DISABLED")
        #     logger.info(" - Load-aware Drop (DepTGL): DISABLED")
        #     logger.info("====================================================")

        # 强制使用 Hash 分区（NeutronStream 的基础负载均衡策略）
        args.partition_strategy = 'hash'
        args.use_distgl_partition = False

        # 必须开启预取（NeutronStream 的核心优化）
        args.disable_prefetch = False

        # 强制关闭自适应策略，退化为固定窗口同步
        args.disable_dynamic_sync = True
        args.disable_load_aware_drop = True
        args.random_ablation = False
    # =====================================================================

    ### Extract data for training, validation and testing
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data = get_data(DATA,
                                  different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
    ### Extract data for training, validation and testing
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data = get_data(DATA,
                                      different_new_nodes_between_val_and_test=args.different_new_nodes,
                                      randomize_features=args.randomize_features)

    # ================= [MemShare: 识别 Hotspot Shared Nodes] =================

    # =========================================================================
    # === [MemShare: 识别 Hotspot Shared Nodes] ===
    import numpy as np
    all_interactions = np.concatenate([full_data.sources, full_data.destinations])
    node_counts = np.bincount(all_interactions)
    max_node_id_in_data = len(node_counts)

    num_hot = int(args.top_k_ratio * max_node_id_in_data)
    if num_hot > 0:
        hot_nodes_array = np.argsort(node_counts)[-num_hot:]
        hot_nodes_set = set(hot_nodes_array.tolist())

        # 将打印保护在仅由 num_hot 大于 0 触发的范围内
        if rank == 0:
            logger.info(f'[MemShare] 热点节点共享机制开启: 发现 {num_hot} 个 Top-K 热点节点 (ratio={args.top_k_ratio})')
    else:
        hot_nodes_set = set()
    # Determine number of nodes
    if args.num_nodes is None:
        num_nodes = max(full_data.sources.max(), full_data.destinations.max()) + 1
        #logger.info(f'Inferred number of nodes from data: {num_nodes}')
    else:
        # 使用指定的节点数量，但需要确保至少能覆盖数据中的所有节点
        data_max_node = max(full_data.sources.max(), full_data.destinations.max()) + 1
        num_nodes = max(args.num_nodes, data_max_node)
        # if args.num_nodes < data_max_node:
        #     logger.warning(f'Specified num_nodes ({args.num_nodes}) is smaller than actual nodes in data ({data_max_node}). Using {num_nodes}.')
        # else:
        #     logger.info(f'Using specified number of nodes: {num_nodes}')

    # Partition data for distributed training
    partitioner = None
    if args.distributed and world_size > 1:
        # ================= [修改：绑定 DistGL STEP 分区] =================
        if args.use_distgl_partition:
            # 如果启用了 DistGL，强制使用 STEP 分区器覆盖原有的 Hash/Range
            from distributed.range_partitioner import STEPPartitioner
            if rank == 0:
                logger.info(f'Rank {rank}: [DistGL Mode] Building STEP partitioner (one-time cost)...')
            partitioner = STEPPartitioner(num_nodes=num_nodes, world_size=world_size, full_data=full_data)
            local_nodes = partitioner.get_local_nodes(rank)
            #logger.info(f'Rank {rank}: [DistGL Mode] Using STEP partitioner, owns {len(local_nodes)} nodes')
        # =====================================================================
        elif args.partition_strategy == 'hash':
            partitioner = HashPartitioner(num_nodes=num_nodes, world_size=world_size)
            local_nodes = partitioner.get_local_nodes(rank)
            #logger.info(f'Rank {rank}: Using hash partitioner, owns {len(local_nodes)} nodes (first 10: {local_nodes[:10]})')
        else:
            partitioner = RangePartitioner(num_nodes=num_nodes, world_size=world_size)
            #logger.info(f'Rank {rank}: Using range partitioner, owns nodes {partitioner.get_range(rank)}')
        
        # ================= [新增：全缓存 Baseline 逻辑] =================
        if args.full_cache:
            logger.warning(f'Rank {rank}: FULL CACHE MODE ENABLED! Overriding partitioner to treat ALL nodes as local. Expect high memory usage.')
            # 动态覆盖 partitioner 的方法，欺骗系统认为所有节点都在本地
            #partitioner.get_local_nodes = lambda r: range(num_nodes)  # 优化：去掉 list()
            #partitioner.is_local = lambda node_id, r: True
            partitioner.is_full_cache = True  # 新增标记，用于后续极速判断
            # if isinstance(partitioner, RangePartitioner):
            #     partitioner.get_range = lambda r: (0, num_nodes)
        # =====================================================================

        # Partition training data by batches
        # 分区策略：
        # 1. 先将所有数据（所有边）划分成 batch
        # 2. 对于每个 batch，按照节点划分将 batch 中的边分配给不同的进程
        # 3. 如果一条边涉及两个不同分区的节点，该边会被复制到这两个分区
        #    这样每个分区都可以更新自己拥有的节点
        original_size = len(train_data.sources)
        
        # ================= [新增：根据开关选择边分区策略] =================
        if args.use_distgl_partition:
            from distributed.data_partitioner import partition_data_by_batches_distgl
            #logger.info(f'Rank {rank}: Using DistGL source-based edge partitioning strategy.')
            train_data_batches, communication_nodes_per_batch = partition_data_by_batches_distgl(
                train_data, partitioner, rank, BATCH_SIZE, world_size=world_size,
                sync_every=args.sync_every, enable_prefetch=not args.disable_prefetch
            )
        else:
            from distributed.data_partitioner import partition_data_by_batches
            #logger.info(f'Rank {rank}: Using default edge replication partitioning strategy.')
            train_data_batches, communication_nodes_per_batch = partition_data_by_batches(
                train_data, partitioner, rank, BATCH_SIZE, world_size=world_size,
                sync_every=args.sync_every, enable_prefetch=not args.disable_prefetch,
                use_both_sides=args.use_neutronstream
            )
        # =====================================================================

        # ================= [新增：DistGL 优化 3 - 静态特征缓存] =================
        distgl_static_cache = set()
        if args.use_distgl_partition and train_data_batches is not None:
            from collections import defaultdict
            remote_freq = defaultdict(int)
            for batch in train_data_batches:
                for src, dst in zip(batch.sources, batch.destinations):
                    if partitioner.owner(src) != rank: remote_freq[src] += 1
                    if partitioner.owner(dst) != rank: remote_freq[dst] += 1
            top_m = 1000
            distgl_static_cache = set(sorted(remote_freq, key=remote_freq.get, reverse=True)[:top_m])
            logger.info(f'Rank {rank}: [DistGL] Built Static Cache with {len(distgl_static_cache)} nodes.')
        # ========================================================================

        # logger.info(f'Rank {rank}: Communication batches: {sorted(communication_nodes_per_batch.keys())}')
        
        # 计算每个 batch 的大小并统计
        total_partitioned_edges = sum(len(batch.sources) for batch in train_data_batches)
        # logger.info(f'Rank {rank}: Partitioned training data into {len(train_data_batches)} batches, '
        #            f'total {total_partitioned_edges} edges (original: {original_size}, '
        #            f'ratio: {total_partitioned_edges/original_size:.2%})')
        #
        # 只有在没有禁用预拉取时，才输出预拉取扩展统计
        if not args.disable_prefetch:
            # 统计预拉取完成后所有机器需要训练的事件总和
            # 收集所有rank的事件数
            local_total_events = total_partitioned_edges
            all_rank_events = all_gather_object(local_total_events)
            total_events_all_ranks = sum(all_rank_events)
            
            # 只在rank 0输出统计信息
            if rank == 0:
                print(f"\n{'='*80}")
                print(f"预拉取统计 (sync_every={args.sync_every}):")
                print(f"  原始训练事件数量: {original_size}")
                print(f"  预拉取完成后所有机器需要训练的事件总和: {total_events_all_ranks}")
                print(f"  事件增加比例: {total_events_all_ranks / original_size:.2%}")
                print(f"  各机器事件数: {all_rank_events}")
                print(f"{'='*80}\n")
                logger.info(f"预拉取统计 (sync_every={args.sync_every}): 原始={original_size}, "
                           f"预拉取后总和={total_events_all_ranks}, 比例={total_events_all_ranks/original_size:.2%}")
        else:
            # 如果禁用了预拉取，打印一条提示信息，保持原始划分数据不变
            if rank == 0:
                print(f"\n{'='*80}")
                print("预拉取已禁用 (--disable_prefetch)。使用原始划分数据进行训练。")
                print(f"{'='*80}\n")
        
        # 统计跨分区边的数量
        # local_node_set = set(partitioner.get_local_nodes(rank))
        # cross_partition_edges = 0
        # for batch in train_data_batches:
        #     for src, dst in zip(batch.sources, batch.destinations):
        #         src_local = src in local_node_set
        #         dst_local = dst in local_node_set
        #         if src_local and not dst_local or (not src_local and dst_local):
        #             # 只有一个节点在本地，说明是跨分区边
        #             cross_partition_edges += 1
        # if total_partitioned_edges > 0:
        #     logger.info(f'Rank {rank}: Cross-partition edges (involving remote nodes): {cross_partition_edges} '
        #                f'({cross_partition_edges/total_partitioned_edges:.2%} of local data)')

        # 为了兼容性，仍然创建完整的 train_data（用于 neighbor finder）
        # 将所有 batch 合并成一个 Data 对象
        if train_data_batches:
            all_sources = np.concatenate([batch.sources for batch in train_data_batches])
            all_destinations = np.concatenate([batch.destinations for batch in train_data_batches])
            all_timestamps = np.concatenate([batch.timestamps for batch in train_data_batches])
            all_edge_idxs = np.concatenate([batch.edge_idxs for batch in train_data_batches])
            all_labels = None
            if train_data_batches[0].labels is not None:
                all_labels = np.concatenate([batch.labels for batch in train_data_batches if batch.labels is not None])
            train_data = Data(all_sources, all_destinations, all_timestamps, all_edge_idxs, all_labels)
        else:
            # 如果没有数据，创建一个空的数据对象
            train_data = Data(
                np.array([], dtype=train_data.sources.dtype),
                np.array([], dtype=train_data.destinations.dtype),
                np.array([], dtype=train_data.timestamps.dtype),
                np.array([], dtype=train_data.edge_idxs.dtype),
                None
            )
        
        # Create partitioned neighbor finder for training
        train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    else:
        train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                          seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                           new_node_test_data.destinations,
                                           seed=3)

    # Set device - 在分布式模式下使用 local_rank 对应的设备
    # 重要：即使 backend=gloo，也要显式 set_device，否则某些 PyTorch 路径可能会拿到 device=-1
    gpu_id = None
    if args.distributed and world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        # 使用 start_gpu 偏移，避免使用被占用的 GPU
        gpu_id = local_rank + args.start_gpu
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device_string = f'cuda:{gpu_id}'
        else:
            device_string = 'cpu'
        #logger.info(f'Rank {rank}: Using GPU {gpu_id} (local_rank={local_rank}, start_gpu={args.start_gpu})')
    else:
        if torch.cuda.is_available():
            gpu_id = GPU
            torch.cuda.set_device(gpu_id)
            device_string = f'cuda:{gpu_id}'
        else:
            device_string = 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
      compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):
      results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
      Path("results/").mkdir(parents=True, exist_ok=True)

      # Initialize Model
      tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                edge_features=edge_features, device=device,
                n_layers=NUM_LAYER,
                n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                memory_update_at_start=not args.memory_update_at_end,
                embedding_module_type=args.embedding_module,
                message_function=args.message_function,
                aggregator_type=args.aggregator,
                memory_updater_type=args.memory_updater,
                n_neighbors=NUM_NEIGHBORS,
                mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                use_source_embedding_in_message=args.use_source_embedding_in_message,
                dyrep=args.dyrep)
      
      # Wrap memory with distributed wrapper if using distributed training
      memory_wrapper = None
      if USE_MEMORY and args.distributed and world_size > 1:
        # Use the same partitioner as data partitioning for consistency
        # 注意：在 DDP 包装之前访问 tgn.memory
        # ================= [修改：绑定 DistGL 动态缓存陈旧度] =================
        # 如果启用 DistGL，允许 Memory 存在 2 个 batch 的陈旧度；否则为 0 (严格同步)
        distgl_staleness = 2 if args.use_distgl_partition else 0
        # =====================================================================
        memory_wrapper = DistributedMemoryWrapper(
            memory_module=tgn.memory,
            num_nodes=num_nodes,
            device=device,
            sync_every=args.sync_every,
            partitioner=partitioner,
            distgl_staleness=distgl_staleness,
            # ================= [新增：传入静态缓存] =================
            static_cache=distgl_static_cache if args.use_distgl_partition else set(),
            # ========================================================
            hot_nodes = hot_nodes_set
        )
        #logger.info(f'Rank {rank}: Initialized distributed memory wrapper (sync_every={args.sync_every}, distgl_staleness={distgl_staleness}, static_cache_size={len(distgl_static_cache) if args.use_distgl_partition else 0})')
        # 注意：预拉取数据已经在数据分区阶段（partition_data_by_batches）计算并包含在train_data_batches中
      
      # 修改：设置 reduction='none'，方便我们手动应用 Mask
      criterion = torch.nn.BCELoss(reduction='none')
      tgn = tgn.to(device)
      
      # 引入 DDP 包装（在 MemoryWrapper 之后）
      if args.distributed and world_size > 1:
        # 只有在 GPU 模式下才指定 device_ids
        if device.type == 'cuda':
          assert gpu_id is not None
          tgn = torch.nn.parallel.DistributedDataParallel(
            tgn, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True
          )
        else:
          # CPU 模式下不指定 device_ids
          tgn = torch.nn.parallel.DistributedDataParallel(tgn, find_unused_parameters=True)
        #logger.info(f'Rank {rank}: Wrapped model with DDP (device: {device.type})')
      
      # Optimizer 定义在 DDP 之后，确保参数正确
      optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
      
      # 辅助函数：获取模型引用（兼容 DDP 和非 DDP）
      def get_model_ref():
        return tgn.module if args.distributed and world_size > 1 else tgn

      # 在分布式模式下，使用全局 batch 数量（所有进程共享相同的 batch 数量）
      if args.distributed and world_size > 1:
        # 计算全局 batch 数量（基于原始数据大小）
        original_train_size = len(full_data.sources)  # 使用完整数据大小作为参考
        global_num_batch = math.ceil(original_train_size / BATCH_SIZE)
        num_batch = len(train_data_batches)  # 本地实际的 batch 数量
        #logger.info(f'Rank {rank}: Local num_batch={num_batch}, global_num_batch={global_num_batch}')
      else:
        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / BATCH_SIZE)
        global_num_batch = num_batch
        train_data_batches = None  # 非分布式模式下不使用 batch 列表

      #logger.info('num of batches per epoch: {}'.format(global_num_batch))

      new_nodes_val_aps = []
      val_aps = []
      epoch_times = []
      total_epoch_times = []
      train_losses = []

      early_stopper = EarlyStopMonitor(max_round=args.patience)

      for epoch in range(NUM_EPOCH):
        if args.distributed and world_size > 1:dist.barrier()
        start_epoch = time.time()
        
        # ================= [修正：移到 Epoch 循环内部！] =================
        ema_grad_norm = 0.0
        last_grad_norm = 0.0  # 新增：用来记录上一个 batch 的真实梯度
        alpha_grad = 0.1  # 梯度 EMA 平滑系数
        
        ema_batch_time = 0.0
        beta_time = 0.1   # 耗时 EMA 平滑系数
        prev_batch_time = 0.0
        # ============================================================

        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
          get_model_ref().memory.__init_memory__()
          if memory_wrapper is not None:
              memory_wrapper.reset()

        # Train using only training graph
        get_model_ref().set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        
        # 快速测试模式：用于自动调优
        quick_test_mode = args.quick_test_batches is not None
        if quick_test_mode and rank == 0:
            logger.info(f'Running in quick test mode: will exit after {args.quick_test_batches} batches')
        
        # 负载均衡监控：记录每次同步前的时间和batch进度
        sync_time_records = []  # 记录每次同步的时间信息
        batch_start_time = time.time()  # epoch开始时间
        
        # 时间分解统计（训练循环级别）
        profiler = TimeProfiler(enabled=args.profile_time)

        # 前向内部细粒度分解统计（模型内部）
        fwd_profiler = BreakdownProfiler(
            enabled=args.profile_forward_breakdown,
            sync_cuda=args.profile_forward_breakdown_sync_cuda
        )
        # 挂到模型与 embedding_module（下层会读取）
        model_ref_for_prof = get_model_ref()
        model_ref_for_prof.op_profiler = fwd_profiler
        if hasattr(model_ref_for_prof, "embedding_module"):
            model_ref_for_prof.embedding_module.op_profiler = fwd_profiler
        
        for k in range(0, global_num_batch, args.backprop_every):
          loss = None
          num_micro_batches = 0
          optimizer.zero_grad()

          # Custom loop to allow to perform backpropagation only every a certain number of batches
          for j in range(args.backprop_every):
            batch_idx = k + j
            if batch_idx >= global_num_batch:
              continue
            
            # 快速测试模式: 检查是否已达到测试batch数量
            if quick_test_mode and batch_idx >= args.quick_test_batches:
                logger.info(f'Quick test completed: processed {batch_idx} batches successfully')
                print(f'[AutoSearch] SUCCESS at sync_every={args.sync_every}: processed {batch_idx} batches')
                sys.exit(0)  # 正常退出

            # 在batch开始前，如果是通信batch，进行memory同步
            if USE_MEMORY and memory_wrapper is not None and args.distributed and world_size > 1:
              # 所有机器在sync batch时都需要调用sync_p2p（确保barrier同步）
              if batch_idx % memory_wrapper.sync_every == 0:
                # ================= [修改：动态 K 值判定 (跳过通信)] =================
                skip_sync = False
                if batch_idx > 0:
                    if not args.random_ablation and not args.disable_dynamic_sync:
                        # 判定：使用上一个 batch 算出的真实梯度 last_grad_norm
                        if ema_grad_norm > 0 and last_grad_norm < args.grad_ema_threshold * ema_grad_norm:
                            skip_sync = True

                # 【关键修复：必须同步 skip_sync 决定，防止多卡死锁！】
                if args.distributed and world_size > 1:
                    if args.random_ablation:
                        # Random ablation 由 Rank 0 统一掷骰子
                        if rank == 0:
                            skip_sync = random.random() < 0.3
                    
                    # 将 Rank 0 的决定广播给所有人，保证大家同进同退
                    # 【性能优化】：使用纯 Tensor 的 broadcast 替代极慢的 all_gather_object (Pickle 序列化)
                    skip_tensor = torch.tensor([1 if skip_sync else 0], dtype=torch.uint8, device=device)
                    dist.broadcast(skip_tensor, src=0)
                    skip_sync = bool(skip_tensor.item())
                    
                #     if skip_sync and rank == 0:
                #         if args.random_ablation:
                #             logger.info(f"Batch {batch_idx}: 随机策略触发，跳过本次通信")
                #         else:
                #             logger.info(f"Batch {batch_idx}: 梯度平稳 (Norm: {last_grad_norm:.4f} < 阈值 {args.grad_ema_threshold * ema_grad_norm:.4f})，跳过本次通信")
                # # =====================================================================

                if not skip_sync:
                    # 记录同步前的时间和进度
                    time_before_sync = time.time()
                    elapsed_since_epoch_start = time_before_sync - batch_start_time
                    
                    # 获取当前机器需要的节点（如果没有则为空集合）
                    needed_nodes = communication_nodes_per_batch.get(batch_idx, set())
                    
                    # ================= [新增：DistGL 优化 4 - 两阶段预取] =================
                    if args.use_distgl_partition:
                        # 获取下个 batch 需要的节点
                        needed_next = communication_nodes_per_batch.get(batch_idx + 1, set())
                        # 找出 Inactive 节点 (下个 batch 需要，但当前 batch 不需要)
                        inactive_next = needed_next - needed_nodes
                        # Stage 1: 把 inactive_next 合并到当前 batch 提前拉取
                        fetch_now = needed_nodes.union(inactive_next)
                    else:
                        fetch_now = needed_nodes
                    # ========================================================================

                    # ================= [新增：注入网络延迟] =================
                    if args.emulate_network:
                        # 全通信基线 (Vanilla DDP / Full Cache) 同步全局节点表；DepTGL 只同步需要的节点
                        comm_nodes = num_nodes if (args.full_cache or args.disable_prefetch) else len(fetch_now)
                        delay_s = simulate_network_delay(
                            num_nodes_communicated=comm_nodes,
                            memory_dim=MEMORY_DIM,
                            bandwidth_gbps=args.emulate_bandwidth,
                            base_latency_ms=args.emulate_latency
                        )
                        if args.print_sync_details and rank == 0:
                            logger.info(f"[Network Emulation] 注入网络延迟: {delay_s:.4f}s (通信节点数: {comm_nodes})")
                    # ========================================================

                    # 记录同步信息（在同步前）
                    if args.print_sync_details:
                        logger.info(f'Rank {rank}: Batch {batch_idx} - About to sync, elapsed time: {elapsed_since_epoch_start:.2f}s')
                    
                    # 执行同步（计时）
                    profiler.start('1_communication_sync')
                    memory_wrapper.sync_p2p(needed_nodes=fetch_now, current_batch_idx=batch_idx)
                    profiler.stop()
                    
                    # 记录同步后的时间
                    time_after_sync = time.time()
                    sync_duration = time_after_sync - time_before_sync
                    
                    # 收集所有rank的时间信息
                    sync_info = {
                        'rank': rank,
                        'batch_idx': batch_idx,
                        'time_before_sync': elapsed_since_epoch_start,
                        'sync_duration': sync_duration
                    }
                    all_sync_info = all_gather_object(sync_info)
                    
                    # 提取时间信息（所有rank都需要）
                    times = [info['time_before_sync'] for info in all_sync_info]
                    durations = [info['sync_duration'] for info in all_sync_info]
                    
                    # 只在rank 0输出统计信息，并且开启了打印参数才输出
                    if rank == 0 and args.print_sync_details:
                        print(f"\n{'='*70}")
                        print(f"Sync at batch {batch_idx} (sync_every={args.sync_every}):")
                        for info in all_sync_info:
                            print(f"  Rank {info['rank']}: arrived at {info['time_before_sync']:.2f}s, sync took {info['sync_duration']:.3f}s")
                        print(f"  Time spread: {max(times) - min(times):.2f}s (fastest: {min(times):.2f}s, slowest: {max(times):.2f}s)")
                        print(f"  Avg sync duration: {sum(durations)/len(durations):.3f}s")
                        print(f"{'='*70}\n")
                    
                    # 记录到列表
                    sync_time_records.append({
                        'batch_idx': batch_idx,
                        'all_ranks_time': times,
                        'time_spread': max(times) - min(times)
                    })

            # 数据准备
            profiler.start('2_data_preparation')
            micro_batch_start_time = time.time() # 记录当前 batch 开始时间
            
            # 在分布式模式下，使用预先划分好的 batch
            # 注意：batch数据已经包含了预拉取的事件（在partition_data_by_batches中计算）
            if args.distributed and world_size > 1 and train_data_batches is not None:
              if batch_idx < len(train_data_batches):
                batch_data = train_data_batches[batch_idx]
                sources_batch = batch_data.sources
                destinations_batch = batch_data.destinations
                edge_idxs_batch = batch_data.edge_idxs
                timestamps_batch = batch_data.timestamps
              else:
                # 如果 batch_idx 超出范围，跳过这个 batch
                profiler.stop()
                continue
            else:
              # 非分布式模式，使用原来的方式
              num_instance = len(train_data.sources)
              if batch_idx >= num_batch:
                profiler.stop()
                continue
              start_idx = batch_idx * BATCH_SIZE
              end_idx = min(num_instance, start_idx + BATCH_SIZE)
              sources_batch = train_data.sources[start_idx:end_idx]
              destinations_batch = train_data.destinations[start_idx:end_idx]
              edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
              timestamps_batch = train_data.timestamps[start_idx:end_idx]

            # ================= [修改：负载感知与动态丢弃预拉取边] =================
            if args.distributed and world_size > 1 and train_data_batches is not None:
                is_heavy_load = False
                
                if args.random_ablation:
                    # Random-DepTGL 变体：随机判定为高负载，触发丢弃 (例如 20% 概率)
                    if random.random() < 0.2:
                        is_heavy_load = True
                elif not args.disable_load_aware_drop:
                    # 评估当前负载：如果上一个 batch 的耗时超过历史平均的 time_ema_threshold 倍，认为是高负载
                    if ema_batch_time > 0 and prev_batch_time > args.time_ema_threshold * ema_batch_time:
                        is_heavy_load = True

                if is_heavy_load:
                    # 找出哪些边是"纯预拉取边"
                    src_t = torch.tensor(sources_batch, device=device, dtype=torch.long)

                    # O(1) 极速查找优化
                    if not hasattr(partitioner, '_local_node_mask'):
                        # if getattr(partitioner, 'is_full_cache', False):
                        #     partitioner._cached_local_nodes_t = torch.arange(num_nodes, device=device, dtype=torch.long)
                        #     local_nodes_len = num_nodes
                        # else:
                        local_nodes_list = list(partitioner.get_local_nodes(rank))
                        partitioner._cached_local_nodes_t = torch.tensor(local_nodes_list, device=device, dtype=torch.long)
                        local_nodes_len = len(local_nodes_list)

                        max_id = max(10000000, num_nodes + 100000)
                        mask = torch.zeros(max_id, dtype=torch.bool, device=device)
                        if local_nodes_len > 0:
                            mask[partitioner._cached_local_nodes_t] = True
                        partitioner._local_node_mask = mask

                    is_src_remote = ~partitioner._local_node_mask[src_t]

                    if args.use_neutronstream:
                        # Both-sides 策略：只有两端都不在本地的边才是纯预拉取边
                        dst_t = torch.tensor(destinations_batch, device=device, dtype=torch.long)
                        is_dst_remote = ~partitioner._local_node_mask[dst_t]
                        is_prefetch_edge = is_src_remote & is_dst_remote
                    else:
                        # Source-based 策略：源节点不在本地即为预拉取边
                        is_prefetch_edge = is_src_remote

                    keep_mask = ~is_prefetch_edge
                    
                    if not keep_mask.all():
                        keep_indices = keep_mask.cpu().numpy()
                        
                        # ================= [新增：打印丢弃日志，证明策略生效！] =================
                        # dropped_count = len(sources_batch) - keep_indices.sum()
                        # logger.info(f"Rank {rank} Batch {batch_idx}: 负载过高 (前一Batch耗时 {prev_batch_time:.3f}s > 阈值 {args.time_ema_threshold * ema_batch_time:.3f}s)，触发计算丢弃，丢弃了 {dropped_count} 条预拉取边！")
                        # # =====================================================================
                        
                        sources_batch = sources_batch[keep_indices]
                        destinations_batch = destinations_batch[keep_indices]
                        edge_idxs_batch = edge_idxs_batch[keep_indices]
                        timestamps_batch = timestamps_batch[keep_indices]
            # =====================================================================

            size = len(sources_batch)
            if size == 0:
              # 如果 batch 为空，跳过
              profiler.stop()
              continue
            _, negatives_batch = train_rand_sampler.sample(size)
            profiler.stop()

            # 数据传输到GPU
            profiler.start('3_data_to_gpu')
            with torch.no_grad():
              pos_label = torch.ones(size, dtype=torch.float, device=device)
              neg_label = torch.zeros(size, dtype=torch.float, device=device)
            profiler.stop()

            # 前向传播
            profiler.start('4_forward_pass')
            tgn = tgn.train()
            pos_prob, neg_prob = get_model_ref().compute_edge_probabilities(
              sources_batch, destinations_batch, negatives_batch,
              timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS
            )
            profiler.stop()

            # 损失计算 (带 Mask，屏蔽预拉取边的梯度)
            profiler.start('5_loss_computation')
            
            loss_mask = torch.ones(size, dtype=torch.float, device=device)
            if args.distributed and world_size > 1 and train_data_batches is not None:
                src_t = torch.tensor(sources_batch, device=device, dtype=torch.long)

                # O(1) 极速查找优化
                if not hasattr(partitioner, '_local_node_mask'):
                    # if getattr(partitioner, 'is_full_cache', False):
                    #     partitioner._cached_local_nodes_t = torch.arange(num_nodes, device=device, dtype=torch.long)
                    #     local_nodes_len = num_nodes
                    # else:
                    local_nodes_list = list(partitioner.get_local_nodes(rank))
                    partitioner._cached_local_nodes_t = torch.tensor(local_nodes_list, device=device, dtype=torch.long)
                    local_nodes_len = len(local_nodes_list)
                    
                    # 直接分配足够大小的布尔数组 (占用仅几MB显存)
                    # 彻底干掉每批次的 .item() 动态检查，消除 CPU-GPU 同步阻塞！
                    max_id = max(10000000, num_nodes + 100000)
                    mask = torch.zeros(max_id, dtype=torch.bool, device=device)
                    if local_nodes_len > 0:
                        mask[partitioner._cached_local_nodes_t] = True
                    partitioner._local_node_mask = mask
                
                # O(1) 瞬间完成查找 (没有任何 CPU-GPU 同步阻塞！)
                is_src_remote = ~partitioner._local_node_mask[src_t]

                if args.use_neutronstream:
                    # Both-sides 策略：只有两端都不在本地的边才是纯预拉取边
                    dst_t = torch.tensor(destinations_batch, device=device, dtype=torch.long)
                    is_dst_remote = ~partitioner._local_node_mask[dst_t]
                    is_prefetch_edge = is_src_remote & is_dst_remote
                else:
                    # Source-based 策略：源节点不在本地即为纯预拉取边
                    is_prefetch_edge = is_src_remote

                # 屏蔽纯预拉取边，防止一条边被两台机器重复计算梯度
                loss_mask[is_prefetch_edge] = 0.0

            pos_loss = criterion(pos_prob.squeeze(-1), pos_label)
            neg_loss = criterion(neg_prob.squeeze(-1), neg_label)
            
            valid_edges = loss_mask.sum()
            if valid_edges > 0:
                batch_loss = (pos_loss * loss_mask).sum() / valid_edges + (neg_loss * loss_mask).sum() / valid_edges
            else:
                # 极端情况防御：如果全被 Mask 掉了，给一个 0 梯度维持 DDP 同步
                batch_loss = (pos_prob.sum() + neg_prob.sum()) * 0.0
                
            loss = batch_loss if loss is None else (loss + batch_loss)
            num_micro_batches += 1
            profiler.stop()

            # 定期输出“前向内部”分解（只看 rank0，输出后 reset，便于看区间占比）
            if args.profile_forward_breakdown and args.profile_forward_breakdown_interval > 0 and rank == 0:
                if batch_idx > 0 and batch_idx % args.profile_forward_breakdown_interval == 0:
                    fwd_profiler.print_summary(
                        title=f"[Intermediate] Forward Breakdown at Batch {batch_idx} (Rank 0)"
                    )
                    fwd_profiler.reset()
            
            # 定期输出时间统计
            if args.profile_time and args.profile_interval > 0 and rank == 0:
                if batch_idx > 0 and batch_idx % args.profile_interval == 0:
                    profiler.print_stats(f"[Intermediate] Time Profiling at Batch {batch_idx} (Rank 0)")
            
            # 内存检查点：在每个batch处理后检查内存使用情况
            try:
                check_memory_threshold(
                    cpu_threshold=args.memory_check_threshold,
                    gpu_threshold=args.memory_check_threshold
                )
            except RuntimeError as e:
                if "SOFT_OOM" in str(e):
                    logger.error(f'Memory threshold exceeded at batch {batch_idx}: {e}')
                    print(f'[AutoSearch] FAILED at sync_every={args.sync_every}, batch {batch_idx}: {e}')
                    sys.exit(55)  # 返回特殊错误码，供外部脚本识别
                raise e

          if num_micro_batches > 0:
            loss = loss / num_micro_batches
            
            # 反向传播
            profiler.start('6_backward_pass')
            loss.backward()
            profiler.stop()
            
            # ================= [新增：在这里计算真实的梯度 Norm！] =================
            if not args.random_ablation and not args.disable_dynamic_sync:
                # 【性能优化】：在 GPU 上一次性计算所有梯度的 Norm，消除 for 循环中致命的 CPU-GPU 同步
                #grads = [p.grad.detach() for p in get_model_ref().parameters() if p.grad is not None]
                # 增加 p.is_leaf 判断，完美避开非叶子节点参数的警告
                grads = [p.grad.detach() for p in get_model_ref().parameters() if p.is_leaf and p.grad is not None]
                if len(grads) > 0:
                    stacked_norms = torch.stack([torch.norm(g) for g in grads])
                    current_grad_norm = torch.norm(stacked_norms).item()  # 整个 batch 只在这里同步 1 次
                else:
                    current_grad_norm = 0.0
                
                # 存下来给下一个 batch 的判定使用
                last_grad_norm = current_grad_norm
                
                if ema_grad_norm == 0.0:
                    ema_grad_norm = current_grad_norm
                else:
                    ema_grad_norm = alpha_grad * current_grad_norm + (1 - alpha_grad) * ema_grad_norm
            # ============================================================
            
            # 优化器更新
            profiler.start('7_optimizer_step')
            optimizer.step()
            profiler.stop()
            
            # ================= [新增：更新耗时统计] =================
            prev_batch_time = time.time() - micro_batch_start_time
            if ema_batch_time == 0.0:
                ema_batch_time = prev_batch_time
            else:
                ema_batch_time = beta_time * prev_batch_time + (1 - beta_time) * ema_batch_time
            # ============================================================
            
            m_loss.append(loss.item())

          # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
          # the start of time
          if USE_MEMORY:
            profiler.start('8_memory_detach')
            get_model_ref().memory.detach_memory()
            
            # [FIX] 强行切断所有残留消息的计算图，防止单卡 GRU inplace 报错
            if hasattr(get_model_ref().memory, 'messages'):
                for node_id, messages in get_model_ref().memory.messages.items():
                    new_messages = []
                    for msg in messages:
                        if isinstance(msg, tuple) and torch.is_tensor(msg[0]):
                            new_messages.append((msg[0].detach(),) + msg[1:])
                        else:
                            new_messages.append(msg)
                    get_model_ref().memory.messages[node_id] = new_messages
                    
            profiler.stop()

        # epoch 结束：所有 rank 都调用一次 sync（只能一次）
        if USE_MEMORY and memory_wrapper is not None:
          # ================= [新增：注入网络延迟] =================
          if args.emulate_network:
              simulate_network_delay(num_nodes, MEMORY_DIM, args.emulate_bandwidth, args.emulate_latency)
          # ========================================================
          profiler.start('9_final_memory_sync')
          memory_wrapper.sync()
          profiler.stop()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        
        # 输出时间分解统计
        if args.profile_time and rank == 0:
            profiler.print_stats(f"Time Profiling for Epoch {epoch} (Rank 0)")

        # 输出“前向内部”分解（epoch 结束）
        if args.profile_forward_breakdown and rank == 0:
            fwd_profiler.print_summary(title=f"Forward Breakdown for Epoch {epoch} (Rank 0)")
        
        # 输出负载均衡统计
        if args.distributed and world_size > 1 and len(sync_time_records) > 0:
            # 只在rank 0输出总结
            if rank == 0:
                print(f"\n{'#'*70}")
                print(f"Load Balance Summary for Epoch {epoch}:")
                print(f"{'#'*70}")
                print(f"Total synchronization points: {len(sync_time_records)}")
                
                # 计算每次同步的时间差异
                spreads = [rec['time_spread'] for rec in sync_time_records]
                print(f"\nTime spread at each sync point:")
                print(f"  Average: {sum(spreads)/len(spreads):.2f}s")
                print(f"  Max: {max(spreads):.2f}s")
                print(f"  Min: {min(spreads):.2f}s")
                
                # 显示最不均衡的几次同步
                sorted_records = sorted(sync_time_records, key=lambda x: x['time_spread'], reverse=True)
                print(f"\nTop 3 most imbalanced sync points:")
                for i, rec in enumerate(sorted_records[:3]):
                    times = rec['all_ranks_time']
                    print(f"  {i+1}. Batch {rec['batch_idx']}: spread={rec['time_spread']:.2f}s")
                    print(f"     Times: [" + ", ".join([f"R{j}:{t:.2f}s" for j, t in enumerate(times)]) + "]")
                
                # 判断负载是否均衡
                avg_spread = sum(spreads) / len(spreads)
                if avg_spread < 1.0:
                    print(f"\n✓ Load is well balanced (avg spread < 1s)")
                elif avg_spread < 3.0:
                    print(f"\n⚠ Load is moderately balanced (avg spread 1-3s)")
                else:
                    print(f"\n✗ Load is imbalanced (avg spread > 3s)")
                    print(f"  Consider adjusting partition strategy or data distribution")
                
                print(f"{'#'*70}\n")

        ### Validation
        # In distributed mode, only rank 0 performs validation to avoid memory timestamp conflicts
        if args.distributed and world_size > 1:
          # Synchronize before validation to ensure all workers are at the same point
          barrier()
          
          if rank == 0:
            # Only rank 0 performs validation
            # Temporarily restore original memory methods to bypass wrapper during validation
            model_ref = get_model_ref()
            if USE_MEMORY and memory_wrapper is not None:
              model_ref.memory.store_raw_messages = memory_wrapper._orig_store_raw_messages
              model_ref.memory.get_memory = memory_wrapper._orig_get_memory
              model_ref.memory.get_last_update = memory_wrapper._orig_get_last_update
              if memory_wrapper._orig_set_memory is not None:
                model_ref.memory.set_memory = memory_wrapper._orig_set_memory
            
            get_model_ref().set_neighbor_finder(full_ngh_finder)

            if USE_MEMORY:
              # Backup memory at the end of training, so later we can restore it and use it for the
              # validation on unseen nodes
              train_memory_backup = model_ref.memory.backup_memory()

            val_ap, val_auc = eval_edge_prediction(model=get_model_ref(),
                                                                   negative_edge_sampler=val_rand_sampler,
                                                                   data=val_data,
                                                                   n_neighbors=NUM_NEIGHBORS)
            if USE_MEMORY:
              val_memory_backup = model_ref.memory.backup_memory()
              # Restore memory we had at the end of training to be used when validating on new nodes.
              # Also backup memory after validation so it can be used for testing (since test edges are
              # strictly later in time than validation edges)
              model_ref.memory.restore_memory(train_memory_backup)

            # Validate on unseen nodes
            nn_val_ap, nn_val_auc = eval_edge_prediction(model=get_model_ref(),
                                                                           negative_edge_sampler=val_rand_sampler,
                                                                           data=new_node_val_data,
                                                                           n_neighbors=NUM_NEIGHBORS)

            if USE_MEMORY:
              # Restore memory we had at the end of validation
              model_ref.memory.restore_memory(val_memory_backup)
            
            # Restore wrapper methods after validation
            if USE_MEMORY and memory_wrapper is not None:
              model_ref.memory.store_raw_messages = memory_wrapper.store_raw_messages
              model_ref.memory.get_memory = memory_wrapper.get_memory
              model_ref.memory.get_last_update = memory_wrapper.get_last_update
              if memory_wrapper._orig_set_memory is not None:
                model_ref.memory.set_memory = memory_wrapper.set_memory

            new_nodes_val_aps.append(nn_val_ap)
            val_aps.append(val_ap)
            train_losses.append(np.mean(m_loss))
          else:
            # Other ranks skip validation but append dummy values to keep lists in sync
            val_ap, val_auc = 0.0, 0.0
            nn_val_ap, nn_val_auc = 0.0, 0.0
            new_nodes_val_aps.append(nn_val_ap)
            val_aps.append(val_ap)
            train_losses.append(np.mean(m_loss))
          
          # Broadcast validation results from rank 0 to all other ranks
          val_results = [val_ap, val_auc, nn_val_ap, nn_val_auc] if rank == 0 else [None, None, None, None]
          val_results = all_gather_object(val_results)
          if rank != 0:
            val_ap, val_auc, nn_val_ap, nn_val_auc = val_results[0]
            # Update the lists with broadcasted values
            val_aps[-1] = val_ap
            new_nodes_val_aps[-1] = nn_val_ap
        else:
          # Non-distributed mode: run validation normally
          tgn.set_neighbor_finder(full_ngh_finder)
          model_ref = get_model_ref()

          if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = model_ref.memory.backup_memory()

          val_ap, val_auc = eval_edge_prediction(model=get_model_ref(),
                                                                 negative_edge_sampler=val_rand_sampler,
                                                                 data=val_data,
                                                                 n_neighbors=NUM_NEIGHBORS)
          if USE_MEMORY:
            val_memory_backup = model_ref.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            model_ref.memory.restore_memory(train_memory_backup)

          # Validate on unseen nodes
          nn_val_ap, nn_val_auc = eval_edge_prediction(model=get_model_ref(),
                                                                       negative_edge_sampler=val_rand_sampler,
                                                                       data=new_node_val_data,
                                                                       n_neighbors=NUM_NEIGHBORS)

          if USE_MEMORY:
            # Restore memory we had at the end of validation
            model_ref.memory.restore_memory(val_memory_backup)

          new_nodes_val_aps.append(nn_val_ap)
          val_aps.append(val_ap)
          train_losses.append(np.mean(m_loss))

        # Save temporary results to disk (only rank 0 in distributed mode)
        if not args.distributed or rank == 0:
          pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
          }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        # ================= [新增：精确统计并打印 GPU 显存峰值] =================
        if torch.cuda.is_available():
            # 获取当前 Epoch 的显存分配峰值 (转换为 MB)
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            # 获取 PyTorch 缓存分配器保留的显存峰值 (类似 nvidia-smi 看到的数值)
            reserved_mem_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
            
            logger.info(f'Rank {rank} Epoch {epoch}: Peak Memory Allocated: {peak_mem_mb:.2f} MB, Reserved: {reserved_mem_mb:.2f} MB')
            
            # 重置峰值统计，以便下一个 Epoch 重新计算
            torch.cuda.reset_peak_memory_stats(device)
        # =====================================================================

        # logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        # logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('epoch: {} training took {:.2f}s, total time (with val) took {:.2f}s'.format(
            epoch, epoch_time, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
          'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
          'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

        # Early stopping (only rank 0 checks, then broadcasts decision)
        should_stop = False
        best_epoch = 0
        if not args.distributed or rank == 0:
          if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            should_stop = True
            best_epoch = early_stopper.best_epoch
          else:
            torch.save(get_model_ref().state_dict(), get_checkpoint_path(epoch))
        
        # In distributed mode, broadcast early stopping decision to all ranks
        if args.distributed and world_size > 1:
          barrier()
          stop_info = [should_stop, best_epoch] if rank == 0 else [None, None]
          stop_info_list = all_gather_object(stop_info)
          should_stop, best_epoch = stop_info_list[0]
        
        # if should_stop:
        #   logger.info(f'Loading the best model at epoch {best_epoch}')
        #   best_model_path = get_checkpoint_path(best_epoch)
        #   get_model_ref().load_state_dict(torch.load(best_model_path))
        #   logger.info(f'Loaded the best model at epoch {best_epoch} for inference')
        #   tgn.eval()
        #   break
        if should_stop:
            # 增加这一行判断，防止非0节点去读本地没有的文件
            if not args.distributed or rank == 0:
                logger.info(f'Loading the best model at epoch {best_epoch}')
                best_model_path = get_checkpoint_path(best_epoch)
                get_model_ref().load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {best_epoch} for inference')
                tgn.eval()
            break  # 所有机器都会跳出循环，只有 rank 0 获得了最好的模型去测试

      # Training has finished, we have loaded the best model, and we want to backup its current
      # memory (which has seen validation edges) so that it can also be used when testing on unseen
      # nodes
      
      ### Test
      # In distributed mode, only rank 0 performs testing
      if args.distributed and world_size > 1:
        if rank == 0:
          # Temporarily restore original memory methods to bypass wrapper during testing
          model_ref = get_model_ref()
          if USE_MEMORY and memory_wrapper is not None:
            model_ref.memory.store_raw_messages = memory_wrapper._orig_store_raw_messages
            model_ref.memory.get_memory = memory_wrapper._orig_get_memory
            model_ref.memory.get_last_update = memory_wrapper._orig_get_last_update
            if memory_wrapper._orig_set_memory is not None:
              model_ref.memory.set_memory = memory_wrapper._orig_set_memory
          
          if USE_MEMORY:
            val_memory_backup = model_ref.memory.backup_memory()

          model_ref.embedding_module.neighbor_finder = full_ngh_finder
          test_ap, test_auc = eval_edge_prediction(model=get_model_ref(),
                                                                     negative_edge_sampler=test_rand_sampler,
                                                                     data=test_data,
                                                                     n_neighbors=NUM_NEIGHBORS)

          if USE_MEMORY:
            model_ref.memory.restore_memory(val_memory_backup)

          # Test on unseen nodes
          nn_test_ap, nn_test_auc = eval_edge_prediction(model=get_model_ref(),
                                                                         negative_edge_sampler=nn_test_rand_sampler,
                                                                         data=new_node_test_data,
                                                                         n_neighbors=NUM_NEIGHBORS)

          # Restore wrapper methods after testing
          if USE_MEMORY and memory_wrapper is not None:
            model_ref.memory.store_raw_messages = memory_wrapper.store_raw_messages
            model_ref.memory.get_memory = memory_wrapper.get_memory
            model_ref.memory.get_last_update = memory_wrapper.get_last_update
            if memory_wrapper._orig_set_memory is not None:
              model_ref.memory.set_memory = memory_wrapper.set_memory

          logger.info(
            'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
          logger.info(
            'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
        else:
          # Other ranks skip testing
          test_ap, test_auc = 0.0, 0.0
          nn_test_ap, nn_test_auc = 0.0, 0.0
      else:
        # Non-distributed mode: run testing normally
        model_ref = get_model_ref()
        if USE_MEMORY:
          val_memory_backup = model_ref.memory.backup_memory()

        model_ref.embedding_module.neighbor_finder = full_ngh_finder
        test_ap, test_auc = eval_edge_prediction(model=get_model_ref(),
                                                                   negative_edge_sampler=test_rand_sampler,
                                                                   data=test_data,
                                                                   n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
          model_ref.memory.restore_memory(val_memory_backup)

        # Test on unseen nodes
        nn_test_ap, nn_test_auc = eval_edge_prediction(model=get_model_ref(),
                                                                       negative_edge_sampler=nn_test_rand_sampler,
                                                                       data=new_node_test_data,
                                                                       n_neighbors=NUM_NEIGHBORS)

        logger.info(
          'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
        logger.info(
          'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
      # Save results for this run (only rank 0 in distributed mode)
      if not args.distributed or rank == 0:
        pickle.dump({
          "val_aps": val_aps,
          "new_nodes_val_aps": new_nodes_val_aps,
          "test_ap": test_ap,
          "new_node_test_ap": nn_test_ap,
          "epoch_times": epoch_times,
          "train_losses": train_losses,
          "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving TGN model')
        if USE_MEMORY:
          # Restore memory at the end of validation (save a model which is ready for testing)
          get_model_ref().memory.restore_memory(val_memory_backup)
        torch.save(get_model_ref().state_dict(), MODEL_SAVE_PATH)
        logger.info('TGN model saved')

    # Clean up distributed training if enabled
    if args.distributed and world_size > 1:
        barrier()
        logger.info(f'Rank {rank}: Training completed successfully')
        destroy_process_group()
        logger.info(f'Rank {rank}: Process group destroyed')


if __name__ == "__main__":
    import os
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.distributed:
        if "LOCAL_RANK" in os.environ:
            # 【修复点】：如果是被 torchrun 启动的，直接运行主函数，拒绝套娃！
            local_rank = int(os.environ["LOCAL_RANK"])
            train_main(local_rank, args)
        else:
            # 如果是用户直接 python xxx.py 运行的（单机多卡），才使用 spawn
            from distributed.dist_utils import spawn_distributed

            master_port = os.environ.get("MASTER_PORT", "29500")
            spawn_distributed(
                train_main,
                args=(args,),
                nprocs=args.world_size,
                backend=args.backend,
                master_port=master_port
            )
    else:
        # 非分布式模式，直接运行
        train_main(0, args)
    # if args.distributed:
    #     # 使用 spawn 模式启动分布式训练
    #     from distributed.dist_utils import spawn_distributed
    #
    #     # 从环境变量获取端口，如果没有则默认使用 29500
    #     master_port = os.environ.get("MASTER_PORT", "29500")
    #
    #     spawn_distributed(
    #         train_main,
    #         args=(args,),
    #         nprocs=args.world_size,
    #         backend=args.backend,
    #         master_port=master_port
    #     )
    # else:
    #     # 非分布式模式，直接运行
    #     train_main(0, args)

