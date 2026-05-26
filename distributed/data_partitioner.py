"""
Data partitioning utilities for distributed TGN training.
按照节点（顶点）划分，然后将事件（边）分配给相关分区。
如果一个事件涉及两个不同分区的节点，该事件会被复制到这两个分区。
"""
import numpy as np
from typing import Union
from utils.data_processing import Data
from distributed.range_partitioner import RangePartitioner, HashPartitioner
from distributed.dist_utils import get_rank, get_world_size
from distributed.data_prefetcher import DataPrefetcher


def partition_data(data: Data, partitioner: Union[RangePartitioner, HashPartitioner], rank: int, 
                   include_edges_with_remote_nodes: bool = True):
    """
    按照节点（顶点）划分数据。
    
    分区策略：
    1. 首先按照节点划分（每个进程拥有一部分节点）
    2. 对于每条边（事件）：
       - 如果边的源节点或目标节点属于当前进程，则包含该边
       - 如果一条边涉及两个不同分区的节点，该边会被复制到这两个分区
       - 这样每个分区可以更新自己拥有的节点
    
    Args:
        data: Data object containing sources, destinations, timestamps, etc.
        partitioner: RangePartitioner or HashPartitioner instance
        rank: Current rank
        include_edges_with_remote_nodes: If True, include edges where at least one endpoint 
                                        is local (用于更新本地节点). If False, only include 
                                        edges where both endpoints are local.
    
    Returns:
        Partitioned Data object - 包含所有涉及本地节点的事件（边）
    """
    local_node_set = set(partitioner.get_local_nodes(rank))
    
    # 对于每条边，检查是否涉及本地节点
    # 如果一条边涉及本地节点（源节点或目标节点），则包含该边
    # 这样如果一条边涉及两个不同分区的节点，它会被复制到两个分区
    if include_edges_with_remote_nodes:
        # 包含所有涉及本地节点的边（源节点或目标节点至少有一个是本地节点）
        # 这样跨分区的边会被复制到相关分区
        mask = np.array([
            (src in local_node_set or dst in local_node_set)
            for src, dst in zip(data.sources, data.destinations)
        ])
    else:
        # 只包含两个端点都是本地节点的边
        mask = np.array([
            (src in local_node_set and dst in local_node_set)
            for src, dst in zip(data.sources, data.destinations)
        ])
    
    return Data(
        sources=data.sources[mask],
        destinations=data.destinations[mask],
        timestamps=data.timestamps[mask],
        edge_idxs=data.edge_idxs[mask],
        labels=data.labels[mask] if hasattr(data, 'labels') and data.labels is not None else None
    )


def get_partitioned_data(data: Data, num_nodes: int, rank: int = None, 
                         world_size: int = None):
    """
    Helper function to get partitioned data.
    
    Args:
        data: Original Data object
        num_nodes: Total number of nodes
        rank: Current rank (if None, uses dist_utils.get_rank())
        world_size: World size (if None, uses dist_utils.get_world_size())
    
    Returns:
        Partitioned Data object
    """
    if rank is None:
        rank = get_rank()
    if world_size is None:
        world_size = get_world_size()
    
    partitioner = RangePartitioner(num_nodes=num_nodes, world_size=world_size)
    return partition_data(data, partitioner, rank, include_edges_with_remote_nodes=True)


def partition_data_by_batches(data: Data, partitioner: Union[RangePartitioner, HashPartitioner], 
                               rank: int, batch_size: int, world_size: int = None,
                               sync_every: int = None, enable_prefetch: bool = True,
                               use_both_sides: bool = False):
    """
    先将所有数据划分成 batch，然后按照节点划分将每个 batch 中的边分配给对应的进程。

    分区策略由 use_both_sides 控制：
    - use_both_sides=False (默认, Source-based)：只有边的源节点属于当前进程才包含该边，
      不产生边复制，适用于 DepTGL。
    - use_both_sides=True (Both-sides)：源节点或目标节点属于当前进程即包含该边，
      跨分区边会被复制到两侧，适用于 NeutronStream 等 baseline。

    Args:
        data: Data object containing sources, destinations, timestamps, etc.
        partitioner: RangePartitioner or HashPartitioner instance
        rank: Current rank
        batch_size: Size of each batch (全局 batch size，所有进程共享)
        world_size: 总进程数
        sync_every: 同步间隔（用于计算预拉取）。如果为None，不进行预拉取
        enable_prefetch: 是否启用预拉取功能
        use_both_sides: 是否使用 Both-sides 边复制策略（NeutronStream baseline 使用）

    Returns:
        Tuple of (batches, communication_nodes_per_batch):
        - batches: List of Data objects, each representing a batch
        - communication_nodes_per_batch: Dict[batch_idx, Set[node_id]] 只包含通信batch的节点
    """
    local_node_set = set(partitioner.get_local_nodes(rank))
    
    # 第一步：先将所有数据划分成 batch（全局划分）
    num_edges = len(data.sources)
    num_batches = (num_edges + batch_size - 1) // batch_size
    
    # 第二步：对于每个 batch，按照节点划分分配给当前进程
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_edges)
        
        # 获取这个 batch 中的所有边
        batch_sources = data.sources[start_idx:end_idx]
        batch_destinations = data.destinations[start_idx:end_idx]
        batch_timestamps = data.timestamps[start_idx:end_idx]
        batch_edge_idxs = data.edge_idxs[start_idx:end_idx]
        batch_labels = data.labels[start_idx:end_idx] if hasattr(data, 'labels') and data.labels is not None else None
        
        # 按照节点划分：根据策略选择该 batch 中属于本进程的边
        if use_both_sides:
            # Both-sides 策略：源节点或目标节点在本地即包含（跨分区边被复制到两侧）
            batch_mask = np.array([
                (src in local_node_set or dst in local_node_set)
                for src, dst in zip(batch_sources, batch_destinations)
            ])
        else:
            # Source-based 策略：只有源节点在本地才包含（无边复制）
            batch_mask = np.array([
                (src in local_node_set)
                for src in batch_sources
            ])
        
        # 如果这个 batch 中有涉及本地节点的边，则包含这个 batch
        if batch_mask.any():
            batch_data = Data(
                sources=batch_sources[batch_mask],
                destinations=batch_destinations[batch_mask],
                timestamps=batch_timestamps[batch_mask],
                edge_idxs=batch_edge_idxs[batch_mask],
                labels=batch_labels[batch_mask] if batch_labels is not None else None
            )
            batches.append(batch_data)
        else:
            # 如果这个 batch 中没有涉及本地节点的边，添加一个空的 batch
            batches.append(Data(
                sources=np.array([], dtype=batch_sources.dtype),
                destinations=np.array([], dtype=batch_destinations.dtype),
                timestamps=np.array([], dtype=batch_timestamps.dtype),
                edge_idxs=np.array([], dtype=batch_edge_idxs.dtype),
                labels=None
            ))
    
    # 第三步：如果启用预拉取，计算并添加预拉取的事件
    # communication_nodes_per_batch: Dict[batch_idx, Set[node_id]]
    # 保存每个通信batch需要通信的节点（包括从当前通信batch到下次通信batch之间所有batch需要的节点）
    communication_nodes_per_batch = {}
    
    if enable_prefetch and sync_every is not None and sync_every > 0:
        if world_size is None:
            world_size = get_world_size()
        
        # 创建DataPrefetcher来计算预拉取数据
        prefetcher = DataPrefetcher(
            full_data=data,
            partitioner=partitioner,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
            use_both_sides=use_both_sides
        )
        
        # 对每个同步周期进行预拉取计算
        sync_batch_indices = list(range(0, num_batches, sync_every))
        if sync_batch_indices[-1] != num_batches:
            sync_batch_indices.append(num_batches)
        
        for sync_idx in range(len(sync_batch_indices) - 1):
            current_sync_batch = sync_batch_indices[sync_idx]
            next_sync_batch = sync_batch_indices[sync_idx + 1]
            
            # 计算这个周期内每个batch需要的节点
            needed_nodes_per_batch,needed_nodes_per_batch_delayed = prefetcher.compute_needed_nodes_per_batch(
                current_sync_batch, next_sync_batch
            )
            
            # 保存从当前通信batch到下次通信batch之间所有需要通信的节点
            # 合并这个周期内所有batch需要的节点
            all_communication_nodes = set()
            for batch_idx, remote_nodes in needed_nodes_per_batch.items():
                all_communication_nodes.update(remote_nodes)
            
            # 同时包含当前通信batch本身需要的节点
            current_batch_communication_nodes = prefetcher.get_communication_nodes(current_sync_batch, next_sync_batch)
            all_communication_nodes.update(current_batch_communication_nodes)
            
            if all_communication_nodes:
                communication_nodes_per_batch[current_sync_batch] = all_communication_nodes
            
            # 对于每个batch，提取并添加预拉取数据
            for batch_idx, remote_nodes in needed_nodes_per_batch_delayed.items():
                if batch_idx >= len(batches) or not remote_nodes:
                    continue
                
                # 提取预拉取数据
                prefetch_data = prefetcher.extract_prefetch_data_for_batch(batch_idx, remote_nodes)
                
                if prefetch_data is not None and len(prefetch_data.sources) > 0:
                    # 合并预拉取数据到对应batch
                    original_batch = batches[batch_idx]
                    
                    if len(original_batch.sources) > 0:
                        # 合并原始数据和预拉取数据
                        combined_sources = np.concatenate([original_batch.sources, prefetch_data.sources])
                        combined_destinations = np.concatenate([original_batch.destinations, prefetch_data.destinations])
                        combined_timestamps = np.concatenate([original_batch.timestamps, prefetch_data.timestamps])
                        combined_edge_idxs = np.concatenate([original_batch.edge_idxs, prefetch_data.edge_idxs])
                        combined_labels = None
                        if original_batch.labels is not None and prefetch_data.labels is not None:
                            combined_labels = np.concatenate([original_batch.labels, prefetch_data.labels])
                        
                        # 按时间戳排序
                        sort_idx = np.argsort(combined_timestamps)
                        
                        batches[batch_idx] = Data(
                            sources=combined_sources[sort_idx],
                            destinations=combined_destinations[sort_idx],
                            timestamps=combined_timestamps[sort_idx],
                            edge_idxs=combined_edge_idxs[sort_idx],
                            labels=combined_labels[sort_idx] if combined_labels is not None else None
                        )
                    else:
                        # 原始batch为空，直接使用预拉取数据
                        batches[batch_idx] = prefetch_data
    
    return batches, communication_nodes_per_batch


def partition_data_by_batches_distgl(data: Data, partitioner: Union[RangePartitioner, HashPartitioner],
                                     rank: int, batch_size: int, world_size: int = None,
                                     sync_every: int = None, enable_prefetch: bool = True):
    """
    专为 DistGL 对比实验设计的分区函数。
    采用基于源节点 (Source-based) 的边划分策略，避免了边的冗余复制。
    """
    local_node_set = set(partitioner.get_local_nodes(rank))

    num_edges = len(data.sources)
    num_batches = (num_edges + batch_size - 1) // batch_size

    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_edges)

        batch_sources = data.sources[start_idx:end_idx]
        batch_destinations = data.destinations[start_idx:end_idx]
        batch_timestamps = data.timestamps[start_idx:end_idx]
        batch_edge_idxs = data.edge_idxs[start_idx:end_idx]
        batch_labels = data.labels[start_idx:end_idx] if hasattr(data, 'labels') and data.labels is not None else None

        # 【DistGL 核心逻辑】：只根据源节点划分边，不产生边复制
        batch_mask = np.array([
            (src in local_node_set)
            for src in batch_sources
        ])

        if batch_mask.any():
            batch_data = Data(
                sources=batch_sources[batch_mask],
                destinations=batch_destinations[batch_mask],
                timestamps=batch_timestamps[batch_mask],
                edge_idxs=batch_edge_idxs[batch_mask],
                labels=batch_labels[batch_mask] if batch_labels is not None else None
            )
            batches.append(batch_data)
        else:
            batches.append(Data(
                sources=np.array([], dtype=batch_sources.dtype),
                destinations=np.array([], dtype=batch_destinations.dtype),
                timestamps=np.array([], dtype=batch_timestamps.dtype),
                edge_idxs=np.array([], dtype=batch_edge_idxs.dtype),
                labels=None
            ))

    communication_nodes_per_batch = {}

    if enable_prefetch and sync_every is not None and sync_every > 0:
        if world_size is None:
            world_size = get_world_size()

        # 引入 DistGL 专用预拉取器
        from distributed.data_prefetcher import DistGLDataPrefetcher
        prefetcher = DistGLDataPrefetcher(
            full_data=data,
            partitioner=partitioner,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size
        )

        sync_batch_indices = list(range(0, num_batches, sync_every))
        if sync_batch_indices[-1] != num_batches:
            sync_batch_indices.append(num_batches)

        for sync_idx in range(len(sync_batch_indices) - 1):
            current_sync_batch = sync_batch_indices[sync_idx]
            next_sync_batch = sync_batch_indices[sync_idx + 1]

            needed_nodes_per_batch, needed_nodes_per_batch_delayed = prefetcher.compute_needed_nodes_per_batch(
                current_sync_batch, next_sync_batch
            )

            all_communication_nodes = set()
            for batch_idx, remote_nodes in needed_nodes_per_batch.items():
                all_communication_nodes.update(remote_nodes)

            current_batch_communication_nodes = prefetcher.get_communication_nodes(current_sync_batch, next_sync_batch)
            all_communication_nodes.update(current_batch_communication_nodes)

            if all_communication_nodes:
                communication_nodes_per_batch[current_sync_batch] = all_communication_nodes

            for batch_idx, remote_nodes in needed_nodes_per_batch_delayed.items():
                if batch_idx >= len(batches) or not remote_nodes:
                    continue

                prefetch_data = prefetcher.extract_prefetch_data_for_batch(batch_idx, remote_nodes)

                if prefetch_data is not None and len(prefetch_data.sources) > 0:
                    original_batch = batches[batch_idx]

                    if len(original_batch.sources) > 0:
                        combined_sources = np.concatenate([original_batch.sources, prefetch_data.sources])
                        combined_destinations = np.concatenate([original_batch.destinations, prefetch_data.destinations])
                        combined_timestamps = np.concatenate([original_batch.timestamps, prefetch_data.timestamps])
                        combined_edge_idxs = np.concatenate([original_batch.edge_idxs, prefetch_data.edge_idxs])
                        combined_labels = None
                        if original_batch.labels is not None and prefetch_data.labels is not None:
                            combined_labels = np.concatenate([original_batch.labels, prefetch_data.labels])

                        sort_idx = np.argsort(combined_timestamps)

                        batches[batch_idx] = Data(
                            sources=combined_sources[sort_idx],
                            destinations=combined_destinations[sort_idx],
                            timestamps=combined_timestamps[sort_idx],
                            edge_idxs=combined_edge_idxs[sort_idx],
                            labels=combined_labels[sort_idx] if combined_labels is not None else None
                        )
                    else:
                        batches[batch_idx] = prefetch_data

    return batches, communication_nodes_per_batch

