"""
Data Prefetcher for Distributed TGN Training
在每次通信后，对数据进行预拉取，实现在通信时拉取memory，不通信时进行本地重算。

预拉取规则：
1. 对于某个机器一个batch里面某个事件，如果有节点不在本地
2. 就将前面batch里面（上次通信之后）和该节点相关原来不在本地机器上的事件拉取到本地机器上进行训练
"""
from __future__ import annotations
from typing import Dict, List, Set, Optional, Union
from collections import defaultdict
import numpy as np
from utils.data_processing import Data
from distributed.range_partitioner import RangePartitioner, HashPartitioner
from distributed.dist_utils import all_gather_object, is_distributed, get_rank, get_world_size
import torch
import torch.distributed as dist


class DataPrefetcher:
    """
    数据预拉取器：在通信后预拉取相关数据，用于本地重算
    """

    def __init__(
        self,
        full_data: Data,
        partitioner: Union[RangePartitioner, HashPartitioner],
        rank: int,
        world_size: int,
        batch_size: int,
        use_both_sides: bool = False,
    ):
        """
        初始化数据预拉取器

        Args:
            full_data: 完整的数据（所有边）
            partitioner: 节点分区器
            rank: 当前进程的rank
            world_size: 总进程数
            batch_size: 每个batch的大小
            use_both_sides: 是否使用 Both-sides 边复制策略（NeutronStream 等 baseline 使用）
        """
        self.full_data = full_data
        self.partitioner = partitioner
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.use_both_sides = use_both_sides

        # 本地节点集合
        self.local_node_set = set(partitioner.get_local_nodes(rank))

        # 跟踪上次通信后的batch索引
        self.last_sync_batch_idx = -1

        # 本地缓存：存储预拉取的事件数据
        # key: batch_idx, value: Data - 该batch的预拉取数据
        self.prefetched_data_cache: Dict[int, Data] = {}

        # 跟踪每个节点在哪些batch中出现过（用于去重和优化）
        self.node_batch_map: Dict[int, Set[int]] = defaultdict(set)

        # 计算全局batch数量
        self.num_global_batches = (len(full_data.sources) + batch_size - 1) // batch_size

    def get_remote_nodes_in_batch(self, batch_idx: int) -> Set[int]:
        """
        获取指定batch中涉及的非本地节点

        Args:
            batch_idx: batch索引

        Returns:
            涉及的非本地节点集合
        """
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.full_data.sources))

        remote_nodes = set()
        for i in range(start_idx, end_idx):
            src = self.full_data.sources[i]
            dst = self.full_data.destinations[i]

            # 如果源节点不在本地，添加到远程节点集合
            if src not in self.local_node_set:
                remote_nodes.add(src)
            # 如果目标节点不在本地，添加到远程节点集合
            if dst not in self.local_node_set:
                remote_nodes.add(dst)

        return remote_nodes
    
    def get_communication_nodes(self, current_batch_idx: int, next_sync_batch_idx: Optional[int] = None) -> Set[int]:
        """
        计算在通信时需要通信的节点（不在本地的节点）

        Source-based 策略：只有源节点在本地的边才在本地训练，
        因此只需通信这些边的远程目标节点。

        Args:
            current_batch_idx: 当前batch索引（通信点）
            next_sync_batch_idx: 下次通信的batch索引（未使用，保留接口兼容性）

        Returns:
            需要通信的节点集合（本地训练边涉及的远程目标节点）
        """
        remote_nodes = set()
        start_idx = current_batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.full_data.sources))

        for i in range(start_idx, end_idx):
            src = self.full_data.sources[i]
            dst = self.full_data.destinations[i]

            if self.use_both_sides:
                # Both-sides 策略：源或目标节点在本地，边就在本地训练
                if src in self.local_node_set or dst in self.local_node_set:
                    if src not in self.local_node_set:
                        remote_nodes.add(src)
                    if dst not in self.local_node_set:
                        remote_nodes.add(dst)
            else:
                # Source-based 策略：只有源节点在本地，这条边才在本地训练
                if src in self.local_node_set:
                    if dst not in self.local_node_set:
                        remote_nodes.add(dst)

        return remote_nodes

    def extract_events_for_nodes(
        self,
        batch_start: int,
        batch_end: int,
        target_nodes: Set[int],
        requester_rank: Optional[int] = None
    ) -> Data:
        """
        从指定batch范围内提取与目标节点相关的事件
        只提取"请求方原来没有的事件"（即两个节点都不在请求方本地的事件）

        Args:
            batch_start: 起始batch索引（包含）
            batch_end: 结束batch索引（不包含）
            target_nodes: 目标节点集合（这些节点是其他进程请求的本地节点）
            requester_rank: 请求方的rank（用于判断哪些事件请求方已经有了）

        Returns:
            包含相关事件的Data对象（这些事件请求方原来没有）
        """
        event_indices = []
        
        # 获取请求方的节点集合
        if requester_rank is not None:
            requester_node_set = set(self.partitioner.get_local_nodes(requester_rank))
        else:
            requester_node_set = set()

        for batch_idx in range(batch_start, batch_end):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.full_data.sources))

            for i in range(start_idx, end_idx):
                src = self.full_data.sources[i]
                dst = self.full_data.destinations[i]

                # 提取条件：
                # 1. 这条边涉及目标节点中的任何一个（这些节点是本地节点，被其他进程请求）
                # 2. 这条边的两个节点都不在请求方本地（这样的事件请求方原来没有）
                # 如果 requester_rank 为 None，则只检查第一个条件
                if src in target_nodes or dst in target_nodes:
                    if requester_rank is None:
                        # 如果没有指定请求方，则提取所有涉及目标节点的事件
                        event_indices.append(i)
                    elif src not in requester_node_set and dst not in requester_node_set:
                        # 只提取两个节点都不在请求方本地的事件
                        event_indices.append(i)

        if not event_indices:
            # 如果没有相关事件，返回空数据
            return Data(
                sources=np.array([], dtype=self.full_data.sources.dtype),
                destinations=np.array([], dtype=self.full_data.destinations.dtype),
                timestamps=np.array([], dtype=self.full_data.timestamps.dtype),
                edge_idxs=np.array([], dtype=self.full_data.edge_idxs.dtype),
                labels=None
            )

        event_indices = np.array(event_indices)
        return Data(
            sources=self.full_data.sources[event_indices],
            destinations=self.full_data.destinations[event_indices],
            timestamps=self.full_data.timestamps[event_indices],
            edge_idxs=self.full_data.edge_idxs[event_indices],
            labels=self.full_data.labels[event_indices] if self.full_data.labels is not None else None
        )

    # def compute_needed_nodes_per_batch(self, current_batch_idx: int, next_sync_batch_idx: Optional[int] = None) -> Dict[
    #     int, Set[int]]:
    #     """
    #     计算每个batch需要的所有非本地节点（用于预拉取）
    #
    #     从next_sync_batch-1往前扫描到current_batch（包括current_batch），逐batch累积需要的节点
    #
    #     Args:
    #         current_batch_idx: 当前batch索引（通信点）
    #         next_sync_batch_idx: 下次通信的batch索引。如果为None，则计算到最后一个batch
    #
    #     Returns:
    #         Dict[batch_idx, Set[node_id]]：每个batch需要的所有非本地节点
    #     """
    #     if next_sync_batch_idx is None:
    #         next_sync_batch_idx = self.num_global_batches
    #
    #     # 如果下次通信点就是current_batch+1，说明只有current_batch需要处理
    #     if next_sync_batch_idx <= current_batch_idx:
    #         return {}
    #
    #     needed_nodes_per_batch = {}
    #
    #     # 从next_sync_batch-1往前扫描到current_batch（包括current_batch）
    #     for batch_idx in range(next_sync_batch_idx - 1, current_batch_idx - 1, -1):
    #         # 1. 这个batch中本地事件涉及的非本地节点
    #         local_remote_nodes = set()
    #         start_idx = batch_idx * self.batch_size
    #         end_idx = min(start_idx + self.batch_size, len(self.full_data.sources))
    #
    #         for i in range(start_idx, end_idx):
    #             src = self.full_data.sources[i]
    #             dst = self.full_data.destinations[i]
    #
    #             if src in self.local_node_set or dst in self.local_node_set:
    #                 if src not in self.local_node_set:
    #                     local_remote_nodes.add(src)
    #                 if dst not in self.local_node_set:
    #                     local_remote_nodes.add(dst)
    #
    #         # 2. 从后面batch传播来的节点
    #         propagated_nodes = set()
    #         if batch_idx + 1 in needed_nodes_per_batch:
    #             propagated_nodes = needed_nodes_per_batch[batch_idx + 1].copy()
    #
    #         # 3. 传播节点在当前batch的事件可能引入新节点
    #         new_nodes = set()
    #         if propagated_nodes:
    #             for i in range(start_idx, end_idx):
    #                 src = self.full_data.sources[i]
    #                 dst = self.full_data.destinations[i]
    #
    #                 if src in propagated_nodes or dst in propagated_nodes:
    #                     if src not in self.local_node_set and src not in local_remote_nodes and src not in propagated_nodes:
    #                         new_nodes.add(src)
    #                     if dst not in self.local_node_set and dst not in local_remote_nodes and dst not in propagated_nodes:
    #                         new_nodes.add(dst)
    #
    #         # 4. 合并所有节点
    #         needed_nodes_per_batch[batch_idx] = local_remote_nodes | propagated_nodes | new_nodes
    #
    #     return needed_nodes_per_batch

    def compute_needed_nodes_per_batch(self, current_batch_idx: int, next_sync_batch_idx: Optional[int] = None) -> Dict[
        int, Set[int]]:
        """
        计算每个batch需要的所有非本地节点（用于预拉取）

        从next_sync_batch-1往前扫描到current_batch（包括current_batch），逐batch累积需要的节点

        Args:
            current_batch_idx: 当前batch索引（通信点）
            next_sync_batch_idx: 下次通信的batch索引。如果为None，则计算到最后一个batch

        Returns:
            Dict[batch_idx, Set[node_id]]：每个batch需要的所有非本地节点（原始版本）
            同时会计算 needed_nodes_per_batch_delayed 并以元组形式返回
        """
        if next_sync_batch_idx is None:
            next_sync_batch_idx = self.num_global_batches

        # 如果下次通信点就是current_batch+1，说明只有current_batch需要处理
        if next_sync_batch_idx <= current_batch_idx:
            return {}, {}, {}

        needed_nodes_per_batch = {}
        needed_nodes_per_batch_delayed = {}

        # 用于跟踪已经出现过的节点（从next_sync_batch往前扫描的过程中）
        seen_nodes = set()

        # 从next_sync_batch-1往前扫描到current_batch（包括current_batch）
        for batch_idx in range(next_sync_batch_idx - 1, current_batch_idx - 1, -1):
            # 1. 这个batch中本地事件涉及的非本地节点
            local_remote_nodes = set()
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.full_data.sources))

            for i in range(start_idx, end_idx):
                src = self.full_data.sources[i]
                dst = self.full_data.destinations[i]

                if self.use_both_sides:
                    # Both-sides 策略：源或目标节点在本地，边才分配给当前进程
                    if src in self.local_node_set or dst in self.local_node_set:
                        if src not in self.local_node_set:
                            local_remote_nodes.add(src)
                        if dst not in self.local_node_set:
                            local_remote_nodes.add(dst)
                else:
                    # Source-based 策略：仅源节点在本地，边才分配给当前进程
                    if src in self.local_node_set:
                        if dst not in self.local_node_set:
                            local_remote_nodes.add(dst)

            # 2. 从后面batch传播来的节点
            propagated_nodes = set()
            if batch_idx + 1 in needed_nodes_per_batch:
                propagated_nodes = needed_nodes_per_batch[batch_idx + 1].copy()

            # 3. 传播节点在当前batch的事件可能引入新节点
            new_nodes = set()
            if propagated_nodes:
                for i in range(start_idx, end_idx):
                    src = self.full_data.sources[i]
                    dst = self.full_data.destinations[i]

                    if src in propagated_nodes or dst in propagated_nodes:
                        if src not in self.local_node_set and src not in local_remote_nodes and src not in propagated_nodes:
                            new_nodes.add(src)
                        if dst not in self.local_node_set and dst not in local_remote_nodes and dst not in propagated_nodes:
                            new_nodes.add(dst)

            # 4. 合并所有节点（原始版本）
            needed_nodes_per_batch[batch_idx] = local_remote_nodes | propagated_nodes | new_nodes

            # 5. 构建延迟版本
            all_current_nodes = local_remote_nodes | propagated_nodes | new_nodes

            # 识别首次出现的节点
            first_time_nodes = all_current_nodes - seen_nodes
            # 识别已经出现过的节点
            already_seen_nodes = all_current_nodes & seen_nodes

            # 更新seen_nodes
            seen_nodes.update(all_current_nodes)

            # 当前batch添加已经见过的节点
            needed_nodes_per_batch_delayed[batch_idx] = already_seen_nodes.copy()

            # 首次出现的节点尝试添加到前一个batch
            if first_time_nodes and batch_idx - 1 >= current_batch_idx:
                if batch_idx - 1 not in needed_nodes_per_batch_delayed:
                    needed_nodes_per_batch_delayed[batch_idx - 1] = set()
                needed_nodes_per_batch_delayed[batch_idx - 1].update(first_time_nodes)

        return needed_nodes_per_batch, needed_nodes_per_batch_delayed

    def prefetch_after_sync(self, current_batch_idx: int, next_sync_batch_idx: Optional[int] = None):
        """
        在通信后执行预拉取（使用点对点通信）

        预拉取规则：
        1. 计算每个batch（从current_batch到next_sync_batch-1）需要的非本地节点
        2. 对于每个batch N，拉取该batch N自己时间范围内这些节点的事件（本来不在本地的）
        3. 按batch组织预拉取数据，训练batch N时使用prefetched_data_cache[N]

        Args:
            current_batch_idx: 当前batch索引（刚完成通信的batch）
            next_sync_batch_idx: 下次通信的batch索引。如果为None，则计算到最后一个batch
        """
        if self.world_size == 1:
            self.last_sync_batch_idx = current_batch_idx
            return

        # 计算每个batch需要的非本地节点
        needed_nodes_per_batch = self.compute_needed_nodes_per_batch(current_batch_idx, next_sync_batch_idx)

        if not needed_nodes_per_batch:
            # 如果没有需要预拉取的batch，直接更新同步点
            self.last_sync_batch_idx = current_batch_idx
            return

        # 对每个batch进行预拉取
        for target_batch_idx in sorted(needed_nodes_per_batch.keys()):
            remote_nodes = needed_nodes_per_batch[target_batch_idx]
            
            if not remote_nodes:
                continue
            
            # 提取该batch的预拉取数据（直接从full_data）
            # 在单机环境中，每个进程可以访问full_data
            # 在真实分布式环境中，full_data只包含本地数据
            prefetch_data = self.extract_prefetch_data_for_batch(target_batch_idx, remote_nodes)
            
            if prefetch_data is not None:
                self.prefetched_data_cache[target_batch_idx] = prefetch_data

        # 更新同步点
        self.last_sync_batch_idx = current_batch_idx
    
    def prefetch_after_sync_p2p(self, current_batch_idx: int, next_sync_batch_idx: Optional[int] = None):
        """
        在通信后执行预拉取（使用点对点通信，真实分布式环境）

        通过torch.distributed的send/recv实现点对点数据传输

        Args:
            current_batch_idx: 当前batch索引（刚完成通信的batch）
            next_sync_batch_idx: 下次通信的batch索引。如果为None，则计算到最后一个batch
        """
        if not is_distributed() or self.world_size == 1:
            self.last_sync_batch_idx = current_batch_idx
            return

        # 计算每个batch需要的非本地节点
        needed_nodes_per_batch = self.compute_needed_nodes_per_batch(current_batch_idx, next_sync_batch_idx)

        if not needed_nodes_per_batch:
            self.last_sync_batch_idx = current_batch_idx
            return

        # 对每个batch进行点对点通信预拉取
        for target_batch_idx in sorted(needed_nodes_per_batch.keys()):
            remote_nodes = needed_nodes_per_batch[target_batch_idx]
            
            if not remote_nodes:
                continue
            
            # 确定哪些节点属于哪个rank
            rank_to_nodes = {}  # rank -> set of nodes
            for node_id in remote_nodes:
                owner_rank = self.partitioner.owner(node_id)
                if owner_rank != self.rank and owner_rank not in rank_to_nodes:
                    rank_to_nodes[owner_rank] = set()
                if owner_rank != self.rank:
                    rank_to_nodes[owner_rank].add(node_id)
            
            # 向每个拥有所需节点的rank发送请求并接收数据
            received_data_list = []
            for owner_rank, requested_nodes in rank_to_nodes.items():
                # 发送请求：(target_batch_idx, requested_nodes)
                request = {
                    "target_batch": target_batch_idx,
                    "requested_nodes": list(requested_nodes),
                    "requester_rank": self.rank
                }
                
                # 使用torch.distributed发送请求
                request_tensor = self._serialize_request(request)
                dist.send(request_tensor, dst=owner_rank)
                
                # 接收响应数据
                # 首先接收数据大小
                size_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(size_tensor, src=owner_rank)
                data_size = size_tensor.item()
                
                if data_size > 0:
                    # 接收实际数据
                    data_tensor = torch.zeros(data_size, dtype=torch.uint8)
                    dist.recv(data_tensor, src=owner_rank)
                    data = self._deserialize_data(data_tensor)
                    if data is not None and len(data.sources) > 0:
                        received_data_list.append(data)
            
            # 处理其他rank的请求（作为发送方）
            for _ in range(self.world_size - 1):
                # 检查是否有请求
                # 这里需要非阻塞接收，实际实现需要协调
                pass
            
            # 合并接收到的数据
            if received_data_list:
                all_sources = np.concatenate([d.sources for d in received_data_list])
                all_destinations = np.concatenate([d.destinations for d in received_data_list])
                all_timestamps = np.concatenate([d.timestamps for d in received_data_list])
                all_edge_idxs = np.concatenate([d.edge_idxs for d in received_data_list])
                
                sort_indices = np.argsort(all_timestamps)
                
                self.prefetched_data_cache[target_batch_idx] = Data(
                    sources=all_sources[sort_indices],
                    destinations=all_destinations[sort_indices],
                    timestamps=all_timestamps[sort_indices],
                    edge_idxs=all_edge_idxs[sort_indices],
                    labels=None
                )

        # 更新同步点
        self.last_sync_batch_idx = current_batch_idx
    
    def _serialize_request(self, request: dict) -> torch.Tensor:
        """序列化请求为tensor"""
        import pickle
        data_bytes = pickle.dumps(request)
        return torch.ByteTensor(list(data_bytes))
    
    def _deserialize_data(self, tensor: torch.Tensor) -> Optional[Data]:
        """反序列化tensor为Data对象"""
        import pickle
        data_bytes = bytes(tensor.cpu().numpy())
        return pickle.loads(data_bytes)

    def extract_prefetch_data_for_batch(self, target_batch_idx: int, remote_nodes: Set[int]) -> Optional[Data]:
        """
        从指定batch中提取预拉取数据
        
        提取该batch中涉及remote_nodes的、源节点不在本地的事件（Source-based 预拉取边）
        
        Args:
            target_batch_idx: 目标batch索引
            remote_nodes: 需要预拉取的非本地节点集合
        
        Returns:
            预拉取的数据，如果没有则返回None
        """
        if not remote_nodes:
            return None
        
        batch_start = target_batch_idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.full_data.sources))
        
        prefetch_events = []
        for i in range(batch_start, batch_end):
            src = self.full_data.sources[i]
            dst = self.full_data.destinations[i]

            if self.use_both_sides:
                # Both-sides 策略：两端都不在本地的边才是纯预拉取边
                if (src in remote_nodes or dst in remote_nodes) and \
                   (src not in self.local_node_set and dst not in self.local_node_set):
                    prefetch_events.append(i)
            else:
                # Source-based 策略：源节点不在本地的边就是预拉取边
                if (src in remote_nodes or dst in remote_nodes) and (src not in self.local_node_set):
                    prefetch_events.append(i)

        if not prefetch_events:
            return None

        prefetch_events = np.array(prefetch_events)
        return Data(
            sources=self.full_data.sources[prefetch_events],
            destinations=self.full_data.destinations[prefetch_events],
            timestamps=self.full_data.timestamps[prefetch_events],
            edge_idxs=self.full_data.edge_idxs[prefetch_events],
            labels=self.full_data.labels[prefetch_events] if self.full_data.labels is not None else None
        )

    def get_prefetched_data_for_batch(self, batch_idx: int) -> Optional[Data]:
        """
        获取指定batch的预拉取数据（用于本地重算）
        返回的数据已按时间戳排序

        Args:
            batch_idx: batch索引

        Returns:
            预拉取的数据（按时间戳排序），如果没有则返回None
        """
        return self.prefetched_data_cache.get(batch_idx, None)

    def clear_cache(self):
        """清空预拉取缓存（在每个epoch开始时调用）"""
        self.prefetched_data_cache.clear()
        self.node_batch_map.clear()
        self.last_sync_batch_idx = -1

    def reset(self):
        """重置预拉取器（等同于clear_cache）"""
        self.clear_cache()


class DistGLDataPrefetcher(DataPrefetcher):
    """
    专为 DistGL 对比实验设计的预拉取器。
    核心区别：DistGL 采用基于源节点 (Source-based) 的边划分策略。
    一条边只有在其源节点属于本地时，才会在本地被训练。
    """

    def get_communication_nodes(self, current_batch_idx: int, next_sync_batch_idx: Optional[int] = None) -> Set[int]:
        remote_nodes = set()
        start_idx = current_batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.full_data.sources))

        for i in range(start_idx, end_idx):
            src = self.full_data.sources[i]
            dst = self.full_data.destinations[i]

            # 【DistGL 核心逻辑】：只有源节点在本地，这条边才在本地训练
            if src in self.local_node_set:
                # 既然源节点在本地，那么只有目标节点可能不在本地
                if dst not in self.local_node_set:
                    remote_nodes.add(dst)

        return remote_nodes

    def compute_needed_nodes_per_batch(self, current_batch_idx: int, next_sync_batch_idx: Optional[int] = None) -> tuple:
        if next_sync_batch_idx is None:
            next_sync_batch_idx = self.num_global_batches

        if next_sync_batch_idx <= current_batch_idx:
            return {}, {}

        needed_nodes_per_batch = {}
        needed_nodes_per_batch_delayed = {}
        seen_nodes = set()

        for batch_idx in range(next_sync_batch_idx - 1, current_batch_idx - 1, -1):
            local_remote_nodes = set()
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.full_data.sources))

            for i in range(start_idx, end_idx):
                src = self.full_data.sources[i]
                dst = self.full_data.destinations[i]

                # 【DistGL 核心逻辑】：仅源节点在本地时，边才分配给当前进程
                if src in self.local_node_set:
                    if dst not in self.local_node_set:
                        local_remote_nodes.add(dst)

            propagated_nodes = set()
            if batch_idx + 1 in needed_nodes_per_batch:
                propagated_nodes = needed_nodes_per_batch[batch_idx + 1].copy()

            new_nodes = set()
            if propagated_nodes:
                for i in range(start_idx, end_idx):
                    src = self.full_data.sources[i]
                    dst = self.full_data.destinations[i]

                    if src in propagated_nodes or dst in propagated_nodes:
                        if src not in self.local_node_set and src not in local_remote_nodes and src not in propagated_nodes:
                            new_nodes.add(src)
                        if dst not in self.local_node_set and dst not in local_remote_nodes and dst not in propagated_nodes:
                            new_nodes.add(dst)

            needed_nodes_per_batch[batch_idx] = local_remote_nodes | propagated_nodes | new_nodes

            all_current_nodes = local_remote_nodes | propagated_nodes | new_nodes
            first_time_nodes = all_current_nodes - seen_nodes
            already_seen_nodes = all_current_nodes & seen_nodes
            seen_nodes.update(all_current_nodes)

            needed_nodes_per_batch_delayed[batch_idx] = already_seen_nodes.copy()

            if first_time_nodes and batch_idx - 1 >= current_batch_idx:
                if batch_idx - 1 not in needed_nodes_per_batch_delayed:
                    needed_nodes_per_batch_delayed[batch_idx - 1] = set()
                needed_nodes_per_batch_delayed[batch_idx - 1].update(first_time_nodes)

        return needed_nodes_per_batch, needed_nodes_per_batch_delayed

    def extract_prefetch_data_for_batch(self, target_batch_idx: int, remote_nodes: Set[int]) -> Optional[Data]:
        if not remote_nodes:
            return None

        batch_start = target_batch_idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.full_data.sources))

        prefetch_events = []
        for i in range(batch_start, batch_end):
            src = self.full_data.sources[i]
            dst = self.full_data.destinations[i]

            # 【DistGL 核心逻辑】：
            # 1. 涉及需要的节点
            # 2. 这条边本来不在本地（对于 DistGL 来说，源节点不在本地，这条边就不在本地）
            if (src in remote_nodes or dst in remote_nodes) and (src not in self.local_node_set):
                prefetch_events.append(i)

        if not prefetch_events:
            return None

        prefetch_events = np.array(prefetch_events)
        return Data(
            sources=self.full_data.sources[prefetch_events],
            destinations=self.full_data.destinations[prefetch_events],
            timestamps=self.full_data.timestamps[prefetch_events],
            edge_idxs=self.full_data.edge_idxs[prefetch_events],
            labels=self.full_data.labels[prefetch_events] if self.full_data.labels is not None else None
        )

