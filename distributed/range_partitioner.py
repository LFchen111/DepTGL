"""
Range-based partitioner for distributing nodes across machines.
For 100 nodes and 4 machines:
- Machine 0: nodes 0-24
- Machine 1: nodes 25-49
- Machine 2: nodes 50-74
- Machine 3: nodes 75-99
"""


class RangePartitioner:
    def __init__(self, num_nodes: int, world_size: int):
        """
        Initialize range partitioner.
        
        Args:
            num_nodes: Total number of nodes in the graph
            world_size: Number of machines/processes
        """
        self.num_nodes = num_nodes
        self.world_size = world_size
        
        # Compute nodes per partition
        nodes_per_partition = num_nodes // world_size
        self.remainder = num_nodes % world_size
        
        # Calculate ranges for each rank
        self.ranges = []
        start = 0
        for rank in range(world_size):
            # Distribute remainder nodes across first few partitions
            extra = 1 if rank < self.remainder else 0
            end = start + nodes_per_partition + extra
            self.ranges.append((start, end))
            start = end
    
    def owner(self, node_id: int) -> int:
        """
        Get the rank that owns a given node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Rank that owns this node
        """
        for rank in range(self.world_size):
            start, end = self.ranges[rank]
            if start <= node_id < end:
                return rank
        # If node_id >= num_nodes, assign to last rank
        return self.world_size - 1
    
    def is_local(self, node_id: int, rank: int) -> bool:
        """
        Check if a node is local to the given rank.
        
        Args:
            node_id: Node ID
            rank: Rank to check
            
        Returns:
            True if node is local to this rank
        """
        return self.owner(node_id) == rank
    
    def get_local_nodes(self, rank: int):
        """
        Get all node IDs local to the given rank.
        
        Args:
            rank: Rank
            
        Returns:
            List of local node IDs
        """
        start, end = self.ranges[rank]
        return list(range(start, end))
    
    def get_range(self, rank: int):
        """
        Get the range of nodes for a given rank.
        
        Args:
            rank: Rank
            
        Returns:
            Tuple (start, end) representing the range
        """
        return self.ranges[rank]


class HashPartitioner:
    """
    Hash-based partitioner for distributing nodes across machines.
    Uses modulo operation to distribute nodes: node_id % world_size
    This helps when node_id correlates with time, as it breaks up sequential patterns.
    """
    
    def __init__(self, num_nodes: int, world_size: int):
        """
        Initialize hash partitioner.
        
        Args:
            num_nodes: Total number of nodes in the graph
            world_size: Number of machines/processes
        """
        self.num_nodes = num_nodes
        self.world_size = world_size
    
    def owner(self, node_id: int) -> int:
        """
        Get the rank that owns a given node using hash (modulo).
        
        Args:
            node_id: Node ID
            
        Returns:
            Rank that owns this node
        """
        return int(node_id) % self.world_size
    
    def is_local(self, node_id: int, rank: int) -> bool:
        """
        Check if a node is local to the given rank.
        
        Args:
            node_id: Node ID
            rank: Rank to check
            
        Returns:
            True if node is local to this rank
        """
        return self.owner(node_id) == rank
    
    def get_local_nodes(self, rank: int):
        """
        Get all node IDs local to the given rank.
        
        Args:
            rank: Rank
            
        Returns:
            List of local node IDs
        """
        return [i for i in range(self.num_nodes) if self.owner(i) == rank]
    
    def get_range(self, rank: int):
        """
        Hash partitioner doesn't have a simple range representation.
        Returns the list of local nodes instead (for compatibility).
        
        Args:
            rank: Rank
            
        Returns:
            Tuple (start, end) where start=0 and end=len(local_nodes) for compatibility
        """
        local_nodes = self.get_local_nodes(rank)
        return (0, len(local_nodes))


import numpy as np


class STEPPartitioner:
    """
    STEP (Streaming Temporal-aware Edge Partitioning) for DistGL.
    Assigns each node to a partition based on temporal locality, spatial affinity
    (co-occurrence with already-assigned neighbours), and load balance.
    """

    def __init__(self, num_nodes: int, world_size: int, full_data, alpha: float = 0.5, gamma: float = 1.5):
        self.num_nodes = num_nodes
        self.world_size = world_size

        self.node_owner = -1 * np.ones(num_nodes, dtype=np.int32)

        partition_avg_time   = np.zeros(world_size, dtype=np.float64)
        partition_node_count = np.zeros(world_size, dtype=np.int32)
        partition_edge_count = np.zeros(world_size, dtype=np.int32)
        partition_nodes      = [set() for _ in range(world_size)]

        sources      = full_data.sources
        destinations = full_data.destinations
        timestamps   = full_data.timestamps

        for i in range(len(sources)):
            src, dst, t = int(sources[i]), int(destinations[i]), float(timestamps[i])

            if self.node_owner[src] == -1:
                best_p = self._select_partition(src, dst, t, partition_avg_time, partition_node_count, partition_nodes, alpha, gamma)
                self.node_owner[src] = best_p
                partition_node_count[best_p] += 1
                partition_nodes[best_p].add(src)

            if self.node_owner[dst] == -1:
                best_p = self._select_partition(dst, src, t, partition_avg_time, partition_node_count, partition_nodes, alpha, gamma)
                self.node_owner[dst] = best_p
                partition_node_count[best_p] += 1
                partition_nodes[best_p].add(dst)

            # Update the owning partition's average timestamp
            p_src = self.node_owner[src]
            curr_avg   = partition_avg_time[p_src]
            curr_count = partition_edge_count[p_src]
            partition_avg_time[p_src]   = (curr_avg * curr_count + t) / (curr_count + 1)
            partition_edge_count[p_src] += 1

        # Assign isolated nodes (never appeared in any edge) round-robin
        for i in range(num_nodes):
            if self.node_owner[i] == -1:
                self.node_owner[i] = i % world_size

    def _select_partition(self, u, v, t, avg_times, node_counts, partition_nodes, alpha, gamma):
        scores = np.zeros(self.world_size)
        for p in range(self.world_size):
            # 1. 空间亲和性 (Spatial Affinity): 0 或 1
            spatial_score = 1.0 if v in partition_nodes[p] else 0.0

            # 2. 时间局部性 (Temporal Locality)
            if node_counts[p] == 0:
                # 如果机器 P 是空的，给一个中等的时间得分，鼓励探索
                temporal_score = 0.5
            else:
                # 计算当前时间 t 与机器 P 平均时间的绝对差值
                time_diff = abs(t - avg_times[p])
                # 使用指数衰减：差值越小 (接近0)，得分越接近 1；差值越大，得分越接近 0
                # 这里的 1e6 是一个缩放因子(温度系数)，根据数据集的时间戳跨度(秒/毫秒)可能需要微调
                temporal_score = np.exp(-time_diff / 1e6)

            # 3. 负载惩罚 (Load Penalty)
            # 使用论文中的多项式惩罚：gamma * (count ^ (gamma - 1))
            load_penalty = gamma * (node_counts[p] ** (gamma - 1)) if node_counts[p] > 0 else 0.0

            # 4. 综合打分 (Additive 融合)
            # alpha 控制空间和时间的权重，通常 alpha=0.5 表示同等重要
            scores[p] = alpha * spatial_score + (1.0 - alpha) * temporal_score - load_penalty

        return int(np.argmax(scores))

    def owner(self, node_id: int) -> int:
        return int(self.node_owner[node_id])

    def is_local(self, node_id: int, rank: int) -> bool:
        return self.owner(node_id) == rank

    def get_local_nodes(self, rank: int):
        return np.where(self.node_owner == rank)[0].tolist()

