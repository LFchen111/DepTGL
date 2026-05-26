"""
Distributed Memory Wrapper for TGN
实现memory和sample的数据缓存，每隔几个batch通信一次memory
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from distributed.dist_utils import all_gather_object, is_distributed, get_rank, get_world_size
import torch.distributed as dist
import pickle
from distributed.range_partitioner import RangePartitioner, HashPartitioner
from typing import Union


def _to_1d_long_tensor(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long).view(-1)
    if isinstance(x, np.ndarray):
        # 直接从numpy array转换，避免通过list
        return torch.from_numpy(x).to(device=device, dtype=torch.long).view(-1)
    if isinstance(x, (list, tuple)):
        # 检查列表中是否包含numpy数组
        if len(x) > 0 and isinstance(x[0], np.ndarray):
            # 先转换为numpy array，再转换为tensor
            return torch.from_numpy(np.array(x)).to(device=device, dtype=torch.long).view(-1)
        # 普通列表，直接转换
        return torch.tensor(list(x), device=device, dtype=torch.long).view(-1)
    # 标量值
    return torch.tensor([x], device=device, dtype=torch.long).view(-1)


def _ts_to_float(ts) -> float:
    if isinstance(ts, torch.Tensor):
        return float(ts.detach().cpu().item())
    return float(ts)


def _detach_msg(m, device) -> torch.Tensor:
    if isinstance(m, torch.Tensor):
        return m.detach().to(device)
    return torch.tensor(m, device=device).detach()


class DistributedMemoryWrapper:
    """
    给现有 tgn.memory 做一层 wrapper，支持分布式memory管理：

    核心设计：
    - 本地节点：直接更新memory，sync时广播给其他机器
    - 远程节点：更新本地cache用于重算，不影响远程机器上的真实值
    - 数据分区保证每个节点的owner会收到所有涉及该节点的边
    - sync时动态读取本地节点的最新状态并广播，其他机器更新cache
    - 不需要维护dirty标记，按需同步
    """

    def reset(self):
        """在每个 epoch memory.__init_memory__() 之后调用，避免 cache 残留导致断言不一致"""
        self._remote_mem_cache = None
        self._remote_last_update_cache = None
        self._messages_dict = {}
        # 重置 DistGL 陈旧度追踪，避免跨 epoch 的脏数据
        self.node_last_update_batch = {}

    def __init__(
            self,
            memory_module: Any,
            num_nodes: int,
            device: torch.device,
            sync_every: int = 20,
            partitioner: Union[RangePartitioner, HashPartitioner, None] = None,
            distgl_staleness: int = 0,
            static_cache: Optional[set] = None,
            hot_nodes: Optional[set] = None,
    ):
        self.mem = memory_module
        self.device = device

        self.static_cache: set = static_cache if static_cache is not None else set()

        # ================= [MemShare: 初始化热点缓存结构] =================
        self.hot_nodes = hot_nodes if hot_nodes is not None else set()
        if self.hot_nodes:
            # 放到 GPU 显存内，供后续执行极其快速的 All-Reduce 密集通信
            self.hot_nodes_tensor = torch.tensor(list(self.hot_nodes), dtype=torch.long, device=self.device)
        else:
            self.hot_nodes_tensor = None

        self.sync_every = int(sync_every)

        self.rank = get_rank()
        self.world_size = get_world_size()

        self.num_nodes = num_nodes
        # Use provided partitioner, or create RangePartitioner as default for backward compatibility
        if partitioner is not None:
            self.partitioner = partitioner
        else:
            self.partitioner = RangePartitioner(num_nodes=self.num_nodes, world_size=self.world_size)

        # 不需要维护dirty_local，sync时直接根据needed_nodes同步

        # 远程节点的memory缓存（只读，用于查询）
        self._remote_mem_cache: Optional[torch.Tensor] = None
        self._remote_last_update_cache: Optional[torch.Tensor] = None

        # 每个节点的待处理消息（用于分布式同步），格式: {nid: [(msg_tensor, ts_tensor), ...]}
        self._messages_dict: Dict[int, list] = {}

        # ================= [新增：DistGL 缓存状态] =================
        self.distgl_staleness = distgl_staleness
        # 记录每个节点最后一次被成功拉取的 batch_idx（用于判断缓存是否新鲜）
        self.node_last_update_batch: Dict[int, int] = {}
        # 优化 3：高频远程节点的静态缓存集合，命中则永久跳过通信
        self.static_cache: set = static_cache if static_cache is not None else set()
        # ==========================================================

        # 备份原方法
        self._orig_store_raw_messages = getattr(self.mem, "store_raw_messages", None)
        self._orig_get_memory = getattr(self.mem, "get_memory", None)
        self._orig_get_last_update = getattr(self.mem, "get_last_update", None)
        self._orig_clear_messages = getattr(self.mem, "clear_messages", None)

        self._orig_set_memory = getattr(self.mem, "set_memory", None)
        if self._orig_set_memory is not None:
            self.mem.set_memory = self.set_memory  # type: ignore

        if self._orig_store_raw_messages is None or self._orig_get_memory is None:
            raise RuntimeError("缺少 store_raw_messages 或 get_memory，无法使用分布式缓存。")

        # 猴子补丁：替换 memory 的关键方法
        self.mem.store_raw_messages = self.store_raw_messages  # type: ignore
        self.mem.get_memory = self.get_memory  # type: ignore
        if self._orig_get_last_update is not None:
            self.mem.get_last_update = self.get_last_update  # type: ignore

    def set_memory(self, node_idxs, values):
        """
        设置memory：
        - 所有节点都更新底层memory和cache
        - 本地节点：会在sync时被其他机器拉取
        - 远程节点：更新本地cache用于重算，不影响远程机器上的真实值
        """
        ids_t = _to_1d_long_tensor(node_idxs, self.device)
        vals = values.detach().to(self.device)

        # 更新底层memory（所有节点）
        self._orig_set_memory(ids_t, vals)

        # 更新本地 cache（所有节点），确保与底层 memory 一致
        self._ensure_cache()
        with torch.no_grad():
            self._remote_mem_cache[ids_t] = vals.detach().clone()
            # 同步 last_update 到 cache，确保与底层 memory 一致
            self._remote_last_update_cache[ids_t] = self.mem.last_update[ids_t].detach().clone()

    def _ensure_cache(self):
        """确保 cache 与底层 memory 同步（本地节点部分始终最新）。"""
        if self._remote_mem_cache is not None:
            with torch.no_grad():
                # 【恢复这里的 is_full_cache 分支】
                if getattr(self.partitioner, 'is_full_cache', False):
                    # 极速优化：全缓存模式下直接整体 copy（原地复制，不分配新内存，不会 OOM）
                    self._remote_mem_cache.copy_(self.mem.memory.detach())
                    self._remote_last_update_cache.copy_(self.mem.last_update.detach())
                elif isinstance(self.partitioner, RangePartitioner):
                    start, end = self.partitioner.get_range(self.rank)
                    self._remote_mem_cache[start:end] = self.mem.memory[start:end].detach()
                    self._remote_last_update_cache[start:end] = self.mem.last_update[start:end].detach()
                else:
                    # 缓存 local_ids_t，避免每次调用都在 CPU 上遍历所有节点
                    if not hasattr(self, '_local_ids_t'):
                        local_nodes = list(self.partitioner.get_local_nodes(self.rank))
                        self._local_ids_t = torch.tensor(local_nodes, device=self.device,
                                                         dtype=torch.long) if local_nodes else None

                    if self._local_ids_t is not None:
                        self._remote_mem_cache[self._local_ids_t] = self.mem.memory[self._local_ids_t].detach()
                        self._remote_last_update_cache[self._local_ids_t] = self.mem.last_update[
                            self._local_ids_t].detach()
            return

        # 首次初始化：直接从底层 memory/last_update 克隆
        self._remote_mem_cache = self.mem.memory.detach().clone()
        self._remote_last_update_cache = self.mem.last_update.detach().clone()

    def store_raw_messages(self, nodes, messages, timestamps):
        """
        存储raw messages，支持本地重算。接受新的 Tensor API：
          nodes      [B]    long tensor of node IDs
          messages   [B, D] float tensor of message vectors
          timestamps [B]    float tensor of edge timestamps

        全向量化实现：无 Python 循环，所有去重/过滤操作在 GPU 上完成。
        """
        self._ensure_cache()

        # ── 统一转为 GPU Tensor ──────────────────────────────────────────────
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.from_numpy(np.asarray(nodes))
        nodes = nodes.to(device=self.device, dtype=torch.long).view(-1)

        if not isinstance(messages, torch.Tensor):
            messages = torch.stack(list(messages))
        messages = messages.to(self.device).detach()

        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(list(timestamps), dtype=torch.float)
        timestamps = timestamps.to(device=self.device, dtype=torch.float).view(-1)

        # ── 过滤超出范围的节点 ───────────────────────────────────────────────
        valid = nodes < self.num_nodes
        if not valid.all():
            nodes = nodes[valid]
            messages = messages[valid]
            timestamps = timestamps[valid]

        if nodes.numel() == 0:
            return

        # ── Step 1: 先按时间戳升序排列，再对节点 ID 做稳定排序 ──────────────
        # 结果：相同节点的消息聚在一起，且时间戳最大的在每组末尾
        ts_order = torch.argsort(timestamps, stable=True)
        nodes_s = nodes[ts_order]
        msgs_s = messages[ts_order]
        ts_s = timestamps[ts_order]

        node_order = torch.argsort(nodes_s, stable=True)
        nodes_sorted = nodes_s[node_order]
        msgs_sorted = msgs_s[node_order]
        ts_sorted = ts_s[node_order]

        # ── Step 2: 每个节点只保留最后一条（时间戳最大）─────────────────────
        unique_nodes, counts = torch.unique_consecutive(nodes_sorted, return_counts=True)
        last_idx = torch.cumsum(counts, 0) - 1
        unique_msgs = msgs_sorted[last_idx]
        unique_ts = ts_sorted[last_idx]

        # ── Step 3: 批量过滤 ts < last_update 的过期消息 ────────────────────
        # _ensure_cache 已将本地节点写入 cache，直接读取即可
        last_updates = self._remote_last_update_cache[unique_nodes]
        keep = (unique_ts + 1e-9) >= last_updates
        if not keep.any():
            return

        filtered_nodes = unique_nodes[keep]
        filtered_msgs = unique_msgs[keep]
        filtered_ts = unique_ts[keep]

        # ── 更新 _messages_dict（仅供 sync 序列化使用，循环量极小）──────────
        for i, nid in enumerate(filtered_nodes.tolist()):
            self._messages_dict[nid] = [(filtered_msgs[i], filtered_ts[i])]

        # ── 写入 flat lists 供 TGN update_memory 使用 ───────────────────────
        self._orig_store_raw_messages(filtered_nodes, filtered_msgs, filtered_ts)

    def get_memory(self, node_idxs):
        """获取节点的 memory。
        _ensure_cache 已将本地节点的最新值同步到 cache，因此所有范围内节点
        直接从 _remote_mem_cache 读取，无需区分本地/远程，避免 is_local 循环。
        超出分布式范围的节点退化到原始方法。
        """
        self._ensure_cache()
        ids_t = _to_1d_long_tensor(node_idxs, self.device)

        valid_mask = ids_t < self.num_nodes
        if valid_mask.all():
            return self._remote_mem_cache[ids_t]

        result = torch.zeros((len(ids_t), self._remote_mem_cache.shape[1]),
                             device=self.device, dtype=self._remote_mem_cache.dtype)
        if valid_mask.any():
            result[valid_mask] = self._remote_mem_cache[ids_t[valid_mask]]
        if (~valid_mask).any():
            result[~valid_mask] = self._orig_get_memory(ids_t[~valid_mask])
        return result

    def get_last_update(self, node_idxs):
        """获取节点的 last_update 时间。
        同 get_memory，_ensure_cache 保证 cache 包含最新本地值，直接索引即可。
        """
        if self._orig_get_last_update is None:
            raise AttributeError("Original memory has no get_last_update")

        self._ensure_cache()
        ids_t = _to_1d_long_tensor(node_idxs, self.device)

        valid_mask = ids_t < self.num_nodes
        if valid_mask.all():
            return self._remote_last_update_cache[ids_t]

        result = torch.zeros(len(ids_t), device=self.device,
                             dtype=self._remote_last_update_cache.dtype)
        if valid_mask.any():
            result[valid_mask] = self._remote_last_update_cache[ids_t[valid_mask]]
        if (~valid_mask).any():
            result[~valid_mask] = self._orig_get_last_update(ids_t[~valid_mask])
        return result

    def sync(self):
        """
        同步memory：广播所有本地节点的最新状态，更新远程节点的cache

        设计说明：
        1. 每个机器广播自己拥有的所有本地节点的最新状态
        2. 其他机器接收后用owner的状态覆盖本地cache和底层memory
        3. Messages也从owner同步，用owner的messages完全替换本地的
        """
        if not is_distributed():
            return

        self._ensure_cache()

        self._sync_hot_nodes_allreduce()
        # 准备本地所有节点的memory、last_update和messages
        # 缓存 local_nodes 列表，避免每次 sync 都重新遍历所有节点
        # if not hasattr(self, '_local_nodes_list'):
        #     # if getattr(self.partitioner, 'is_full_cache', False):
        #     #     self._local_nodes_list = range(self.num_nodes)
        #     # else:
        #     self._local_nodes_list = list(self.partitioner.get_local_nodes(self.rank))
        # local_nodes = self._local_nodes_list
        if not hasattr(self, '_local_nodes_list'):
            self._local_nodes_list = list(self.partitioner.get_local_nodes(self.rank))
        local_nodes = self._local_nodes_list

        # === [MemShare] 在常规单边强制覆盖(Owner->Remotes)中剔除全局共享节点 ===
        if self.hot_nodes:
            local_nodes = [n for n in local_nodes if n not in self.hot_nodes]

        if local_nodes:
            # if getattr(self.partitioner, 'is_full_cache', False):
            #     local_ids_t = torch.arange(self.num_nodes, device=self.device, dtype=torch.long)
            # else:
            local_ids_t = torch.tensor(local_nodes, device=self.device, dtype=torch.long)
            local_mem = self._orig_get_memory(local_ids_t).detach().cpu()
            local_last = self._orig_get_last_update(local_ids_t).detach().cpu()

            # 收集这些节点的messages（待处理的更新）
            local_messages = {}
            for nid in local_nodes:
                msgs = self._messages_dict.get(nid)
                if msgs:
                    local_messages[nid] = [
                        (msg.detach().cpu(),
                         float(ts.detach().cpu().item()) if isinstance(ts, torch.Tensor) else float(ts))
                        for msg, ts in msgs
                    ]
        else:
            local_nodes = []
            local_mem = None
            local_last = None
            local_messages = {}

        # 准备payload
        payload = {
            "rank": self.rank,
            "node_ids": local_nodes,
            "memory": local_mem,
            "last_update": local_last,
            "messages": local_messages,
        }

        # All-gather所有机器的数据
        gathered = all_gather_object(payload)

        # 首先清空所有未同步的远程节点的messages（避免使用过期的预拉取数据）
        synced_remote_nodes = set()
        for p in gathered:
            if p["rank"] != self.rank:
                synced_remote_nodes.update(p.get("node_ids", []))

        # 清空所有远程节点的 messages（dict + flat lists）
        # 优化：直接遍历 _messages_dict 现有的 key，避免 O(N) 全局遍历
        remote_to_clear = [
            node_id for node_id in list(self._messages_dict.keys())
            if not self.partitioner.is_local(node_id, self.rank)
        ]
        for node_id in remote_to_clear:
            del self._messages_dict[node_id]
        if remote_to_clear:
            self.mem.clear_messages_for_nodes(
                torch.tensor(remote_to_clear, device=self.device, dtype=torch.long)
            )

        # 更新远程memory缓存并同步到底层memory
        for p in gathered:
            if p["rank"] == self.rank:
                # 本地节点的缓存已经在 _ensure_cache() 中更新，无需重复更新
                continue

            node_ids = p["node_ids"]
            mem = p["memory"]
            last_updates = p["last_update"]
            messages = p.get("messages", {})

            if not node_ids or mem is None:
                continue

            ids_t = torch.tensor(node_ids, device=self.device, dtype=torch.long)
            mem_dev = mem.to(self.device)

            # 同时更新cache和底层memory（确保两者完全一致）
            # 这样 TGN 直接访问 self.memory.last_update 时也能获取到同步后的值
            with torch.no_grad():
                self.mem.memory[ids_t] = mem_dev
                self._remote_mem_cache[ids_t] = mem_dev.clone()

            if last_updates is not None:
                last_dev = last_updates.to(self.device)
                with torch.no_grad():
                    self.mem.last_update[ids_t] = last_dev
                    self._remote_last_update_cache[ids_t] = last_dev.clone()

            # 同步messages（用owner的messages完全替换本地的）
            recv_nodes_f, recv_msgs_f, recv_ts_f = [], [], []
            for nid in node_ids:
                if nid in messages:
                    converted = [
                        (_detach_msg(msg_cpu, self.device),
                         torch.tensor(ts, device=self.device, dtype=torch.float))
                        for msg_cpu, ts in messages[nid]
                    ]
                    self._messages_dict[nid] = converted
                    for msg, ts_t in converted:
                        recv_nodes_f.append(nid)
                        recv_msgs_f.append(msg)
                        recv_ts_f.append(ts_t)
                else:
                    self._messages_dict.pop(nid, None)
            if recv_nodes_f:
                self._orig_store_raw_messages(
                    torch.tensor(recv_nodes_f, device=self.device, dtype=torch.long),
                    torch.stack(recv_msgs_f),
                    torch.stack(recv_ts_f),
                )

    # def sync_p2p(self, needed_nodes: Optional[set] = None, current_batch_idx: int = 0):
    #     """
    #     使用点对点通信同步memory
    #
    #     设计说明：
    #     1. 根据needed_nodes动态确定需要从哪些机器拉取哪些节点
    #     2. 不依赖dirty标记，直接从owner的当前memory读取最新状态
    #     3. 通过点对点通信减少通信量
    #     4. 用owner的最新状态覆盖本地cache，不预先清空（避免丢失信息）
    #
    #     Args:
    #         needed_nodes: 当前机器需要的远程节点集合。如果为None，使用all_gather同步所有节点
    #         current_batch_idx: 当前 batch 索引，用于 DistGL 陈旧度过滤
    #     """
    #     if not is_distributed():
    #         return
    #
    #     # Barrier确保所有机器都到达通信点
    #     # 过滤 NCCL 后端关于 device_id 的警告（该警告是无害的）
    #     import warnings
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings("ignore", message=".*barrier.*device.*", category=UserWarning)
    #         dist.barrier()
    #
    #     self._ensure_cache()
    #
    #     # ================= [修改：全缓存模式下强制全局同步] =================
    #     # if getattr(self.partitioner, 'is_full_cache', False):
    #     #     is_full_cache = True
    #     # else:
    #     #     if not hasattr(self, '_local_nodes_len'):
    #     #         self._local_nodes_len = len(list(self.partitioner.get_local_nodes(self.rank)))
    #     #     is_full_cache = self._local_nodes_len == self.num_nodes
    #     #is_full_cache = getattr(self.partitioner, 'is_full_cache', False)
    #     if needed_nodes is None:
    #         # 如果没有指定需要的节点，或者处于全缓存模式，使用all_gather方式同步所有节点
    #         self.sync()
    #         return
    #
    #     # ================= [DistGL 动态缓存过滤逻辑（含优化 3 静态缓存）] =================
    #     # 对于最近已拉取过（陈旧度在容忍范围内）或命中静态缓存的节点，跳过本次通信
    #     if (self.distgl_staleness > 0 or self.static_cache) and needed_nodes:
    #         actual_needed = set()
    #         for node in needed_nodes:
    #             # 优化 3：命中静态缓存，永久跳过通信
    #             if node in self.static_cache:
    #                 continue
    #             last_update = self.node_last_update_batch.get(node, -999)
    #             # 距离上次成功拉取的 batch 数 <= 陈旧度容忍值，缓存仍然新鲜，跳过
    #             if (current_batch_idx - last_update) <= self.distgl_staleness:
    #                 continue
    #             actual_needed.add(node)
    #         needed_nodes = actual_needed  # 替换为真正需要通信的节点（大幅减少通信量）
    #     # =====================================================================
    #
    #     # 确定需要从哪些rank接收哪些节点（只请求远程节点）
    #     rank_to_request_nodes = {}  # rank -> list of nodes
    #     for node_id in needed_nodes:
    #         owner_rank = self.partitioner.owner(node_id)
    #         if owner_rank != self.rank:  # 只请求远程节点
    #             if owner_rank not in rank_to_request_nodes:
    #                 rank_to_request_nodes[owner_rank] = []
    #             rank_to_request_nodes[owner_rank].append(node_id)
    #
    #     # 使用all_gather交换请求信息
    #     request_info = {
    #         "rank": self.rank,
    #         "requests": rank_to_request_nodes  # {owner_rank: [node_ids]}
    #     }
    #     all_requests = all_gather_object(request_info)
    #
    #     # 确定哪些rank需要我的哪些节点
    #     nodes_to_send_per_rank = {}  # rank -> list of nodes
    #     for req_info in all_requests:
    #         requester_rank = req_info["rank"]
    #         if requester_rank == self.rank:
    #             continue
    #         requests = req_info["requests"]
    #         if self.rank in requests:
    #             nodes_to_send_per_rank[requester_rank] = requests[self.rank]
    #
    #     # 检测后端类型，选择合适的通信方式
    #     backend = dist.get_backend()
    #
    #     if backend == "nccl":
    #         # NCCL 后端：使用 all_gather_object 方式（更可靠）
    #         # NCCL 对序列化数据的点对点通信支持有限，使用 all_gather_object 更稳定
    #         self._sync_p2p_allgather(nodes_to_send_per_rank, rank_to_request_nodes, needed_nodes)
    #     else:
    #         # Gloo 后端：使用点对点通信（更高效）
    #         self._sync_p2p_sendrecv(nodes_to_send_per_rank, rank_to_request_nodes, needed_nodes)
    #
    #     # ================= [新增：通信完成后更新陈旧度时间戳] =================
    #     # 记录本次成功拉取的节点，供下次陈旧度过滤使用
    #     if self.distgl_staleness > 0 and needed_nodes:
    #         for node in needed_nodes:
    #             self.node_last_update_batch[node] = current_batch_idx
    #     # =====================================================================
    def sync_p2p(self, needed_nodes: Optional[set] = None, current_batch_idx: int = 0):
        """
        使用点对点通信同步memory

        设计说明：
        1. 根据needed_nodes动态确定需要从哪些机器拉取哪些节点
        2. 不依赖dirty标记，直接从owner的当前memory读取最新状态
        3. 通过点对点通信减少通信量
        4. 用owner的最新状态覆盖本地cache，不预先清空（避免丢失信息）

        Args:
            needed_nodes: 当前机器需要的远程节点集合。如果为None，使用all_gather同步所有节点
            current_batch_idx: 当前 batch 索引，用于 DistGL 陈旧度过滤
        """
        if not is_distributed():
            return

        # Barrier确保所有机器都到达通信点
        # 过滤 NCCL 后端关于 device_id 的警告（该警告是无害的）
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*barrier.*device.*", category=UserWarning)
            dist.barrier()

        self._ensure_cache()

        if needed_nodes is None:
            # 如果没有指定需要的节点，或者处于全缓存模式，使用all_gather方式同步所有节点
            self.sync()
            return

        # ================= [MemShare & DistGL 动态缓存过滤逻辑] =================
        # 支持了原有的陈旧度过滤、静态缓存过滤，并新增了 MemShare 的共享节点过滤
        has_hot_nodes = hasattr(self, 'hot_nodes') and self.hot_nodes
        if (self.distgl_staleness > 0 or self.static_cache or has_hot_nodes) and needed_nodes:
            actual_needed = set()
            for node in needed_nodes:
                # === [MemShare核心]: 热点节点属于全量共享内存，阻断 P2P 的请求流量 ===
                if has_hot_nodes and node in self.hot_nodes:
                    continue
                # ==============================================================

                # 优化 3：命中静态缓存，永久跳过通信
                if node in self.static_cache:
                    continue
                last_update = self.node_last_update_batch.get(node, -999)
                # 距离上次成功拉取的 batch 数 <= 陈旧度容忍值，缓存仍然新鲜，跳过
                if (current_batch_idx - last_update) <= self.distgl_staleness:
                    continue
                actual_needed.add(node)
            needed_nodes = actual_needed  # 替换为真正需要通信的节点（大幅减少通信量）
        # =====================================================================

        # 确定需要从哪些rank接收哪些节点（只请求远程节点）
        rank_to_request_nodes = {}  # rank -> list of nodes
        for node_id in needed_nodes:
            owner_rank = self.partitioner.owner(node_id)
            if owner_rank != self.rank:  # 只请求远程节点
                if owner_rank not in rank_to_request_nodes:
                    rank_to_request_nodes[owner_rank] = []
                rank_to_request_nodes[owner_rank].append(node_id)

        # 使用all_gather交换请求信息
        request_info = {
            "rank": self.rank,
            "requests": rank_to_request_nodes  # {owner_rank: [node_ids]}
        }
        all_requests = all_gather_object(request_info)

        # 确定哪些rank需要我的哪些节点
        nodes_to_send_per_rank = {}  # rank -> list of nodes
        for req_info in all_requests:
            requester_rank = req_info["rank"]
            if requester_rank == self.rank:
                continue
            requests = req_info["requests"]
            if self.rank in requests:
                nodes_to_send_per_rank[requester_rank] = requests[self.rank]

        # 检测后端类型，选择合适的通信方式
        backend = dist.get_backend()

        if backend == "nccl":
            # NCCL 后端：使用 all_gather_object 方式（更可靠）
            self._sync_p2p_allgather(nodes_to_send_per_rank, rank_to_request_nodes, needed_nodes)
        else:
            # Gloo 后端：使用点对点通信（更高效）
            self._sync_p2p_sendrecv(nodes_to_send_per_rank, rank_to_request_nodes, needed_nodes)

        # ================= [新增：通信完成后更新陈旧度时间戳] =================
        # 记录本次成功拉取的节点，供下次陈旧度过滤使用
        if self.distgl_staleness > 0 and needed_nodes:
            for node in needed_nodes:
                self.node_last_update_batch[node] = current_batch_idx
        # =====================================================================

        # === [MemShare] 通信末尾：针对热点节点执行一次全局同步融合 ===
        if hasattr(self, '_sync_hot_nodes_allreduce'):
            self._sync_hot_nodes_allreduce()
        # ===========================================================
    def _sync_p2p_allgather(self, nodes_to_send_per_rank, rank_to_request_nodes, needed_nodes):
        """NCCL 后端：使用真正的 GPU 张量通信（高性能版本）"""
        import time

        start_time = time.time()
        total_send_bytes = 0

        # 准备要发送的数据（保持在 GPU 上）
        send_node_ids_list = []
        send_memory_list = []
        send_last_update_list = []

        for target_rank, node_list in nodes_to_send_per_rank.items():
            if not node_list:
                continue

            node_ids_t = torch.tensor(node_list, device=self.device, dtype=torch.long)
            node_mem = self._orig_get_memory(node_ids_t).detach()  # 保持在 GPU 上
            node_last = self._orig_get_last_update(node_ids_t).detach()  # 保持在 GPU 上

            send_node_ids_list.append(node_ids_t)
            send_memory_list.append(node_mem)
            send_last_update_list.append(node_last)

            # 统计发送的数据量
            total_send_bytes += node_mem.numel() * node_mem.element_size()
            total_send_bytes += node_last.numel() * node_last.element_size()

        # 合并所有要发送的数据
        if send_node_ids_list:
            all_send_node_ids = torch.cat(send_node_ids_list, dim=0)
            all_send_memory = torch.cat(send_memory_list, dim=0)
            all_send_last_update = torch.cat(send_last_update_list, dim=0)
        else:
            # 如果没有数据要发送，创建空张量
            all_send_node_ids = torch.zeros(0, device=self.device, dtype=torch.long)
            all_send_memory = torch.zeros(0, self.mem.memory.size(1), device=self.device, dtype=self.mem.memory.dtype)
            all_send_last_update = torch.zeros(0, device=self.device, dtype=self.mem.last_update.dtype)

        # 步骤 1: 使用 all_gather 交换每个 rank 的数据大小
        local_size = torch.tensor([all_send_node_ids.size(0)], device=self.device, dtype=torch.long)
        size_list = [torch.zeros(1, device=self.device, dtype=torch.long) for _ in range(self.world_size)]
        dist.all_gather(size_list, local_size)

        # 步骤 2: 使用 all_gather 交换实际数据（GPU 张量直接通信）
        max_size = max(s.item() for s in size_list)

        if max_size > 0:
            # 填充到最大长度
            if all_send_node_ids.size(0) < max_size:
                pad_size = max_size - all_send_node_ids.size(0)
                all_send_node_ids = torch.cat([
                    all_send_node_ids,
                    torch.zeros(pad_size, device=self.device, dtype=torch.long)
                ], dim=0)
                all_send_memory = torch.cat([
                    all_send_memory,
                    torch.zeros(pad_size, self.mem.memory.size(1), device=self.device, dtype=self.mem.memory.dtype)
                ], dim=0)
                all_send_last_update = torch.cat([
                    all_send_last_update,
                    torch.zeros(pad_size, device=self.device, dtype=self.mem.last_update.dtype)
                ], dim=0)

            # all_gather node_ids
            node_ids_gathered = [torch.zeros(max_size, device=self.device, dtype=torch.long)
                                 for _ in range(self.world_size)]
            dist.all_gather(node_ids_gathered, all_send_node_ids)

            # all_gather memory
            memory_gathered = [
                torch.zeros(max_size, self.mem.memory.size(1), device=self.device, dtype=self.mem.memory.dtype)
                for _ in range(self.world_size)]
            dist.all_gather(memory_gathered, all_send_memory)

            # all_gather last_update
            last_update_gathered = [torch.zeros(max_size, device=self.device, dtype=self.mem.last_update.dtype)
                                    for _ in range(self.world_size)]
            dist.all_gather(last_update_gathered, all_send_last_update)
        else:
            node_ids_gathered = []
            memory_gathered = []
            last_update_gathered = []

        # 清空即将被覆盖的远程节点 messages（dict + flat lists）
        remote_needed = [
            nid for nid in needed_nodes
            if not self.partitioner.is_local(nid, self.rank)
        ]
        for nid in remote_needed:
            self._messages_dict.pop(nid, None)
        if remote_needed:
            self.mem.clear_messages_for_nodes(
                torch.tensor(remote_needed, device=self.device, dtype=torch.long)
            )

        # 步骤 3: 处理接收到的数据
        for source_rank in range(self.world_size):
            if source_rank == self.rank:
                continue

            # 获取该 rank 需要发送给我的节点
            if source_rank not in rank_to_request_nodes:
                continue

            requested_nodes = set(rank_to_request_nodes[source_rank])
            actual_size = size_list[source_rank].item()

            if actual_size == 0:
                continue

            # 获取该 rank 发送的数据
            recv_node_ids = node_ids_gathered[source_rank][:actual_size]
            recv_memory = memory_gathered[source_rank][:actual_size]
            recv_last_update = last_update_gathered[source_rank][:actual_size]

            # 只更新我们请求的节点
            mask = torch.tensor([nid.item() in requested_nodes for nid in recv_node_ids],
                                device=self.device, dtype=torch.bool)

            if mask.any():
                filtered_node_ids = recv_node_ids[mask]
                filtered_memory = recv_memory[mask]
                filtered_last_update = recv_last_update[mask]

                # 更新 memory 和 cache（直接在 GPU 上操作）
                with torch.no_grad():
                    self.mem.memory[filtered_node_ids] = filtered_memory
                    self.mem.last_update[filtered_node_ids] = filtered_last_update
                    self._remote_mem_cache[filtered_node_ids] = filtered_memory.clone()
                    self._remote_last_update_cache[filtered_node_ids] = filtered_last_update.clone()

                # messages 不通过 NCCL 传输；dict 中的过时条目已在上方清空
                for nid in filtered_node_ids.cpu().tolist():
                    self._messages_dict.pop(nid, None)

        # 性能统计
        elapsed = time.time() - start_time
        # if total_send_bytes > 0:
        #     bandwidth_mbps = (total_send_bytes * 8 / 1024 / 1024) / elapsed if elapsed > 0 else 0
        #     print(f"[PERF] Rank {self.rank}: sync_p2p (NCCL/GPU) took {elapsed:.3f}s, "
        #           f"sent {total_send_bytes / 1024 / 1024:.2f}MB, bandwidth: {bandwidth_mbps:.2f} Mbps")

    def _sync_p2p_sendrecv(self, nodes_to_send_per_rank, rank_to_request_nodes, needed_nodes):
        """Gloo 后端：使用点对点通信"""
        import time
        start_time = time.time()

        comm_device = torch.device("cpu")

        # 1. 准备发送数据
        send_requests = []
        send_buffers = {}
        total_send_bytes = 0

        for target_rank, node_list in nodes_to_send_per_rank.items():
            if not node_list:
                continue

            # 从当前memory读取这些节点的最新状态
            # 使用 non_blocking=True 进行异步 GPU->CPU 传输（避免阻塞训练）
            node_ids_t = torch.tensor(node_list, device=self.device, dtype=torch.long)
            node_mem = self._orig_get_memory(node_ids_t).detach().to('cpu', non_blocking=True)
            node_last = self._orig_get_last_update(node_ids_t).detach().to('cpu', non_blocking=True)

            # 收集messages（使用内部 dict，避免访问已废弃的 self.mem.messages）
            node_messages = {}
            for nid in node_list:
                msg_list = self._messages_dict.get(nid)
                if msg_list:
                    node_messages[nid] = [
                        (msg.detach().to('cpu', non_blocking=True),
                         float(ts.detach().cpu().item()) if isinstance(ts, torch.Tensor) else float(ts))
                        for msg, ts in msg_list
                    ]

            # 确保异步 GPU->CPU 传输完成后再序列化
            # non_blocking=True 时需要显式同步
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

            # 序列化
            data_to_send = {
                "node_ids": node_list,
                "memory": node_mem,
                "last_update": node_last,
                "messages": node_messages,
            }
            data_bytes = pickle.dumps(data_to_send)
            # 创建可写的 tensor（避免 buffer 只读警告）
            data_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(data_bytes))

            # 统计发送的数据量
            total_send_bytes += len(data_bytes)

            # 发送大小
            size_tensor = torch.tensor([len(data_bytes)], dtype=torch.long, device=comm_device)
            size_req = dist.isend(size_tensor, dst=target_rank)
            send_requests.append(size_req)

            # 发送数据
            data_req = dist.isend(data_tensor, dst=target_rank)
            send_requests.append(data_req)

            send_buffers[target_rank] = (size_tensor, data_tensor)

        # 2. 接收数据
        recv_requests = []
        recv_buffers = {}

        for source_rank, requested_nodes in rank_to_request_nodes.items():
            if not requested_nodes:
                continue

            # 接收大小
            size_tensor = torch.zeros(1, dtype=torch.long, device=comm_device)
            size_req = dist.irecv(size_tensor, src=source_rank)
            recv_requests.append((size_req, source_rank, size_tensor))

        # 3. 等待大小接收完成，开始接收数据
        for size_req, source_rank, size_tensor in recv_requests:
            size_req.wait()
            data_size = size_tensor.item()

            data_tensor = torch.zeros(data_size, dtype=torch.uint8, device=comm_device)
            data_req = dist.irecv(data_tensor, src=source_rank)
            recv_buffers[source_rank] = (data_req, data_tensor)

        # 3.5. 清空即将同步的远程节点的 messages（dict + flat lists）
        # 优化：直接遍历 _messages_dict 现有的 key，避免 O(N) 全局遍历
        remote_to_clear = [
            node_id for node_id in list(self._messages_dict.keys())
            if not self.partitioner.is_local(node_id, self.rank)
        ]
        for node_id in remote_to_clear:
            del self._messages_dict[node_id]
        if remote_to_clear:
            self.mem.clear_messages_for_nodes(
                torch.tensor(remote_to_clear, device=self.device, dtype=torch.long)
            )

        # 4. 等待数据接收完成并更新cache
        for source_rank, (data_req, data_tensor) in recv_buffers.items():
            data_req.wait()

            # 优化：tensor 已在 CPU 上，无需再调用 .cpu()
            data_bytes = bytes(data_tensor.numpy())
            received_data = pickle.loads(data_bytes)

            node_ids = received_data["node_ids"]
            node_mem = received_data["memory"].to(self.device)
            node_last = received_data["last_update"].to(self.device)
            node_messages = received_data.get("messages", {})

            # 同步到底层memory和cache（确保两者完全一致）
            # 这样 TGN 直接访问 self.memory.last_update 时也能获取到同步后的值
            node_ids_t = torch.tensor(node_ids, device=self.device, dtype=torch.long)
            with torch.no_grad():
                self.mem.memory[node_ids_t] = node_mem
                self.mem.last_update[node_ids_t] = node_last
                # 更新缓存，确保与底层 memory 一致
                self._remote_mem_cache[node_ids_t] = node_mem.clone()
                self._remote_last_update_cache[node_ids_t] = node_last.clone()

            # 同步messages：写入内部 dict，同时追加到 flat lists 供 TGN 使用
            recv_nodes_f, recv_msgs_f, recv_ts_f = [], [], []
            for nid in node_ids:
                if nid in node_messages:
                    converted = [
                        (_detach_msg(msg_cpu, self.device),
                         torch.tensor(ts, device=self.device, dtype=torch.float))
                        for msg_cpu, ts in node_messages[nid]
                    ]
                    self._messages_dict[nid] = converted
                    for msg, ts_t in converted:
                        recv_nodes_f.append(nid)
                        recv_msgs_f.append(msg)
                        recv_ts_f.append(ts_t)
                else:
                    self._messages_dict.pop(nid, None)
            if recv_nodes_f:
                self._orig_store_raw_messages(
                    torch.tensor(recv_nodes_f, device=self.device, dtype=torch.long),
                    torch.stack(recv_msgs_f),
                    torch.stack(recv_ts_f),
                )

        # 5. 等待所有发送完成
        for req in send_requests:
            req.wait()

        # 诊断信息
        elapsed = time.time() - start_time
        # if total_send_bytes > 0:
        #     bandwidth_mbps = (total_send_bytes * 8 / 1024 / 1024) / elapsed if elapsed > 0 else 0
        #     print(
        #         f"[PERF] Rank {self.rank}: sync_p2p took {elapsed:.3f}s, sent {total_send_bytes / 1024 / 1024:.2f}MB to {len(nodes_to_send_per_rank)} ranks, bandwidth: {bandwidth_mbps:.2f} Mbps")

    def maybe_sync(self, batch_idx: int):
        """每隔sync_every个batch同步一次"""
        if self.sync_every > 0 and (batch_idx + 1) % self.sync_every == 0:
            self.sync()

    def _sync_hot_nodes_allreduce(self):
        """
        [MemShare] Synchronous Smoothing Aggregation
        利用全局 All-Reduce (Sum) 获取各台机器最新梯度的平均偏差
        """
        if not hasattr(self, 'hot_nodes_tensor') or self.hot_nodes_tensor is None or self.world_size <= 1:
            return

        hot_mem = self.mem.memory[self.hot_nodes_tensor].clone()
        hot_last_update = self.mem.last_update[self.hot_nodes_tensor].clone()

        # Memory 取所有进程更新向量的平均值
        dist.all_reduce(hot_mem, op=dist.ReduceOp.SUM)
        hot_mem.div_(self.world_size)

        # TimeStamp 取拥有最新特征的时间戳
        dist.all_reduce(hot_last_update, op=dist.ReduceOp.MAX)

        with torch.no_grad():
            self.mem.memory[self.hot_nodes_tensor] = hot_mem
            self.mem.last_update[self.hot_nodes_tensor] = hot_last_update
            self._remote_mem_cache[self.hot_nodes_tensor] = hot_mem.clone()
            self._remote_last_update_cache[self.hot_nodes_tensor] = hot_last_update.clone()