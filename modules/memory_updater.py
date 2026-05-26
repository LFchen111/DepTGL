from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


# class SequenceMemoryUpdater(MemoryUpdater):
#   def __init__(self, memory, message_dimension, memory_dimension, device):
#     super(SequenceMemoryUpdater, self).__init__()
#     self.memory = memory
#     self.layer_norm = torch.nn.LayerNorm(memory_dimension)
#     self.message_dimension = message_dimension
#     self.device = device
#
#   def update_memory(self, unique_node_ids, unique_messages, timestamps):
#     if len(unique_node_ids) <= 0:
#       return
#
#     assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
#                                                                                      "update memory to time in the past"
#
#     memory = self.memory.get_memory(unique_node_ids)
#     self.memory.last_update[unique_node_ids] = timestamps
#
#     updated_memory = self.memory_updater(unique_messages, memory)
#
#     self.memory.set_memory(unique_node_ids, updated_memory)
#
#   def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
#     if len(unique_node_ids) <= 0:
#       return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
#
#     assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
#                                                                                      "update memory to time in the past"
#
#     updated_memory = self.memory.memory.data.clone()
#     updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])
#
#     updated_last_update = self.memory.last_update.data.clone()
#     updated_last_update[unique_node_ids] = timestamps
#
#     return updated_memory, updated_last_update
class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    # [MemShare 容错] 移除硬性的 assert 报错
    # 找出现有内存里的更新时间
    current_last_update = self.memory.get_last_update(unique_node_ids)

    # 找出那些是“向未来推进”的合法更新（或者时间等于现在的更新）
    valid_mask = timestamps >= current_last_update

    # 如果全是不合法的倒流时间，直接跳过本次更新
    if not valid_mask.any():
      return

    # 如果只有部分节点合法，过滤出合法的节点、消息和时间戳
    if not valid_mask.all():
      unique_node_ids = unique_node_ids[valid_mask]
      unique_messages = unique_messages[valid_mask]
      timestamps = timestamps[valid_mask]

    # 常规的更新逻辑
    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    # 初始化返回的、完全克隆的一份新副本
    updated_memory = self.memory.memory.data.clone()
    updated_last_update = self.memory.last_update.data.clone()

    # [MemShare 容错] 检查时序倒流
    current_last_update = self.memory.get_last_update(unique_node_ids)
    valid_mask = timestamps >= current_last_update

    # 只有存在合法节点时，才去计算 RNN/GRU 的新特征并在副本上覆盖更新
    if valid_mask.any():
      valid_node_ids = unique_node_ids[valid_mask]
      valid_messages = unique_messages[valid_mask]
      valid_timestamps = timestamps[valid_mask]

      # 只对时间戳合法（向前）的节点做前向推演
      updated_memory[valid_node_ids] = self.memory_updater(
        valid_messages,
        updated_memory[valid_node_ids]
      )
      updated_last_update[valid_node_ids] = valid_timestamps

    # 无论有无更新（哪怕全是过期的倒流请求），都应该把这个完整的克隆丢回去
    return updated_memory, updated_last_update

class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
