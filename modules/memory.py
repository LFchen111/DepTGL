import torch
from torch import nn


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    self.messages_nodes = []
    self.messages_data = []
    self.messages_ts = []

  def store_raw_messages(self, nodes, messages, timestamps):
    self.messages_nodes.append(nodes)
    self.messages_data.append(messages)
    self.messages_ts.append(timestamps)

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    return (self.memory.data.clone(),
            self.last_update.data.clone(),
            [m.clone() for m in self.messages_nodes],
            [m.clone() for m in self.messages_data],
            [m.clone() for m in self.messages_ts])

  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()
    self.messages_nodes = [m.clone() for m in memory_backup[2]]
    self.messages_data = [m.clone() for m in memory_backup[3]]
    self.messages_ts = [m.clone() for m in memory_backup[4]]

  def detach_memory(self):
    self.memory.detach_()
    self.messages_nodes = [m.detach() for m in self.messages_nodes]
    self.messages_data = [m.detach() for m in self.messages_data]
    self.messages_ts = [m.detach() for m in self.messages_ts]

  def clear_messages(self, nodes):
    self.messages_nodes = []
    self.messages_data = []
    self.messages_ts = []

  def clear_messages_for_nodes(self, node_ids):
    """Remove pending messages for specific node IDs, leaving other nodes untouched."""
    if not self.messages_nodes:
      return
    if not isinstance(node_ids, torch.Tensor):
      node_ids = torch.tensor(list(node_ids), dtype=torch.long)

    all_nodes = torch.cat(self.messages_nodes)
    all_data = torch.cat(self.messages_data)
    all_ts = torch.cat(self.messages_ts)

    keep_mask = ~torch.isin(all_nodes, node_ids.to(all_nodes.device))
    if keep_mask.all():
      return
    if not keep_mask.any():
      self.messages_nodes = []
      self.messages_data = []
      self.messages_ts = []
      return
    self.messages_nodes = [all_nodes[keep_mask]]
    self.messages_data = [all_data[keep_mask]]
    self.messages_ts = [all_ts[keep_mask]]
