import torch


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a tuple of
  (nodes_list, data_list, ts_list) Tensor lists, aggregates messages with the
  same node id using one of the possible strategies.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    :param node_ids: unused (kept for API compatibility); pass None.
    :param messages: tuple (nodes_list, data_list, ts_list) where each element
                     is a list of Tensors accumulated since the last clear.
    :return: (unique_nodes [K], aggregated_messages [K, D], timestamps [K])
    """


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Keep only the last (most recent) message for each node."""
    msg_nodes_list, msg_data_list, msg_ts_list = messages

    if not msg_nodes_list:
      return (torch.empty(0, device=self.device, dtype=torch.long),
              torch.empty(0, device=self.device),
              torch.empty(0, device=self.device))

    all_nodes = torch.cat(msg_nodes_list)   # [N]
    all_msgs = torch.cat(msg_data_list)     # [N, D]
    all_ts = torch.cat(msg_ts_list)         # [N]

    # Stable sort by node id so that relative (temporal) insertion order is preserved
    sort_idx = torch.argsort(all_nodes, stable=True)
    sorted_nodes = all_nodes[sort_idx]
    sorted_msgs = all_msgs[sort_idx]
    sorted_ts = all_ts[sort_idx]

    unique_nodes, counts = torch.unique_consecutive(sorted_nodes, return_counts=True)
    # Last entry per group = cumulative end index
    end_indices = torch.cumsum(counts, 0) - 1

    return unique_nodes, sorted_msgs[end_indices], sorted_ts[end_indices]


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Average all messages for each node; timestamp taken from the last message."""
    msg_nodes_list, msg_data_list, msg_ts_list = messages

    if not msg_nodes_list:
      return (torch.empty(0, device=self.device, dtype=torch.long),
              torch.empty(0, device=self.device),
              torch.empty(0, device=self.device))

    all_nodes = torch.cat(msg_nodes_list)   # [N]
    all_msgs = torch.cat(msg_data_list)     # [N, D]
    all_ts = torch.cat(msg_ts_list)         # [N]

    sort_idx = torch.argsort(all_nodes, stable=True)
    sorted_nodes = all_nodes[sort_idx]
    sorted_msgs = all_msgs[sort_idx]
    sorted_ts = all_ts[sort_idx]

    unique_nodes, counts = torch.unique_consecutive(sorted_nodes, return_counts=True)
    end_indices = torch.cumsum(counts, 0) - 1

    # Compute per-group mean via index_add on a zero buffer
    n_unique = unique_nodes.shape[0]
    msg_dim = sorted_msgs.shape[1]
    segment_ids = torch.repeat_interleave(
        torch.arange(n_unique, device=self.device), counts
    )
    msg_sums = torch.zeros(n_unique, msg_dim, device=self.device, dtype=sorted_msgs.dtype)
    msg_sums.index_add_(0, segment_ids, sorted_msgs)
    to_update_msgs = msg_sums / counts.unsqueeze(1).float()

    return unique_nodes, to_update_msgs, sorted_ts[end_indices]


def get_message_aggregator(aggregator_type, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
