import logging
import numpy as np
import torch

from utils.utils import MergeLayer, prof_section
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)
    # Optional fine-grained forward breakdown profiler (set by training script)
    self.op_profiler = None

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    prof = getattr(self, "op_profiler", None)
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        with prof_section(prof, "tgn/memory_get_updated_memory"):
          memory, last_update = self.get_updated_memory(
            (self.memory.messages_nodes,
             self.memory.messages_data,
             self.memory.messages_ts))
      else:
        with prof_section(prof, "tgn/memory_read_full"):
          memory = self.memory.get_memory(list(range(self.n_nodes)))
          last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      with prof_section(prof, "tgn/memory_time_diffs"):
        edge_times_t = torch.tensor(edge_times, dtype=torch.long, device=self.device)
        source_nodes_t = torch.tensor(source_nodes, dtype=torch.long, device=self.device)
        destination_nodes_t = torch.tensor(destination_nodes, dtype=torch.long, device=self.device)
        negative_nodes_t = torch.tensor(negative_nodes, dtype=torch.long, device=self.device)

        source_time_diffs = edge_times_t - last_update[source_nodes_t].long()
        source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src

        destination_time_diffs = edge_times_t - last_update[destination_nodes_t].long()
        destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

        negative_time_diffs = edge_times_t - last_update[negative_nodes_t].long()
        negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

        time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                               dim=0)

    # Compute the embeddings using the embedding module
    with prof_section(prof, "tgn/embedding_module_compute"):
      node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                               source_nodes=nodes,
                                                               timestamps=timestamps,
                                                               n_layers=self.n_layers,
                                                               n_neighbors=n_neighbors,
                                                               time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        with prof_section(prof, "tgn/memory_update_positives"):
          self.update_memory(
            (self.memory.messages_nodes,
             self.memory.messages_data,
             self.memory.messages_ts))

        # assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
        #   "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      with prof_section(prof, "tgn/raw_messages_sources"):
        src_nodes_t, src_messages, src_ts = self.get_raw_messages(
          source_nodes, source_node_embedding,
          destination_nodes, destination_node_embedding,
          edge_times, edge_idxs)
      with prof_section(prof, "tgn/raw_messages_destinations"):
        dst_nodes_t, dst_messages, dst_ts = self.get_raw_messages(
          destination_nodes, destination_node_embedding,
          source_nodes, source_node_embedding,
          edge_times, edge_idxs)

      # if self.memory_update_at_start:
      #   with prof_section(prof, "tgn/memory_store_raw_messages"):
      #     self.memory.store_raw_messages(src_nodes_t, src_messages, src_ts)
      #     self.memory.store_raw_messages(dst_nodes_t, dst_messages, dst_ts)
      # else:
      #   with prof_section(prof, "tgn/memory_update_from_raw_messages"):
      #     self.update_memory(([src_nodes_t], [src_messages], [src_ts]))
      #     self.update_memory(([dst_nodes_t], [dst_messages], [dst_ts]))
      if self.memory_update_at_start:
          with prof_section(prof, "tgn/memory_store_raw_messages"):
              self.memory.store_raw_messages(src_nodes_t, src_messages, src_ts)
              self.memory.store_raw_messages(dst_nodes_t, dst_messages, dst_ts)
      else:
          with prof_section(prof, "tgn/memory_update_from_raw_messages"):
              # 【修复 Bug】：将 src 和 dst 打包成一个列表传入
              # 底层的 message_aggregator 会自动将它们拼接，并按时间戳严格排序，彻底消除时间倒流报错！
              self.update_memory(([src_nodes_t, dst_nodes_t],
                                  [src_messages, dst_messages],
                                  [src_ts, dst_ts]))
      if self.dyrep:
        source_nodes_t = torch.tensor(source_nodes, dtype=torch.long, device=self.device)
        destination_nodes_t = torch.tensor(destination_nodes, dtype=torch.long, device=self.device)
        negative_nodes_t = torch.tensor(negative_nodes, dtype=torch.long, device=self.device)
        source_node_embedding = memory[source_nodes_t]
        destination_node_embedding = memory[destination_nodes_t]
        negative_node_embedding = memory[negative_nodes_t]

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    prof = getattr(self, "op_profiler", None)
    n_samples = len(source_nodes)
    with prof_section(prof, "tgn/compute_temporal_embeddings_total"):
      source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
        source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    with prof_section(prof, "tgn/affinity_score"):
      score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                  torch.cat([destination_node_embedding,
                                             negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, messages):
    """
    :param messages: tuple (nodes_list, data_list, ts_list) – the three Tensor
                     lists held by Memory, or a freshly-constructed single-batch
                     equivalent.
    """
    prof = getattr(self, "op_profiler", None)
    with prof_section(prof, "tgn/message_aggregate"):
      unique_nodes, unique_messages, unique_timestamps = \
        self.message_aggregator.aggregate(None, messages)

    if len(unique_nodes) > 0:
      with prof_section(prof, "tgn/message_function"):
        unique_messages = self.message_function.compute_message(unique_messages)

    with prof_section(prof, "tgn/memory_updater_update"):
      self.memory_updater.update_memory(unique_nodes, unique_messages,
                                        timestamps=unique_timestamps)

  def get_updated_memory(self, messages):
    """
    :param messages: tuple (nodes_list, data_list, ts_list) – same format as
                     update_memory.
    """
    prof = getattr(self, "op_profiler", None)
    with prof_section(prof, "tgn/message_aggregate"):
      unique_nodes, unique_messages, unique_timestamps = \
        self.message_aggregator.aggregate(None, messages)

    if len(unique_nodes) > 0:
      with prof_section(prof, "tgn/message_function"):
        unique_messages = self.message_function.compute_message(unique_messages)

    with prof_section(prof, "tgn/memory_updater_get_updated_memory"):
      updated_memory, updated_last_update = self.memory_updater.get_updated_memory(
        unique_nodes, unique_messages, timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    """
    Returns:
      source_nodes_t   [B]    – source node ids as a long Tensor
      source_message   [B, D] – concatenated raw message vectors
      edge_times_t     [B]    – edge timestamps as a float Tensor
    """
    edge_times_t = torch.from_numpy(edge_times).float().to(self.device)
    edge_idxs_t = torch.from_numpy(np.asarray(edge_idxs)).long().to(self.device)
    source_nodes_t = torch.tensor(source_nodes, dtype=torch.long, device=self.device)
    destination_nodes_t = torch.tensor(destination_nodes, dtype=torch.long, device=self.device)

    edge_features = self.edge_raw_features[edge_idxs_t]

    source_memory = self.memory.get_memory(source_nodes_t) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes_t) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times_t - self.memory.last_update[source_nodes_t]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(
      len(source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding], dim=1)

    return source_nodes_t, source_message, edge_times_t

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
