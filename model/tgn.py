import logging
import numpy as np
import torch
from collections import defaultdict
from torch import nn
from utils.utils import MergeLayer, MLP
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.memory_map_updater import get_memory_map_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from modules.memory_map import MemoryMap
from sklearn.metrics.pairwise import euclidean_distances

class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False, use_map=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500,
                 map_size=55, map_layer=2,
                 embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru", assign_type="position",
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
        self.use_memory_map = use_map
        time_dim=self.n_node_features
        self.time_encoder = TimeEncode(dimension=time_dim)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            self.num_dim=1
            self.num_encode_layer = nn.Sequential(
                nn.Linear(in_features=1, out_features=self.num_dim),
                nn.ReLU(),
                nn.Linear(in_features=self.num_dim, out_features=self.num_dim))
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension  + self.num_dim
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.memory_map = MemoryMap(n_nodes=self.n_nodes,
                                        map_init_type="zero",
                                        size=map_size,
                                        neighbor=self.neighbor_finder,
                                        memory_dimension=self.memory_dimension,
                                        device=device)
            self.memory_map_gru = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)
            self.memory_map_gru = self.memory_map_gru.memory_updater
            self.aggregator_type=aggregator_type

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
            memory_gru = self.memory_updater.memory_updater


            self.memory_map_gru.weight_ih.data = memory_gru.weight_ih.data.clone()
            self.memory_map_gru.weight_hh.data = memory_gru.weight_hh.data.clone()
            self.memory_map_gru.bias_ih.data = memory_gru.bias_ih.data.clone()
            self.memory_map_gru.bias_hh.data = memory_gru.bias_hh.data.clone()

            self.memory_map_updater = get_memory_map_updater(map_update_type="net",
                                                             assign_type=assign_type,
                                                             memory_map=self.memory_map,
                                                             map_dimension=memory_dimension,
                                                             message_dimension=message_dimension,
                                                             map_layer=map_layer,
                                                             message_aggregator = self.message_aggregator,
                                                             time_encoder = self.time_encoder,
                                                             time_dimension = time_dim,
                                                             message_function = self.message_function,
                                                             neighbor=self.neighbor_finder,
                                                             dropout=dropout,
                                                             device=device)
            self.map_use_message = False
        self.embedding_module_type = embedding_module_type
        self.emb_linear = torch.nn.Linear(self.memory_dimension * 2, self.memory_dimension)
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_raw_features,
                                                     edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=time_dim,
                                                     embedding_dimension=self.memory_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors)

        # MLP to compute probability on an edge given two node embeddings
        # if
        # self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
        #                                  self.n_node_features,
        #                                  1)

        self.affinity_score = MergeLayer(self.memory_dimension, self.memory_dimension,
                                         self.memory_dimension,
                                         1)
        self.node_predictor = MLP(node_features.shape[1], drop=dropout)
        self.use_connections = True


    def dice_loss(self, input, target):
        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()
        a = torch.sum(input * target, 1)  # |X⋂Y|
        b = torch.sum(input * input, 1) + 0.001  # |X|
        c = torch.sum(target * target, 1) + 0.001  # |Y|
        d = (2 * a) / (b + c)
        return 1 - d


    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, batch_i, epoch, batch_num, flist, n_neighbors=20, inductive=False):
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
        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                if self.use_memory_map:
                    if not inductive:
                        self.update_memory_map(batch_i, batch_num, flist, self.memory_updater.memory_updater)
                    memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                                  self.memory.messages,
                                                                  )
                else:
                    memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                                  self.memory.messages,
                                                                  )

            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            time_diffs = self.compute_time(edge_times, source_nodes, destination_nodes, negative_nodes, last_update)
            # self.memory_map_updater.check_center(batch_i,memory,nodes)

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

        assign_after_batch = False
        if not assign_after_batch and self.use_memory_map:
            self.memory_map_updater.assign_map(source_nodes, destination_nodes, edge_times, memory, batch_i)
        if self.use_memory and self.use_memory_map and not inductive:
            node_embedding_list = self.memory_map_updater.aggregate_trajectory(batch_i, epoch, nodes, memory, timestamps, flist, assign_after_batch)



        if assign_after_batch:
            self.memory_map_updater.assign_map_after_batch(source_nodes, destination_nodes, edge_times, memory, batch_i)

        if self.use_memory and self.use_memory_map and self.use_connections:
            self.memory_map_updater.add_connections(source_nodes, destination_nodes, edge_times)
        # if self.use_memory and self.use_memory_map:
        #     self.memory_map_updater.set_nodes_after_batch
        # self.memory_map_updater.set_time(source_nodes, destination_nodes, edge_times)


        if self.use_memory:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self.update_memory(positives, self.memory.messages)

                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-3), \
                    "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)
                if self.use_memory_map:
                    #if batch_i > 1:
                    self.memory_map_updater.clear_messages(positives)

            self.memory.add_connections(source_nodes,destination_nodes,edge_times)
            num = self.memory.get_connections(source_nodes,destination_nodes)
            use_unique_edge = False
            if use_unique_edge:
                source_nodes = torch.tensor(source_nodes).to(self.device)
                destination_nodes = torch.tensor(destination_nodes).to(self.device)
                edge = torch.stack((source_nodes, destination_nodes), dim=1)
                edge_unique, inverse = torch.unique(edge, dim=0, return_inverse=True)
                index = torch.argmax((inverse == torch.arange(edge_unique.shape[0]).to(self.device).unsqueeze(1)).int(),
                                     dim=1)
                source_nodes = edge_unique[:, 0].cpu().numpy()
                destination_nodes = edge_unique[:, 1].cpu().numpy()
                index = index.cpu().numpy()
                edge_times = edge_times[index]
                edge_idxs = edge_idxs[index]
                num = np.array(num)[index]

            unique_sources, source_id_to_messages, cor_destinstion = self.get_raw_messages(source_nodes,
                                                                          source_node_embedding,
                                                                          destination_nodes,
                                                                          destination_node_embedding,
                                                                          edge_times, edge_idxs, num)
            unique_destinations, destination_id_to_messages, cor_source = self.get_raw_messages(destination_nodes,
                                                                                    destination_node_embedding,
                                                                                    source_nodes,
                                                                                    source_node_embedding,
                                                                                    edge_times, edge_idxs, num)
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages, cor_destinstion)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages, cor_source)
                if self.use_memory_map:
                    # if self.map_use_message:
                    self.memory_map_updater.store_raw_messages(unique_sources, source_id_to_messages, cor_destinstion)
                    self.memory_map_updater.store_raw_messages(unique_destinations, destination_id_to_messages, cor_source)
                    # if not self.map_use_message:
                    self.memory_map_updater.store_last_node([source_nodes, destination_nodes])
                    #if (batch_i > 0):
                        # self.memory_map_updater.aggregate_trajectory([source_nodes, destination_nodes])
                        # map_source, s_center = self.memory_map_updater.get_map_from_embeddings(source_nodes, memory[source_nodes])
                        # map_destination, d_center = self.memory_map_updater.get_map_from_embeddings(destination_nodes, memory[destination_nodes])
                        # map_negative, n_center = self.memory_map_updater.get_map_from_embeddings(negative_nodes, memory[negative_nodes])
                        # source_node_embedding = self.emb_linear(
                        #     torch.cat((memory[source_nodes], map_source), dim=1)
                        # )
                        # destination_node_embedding = self.emb_linear(
                        #     torch.cat((memory[destination_nodes], map_destination), dim=1)
                        # )
                        # negative_node_embedding = self.emb_linear(
                        #     torch.cat((memory[negative_nodes], map_negative), dim=1)
                        # )
                        # assign k batch nodes to memory map
                    # self.memory_map_updater.assign_map(source_nodes, destination_nodes, memory, batch_i)
                        # self.memory_map.analyze_center(batch_i, epoch, flist)
                        # self.memory_map_updater.analyze_center(s_center, source_nodes, d_center, destination_nodes,
                        #                                        n_center, negative_nodes, batch_i, flist, timestamps)
                    '''unique_nodes = np.concatenate([source_nodes, destination_nodes])
                    map_nodes = self.update_map(unique_nodes,memory, batch_i, batch_num, flist)
                    if (batch_i >= self.memory_map.size):
                        map_source = self.memory_map_updater.get_map_from_embeddings(memory[source_nodes])
                        map_destination = self.memory_map_updater.get_map_from_embeddings(memory[destination_nodes])
                        map_negative = self.memory_map_updater.get_map_from_embeddings(memory[negative_nodes])
                        source_node_embedding = self.emb_linear(
                            torch.cat((memory[source_nodes], map_source), dim=1)
                        )
                        destination_node_embedding = self.emb_linear(
                            torch.cat((memory[destination_nodes], map_destination), dim=1)
                        )
                        negative_node_embedding = self.emb_linear(
                            torch.cat((memory[negative_nodes], map_negative), dim=1)
                        )
                    if (batch_i >= self.memory_map.size):
                        # node_embedding = memory[unique_nodes] + map_nodes.detach()
                        node_embedding = self.emb_linear(
                            torch.cat((memory[unique_nodes], map_nodes), dim=1)
                        )
                        map_negative = self.memory_map_updater.get_map_from_embeddings(memory[negative_nodes])
                        # negative_embedding = memory[negative_nodes] + map_negative.detach()
                        negative_embedding = self.emb_linear(
                            torch.cat((memory[negative_nodes], map_negative), dim=1))
                        source_node_embedding = node_embedding[:len(source_nodes), :]
                        destination_node_embedding = node_embedding[:len(source_nodes), :]
                        negative_node_embedding = negative_embedding'''


            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

            #

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                negative_node_embedding = memory[negative_nodes]

        if self.use_memory and self.use_memory_map and (batch_i > 1) and not inductive:
            return memory, node_embedding_list
        return memory, node_embedding

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, batch_i, epoch, batch_num, flist, n_neighbors=20, inductive=False):
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
        n_samples = len(source_nodes)
        memory, node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, batch_i, epoch, batch_num, flist,
            n_neighbors, inductive)

        if self.use_memory and self.use_memory_map and (batch_i > 1):
        # print(source_node_embedding.shape)
            if self.use_connections:
                source_emb_d, source_emb_n, destination_emb, negative_emb = node_embedding
                score = self.affinity_score(torch.cat([source_emb_d, source_emb_n], dim=0),
                                            torch.cat([destination_emb,
                                                       negative_emb])).squeeze(dim=0)
            else:
                source_node_embedding = node_embedding[:n_samples]
                destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
                negative_node_embedding = node_embedding[2 * n_samples:]
                score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                            torch.cat([destination_node_embedding,
                                                       negative_node_embedding])).squeeze(dim=0)
        else:
            source_node_embedding = node_embedding[:n_samples]
            destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
            negative_node_embedding = node_embedding[2 * n_samples:]
            score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                        torch.cat([destination_node_embedding,
                                                   negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return memory, pos_score.sigmoid(), neg_score.sigmoid()

    def compute_node_classfication(self, source_nodes, destination_nodes, edge_times,
                                   edge_idxs, batch_i, batch_num, flist, n_neighbors=20):
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
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, _ = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, destination_nodes, edge_times, edge_idxs, batch_i, batch_num, flist,
            n_neighbors)

        pred = self.node_predictor(source_node_embedding).sigmoid()


        return pred

    def compute_time(self, edge_times, source_nodes, destination_nodes, negative_nodes, last_update):
        ### Compute differences between the time the memory of a node was last updated,
        ### and the time for which we want to compute the embedding of a node
        source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            source_nodes].long()
        source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
        destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            destination_nodes].long()
        destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
        negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            negative_nodes].long()
        negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

        time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                               dim=0)
        return time_diffs

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages,
                self.memory.cor_destination)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)


    # def update_map(self, nodes, memory, batch_i, batch_num, flist, last=True):
    #     if last:
    #         memory = memory
    #     else:
    #         memory = self.memory.get_memory(list(range(self.n_nodes)))
    #     if batch_i > 0:
    #         self.memory_map_updater.assign_map(nodes, memory[nodes])
    #     # update_center_emb_list, update_node_ids_list = self.memory_map_updater.update_map(nodes,
    #     #                             memory[nodes], batch_i, batch_num, flist, self.map_use_message)
    #     return #update_center_emb_list, update_node_ids_list



    def update_memory_map(self,  batch_i, batch_num, flist, memory_updater):
        if self.use_memory_map:
            # aggregate messages from k-1 batch
            unique_nodes, unique_messages, unique_timestamps, unique_destination = \
                self.memory_map_updater.aggregate(self.aggregator_type)
            #if len(unique_nodes2) > 0:
                #unique_messages2 = self.message_function.compute_message(unique_messages2)
            embedding = self.memory.get_memory(unique_nodes)
            self.memory_map_updater.update_memory_map(unique_nodes, embedding, batch_i, batch_num, flist,
                                        self.map_use_message, unique_messages, unique_timestamps, unique_destination,memory_updater)

    def get_updated_memory(self, nodes, messages, ): # positives, batch_i, batch_num, flist
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages,
                self.memory.cor_destination)  # message before

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)
        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, edge_times, edge_idxs, num):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        num = torch.tensor(num).float().to(self.device).unsqueeze(dim=1)
        edge_features = self.edge_raw_features[edge_idxs]

        # num_emb = self.num_encode_layer(num)
        source_memory = self.memory.get_memory(source_nodes) if not \
            self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
            not self.use_destination_embedding_in_message else destination_node_embedding

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
            source_nodes), -1)

        # _,source_cell =self.memory_map.get_map_from_nodeid(source_nodes)
        # _, des_cell = self.memory_map.get_map_from_nodeid(destination_nodes)
        source_message = torch.cat([source_memory, destination_memory, edge_features, num,
                                    source_time_delta_encoding],
                                   dim=1)


        messages = defaultdict(list)
        cor_destinstion = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))
            cor_destinstion[source_nodes[i]].append(destination_nodes[i])

        return unique_sources, messages, cor_destinstion

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.memory_map_updater.neighbor = neighbor_finder
        self.memory_map.neighbor = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
