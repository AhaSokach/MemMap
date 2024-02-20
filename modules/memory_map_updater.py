import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch_scatter import scatter_add, scatter_max
from utils.utils import MergeLayer # , get_temporal_neighbor_unique


class MemoryMapUpdater(nn.Module):
    def update_map(self, nodes, embeddings, batch_i, batch_num, flist, map_use_message):
        pass
        update_center_emb_list=[]
        update_node_ids_list=[]
        return update_center_emb_list, update_node_ids_list

    def find_center_distances(self, unique_nodes, unique_embeddings, unique_times,
                          batch_i):
        source_distances = self.distance_calculator(unique_embeddings,)
        min_indices = torch.argmin(source_distances, dim=1)
        list_num = torch.arange(self.size * self.size).to(self.device)
        cells = list_num[min_indices]
        self.memory_map.set_nodes_distance(unique_nodes, cells, unique_times)

    def find_center_distances_edge(self, nodes, embeddings, times,
                          batch_i):

        nodes_unique, index, indices = np.unique(nodes, return_index=True, return_inverse=True)
        nodes = torch.tensor(nodes).to(self.device)
        unique_embeddings = embeddings[index]
        times = torch.tensor(times).to(self.device)
        source_distances = self.distance_calculator(unique_embeddings,)
        min_indices = torch.argmin(source_distances, dim=1)
        list_num = torch.arange(self.size * self.size).to(self.device)
        cells = list_num[min_indices]
        cells_all = cells[indices]
        self.memory_map.set_nodes_distance(nodes, cells_all, times)


    def find_center_iteration(self, source_nodes_unique, destination_nodes_unique, edge_times,
                          unique_source_embeddings, unique_destination_embeddings,
                          batch_i):

        # deal all -1 data
        # source_cells, _ = self.memory_map.get_map_from_nodeid(source_nodes_unique_unique)
        # source_cor_cells, _ = self.memory_map.get_map_from_nodeid(source_cor_nodes_unique)
        #
        # no_source_cell_bool = torch.eq(source_cells, -1)
        # no_source_cor_cell_bool = torch.eq(source_cor_cells, -1)
        #
        # both_no_source_cell_bool = no_source_cell_bool * no_source_cor_cell_bool
        # both_no_nodes = source_nodes_unique_unique[both_no_source_cell_bool]
        # random_cells = self.memory_map.get_random_cells(both_no_nodes.shape[0])
        # self.memory_map.set_new_nodes(both_no_nodes, random_cells, source_times[both_no_source_cell_bool])

        instance_nums = source_nodes_unique.shape[0]
        j=0
        for i in range(instance_nums):
            source_node = source_nodes_unique[i]
            destination_node = destination_nodes_unique[i]
            source_cell, _ = self.memory_map.get_map_from_nodeid(source_node)
            destination_cell, _ = self.memory_map.get_map_from_nodeid(destination_node)
            if (source_cell == -1) and (destination_cell == -1):
                random_cell = self.memory_map.get_random_cells(1)[0]
                self.memory_map.set_new_nodes(torch.stack((source_node, destination_node)), torch.stack((random_cell,random_cell)),
                                              torch.stack((edge_times[i], edge_times[i])))
                j+=1
            elif (source_cell != -1) and (destination_cell == -1):
                source_position = self.memory_map.get_position(source_node)
                self.memory_map.set_new_nodes(destination_node.view(-1), source_cell.view(-1),
                                              edge_times[i].view(-1), source_position.view(-1))
            elif (source_cell == -1) and (destination_cell != -1):
                destination_position = self.memory_map.get_position(destination_node)
                self.memory_map.set_new_nodes(source_node.view(-1), destination_cell.view(-1),
                                              edge_times[i].view(-1), destination_position.view(-1))
            else:
                source_position = self.memory_map.get_position(source_node)
                destination_position = self.memory_map.get_position(destination_node)
                if source_position < destination_position:
                    neighbor_cells = self.memory_map.get_neighbor_cells(source_node.view(1,-1))
                    cell_embeddings = self.memory_map.get_map(neighbor_cells)
                    neighbor_distances = self.distance_calculator(
                        unique_destination_embeddings[i].view(1,-1),
                        cell_embeddings)  # (117, 484)
                    min_indices = torch.argmin(neighbor_distances)  # (117,)
                    change_cells = neighbor_cells[0][min_indices]
                    self.memory_map.set_nodes(destination_node.view(-1), change_cells.view(-1),
                                              source_position.view(-1), edge_times[i].view(-1))
                elif source_position > destination_position:
                    neighbor_cells = self.memory_map.get_neighbor_cells(destination_node.view(1, -1))
                    cell_embeddings = self.memory_map.get_map(neighbor_cells)
                    neighbor_distances = self.distance_calculator(
                        unique_source_embeddings[i].view(1, -1),
                        cell_embeddings)  # (117, 484)
                    min_indices = torch.argmin(neighbor_distances)  # (117,)
                    change_cells = neighbor_cells[0][min_indices]
                    self.memory_map.set_nodes(source_node.view(-1), change_cells.view( -1),
                                              destination_position.view( -1), edge_times[i].view( -1))



    def find_center(self, source_nodes_unique, source_cor_nodes, source_times,
                                     unique_source_embeddings, batch_i, des=False):
        # print(batch_i)
        # if des:
        #     use_change = True
        # else:
        #     use_change = False
        use_change = True

        source_cells,_ = self.memory_map.get_map_from_nodeid(source_nodes_unique)
        source_cor_cells, _ = self.memory_map.get_map_from_nodeid(source_cor_nodes)
        no_source_cell_bool = torch.eq(source_cells, -1)
        no_source_cor_cell_bool = torch.eq(source_cor_cells, -1)
        #return_cells = torch.tensor([-1]*source_nodes_unique.shape[0]).to(self.device)

        # source -1 des -1
        both_no_source_cell_bool = no_source_cell_bool * no_source_cor_cell_bool
        both_no_nodes = source_nodes_unique[both_no_source_cell_bool]
        random_cells = self.memory_map.get_random_cells(both_no_nodes.shape[0])
        self.memory_map.set_new_nodes(both_no_nodes, random_cells, source_times[both_no_source_cell_bool])
        #return

        # source 1 dis -1
        # self.memory_map.set

        # source -1 dis 1
        no_source_with_cor_cell_bool = no_source_cell_bool * (~no_source_cor_cell_bool)
        no_source_with_cor_nodes = source_nodes_unique[no_source_with_cor_cell_bool].long()
        no_source_with_cor_nodes_cells = source_cor_cells[no_source_with_cor_cell_bool]
        no_source_with_cor_nodes_positions = self.memory_map.get_position(source_cor_nodes[no_source_with_cor_cell_bool])
        self.memory_map.set_new_nodes(no_source_with_cor_nodes, no_source_with_cor_nodes_cells,
                                      source_times[no_source_with_cor_cell_bool],
                                      no_source_with_cor_nodes_positions, )

        # source 1 des 1
        source_with_cor_cell_bool = (~no_source_cell_bool) * (~no_source_cor_cell_bool)
        source_with_cor_nodes = source_nodes_unique[source_with_cor_cell_bool]
        source_with_cor_cor_nodes = source_cor_nodes[source_with_cor_cell_bool]
        source_with_cor_nodes_positions = self.memory_map.get_position(source_with_cor_nodes)
        source_with_cor_cor_nodes_positions = self.memory_map.get_position(source_with_cor_cor_nodes)

        # find nodes need to change cells
        cor_priority = torch.lt(source_with_cor_nodes_positions, source_with_cor_cor_nodes_positions)
        change_source_nodes = source_with_cor_nodes[cor_priority]
        change_source_nodes_positions = source_with_cor_cor_nodes_positions[cor_priority]
        change_source_cor_nodes_cells = source_cor_cells[source_with_cor_cell_bool][cor_priority]
        if use_change:
            change_bool = torch.eq(source_with_cor_nodes_positions, source_with_cor_cor_nodes_positions)
            change_source_nodes_eq = source_with_cor_nodes[change_bool]
            change_cor_nodes_cells_eq = source_cor_cells[source_with_cor_cell_bool][change_bool]
            if change_source_nodes_eq.shape[0] != 0:
                neighbor_cells_eq = self.memory_map.get_neighbor_cells(change_cor_nodes_cells_eq,offset=1)
                cell_embeddings_eq = self.memory_map.get_map(neighbor_cells_eq)
                neighbor_distances_eq = self.distance_calculator(
                    unique_source_embeddings[source_with_cor_cell_bool][change_bool], cell_embeddings_eq)  # (117, 484)
                min_distances_eq, min_indices_eq = torch.min(neighbor_distances_eq, dim=1)  # (117,)
                change_cells_eq = neighbor_cells_eq[torch.arange(min_indices_eq.shape[0]), min_indices_eq]
                self.memory_map.set_nodes(change_source_nodes_eq, change_cells_eq, source_with_cor_cor_nodes_positions[change_bool],
                                          source_times[source_with_cor_cell_bool][change_bool], self.use_connections)

        # change_source_nodes = source_with_cor_nodes
        # change_source_nodes_positions = source_with_cor_cor_nodes_positions
        # change_source_cor_nodes_cells = source_cor_cells[source_with_cor_cell_bool]
        if change_source_nodes.shape[0] != 0:
            neighbor_cells = self.memory_map.get_neighbor_cells(change_source_cor_nodes_cells)
            cell_embeddings = self.memory_map.get_map(neighbor_cells)
            neighbor_distances = self.distance_calculator(unique_source_embeddings[source_with_cor_cell_bool][cor_priority], cell_embeddings)  # (117, 484)
            min_distances, min_indices = torch.min(neighbor_distances, dim=1)  # (117,)
            change_cells = neighbor_cells[torch.arange(min_indices.shape[0]), min_indices]
            self.memory_map.set_nodes(change_source_nodes, change_cells, change_source_nodes_positions,
                                      source_times[source_with_cor_cell_bool][cor_priority], self.use_connections)


    def get_map_from_embeddings(self, nodes, embeddings):
        centers, _ = self.find_center(embeddings)
        dict_be = self.memory_map.belongingness
        for i in range(len(nodes)):
            n = nodes[i]
            c = dict_be.get(n)
            if c != None:
                centers[i] = torch.tensor(c[-1]).to(self.device)

        # memory_map_reshape = torch.reshape(self.memory_map, (-1, self.memory_dimension))
        map_nodes = self.memory_map.get_map(centers)
        return map_nodes, centers

    def distance_calculator(self, batch_x, map_embeddings=None):
        if (self.distance_function == "euclidean"):
            return self._euclidean_distance(batch_x, map_embeddings)
        elif (self.distance_function == "cosine"):
            return self._cosine_distance(batch_x, map_embeddings)

    def _euclidean_distance(self, batch_x, map_embeddings=None):
        # map_reshape = torch.reshape(self.memory_map.memory_map, (-1, 172))
        if map_embeddings != None:
            # ...
            distances = torch.cdist(batch_x.view(batch_x.shape[0],1,-1), map_embeddings, p=2).squeeze(1)
            # torch.sqrt(torch.sum((map_embeddings - batch_x.view(batch_x.shape[0],1,-1)) ** 2, dim=2)))
            #torch.cdist(batch_x[0].view(1,-1), map_embeddings[1], p=2)
            #
        else:
            distances = torch.cdist(batch_x, self.memory_map.memory_map.data.clone(), p=2)
            # distances = torch.mm(batch_x, self.memory_map.memory_map.T)

        self.memory_map.detach_map()
        #distances.detach_()
        return distances

    def _cosine_distance(self, batch_x, map_embeddings=None):
        # map_reshape = torch.reshape(self.memory_map.memory_map, (-1, 172))
        distances = F.cosine_similarity(batch_x.view(batch_x.shape[0],1,batch_x.shape[1]),
                                        self.memory_map.memory_map.view(1,-1,batch_x.shape[-1]), dim=-1)
        self.memory_map.detach_map()
        #distances.detach_()
        return distances

    def translate_id(self, ids):
        rows = ids // self.size
        columns = ids % self.size
        return rows, columns

    def translate_id_batch(self, ids, ):
        batch_size = ids.shape[0]
        num = torch.arange(batch_size)
        num_base = num * self.size * self.size
        ids_translate = num_base + ids
        return ids_translate

    def decay_function(self, learning_rate, t, max_iter, neigh=True):
        w = learning_rate / (1 + self.size // 2 * t / (max_iter / 2))
        # if ((w<2.5) and (neigh) ):
            # w =2.5
        return w

    def neighbor_finder(self, cell_ids, rows, columns, sigma, batch_size, map_use_message=False):
        if (self.neighbor_function == "bubble"):
            return self._bubble(rows, columns, sigma, batch_size)
        elif (self.neighbor_function == "_mexican_hat"):
            return self._mexican_hat_list(cell_ids,  rows, columns, sigma, batch_size, map_use_message)

    def _bubble(self, rows, columns, sigma, batch_size):
        list = torch.arange(self.size)
        # neigx = list.repeat(batch_size,1)
        # neigy = list.repeat(batch_size, 1)
        neigx = torch.reshape(torch.repeat_interleave(list, batch_size, dim=0), (-1, batch_size)).to(self.device)
        neigy = torch.reshape(torch.repeat_interleave(list, batch_size, dim=0), (-1, batch_size)).to(self.device)

        ax = torch.logical_and(torch.gt(neigx, rows - sigma), torch.lt(neigx, rows + sigma))
        ay = torch.logical_and(torch.gt(neigy, columns - sigma), torch.lt(neigy, columns + sigma))
        ax = torch.transpose(ax, 0, 1)
        ay = torch.transpose(ay, 0, 1)
        neighbor_map_update_bool = torch.tensor([]).to(self.device)
        for i in range(batch_size):
            ax_i = ax[i]
            ay_i = ay[i]
            neighbor = torch.outer(ax_i, ay_i) * 1.
            neighbor = torch.reshape(neighbor, (1, self.size, self.size))
            if (i == 0):
                neighbor_map_update_bool = neighbor
            else:
                neighbor_map_update_bool = torch.cat((neighbor_map_update_bool, neighbor), dim=0)

        return neighbor_map_update_bool

    def _mexican_hat(self, rows, columns, sigma, batch_size, map_use_message=False):
        if map_use_message:
            xy = torch.tensor([rows[0], columns[0]]).to(self.device)
            p = torch.pow(self.grid_x - self.grid_x.T[rows[0]][columns[0]], 2) + torch.pow(self.grid_y - self.grid_y.T[rows[0]][columns[0]], 2)
            d = 4 * sigma * sigma
            neigh = (torch.exp(-p / d) * (1 - 2 / d * p)).T
            neigh[rows[0]][columns[0]] = 0.0
            return neigh
        else:
            s_xy = torch.tensor([rows[0], columns[0]]).to(self.device)
            d_xy = torch.tensor([rows[1], columns[1]]).to(self.device)
            p_s = torch.pow(self.grid_x - self.grid_x.T[rows[0]][columns[0]], 2) + torch.pow(self.grid_y - self.grid_y.T[rows[0]][columns[0]], 2)
            p_d = torch.pow(self.grid_x - self.grid_x.T[rows[1]][columns[1]], 2) + torch.pow(self.grid_y - self.grid_y.T[rows[1]][columns[1]], 2)
            d = 4 * sigma * sigma
            neigh_s = (torch.exp(-p_s/d)*(1-2/d*p_s)).T
            neigh_d = (torch.exp(-p_d / d) * (1 - 2 / d * p_d)).T
            neigh_s[rows[0]][columns[0]] = 0.0
            neigh_d[rows[1]][columns[1]] = 0.0
        return torch.stack((neigh_s, neigh_d), dim=0)


    def _mexican_hat_list(self, cell_ids, rows, columns, sigma, num, map_use_message=False):
        map_weight_batch = []
        weight_index_batch = torch.tensor([]).to(self.device)
        weight_bool_batch = []
        center_index_batch = []
        use_cell = torch.zeros(self.size * self.size).to(self.device)
        index = torch.arange(self.size * self.size).to(self.device)
        for i in range(num):
            p = torch.pow(self.grid_x - self.grid_x.T[rows[i]][columns[i]], 2) + torch.pow(self.grid_y - self.grid_y.T[rows[i]][columns[i]], 2)
            d = 4 * sigma * sigma
            neigh = (torch.exp(-p / d) * (1 - 2 / d * p)).T # * 0.5
            neigh[rows[i]][columns[i]] = 0.0

            weight_index, weight_bool = self.memory_map.get_true_weight(neigh)
            neigh = torch.reshape(neigh, (self.size * self.size,))
            map_weight_batch.append(neigh)
            weight_index_batch = torch.concat((weight_index_batch, weight_index))
            weight_bool_batch.append(weight_bool)
            use_cell = use_cell.bool() + weight_bool.bool()
            # l = [[0]* weight_index.shape[0]]
            center_index_batch.extend([cell_ids[i]] * weight_index.shape[0])
        # use_cell_batch = torch.gt(use_cell, 0)
        use_index = index[use_cell]
        # print(weight_index.shape)
        # print(sigma)

        map_weight_batch = torch.stack(map_weight_batch)
        weight_bool_batch = torch.stack(weight_bool_batch)
        # weight_index_batch = torch.tensor(weight_index_batch).to(self.device)
        center_index_batch = torch.tensor(center_index_batch).to(self.device)

        return map_weight_batch, weight_index_batch, weight_bool_batch, use_index, center_index_batch



    def display_map(self, flist, batch_i, batch_num, sig, z_center, z_neighbor, dis_num = 5):
        interval = batch_num // dis_num
        print(f'batch:{batch_i}/{batch_num}, neighbor:{sig}, '
              f'weight:{z_center}, w_n:{z_neighbor}', file=flist[2])
        if(batch_i % interval == 0):
            distances = self.memory_map.display_map()
            choose_dis = [30, 100, 200, 300]
            for i in choose_dis:
                print(f'node{i} batch{batch_i}:', file=flist[0])
                print(np.round(distances[i].cpu().detach().numpy(),2), file=flist[0])
                print(f'neighbor:{sig}, weight:{z_center}', file=flist[0])
            print(f'batch:{batch_i}/{batch_num}',file=flist[1])
            self.memory_map.print_map(flist[1])
        #print(distances, file=file)


class NetMemoryUpdater(MemoryMapUpdater):
    def __init__(self, assign_type,memory_map, map_dimension, message_dimension, map_layer, message_aggregator, time_encoder, time_dim, neighbor, dropout, device, bias=True):
        super(NetMemoryUpdater, self).__init__()
        self.memory_map = memory_map
        self.distance_function = "euclidean"
        self.map_dimension = map_dimension
        self.neighbor = neighbor
        self.linear_map1 = torch.nn.Linear(map_dimension, map_dimension, bias=bias)
        self.linear_map2 = torch.nn.Linear(map_dimension, map_dimension, bias=bias)
        self.linear_emb = torch.nn.Linear(map_dimension, map_dimension, bias=bias)
        self.linear_neighbor1 = torch.nn.Linear(map_dimension, map_dimension, bias=bias)
        self.linear_neighbor2 = torch.nn.Linear(map_dimension, map_dimension, bias=bias)
        self.linear_center = torch.nn.Linear(map_dimension, map_dimension, bias=bias)
        self.activation_center = torch.nn.ReLU()
        self.activation_neighbor = torch.nn.ReLU()
        self.message_aggregator = message_aggregator
        self.time_encoder = time_encoder
        self.map_layer = map_layer
        self.assign_type=assign_type
        #self.linear_z = nn.Linear(map_dimension * 2 + 0, map_dimension,
        #                          bias=bias)  # TODO the dimension of weight is 1 or map_dimension
        self.linear_z = nn.Linear(344, 1, bias=bias)
        self.linear_message = nn.Linear(688, 172, bias=bias)
        self.linear_dis = nn.Linear(1,1)
        self.size = memory_map.size
        self.last_batch_node = []
        self.messages = defaultdict(list)
        self.cor_destination = defaultdict(list)
        self.neighbor_function = "_mexican_hat"
        self.learning_rate = 0.1
        self.sigma = int(self.size // 10.0)  
        self.x = torch.arange(self.size)
        self.y = torch.arange(self.size)
        self.grid_x, self.grid_y = torch.meshgrid(self.x, self.y)
        self.cell_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=map_dimension)

        self.trajectory_updater = nn.GRUCell(input_size=map_dimension,
                                         hidden_size=map_dimension)
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)



        self.use_feature = True
        self.use_neighbor_feature = True
        self.use_connections= False
        self.query_dim = map_dimension
        self.key_dim = map_dimension
        self.query_dim_traj = map_dimension
        self.key_dim_traj = map_dimension
        self.use_neighbor_num = False
        if self.use_feature:
            # if self.use_neighbor_num:
            #     self.query_dim_traj = map_dimension
            # else:
            self.query_dim_traj = map_dimension + time_dim
            self.key_dim_traj = map_dimension + time_dim
        if self.use_neighbor_feature:
            self.query_dim = map_dimension + time_dim
            self.key_dim = map_dimension + time_dim
        # self.trajectory_updater = nn.GRUCell
        num_dim = 2
        self.position_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.map_dimension),
            nn.ReLU(),
            nn.Linear(in_features=self.map_dimension, out_features=self.map_dimension))
        self.num_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=num_dim),
            nn.ReLU(),
            nn.Linear(in_features=num_dim, out_features=num_dim))
        n_head = 4
        dropout = dropout
        self.use_position = False
        self.aggregate_node = True
        self.multi_head_target_neighbor = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=n_head,
                                                       dropout=dropout)

        self.multi_head_target_traj = nn.MultiheadAttention(embed_dim=self.query_dim_traj,
                                                       kdim=self.key_dim_traj,
                                                       vdim=self.key_dim_traj,
                                                       num_heads=n_head,
                                                       dropout=dropout)
        self.use_num=False
        if self.aggregate_node:
            nodes_head = 2
            if self.use_num:
                self.nodes_query_dim = map_dimension + time_dim + num_dim

                self.nodes_key_dim = map_dimension + time_dim + num_dim
            else:
                self.nodes_key_dim = map_dimension + time_dim
                self.nodes_query_dim = map_dimension + time_dim
            self.multi_head_target_nodes = nn.MultiheadAttention(embed_dim=self.nodes_query_dim,
                                                                kdim=self.nodes_key_dim,
                                                                vdim=self.nodes_key_dim,
                                                                num_heads=nodes_head,
                                                                dropout=dropout)


        merge_dimension1 = self.map_dimension # node_emb
        merge_dimension2 = 0 #self.map_dimension * 2
        self.use_latest_time = False

        flag=False

        if flag:
            self.use_neighbor_cell = True
            self.use_traj_cell = True
            self.use_cell = True
            self.use_batch_nodes = True



            self.use_nodes = False
            self.use_neighbor_batch = False
        else:
            self.use_neighbor_cell = True
            self.use_traj_cell = True
            self.use_cell = True
            self.use_batch_nodes = False

            self.use_nodes = False
            self.use_neighbor_batch = False


        if self.use_cell:
            merge_dimension1 += self.map_dimension

        if self.aggregate_node and (self.use_nodes or self.use_batch_nodes or self.use_neighbor_batch):
            merge_dimension1 = merge_dimension1 + map_dimension + time_dim
            if self.use_num:
                merge_dimension1 += num_dim

        if self.use_traj_cell:
            merge_dimension2 = merge_dimension2 + map_dimension + time_dim

        if self.use_neighbor_cell:
            merge_dimension2 = merge_dimension2 + map_dimension + time_dim

        # if self.use_feature and self.use_neighbor_feature:
        #     merge_dimension1 += 2 * time_dim
        # if self.use_latest_time:
        #     merge_dimension2 += time_dim

        # merge_dimension1 = self.map_dimension
        # merge_dimension2 = self.map_dimension * 2
        self.merger = MergeLayer(merge_dimension1, merge_dimension2, map_dimension * 2, map_dimension)


        self.merger2 = MergeLayer(self.map_dimension, map_dimension * 2, map_dimension * 2, map_dimension)

        self.device = device

    def store_last_node(self, node_list):
        self.last_batch_node = []
        self.last_batch_node.extend(node_list[0])
        self.last_batch_node.extend(node_list[1])

    def store_raw_messages(self, nodes, node_id_to_messages, node_id_to_cordestinstion):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])
            self.cor_destination[node].extend(node_id_to_cordestinstion[node])


    def get_last_update(self, batch_node_ids, embeddings, batch_i, batch_num, flist):
        unique_node_ids, index, indices, counts = np.unique(batch_node_ids, return_index=True,
                                                            return_inverse=True, return_counts=True)
        unique_embeddings = embeddings[index]
        batch_size = unique_embeddings.shape[0]
        '''unique'''
        centers, distances_all = self.find_center(unique_embeddings)  # (117,484)
        true_update = (centers.view(1,-1) == self.last_update.view(-1,1)).any(dim=0)
        memory_map_clone = self.memory_map.get

    def init_map_pair(self, nodes):
        ...


    def update_memory_map(self,  node_ids, embeddings, batch_i, batch_num, flist, map_use_message, unique_messages=None, unique_timestamps=None,unique_destination=None,memory_updater=None):
        if batch_i < 1:
            return
        # print(batch_i)

        unique_node = torch.tensor(node_ids).to(self.device)
        node_num = unique_node.shape[0]
        cell_ids, cell_embs = self.memory_map.get_map_from_nodeid(unique_node)

        # center update
        ones_list_center = torch.tensor([1] * cell_ids.shape[0]).to(self.device)
        cell_nums = scatter_add(ones_list_center, cell_ids.long(), dim=0)
        unique_messages = unique_messages.data.clone()
        aggregated_messages = scatter_add(unique_messages, cell_ids, dim=0)
        aggregated_messages = aggregated_messages[cell_ids] / cell_nums[cell_ids].view(-1,1) # TODO div du shu
        center_cell_update = memory_updater(aggregated_messages, cell_embs)
        self.memory_map.set_map(cell_ids, center_cell_update)
        # self.memory_map.detach_map()

        # updated_trajectory = self.memory_map.get_node_trajectory(unique_node)
        #  # self.memory_map.memory_map.data.clone()[cell_ids]
        # map_emb = center_cell_update.detach().clone()
        # updated_trajectory = self.trajectory_updater(map_emb, updated_trajectory)
        # self.memory_map.set_node_trajectory(unique_node, updated_trajectory)

        # weight calculate
        sig = self.decay_function(self.sigma, batch_i, batch_num)
        eta = self.decay_function(self.learning_rate, batch_i, batch_num, neigh=False)
        rows, columns = self.translate_id(cell_ids)
        map_weight_batch, weight_index_batch, weight_bool_batch, use_index, center_index_batch\
            = self.neighbor_finder(cell_ids, rows, columns, sig, node_num, map_use_message)  # (117,22,22)


        map_weight_use = map_weight_batch.view(-1)[weight_bool_batch.bool().view(-1)]
        minus_weight_use = 1 - map_weight_use
        # neighbor_weight_batch = map_weight_batch[weight_bool_batch]
        # weight_index_batch_reshape = torch.reshape(weight_index_batch, (1,-1))
        ones_list = torch.tensor([1] * weight_index_batch.shape[0]).to(self.device)
        center_nums = scatter_add(ones_list, weight_index_batch.long(), dim=0)
        neighbor_emb_batch = self.memory_map.get_map(weight_index_batch.int())
        center_emb_batch = self.memory_map.get_map(center_index_batch.long())
        #neighbor_weight_batch_reshape = torch.reshape(neighbor_weight_batch, (1,-1))
        new_map_emb = map_weight_use.view(-1,1) * center_emb_batch + minus_weight_use.view(-1,1) * neighbor_emb_batch
        # new_map_emb = new_map_emb
        #weight_neighbor_emb_betch = neighbor_weight_batch_reshape * neighbor_emb_batch
        emb_aggregated = scatter_add(new_map_emb, weight_index_batch.long(), dim=0)
        self.memory_map.set_map(use_index, emb_aggregated[use_index] / center_nums[use_index].view(-1,1))
        #

    def check_center(self, batch_i,memory,nodes):
        if batch_i <= 1:
            return
        distances = self.distance_calculator(memory[nodes])  # (117, 484)
        min_distances, min_indices = torch.min(distances, dim=1)
        #print("")


    def assign_center(self, source_nodes_unique, source_cor_nodes, source_times, destination_nodes_unique, destination_cor_nodes,
                                     destination_times, unique_source_embeddings, unique_destination_embeddings, batch_i):
        self.find_center(source_nodes_unique, source_cor_nodes, source_times,
                                     unique_source_embeddings,  batch_i)
        self.find_center(destination_nodes_unique,
                                   destination_cor_nodes, destination_times,
                                   unique_destination_embeddings, batch_i, des=True)
        # return torch.concat((centers_s,centers_d))

    def assign_center_iteration(self, source_nodes_unique, source_cor_nodes, source_times, destination_nodes_unique, destination_cor_nodes,
                                     destination_times, unique_source_embeddings, unique_destination_embeddings, batch_i):
        self.find_center_iteration(source_nodes_unique, destination_nodes_unique, source_times,
                                     unique_source_embeddings, unique_destination_embeddings, batch_i)
        # self.find_center(destination_nodes_unique,
        #                            destination_cor_nodes, destination_times,
        #                            unique_destination_embeddings, batch_i)


    def assign_map(self, source_nodes, destination_nodes, edge_times, memory, batch_i):
        # if batch_i == 0:
        #     return
        center_type = self.assign_type
        if center_type == "distances":
            nodes = np.concatenate([source_nodes, destination_nodes])
            times = np.concatenate([edge_times, edge_times])
            nodes_unique, index = np.unique(nodes, return_index=True)
            node_embeddings = memory[nodes_unique]
            times = times[index]
            nodes_unique = torch.tensor(nodes_unique).to(self.device)
            times = torch.tensor(times).to(self.device)
            self.find_center_distances(nodes_unique, node_embeddings, times, batch_i)
        elif center_type == "distances_edge":
            nodes = np.concatenate([source_nodes, destination_nodes])
            times = np.concatenate([edge_times, edge_times])
            # nodes_unique , index = np.unique(nodes, return_index=True, axis=0)
            node_embeddings = memory[nodes]
            # times = times[index]
            # nodes_unique = torch.tensor(nodes).to(self.device)
            # times = torch.tensor(times).to(self.device)
            self.find_center_distances_edge(nodes, node_embeddings, times, batch_i)
        elif center_type == "unique_edge":
            # source_nodes_unique_unique, source_index_unique, source_inverse_unique = np.unique(source_nodes,
            #                                                               return_index=True, return_inverse=True)
            # source_nodes_unique_unique = torch.tensor(source_nodes_unique_unique).to(self.device)
            source_nodes = torch.tensor(source_nodes).to(self.device)
            destination_nodes = torch.tensor(destination_nodes).to(self.device)
            edge = torch.stack((source_nodes, destination_nodes), dim=1)
            edge_unique, inverse = torch.unique(edge, dim=0, return_inverse=True)
            index = torch.argmax((inverse == torch.arange(edge_unique.shape[0]).to(self.device).unsqueeze(1)).int(), dim=1)
            source_nodes_unique = edge_unique[:,0]
            source_index = index
            destination_nodes_unique = edge_unique[:,1]
            #destination_index = index
            # source_cor_nodes = destination_nodes_unique
            # destination_cor_nodes = source_nodes_unique
            source_times = torch.tensor(edge_times).to(self.device)[source_index]
            # destination_times = source_times

            unique_source_embeddings = memory[source_nodes_unique]
            unique_destination_embeddings = memory[destination_nodes_unique]

            self.find_center_iteration(source_nodes_unique, destination_nodes_unique, source_times,

                               unique_source_embeddings, unique_destination_embeddings,
                               batch_i)


        elif center_type == "position":

            source_nodes_unique, source_index, source_inverse = np.unique(source_nodes,
                                                                             return_index=True, return_inverse=True)

            destination_nodes_unique, destination_index, destination_inverse = np.unique(destination_nodes,
                                                                                             return_index=True,
                                                                                             return_inverse=True)

            source_cor_nodes = destination_nodes[source_index]
            destination_cor_nodes = source_nodes[destination_index]
            source_nodes_unique = torch.tensor(source_nodes_unique).to(self.device)
            source_times = torch.tensor(edge_times[source_index]).to(self.device)
            destination_nodes_unique = torch.tensor(destination_nodes_unique).to(self.device)
            destination_times = torch.tensor(edge_times[destination_index]).to(self.device)
            source_cor_nodes = torch.tensor(source_cor_nodes).to(self.device)
            destination_cor_nodes = torch.tensor(destination_cor_nodes).to(self.device)



            # nodes =  np.concatenate([source_nodes, destination_nodes])
            # cor_destination =np.concatenate([destination_nodes, source_nodes])
            # unique_nodes, index = np.unique(nodes, return_index=True)
            unique_source_embeddings = memory[source_nodes_unique]
            unique_destination_embeddings = memory[destination_nodes_unique]

            # nodes_embeddings = memory[unique_nodes]
            # unique_cor_destination = cor_destination[index]
            self.assign_center(source_nodes_unique, source_cor_nodes, source_times, destination_nodes_unique,
                               destination_cor_nodes, destination_times,
                                         unique_source_embeddings, unique_destination_embeddings, batch_i)
            #self.memory_map.set_node_to_cell(unique_nodes, centers)
        # if self.use_batch_nodes:
        #     self.memory_map.add_nodes_batch(source_nodes, destination_nodes, edge_times)


    def assign_map_after_batch(self, source_nodes, destination_nodes, edge_times, memory, batch_i):
        # if batch_i == 0:
        #     return
        center_type = "distances"
        if center_type == "distances":
            nodes = np.concatenate([source_nodes, destination_nodes])
            times = np.concatenate([edge_times, edge_times])
            nodes_unique, index = np.unique(nodes, return_index=True)
            node_embeddings = memory[nodes_unique]
            times = times[index]
            nodes_unique = torch.tensor(nodes_unique).to(self.device)
            times = torch.tensor(times).to(self.device)
            self.find_center_distances(nodes_unique, node_embeddings, times, batch_i)

        elif center_type == "unique_edge":
            # source_nodes_unique_unique, source_index_unique, source_inverse_unique = np.unique(source_nodes,
            #                                                               return_index=True, return_inverse=True)
            # source_nodes_unique_unique = torch.tensor(source_nodes_unique_unique).to(self.device)
            source_nodes = torch.tensor(source_nodes).to(self.device)
            destination_nodes = torch.tensor(destination_nodes).to(self.device)
            edge = torch.stack((source_nodes, destination_nodes), dim=1)
            edge_unique, inverse = torch.unique(edge, dim=0, return_inverse=True)
            index = torch.argmax((inverse == torch.arange(edge_unique.shape[0]).to(self.device).unsqueeze(1)).int(), dim=1)
            source_nodes_unique = edge_unique[:,0]
            source_index = index
            destination_nodes_unique = edge_unique[:,1]
            #destination_index = index
            # source_cor_nodes = destination_nodes_unique
            # destination_cor_nodes = source_nodes_unique
            source_times = torch.tensor(edge_times).to(self.device)[source_index]
            # destination_times = source_times

            unique_source_embeddings = memory[source_nodes_unique]
            unique_destination_embeddings = memory[destination_nodes_unique]

            self.find_center_iteration(source_nodes_unique, destination_nodes_unique, source_times,

                               unique_source_embeddings, unique_destination_embeddings,
                               batch_i)


        elif center_type == "position":

            source_nodes_unique, source_index, source_inverse = np.unique(source_nodes,
                                                                             return_index=True, return_inverse=True)

            destination_nodes_unique, destination_index, destination_inverse = np.unique(destination_nodes,
                                                                                             return_index=True,
                                                                                             return_inverse=True)

            source_cor_nodes = destination_nodes[source_index]
            destination_cor_nodes = source_nodes[destination_index]
            source_nodes_unique = torch.tensor(source_nodes_unique).to(self.device)
            source_times = torch.tensor(edge_times[source_index]).to(self.device)
            destination_nodes_unique = torch.tensor(destination_nodes_unique).to(self.device)
            destination_times = torch.tensor(edge_times[destination_index]).to(self.device)
            source_cor_nodes = torch.tensor(source_cor_nodes).to(self.device)
            destination_cor_nodes = torch.tensor(destination_cor_nodes).to(self.device)



            # nodes =  np.concatenate([source_nodes, destination_nodes])
            # cor_destination =np.concatenate([destination_nodes, source_nodes])
            # unique_nodes, index = np.unique(nodes, return_index=True)
            unique_source_embeddings = memory[source_nodes_unique]
            unique_destination_embeddings = memory[destination_nodes_unique]

            # nodes_embeddings = memory[unique_nodes]
            # unique_cor_destination = cor_destination[index]
            self.assign_center(source_nodes_unique, source_cor_nodes, source_times, destination_nodes_unique,
                               destination_cor_nodes, destination_times,
                                         unique_source_embeddings, unique_destination_embeddings, batch_i)
            #self.memory_map.set_node_to_cell(unique_nodes, centers)

    # def assign_map_interation(self, source_nodes, destination_nodes, timestamps):
    #     source_nodes_tensor = torch.tensor(source_nodes).to(self.device)
    #     destination_nodes_tensor = torch.tensor(destination_nodes).to(self.device)
    #     edges = torch.stack((source_nodes_tensor,destination_nodes_tensor), dim=1)
    #     unique_edges, inverse = torch.unique(edges, dim=0, return_inverse=True)


    def calculate_attention(self, query, key, neighbors_padding_mask, multi_head_target):
        query = query.data.clone()
        key = key.data.clone()
        query = query.permute([1, 0, 2])
        key = key.permute([1, 0, 2])

        neighbors_padding_mask = ~neighbors_padding_mask.bool()
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)

        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False


        attn_output, attn_output_weights = multi_head_target(query=query, key=key, value=key,
                                                                  key_padding_mask=neighbors_padding_mask)

        attn_output = attn_output.squeeze()
        attn_output_weights = attn_output_weights.squeeze()

        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        # Skip connection with temporal attention over neighborhood and the features of the node itself
        #attn_output = self.merger(attn_output, src_node_features)

        return attn_output # , attn_output_weights

    def aggregate_trajectory(self, batch_i, epoch, nodes, memory, timestamps, flist, assign_after_batch, write=False):
        if batch_i <= 1:
            instance_num = nodes.shape[0] // 3
            nodes_unique, index = np.unique(nodes[:instance_num*2], return_index=True)
            index = torch.tensor(index).to(self.device)
            times = torch.tensor(timestamps).to(self.device)[index]
            cells = self.memory_map.get_cells_from_trajectory(nodes_unique,times)
            self.memory_map.set_nodes_after_batch(nodes_unique, cells, times)
            # self.memory_map.nodes_batch.clear()
            return

        max_num=25
        trajectory_nodes, trajectory_bool, trajectory_times, cells = self.memory_map.get_trajectory_from_nodeid(nodes, timestamps, max_num)
        # cells = self.memory_map.node_to_map[nodes]
        #unique_nodes, index, inverse = np.unique(nodes, return_index=True, return_inverse=True)
        if assign_after_batch:
            cells = self.memory_map.get_cellid(nodes)
        cell_ids, unknown_ind = self.get_cell_unknown(nodes, cells, memory)
        cell_embedding = self.memory_map.get_map(cell_ids)


        nodes = np.array(nodes)
        if self.use_batch_nodes:
            nodes_max_num = 10
            nodes_batch, nodes_batch_time, nodes_bool = self.memory_map.get_nodes_batch(nodes,timestamps,nodes_max_num)

            if self.use_num:

                nodes_emb, cells_nodes_emb, cells_nodes_bool = self.memory_map.get_connections3(nodes, nodes_batch,
                                                                                                memory,
                                                                                                timestamps,
                                                                                                self.time_encoder,
                                                                                                self.num_encode_layer,
                                                                                                nodes_max_num,
                                                                                                edge_times=nodes_batch_time,
                                                                                                neigh=True,
                                                                                                calculate=True)
            else:
                nodes_emb, cells_nodes_emb, cells_nodes_bool = self.memory_map.get_connections2(nodes, nodes_batch,
                                                                                                memory,
                                                                                                timestamps,
                                                                                                self.time_encoder,
                                                                                                self.num_encode_layer,
                                                                                                nodes_max_num,
                                                                                                edge_times=nodes_batch_time,
                                                                                                neigh=True,
                                                                                                calculate=False)
            # cell_nodes, _, edge_times = self.neighbor.get_temporal_neighbor_unique(nodes, timestamps, nodes_max_num)
            instance_num = nodes.shape[0] // 3
            nodes_bool = torch.cat((nodes_bool,nodes_bool[:instance_num]))
            neighbor_node_output = self.calculate_attention(nodes_emb,
                                                            cells_nodes_emb,
                                                            nodes_bool, self.multi_head_target_nodes)
        elif self.use_nodes:
            nodes_max_num = 10
            # neighbor_nodes, _, edge_times = self.neighbor.get_temporal_neighbor_unique(nodes, timestamps, nodes_max_num)
            # neighbor_nodes=None
            cell_nodes = self.memory_map.get_nodes_in_cells(cells, nodes_max_num, batch_i,epoch,flist,nodes2=nodes,
                                                            neighbornodes=None)
            # cell_nodes, _, edge_times = self.neighbor.get_temporal_neighbor_unique(nodes, timestamps, nodes_max_num)
            # cell_nodes = torch.tensor(cell_nodes).to(self.device)
            # cell_node_embs = memory[cell_nodes]
            nodes_emb, cells_nodes_emb, cells_nodes_bool = self.memory_map.get_connections2(nodes, cell_nodes, memory,
                                                        timestamps, self.time_encoder, self.num_encode_layer, nodes_max_num,
                                                        edge_times=None, neigh=False,calculate=True)
            # instance_num = cell_nodes.shape[0] // 3
            # cell_nodes_dup = torch.cat((cell_nodes,cell_nodes[:instance_num]))
            # cells_nodes_bool = cell_nodes_dup != -1
            neighbor_node_output = self.calculate_attention(nodes_emb,
                                                    cells_nodes_emb,
                                                    cells_nodes_bool, self.multi_head_target_nodes)
        elif self.use_neighbor_batch:
            nodes_max_num = 10
            neighbors,times,neighbors_bool = self.memory_map.get_neighbor_batch(nodes,nodes_max_num)
            nodes_emb, cells_nodes_emb, cells_nodes_bool = self.memory_map.get_connections2(nodes, neighbors, memory,
                                                                                            timestamps,
                                                                                            self.time_encoder,
                                                                                            self.num_encode_layer,
                                                                                            nodes_max_num,
                                                                                            edge_times=times,
                                                                                            neigh=True)
            instance_num = nodes.shape[0] // 3
            nodes_bool = torch.cat((neighbors_bool, neighbors_bool[:instance_num]))
            neighbor_node_output = self.calculate_attention(nodes_emb,
                                                            cells_nodes_emb,
                                                            nodes_bool, self.multi_head_target_nodes)
        else:
            neighbor_node_output = None
            # instance_num = nodes.shape[0] // 3
            # source_nodes_tensor = torch.tensor(nodes[0:instance_num]).to(self.device)
            # destination_nodes_tensor = torch.tensor(nodes[instance_num:2*instance_num]).to(self.device)
            # source_nodes_tensor_dup = source_nodes_tensor.repeat(instance_num*3)
            # destination_nodes_tensor_dup = destination_nodes_tensor.repeat(instance_num * 3)
            #
            # edges =
            # time_dup = timestamps.repeat(1,nodes.shape[0])
            # use_time = torch.lt(time_dup,timestamps.unsqueeze(1))



        if self.use_neighbor_cell:
            layer = self.map_layer
            neigh_max_num = (2 * layer + 1) * (2 * layer + 1)
            neighbor_cells, neighbor_bool = self.memory_map.get_neighbor_from_nodeid(cells,nodes, layer)
            neighbor_embeddings = self.memory_map.get_map(neighbor_cells)
            if self.use_neighbor_feature:
                neighbor_times = self.memory_map.get_time(neighbor_cells)
                timestamps_tensor = torch.tensor(timestamps).to(self.device).view(-1, 1)
                # neighbor_edge_deltas = timestamps_tensor - neighbor_times
                # neighbor_edge_deltas_torch = neighbor_edge_deltas.float()
                neighbor_edge_time_embeddings = self.time_encoder((timestamps_tensor - neighbor_times).float())
                neighbor_embeddings = torch.cat([neighbor_embeddings, neighbor_edge_time_embeddings], dim=2)
                source_time_embedding = self.time_encoder(
                    torch.zeros_like(timestamps_tensor).unsqueeze(dim=1).float()).squeeze(dim=1)
                cell_embedding_cat = torch.cat([cell_embedding.unsqueeze(dim=1), source_time_embedding], dim=2)
            else:
                #cell_embedding = cell_embedding.view(nodes.shape[0], 1, -1)
                cell_embedding_cat = cell_embedding.unsqueeze(dim=1)
            # if self.aggregate_nodes:
            #     cell_neighbor_nodes = self.memory_map.get_neighbor_nodes()


            neigh_output = self.calculate_attention(cell_embedding_cat,
                                                   neighbor_embeddings.view(nodes.shape[0], neigh_max_num, -1),
                                                   neighbor_bool, self.multi_head_target_neighbor)
        else:
            neigh_output = None
        # neigh_output = self.calculate_attention(cell_embedding.view(unique_nodes.shape[0], 1, -1),
        #                                        neighbor_embeddings.view(unique_nodes.shape[0], neigh_max_num, -1),
        #                                        neighbor_bool)
        # if self.use_nodes:
        #     if self.aggregate_node:
        #         # node_embs = memory[nodes]
        #         nodes_max_num = 10
        #         neighbor_nodes, _, edge_times = self.neighbor.get_temporal_neighbor_unique(nodes, timestamps, nodes_max_num)
        #         # neighbor_nodes=None
        #         # cell_nodes = self.memory_map.get_nodes_in_cells(cells, nodes_max_num, batch_i,epoch,flist,nodes2=nodes,neighbornodes=neighbor_nodes)
        #         cell_nodes, _, edge_times = self.neighbor.get_temporal_neighbor_unique(nodes, timestamps, nodes_max_num)
        #         cell_nodes = torch.tensor(cell_nodes).to(self.device)
        #         # cell_node_embs = memory[cell_nodes]
        #         nodes_emb, cells_nodes_emb, cells_nodes_bool = self.memory_map.get_connections2(nodes, cell_nodes, memory,
        #                                                     timestamps, self.time_encoder, self.num_encode_layer, nodes_max_num,
        #                                                     edge_times=edge_times, neigh=True)
        #         instance_num = cell_nodes.shape[0] // 3
        #         cell_nodes_dup = torch.cat((cell_nodes,cell_nodes[:instance_num]))
        #         cells_nodes_bool = cell_nodes_dup != -1
        #         neighbor_node_output = self.calculate_attention(nodes_emb,
        #                                                 cells_nodes_emb,
        #                                                 cells_nodes_bool, self.multi_head_target_nodes)
        # else:
        #     neighbor_node_output = None



        # updated_trajectory = self.memory_map.node_trajectory.data.clone()[unique_nodes]
        # map_emb = self.memory_map.memory_map.data.clone()[cell_ids]
        # updated_trajectory = self.trajectory_updater(map_emb,updated_trajectory)
        # self.memory_map.set_node_trajectory(unique_nodes, updated_trajectory)

        # traj_output = self.memory_map.get_node_trajectory(unique_nodes)
        # trajectory_nodes = unique_nodes

        # trajectory_nodes, trajectory_bool = self.memory_map.get_trajectory_from_nodeid(nodes, max_num)
        if self.use_traj_cell:
            trajectory_embeddings = self.memory_map.get_map(trajectory_nodes.long())
            if self.use_feature:
                timestamps_tensor = torch.tensor(timestamps).to(self.device).view(-1, 1)
                edge_deltas = timestamps_tensor - trajectory_times
                edge_deltas_torch = edge_deltas.float()
                trajectory_edge_time_embeddings = self.time_encoder(edge_deltas_torch)
                # trajectory_embeddings = self.memory_map.get_map(trajectory_nodes.view(-1).long()).view(nodes.shape[0],max_num,-1)
                trajectory_embeddings = torch.cat([trajectory_embeddings, trajectory_edge_time_embeddings], dim=2)
                # cell_embedding = cell_embedding.view(nodes.shape[0],1,-1)
                source_time_embedding = self.time_encoder(torch.zeros_like(timestamps_tensor).unsqueeze(dim=1).float()).squeeze(dim=1)
                cell_embedding_cat = torch.cat([cell_embedding.unsqueeze(dim=1), source_time_embedding], dim=2)

            else:

                # cell_embedding = cell_embedding.view(nodes.shape[0], 1, -1)
                cell_embedding_cat = cell_embedding.unsqueeze(dim=1)
            traj_output = self.calculate_attention(cell_embedding_cat, trajectory_embeddings, trajectory_bool, self.multi_head_target_traj)
        else:
            traj_output = None
        # if batch_i <= 1:
        #     embs = self.compute_embedding(memory[unique_nodes], memory[unique_nodes], memory[unique_nodes], memory[unique_nodes])
        # else:

        node_emb = memory[nodes]

        if self.use_connections:
            s_d_nums, s_d_times, s_n_nums, s_n_times = self.memory_map.get_connections(nodes)
            num_tensor = torch.tensor(np.concatenate([s_d_nums,s_n_nums])).to(self.device)
            num_embedding = self.num_encode_layer(num_tensor.unsqueeze(dim=1).float())
            time_tensor = (timestamps_tensor[0:nodes.shape[0]//3*2] - torch.tensor(np.concatenate([s_d_times,s_n_times])).to(self.device).unsqueeze(dim=1)).float()
            time_embedding = self.time_encoder(time_tensor).squeeze(dim=1)
            embs = self.compute_embedding(node_emb, cell_embedding, traj_output, neigh_output, num_embedding, time_embedding)
        elif self.aggregate_node:
            embs = self.compute_embedding(node_emb, cell_embedding, traj_output, neigh_output, neighbor_node_output=neighbor_node_output)
        else:
            embs = self.compute_embedding(node_emb, cell_embedding, traj_output, neigh_output)
        # self.memory_map.set_time(cell_ids, nodes, torch.tensor(timestamps).to(self.device).float())
        # embs = embs[inverse]
        if write:
            unknown_ind_inverse = unknown_ind# [inverse]
            # if self.use_connections:
            #     self.memory_map.analyze_center(batch_i, epoch, flist, nodes, timestamps, cell_ids, neighbor_cells,
            #                                    trajectory_nodes, unknown_ind_inverse, s_d_nums, s_d_times, s_n_nums, s_n_times)
            # elif self.use_latest_time:
            #     self.memory_map.analyze_center(batch_i, epoch, flist, nodes, timestamps, cell_ids, neighbor_cells,
            #                                    trajectory_nodes, unknown_ind_inverse,nodes_time=time_tensor_nodes)
            # else:
            self.memory_map.analyze_center(batch_i, epoch, flist, nodes, timestamps, cell_ids, neighbor_cells,
                                               trajectory_nodes, unknown_ind_inverse)
            self.memory_map.print_node_information(batch_i, epoch, flist)
        instance_num = nodes.shape[0] // 3
        nodes_unique, index = np.unique(nodes[:instance_num * 2], return_index=True)
        index = torch.tensor(index).to(self.device)
        times = torch.tensor(timestamps).to(self.device)[index]
        cells = self.memory_map.get_cells_from_trajectory(nodes_unique, times)
        self.memory_map.set_nodes_after_batch(nodes_unique, cells, times)
        # self.memory_map.nodes_batch.clear()
        return embs

    # def set_time(self, source_nodes, destination_nodes, edge_times):
    #     self.memory_map.set_time(cell_ids, nodes, torch.tensor(timestamps).to(self.device).float())

    def compute_embedding(self, node_embedding, cell_embedding, traj_embedding, neigh_embedding, num_embedding=None,
                          time_embedding=None, neighbor_node_output=None):
        if self.use_neighbor_num:
            instance_num = node_embedding.shape[0]//3
            cell_embedding = torch.cat((cell_embedding, cell_embedding[0:instance_num]))
            node_embedding = torch.cat((node_embedding, node_embedding[0:instance_num]))
            traj_embedding = torch.cat((traj_embedding, traj_embedding[0:instance_num]))
            emb_aggr = torch.cat([traj_embedding, neigh_embedding], dim=1)
            emb_self = torch.cat([node_embedding, cell_embedding], dim=1)
            embs = self.merger(emb_self, emb_aggr)
            return embs[:instance_num],embs[instance_num*3:],embs[instance_num:2*instance_num],embs[2 * instance_num:instance_num*3]
        elif self.aggregate_node:
            instance_num = node_embedding.shape[0] // 3
            cell_embedding = torch.cat((cell_embedding, cell_embedding[0:instance_num]))
            node_embedding = torch.cat((node_embedding, node_embedding[0:instance_num]))
            if self.use_traj_cell:
                traj_embedding = torch.cat((traj_embedding, traj_embedding[0:instance_num]))
            if self.use_neighbor_cell:
                neigh_embedding = torch.cat((neigh_embedding, neigh_embedding[0:instance_num]))
            #emb_aggr = torch.cat([traj_embedding, neigh_embedding, ], dim=1)
            if self.use_neighbor_cell and self.use_traj_cell:
                emb_aggr = torch.cat([traj_embedding, neigh_embedding, ], dim=1)
            elif self.use_neighbor_cell:
                emb_aggr = torch.cat([neigh_embedding, ], dim=1)
            elif self.use_traj_cell:
                emb_aggr = torch.cat([traj_embedding, ], dim=1)
            else:
                emb_aggr = None
            if self.use_cell and (self.use_nodes or self.use_batch_nodes or self.use_neighbor_batch):
                emb_self = torch.cat([node_embedding, cell_embedding, neighbor_node_output], dim=1)
            elif self.use_cell:
                emb_self = torch.cat([node_embedding, cell_embedding], dim=1)
            elif (self.use_nodes or self.use_batch_nodes or self.use_neighbor_batch):
                emb_self = torch.cat([node_embedding, neighbor_node_output], dim=1)
            else:
                emb_self = torch.cat([node_embedding,], dim=1)

            embs = self.merger(emb_self, emb_aggr)
            # embs = neighbor_node_output
            # embs = self.merger(node_embedding, neighbor_node_output)
            return embs[:instance_num], embs[instance_num * 3:], embs[instance_num:2 * instance_num], embs[
                                                                                                      2 * instance_num:instance_num * 3]


        emb_aggr = torch.cat([traj_embedding, neigh_embedding], dim=1)
        emb_self = torch.cat([node_embedding, cell_embedding], dim=1)
        embs = self.merger(emb_self, emb_aggr)
        if self.use_connections:
            con_emb = torch.cat([num_embedding, time_embedding], dim=1)
            instance_num = emb_self.shape[0] // 3
            source_emb_d = self.merger2(embs[:instance_num], con_emb[:instance_num])
            source_emb_n = self.merger2(embs[:instance_num], con_emb[instance_num:])
            destination_emb = self.merger2(embs[instance_num:2*instance_num], con_emb[:instance_num])
            negative_emb = self.merger2(embs[2 * instance_num:], con_emb[instance_num:])
            return source_emb_d,source_emb_n,destination_emb,negative_emb
        #elif self.use_neighbor_num:



        # embs = torch.cat([emb_self, emb_aggr], dim=1)# self.merger(emb_self, emb_aggr)
        return embs

    def calculate_attention2(self,  query, key,):
        ...

    def add_connections(self, source_nodes, destination_nodes, times ):
        if self.use_connections or self.use_neighbor_num or self.aggregate_node:
            # source_cells,_ = self.memory_map.get_map_from_nodeid(source_nodes)
            # destination_cells, _ = self.memory_map.get_map_from_nodeid(destination_nodes)
            # self.memory_map.add_connections(source_nodes, destination_nodes, times)
            ...

    def get_cell_unknown(self, nodes, cell_ids, memory):
        nodes = torch.tensor(nodes).to(self.device)
        # cell_ids = self.memory_map.get_cellid(nodes)
        unknown_ind = torch.eq(cell_ids, -1)
        unknown_nodes = nodes[unknown_ind]
        distances = self.distance_calculator(memory[unknown_nodes])  # (117, 484)
        min_distances, min_indices = torch.min(distances, dim=1)  # (117,)
        cell_arrange = torch.arange(self.size * self.size).to(self.device)
        unknown_cells = cell_arrange[min_indices]
        cell_ids[unknown_ind] = unknown_cells
        return cell_ids, unknown_ind

    def clear_messages(self, nodes):
        self.messages = defaultdict(list)
        self.cor_destination = defaultdict(list)


    def aggregate(self, aggregator_type="last"):
        """Only keep the last message for each node"""
        #unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        unique_destinations = []
        # aggregate_type = "unique edge"
        # aggregate_type = "last edge"

        to_update_node_ids = []
        nodes = self.messages.keys()
        if aggregator_type == "unique":
            for n in nodes:
                cor_des = self.cor_destination[n]
                unique_des, index = np.unique(cor_des, return_index=True)
                for i in range(index.shape[0]):
                    to_update_node_ids.append(n)
                    unique_messages.append(self.messages[n][index[i]][0])
                    unique_timestamps.append(self.messages[n][index[i]][1])
                    unique_destinations.append(self.cor_destination[n][index[i]])
        else:
            for n in nodes:
                to_update_node_ids.append(n)
                unique_messages.append(self.messages[n][-1][0])
                unique_timestamps.append(self.messages[n][-1][1])
                unique_destinations.append(self.cor_destination[n][-1])

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
        #unique_destinations = torch.stack(unique_destinations) if len(to_update_node_ids) > 0 else []
        # self.messages = defaultdict(list)
        # self.cor_destination = defaultdict(list)
        return to_update_node_ids, unique_messages, unique_timestamps, unique_destinations


def get_memory_map_updater(map_update_type, assign_type,memory_map, map_dimension, message_dimension, map_layer, message_aggregator,time_encoder,time_dimension, neighbor, message_function,dropout, device):
    if map_update_type == "net":
        return NetMemoryUpdater(assign_type,memory_map, map_dimension, message_dimension, map_layer, message_aggregator,
                                time_encoder, time_dimension, neighbor, dropout, device)
