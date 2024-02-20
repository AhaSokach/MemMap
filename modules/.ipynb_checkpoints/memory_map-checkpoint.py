import torch
from torch import nn
import numpy as np
import math
import random
from collections import defaultdict
from copy import deepcopy


class MemoryMap(nn.Module):

    def __init__(self, n_nodes, map_init_type, size, neighbor,
                 memory_dimension, input_dimension=None, message_dimension=None,
                             device="cpu", combination_method='sum'):
        super(MemoryMap, self).__init__()
        self.size = size   #math.ceil(np.sqrt(5 * np.sqrt(n_nodes)))
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.neighbor = neighbor
        self.device = device
        self.map_init_type = "zero"
        self.update_type = "net"
        self.learning_rate = 0.5
        self.sigma = 3.0
        #self.belongingness = {}
        #self.belongingness_map = {}
        #self.final_cell_map = {}
        #self.empty_cells = [i for i in range(0, self.size * self.size)]
        self.trajectory_updater = nn.GRUCell(input_size=memory_dimension,
                                         hidden_size=memory_dimension)


        #self.combination_method = combination_method

        self.__init_map__()

    def __init_map__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        if(self.map_init_type == "zero"):
            if(self.update_type == "som"):
                self.memory_map = torch.zeros((self.size * self.size, self.memory_dimension)).to(self.device)

                self.belongingness = {}
                self.belongingness_map = {}
                self.final_cell_map = {}
                self.empty_cells = [i for i in range(0, self.size * self.size)]
            else:
                self.memory_map = nn.Parameter(torch.zeros((self.size * self.size, self.memory_dimension)).to(self.device),
                                            requires_grad=False)
                self.map_times = nn.Parameter(
                    torch.zeros((self.size * self.size,)).to(self.device),
                    requires_grad=False)
                self.nodes_latest_time = torch.zeros((self.n_nodes,)).to(self.device)
                self.node_trajectory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                                            requires_grad=False)
                self.node_to_map = torch.tensor([-1] * self.n_nodes).to(self.device)
                self.position_encode = torch.tensor([-1] * self.n_nodes).to(self.device)
                trajectory_len = 20
                #self.node_trajectory_id = torch.zeros((self.n_nodes, trajectory_len)).to(self.device)
                self.node_trajectory_id = defaultdict(list)
                self.nodes_connections = defaultdict(dict)
                self.nodes_neighbor = defaultdict(list)
                self.nodes_neighbor_time = defaultdict(list)
                self.cell_to_node = defaultdict(list)
                # self.cell_to_node_time = defaultdict(list)
                self.empty_map = torch.ones((self.size * self.size,)).to(self.device)
                self.nodes_batch = defaultdict(list)
                self.all_indices = torch.arange(self.size * self.size).to(self.device)
                #self.belongingness = {}
                #self.belongingness_map = {}
                #self.final_cell_map = {}
                #self.empty_cells = [i for i in range(0, self.size * self.size)]
                self.new_position = 1
                # TODO other init for map



    def get_random_cells(self, num):
        empty_cell_ids = self.all_indices[self.empty_map.bool()]
        shuffled_indices = torch.randperm(empty_cell_ids.shape[0])
        cell_ids = empty_cell_ids[shuffled_indices[:num]]
        if (cell_ids.shape[0] < num):
            remaining_num = num - cell_ids.shape[0]
            if(empty_cell_ids.shape[0] < remaining_num):
                max_num = self.size * self.size-1
                random_cells = torch.randint(0, max_num, (remaining_num,)).to(self.device)
                cells = self.all_indices[random_cells]
            else:
                max_num = empty_cell_ids.shape[0]-1
                random_cells = torch.randint(0, max_num,(remaining_num,)).to(self.device)
                cells = empty_cell_ids[random_cells]
            cell_ids = torch.concat((cell_ids, cells))
        return cell_ids

    # def get_random_cells_embeddings(self, num, node_embeddings):
    #     empty_cell_ids = self.all_indices[self.empty_map.bool()]
    #     shuffled_indices = torch.randperm(empty_cell_ids.shape[0])
    #     cell_ids = empty_cell_ids[shuffled_indices[:num]]
    #     if (cell_ids.shape[0] < num):
    #         remaining_num = num - cell_ids.shape[0]
    #         if(empty_cell_ids.shape[0] < remaining_num):
    #             max_num = self.size * self.size-1
    #             random_cells = torch.randint(0, max_num, (remaining_num,)).to(self.device)
    #             cells = self.all_indices[random_cells]
    #         else:
    #             max_num = empty_cell_ids.shape[0]-1
    #             random_cells = torch.randint(0, max_num,(remaining_num,)).to(self.device)
    #             cells = empty_cell_ids[random_cells]
    #         cell_ids = torch.concat((cell_ids, cells))
    #     return cell_ids

    def add_nodes_batch(self,source_nodes, destination_nodes,times):
        instance_num = source_nodes.shape[0]
        for i in range(instance_num):
            # self.nodes_batch[source_nodes[i]] = np.vstack((self.nodes_batch[source_nodes[i]],np.array([destination_nodes[i],times[i]])))
            # self.nodes_batch[destination_nodes[i]] = np.vstack((self.nodes_batch[destination_nodes[i]],np.array([source_nodes[i],times[i]])))

            source_node_batch = self.nodes_batch[source_nodes[i]]
            destination_node_batch = self.nodes_batch[destination_nodes[i]]
            if len(source_node_batch) == 0:
                source_node_batch.append([destination_nodes[i]])
                source_node_batch.append([times[i]])
            else:
                source_node_batch[0].append(destination_nodes[i])
                source_node_batch[1].append(times[i])
            if len(destination_node_batch) == 0:
                destination_node_batch.append([source_nodes[i]])
                destination_node_batch.append([times[i]])
            else:
                destination_node_batch[0].append(source_nodes[i])
                destination_node_batch[1].append(times[i])
            # self.nodes_batch[source_nodes[i]].append((destination_nodes[i],times[i]))
            # self.nodes_batch[destination_nodes[i]].append((source_nodes[i], times[i]))

    def get_nodes_batch(self,nodes,times,nodes_max_num):
        instance_num = nodes.shape[0] // 3
        nodes_batch = torch.full((nodes.shape[0], nodes_max_num), 0).to(self.device)
        nodes_batch_time = np.zeros((nodes.shape[0], nodes_max_num)).astype(np.float32)
        nodes_bool2 = torch.full((nodes.shape[0], nodes_max_num), 0).to(self.device)

        nodes_tensor = torch.tensor(nodes).to(self.device)
        nodes_bool = nodes_tensor.view(-1,1) == nodes_tensor[:2*instance_num].view(1,-1)
        times_tensor = torch.tensor(times).to(self.device)
        times_tensor_positive = torch.tensor(times[:2*instance_num]).to(self.device)
        times_repeat_tensor = times_tensor_positive.repeat(nodes.shape[0],1)
        cor_nodes = torch.cat((nodes_tensor[instance_num:2*instance_num],nodes_tensor[:instance_num]))
        nodes_repeat_tensor = cor_nodes.repeat(nodes.shape[0],1)
        time_bool = times_repeat_tensor < (times_tensor.view(-1,1))
        index_bool = nodes_bool * time_bool
        all_batch_nodes = nodes_repeat_tensor[index_bool]
        range_line = torch.arange(0,3*instance_num).to(self.device)
        line_repeat_tensor = range_line.view(-1,1).repeat(1, index_bool.shape[1])
        all_batch_lines = line_repeat_tensor[index_bool]
        all_batch_times = times_repeat_tensor[index_bool]
        batch_nodes_sum = torch.sum(index_bool,dim=1).to(self.device)

        ind = 0
        for i in range(instance_num*3):
            if ind >= all_batch_lines.shape[0]:
                break
            li = all_batch_lines[ind]

            num = batch_nodes_sum[i]
            size = min(num,nodes_max_num)
            # print(f'num {num}')
            if i==li:
                # print(f'ind num {ind+num}')
                # if (ind+num)==2964:
                #     print("aaa")

                nodes_batch[li,:size] = all_batch_nodes[ind+num-size:ind+num]
                nodes_batch_time[li][:size] = all_batch_times[ind+num-size:ind+num].cpu().numpy()
                nodes_bool2[li,:size] = 1
                ind+=num


        #
        # max_sum = torch.sum(nodes_bool,dim=0)
        # nodes_batch = torch.full((nodes.shape[0], max_sum), 0).to(self.device)
        # true_neighbors =

        # times = np.concatenate([times,times,times])
        # for i in range(nodes.shape[0]):
        #     nodes_list = self.nodes_batch[nodes[i]]
        #     # print(nodes[i])
        #     if len(nodes_list)==0:
        #         nodes_list_node = np.array([])
        #         nodes_list_time = np.array([])
        #     else:
        #         nodes_list_node = np.array(nodes_list[0])
        #         nodes_list_time = np.array(nodes_list[1])
        #
        #     true_time = nodes_list_time < times[i]
        #     true_num = min(np.sum(true_time),nodes_max_num)
        #     nodes_batch[i][:true_num] = torch.tensor(nodes_list_node[true_time][-nodes_max_num:]).to(self.device)
        #     nodes_batch_time[i][:true_num] = nodes_list_time[true_time][-nodes_max_num:]
        #     nodes_bool2[i][:true_num] = 1
            # self.nodes_batch[source_nodes[i]].append([destination_nodes[i],times[i]])
            # self.nodes_batch[destination_nodes[i]].append([source_nodes[i], times[i]])
        # self.nodes_batch.clear()
        return nodes_batch, nodes_batch_time, nodes_bool2

    def set_nodes_distance(self, nodes, cells, times):
        if nodes.shape[0] == 0:
            return
        node_id_item = nodes.cpu().detach().numpy()
        cell_id_item = cells.cpu().detach().numpy()
        times = times.cpu().numpy()
        for i in range(node_id_item.shape[0]):
            self.node_trajectory_id[node_id_item[i]].append((times[i], cell_id_item[i]))
            # self.cell_to_node[cell_id_item[i]].append(node_id_item[i])
        # self.node_trajectory_id =
        # self.node_to_map[nodes] = cells.clone()
        # self.empty_map[cells] = False

    def set_connections(self, nodes, cells, cor_cells):
        cells = cells.cpu().numpy()
        cor_cells = cor_cells.cpu().numpy()
        for i in range(len(cells)):
            cell = cells[i]
            cor_cell = cor_cells[i]
            cell_con = self.cell_connections[cell]
            cell_con_cor = cell_con[cor_cell]
            if cell_con_cor == None:
                ...






    def set_nodes_after_batch(self, nodes, cells, times):
        if nodes.shape[0] == 0:
            return
        node_id_item = nodes# .cpu().detach().numpy()
        cell_id_item = cells.cpu().detach().numpy()
        # old_cells = self.node_to_map[nodes].cpu().numpy()
        for i in range(node_id_item.shape[0]):
            old_cell = self.node_to_map[nodes[i]].item()# data().cpu().numpy()[0]
            if(cell_id_item[i]==-1):
                continue
            self.map_times[cells[i]]=times[i]
            if(old_cell == -1):
                self.cell_to_node[cell_id_item[i]].append(node_id_item[i])
            else:
                self.cell_to_node[cell_id_item[i]].append(node_id_item[i])
                self.cell_to_node[old_cell].remove(node_id_item[i])
                if (len(self.cell_to_node[old_cell]) == 0):
                    self.empty_map[old_cell] = True
        self.node_to_map[nodes] = cells

    def set_new_nodes(self, nodes, cells, times, positions=None):
        # print(nodes)
        if nodes.shape[0] == 0:
            return
        node_id_item = nodes.cpu().detach().numpy()
        cell_id_item = cells.cpu().detach().numpy()
        times = times.cpu().numpy()
        for i in range(node_id_item.shape[0]):
            self.node_trajectory_id[node_id_item[i]].append((times[i], cell_id_item[i]))
            # self.cell_to_node[cell_id_item[i]].append(node_id_item[i])
        # self.node_trajectory_id =
        # self.node_to_map[nodes] = cells.clone()
        self.empty_map[cells] = False
        if(positions != None):
            self.set_position(nodes, positions)
        else:
            self.set_new_position(nodes)

    def set_nodes(self, nodes, cells, positions, times, use_connections=False):
        # definitly changes
        if nodes.shape[0] == 0:
            return
        # old_cells = self.node_to_map[nodes].cpu().numpy()
        node_id_item = nodes.cpu().numpy()
        cell_id_item = cells.cpu().numpy()
        times = times.cpu().numpy()
        for i in range(node_id_item.shape[0]):
            self.node_trajectory_id[node_id_item[i]].append((times[i], cell_id_item[i]))
            # self.cell_to_node[cell_id_item[i]].append(node_id_item[i])
            # self.cell_to_node[old_cells[i]].remove(node_id_item[i])
            # if (len(self.cell_to_node[old_cells[i]]) == 0):
            #     self.empty_map[old_cells[i]] = True
            # if use_connections:
            #     num_list = self.cell_connections[old_cells[i]]
            #     self.cell_connections[cell_id_item]=num_list
            #     self.cell_connections[old_cells[i]]=dict()

        # self.node_to_map[nodes] = cells
        self.empty_map[cells] = False
        self.set_position(nodes, positions)



    def set_time(self, cell_ids, nodes, times):
        # instance_num = cell_ids.shape[0]//3
        # cell_ids = cell_ids[0:instance_num*2]
        # nodes = nodes[0:instance_num*2]
        # times = times[0:instance_num*2]
        self.map_times[cell_ids] = times
        self.nodes_latest_time[nodes] = times

    def get_time(self, cell_ids):
        return self.map_times[cell_ids]

    def get_latest_time(self, nodes):
        return self.nodes_latest_time[nodes]

    def get_position(self, node_id):
        return self.position_encode[node_id]

    def set_position(self, node_id, position_id):
        self.position_encode[node_id] = position_id.clone()

    def set_new_position(self, node_id, ):
        num = node_id.shape[0]
        positions = torch.arange(self.new_position, self.new_position+num).to(self.device)
        self.position_encode[node_id] = positions
        self.new_position += num



    def get_map_from_embeddings(self, embeddings):
        centers, _ = self.find_center(embeddings)
        # memory_map_reshape = torch.reshape(self.memory_map, (-1, self.memory_dimension))
        map_nodes = self.memory_map[centers]
        return map_nodes

    def set_center_cell(self, cell_ids, center_cell_update):
        self.memory_map[cell_ids] = center_cell_update

    # def set_node_to_map(self, node_id, cell_id, node_emb):
    #     # for just one node
    #     node_id_item = node_id.item()
    #     cell_id_item = cell_id.item()
    #
    #     old_cell = self.node_to_map[node_id_item].item()
    #     if ((old_cell != -1) and (old_cell != cell_id_item)):
    #         self.cell_to_node[old_cell].remove(node_id_item)
    #         if(len(self.cell_to_node[old_cell]) == 0):
    #             self.empty_map[old_cell] = True
    #     if (old_cell == -1):
    #         self.set_map(cell_id_item,node_emb) #[cell_id] = node_emb
    #     if (old_cell != cell_id_item):
    #         #self.node_to_map[node_id] = cell_id
    #         self.node_trajectory_id[node_id_item].append(cell_id_item)
    #         self.cell_to_node[cell_id_item].append(node_id_item)
    #         # updated_trajectory = self.trajectory_updater(self.memory_map[cell_id].detach().clone(), self.node_trajectory[node_id])
    #         # self.set_node_trajectory(node_id, updated_trajectory)
    #     self.node_to_map[node_id] = cell_id
    #     self.empty_map[cell_id] = False

    # def get_connections(self, nodes):
    #     instance_num = nodes.shape[0] // 3
    #     source_nodes = nodes[0:instance_num]
    #     s_d_nums = []
    #     s_d_times = []
    #     s_n_nums = []
    #     s_n_times = []
    #     for i in range(instance_num):
    #         s = nodes[i]
    #         d = nodes[i + instance_num]
    #         n = nodes[i + 2 * instance_num]
    #         s_list = self.nodes_connections[s]
    #         if d not in s_list:
    #             s_d_nums.append(0)
    #             s_d_times.append(0)
    #         else:
    #             num, t=s_list[d]
    #             s_d_nums.append(num)
    #             s_d_times.append(t)
    #
    #         if n not in s_list:
    #             s_n_nums.append(0)
    #             s_n_times.append(0)
    #         else:
    #             num, t = s_list[n]
    #             s_n_nums.append(num)
    #             s_n_times.append(t)
    #     return s_d_nums, s_d_times, s_n_nums, s_n_times

    # def get_neighbor_cell_connections(self, nodes, cell_ids, cells_neighbor):
    #     instance_num = nodes.shape[0] // 3
    #     neighbor_num = cells_neighbor.shape[1]
    #     source_cell_source_node_num = torch.zeros()
    #     source_cell_destination_node_num = []
    #     source_cell_negative_node_num = []
    #     destination_cell_source_node_num = []
    #     destination_cell_destination_node_num = []
    #     destination_cell_negative_node_num = []
    #     negative_cell_source_node_num = []
    #     negative_cell_destination_node_num = []
    #     negative_cell_negative_node_num = []
    #     for i in range(instance_num):
    #         source_node = nodes[i]
    #         destination_node = nodes[i+instance_num]
    #         negative_node = nodes[i+2*instance_num]
    #         source_neighbor_cells = cells_neighbor[i].cpu().numpy()
    #         destination_neighbor_cells = cells_neighbor[i + instance_num].cpu().numpy()
    #         negative_neighbor_cells = cells_neighbor[i + 2 * instance_num].cpu().numpy()
    #         # source_cell = cell_ids[i].cpu().numpy()
    #         # destination_cell = cell_ids[i + instance_num].cpu().numpy()
    #         # negative_cell = cell_ids[i + 2 * instance_num].cpu().numpy()
    #         for j in range(neighbor_num):
    #             source_neighbor_nodes = self.cell_to_node[source_neighbor_cells[j]]
    #             destination_neighbor_nodes = self.cell_to_node[destination_neighbor_cells[j]]
    #             negative_neighbor_nodes = self.cell_to_node[negative_neighbor_cells[j]]
    #             num_s = 0
    #             num_d = 0
    #             num_n = 0
    #             source_node_connections = self.nodes_connections[source_node]
    #             destination_node_connections = self.nodes_connections[destination_node]
    #             negative_node_connections = self.nodes_connections[negative_node]
    #             for z in range(len(source_neighbor_nodes)):
    #                 if source_neighbor_nodes[z] in source_node_connections:
    #                     num_s += source_node_connections[source_neighbor_nodes[z]][0]
    #                 if source_neighbor_nodes[z] in destination_node_connections:
    #                     num_d += destination_node_connections[source_neighbor_nodes[z]][0]
    #                 if source_neighbor_nodes[z] in negative_node_connections:
    #                     num_n += negative_node_connections[source_neighbor_nodes[z]][0]
    #             source_cell_source_node_num.append(num_s)
    #             source_cell_destination_node_num.append(num_d)
    #             source_cell_negative_node_num.append(num_n)
    #             num_s = 0
    #             num_d = 0
    #             num_n = 0
    #             for z in range(len(destination_neighbor_nodes)):
    #                 if destination_neighbor_nodes[z] in source_node_connections:
    #                     num_s += source_node_connections[destination_neighbor_nodes[z]][0]
    #                 if destination_neighbor_nodes[z] in destination_node_connections:
    #                     num_d += destination_node_connections[destination_neighbor_nodes[z]][0]
    #                 # num_n += negative_node_connections[destination_neighbor_nodes[j]]
    #             destination_cell_source_node_num.append(num_s)
    #             destination_cell_destination_node_num.append(num_d)
    #             # destination_cell_negative_node_num.append(num_n)
    #
    #             num_s = 0
    #             num_d = 0
    #             num_n = 0
    #             for z in range(len(negative_neighbor_nodes)):
    #                 if negative_neighbor_nodes[z] in source_node_connections:
    #                     num_s += source_node_connections[negative_neighbor_nodes[z]][0]
    #                 # num_d += destination_node_connections[negative_neighbor_nodes[j]]
    #                 if negative_neighbor_nodes[z] in negative_node_connections:
    #                     num_n += negative_node_connections[negative_neighbor_nodes[z]][0]
    #             negative_cell_source_node_num.append(num_s)
    #                 # negative_cell_destination_node_num.append(num_d)
    #             negative_cell_negative_node_num.append(num_n)
    #
    #     source_cell_source_node_num = torch.tensor(source_cell_source_node_num).to(self.device).view(-1,neighbor_num)
    #     source_cell_destination_node_num = torch.tensor(source_cell_destination_node_num).to(self.device).view(-1,neighbor_num)
    #     source_cell_negative_node_num = torch.tensor(source_cell_negative_node_num).to(self.device).view(-1,neighbor_num)
    #     destination_cell_source_node_num = torch.tensor(destination_cell_source_node_num).to(self.device).view(-1,neighbor_num)
    #     destination_cell_destination_node_num = torch.tensor(destination_cell_destination_node_num).to(self.device).view(-1,neighbor_num)
    #     #destination_cell_negative_node_num = torch.tensor(destination_cell_negative_node_num).to(self.device).view(-1,neighbor_num)
    #     negative_cell_source_node_num = torch.tensor(negative_cell_source_node_num).to(self.device).view(-1,neighbor_num)
    #     #negative_cell_destination_node_num = torch.tensor(negative_cell_destination_node_num).to(self.device).view(-1,neighbor_num)
    #     negative_cell_negative_node_num = torch.tensor(negative_cell_negative_node_num).to(self.device).view(-1,neighbor_num)
    #     source_destination_num = torch.stack((source_cell_source_node_num,source_cell_destination_node_num),dim=2)
    #     destination_source_num = torch.stack((destination_cell_destination_node_num,destination_cell_source_node_num),dim=2)
    #     source_negative_num = torch.stack((source_cell_source_node_num,source_cell_negative_node_num),dim=2)
    #     negative_source_num = torch.stack((negative_cell_negative_node_num,negative_cell_source_node_num),dim=2)
    #     return source_destination_num, destination_source_num, source_negative_num, negative_source_num


    def get_nodes_in_cells(self, cells, max_num, batch_i, epoch, flist,  nodes2=None,neighbornodes=None,write=False):
        # max_num = 25
        neighbor_cells = self.get_neighbor_cells(cells, offset=1)
        nodes = torch.full((cells.shape[0], max_num), -1).to(self.device)
        cells_item = cells.cpu().numpy()
        if write:
            neighbor_file = flist[5]
            print(f'epoch:{epoch} batch:{batch_i}', file=neighbor_file)

        for i in range(cells_item.shape[0]):
            neic = neighbor_cells[i].cpu().numpy()
            n = self.cell_to_node[cells_item[i]][-max_num:]
            nodes[i,max_num-len(n):] = torch.tensor(n).to(self.device)
            # left = max_num-len(n)
            # j=0
            # while(left>0):
            #     if j>8:
            #         break
            #     if neic[j]==cells_item[i]:
            #         j += 1
            #         continue
            #
            #     n_2=self.cell_to_node[neic[j]][-left:]
            #     nodes[i,left-len(n_2):left] = torch.tensor(n_2).to(self.device)
            #     left = left-len(n_2)
            #     j += 1
            if write:
                print(f'i: {nodes2[i]} {cells_item[i]}',file=neighbor_file)
                print(f'cell node:{n}',file=neighbor_file)
                print(f'neighbor node:{neighbornodes[i]}',file=neighbor_file)
        return nodes

    def get_connections2(self, nodes, cell_nodes, memory, times, time_encoder, num_encoder, max_num=25,
                         edge_times=None, neigh=False, calculate=False):
        instance_num = nodes.shape[0] // 3
        neighbor_num = cell_nodes.shape[1]
        source_cell_source_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_node_times = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_node_bool = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_destination_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_negative_node_num = torch.zeros((instance_num, max_num)).to(self.device)

        destination_cell_source_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        destination_cell_node_times = torch.zeros((instance_num, max_num)).to(self.device)
        destination_cell_node_bool = torch.zeros((instance_num, max_num)).to(self.device)
        destination_cell_destination_node_num = torch.zeros((instance_num, max_num)).to(self.device)

        negative_cell_source_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        negative_cell_node_times = torch.zeros((instance_num, max_num)).to(self.device)
        negative_cell_node_bool = torch.zeros((instance_num, max_num)).to(self.device)
        negative_cell_negative_node_num = torch.zeros((instance_num, max_num)).to(self.device)

        if calculate:
            for i in range(instance_num):
                source_node = nodes[i]
                destination_node = nodes[i+instance_num]
                negative_node = nodes[i+2*instance_num]
                source_cell_nodes = cell_nodes[i].cpu().numpy()
                destination_cell_nodes = cell_nodes[i + instance_num].cpu().numpy()
                negative_cell_nodes = cell_nodes[i + 2 * instance_num].cpu().numpy()

                source_node_connections = self.nodes_connections[source_node]
                destination_node_connections = self.nodes_connections[destination_node]
                negative_node_connections = self.nodes_connections[negative_node]

                for z in range(max_num):
                    compare_node = source_cell_nodes[z]
                    if compare_node in source_node_connections:
                        source_cell_source_node_num[i,z] = source_node_connections[compare_node][0]
                        source_cell_node_times[i, z] = source_node_connections[compare_node][1]
                        source_cell_node_bool[i, z] = 1
                    if compare_node in destination_node_connections:
                        source_cell_destination_node_num[i,z] = destination_node_connections[compare_node][0]
                    if compare_node in negative_node_connections:
                        source_cell_negative_node_num[i, z] = negative_node_connections[compare_node][0]

                for z in range(max_num):
                    compare_node = destination_cell_nodes[z]
                    if compare_node in destination_node_connections:
                        destination_cell_destination_node_num[i, z] = destination_node_connections[compare_node][0]
                        destination_cell_node_times[i, z] = destination_node_connections[compare_node][1]
                        destination_cell_node_bool[i, z] = 1
                    if compare_node in source_node_connections:
                        destination_cell_source_node_num[i, z] = source_node_connections[compare_node][0]

                for z in range(max_num):
                    compare_node = negative_cell_nodes[z]
                    if compare_node in negative_node_connections:
                        negative_cell_negative_node_num[i, z] = negative_node_connections[compare_node][0]
                        negative_cell_node_times[i, z] = negative_node_connections[compare_node][1]
                        negative_cell_node_bool[i, z] = 1
                    if compare_node in source_node_connections:
                        negative_cell_source_node_num[i, z] = source_node_connections[compare_node][0]

        cell_nodes_memory = memory[cell_nodes.long()]

        if neigh:

            time_tensor = torch.tensor(np.concatenate([edge_times,edge_times[0:instance_num]])).to(self.device)#.unsqueeze(dim=1)
        else:
            time_tensor = torch.cat((source_cell_node_times, destination_cell_node_times, negative_cell_node_times, source_cell_node_times))
        edge_times = torch.tensor(np.concatenate([times,times[0:instance_num]])).to(self.device).unsqueeze(dim=1)
        time_emb = time_encoder((edge_times - time_tensor).unsqueeze(dim=-1).float()).squeeze(dim=2)



        source_destination_num = torch.stack((source_cell_source_node_num, source_cell_destination_node_num), dim=2)
        destination_source_num = torch.stack((destination_cell_source_node_num, destination_cell_destination_node_num),
                                             dim=2)
        source_negative_num = torch.stack((source_cell_source_node_num, source_cell_negative_node_num), dim=2)
        negative_source_num = torch.stack((negative_cell_source_node_num, negative_cell_negative_node_num), dim=2)

        num_tensor = torch.cat((source_destination_num,destination_source_num,negative_source_num,source_negative_num))
        num_emb = num_tensor #num_encoder(num_tensor.unsqueeze(dim=-1)).sum(dim=2)
        node_num_tensor = num_tensor.sum(dim=1)

        cell_nodes_memory = torch.cat((cell_nodes_memory,cell_nodes_memory[0:instance_num]))
        # cells_nodes_emb = torch.cat((cell_nodes_memory, time_emb, num_emb), dim=2)
        cells_nodes_emb = torch.cat((cell_nodes_memory, time_emb), dim=2)

        #cells_nodes_emb = torch.cat((cell_nodes_memory, time_emb), dim=2)
        cells_nodes_bool = torch.cat((source_cell_node_bool, destination_cell_node_bool, negative_cell_node_bool, source_cell_node_bool))

        nodes = np.concatenate((nodes,nodes[0:instance_num]))
        nodes_memory = memory[nodes]
        nodes_time_emb = time_encoder(
                torch.zeros_like(edge_times).unsqueeze(dim=1).float()).squeeze(dim=1)
        # nodes_emb = torch.cat((nodes_memory.unsqueeze(dim=1), nodes_time_emb),dim=2)
        nodes_emb = torch.cat((nodes_memory.unsqueeze(dim=1),nodes_time_emb),dim=2)


        return nodes_emb, cells_nodes_emb, cells_nodes_bool

    def get_connections3(self, nodes, cell_nodes, memory, times, time_encoder, num_encoder, max_num=25,
                         edge_times=None, neigh=False, calculate=False):
        instance_num = nodes.shape[0] // 3
        neighbor_num = cell_nodes.shape[1]
        source_cell_source_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_node_times = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_node_bool = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_destination_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        source_cell_negative_node_num = torch.zeros((instance_num, max_num)).to(self.device)

        destination_cell_source_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        destination_cell_node_times = torch.zeros((instance_num, max_num)).to(self.device)
        destination_cell_node_bool = torch.zeros((instance_num, max_num)).to(self.device)
        destination_cell_destination_node_num = torch.zeros((instance_num, max_num)).to(self.device)

        negative_cell_source_node_num = torch.zeros((instance_num, max_num)).to(self.device)
        negative_cell_node_times = torch.zeros((instance_num, max_num)).to(self.device)
        negative_cell_node_bool = torch.zeros((instance_num, max_num)).to(self.device)
        negative_cell_negative_node_num = torch.zeros((instance_num, max_num)).to(self.device)

        source_destination_node_num = torch.zeros((instance_num,)).to(self.device)
        source_negative_node_num = torch.zeros((instance_num,)).to(self.device)

        if calculate:
            for i in range(instance_num):
                source_node = nodes[i]
                destination_node = nodes[i + instance_num]
                negative_node = nodes[i + 2 * instance_num]
                source_cell_nodes = cell_nodes[i].cpu().numpy()
                destination_cell_nodes = cell_nodes[i + instance_num].cpu().numpy()
                negative_cell_nodes = cell_nodes[i + 2 * instance_num].cpu().numpy()

                source_node_connections = self.nodes_connections[source_node]
                destination_node_connections = self.nodes_connections[destination_node]
                negative_node_connections = self.nodes_connections[negative_node]

                if destination_node in source_node_connections:
                    source_destination_node_num[i] = source_node_connections[destination_node][0]
                if negative_node in source_node_connections:
                    source_negative_node_num[i] = source_node_connections[negative_node][0]

                for z in range(max_num):
                    compare_node = source_cell_nodes[z]
                    if compare_node in source_node_connections:
                        source_cell_source_node_num[i, z] = source_node_connections[compare_node][0]
                        source_cell_node_times[i, z] = source_node_connections[compare_node][1]
                        source_cell_node_bool[i, z] = 1
                    if compare_node in destination_node_connections:
                        source_cell_destination_node_num[i, z] = destination_node_connections[compare_node][0]
                    if compare_node in negative_node_connections:
                        source_cell_negative_node_num[i, z] = negative_node_connections[compare_node][0]

                for z in range(max_num):
                    compare_node = destination_cell_nodes[z]
                    if compare_node in destination_node_connections:
                        destination_cell_destination_node_num[i, z] = destination_node_connections[compare_node][0]
                        destination_cell_node_times[i, z] = destination_node_connections[compare_node][1]
                        destination_cell_node_bool[i, z] = 1
                    if compare_node in source_node_connections:
                        destination_cell_source_node_num[i, z] = source_node_connections[compare_node][0]

                for z in range(max_num):
                    compare_node = negative_cell_nodes[z]
                    if compare_node in negative_node_connections:
                        negative_cell_negative_node_num[i, z] = negative_node_connections[compare_node][0]
                        negative_cell_node_times[i, z] = negative_node_connections[compare_node][1]
                        negative_cell_node_bool[i, z] = 1
                    if compare_node in source_node_connections:
                        negative_cell_source_node_num[i, z] = source_node_connections[compare_node][0]

        cell_nodes_memory = memory[cell_nodes.long()]

        if neigh:

            time_tensor = torch.tensor(np.concatenate([edge_times, edge_times[0:instance_num]])).to(
                self.device)  # .unsqueeze(dim=1)
        else:
            time_tensor = torch.cat(
                (source_cell_node_times, destination_cell_node_times, negative_cell_node_times, source_cell_node_times))
        edge_times = torch.tensor(np.concatenate([times, times[0:instance_num]])).to(self.device).unsqueeze(dim=1)
        time_emb = time_encoder((edge_times - time_tensor).unsqueeze(dim=-1).float()).squeeze(dim=2)

        source_destination_num = torch.stack((source_cell_source_node_num,source_cell_source_node_num),dim=2)
        destination_source_num = torch.stack((destination_cell_destination_node_num,destination_cell_destination_node_num),dim=2)
        source_negative_num = torch.stack((source_cell_source_node_num,source_cell_source_node_num),dim=2)
        negative_source_num = torch.stack((negative_cell_negative_node_num,negative_cell_negative_node_num),dim=2)

        num_tensor = torch.cat(
            (source_destination_num, destination_source_num, negative_source_num, source_negative_num))
        num_emb = num_tensor# .unsqueeze(2)  # num_encoder(num_tensor.unsqueeze(dim=-1)).sum(dim=2)

        source_destination_node_num = torch.stack((source_destination_node_num, source_destination_node_num), dim=1)
        source_negative_node_num = torch.stack(
            (source_negative_node_num, source_negative_node_num), dim=1)
        node_num_tensor = torch.cat((source_destination_node_num,source_destination_node_num,
                                     source_negative_node_num,source_negative_node_num))
        node_num_emb = node_num_tensor.unsqueeze(1)

        cell_nodes_memory = torch.cat((cell_nodes_memory, cell_nodes_memory[0:instance_num]))
        cells_nodes_emb = torch.cat((cell_nodes_memory, time_emb, num_emb), dim=2)
        # cells_nodes_emb = torch.cat((cell_nodes_memory, time_emb), dim=2)

        # cells_nodes_emb = torch.cat((cell_nodes_memory, time_emb), dim=2)
        cells_nodes_bool = torch.cat(
            (source_cell_node_bool, destination_cell_node_bool, negative_cell_node_bool, source_cell_node_bool))

        nodes = np.concatenate((nodes, nodes[0:instance_num]))
        nodes_memory = memory[nodes]
        nodes_time_emb = time_encoder(
            torch.zeros_like(edge_times).unsqueeze(dim=1).float()).squeeze(dim=1)
        # nodes_emb = torch.cat((nodes_memory.unsqueeze(dim=1), nodes_time_emb),dim=2)
        nodes_emb = torch.cat((nodes_memory.unsqueeze(dim=1), nodes_time_emb, node_num_emb), dim=2)

        return nodes_emb, cells_nodes_emb, cells_nodes_bool

    def get_neighbor_batch(self, nodes, max_num):
        unique_nodes, index, indices = np.unique(nodes,return_index=True,return_inverse=True)

        neighbors = torch.full((unique_nodes.shape[0], max_num), -1).to(self.device)
        times = np.zeros((nodes.shape[0], max_num)).astype(np.float32)
        neighbors_bool = torch.full((unique_nodes.shape[0], max_num), 0).to(self.device)
        # unique_nodes, index, indices = np.unique(nodes,return_index=True,return_inverse=True)
        for i in range(unique_nodes.shape[0]):
            s = unique_nodes[i]
            nei_s = self.nodes_neighbor[s][-max_num:]
            nei_t = self.nodes_neighbor_time[s][-max_num:]
            neighbors[i,max_num-len(nei_s):] = torch.tensor(nei_s).to(self.device)
            times[i][max_num - len(nei_s):] = nei_t
            neighbors_bool[i, max_num - len(nei_s):] = 1

        indices_torch = torch.tensor(indices).to(self.device)
        neighbors1 = neighbors[indices_torch]
        times1 = times[indices]
        neighbors_bool1 = neighbors_bool[indices_torch]
        return neighbors1,times1,neighbors_bool1




    def add_connections(self, source_nodes, destination_nodes, times):
        instance_nums = source_nodes.shape[0]
        for i in range(instance_nums):
            s = source_nodes[i]
            d = destination_nodes[i]
            t = times[i]
            self.nodes_neighbor[s].append(d)
            self.nodes_neighbor_time[s].append(t)
            self.nodes_neighbor[d].append(s)
            self.nodes_neighbor_time[d].append(t)
            s_con = self.nodes_connections[s]
            if d not in s_con:
                self.nodes_connections[s][d] = [1,t]
            else:
                s_con[d][0] += 1
                s_con[d][1] = t

            d_con = self.nodes_connections[d]
            if s not in d_con:
                self.nodes_connections[d][s] = [1,t]
            else:
                d_con[s][0] += 1
                d_con[s][1] = t
            # s = source_nodes[i]
            # d = destination_nodes[i]
            # s_c = source_cells[i]
            # d_c = destination_cells[i]
            # s_con_d = self.cell_connections[s_c][d]
            # if s_con_d == None:
            #     self.cell_connections[s_c][d] = 1
            # else:
            #     self.cell_connections[s_c][d] += 1
            # d_con_s = self.cell_connections[d_c][s]
            # if d_con_s == None:
            #     self.cell_connections[d_c][s] = 1
            # else:
            #     self.cell_connections[d_c][s] += 1






    def set_node_trajectory(self, node_ids, trajectory_embedding):
        self.node_trajectory[node_ids] = trajectory_embedding

    def get_node_trajectory(self, node_ids):
        trajectory_embedding = self.node_trajectory[node_ids]
        return trajectory_embedding

    # def set_node_to_cell(self, nodes, cell_ids):
    #     ...
        #self.empty_map[cell_ids] = False
        #old_cells = self.node_to_map[nodes]
        # self.node_to_map[nodes] = cell_ids
        # for i in range(len(nodes)):
        #     self.node_trajectory_id[nodes[i]].append(cell_ids[i])
        #     self.cell_to_node[cell_ids[i]].append([nodes[i]])
            # if (old_cells[i] != -1):
            #     self.cell_to_node[old_cells[i]].remove([nodes[i]])


    def get_map_from_bool(self, map_bool):
        '''need bool for the whole map'''
        list = torch.arange(self.size * self.size).to(self.device)
        #map_bool = map_bool.bool()
        map_bool = torch.gt(map_bool, 0.0)
        return list[map_bool]

    def get_trajectory_from_nodeid(self, nodes, times, max_num=10):
        # nodes = torch.tensor(nodes).to(self.device)
        cell_ids = torch.full((len(nodes), ), -1).to(self.device)
        trajectory_nodes = torch.zeros((len(nodes), max_num)).to(self.device)
        trajectory_times = torch.zeros((len(nodes), max_num)).to(self.device)
        trajectory_bool = torch.zeros((len(nodes), max_num)).to(self.device)
        for i in range(len(nodes)):
            n = nodes[i]
            time_now = times[i]
            traj_list = self.node_trajectory_id[n]
            traj = []
            for j in range(len(traj_list)-1, -1, -1):
                t, c = traj_list[j]
                if (t<time_now):
                    ind = j
                    traj = traj_list[:ind+1]
                    break
            if(len(traj)!=0):
                cell_ids[i] = traj[-1][1]
                traj = np.array(traj)
                traj_cells = traj[:,1]
                traj_times = traj[:,0]
                # traj_cells = np.flip(traj_cells)
                # traj_times = np.flip(traj_times)
                # traj_cell_unique, index = np.unique(traj_cells, return_index=True)
                # traj_time_unique = traj_times[index]
                # traj_cell_max = traj_cell_unique[:max_num]
                # traj_time_max = traj_time_unique[:max_num]
                traj_cell_max = traj_cells[-max_num:]
                traj_time_max = traj_times[-max_num:]
                trajectory_nodes[i, max_num-len(traj_cell_max):] = torch.tensor(traj_cell_max).to(self.device)
                trajectory_times[i, max_num-len(traj_cell_max):] = torch.tensor(traj_time_max).to(self.device)
                trajectory_bool[i, max_num - len(traj_cell_max):] = 1
        return trajectory_nodes, trajectory_bool, trajectory_times, cell_ids

    def get_cells_from_trajectory(self, nodes, times):
        cell_ids = torch.full((nodes.shape[0],), -1).to(self.device)
        for i in range(len(nodes)):
            n = nodes[i]
            time_now = times[i]
            traj = self.node_trajectory_id[n][-1]
            # traj = []
            # for j in range(len(traj_list) - 1, -1, -1):
            #     t, c = traj_list[j]
            #     if (t < time_now):
            #         ind = j
            #         traj = traj_list[:ind + 1]
            #         break
            # if (len(traj) != 0):
            cell_ids[i] = traj[1]
        return cell_ids

    def translate_id(self, ids):
        rows = ids // self.size
        columns = ids % self.size
        return rows, columns


    def get_neighbor_cells(self, cells, offset=1):
        #if cells.shape[0]==
        rows, cols = self.translate_id(cells)
        cols_0 = (cols < offset)
        cols[cols_0] = offset
        cols_size = (cols >= (self.size-offset))
        cols[cols_size] = self.size-offset-1
        rows_0 = (rows < offset)
        rows[rows_0] = offset
        rows_size = (rows >= (self.size-offset))
        rows[rows_size] = self.size-offset-1

        cells_tensor = torch.stack((rows,cols),dim=1).view(-1, 1, 1, 2)
        row_offsets, col_offsets = torch.meshgrid(torch.arange(-offset, offset+1).to(self.device),torch.arange(-offset, offset+1).to(self.device))




        new_rows = cells_tensor[:, :, :, 0] + row_offsets # num_cell,3,3
        new_cols = cells_tensor[:, :, :, 1] + col_offsets

        # valid_indices = (new_rows >= 0) & (new_rows < self.size) & (new_cols >= 0) & (new_cols < self.size)
        # new_rows = new_rows[valid_indices]
        # new_cols = new_cols[valid_indices]
        cell_ids = new_rows * self.size + new_cols
        return cell_ids.view(cells.shape[0], -1)


    def get_neighbor_from_nodeid(self, cells, nodes, layer=2, max_num=25):
        cells = self.node_to_map[nodes]
        neighbor_cells = self.get_neighbor_cells(cells, layer)
        # neighbor_times = self.map_times[neighbor_cells]
        # times_change = (neighbor_times > times.view(-1,1))
        # neighbor_times[times_change] = times.view(-1,1)[times_change]
        neighbor_bool = torch.ones((neighbor_cells.shape)).bool().to(self.device)
        # cells = self.node_to_map[nodes]
        # rows, columns = self.translate_id(cells)
        # batch_size = len(nodes)
        # neighbor_nodes = torch.zeros((len(nodes), max_num)).to(self.device)
        # neighbor_bool = torch.zeros((len(nodes), max_num)).to(self.device)
        # list_size = torch.arange(self.size)
        #
        # neigx = torch.reshape(torch.repeat_interleave(list_size, batch_size, dim=0), (-1, batch_size)).to(self.device)
        # neigy = torch.reshape(torch.repeat_interleave(list_size, batch_size, dim=0), (-1, batch_size)).to(self.device)
        #
        # ax = torch.logical_and(torch.gt(neigx, rows - layer), torch.lt(neigx, rows + layer))
        # ay = torch.logical_and(torch.gt(neigy, columns - layer), torch.lt(neigy, columns + layer))
        # ax = torch.transpose(ax, 0, 1)
        # ay = torch.transpose(ay, 0, 1)
        # neighbor_map_update_bool = torch.tensor([]).to(self.device)
        # list484 = torch.arange(self.size * self.size).to(self.device)
        #
        # max_num = torch.tensor(max_num).to(self.device)
        # for i in range(batch_size):
        #     ax_i = ax[i]
        #     ay_i = ay[i]
        #     neighbor = torch.outer(ax_i, ay_i) * 1.
        #     neighbor = torch.reshape(neighbor, (self.size * self.size,)).bool()
        #     neighbor_i = neighbor * (1 - self.empty_map).bool()
        #     neighbor_i[cells[i]] = False
        #     true_neighbor = list484[neighbor_i][-max_num:]
        #     neighbor_nodes[i, max_num - true_neighbor.shape[0]:] = true_neighbor
        #     neighbor_bool[i, max_num - true_neighbor.shape[0]:] = 1
        #     # tn_list = []
        #     # for tn in true_neighbor:
        #     #     tn_list.extend(self.cell_to_node[tn.item()])
        #     # tn_list = tn_list[-max_num:]
        #
        #     # neighbor_nodes[i, max_num-len(tn_list):] = torch.tensor(tn_list).to(self.device)
        #     # neighbor_bool[i, max_num - len(tn_list):] = 1
        return neighbor_cells, neighbor_bool


    # def get_neighborid_from_cellid(self, cells, layer=2):
    #
    #     batch_size = len(cells)
    #     max_num = 9
    #     neighbor_cells = torch.zeros((len(cells), max_num)).to(self.device)
    #     neighbor_bool = torch.zeros((len(cells), max_num)).to(self.device)
    #     list_size = torch.arange(self.size)
    #
    #     cells = torch.tensor(cells).to(self.device)
    #     rows, columns = self.translate_id(cells)
    #
    #     neigx = torch.reshape(torch.repeat_interleave(list_size, batch_size, dim=0), (-1, batch_size)).to(self.device)
    #     neigy = torch.reshape(torch.repeat_interleave(list_size, batch_size, dim=0), (-1, batch_size)).to(self.device)
    #
    #     ax = torch.logical_and(torch.gt(neigx, rows - layer), torch.lt(neigx, rows + layer))
    #     ay = torch.logical_and(torch.gt(neigy, columns - layer), torch.lt(neigy, columns + layer))
    #     ax = torch.transpose(ax, 0, 1)
    #     ay = torch.transpose(ay, 0, 1)
    #     neighbor_map_update_bool = torch.tensor([]).to(self.device)
    #     list484 = torch.arange(self.size * self.size).to(self.device)
    #
    #     max_num = torch.tensor(max_num).to(self.device)
    #     for i in range(batch_size):
    #         ax_i = ax[i]
    #         ay_i = ay[i]
    #         neighbor = torch.outer(ax_i, ay_i) * 1.
    #         neighbor = torch.reshape(neighbor, (self.size * self.size,)).bool()
    #         true_neighbor = list484[neighbor][-max_num:]
    #         neighbor_cells[i, max_num - true_neighbor.shape[0]:] = true_neighbor
    #         neighbor_bool[i, max_num - true_neighbor.shape[0]:] = 1
    #     return neighbor_cells, neighbor_bool





    # def get_mapid_before(self, source_node, destination_node):
    #
    #     cells_source = self.belongingness.get(source_node)
    #     cells_destination = self.belongingness.get(destination_node)
    #
    #     if (cells_source == None):
    #         if cells_destination == None:
    #             if len(self.empty_cells) == 0:
    #                 ran_cell = random.randint(0, self.size * self.size - 1)
    #                 ran_cell = torch.tensor(ran_cell).to(self.device)
    #                 return ran_cell, ran_cell
    #             ran_cell = random.randint(0, len(self.empty_cells)-1)
    #             ran_cell = torch.tensor(self.empty_cells[ran_cell]).to(self.device)
    #             return ran_cell, ran_cell
    #         else:
    #             return torch.tensor(cells_destination[-1]).to(self.device), None
    #
    #     else:
    #         if cells_destination == None:
    #             return None, torch.tensor(cells_source[-1]).to(self.device)
    #         else:
    #             return None, None

    # def get_mapid_before_list(self, node_ids):
    #
    #     cells = []
    #     nums = []
    #     batch_size = len(node_ids) // 2
    #     for i in range(batch_size):
    #         s_id = i
    #         d_id = batch_size + i
    #         source_node = node_ids[s_id]
    #         destination_node = node_ids[d_id]
    #         cells_source = self.belongingness.get(source_node)
    #         cells_destination = self.belongingness.get(destination_node)
    #         if (cells_source == None):
    #             if cells_destination == None:
    #                 if len(self.empty_cells) == 0:
    #                     ran_cell = random.randint(0, self.size * self.size - 1)
    #                     #ran_cell = torch.tensor(ran_cell).to(self.device)
    #                     cells.extend([ran_cell, ran_cell])
    #                     nums.extend([s_id, d_id])
    #                     self.set_belongingness([source_node, destination_node], [ran_cell, ran_cell])
    #
    #                 else:
    #                     ran_cell = random.randint(0, len(self.empty_cells)-1)
    #                     cells.extend([self.empty_cells[ran_cell], self.empty_cells[ran_cell]])
    #                     nums.extend([s_id, d_id])
    #                     self.set_belongingness([source_node, destination_node], [self.empty_cells[ran_cell], self.empty_cells[ran_cell]])
    #
    #             else:
    #                 cells.append(cells_destination[-1])
    #                 nums.append(s_id)
    #                 self.set_belongingness([source_node], [cells_destination[-1]])
    #
    #
    #         else:
    #             if cells_destination == None:
    #                 cells.append(cells_source[-1])
    #                 nums.append(d_id)
    #                 self.set_belongingness([destination_node], [cells_source[-1]])
    #
    #     cells = torch.tensor(cells).to(self.device)
    #     #self.set_belongingness(node_ids[nums], cells)
    #     nums = torch.tensor(nums).to(self.device)
    #     return  cells, nums


    # def get_mapid_before_list_unique(self, unique_nodes, unique_destinations):
    #
    #     cells = []
    #     nums = []
    #     batch_size = len(unique_nodes)
    #     for i in range(batch_size):
    #         s_id = i
    #         source_node = unique_nodes[s_id]
    #         destination_node = unique_destinations[i]
    #         cells_source = self.belongingness.get(source_node)
    #         cells_destination = self.belongingness.get(destination_node)
    #         if (cells_source == None):
    #             if cells_destination == None:
    #                 if len(self.empty_cells) == 0:
    #                     ran_cell = random.randint(0, self.size * self.size - 1)
    #                     ran_cell = torch.tensor(ran_cell).to(self.device)
    #                     cells.extend([ran_cell])
    #                     nums.extend([s_id])
    #                     self.set_belongingness([source_node], [ran_cell])
    #
    #                 else:
    #                     ran_cell = random.randint(0, len(self.empty_cells)-1)
    #                     ran_cell = torch.tensor(self.empty_cells[ran_cell]).to(self.device)
    #                     cells.extend([ran_cell])
    #                     nums.extend([s_id])
    #                     self.set_belongingness([source_node], [ran_cell])
    #
    #             else:
    #                 ran_cell = torch.tensor(cells_destination[-1]).to(self.device)
    #                 cells.append(ran_cell)
    #                 nums.append(s_id)
    #
    #                 self.set_belongingness([source_node], [ran_cell])
    #         # else:
    #         #     ran_cell = torch.tensor(cells_source[-1]).to(self.device)
    #         #     cells.append(ran_cell)
    #         #     nums.append(s_id)
    #
    #         # else:
    #         #     if cells_destination == None:
    #         #         cells.append(cells_source[-1])
    #         #         nums.append(d_id)
    #         #         self.set_belongingness([destination_node], [cells_source[-1]])
    #
    #     cells = torch.tensor(cells).to(self.device)
    #     #self.set_belongingness(node_ids[nums], cells)
    #     #print(nums)
    #     nums = torch.tensor(nums).to(self.device)
    #     return  cells, nums
    #




    def set_map(self, ids, embeddings):
        self.memory_map[ids] = embeddings# .data.clone()

    def get_true_weight(self, map_weight):
        map_weight = torch.reshape(map_weight, (self.size * self.size, ))
        list = torch.arange(self.size * self.size).to(self.device)
        # map_bool = map_bool.bool()
        weight_bool = torch.gt(map_weight, 0.0)
        weight_bool = weight_bool #  * self.empty_map
        weight_index = list[weight_bool.bool()]
        return weight_index, weight_bool



    def get_map_from_nodeid(self, node_ids):
        cell_ids = self.node_to_map[node_ids]
        cell_embs = self.memory_map[cell_ids]
        return cell_ids, cell_embs

    def get_cellid(self, node_ids):
        cell_ids = self.node_to_map[node_ids]
        return cell_ids

    def get_map(self, ids):
        return self.memory_map[ids, :]

    def detach_map(self):
        self.memory_map.detach_()
        self.node_trajectory.detach_()

    def display_map(self):
        distances = torch.cdist(self.memory_map, self.memory_map, p=2)
        all_distances = torch.reshape(distances,(-1,self.size,self.size))
        '''all_distances = []
        for i in range(self.size * self.size):
            distances = torch.cdist(self.memory_map[i], self.memory_map, p=2)
            distances_reshape = torch.reshape(distances,(self.size, self.size))
            all_distances.append(distances_reshape)'''
        return all_distances

    def print_map(self,file):
        print(self.memory_map.cpu().detach().numpy(), file=file)

    def pca_init(self):
        return

    def set_belongingness(self, nodes, cells):
        for i in range(len(nodes)):
            n = nodes[i]
            c = cells[i].cpu().detach().item()
            if c in self.empty_cells:
                self.empty_cells.remove(c)
            clist = self.belongingness.get(n)
            #c=torch.tensor(c)
            if clist == None:
                new_list = [c]
                self.belongingness[n] = new_list
                self.final_cell_map[n] = c
            else:
                clist.append(c)
                self.final_cell_map.update({n:c})
            '''nlist = self.belongingness_map.get(c)
            if nlist == None:
                new_n_list = [n]
                self.belongingness_map[c] = new_n_list
            else:
                nlist.append(n)'''

    # def display_nodes_map(self,file):
    #     cell2node = {}
    #     for k,v in self.belongingness.items():
    #         print(f'{k}:{v}',file=file)
    #         final_cell = v[-1]
    #         n_list = cell2node.get(final_cell)
    #         if n_list == None:
    #             new_n_list = [k]
    #             cell2node[final_cell]=new_n_list
    #         else:
    #             n_list.append(k)
    #     print("****************",file=file)
    #     sorted_dict=sorted(cell2node.items(), key=lambda x:x[0], reverse=True)
    #     '''for k,v in self.belongingness_map.items():
    #         print(f'{k}:{v}',file=file)'''
    #     #print(self.belongingness,file=file)
    #     for k, v in sorted_dict:
    #         print(f'{k}:{v}', file=file)


    def backup_map(self):
        node_trajectory_id_clone = {}
        for k, v in self.node_trajectory_id.items():
            node_trajectory_id_clone[k] = [x for x in v]

        cell_to_node_clone = {}
        for k, v in self.cell_to_node.items():
            cell_to_node_clone[k] = [x for x in v]
        return (self.memory_map.data.clone(), self.node_trajectory.data.clone(),
                self.node_to_map.data.clone(), self.position_encode.data.clone(),
                self.empty_map.data.clone(), node_trajectory_id_clone, cell_to_node_clone)

    def restore_map(self, map_back_up):
        self.memory_map.data, self.node_trajectory.data, self.node_to_map.data, self.position_encode.data, self.empty_map.data =\
            map_back_up[0].clone(), map_back_up[1].clone(), map_back_up[2].clone(), map_back_up[3].clone(), map_back_up[4].clone()

        self.node_trajectory_id = defaultdict(list)
        self.cell_to_node = defaultdict(list)

        for k, v in map_back_up[5].items():
            self.node_trajectory_id[k] = [x for x in v]

        for k, v in map_back_up[6].items():
            self.cell_to_node[k] = [x for x in v]


    def print_node_information(self, batch_i, epoch, flist):
        center_file = flist[6]
        node_file = flist[7]
        print(f'batch {batch_i} epoch {epoch}:', file=center_file)
        print(f'batch {batch_i} epoch {epoch}:', file=node_file)
        sorted_dict = sorted(self.cell_to_node.items(), key=lambda x: x[0], reverse=True)
        for k,v in sorted_dict:
            if self.empty_map[k] == False:
                print(f'{k}:{v}', file=center_file)
        if (epoch % 7==0) and (batch_i%100==0):
            print(self.node_to_map.cpu().detach().numpy(), file=node_file)
        #print(self.cell_to_node, file=center_file)

    def analyze_center(self, batch_i, epoch, flist, unique_nodes, timestamps, cell_ids,
                       neighbor_nodes, trajectory_nodes,  unknown_ind_inverse, #use_connections=False,
                       s_d_nums=None, s_d_times=None, s_n_nums=None, s_n_times=None,
                       nodes_time=None):
        file1 = flist[3]
        file2 = flist[4]
        neighbor_file = flist[5]
        cell_ids = cell_ids#[inverse]
        nodes = unique_nodes#[inverse]
        neighbor_nodes = neighbor_nodes#[inverse]
        trajectory_nodes = trajectory_nodes#[inverse]
        s_same_d_num = 0
        s_same_n_num = 0
        s_d_unknown = 0
        s_n_unknown = 0
        s_unknown_sum = 0
        d_unknown_sum = 0
        s_same_d_num1 = 0
        s_same_n_num1 = 0
        s_same_d_num2 = 0
        s_same_n_num2 = 0
        s_same_d_num3 = 0
        s_same_n_num3 = 0
        print(f'batch {batch_i} epoch {epoch}:',file=file1)
        print(f'batch {batch_i} epoch {epoch}:', file=neighbor_file)
        instance_num = nodes.shape[0] // 3
        for i in range(instance_num):
            s_center = cell_ids[i]
            d_center = cell_ids[i+instance_num]
            n_center = cell_ids[i+2*instance_num]
            s_unknown = unknown_ind_inverse[i]
            d_unknown = unknown_ind_inverse[i + instance_num]
            n_unknown = unknown_ind_inverse[i + 2 * instance_num]
            s_row = s_center // self.size
            s_col = s_center % self.size
            d_row = d_center // self.size
            d_col = d_center % self.size
            n_row = n_center // self.size
            n_col = n_center % self.size
            if s_center == d_center:
                s_same_d_num += 1
            elif ((s_row - d_row) <= 1) and ((s_row - d_row) >= -1) and ((s_col - d_col) <= 1) and ((s_col - d_col) >= -1):
                s_same_d_num1+=1
            elif ((s_row - d_row) <= 2) and ((s_row - d_row) >= -2) and ((s_col - d_col) <= 2) and ((s_col - d_col) >= -2):
                s_same_d_num2+=1
            elif ((s_row - d_row) <= 3) and ((s_row - d_row) >= -3) and ((s_col - d_col) <= 3) and ((s_col - d_col) >= -3):
                s_same_d_num3+=1
            if s_center == n_center:
                s_same_n_num += 1
            elif ((s_row - n_row) <= 1) and ((s_row - n_row) >= -1) and ((s_col - n_col) <= 1) and ((s_col - n_col) >= -1):
                s_same_n_num1+=1
            elif ((s_row - n_row) <= 2) and ((s_row - n_row) >= -2) and ((s_col - n_col) <= 2) and ((s_col - n_col) >= -2):
                s_same_n_num2+=1
            elif ((s_row - n_row) <= 3) and ((s_row - n_row) >= -3) and ((s_col - n_col) <= 3) and ((s_col - n_col) >= -3):
                s_same_n_num3+=1
            if s_unknown and d_unknown:
                s_d_unknown+=1
            elif s_unknown and n_unknown:
                s_n_unknown+=1
            elif s_unknown:
                s_unknown_sum+=1
            elif d_unknown:
                d_unknown_sum+=1
            print(f'{i}:s_center:{s_center}/{nodes[i]}, d:{d_center}/{nodes[i+instance_num]}, '
                   f'n:{n_center}/{nodes[i+2*instance_num]}, time:{timestamps[i]}, '
                  , file=file1)
            # print(f'{i}:s_d_nums:{s_d_nums[i]}, s_d_times:{s_d_times[i]}, '
            #       f's_n_nums:{s_n_nums[i]}, s_n_times:{s_n_times[i]}, '
            #       , file=file1)
            # print(f'{i}:s_center:{s_center}/{nodes[i]}, d:{d_center}/{nodes[i+instance_num]}, '
            #        f'n:{n_center}/{nodes[i+2*instance_num]}, time:{timestamps[i]}, '
            #       f'sou time:{nodes_time.cpu().numpy()[i]}, des time:{nodes_time.cpu().numpy()[i+instance_num]},'
            #       f' neg time:{nodes_time.cpu().numpy()[i + 2 * instance_num]}', file=file1)
            # neighbor_s, _, times_s = self.neighbor.get_temporal_neighbor2(
            #     nodes[i],
            #     timestamps[i],
            #     n_neighbors=30)
            # print(f'{i}:s{nodes[i]}:{neighbor_s}',file=neighbor_file)
            # print(f'{i}:s{nodes[i]}:{times_s}', file=neighbor_file)
            # print(f'{i}:s{nodes[i]} nei:{neighbor_nodes[i]}', file=neighbor_file)
            # print(f'{i}:s{nodes[i]} tra:{trajectory_nodes[i]}', file=neighbor_file)
            #
            # neighbor_d, _, times_d = self.neighbor.get_temporal_neighbor2(
            #     nodes[i+instance_num],
            #     timestamps[i],
            #     n_neighbors=30)
            # print(f'{i}:d{nodes[i+instance_num]}:{neighbor_d}', file=neighbor_file)
            # print(f'{i}:d{nodes[i+instance_num]}:{times_d}', file=neighbor_file)
            # print(f'{i}:d{nodes[i+instance_num]} nei:{neighbor_nodes[i+instance_num]}', file=neighbor_file)
            # print(f'{i}:d{nodes[i+instance_num]} tra:{trajectory_nodes[i+instance_num]}', file=neighbor_file)
            # neighbor_n, _, time_n = self.neighbor.get_temporal_neighbor2(
            #     nodes[i+2*instance_num],
            #     timestamps[i],
            #     n_neighbors=30)
            # print(f'{i}:n{nodes[i+2*instance_num]}:{neighbor_n}', file=neighbor_file)
            # print(f'{i}:n{nodes[i + 2 * instance_num]}:{time_n}', file=neighbor_file)
            # print(f'{i}:n{nodes[i+2*instance_num]} nei:{neighbor_nodes[i+2*instance_num]}', file=neighbor_file)
            # print(f'{i}:n{nodes[i+2*instance_num]} tra:{trajectory_nodes[i+2*instance_num]}', file=neighbor_file)
            # print(f'{i}:d{s_d_nums[i]} {s_d_times[i]} n:{s_n_nums[i]} {s_n_times[i]}', file=neighbor_file)




        print(f'batch {batch_i} epoch {epoch}:', file=file2)
        print(f'source same with d:{s_same_d_num}/{instance_num}',file=file2)
        print(f'source unknown with d:{s_d_unknown}/{instance_num}', file=file2)
        print(f'source same with n:{s_same_n_num}/{instance_num}',file=file2)
        print(f'source unknown with n:{s_n_unknown}/{instance_num}', file=file2)
        print(f'source unknown:{s_unknown_sum}/{instance_num}', file=file2)
        print(f'destination unknown:{d_unknown_sum}/{instance_num}', file=file2)

        print(f'source same with d1:{s_same_d_num1}/{instance_num}', file=file2)
        print(f'source same with n1:{s_same_n_num1}/{instance_num}', file=file2)

        print(f'source same with d2:{s_same_d_num2}/{instance_num}',file=file2)
        print(f'source same with n2:{s_same_n_num2}/{instance_num}',file=file2)

        print(f'source same with d3:{s_same_d_num3}/{instance_num}', file=file2)
        print(f'source same with n3:{s_same_n_num3}/{instance_num}', file=file2)

