# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

# UPDATE
# @Time   : 2021/7/23, 2020/8/31, 2020/10/6, 2020/9/18, 2021/3/19
# @Author : Xingyu Pan, Kaiyuan Li, Yupeng Hou, Yushuo Chen, Zhichao Feng
# @email  : xy_pan@foxmail.com, tsotfsk@outlook.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, fzcbupt@gmail.com
# Addition from gitHub tripleDoubleE

"""
recbole.sampler
########################
"""

import copy
import random
import powerlaw

import numpy as np
from numpy.random import sample
import torch
from collections import Counter
from collections import defaultdict
import networkx as nx
import scipy.sparse as sp
from recbole.model import general_recommender 
import pandas as pd
from recbole.config import configurator



class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.

    Args:
        distribution (str): The string of distribution, which is used for subclass.

    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution, alpha, datasets):
        self.distribution = ""
       #self.sample_num = sample_num
        self.alpha = alpha
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()
        self.datasets = datasets

    def set_distribution(self, distribution):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == "popularity":
            self._build_alias_table()


    def _uni_sampling(self, sample_num, user_ids):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_uni_sampling] should be implemented")

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()

        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError("Method [_get_candidates_list] should be implemented")

    def _get_candidates_list_per_user(self, user_id):
            """Get sample candidates list for _pop_sampling()

            Returns:
                candidates_list (list): a list of candidates id.
            """
            raise NotImplementedError("Method [_get_candidates_list_per_user] should be implemented")

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling."""
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        large_q = []
        small_q = []
        for i in self.prob:
            self.alias[i] = -1
            self.prob[i] = self.prob[i] / len(candidates_list)
            self.prob[i] = pow(self.prob[i], self.alpha)
        normalize_count = sum(self.prob.values())
        for i in self.prob:
            self.prob[i] = self.prob[i] / normalize_count * len(self.prob)
            if self.prob[i] > 1:
                large_q.append(i)
            elif self.prob[i] < 1:
                small_q.append(i)
        while len(large_q) != 0 and len(small_q) != 0:
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l
            self.prob[l] = self.prob[l] - (1 - self.prob[s])
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def _get_closest(self,list1, list2):
        # list1 = fake, low number of PA scores
        # list2 = Real, huge amount of PA scores

        real_pa_vals = np.zeros(len(list1))       

        for d in range(len(list1)):
            temp_result = abs(list1[d] - list2)

            min_val = np.amin(temp_result)
            
            min_val_index = np.where(temp_result == min_val)
            min_val_index = min_val_index[0][0]
            closest_element = list2[min_val_index]

            #list2 = np.delete(list2, min_val_index) ### silinisin mi ???? 
            real_pa_vals[d] = closest_element

        return real_pa_vals 
    

    def _calculate_powerlaw_dist(self, sample_num, degree_dict):

        min_it_deg = min(list(degree_dict.values()))
        max_it_deg = max(list(degree_dict.values()))

        pw_fitted = powerlaw.Fit(
                list(degree_dict.values()), xmin=min_it_deg, xmax=max_it_deg
            )
        
        
        #print('sample_num _calculate_powerlaw: ', sample_num)
        art_it_deg = pw_fitted.power_law.generate_random(sample_num)

        chosen_it_deg_user = self._get_closest(art_it_deg, np.array(list((degree_dict.values()))))

        degree_item_dict = {}
        for element, degree in degree_dict.items():
            if degree in degree_item_dict:
                degree_item_dict[degree].append(element)
            else:
                degree_item_dict[degree] = [element]


        sampled_items = []
        for degree in chosen_it_deg_user:
            matching_item_ids = degree_item_dict[degree]
            chosen_item_id = random.choice(matching_item_ids)
            sampled_items.append(chosen_item_id)

        return np.array(sampled_items)
    
    def _powerlaw_sampling(self, sample_num, condition):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_powerlaw_sampling] should be implemented")
    
    def _mcns_sampling(self):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_powerlaw_sampling] should be implemented")
    
    def _item_proj_sampling(self, user_ids):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_powerlaw_sampling] should be implemented")
    
    def _dise_negative_sampling(self, user_ids):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_powerlaw_sampling] should be implemented")
    def _dynamic_negative_sampling(self, user_ids):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_powerlaw_sampling] should be implemented")


    def sampling(self, sample_num, user_ids, dataset):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.distribution == "uniform":
            return self._uni_sampling(sample_num, user_ids)
        elif self.distribution == "popularity":
            return self._pop_sampling(sample_num, user_ids)
        elif self.distribution == "powerlaw-1":
            condition = 1
            return self._powerlaw_sampling(sample_num, condition, user_ids)
        elif self.distribution == "powerlaw-2":
            return self._powerlaw_sampling(sample_num, condition, user_ids)
        elif self.distribution == "mcns":
            return self._mcns_sampling()
        elif self.distribution == "item_proj":
            return self._item_proj_sampling(sample_num, user_ids)
        elif self.distribution == "dens":
            return self._dise_negative_sampling(user_ids)
        elif self.distribution == "dns":
            return self._dynamic_negative_sampling(user_ids)
        
        else:
            raise NotImplementedError(
                f"The sampling distribution [{self.distribution}] is not implemented."
            )

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError("Method [get_used_ids] should be implemented")

    def sample_by_key_ids(self, key_ids, num, dataset):
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """

        ## bu fonksiyona sample icin candidate user id'key key id adi altinda liste seklinde veriliyor

        key_ids = np.array(key_ids)
        #print('key_ids: ', key_ids)
        key_num = len(key_ids)
        #print('key_num: ', key_num)
        total_num = key_num * num
        #print('num: ', num)
        #print('total_num: ', total_num)


        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num, key_ids)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list), key_ids)
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list), key_ids, dataset)
                check_list = np.array(
                    [
                        i
                        for i, used, v in zip(
                            check_list,
                            self.used_ids[key_ids[check_list]],
                            value_ids[check_list],
                        )
                        if v in used
                    ]
                )


        return torch.tensor(value_ids, dtype=torch.long)


class Sampler(AbstractSampler):
    """:class:`Sampler` is used to sample negative items for each input user. In order to avoid positive items
    in train-phase to be sampled in valid-phase, and positive items in train-phase or valid-phase to be sampled
    in test-phase, we need to input the datasets of all phases for pre-processing. And, before using this sampler,
    it is needed to call :meth:`set_phase` to get the sampler of corresponding phase.

    Args:
        phases (str or list of str): All the phases of input.
        datasets (Dataset or list of Dataset): All the dataset for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, datasets, distribution="uniform", alpha=1.0):
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError(
                f"Phases {phases} and datasets {datasets} should have the same length."
            )

        self.phases = phases
        self.datasets = datasets

        self.uid_field = datasets[0].uid_field
        self.iid_field = datasets[0].iid_field

        self.user_num = datasets[0].user_num
        self.item_num = datasets[0].item_num

        #self.sample_num = sample_num

        super().__init__(distribution=distribution, alpha=alpha, datasets = datasets) 

    def _get_candidates_list(self):
        candidates_list = []
        
        for dataset in self.datasets:
            candidates_list.extend(dataset.inter_feat[self.iid_field].numpy())
        return candidates_list
    
    def _get_candidates_list_per_user(self, user_id):
        candidates_list = []

        for dataset in self.datasets:
            user_interacted_items = list(dataset.inter_feat[self.iid_field][dataset.inter_feat[self.uid_field] == user_id].numpy())
            #candidates_list.update(list(dataset.inter_feat[self.iid_field].numpy()) - list(user_interacted_items))
            candidates_list = [iid for iid in list(dataset.inter_feat[self.iid_field].numpy()) if iid not in list(user_interacted_items)]

        return candidates_list
    
    def _get_degree(self, condition):

        degree_dict = {}

        if condition not in [1, 2]:
            raise ValueError("Invalid condition variable. Use 1 for all items or 2 for per user.")

        if condition == 1:
            candidates_list = self._get_candidates_list()
            degree_dict = dict(Counter(candidates_list))
        
        
        elif condition == 2:
            for dataset in self.datasets:
                user_ids = set(dataset.inter_feat[self.uid_field].numpy())
                for user_id in user_ids:
                    candidates_list_per_user = self._get_candidates_list_per_user(user_id)
                    degree_dict[user_id] = dict(Counter(candidates_list_per_user))

        return degree_dict

    def dfs(self, mask, nx_graph, start_node, walks_num=100):

        user_num = self.datasets[0].user_num

        stack=[]
        stack.append(start_node)
        seen=set()
        seen.add(start_node)
        walks = []
        mask_list = set(mask[start_node])
        while (len(stack)>0):
            vertex=stack.pop()
            nodes=nx_graph[vertex]
            # print("nodes", nodes)
            for w in nodes:
                if w not in seen:
                    stack.append(w)
                    seen.add(w)
            if start_node < user_num:
                # print("user...")
                if vertex > user_num:
                    if vertex in mask_list:
                        pass
                    else:
                        walks.append(vertex)
                else:
                    pass
            else:
                # print("item...")
                if vertex > user_num:
                    if vertex in mask_list:
                        pass
                    else:
                        if vertex == start_node:
                            pass
                        else:
                            walks.append(vertex)
                else:
                    pass
            if len(walks) >= walks_num:
                break
        return walks

    def get_length(self, walks):
        length = 0
        for key in walks.keys():
            length += len(walks[key])
        return length

    def intermediate(self, nx_graph, mask):
        
        #user_num = dataset.n_users
        candidate = defaultdict(list)

        for node in nx_graph.nodes():
            walk = self.dfs(mask, nx_graph, node, walks_num=100)
            candidate[node].extend(walk)
            '''
            if node < user_num:
                pass
            else:
                walk = dfs(dataset, mask, nx_graph, node, walks_num=100)
                candidate[node].extend(walk)
            '''

        return candidate

    def _convert_sparse_to_networkx(self):
        csr_usr_item_graph = self.datasets[0].inter_matrix(form="csr").astype(np.float32)
        if isinstance(csr_usr_item_graph, torch.Tensor):
            # Convert PyTorch sparse tensor to a dense tensor and then to a SciPy sparse matrix
            csr_usr_item_graph = csr_usr_item_graph.to_dense().cpu().numpy()
            csr_usr_item_graph = sp.csr_matrix(csr_usr_item_graph)

        # Convert to COO matrix for easier iteration
        coo_adj = csr_usr_item_graph.tocoo()

        # Create a NetworkX graph
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(range(csr_usr_item_graph.shape[0]))

        # Add edges
        for i, j, _ in zip(coo_adj.row, coo_adj.col, coo_adj.data):
            if i != j:  # Exclude diagonal entries
                G.add_edge(i, j)

        return G
    
    def _load_item_pop(self):
        csr_usr_item_graph = self.datasets[0].inter_matrix(form="csr").astype(np.float32)
        item_pop = list()
        node_deg = dict()
        dd = defaultdict(list)

        for row, col in zip(*csr_usr_item_graph.nonzero()):
            dd[row].append(col)
            dd[col].append(row)

        for key in dd.keys():
            item_pop.append(1)
        deg_sum = np.sum(item_pop)
        for key in dd.keys():
            node_deg[key] = 1 / deg_sum

        return node_deg, dd
    
    def _mcns_sampling(self):
        print('INSIDE MCNS SAMPLING FUNCTION   .... ')

        nx_graph = self._convert_sparse_to_networkx()
        q_1_dict, mask = self._load_item_pop()

        candidates = self.intermediate(nx_graph, mask)

        user_num = self.datasets[0].user_num
        start_given = None
        N_steps = 10
        user_list = list(set(self.datasets[0].inter_feat[self.datasets[0].uid_field].numpy()))


        distribution = [0.01] * 100
        distribution = [i/np.sum(distribution) for i in distribution]

        batch_size = 2048

        if start_given is None:
            start = np.random.choice(list(candidates.keys()), batch_size)  # random init (user and item)

        else:
            start = start_given

        count = 0
        cur_state = start

        walks = defaultdict(list)
        generate_examples = list()
        
        while True:
            y_list = list()
            q_probs_list = list()
            q_probs_next_list = list()
            count += 1
            sample_num = np.random.random()

            if sample_num < 0.5:
                y_list = np.random.choice(list(q_1_dict.keys()), len(cur_state), p=list(q_1_dict.values()))
                q_probs_list = [q_1_dict[i] for i in y_list]
                q_probs_next_list = [q_1_dict[i] for i in cur_state]
            else:
                for i in cur_state:
                    y = np.random.choice(candidates[i], 1, p=distribution)[0]
                    y_list.append(y)
                    index = candidates[i].index(y)
                    q_probs = distribution[index]
                    q_probs_list.append(q_probs)
                    
                    node_list_next = candidates[y]
                    
                    if i in node_list_next:
                        index_next = node_list_next.index(i)
                        q_probs_next = distribution[index_next]
                    else:
                        q_probs_next = q_1_dict[i]
                    q_probs_next_list.append(q_probs_next) 

            u = np.random.rand()
            user = torch.tensor(user_list).long()
            item = torch.tensor(y_list).long()
            print("deneme")
            print(item)
            p_probs = general_recommender.LightGCN.get_p_probs(user, item).detach().cpu().numpy()
            p_probs_next = general_recommender.LightGCN.get_p_probs(torch.tensor(user_list).long(), 
                                                                    torch.tensor(cur_state).long()).detach().cpu().numpy()
            
            try:
                #print("... INSIDE TRY !!! ...")
                A_a_list = np.multiply(np.array(p_probs), np.array(q_probs_next_list))/ np.multiply(np.array(p_probs_next), np.array(q_probs_list))
            except: ### bu kisim sorunlu dogru hesapliyor mu diye emin olmak lazim ### 
                #print("... INSIDE EXCEPT !!!!! ...")
                A_a_list = np.multiply(np.array(p_probs), np.array(q_probs_next_list)[:, np.newaxis]) / np.multiply(np.array(p_probs_next), np.array(q_probs_list)[:, np.newaxis])
            
            next_state = list()
            next_user = list()
            if count > N_steps:
                for i in list(range(len(cur_state))):
                    if y_list[i] >= user_num:
                        walks[user_list[i]].append(cur_state[i])
                    else:
                        next_state.append(y_list[i])
                        next_user.append(user_list[i])
                    cur_state = next_state
                    user_list = next_user

            else:
                for i in list(range(len(cur_state))):
                    A_a = A_a_list[i]                        
                    alpha = min(1, np.min(A_a))
                    if u < alpha:
                        next_state.append(y_list[i])
                    else:
                        next_state.append(cur_state[i])
                cur_state = next_state
            
            length = self.get_length(walks)

            if length == batch_size:
                generate_examples = list()
                for user in list(walks.keys()):
                    d = walks[user]
                    if len(d) == 1:
                        generate_examples.append(d[0]) 

                    else:
                        generate_examples.append(d[0])
                        del walks[user][0]

                break
            else:
                continue  
            
        return generate_examples
    

    def _item_proj_sampling(self, sample_num, user_ids):
        csr_usr_item_graph = self.datasets[0].inter_matrix(form="csr").astype(np.float32)
        rows, cols = csr_usr_item_graph.nonzero()
        edges = np.column_stack((rows, cols))
        df = pd.DataFrame(edges, columns=['user', 'item'])
        G_bipartite = nx.from_pandas_edgelist(df, 'user', 'item')

        G_item_projected = nx.bipartite.projected_graph(G_bipartite, nodes=df['item'].unique())

        for user_id in user_ids:
            allPos = list(self.datasets[0].inter_feat[self.iid_field][self.datasets[0].inter_feat[self.uid_field] == user_id].numpy())
            
            while True:
                random_element = random.choice(allPos)

                # Initialize a dictionary to store the distance of each node from the start node
                distances = {node: float('inf') for node in G_item_projected.nodes}

                distances[random_element] = 0

                # Perform a breadth-first search
                queue = [random_element]

                while queue:
                    current_node = queue.pop(0)
                    for neighbor in G_item_projected.neighbors(current_node):
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)

                # Find the node with the maximum distance
                furthest_node = max(distances, key=distances.get)

                if furthest_node in allPos:
                    continue
                else:
                    break

            return furthest_node
        

    def _dise_negative_sampling(self, user_ids):
        #cur_epoch, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item
        csr_usr_item_graph = self.datasets[0].inter_matrix(form="csr").astype(np.float32)

        non_interacted_indices = np.where(csr_usr_item_graph.toarray() == 0)
        # Convert the non-interacted indices to a list of tuples
        non_interacted_indices_list = list(zip(non_interacted_indices[0], non_interacted_indices[1]))

        warmup = 100

        config = configurator.Config(model='LightGCN', config_file_list=['/home/ece/Desktop/Negative_Sampling/lightgcn_parameters.yaml'])
        final_config_dict = config._get_final_config_dict()
        model = general_recommender.LightGCN(final_config_dict, self.datasets[0])
        pos_item = list()

        for user_id in user_ids:
            pos_item.append(self.datasets[0].inter_feat[self.iid_field][self.datasets[0].inter_feat[self.uid_field] == user_id].numpy())
        
            # Find the indices where the user has not interacted (0 values)
            neg_candidates = set([item_id for user_id_, item_id in non_interacted_indices_list if user_id_ == user_id])
            

        user_gcn_emb = model.user_embedding.weight
        item_gcn_emb = model.item_embedding.weight

        batch_size = len(user_ids)
        s_e, p_e = user_gcn_emb[user_ids], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]
        
        gate_p = torch.sigmoid(model.item_gate(p_e) + model.user_gate(s_e))
        gated_p_e = p_e * gate_p    # [batch_size, n_hops+1, channel]

        gate_n = torch.sigmoid(model.neg_gate(n_e) + model.pos_gate(gated_p_e).unsqueeze(1))
        gated_n_e = n_e * gate_n    # [batch_size, n_negs, n_hops+1, channel]
        
        # default bu idi --> n_e_sel = (1 - min(1, cur_epoch / warmup)) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]
        # n_e_sel = (1 - max(0, 1 - (cur_epoch / self.warmup))) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]
        n_e_sel = (1 - self.alpha) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]

        """dynamic negative sampling"""
        scores = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
               range(neg_items_emb_.shape[1]), indices, :]

    def _dynamic_negative_sampling(self, user_ids):
        #user_gcn_emb, item_gcn_emb, user, neg_candidates
        csr_usr_item_graph = self.datasets[0].inter_matrix(form="csr").astype(np.float32)

        non_interacted_indices = np.where(csr_usr_item_graph.toarray() == 0)
        non_interacted_indices_list = list(zip(non_interacted_indices[0], non_interacted_indices[1]))
        config = configurator.Config(model='LightGCN', config_file_list=['/home/ece/Desktop/Negative_Sampling/lightgcn_parameters.yaml'])
        final_config_dict = config._get_final_config_dict()
        model = general_recommender.LightGCN(final_config_dict, self.datasets[0])
        user_gcn_emb = model.user_embedding.weight
        item_gcn_emb = model.item_embedding.weight

        s_e = user_gcn_emb[user_ids]  # [batch_size, n_hops+1, channel]

        for user_id in user_ids:        
            # Find the indices where the user has not interacted (0 values)
            neg_candidates = set([item_id for user_id_, item_id in non_interacted_indices_list if user_id_ == user_id])        
        
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]

        if self.pool == 'mean':
            s_e = s_e.mean(dim=1)  # [batch_size, channel]
            n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]

        """dynamic negative sampling"""
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs]
        indices = torch.max(scores, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()

        return item_gcn_emb[neg_item]

    def _uni_sampling(self, sample_num, user_ids):
        return np.random.randint(1, self.item_num, sample_num)
    

    def _powerlaw_sampling(self, sample_num, condition, user_ids):

        """Sample [sample_num] items in the powerlaw distribution.

        Args:
            

        Returns:
            sample_list (np.array): a list of samples.
        """

        if condition == 1:
            degree_dict = self._get_degree(condition)
            return self._calculate_powerlaw_dist(sample_num, degree_dict)

        elif condition == 2:
            all_sampled_items = []
            degree_dict = self._get_degree(condition)
            
            #for dataset in self.datasets:

            #user_ids = set(self.datasets[0].inter_feat[self.uid_field].numpy())
            #print(user_ids)
            #print('user_id_shape', len(user_ids))
            #user_ids = np.array(125) 
            #print('user_ids.shape: ',user_ids.shape)
            ### dogru oldu  mu tartismak lazim !!!!!!!
            ### belki get candidate_per_user fonk. degistirmek lazim !!!
            
            for user_id in np.array(user_ids[:(len(user_ids)//5)]):
                ###############
                sampled_items_per_user = self._calculate_powerlaw_dist(5, degree_dict[user_id])
                #print('shape of itemperuser:', sampled_items_per_user.shape)
                all_sampled_items.append(sampled_items_per_user)

            concatenated_items = np.concatenate(all_sampled_items)
            #print('shape of concat item:', concatenated_items.shape)
            
            return concatenated_items
                    

    def get_used_ids(self):
        """
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        used_item_id = dict()
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid in zip(
                dataset.inter_feat[self.uid_field].numpy(),
                dataset.inter_feat[self.iid_field].numpy(),
            ):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                raise ValueError(
                    "Some users have interacted with all items, "
                    "which we can not sample negative items for them. "
                    "Please set `user_inter_num_interval` to filter those users."
                )
        return used_item_id

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, :attr:`phase` is set the same as input phase, and :attr:`used_ids`
            is set to the value of corresponding phase.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase [{phase}] not exist.")
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    
    def sample_by_user_ids(self, user_ids, item_ids, num, datasets):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        
        try:
            return self.sample_by_key_ids(user_ids, num, datasets)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f"user_id [{user_id}] not exist.")
  


class KGSampler(AbstractSampler):
    """:class:`KGSampler` is used to sample negative entities in a knowledge graph.

    Args:
        dataset (Dataset): The knowledge graph dataset, which contains triplets in a knowledge graph.
        distribution (str, optional): Distribution of the negative entities. Defaults to 'uniform'.
    """

    def __init__(self, dataset, distribution="uniform", alpha=1.0):
        self.dataset = dataset

        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field
        self.hid_list = dataset.head_entities
        self.tid_list = dataset.tail_entities

        self.head_entities = set(dataset.head_entities)
        self.entity_num = dataset.entity_num

        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num, user_ids):
        return np.random.randint(1, self.entity_num, sample_num)

    def _get_candidates_list(self):
        return list(self.hid_list) + list(self.tid_list)

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used entity_ids is the same as tail_entity_ids in knowledge graph.
            Index is head_entity_id, and element is a set of tail_entity_ids.
        """
        used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)

        for used_tail_set in used_tail_entity_id:
            if len(used_tail_set) + 1 == self.entity_num:  # [pad] is a entity.
                raise ValueError(
                    "Some head entities have relation with all entities, "
                    "which we can not sample negative entities for them."
                )
        return used_tail_entity_id

    def sample_by_entity_ids(self, head_entity_ids, num=1):
        """Sampling by head_entity_ids.

        Args:
            head_entity_ids (numpy.ndarray or list): Input head_entity_ids.
            num (int, optional): Number of sampled entity_ids for each head_entity_id. Defaults to ``1``.

        Returns:
            torch.tensor: Sampled entity_ids.
            entity_ids[0], entity_ids[len(head_entity_ids)], entity_ids[len(head_entity_ids) * 2], ...,
            entity_id[len(head_entity_ids) * (num - 1)] is sampled for head_entity_ids[0];
            entity_ids[1], entity_ids[len(head_entity_ids) + 1], entity_ids[len(head_entity_ids) * 2 + 1], ...,
            entity_id[len(head_entity_ids) * (num - 1) + 1] is sampled for head_entity_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(head_entity_ids, num)
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError(f"head_entity_id [{head_entity_id}] not exist.")

class RepeatableSampler(AbstractSampler):
    """:class:`RepeatableSampler` is used to sample negative items for each input user. The difference from
    :class:`Sampler` is it can only sampling the items that have not appeared at all phases.

    Args:
        phases (str or list of str): All the phases of input.
        dataset (Dataset): The union of all datasets for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, dataset, distribution="uniform", alpha=1.0):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num, user_ids):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as positive item_ids.
            Index is user_id, and element is a set of item_ids.
        """
        return np.array([set() for _ in range(self.user_num)])

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            self.used_ids = np.array([{i} for i in item_ids])
            return self.sample_by_key_ids(np.arange(len(user_ids)), num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f"user_id [{user_id}] not exist.")

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, and :attr:`phase` is set the same as input phase.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase [{phase}] not exist.")
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler

class SeqSampler(AbstractSampler):
    """:class:`SeqSampler` is used to sample negative item sequence.

    Args:
        datasets (Dataset or list of Dataset): All the dataset for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.
    """

    def __init__(self, dataset, distribution="uniform", alpha=1.0):
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num, user_ids):
        return np.random.randint(1, self.item_num, sample_num)

    def get_used_ids(self):
        pass

    def sample_neg_sequence(self, pos_sequence):
        """For each moment, sampling one item from all the items except the one the user clicked on at that moment.

        Args:
            pos_sequence (torch.Tensor):  all users' item history sequence, with the shape of `(N, )`.

        Returns:
            torch.tensor : all users' negative item history sequence.

        """
        total_num = len(pos_sequence)
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        while len(check_list) > 0:
            value_ids[check_list] = self.sampling(len(check_list))
            check_index = np.where(value_ids[check_list] == pos_sequence[check_list])
            check_list = check_list[check_index]

        return torch.tensor(value_ids)
