"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from tqdm import tqdm
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

from collections import defaultdict
import networkx as nx
import pickle

import dgl
import dgl.nn as dglnn
from torch.optim import SparseAdam
from sklearn.metrics.pairwise import cosine_similarity


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
    def convert_sparse_to_networkx(self):
        raise NotImplementedError
    
    def create_item_projected_graph(self):
        raise NotImplementedError

    def load_item_pop(self):
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, config, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
        
        if config['neg_sample'] == 'alpha75':
            rows, cols = self.UserItemNet.nonzero()
            edges = np.column_stack((rows, cols))
            df = pd.DataFrame(edges, columns=['user', 'item'])

            def convert_id_to_string(id_value, prefix):
                return f'{prefix}_{id_value}'

            # Convert user IDs
            df['user'] = df['user'].apply(lambda x: convert_id_to_string(x, 'u'))

            # Convert item IDs
            df['item'] = df['item'].apply(lambda x: convert_id_to_string(x, 'i'))

            #G_bipartite = nx.from_pandas_edgelist(df, 'user', 'item', create_using=nx.Graph)

            self.G_bipartite=nx.Graph()
            self.G_bipartite.add_nodes_from(df['user'],bipartite=0)
            self.G_bipartite.add_nodes_from(df['item'],bipartite=1)
            self.G_bipartite.add_edges_from([(row.user, row.item) for row in df.itertuples(index=False)])

            nodes = list(self.G_bipartite.nodes())
            mapping = {node: int(node.split('_')[1]) for node in nodes}
            self.G_bipartite = nx.relabel_nodes(self.G_bipartite, mapping)

        if config['neg_sample'] == 'item_proj':
            rows, cols = self.UserItemNet.nonzero()
            edges = np.column_stack((rows, cols))
            df = pd.DataFrame(edges, columns=['user', 'item'])
            
            def convert_id_to_string(id_value, prefix):
                return f'{prefix}_{id_value}'

            # Convert user IDs
            df['user'] = df['user'].apply(lambda x: convert_id_to_string(x, 'u'))

            # Convert item IDs
            df['item'] = df['item'].apply(lambda x: convert_id_to_string(x, 'i'))

            #G_bipartite = nx.from_pandas_edgelist(df, 'user', 'item', create_using=nx.Graph)
            
            G_bipartite=nx.Graph()
            G_bipartite.add_nodes_from(list(set(df['user'])),bipartite=0)
            G_bipartite.add_nodes_from(list(set(df['item'])),bipartite=1)
            G_bipartite.add_edges_from([(row.user, row.item) for row in df.itertuples(index=False)])
            

            self.G_item_projected = nx.bipartite.projected_graph(G_bipartite, nodes=df['item'].unique())

            num_pairs_user = nx.number_of_nodes(self.G_item_projected) * (nx.number_of_nodes(self.G_item_projected) - 1)


            nodes = list(self.G_item_projected.nodes())
            mapping = {node: int(node.split('_')[1]) for node in nodes}
            self.G_item_projected = nx.relabel_nodes(self.G_item_projected, mapping)
    
            # Initialize a progress bar
            pbar = tqdm(total=num_pairs_user, desc="Converting generator to dictionary")

            # Initialize an empty dictionary
            self.path_length_dict_item = {}
            self.remap_dict = {}
            count = 0
            # Convert the generator to a dictionary
            length_generator_item = nx.all_pairs_shortest_path_length(self.G_item_projected)

            for source, distances in length_generator_item:
                self.remap_dict[source] = count
                self.path_length_dict_item[source] = dict(distances)
                count += 1
                pbar.update(len(distances))
            pbar.close()

        # for user projected 
        if config['neg_sample'] == 'user_proj':

            rows, cols = self.UserItemNet.nonzero()
            edges = np.column_stack((rows, cols))
            df = pd.DataFrame(edges, columns=['user', 'item'])
           
            def convert_id_to_string(id_value, prefix):
                return f'{prefix}_{id_value}'

            # Convert user IDs
            df['user'] = df['user'].apply(lambda x: convert_id_to_string(x, 'u'))

            # Convert item IDs
            df['item'] = df['item'].apply(lambda x: convert_id_to_string(x, 'i'))
            
            G_bipartite=nx.Graph()
            G_bipartite.add_nodes_from(df['user'],bipartite=0)
            G_bipartite.add_nodes_from(df['item'],bipartite=1)
            G_bipartite.add_edges_from([(row.user, row.item) for row in df.itertuples(index=False)])
            
            #is_bipartite = nx.is_bipartite(G_bipartite)

            self.G_user_projected = nx.bipartite.projected_graph(G_bipartite, nodes=df['user'].unique())

            num_pairs_user = nx.number_of_nodes(self.G_user_projected) * (nx.number_of_nodes(self.G_user_projected) - 1)

            nodes = list(self.G_user_projected.nodes())
            mapping = {node: int(node.split('_')[1]) for node in nodes}
            self.G_user_projected = nx.relabel_nodes(self.G_user_projected, mapping)

            # Initialize a progress bar
            pbar = tqdm(total=num_pairs_user, desc="Converting generator to dictionary")

            # Initialize an empty dictionary
            self.path_length_dict_user = {}
            self.remap_dict = {}
            count = 0
            # Convert the generator to a dictionary
            length_generator_user = nx.all_pairs_shortest_path_length(self.G_user_projected)

            for source, distances in length_generator_user:
                self.remap_dict[source] = count
                self.path_length_dict_user[source] = dict(distances)
                count += 1
                pbar.update(len(distances))


            pbar.close()
        
            #with open('/home/ece/Desktop/Negative_Sampling/LightGCN-PyTorch/ABC_user_my_dict.pkl', 'wb') as file:
            #    pickle.dump(self.path_length_dict_user, file)

        if config['neg_sample'] == 'item_proj_SimRank':
            rows, cols = self.UserItemNet.nonzero()
            edges = np.column_stack((rows, cols))
            df = pd.DataFrame(edges, columns=['user', 'item'])
            
            def convert_id_to_string(id_value, prefix):
                return f'{prefix}_{id_value}'

            # Convert user IDs
            df['user'] = df['user'].apply(lambda x: convert_id_to_string(x, 'u'))

            # Convert item IDs
            df['item'] = df['item'].apply(lambda x: convert_id_to_string(x, 'i'))

            #G_bipartite = nx.from_pandas_edgelist(df, 'user', 'item', create_using=nx.Graph)
            
            G_bipartite=nx.Graph()
            G_bipartite.add_nodes_from(list(set(df['user'])),bipartite=0)
            G_bipartite.add_nodes_from(list(set(df['item'])),bipartite=1)
            G_bipartite.add_edges_from([(row.user, row.item) for row in df.itertuples(index=False)])
            

            self.G_item_projected = nx.bipartite.projected_graph(G_bipartite, nodes=df['item'].unique())  

            nodes = list(self.G_item_projected.nodes())
            mapping = {node: int(node.split('_')[1]) for node in nodes}
            self.G_item_projected = nx.relabel_nodes(self.G_item_projected, mapping)

            self.simRank_dict = nx.simrank_similarity(self.G_item_projected)

        if config['neg_sample'] == 'item_proj_Panther':
            rows, cols = self.UserItemNet.nonzero()
            edges = np.column_stack((rows, cols))
            df = pd.DataFrame(edges, columns=['user', 'item'])
            
            def convert_id_to_string(id_value, prefix):
                return f'{prefix}_{id_value}'

            # Convert user IDs
            df['user'] = df['user'].apply(lambda x: convert_id_to_string(x, 'u'))

            # Convert item IDs
            df['item'] = df['item'].apply(lambda x: convert_id_to_string(x, 'i'))

            #G_bipartite = nx.from_pandas_edgelist(df, 'user', 'item', create_using=nx.Graph)
            
            G_bipartite=nx.Graph()
            G_bipartite.add_nodes_from(list(set(df['user'])),bipartite=0)
            G_bipartite.add_nodes_from(list(set(df['item'])),bipartite=1)
            G_bipartite.add_edges_from([(row.user, row.item) for row in df.itertuples(index=False)])
            

            self.G_item_projected = nx.bipartite.projected_graph(G_bipartite, nodes=df['item'].unique())  

            nodes = list(self.G_item_projected.nodes())
            mapping = {node: int(node.split('_')[1]) for node in nodes}
            self.G_item_projected = nx.relabel_nodes(self.G_item_projected, mapping)

            self.panther_sim_dict = {}
            num_pairs = nx.number_of_nodes(self.G_item_projected) * (nx.number_of_nodes(self.G_item_projected) - 1)

            pbar = tqdm(total=num_pairs, desc="Calculating Panther similarity")

            for node in list(mapping.values()):
                panther_sim = nx.panther_similarity(self.G_item_projected, node, k=nx.number_of_nodes(self.G_item_projected) - 1)
                self.panther_sim_dict[node] = panther_sim
                pbar.update(1)
            
            pbar.close()

        if config['neg_sample'] == 'metapath2vec':

            """
            Trains a MetaPath2Vec model on the user-item interaction data and returns user and item embeddings.

            Args:
                UserItemNet: A scipy sparse matrix representing user-item interactions.

            Returns:
                A tuple containing two dictionaries:
                    - user_embeddings: Dictionary mapping real user IDs to their corresponding embeddings.
                    - item_embeddings: Dictionary mapping real item IDs to their corresponding embeddings.
            """

            # Node ID mapping
            id_mapping = {}

            g = dgl.bipartite_from_scipy(self.UserItemNet, utype='_U', etype='_user_item_edge', vtype='_I')
            
            print(g)

            # Define metapath for user-item interactions
            metapath = ['_user_item_edge']

            # Create the MetaPath2Vec model
            model = dglnn.MetaPath2Vec(g, metapath, emb_dim=128, window_size=1, negative_size=5, sparse=True)

            # Define data loader for efficient batch training
            dataloader = DataLoader(torch.arange(g.num_nodes('_U')), batch_size=128, shuffle=True, collate_fn=model.sample)

            # Define optimizer (adjust parameters as needed)
            optimizer = SparseAdam(model.parameters(), lr=0.025)

            # Train the model
            for epoch in range(100):  # Adjust number of epochs based on dataset size and complexity
                for (pos_u, pos_v, neg_v) in dataloader:
                    loss = model(pos_u, pos_v, neg_v)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Retrieve learned embeddings
            user_nids = torch.LongTensor(model.local_to_global_nid['_U'])
            user_emb = model.node_embed(user_nids).detach().cpu().numpy()

            item_nids = torch.LongTensor(model.local_to_global_nid['_I'])
            item_emb = model.node_embed(item_nids).detach().cpu().numpy()

            similarity_matrix = cosine_similarity(item_emb)

            self.min_indices = np.argmin(similarity_matrix, axis=1)
            self.min_indices_dict = {}
            for i, row in enumerate(similarity_matrix):
                # Get indices of the 5 smallest cosine values excluding the diagonal (self-similarity)
                min_indices = np.argsort(row)[1:11]
                self.min_indices_dict[i] = min_indices

    
    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
    
    def create_item_projected_graph(self):
        csr_usr_item_graph = self.UserItemNet

        rows, cols = csr_usr_item_graph.nonzero()
        edges = np.column_stack((rows, cols))
        df = pd.DataFrame(edges, columns=['user', 'item'])
        G_bipartite = nx.from_pandas_edgelist(df, 'user', 'item')

        G_item_projected = nx.bipartite.projected_graph(G_bipartite, nodes=df['item'].unique())
        # Graph with 9177 nodes and 3892067 edges

        return G_item_projected

    def get_distance_matrix(self, nx_graph):
        
        # print("a")
        # length_generator = nx.all_pairs_shortest_path_length(nx_graph)
        # print("b")
        # length_dict = dict(length_generator)

        # print(length_dict)
        
        # Calculate the number of pairs of nodes
        num_pairs = nx.number_of_nodes(nx_graph) * (nx.number_of_nodes(nx_graph) - 1)

        # Initialize a progress bar
        pbar = tqdm(total=num_pairs, desc="Converting generator to dictionary")

        # Initialize an empty dictionary
        length_dict = {}

        # Convert the generator to a dictionary
        length_generator = nx.all_pairs_shortest_path_length(nx_graph)
        for source, distances in length_generator:
            length_dict[source] = dict(distances)
            pbar.update(len(distances))

        pbar.close()

        return length_dict               
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems

    def load_item_pop(self):
        csr_usr_item_graph = self.UserItemNet

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
    

    def convert_sparse_to_networkx(self):
        csr_usr_item_graph = self.UserItemNet
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
    
    def create_item_projected_graph(self):
        csr_usr_item_graph = self.UserItemNet

        rows, cols = csr_usr_item_graph.nonzero()
        edges = np.column_stack((rows, cols))
        df = pd.DataFrame(edges, columns=['user', 'item'])
        G_bipartite = nx.from_pandas_edgelist(df, 'user', 'item')

        G_item_projected = nx.bipartite.projected_graph(G_bipartite, nodes=df['item'].unique())
        # Graph with 9177 nodes and 3892067 edges

        return G_item_projected
    
    ################################# YAPILACAKLAR ######################################
    
    # distance matrix fonksiyonu (np array donsun)
    ###############################################
    # self.UserItemNet --> node id'ler nasil? datadaki id'ler m' yoksa remapped id ler mi?
    # eger remapped id'ler ise GO ON 
    # eger remapped id defilse donusum icin ekleme yapmak gerekicek (remapped dict gibi)

    def get_distance_matrix(self, nx_graph):
        
        # print("a")
        # length_generator = nx.all_pairs_shortest_path_length(nx_graph)
        # print("b")
        # length_dict = dict(length_generator)

        # print(length_dict)
        
        # Calculate the number of pairs of nodes
        num_pairs = nx.number_of_nodes(nx_graph) * (nx.number_of_nodes(nx_graph) - 1)

        # Initialize a progress bar
        pbar = tqdm(total=num_pairs, desc="Converting generator to dictionary")

        # Initialize an empty dictionary
        length_dict = {}

        # Convert the generator to a dictionary
        length_generator = nx.all_pairs_shortest_path_length(nx_graph)
        for source, distances in length_generator:
            length_dict[source] = dict(distances)
            pbar.update(len(distances))

        pbar.close()

        return length_dict


    # utils.ItemProjSample_original_python
    ##########################################
    # dataset.create_item_projected_graph --> gerek yok ???? 
    # dataset.get_distance_matrix
    # neg_item --> get_furthest_element(distance_matrix, pos_node) 

    # get_furthest_element(distance_matrix, pos_node)
    ##################################################
    # pos node satirindaki en yuksek distance'a sahip elemanin index'ini dondur.