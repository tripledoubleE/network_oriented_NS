'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
import model
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
import pickle
from tqdm import tqdm
import math
import itertools
import pandas as pd

from sources import mcns_sampling, item_projected_sampling

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
    
def NoSampling_original(dataset, neg_ratio = 1):

    dataset : BasicDataset

    S = NoSampling_original_python(dataset)

    with open('/home/ece/Desktop/Negative_Sampling/LightGCN-PyTorch/data/lastfm/train_with_negs_1_100.pkl', 'wb') as f:
        pickle.dump(S, f)
    
    return S  
 
def NoSampling_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    dataset : BasicDataset

    print("NO SAMPLING !!!")
    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)
    

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    allNegs = dataset.allNeg
    
    S = []

    for i, user in enumerate(tqdm(users, desc='Sampling')):
        posForUser = allPos[user]
        negForUser = allNegs[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]

        random_neg_items = random.sample(list(negForUser), 100)
        
        for negitem in random_neg_items:
            S.append([user, positem, negitem])

    return np.array(S)



def UniformSample_original(dataset, neg_ratio = 1):

    dataset : BasicDataset
    allPos = dataset.allPos
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    
    print("UniformSample_original ")
    print(S.shape)

    with open('/home/ece/Desktop/Negative_Sampling/LightGCN-PyTorch/data/lastfm/train_with_negs_1_1.pkl', 'wb') as f:
        pickle.dump(S, f)
    
    return S  
 
def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    dataset : BasicDataset

    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

def Alpha75_Sample_original(dataset, neg_ratio = 1):

    dataset : BasicDataset
    
    S = Alpha75_Sample_original_python(dataset)
 
    print(S.shape)
    
    return S  
 
def Alpha75_Sample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    dataset : BasicDataset

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    bipartite_graph = dataset.G_bipartite
    S = []

    for i, user in enumerate(users):
        user_degree = bipartite_graph.degree(user)
        degree_of_75 = math.ceil(user_degree ** 0.75)
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitems = np.random.randint(0, dataset.m_items, size = degree_of_75)

            if any(negitem in posForUser for negitem in negitems):
                continue
            else:
                break
        
        for negitem in negitems:
            S.append([user, positem, negitem])
        
    return np.array(S)


def MCNS_Sample(dataset, Recmodel):
    '''
    print("MCNS SAMPLING FUNCTION !!!!!")
    dataset : BasicDataset
    Recmodel: model.LightGCN
    #graph_tensor = dataset.getSparseGraph()
    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)
    nx_graph = dataset.convert_sparse_to_networkx()

    q_1_dict, mask = dataset.load_item_pop()
    #load_item_pop- > returns node degree dist and neighbor dict

    candidates = mcns_sampling.intermediate(dataset, nx_graph, mask)
    #print(list(candidates.keys()))
    print("MCNS negative samling candidates calculated")
    
    N_steps = 10
    start_given = None
    user_list = dataset.trainUniqueUsers.tolist()

    user_neg_pairs = mcns_sampling.negative_sampling(Recmodel, dataset, candidates, start_given, q_1_dict, N_steps, user_list) 
    print("generate examples finished ... ") 
    #print(user_neg_pairs) 

    #user-pos-pairs calculation
    user_pos_pairs = mcns_sampling.positive_sampling(dataset, user_list)
    print("pos sampling is finished ... ")
    #print(user_pos_pairs)

    # Combine the two arrays into pairs
    combined_pairs = list(zip(user_neg_pairs, user_pos_pairs))
    print("Combine positive and negative instances ... ")
    #print(combined_pairs)

    # Create the new NumPy array with the desired structure
    result = np.array([[x[0][0], x[1][1], x[0][1]] for x in combined_pairs])
    print("Combining finished >>>> S nd array calculated [[u_id, po_id, neg_id]] ...")
    #print(result)

    print(result.shape)

    return result
    '''

    dataset : BasicDataset
    #allPos = dataset.allPos

    S = MCNS_Sample_original_python(dataset, Recmodel)
    
    print("UniformSample_original ")
    print(S.shape)
    
    return S   

def MCNS_Sample_original_python(dataset, Recmodel):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """

    print("MCNS SAMPLING FUNCTION !!!!!")
    dataset : BasicDataset
    Recmodel: model.LightGCN
    #graph_tensor = dataset.getSparseGraph()
    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)
    nx_graph = dataset.convert_sparse_to_networkx()

    q_1_dict, mask = dataset.load_item_pop()
    #load_item_pop- > returns node degree dist and neighbor dict

    candidates = mcns_sampling.intermediate(dataset, nx_graph, mask)
    #print(list(candidates.keys()))
    print("MCNS negative samling candidates calculated")
    
    N_steps = 10
    start_given = None
    user_list = dataset.trainUniqueUsers.tolist()
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            
            generated_examples = mcns_sampling.negative_sampling(Recmodel, dataset, candidates, start_given, q_1_dict, N_steps, user_list)
            negitem = generated_examples[0]
            #print(negitem)


            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)



def ItemProjSample_original(dataset, neg_ratio = 1):

    dataset : BasicDataset

    S = ItemProjSample_original_python(dataset)
    
    print("ItemProjSample_original ")
    print(S.shape)
    
    return S  
 
def ItemProjSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    print("ITEM PROJECTED SAMPLING FUNCTION !!!!!")

    dataset : BasicDataset

    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)

    distance_dict = dataset.path_length_dict_item
    print("distance is calculated .... ")

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []

    for i, user in enumerate(tqdm(users, desc='Sampling')):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            negitem = item_projected_sampling.get_longest_path_node(distance_dict, positem)
            #negitem_id = item_projected_sampling.get_longest_path_node(distance_dict, positem)
            #negitem = dataset.remap_dict[negitem_id]
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        #print(S)

    return np.array(S)


def UserProjSample_original(dataset, neg_ratio = 1):

    dataset : BasicDataset

    S = UserProjSample_original_python(dataset)
    
    print(S.shape)
    
    return S  
 
def UserProjSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """

    dataset : BasicDataset

    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)

    distance_dict = dataset.path_length_dict_user   
    print("distance is calculated .... ")

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []

    for i, user in enumerate(tqdm(users, desc='Sampling')):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            negUser = item_projected_sampling.get_longest_path_node(distance_dict, user) #datadaki user id
            
            pos_items_for_negUser = allPos[negUser]

            posindex_negUser = np.random.randint(0, len(pos_items_for_negUser))
            positem_negUser = pos_items_for_negUser[posindex_negUser]
            negitem = positem_negUser

            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])

    return np.array(S)


def Dens_Sample_original(dataset, Recmodel, cur_epoch):

    dataset : BasicDataset

    S = Dens_Sample_original_python(dataset, Recmodel, cur_epoch)
    
    print(S.shape)
    
    return S  
 
def Dens_Sample_original_python(dataset, Recmodel, cur_epoch):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    # dataset, Recmodel, cur_epoch, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item

    dataset : BasicDataset
    Recmodel: model.LightGCN
    
    #user_gcn_emb = Recmodel.embedding_user.weight
    #item_gcn_emb = Recmodel.embedding_item.weight

    all_users, all_items = Recmodel.computer()

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    allNegs = dataset.allNeg
    S = []

    for i, user in tqdm(enumerate(users)):
        #batch_size = user.shape[0]
        posForUser = allPos[user]
        neg_candidate_for_user = allNegs[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            s_e, p_e = all_users[user], all_items[positem]
            n_e = all_items[neg_candidate_for_user]
            gate_p = torch.sigmoid(Recmodel.item_gate(p_e) + Recmodel.user_gate(s_e))
            gated_p_e = p_e * gate_p 

            gate_n = torch.sigmoid(Recmodel.neg_gate(n_e) + Recmodel.pos_gate(gated_p_e).unsqueeze(1))
            gated_n_e = n_e * gate_n 

            warmup = 100
            n_e_sel = (1 - min(1, cur_epoch / warmup)) * n_e - gated_n_e 
            print("n_e_sel")
            print(n_e_sel)
            print(n_e_sel.size())

            # scores = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
            # indices = torch.max(scores, dim=1)[1].detach()
            # neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
            # # [batch_size, n_hops+1, channel]
            # negitem =  neg_items_emb_[[[i] for i in range(batch_size)],
            #     range(neg_items_emb_.shape[1]), indices, :]
            scores = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1) # [batch_size, n_negs]
            indices = torch.argmax(scores, dim=0) # [batch_size]
            negitem = neg_candidate_for_user[indices]

            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])

    return np.array(S)

def Dynamic_Sample_original(dataset, Recmodel):

    dataset : BasicDataset

    S = Dynamic_Sample_original_python(dataset, Recmodel)
    
    print(S.shape)
    
    return S  
 
def Dynamic_Sample_original_python(dataset, Recmodel):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    # dataset, Recmodel, cur_epoch, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item

    dataset : BasicDataset
    Recmodel: model.LightGCN
    user_gcn_emb = Recmodel.embedding_user.weight
    item_gcn_emb = Recmodel.embedding_item.weight

    all_users, all_items = Recmodel.computer()

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    allNegs = dataset.allNeg
    S = []

    for i, user in enumerate(users):
        posForUser = allPos[user]
        neg_candidate_for_user = allNegs[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            s_e = all_users[user] # OR user_gcn_emb ???
            n_e = all_items[neg_candidate_for_user] # OR item_gcn_emb ???

            #if self.pool == 'mean':
            #s_e = s_e.mean(dim=1)  # [batch_size, channel]
            #n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]

            """dynamic negative sampling"""
            # scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs]
            # indices = torch.max(scores, dim=1)[1].detach()  # [batch_size]
            # negitem = torch.gather(neg_candidate_for_user, dim=1, index=indices.unsqueeze(-1)).squeeze()

            """dynamic negative sampling"""
            # Compute scores
            scores = torch.matmul(n_e, s_e.unsqueeze(-1)) # [batch_size, n_negs]

            # Get indices of highest scoring items
            indices = torch.argmax(scores, dim=0) # [batch_size]

            # Select highest scoring items
            negitem = neg_candidate_for_user[indices]

            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])

    return np.array(S)


def SimRank_Sample_original(dataset, neg_ratio = 1):

    dataset : BasicDataset

    S = SimRank_Sample_original_python(dataset)
    
    print("ItemProjSample_original ")
    print(S.shape)
    
    return S  
 
def SimRank_Sample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    print("ITEM PROJECTED SAMPLING FUNCTION !!!!!")

    dataset : BasicDataset

    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)

    simRank_dict = dataset.simRank_dict
    print("simRank is calculated .... ")

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []

    for i, user in enumerate(tqdm(users, desc='Sampling')):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            negitem = item_projected_sampling.get_min_sim_score(simRank_dict, positem)
            #negitem_id = item_projected_sampling.get_longest_path_node(distance_dict, positem)
            #negitem = dataset.remap_dict[negitem_id]
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        #print(S)

    return np.array(S)

def Panther_Sample_original(dataset, neg_ratio = 1):

    dataset : BasicDataset

    S = Panther_Sample_original_python(dataset)
    
    print("ItemProjSample_original ")
    print(S.shape)
    
    return S  
 
def Panther_Sample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    print("ITEM PROJECTED SAMPLING FUNCTION !!!!!")

    dataset : BasicDataset

    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)

    panther_sim_dict = dataset.panther_sim_dict
    print(panther_sim_dict)
    print("panther_sim_dict is calculated .... ")

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []

    for i, user in enumerate(tqdm(users, desc='Sampling')):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            negitem = item_projected_sampling.get_min_sim_score(panther_sim_dict, positem)
            print(negitem)
            #negitem_id = item_projected_sampling.get_longest_path_node(distance_dict, positem)
            #negitem = dataset.remap_dict[negitem_id]
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        #print(S)

    return np.array(S)

def MetaPath2Vec_Sample_original(dataset, neg_ratio = 1):

    dataset : BasicDataset

    S = MetaPath2Vec_Sample_original_python(dataset)
    
    return S  
 
def MetaPath2Vec_Sample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    print("METAPATH2VEC SAMPLING FUNCTION !!!!!")

    dataset : BasicDataset

    print("user num: ", dataset.n_users)
    print("item num: ", dataset.m_items)


    #metapath2vec_item_embed_sim_dict = dataset.

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []

    for i, user in enumerate(tqdm(users, desc='Sampling')):
        #print("user: ", user)
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            #negitem = np.random.randint(0, dataset.m_items)
            #print("positem: ", positem)
            #print("positem: ", positem)
            negitems = dataset.min_indices_dict[positem]
            negitem = random.choice(negitems)
            
            #print("negitem ", negitem)
            #print("negitem: ", negitem)
            #negitem_id = item_projected_sampling.get_longest_path_node(distance_dict, positem)
            #negitem = dataset.remap_dict[negitem_id]
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        #print(S)

    return np.array(S)


# Naive_random_walk_original
def Naive_random_walk_original(dataset, neg_ratio = 1):

    dataset : BasicDataset

    S = Naive_random_walk_original_python(dataset)
    
    return S  
 
def Naive_random_walk_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    print("NAIVE RANDOM WALK SAMPLING FUNCTION !!!!!")

    dataset : BasicDataset

    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    path_length_dict = dataset.path_length_dict
    path_length_prob_dict = dataset.path_length_prob_dict
    S = []

    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]

        # uniform negative sampling
        while True:
            negitem_uniform = np.random.randint(0, dataset.m_items)
            if negitem_uniform in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem_uniform])

        # naive random walk sampling
        while True:
            user_id = 'u_' + str(user)
            path_length_prob_list_for_user = path_length_prob_dict[user_id]
            
            # create candidate list that contains only items
            filtered_list = [item for item in path_length_prob_list_for_user if item.startswith('i_')]

            # remove randomly choosen negative item from candidate list (if exists)   
            negitem_uniform_id = 'i_' + str(negitem_uniform)
            if negitem_uniform_id in filtered_list:
                filtered_list.remove(negitem_uniform_id)
    
            selected_elements = random.sample(filtered_list, k=4)
            selected_elements_int = [int(item.split('_')[1]) for item in selected_elements]

            if any(negitem in posForUser for negitem in selected_elements_int):
                continue
            else:
                break
           

        for selected_negitem in selected_elements_int:
            S.append([user, positem, selected_negitem])

        print(np.array(S))
        print(a)
        

    return np.array(S)




# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
