from collections import defaultdict
import torch
import numpy as np
from dataloader import BasicDataset
import model
        
# iterative version
def dfs(dataset, mask, nx_graph, start_node, walks_num=100):

    user_num = dataset.n_users

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

def intermediate(dataset, nx_graph, mask):
    
    #user_num = dataset.n_users
    candidate = defaultdict(list)

    for node in nx_graph.nodes():
        walk = dfs(dataset, mask, nx_graph, node, walks_num=100)
        candidate[node].extend(walk)
        '''
        if node < user_num:
            pass
        else:
            walk = dfs(dataset, mask, nx_graph, node, walks_num=100)
            candidate[node].extend(walk)
        '''

    return candidate

def get_length(walks):
    length = 0
    for key in walks.keys():
        length += len(walks[key])
    return length

def negative_sampling(Recmodel, dataset, candidates, start_given, q_1_dict, N_steps, node1):
        
    user_num = dataset.n_users

    distribution = [0.01] * 100
    # distribution = norm.pdf(np.arange(0,100,1), 50, 10)
    # distribution = norm.pdf(np.arange(0,100,1), 0, 50)
    # distribution = norm.pdf(np.arange(0,100,1), 100, 100)
    # distribution = norm.pdf(np.arange(0,100,1), 50, 100)
    distribution = [i/np.sum(distribution) for i in distribution]
    batch_size = 64

    if start_given is None:
        start = np.random.choice(list(candidates.keys()), batch_size)  # random init (user and item)
        #np_set = set(start.flatten())
        #print(np_set.issubset(list(candidates.keys()))) # TRUE donuyor
        #start = np.random.choice(list(q_1_dict.keys()), batch_size)
    else:
        start = start_given

    count = 0
    cur_state = start

    user_list = node1
    walks = defaultdict(list)
    generate_examples = list()
    
    while True:
        y_list = list()
        q_probs_list = list()
        q_probs_next_list = list()
        count += 1
        sample_num = np.random.random()

        if sample_num < 0.5:
            # print("inside if .. ")
            # print(len(cur_state))
            y_list = np.random.choice(list(q_1_dict.keys()), len(cur_state), p=list(q_1_dict.values()))
            q_probs_list = [q_1_dict[i] for i in y_list]
            q_probs_next_list = [q_1_dict[i] for i in cur_state]
        else:
            # print("inside else .. ")
            # print(len(cur_state))
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
                

            # print("time_1", time.time() - tt_0) 
        u = np.random.rand()
        # print("outside if-else...")
        # print(len(cur_state))
        # print(len(y_list))
        # print(len(q_probs_list))
        # print(len(q_probs_next_list))

        Recmodel: model.LightGCN
        #print(torch.tensor(user_list).long().size())#75
        #print(torch.tensor(y_list).long().size()) #9102
        #p_probs = Recmodel.get_p_probs(torch.tensor(user_list).long(), torch.tensor(y_list)).numpy() 
        #p_probs = Recmodel.get_p_probs(torch.tensor(user_list).long(), torch.tensor(y_list).long()).numpy()
        p_probs = Recmodel.get_p_probs(torch.tensor(user_list).long(), torch.tensor(y_list).long()).detach().cpu().numpy()
        # print( 'p_probs')
        # print(p_probs.shape) # (75,64)
        # print(p_probs)
        #print(torch.tensor(cur_state).long().size()) #9177
        p_probs_next = Recmodel.get_p_probs(torch.tensor(user_list).long(), torch.tensor(cur_state).long()).detach().cpu().numpy()
        # print("p_probs_next")
        # print(p_probs_next.shape) # (75,64)
        # print(p_probs_next)

        # print("q_probs_next_list")
        # print(len(q_probs_next_list))
        # print((q_probs_next_list))
        # print("q_probs_list")
        # print(len(q_probs_list))
        # print((q_probs_list))

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

        length = get_length(walks)
        #print("length ", length)

        #print("walks ", walks)

        if length == batch_size:
            generate_examples = list()
            #for user in node1:
            for user in list(walks.keys()):
                #print(user)
                d = walks[user]
                if len(d) == 1:
                    generate_examples.append(d[0]) 
                    
                    #print("generate_examples inside if")  
                    #print(generate_examples) 
                else:
                    generate_examples.append(d[0])
                    del walks[user][0]
                    #print("generate_examples inside else")
                    #print(generate_examples)
            break
        else:
            #print("!!!!")
            continue  

    #print("GENERATED NEGATIVE INSTANCES ... ")
    #print(len(generate_examples))
    #print(generate_examples)

    
    #user_neg_pairs = np.array(list(zip(node1, generate_examples)))
    #print("user neg pairs ... ")
    #print(user_neg_pairs)
        
    
    return generate_examples

def positive_sampling(dataset, user_list):
    dataset : BasicDataset
    allPos = dataset.allPos
    user_pos_dict = {}

    for user in user_list:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]

        user_pos_dict[user] = positem
    
    pairs = [(key, value) for key, value in user_pos_dict.items()]

    user_pos_pairs = np.array(pairs)

    return user_pos_pairs


'''
def load_item_pop(X_train):
    item_pop = list()
    node_deg = dict()
    dd = defaultdict(list)
    for edge in X_train:
        dd[int(edge[0])].append(int(edge[1]))
        dd[int(edge[1])].append(int(edge[0]))

    for key in dd.keys():
        item_pop.append(1)
    deg_sum = np.sum(item_pop)
    for key in dd.keys():
        node_deg[key] = 1 /deg_sum
    return node_deg, dd

def load_edges(filename):
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            user = int(parts[0])
            items = list(map(int, parts[1:]))
            for item in items:
                edges.append((user, item))
    return edges
'''

'''     
def dfs(graph_tensor, start_node, mask, walks_num = 100, stack=None, seen=None, walks=None):
    dataset : BasicDataset
    user_num = dataset.trainDataSize

    if stack is None:
        stack = []
    if seen is None:
        seen = set()
    if walks is None:
        walks = []

    stack.append(start_node)
    seen.add(start_node)
    mask_list = set(mask[start_node])

    for neighbor in graph_tensor[start_node].nonzero().squeeze(1):
        neighbor = neighbor.item()
        if neighbor not in seen:
            if start_node < user_num:
                if neighbor > user_num and neighbor not in mask_list:
                    walks.append(neighbor)
            else:
                if neighbor > user_num and neighbor not in mask_list and neighbor != start_node:
                    walks.append(neighbor)
            if len(walks) >= walks_num:
                break
            dfs(graph_tensor, neighbor, mask, walks_num, stack, seen, walks)

    if len(walks) >= walks_num:
        return walks
    else:
        stack.pop()
        if len(stack) > 0:
            return dfs(graph_tensor, stack[-1], mask, walks_num, stack, seen, walks)
        else:
            return walks



def dfs(dataset, graph_tensor, start_node, mask, walks=[], walks_num=100):

    dataset : BasicDataset
    user_num = dataset.n_users

    walks.append(start_node)
    mask_list = set(mask[start_node])
    ## GERCEK DENEY YAPARKEN TRY-EXCEPT SIL !!!!
    try:
        for neighbor in graph_tensor[start_node].to_dense().nonzero()[1]:
            neighbor = neighbor.item()
            if neighbor not in walks:
                if len(walks) >= walks_num:
                    break
                if neighbor >= user_num:
                    if neighbor in mask_list:
                        pass
                    else:
                        if neighbor == start_node:
                            pass
                        else:
                            dfs(graph_tensor, mask, neighbor, walks, walks_num)
                else:
                    pass
    except:
        pass
    return walks

    
    
def dfs(dataset, graph_tensor, start_node, mask, walks_num=100):
    walk = []
    for _ in range(walks_num):
        current_node = start_node
        walk.append(current_node)
        while len(walk) < dataset.max_t:
            neighbors = graph_tensor[current_node].to_dense().nonzero()[0]  # Convert to dense tensor
            if len(neighbors) == 0:
                break
            next_node = torch.randint(0, len(neighbors), (1,))
            current_node = neighbors[next_node].item()
            if mask[current_node] == 0:
                break
            walk.append(current_node)
    return walk
'''

