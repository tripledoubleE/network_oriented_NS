from collections import defaultdict
import torch
import numpy as np
from dataloader import BasicDataset
import model
import networkx as nx
from tqdm import tqdm

'''
# Function to calculate the furthest node to given node = Sampling func

def furthest_node_from_given_node(item_proj_graph, start_node):
    # Initialize a dictionary to store the distance of each node from the start node
    distances = {node: float('inf') for node in item_proj_graph.nodes}
    distances[start_node] = 0

    # Perform a breadth-first search
    queue = [start_node]
    while queue:
        current_node = queue.pop(0)
        for neighbor in item_proj_graph.neighbors(current_node):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    # Find the node with the maximum distance
    furthest_node = max(distances, key=distances.get)
    return furthest_node, distances[furthest_node]

'''
'''
def furthest_node_from_given_node(item_proj_graph, start_node):
    # Initialize distances and queue
    print("inside furthest_node_from_given_node_with_heuristic func ... ")
    distances = {node: float('inf') for node in item_proj_graph.nodes}
    distances[start_node] = 0
    queue = [(start_node, item_proj_graph.degree(start_node))]  # (node, degree)

    # Priority queue based on estimated remaining distance (lower degree first)
    import heapq
    heapq.heapify(queue)

    # Initialize progress bar
    total_nodes = len(item_proj_graph.nodes)  # Assuming this is the total node count
    pbar = tqdm(total=total_nodes)

    while queue:
        current_node, current_degree = heapq.heappop(queue)
        #print(current_node)
        for neighbor in item_proj_graph.neighbors(current_node):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                # Estimated remaining distance based on degree (can be modified)
                estimated_distance = distances[neighbor] + (item_proj_graph.degree(neighbor) / 2)
                heapq.heappush(queue, (neighbor, estimated_distance))
        
    # Find the furthest node
    furthest_node = max(distances, key=distances.get)
    pbar.update(1)
    pbar.close()
    return furthest_node, distances[furthest_node]

'''

def get_longest_path_node(length_dict, start_node):

    # Get the dictionary of distances for the specific node
    distances = length_dict[start_node]

    # Convert the distances to a NumPy array
    distances_array = np.array(list(distances.values()))

    # Find the index of the node with the longest path
    max_node_index = np.argmax(distances_array)

    # Get the node ID with the longest path
    max_node = list(distances.keys())[max_node_index]

    return max_node


    