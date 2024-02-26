from collections import defaultdict
import torch
import numpy as np
from dataloader import BasicDataset
import model
import networkx as nx


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