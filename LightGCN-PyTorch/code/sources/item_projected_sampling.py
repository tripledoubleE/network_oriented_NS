import numpy as np
import itertools


def get_longest_path_node(length_dict, start_node):

    sorted_dict = sorted(length_dict[start_node].items(), key=lambda x: x[1], reverse=True)
   
    # Use itertools.groupby to group items by their values
    grouped = itertools.groupby(sorted_dict, key=lambda x: x[1])

    # Get the first group (which contains the keys with the highest value values)
    keys_with_highest_value = next(grouped)[1]
    keys_np_array = np.array([key for key, _ in keys_with_highest_value])

    random_key = np.random.choice(keys_np_array, size=1)[0]
    
    return random_key


    