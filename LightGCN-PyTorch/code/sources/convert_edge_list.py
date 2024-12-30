import pandas as pd

# Define the input and output file paths
train_file_path = "/Users/eceerdem/Desktop/network_oriented_NS/LightGCN-PyTorch/data/gowalla/train.txt"
output_csv_path = "/Users/eceerdem/Desktop/network_oriented_NS/LightGCN-PyTorch/data/gowalla/graph_edge_list.csv"

# Initialize an empty list to store the edges
edges = []

# Read the train.txt file and parse it
with open(train_file_path, "r") as file:
    for line in file:
        nodes = list(map(int, line.strip().split()))
        source_node = nodes[0]
        destination_nodes = nodes[1:]
        # Create edges for the source node to each destination node
        edges.extend([(source_node, dest_node) for dest_node in destination_nodes])

# Convert the edges to a DataFrame
edge_df = pd.DataFrame(edges, columns=["source", "target"])

# Save the DataFrame to a CSV file
edge_df.to_csv(output_csv_path, index=False)
