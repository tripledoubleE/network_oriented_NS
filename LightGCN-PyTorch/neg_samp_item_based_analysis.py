import pandas as pd
import networkx as nx

def main():
    # Read the CSV file into a DataFrame
    df_graph = pd.read_csv('/home/ece/Desktop/Negative_Sampling/LightGCN-PyTorch/data/lastfm/all_data.csv', header=None)

    # Add prefixes to the columns
    df_graph[1] = 'u_' + df_graph[1].astype(str)
    df_graph[2] = 'i_' + df_graph[2].astype(str)

    B = nx.Graph()
    B.add_nodes_from(df_graph[1], bipartite=0)
    B.add_nodes_from(df_graph[2], bipartite=1)
    B.add_edges_from(df_graph[[1, 2]].itertuples(index=False))

    not_in_pred_FN_items_1_1 = [319, 731, 437, 298, 324, 828, 323, 167, 582, 337]
    not_in_gt_FP_items_1_1 = [3887, 4393, 4353, 4347, 749, 4389, 4335, 965, 666, 4091]

    not_in_pred_FN_items_1_100 = [731, 319, 437, 753, 323, 82, 828, 167, 337, 98]
    not_in_gt_FP_items_1_100 = [731, 319, 437, 753, 323, 82, 828, 167, 337, 98]

    print("Analysis for not in Pred 1:1 = FN items ...")
    print('----------------------------------------')
    for element in not_in_pred_FN_items_1_1:
        print("Node: ", element)
        node_id = 'i_' + str(element)
        degree = B.degree(node_id)
        print("Degree: ", degree)
        print("degree centrality: ", nx.bipartite.degree_centrality(B, nodes={node_id})[str(node_id)])
        print("closeness centrality: ", nx.bipartite.closeness_centrality(B, nodes={node_id})[str(node_id)])
        print('Node redundancy: ', nx.bipartite.node_redundancy(B, [str(node_id)]))
        try:
            print("betweenless centrality: ", nx.bipartite.betweenness_centrality(B, nodes={str(node_id)}))
        except:
            print('Divide by zero error')
        print('---------------------------------------')

    
    print("Analysis for not in GT 1:1 = FP items ...")
    print('----------------------------------------')
    for element in not_in_gt_FP_items_1_1:
        print("Node: ", element)
        node_id = 'i_' + str(element)
        degree = B.degree(node_id)
        print("Degree: ", degree)
        print("degree centrality: ", nx.bipartite.degree_centrality(B, nodes={node_id})[str(node_id)])
        print("closeness centrality: ", nx.bipartite.closeness_centrality(B, nodes={node_id})[str(node_id)])
        print('Node redundancy: ', nx.bipartite.node_redundancy(B, [str(node_id)]))
        try:
            print("betweenless centrality: ", nx.bipartite.betweenness_centrality(B, nodes={str(node_id)}))
        except:
            print('Divide by zero error')
        print('---------------------------------------')



    print("Analysis for not in Pred 1:100 = FN and FP items ...")
    print('----------------------------------------')
    for element in not_in_pred_FN_items_1_100:
        print("Node: ", element)
        node_id = 'i_' + str(element)
        degree = B.degree(node_id)
        print("Degree: ", degree)
        print("degree centrality: ", nx.bipartite.degree_centrality(B, nodes={node_id})[str(node_id)])
        print("closeness centrality: ", nx.bipartite.closeness_centrality(B, nodes={node_id})[str(node_id)])
        print('Node redundancy: ', nx.bipartite.node_redundancy(B, [str(node_id)]))
        try:
            print("betweenless centrality: ", nx.bipartite.betweenness_centrality(B, nodes={str(node_id)}))
        except:
            print('Divide by zero error')
        print('---------------------------------------')


if __name__ == "__main__":
    main()
