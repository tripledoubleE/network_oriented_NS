library(igraph)
library(diffudist)
path="/home/ece/Desktop/Negative_Sampling/LightGCN-PyTorch/data/"


data=read.csv(paste(path,"/graph_edge_list.csv",sep=""))
g_ece=graph_from_data_frame(data,directed=FALSE)

i=2

dist=get_distance_matrix(g_ece,tau=i,type = "Normalized Laplacian", verbose = FALSE)
write.csv(dist,paste(path,"distance_tau",i,".csv"))