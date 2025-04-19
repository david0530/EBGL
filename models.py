import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch_geometric.nn import global_mean_pool  # or global_add_pool

class GCN(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # First Graph Convolution Layer (Node-focused)
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)

        # Second Graph Convolution Layer (Edge-focused)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e, node_layer=False)

        # Third Graph Convolution Layer (Node-focused)
        self.gc3 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e, node_layer=True)

        self.dropout = dropout

        # Global pooling layer
        self.pool = global_mean_pool  # or global_add_pool

        # Linear layer for graph classification
        self.linear = nn.Linear(nhid, nclass)

    def forward(self, X, Z, adj_e, adj_v, T, batch):
        # Apply first Graph Convolution Layer (Node-focused)
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])
        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # Apply second Graph Convolution Layer (Edge-focused)
        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])
        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # Apply third Graph Convolution Layer (Node-focused)
        gc3 = self.gc3(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc3[0]), F.relu(gc3[1])
        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # Global mean pooling
        X = self.pool(X, batch)  # Assuming 'batch' indicates graph membership

        # Apply Linear layer for graph classification
        X = self.linear(X)

        return X
