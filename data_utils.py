import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset

def threshold_matrix(matrix, thr=0.3):
    adj_matrix = (matrix > thr).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

class GraphDataset(InMemoryDataset):
    def __init__(self, root, func_matrices, labels, transform=None, pre_transform=None):
        self.func_matrices = func_matrices
        self.labels = labels  
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []

        for idx, matrix in enumerate(self.func_matrices):
            adj_matrix = threshold_matrix(matrix)
            node_features = torch.tensor(matrix, dtype=torch.float32)
            edge_indices = np.array(np.nonzero(adj_matrix))
            edge_indices = torch.tensor(edge_indices, dtype=torch.long)

            graph = Data(
                x=node_features,
                edge_index=edge_indices,
                y=torch.tensor(self.labels[idx], dtype=torch.long)
            )
            data_list.append(graph)
        
        return self.collate(data_list)