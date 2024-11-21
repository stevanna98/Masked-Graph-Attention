import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import unbatch_edge_index

def threshold_matrix(matrix, thr=0.3):
    adj_matrix = (matrix > thr).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

def node_masking(b_ei, b_map, B, m):
    mask = torch.full((B, m, m), fill_value=False, dtype=torch.bool)
    graph_idx = b_map.index_select(0, b_ei[0, :])
    eis = unbatch_edge_index(b_ei, b_map)
    ei = torch.cat(eis, dim=1)
    mask[graph_idx, ei[0, :], ei[1, :]] = True
    return ~mask

class GraphDataset(InMemoryDataset):
    def __init__(self, root, func_matrices, labels, threshold=0.3, transform=None, pre_transform=None):
        self.func_matrices = func_matrices
        self.labels = labels  
        self.threshold = threshold
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []

        for idx, matrix in enumerate(self.func_matrices):
            adj_matrix = threshold_matrix(matrix, self.threshold)
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