import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import warnings

from network import MaskedAttentionGraphs
from data_utils import GraphDataset, node_masking

from torch_geometric.data import DataLoader

warnings.filterwarnings('ignore')
device = "mps" if torch.backends.mps.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Masked Attention Graphs')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads')
parser.add_argument('--dim_hidden', type=int, default=64, help='Hidden dimension')
parser.add_argument('--ln', action='store_true', help='Layer normalization')
parser.add_argument('--d', type=str, required=True, help='path to data')
parser.add_argument('--thr', type=float, default=0.3, help='Threshold')
args = parser.parse_args()

func_matrices = np.load(args.d + '/func_matrices.npy')
labels = np.load(args.d + '/study_group.npy')

dataset = GraphDataset(
    root=args.d + '/processed',
    func_matrices=func_matrices,
    labels=labels
)

n = len(dataset)
n_train = int(0.8 * n)
n_test = n - n_train

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

n_features = dataset[0].num_node_features
model = MaskedAttentionGraphs(dim_input=n_features, num_outputs=4, dim_output=1, 
                              dim_hidden=args.dim_hidden, num_heads=args.num_heads, ln=args.ln).to(device)

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            b_ei = batch.edge_index
            b_map = batch.batch
            B, m = batch.num_graphs, batch[0].num_nodes

            mask = node_masking(b_ei, b_map, B, m).to(device)
            matrix_product = batch.x.unsqueeze(0).bmm(batch.x.unsqueeze(0).transpose(1, 2))
            matrix_masked = matrix_product.masked_fill(mask == 0, float('-inf'))

            logits = model(matrix_masked)
            loss = loss_fn(logits, batch.y.unsqueeze(0))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate predictions and accuracy
            _, predicted = torch.max(logits, dim=1)  # Get the predicted class (logits -> predicted class)
            correct_predictions += (predicted == batch.y).sum().item()  # Count correct predictions
            total_predictions += batch.y.size(0)  # Count total predictions

        accuracy = correct_predictions / total_predictions * 100  # Calculate accuracy as percentage
        print(f'Epoch {epoch+1}/{args.epochs} Loss: {epoch_loss:.4f} Train Accuracy: {accuracy:.2f}%')

        # Call the test method at the end of each epoch to print test accuracy
        test_accuracy = test()
        print(f'Epoch {epoch+1}/{args.epochs} Test Accuracy: {test_accuracy:.2f}%')

    print('Training done!')

def test():
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            b_ei = batch.edge_index
            b_map = batch.batch
            B, m = batch.num_graphs, batch[0].num_nodes

            # Use the same masking and input preparation as in train
            mask = node_masking(b_ei, b_map, B, m).to(device)
            matrix_product = batch.x.unsqueeze(0).bmm(batch.x.unsqueeze(0).transpose(1, 2))
            matrix_masked = matrix_product.masked_fill(mask == 0, float('-inf'))

            logits = model(matrix_masked)

            # Calculate predictions and accuracy
            _, predicted = torch.max(logits, dim=1)  # Get the predicted class (logits -> predicted class)
            correct_predictions += (predicted == batch.y).sum().item()  # Count correct predictions
            total_predictions += batch.y.size(0)  # Count total predictions

    accuracy = correct_predictions / total_predictions * 100  # Calculate accuracy as percentage
    return accuracy

if __name__ == '__main__':
    train()
