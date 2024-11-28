import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import warnings
import sys

from tqdm import tqdm
from network import MaskedAttentionGraphs
from data_utils import GraphDataset, node_masking

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from torch_geometric.data import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/masked_attention_graphs')

warnings.filterwarnings('ignore')
device = "mps" if torch.backends.mps.is_available() else "cpu"

data_path = '/Users/stefanovannoni/Desktop/PhD/PROGETTI/DTI-fMRI INTEGRATION USING GNN/carmen/data'

parser = argparse.ArgumentParser(description='Masked Attention Graphs')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
parser.add_argument('--dim_hidden', type=int, default=32, help='Hidden dimension')
parser.add_argument('--seeds', type=int, default=1, help='Number of seeds')
parser.add_argument('--ln', action='store_false', help='Layer normalization')
parser.add_argument('--d', type=str, default=data_path, help='path to data')
parser.add_argument('--thr', type=float, default=95, help='Threshold')
parser.add_argument('--dataset', type=str, default='pronia', help='Dataset')
args = parser.parse_args()

func_matrices = np.load(args.d + '/func_matrices.npy')
labels = np.load(args.d + '/study_group.npy')

if args.dataset == 'pronia':
    dataset = GraphDataset(
        root=args.d + '/processed',
        func_matrices=func_matrices,
        threshold=args.thr,
        labels=labels
    )
    n = len(dataset)
    n_train = int(0.8 * n)
    n_test = n - n_train

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

elif args.dataset == 'ogb':
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

n_features = dataset.num_node_features
n_classes = dataset.num_classes

model = MaskedAttentionGraphs(
    dim_input=n_features,
    num_seeds=args.seeds,
    dim_output=1,
    dim_hidden=args.dim_hidden,
    num_heads=args.num_heads,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()

# writer.add_graph(model, (torch.randn(32, 160, 160).to(device), torch.randn(32, 160, 160).to(device)))
# writer.close()
# sys.exit()

def train(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc='Training', total=len(loader)):
        batch = batch.to(device)
        x, edge_index, y = batch.x, batch.edge_index, batch.y.float()
        batch_size = batch.batch.max().item() + 1
        n_nodes = batch[0].num_nodes
        n_features = x.size(-1)

        x = x.view(batch_size, n_nodes, n_features).to(device)

        masks = node_masking(edge_index, batch.batch, batch_size, n_nodes).to(device)

        optimizer.zero_grad()
        output = model(x, masks).squeeze(-1)
        loss = criterion(output, y.view(-1,1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(loader, model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', total=len(loader)):
            batch = batch.to(device)
            x, edge_index, y = batch.x, batch.edge_index, batch.y.float()
            batch_size = batch.batch.max().item() + 1
            n_nodes = batch[0].num_nodes
            n_features = x.size(-1)

            x = x.view(batch_size, n_nodes, n_features)

            masks = node_masking(edge_index, batch.batch, batch_size, n_nodes).to(device)

            output = model(x, masks).squeeze(-1)
            predicted = output.argmax(dim=1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Flatten lists to compute metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

for epoch in range(args.epochs):
    train_loss = train(train_loader, model, optimizer, criterion)
    train_acc, train_precision, train_recall, train_f1 = evaluate(train_loader, model)
    test_acc, test_precision, test_recall, test_f1 = evaluate(test_loader, model)


    # writer.add_scalar('training loss', train_loss, epoch)
    # writer.add_scalar('training accuracy', train_acc, epoch)
    # writer.add_scalar('test accuracy', test_acc, epoch)

    print(f'Epoch {epoch+1}/{args.epochs} Loss: {train_loss:.4f} '
          f'Train Accuracy: {train_acc:.2f}% Test Accuracy: {test_acc:.2f}% '
          f'Train F1: {train_f1:.2f} Test F1: {test_f1:.2f}')

torch.save(model.state_dict(), 'model.pth')
