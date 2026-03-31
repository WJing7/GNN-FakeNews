import argparse
import atexit
import time
from tqdm import tqdm
import copy as cp
import os
import sys
import random
import numpy as np
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader


from utils.data_loader import *
from utils.eval_helper import *
from utils.node_weight import WeightedReadout


"""

The GCN, GAT, and GraphSAGE implementation

"""


class Model(torch.nn.Module):
	def __init__(self, args, concat=False):
		super(Model, self).__init__()
		self.args = args
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.dropout_ratio = args.dropout_ratio
		self.model = args.model
		self.concat = concat
		self.use_weighted_readout = args.weighted_readout
		self.readout_layer = None
		self.readout_gate = None
		if self.use_weighted_readout:
			self.readout_layer = WeightedReadout(
				node_dim=self.nhid,
				attr_dim=args.profile_dim,
				hidden_dim=self.nhid,
				use_root_prox=args.readout_use_root_prox,
				use_subtree=args.readout_use_subtree,
				use_degree=args.readout_use_degree,
			)
			self.readout_gate = torch.nn.Linear(self.nhid * 2, self.nhid)

		if self.model == 'gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
		elif self.model == 'sage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
		elif self.model == 'gat':
			self.conv1 = GATConv(self.num_features, self.nhid)

		if self.concat:
			self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
			self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

		self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

	def forward(self, data):

		x, edge_index, batch = data.x, data.edge_index, data.batch

		edge_attr = None

		x = F.relu(self.conv1(x, edge_index, edge_attr))
		if self.use_weighted_readout:
			attr_x = getattr(data, 'profile_x', None)
			if attr_x is None and self.args.feature == 'profile':
				attr_x = data.x
			readout_edge_index = getattr(data, 'readout_edge_index', edge_index)
			struct_x = getattr(data, 'struct_x', None)
			weighted_x = self.readout_layer(
				x,
				data,
				attr_x=attr_x,
				edge_index=readout_edge_index,
				struct_x=struct_x,
			)
			base_x = gmp(x, batch)
			gate = torch.sigmoid(self.readout_gate(torch.cat([weighted_x, base_x], dim=1)))
			x = gate * weighted_x + (1.0 - gate) * base_x
		else:
			x = gmp(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))

		x = F.log_softmax(self.lin2(x), dim=-1)

		return x


class Tee:
	def __init__(self, *streams):
		self.streams = streams

	def write(self, data):
		for stream in self.streams:
			stream.write(data)
		return len(data)

	def flush(self):
		for stream in self.streams:
			stream.flush()


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)


def resolve_readout_struct_flags(args):
	explicit_struct = (
		args.readout_use_root_prox or
		args.readout_use_subtree or
		args.readout_use_degree
	)
	if args.readout_no_struct and explicit_struct:
		raise ValueError('`--readout_no_struct` cannot be combined with per-feature readout flags.')
	if args.readout_no_struct:
		return False, False, False
	if explicit_struct:
		return args.readout_use_root_prox, args.readout_use_subtree, args.readout_use_degree
	return True, True, True


def build_readout_struct_tag(args):
	struct_parts = []
	if args.readout_use_root_prox:
		struct_parts.append('root')
	if args.readout_use_subtree:
		struct_parts.append('subtree')
	if args.readout_use_degree:
		struct_parts.append('degree')
	return '_'.join(struct_parts) if struct_parts else 'none'


def build_log_path(args):
	log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
	os.makedirs(log_dir, exist_ok=True)
	if args.weighted_readout:
		mode = f'weighted_{build_readout_struct_tag(args)}'
	else:
		mode = 'origin'
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	return os.path.join(log_dir, f'{args.dataset}_{mode}_{timestamp}.log')


@torch.no_grad()
def compute_test(model, loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if not args.multi_gpu:
			data = data.to(args.device)
		out = model(data)
		if args.multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=35, help='maximum number of epochs')
parser.add_argument('--runs', type=int, default=10, help='number of repeated runs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='sage', help='model type, [gcn, gat, sage]')
parser.add_argument('--weighted_readout', action='store_true', help='use weighted readout instead of max pooling')
parser.add_argument('--readout_use_root_prox', action='store_true', help='use root proximity structural feature in weighted readout')
parser.add_argument('--readout_use_subtree', action='store_true', help='use subtree size structural feature in weighted readout')
parser.add_argument('--readout_use_degree', action='store_true', help='use node degree structural feature in weighted readout')
parser.add_argument('--readout_no_struct', action='store_true', help='disable all structural features in weighted readout')

args = parser.parse_args()
(
	args.readout_use_root_prox,
	args.readout_use_subtree,
	args.readout_use_degree,
) = resolve_readout_struct_flags(args)
log_path = build_log_path(args)
log_handle = open(log_path, 'a', encoding='utf-8')
stdout_stream = sys.stdout
stderr_stream = sys.stderr
tee_stream = Tee(stdout_stream, log_handle)
sys.stdout = tee_stream
sys.stderr = Tee(stderr_stream, log_handle)
atexit.register(log_handle.close)
set_seed(args.seed)

aux_feature = 'profile' if args.weighted_readout and args.feature != 'profile' else None
dataset = FNNDataset(
	root='data',
	feature=args.feature,
	aux_feature=aux_feature,
	include_readout=args.weighted_readout and (
		args.readout_use_root_prox or args.readout_use_subtree or args.readout_use_degree
	),
	use_root_prox=args.readout_use_root_prox,
	use_subtree=args.readout_use_subtree,
	use_degree=args.readout_use_degree,
	empty=False,
	name=args.dataset,
	transform=ToUndirected(),
)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
args.profile_dim = dataset.num_profile_features if dataset.num_profile_features > 0 else (
	args.num_features if args.feature == 'profile' else 0
)

print(f'Log file: {log_path}')
print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
loader_cls = DataListLoader if args.multi_gpu else DataLoader


if __name__ == '__main__':
	# Model training

	metrics_log = []
	for run in range(args.runs):
		run_seed = args.seed + run
		set_seed(run_seed)
		split_generator = torch.Generator().manual_seed(run_seed)
		training_set, validation_set, test_set = random_split(
			dataset, [num_training, num_val, num_test], generator=split_generator
		)

		train_loader = loader_cls(training_set, batch_size=args.batch_size, shuffle=True)
		val_loader = loader_cls(validation_set, batch_size=args.batch_size, shuffle=False)
		test_loader = loader_cls(test_set, batch_size=args.batch_size, shuffle=False)

		model = Model(args, concat=args.concat)
		if args.multi_gpu:
			model = DataParallel(model)
		model = model.to(args.device)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		print(f'Run {run + 1}/{args.runs}, seed={run_seed}')
		model.train()
		for epoch in tqdm(range(args.epochs)):
			loss_train = 0.0
			out_log = []
			for i, data in enumerate(train_loader):
				optimizer.zero_grad()
				if not args.multi_gpu:
					data = data.to(args.device)
				out = model(data)
				if args.multi_gpu:
					y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
				else:
					y = data.y
				loss = F.nll_loss(out, y)
				loss.backward()
				optimizer.step()
				loss_train += loss.item()
				out_log.append([F.softmax(out, dim=1), y])
			acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
			[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(model, val_loader)
			print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
				  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
				  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
				  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

		[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(model, test_loader, verbose=False)
		print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
			  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
		metrics_log.append([acc, f1_macro, f1_micro, precision, recall, auc, ap])

	metrics_arr = np.array(metrics_log, dtype=np.float32)
	mean_metrics = metrics_arr.mean(axis=0)
	print(f'Average results over {args.runs} runs: acc: {mean_metrics[0]:.4f}, '
		  f'f1_macro: {mean_metrics[1]:.4f}, f1_micro: {mean_metrics[2]:.4f}, '
		  f'precision: {mean_metrics[3]:.4f}, recall: {mean_metrics[4]:.4f}, '
		  f'auc: {mean_metrics[5]:.4f}, ap: {mean_metrics[6]:.4f}')
