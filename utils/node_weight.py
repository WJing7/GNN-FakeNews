import torch
from torch import nn
from torch_geometric.utils import softmax


def _select_structural_features(root_prox, subtree, degree_norm, use_root_prox, use_subtree, use_degree):
	feature_parts = []
	if use_root_prox:
		feature_parts.append(root_prox.unsqueeze(1))
	if use_subtree:
		feature_parts.append(subtree.unsqueeze(1))
	if use_degree:
		feature_parts.append(degree_norm.unsqueeze(1))
	if feature_parts:
		return torch.cat(feature_parts, dim=1)
	return root_prox.new_zeros((root_prox.size(0), 0))


def build_structural_features(
	edge_index,
	batch,
	device=None,
	dtype=torch.float32,
	root_index=None,
	use_root_prox=True,
	use_subtree=True,
	use_degree=True,
):
	struct_dim = int(use_root_prox) + int(use_subtree) + int(use_degree)
	num_nodes = batch.size(0)
	if num_nodes == 0:
		target_device = device if device is not None else batch.device
		return torch.zeros((0, struct_dim), dtype=dtype, device=target_device)

	target_device = device if device is not None else batch.device
	if struct_dim == 0:
		return torch.zeros((num_nodes, 0), dtype=dtype, device=target_device)
	if edge_index is None or edge_index.numel() == 0:
		root_prox = torch.ones(num_nodes, dtype=dtype, device=target_device)
		subtree = torch.ones(num_nodes, dtype=dtype, device=target_device)
		degree = torch.zeros(num_nodes, dtype=dtype, device=target_device)
		return _select_structural_features(
			root_prox,
			subtree,
			degree,
			use_root_prox,
			use_subtree,
			use_degree,
		)

	edge_index_cpu = edge_index.cpu()
	batch_cpu = batch.cpu()
	num_graphs = int(batch_cpu.max().item()) + 1 if batch_cpu.numel() else 1
	counts = torch.bincount(batch_cpu, minlength=num_graphs)
	ptr_cpu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

	if root_index is None:
		root_cpu = ptr_cpu[:-1]
	else:
		root_cpu = root_index.view(-1).cpu()

	deg_cpu = torch.bincount(edge_index_cpu[0], minlength=num_nodes).float()
	root_prox = torch.zeros(num_nodes, dtype=torch.float32)
	subtree = torch.ones(num_nodes, dtype=torch.float32)
	degree_norm = torch.zeros(num_nodes, dtype=torch.float32)

	edge_batch = batch_cpu[edge_index_cpu[0]] if edge_index_cpu.numel() else torch.zeros(0, dtype=batch_cpu.dtype)
	for graph_idx in range(num_graphs):
		start = int(ptr_cpu[graph_idx])
		end = int(ptr_cpu[graph_idx + 1])
		if end <= start:
			continue

		root = int(root_cpu[graph_idx]) if graph_idx < root_cpu.numel() else start
		if root < start or root >= end:
			root = start

		local_num_nodes = end - start
		if edge_index_cpu.numel():
			mask = edge_batch == graph_idx
			local_edge_index = edge_index_cpu[:, mask]
			rows = (local_edge_index[0] - start).tolist()
			cols = (local_edge_index[1] - start).tolist()
		else:
			rows, cols = [], []

		adj = [[] for _ in range(local_num_nodes)]
		for src, dst in zip(rows, cols):
			adj[src].append(dst)

		dist = [-1] * local_num_nodes
		parent = [-1] * local_num_nodes
		root_local = root - start
		dist[root_local] = 0
		parent[root_local] = root_local
		queue = [root_local]
		order = []
		for node in queue:
			order.append(node)
			for neighbor in adj[node]:
				if dist[neighbor] < 0:
					dist[neighbor] = dist[node] + 1
					parent[neighbor] = node
					queue.append(neighbor)

		if all(value < 0 for value in dist):
			dist = [0] * local_num_nodes
		max_dist = max(dist)
		if max_dist < 0:
			max_dist = 0
		dist = [value if value >= 0 else max_dist + 1 for value in dist]

		subtree_size = [1] * local_num_nodes
		for node in reversed(order):
			ancestor = parent[node]
			if ancestor >= 0 and ancestor != node:
				subtree_size[ancestor] += subtree_size[node]

		max_subtree = max(subtree_size) if subtree_size else 1
		dist_t = torch.tensor(dist, dtype=torch.float32)
		subtree_t = torch.tensor(subtree_size, dtype=torch.float32)
		dist_norm = dist_t / float(max_dist) if max_dist > 0 else dist_t
		subtree_norm = subtree_t / float(max_subtree) if max_subtree > 0 else subtree_t

		root_prox[start:end] = 1.0 - dist_norm
		subtree[start:end] = subtree_norm

		deg_slice = deg_cpu[start:end]
		max_deg = float(deg_slice.max().item()) if deg_slice.numel() else 0.0
		degree_norm[start:end] = deg_slice / max_deg if max_deg > 0 else 0.0

	struct = _select_structural_features(
		root_prox,
		subtree,
		degree_norm,
		use_root_prox,
		use_subtree,
		use_degree,
	)
	return struct.to(device=target_device, dtype=dtype)


class WeightedReadout(nn.Module):
	def __init__(
		self,
		node_dim=0,
		attr_dim=0,
		hidden_dim=64,
		attr_from='tail',
		use_root_prox=True,
		use_subtree=True,
		use_degree=True,
	):
		super().__init__()
		self.node_dim = node_dim
		self.attr_dim = attr_dim
		self.attr_from = attr_from
		self.use_root_prox = use_root_prox
		self.use_subtree = use_subtree
		self.use_degree = use_degree
		self.struct_dim = int(use_root_prox) + int(use_subtree) + int(use_degree)
		in_dim = self.node_dim + self.struct_dim + (attr_dim if attr_dim > 0 else 0)
		self.weight_mlp = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, node_embed, data, attr_x=None, edge_index=None, struct_x=None):
		if attr_x is None:
			attr_x = getattr(data, 'profile_x', None)
		if attr_x is None:
			attr_x = getattr(data, 'x', None)
		attr = self._select_attr(attr_x, node_embed.device, node_embed.dtype)

		batch = getattr(data, 'batch', None)
		if batch is None:
			batch = node_embed.new_zeros(node_embed.size(0), dtype=torch.long)

		struct = None
		if self.struct_dim > 0:
			if struct_x is None:
				struct_x = getattr(data, 'struct_x', None)
			if struct_x is None:
				edge_index = edge_index if edge_index is not None else getattr(data, 'edge_index', None)
				root_index = getattr(data, 'root_index', None)
				struct = build_structural_features(
					edge_index,
					batch,
					device=node_embed.device,
					dtype=node_embed.dtype,
					root_index=root_index,
					use_root_prox=self.use_root_prox,
					use_subtree=self.use_subtree,
					use_degree=self.use_degree,
				)
			else:
				struct = struct_x.to(device=node_embed.device, dtype=node_embed.dtype)

		parts = []
		if self.node_dim > 0:
			parts.append(node_embed)
		if struct is not None and struct.size(1) > 0:
			parts.append(struct)
		if attr is not None:
			parts.append(attr)
		weight_in = torch.cat(parts, dim=1)

		weight_logit = self.weight_mlp(weight_in).squeeze(-1)
		weights = softmax(weight_logit, batch)
		out = self._segment_sum(node_embed * weights.unsqueeze(-1), batch)
		return out

	def _select_attr(self, attr_x, device, dtype):
		if self.attr_dim <= 0 or attr_x is None or attr_x.numel() == 0:
			return None
		if attr_x.dim() != 2:
			attr_x = attr_x.view(attr_x.size(0), -1)
		attr_x = attr_x.to(device=device, dtype=dtype)
		dim = min(self.attr_dim, attr_x.size(1))
		if dim == 0:
			return None
		if self.attr_from == 'head':
			return attr_x[:, :dim]
		if self.attr_from == 'tail':
			return attr_x[:, -dim:]
		return None

	def _segment_sum(self, src, index):
		num_groups = int(index.max().item()) + 1 if index.numel() else 1
		out = src.new_zeros((num_groups, src.size(1)))
		out.index_add_(0, index, src)
		return out
