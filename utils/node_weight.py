import torch
from torch import nn
from torch_geometric.utils import softmax


class WeightedReadout(nn.Module):
	def __init__(self, attr_dim=10, hidden_dim=64, attr_from='tail'):
		super().__init__()
		self.attr_dim = attr_dim
		self.attr_from = attr_from
		self.struct_dim = 3
		in_dim = self.struct_dim + (attr_dim if attr_dim > 0 else 0)
		self.weight_mlp = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, node_embed, data, attr_x=None, edge_index=None):
		if attr_x is None:
			attr_x = getattr(data, 'x', node_embed)
		attr = self._select_attr(attr_x, node_embed.device, node_embed.dtype)

		batch = getattr(data, 'batch', None)
		if batch is None:
			batch = node_embed.new_zeros(node_embed.size(0), dtype=torch.long)

		edge_index = edge_index if edge_index is not None else getattr(data, 'edge_index', None)
		struct = self._struct_features(edge_index, batch, data, node_embed.device, node_embed.dtype)

		if attr is None:
			weight_in = struct
		else:
			weight_in = torch.cat([attr, struct], dim=1)

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

	def _struct_features(self, edge_index, batch, data, device, dtype):
		num_nodes = batch.size(0)
		if edge_index is None or edge_index.numel() == 0:
			root_prox = torch.ones(num_nodes, dtype=dtype, device=device)
			subtree = torch.ones(num_nodes, dtype=dtype, device=device)
			degree = torch.zeros(num_nodes, dtype=dtype, device=device)
			return torch.stack([root_prox, subtree, degree], dim=1)

		edge_index_cpu = edge_index.cpu()
		batch_cpu = batch.cpu()

		ptr = getattr(data, 'ptr', None)
		if ptr is None:
			num_graphs = int(batch_cpu.max().item()) + 1 if batch_cpu.numel() else 1
			counts = torch.bincount(batch_cpu, minlength=num_graphs)
			ptr_cpu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
		else:
			ptr_cpu = ptr.cpu()

		root_index = getattr(data, 'root_index', None)
		if root_index is None:
			root_cpu = ptr_cpu[:-1]
		else:
			root_cpu = root_index.view(-1).cpu()

		num_graphs = int(ptr_cpu.numel() - 1)
		deg_cpu = torch.bincount(edge_index_cpu[0], minlength=num_nodes).float()
		root_prox = torch.zeros(num_nodes, dtype=torch.float32)
		subtree = torch.ones(num_nodes, dtype=torch.float32)
		degree_norm = torch.zeros(num_nodes, dtype=torch.float32)

		edge_batch = batch_cpu[edge_index_cpu[0]] if edge_index_cpu.numel() else torch.zeros(0, dtype=batch_cpu.dtype)
		for g in range(num_graphs):
			start = int(ptr_cpu[g])
			end = int(ptr_cpu[g + 1])
			if end <= start:
				continue

			root = int(root_cpu[g]) if g < root_cpu.numel() else start
			if root < start or root >= end:
				root = start

			num_nodes_g = end - start
			if edge_index_cpu.numel():
				mask = edge_batch == g
				e_idx = edge_index_cpu[:, mask]
				rows = (e_idx[0] - start).tolist()
				cols = (e_idx[1] - start).tolist()
			else:
				rows, cols = [], []

			adj = [[] for _ in range(num_nodes_g)]
			for s, t in zip(rows, cols):
				adj[s].append(t)

			dist_g = [-1] * num_nodes_g
			parent = [-1] * num_nodes_g
			root_local = root - start
			dist_g[root_local] = 0
			parent[root_local] = root_local
			queue = [root_local]
			order = []
			for u in queue:
				order.append(u)
				for v in adj[u]:
					if dist_g[v] < 0:
						dist_g[v] = dist_g[u] + 1
						parent[v] = u
						queue.append(v)

			if all(d < 0 for d in dist_g):
				dist_g = [0] * num_nodes_g
			max_dist = max(dist_g)
			if max_dist < 0:
				max_dist = 0
			dist_g = [d if d >= 0 else max_dist + 1 for d in dist_g]

			subtree_g = [1] * num_nodes_g
			for u in reversed(order):
				p = parent[u]
				if p >= 0 and p != u:
					subtree_g[p] += subtree_g[u]

			max_sub = max(subtree_g) if subtree_g else 1
			dist_t = torch.tensor(dist_g, dtype=torch.float32)
			subtree_t = torch.tensor(subtree_g, dtype=torch.float32)
			if max_dist > 0:
				dist_norm = dist_t / float(max_dist)
			else:
				dist_norm = dist_t
			if max_sub > 0:
				subtree_norm = subtree_t / float(max_sub)
			else:
				subtree_norm = subtree_t

			root_prox[start:end] = 1.0 - dist_norm
			subtree[start:end] = subtree_norm

			deg_slice = deg_cpu[start:end]
			max_deg = float(deg_slice.max().item()) if deg_slice.numel() else 0.0
			if max_deg > 0:
				degree_norm[start:end] = deg_slice / max_deg
			else:
				degree_norm[start:end] = 0.0

		struct = torch.stack([root_prox, subtree, degree_norm], dim=1)
		return struct.to(device=device, dtype=dtype)
