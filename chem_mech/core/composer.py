from typing import List
from .loader import InternalGraph
import numpy as np

def compose(graphs: List[InternalGraph]) -> InternalGraph:
    # 将多个图按序号偏移合并为一个 CompositeGraph
    atom_offsets = np.cumsum([0] + [g.n_atoms for g in graphs[:-1]]).tolist()
    xs, edges, adjs, globals_ = [], [], [], []
    total_n = sum(g.n_atoms for g in graphs)
    # 合并 atom feature
    xs = np.vstack([g.x for g in graphs])
    # 合并 adjacency: block-diagonal
    adj = np.zeros((total_n, total_n), dtype=int)
    for off, g in zip(atom_offsets, graphs):
        idx = slice(off, off + g.n_atoms)
        adj[idx, idx] = g.adj
    # 合并 global
    global_vec = np.concatenate([g.global_feat for g in graphs], axis=0)
    # 返回新的 InternalGraph
    return InternalGraph(atom_matrix=xs, bond_matrix=np.array([]),
                         adjacency_matrix=adj, global_vector=global_vec)