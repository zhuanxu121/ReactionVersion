from typing import Dict, Any
import numpy as np

class InternalGraph:
    def __init__(self, atom_matrix: np.ndarray, bond_matrix: np.ndarray,
                 adjacency_matrix: np.ndarray, global_vector: np.ndarray):
        self.x = atom_matrix                # (n_atoms, feat_dim)
        self.edge_attr = bond_matrix        # (n_bonds, feat_dim)
        self.adj = adjacency_matrix         # (n_atoms, n_atoms)
        self.global_feat = global_vector    # (global_dim,)
        self.n_atoms = atom_matrix.shape[0]


def to_internal(graph_dict: Dict[str, Any]) -> InternalGraph:
    return InternalGraph(
        atom_matrix=np.array(graph_dict['atom_matrix']),
        bond_matrix=np.array(graph_dict['bond_matrix']),
        adjacency_matrix=np.array(graph_dict['adjacency_matrix']),
        global_vector=np.array(graph_dict['global_vector']),
    )
