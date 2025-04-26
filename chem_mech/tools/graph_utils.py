
from rdkit.Chem import RWMol, Atom, BondType, SanitizeMol
from chem_mech.core.loader import InternalGraph
import numpy as np


def graph_to_rdkit(graph: InternalGraph) -> RWMol:
    """
    将 InternalGraph 转换为 RDKit RWMol 对象：
    - 假设 graph.x[:,0] 存原子序数（atomic number）
    - graph.adj 为邻接矩阵，1 表示有键（暂按单键处理）
    """
    n_atoms = graph.n_atoms
    mol = RWMol()

    # 1. 添加原子
    for i in range(n_atoms):
        # 原子序数存于特征矩阵第一列
        atomic_num = int(graph.x[i, 0])
        atom = Atom(atomic_num)
        mol.AddAtom(atom)

    # 2. 添加键
    adj = np.array(graph.adj)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if adj[i, j] == 1:
                # 默认单键
                mol.AddBond(i, j, BondType.SINGLE)

    # 3. Sanitize 并返回
    # 可能会自动设置杂化、芳香性等
    SanitizeMol(mol)
    return mol
