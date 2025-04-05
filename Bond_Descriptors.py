import numpy as np
from rdkit import Chem

def generate_adjacency_and_bond_descriptor_matrix(smiles):
    """
    输入：SMILES字符串
    输出：
      - 邻接矩阵 (numpy数组, shape: [num_atoms, num_atoms])
      - 键描述矩阵 (numpy数组, shape: [num_atoms, num_atoms, 8])
        每个非零位置的8维向量依次表示：
          [单键独热, 双键独热, 三键独热, 芳香键独热,
           是否在环上, 是否在芳香环上,
           环的最小元数（若存在，否则0）, 是否共轭]
    """
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    adjacency_matrix = np.zeros((num_atoms, num_atoms))
    bond_descriptor_matrix = np.zeros((num_atoms, num_atoms, 8))

    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()  # 返回所有环中包含的键的索引列表

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

        descriptor = []

        # 1-4. 键的类型（独热编码），顺序：[SINGLE, DOUBLE, TRIPLE, AROMATIC]
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            descriptor.extend([1, 0, 0, 0])
        elif bond_type == Chem.BondType.DOUBLE:
            descriptor.extend([0, 1, 0, 0])
        elif bond_type == Chem.BondType.TRIPLE:
            descriptor.extend([0, 0, 1, 0])
        elif bond_type == Chem.BondType.AROMATIC:
            descriptor.extend([0, 0, 0, 1])
        else:
            descriptor.extend([0, 0, 0, 0])

        # 5. 是否在环上
        in_ring = 1 if bond.IsInRing() else 0
        descriptor.append(in_ring)

        # 6. 是否在芳香环上（用 GetIsAromatic() 判断）
        in_aromatic_ring = 1 if bond.GetIsAromatic() else 0
        descriptor.append(in_aromatic_ring)

        # 7. 环的最小元数（若存在多个环，则取最小，否则为0）
        ring_sizes = []
        if bond.IsInRing():
            bond_idx = bond.GetIdx()
            for ring in bond_rings:
                if bond_idx in ring:
                    ring_sizes.append(len(ring))
            ring_size = min(ring_sizes) if ring_sizes else 0
        else:
            ring_size = 0
        descriptor.append(ring_size)

        # 8. 是否共轭
        is_conjugated = 1 if bond.GetIsConjugated() else 0
        descriptor.append(is_conjugated)

        bond_descriptor_matrix[i, j, :] = descriptor
        bond_descriptor_matrix[j, i, :] = descriptor

    return adjacency_matrix, bond_descriptor_matrix

def get_bonded_atoms_from_adjacency(adjacency_matrix):
    """
    遍历邻接矩阵上三角，记录存在键连接的原子对（编号从1开始）
    """
    bonded_atoms = []
    n_atoms = adjacency_matrix.shape[0]
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if adjacency_matrix[i, j] != 0:
                bonded_atoms.append([i+1, j+1])
    return bonded_atoms

# 定义每个维度的中文解释标签（共8维）
bond_descriptor_labels = [
    "单键独热编码",          # 第1维
    "双键独热编码",          # 第2维
    "三键独热编码",          # 第3维
    "芳香键独热编码",        # 第4维
    "是否在环上 (1:在, 0:不在)",  # 第5维
    "是否在芳香环上 (1:在, 0:不在)",  # 第6维
    "所在环的最小元数（若无环则为0）",  # 第7维
    "是否共轭 (1:是, 0:否)"   # 第8维
]

def Bond_Descriptors_test(adjacency_matrix,bond_descriptor_matrix):
    bonded_atoms = get_bonded_atoms_from_adjacency(adjacency_matrix)
    print("邻接矩阵:")
    print(adjacency_matrix)

    print("\n键描述矩阵 (存在键连接处的每个向量及其解释):")
    for pair in bonded_atoms:
        i, j = pair[0] - 1, pair[1] - 1
        descriptor = bond_descriptor_matrix[i, j, :]
        print(f"原子对 {pair} 的键描述向量：")
        for idx, (label, value) in enumerate(zip(bond_descriptor_labels, descriptor), start=1):
            print(f"  维度 {idx:2d}: {value:12.8f}  --> {label}")
        print("------------------------------------------------------")

    print("\n键连接的原子对 (1-based 编号):")
    print(bonded_atoms)


if __name__ == "__main__":
    # 示例测试
    smiles = 'ClCCCOc1ccc2ccccc2c1'
    adjacency_matrix, bond_descriptor_matrix = generate_adjacency_and_bond_descriptor_matrix(smiles)
    Bond_Descriptors_test(adjacency_matrix,bond_descriptor_matrix)




