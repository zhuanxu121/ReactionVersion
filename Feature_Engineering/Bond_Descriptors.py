import numpy as np
from rdkit import Chem


def generate_adjacency_and_bond_descriptor_matrix(smiles):
    """
    生成邻接矩阵和键描述矩阵，矩阵只包含实际存在的键描述
    :param smiles: SMILES 表示的分子
    :return: 邻接矩阵和键描述矩阵（稀疏形式）
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    # 获取邻接矩阵（稀疏存储形式：只存储实际连接的键对）
    adjacency_matrix = []
    bond_descriptor_matrix = []

    # 获取分子的环信息
    bond_rings = mol.GetRingInfo().AtomRings()  # 获取所有环信息

    # 遍历分子中的所有键，并生成描述矩阵
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # 确保邻接矩阵中有连接
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

        # 只为实际存在的键生成描述，添加到矩阵
        bond_descriptor_matrix.append(descriptor)

        # 记录实际连接的键对
        adjacency_matrix.append((i, j))

    return adjacency_matrix, bond_descriptor_matrix


def Bond_Descriptors_test(adjacency_matrix, bond_descriptor_matrix):
    print("邻接矩阵 (稀疏表示):")
    print(adjacency_matrix)

    print("\n键描述矩阵 (存在键连接处的每个向量及其解释):")
    for idx, descriptor in enumerate(bond_descriptor_matrix):
        i, j = adjacency_matrix[idx]  # 取对应的原子对
        print(f"原子对 {i + 1}-{j + 1} 的键描述向量：")  # 1-based 编号
        for idx, (label, value) in enumerate(zip(bond_descriptor_labels, descriptor), start=1):
            print(f"  维度 {idx:2d}: {value:12.8f}  --> {label}")
        print("------------------------------------------------------")

    print("\n键连接的原子对 (1-based 编号):")
    bonded_atoms = [(pair[0] + 1, pair[1] + 1) for pair in adjacency_matrix]
    print(bonded_atoms)


# 定义每个维度的中文解释标签（共8维）
bond_descriptor_labels = [
    "单键独热编码",  # 第1维
    "双键独热编码",  # 第2维
    "三键独热编码",  # 第3维
    "芳香键独热编码",  # 第4维
    "是否在环上 (1:在, 0:不在)",  # 第5维
    "是否在芳香环上 (1:在, 0:不在)",  # 第6维
    "所在环的最小元数（若无环则为0）",  # 第7维
    "是否共轭 (1:是, 0:否)"  # 第8维
]

if __name__ == "__main__":
    # 示例测试
    smiles = 'ClCC1CO1'  # 例如甲醇的SMILES
    adjacency_matrix, bond_descriptor_matrix = generate_adjacency_and_bond_descriptor_matrix(smiles)
    Bond_Descriptors_test(adjacency_matrix, bond_descriptor_matrix)
