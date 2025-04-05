import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, Descriptors
from sklearn.preprocessing import OneHotEncoder
import time

# ------------------ 基本设置 ------------------
atom_types = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'N', 'Na', 'O', 'P', 'S']
hybridizations = ['S','SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'SP3D3']

electronegativity_dict = {
    'Br': 2.96, 'C': 2.55, 'Cl': 3.16, 'F': 3.98, 'H': 2.20,
    'I': 2.66, 'K': 0.82, 'N': 3.04, 'Na': 0.93, 'O': 3.44,
    'P': 2.19, 'S': 2.58
}

functional_groups = {
    'C': 'C',
    'F': 'F',
    'Cl': 'Cl',
    'Br': 'Br',
    'I': 'I',
    'C=C': 'C=C',
    'C#C': 'C#C',
    'O': 'O',
    'C(=O)O': 'C(=O)O',
    'C(=O)': 'C(=O)',
    'N': 'N',
    'N(=O)O': 'N(=O)O',
    'S(=O)(=O)O': 'S(=O)(=O)O',
    'C#N': 'C#N',
    'C(=O)N': 'C(=O)N',
    'S': 'S',
    'S(=O)N': 'S(=O)N'
}

# ------------------ OneHotEncoder 处理 ------------------
encoder_atom = OneHotEncoder(sparse_output=False)
encoder_hybrid = OneHotEncoder(sparse_output=False)

encoded_atoms = encoder_atom.fit_transform(np.array(atom_types).reshape(-1, 1))
encoded_hybridizations = encoder_hybrid.fit_transform(np.array(hybridizations).reshape(-1, 1))


# ------------------ 官能团特征函数 ------------------
def get_functional_group_features(smiles, atom_idx, functional_groups, max_distance=5):
    """
    返回目标原子在各官能团上的特征向量，每个官能团贡献三个维度：
      1. 是否包含该官能团（0或1）
      2. 目标原子到该官能团的最小距离（若无匹配则为max_distance）
      3. 在该最小距离下，该官能团实例的数量
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    fg_vector = []
    for fg, pattern in functional_groups.items():
        fg_mol = Chem.MolFromSmarts(pattern)
        matches = mol.GetSubstructMatches(fg_mol)
        in_group = any(atom_idx in match for match in matches)
        fg_vector.append(int(in_group))

        if matches:
            match_distances = []
            for match in matches:
                distances = []
                for idx in match:
                    if atom_idx == idx:
                        distances.append(0)
                    else:
                        path = rdmolops.GetShortestPath(mol, atom_idx, idx)
                        distances.append(len(path) - 1)
                match_distances.append(min(distances))
            global_min = min(match_distances)
            global_min = min(global_min, max_distance)
        else:
            global_min = max_distance
            match_distances = []
        fg_vector.append(global_min)

        if match_distances:
            count_eq = sum(1 for d in match_distances if d == global_min)
        else:
            count_eq = 0
        fg_vector.append(count_eq)

    return np.array(fg_vector)


# ------------------ 原子类别与杂化编码函数 ------------------
def get_atom_encoding(atom_symbol):
    try:
        idx = atom_types.index(atom_symbol)
        return encoded_atoms[idx]
    except ValueError:
        raise ValueError(f"Unsupported atom type: {atom_symbol}")


def get_hybridization_encoding(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    hybridization = str(atom.GetHybridization())  # 如 'SP3'
    try:
        idx = hybridizations.index(hybridization)
        return encoded_hybridizations[idx]
    except ValueError:
        raise ValueError(f"Unsupported hybridization type: {hybridization}")


# ------------------ 电负性、Gasteiger电荷、价态函数 ------------------
def get_electronegativity(atom_symbol):
    return electronegativity_dict.get(atom_symbol, None)


def get_gasteiger_charge(mol, atom_idx):
    AllChem.ComputeGasteigerCharges(mol)
    atom = mol.GetAtomWithIdx(atom_idx)
    return float(atom.GetProp('_GasteigerCharge'))


def get_valency(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetTotalValence()


# ------------------ 环和芳香环境特征函数 ------------------
def get_ring_feature(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return int(atom.IsInRing())


def get_aromatic_feature(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return int(atom.GetIsAromatic())


def get_nearest_aromatic_distance(mol, atom_idx, max_distance=5):
    aromatic_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not aromatic_indices:
        return max_distance
    distances = []
    for idx in aromatic_indices:
        if idx == atom_idx:
            distances.append(0)
        else:
            path = rdmolops.GetShortestPath(mol, atom_idx, idx)
            distances.append(len(path) - 1)
    return min(distances) if distances else max_distance


def get_aromatic_count(mol, atom_idx, max_distance=5):
    aromatic_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    count = 0
    for idx in aromatic_indices:
        if idx == atom_idx:
            dist = 0
        else:
            path = rdmolops.GetShortestPath(mol, atom_idx, idx)
            dist = len(path) - 1
        if dist <= max_distance:
            count += 1
    return count


# ------------------ 综合特征向量函数 ------------------
def get_atom_feature_vector(smiles, atom_idx, functional_groups, max_distance=5):
    """
    根据分子的 SMILES 和目标原子索引返回其综合特征向量，包含：
      1. 原子类别独热编码 (12维)
      2. 杂化方式独热编码 (7维)
      3. 电负性 (1维)
      4. Gasteiger电荷 (1维)
      5. 官能团特征（每个官能团3维，总共17*3=51维）
      6. 是否在环上 (1维)
      7. 是否处于芳香环境 (1维)
      8. 最近芳香原子距离 (1维)
      9. max_distance内芳香原子数量 (1维)

    最终特征向量维度 = 12+7+1+1+51+1+1+1+1 = 76
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    atom = mol.GetAtomWithIdx(atom_idx)
    atom_symbol = atom.GetSymbol()
    atom_encoding = get_atom_encoding(atom_symbol)
    hybrid_encoding = get_hybridization_encoding(mol, atom_idx)

    eneg = get_electronegativity(atom_symbol)
    if eneg is None:
        raise ValueError(f"Electronegativity for {atom_symbol} is not defined.")
    eneg_feature = np.array([eneg])

    charge = get_gasteiger_charge(mol, atom_idx)
    charge_feature = np.array([charge])

    fg_feature = get_functional_group_features(smiles, atom_idx, functional_groups, max_distance)

    ring_feat = np.array([get_ring_feature(mol, atom_idx)])
    aromatic_feat = np.array([get_aromatic_feature(mol, atom_idx)])
    nearest_arom_dist = np.array([get_nearest_aromatic_distance(mol, atom_idx, max_distance)])
    arom_count = np.array([get_aromatic_count(mol, atom_idx, max_distance)])

    feature_vector = np.concatenate([
        atom_encoding,
        hybrid_encoding,
        eneg_feature,
        charge_feature,
        fg_feature,
        ring_feat,
        aromatic_feat,
        nearest_arom_dist,
        arom_count
    ])
    return feature_vector


def get_molecule_feature_matrix(smiles, functional_groups, max_distance=5):
    """
    对于给定的 SMILES，遍历所有原子，返回一个矩阵，
    每一行为一个原子的综合特征向量（维度76）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    num_atoms = mol.GetNumAtoms()
    feature_matrix = []
    for i in range(num_atoms):
        fv = get_atom_feature_vector(smiles, i, functional_groups, max_distance)
        feature_matrix.append(fv)
    return np.array(feature_matrix)


# ------------------ 构建特征向量的解释标签 ------------------
# 1. 原子类别（12维）：依次对应 atom_types 列表
labels_atom = [f"原子类别 - {atom}" for atom in atom_types]
# 2. 杂化方式（7维）：依次对应 hybridizations 列表
labels_hybrid = [f"杂化方式 - {hyb}" for hyb in hybridizations]
# 3. 电负性（1维）
label_eneg = ["电负性"]
# 4. Gasteiger电荷（1维）
label_charge = ["Gasteiger电荷"]
# 5. 官能团特征（每个官能团3维，共17*3=51维）
labels_fg = []
for fg in functional_groups:
    labels_fg.append(f"官能团 '{fg}' 是否存在 (0/1)")
    labels_fg.append(f"目标原子到官能团 '{fg}' 的最短距离")
    labels_fg.append(f"在该最短距离下官能团 '{fg}' 的匹配数量")
# 6. 是否在环上（1维）
label_ring = ["是否在环上 (1:在, 0:不在)"]
# 7. 是否处于芳香环境（1维）
label_arom = ["是否处于芳香环境 (1:是, 0:否)"]
# 8. 距离最近芳香原子的距离（1维）
label_nearest_arom = ["目标原子到最近芳香原子的距离"]
# 9. max_distance内芳香原子数量（1维）
label_arom_count = ["max_distance内芳香原子数量"]

feature_labels = (labels_atom + labels_hybrid + label_eneg + label_charge +
                  labels_fg + label_ring + label_arom + label_nearest_arom + label_arom_count)

def Atom_Descriptors_test(mol_feature_matrix):
    print("分子特征矩阵 (每一行对应一个原子的特征向量):")
    num_atoms = mol_feature_matrix.shape[0]
    for i in range(num_atoms):
        print(f"原子 {i + 1} 的特征向量：")
        for j, (label, value) in enumerate(zip(feature_labels, mol_feature_matrix[i]), start=1):
            print(f"  维度 {j:2d}: {value:12.8f}  --> {label}")
        print("------------------------------------------------------")


if __name__ == "__main__":
    # ------------------ 示例测试 ------------------

    smiles = "[Na+]"
    mol_feature_matrix = get_molecule_feature_matrix(smiles, functional_groups, max_distance=5)
    Atom_Descriptors_test(mol_feature_matrix)



