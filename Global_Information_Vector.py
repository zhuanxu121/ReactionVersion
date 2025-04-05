import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# 定义官能团字典（SMARTS模式），注意“C(=O)O”和“C(=O)”均包含在内
functional_groups = {
    'C': 'C',
    'F': 'F',
    'Cl': 'Cl',
    'Br': 'Br',
    'I': 'I',
    'C=C': 'C=C',
    'C#C': 'C#C',
    'O': 'O',
    'C(=O)O': 'C(=O)O',  # 羧酸
    'C(=O)': 'C(=O)',  # 羰基
    'N': 'N',
    'N(=O)O': 'N(=O)O',
    'S(=O)(=O)O': 'S(=O)(=O)O',
    'C#N': 'C#N',
    'C(=O)N': 'C(=O)N',
    'S': 'S',
    'S(=O)N': 'S(=O)N'
}

labels = [
         "官能团 'C' 的计数",
         "官能团 'F' 的计数",
         "官能团 'Cl' 的计数",
         "官能团 'Br' 的计数",
         "官能团 'I' 的计数",
         "官能团 'C=C' 的计数",
         "官能团 'C#C' 的计数",
         "官能团 'O' 的计数",
         "官能团 'C(=O)O'（羧酸）的计数",
         "官能团 'C(=O)'（羰基，调整后）的计数",
         "官能团 'N' 的计数",
         "官能团 'N(=O)O' 的计数",
         "官能团 'S(=O)(=O)O' 的计数",
         "官能团 'C#N' 的计数",
         "官能团 'C(=O)N' 的计数",
         "官能团 'S' 的计数",
         "官能团 'S(=O)N' 的计数",
         "共轭键总数",
         "环总数",
         "环大小平均值",
         "环大小方差",
         "芳香环总数",
         "芳香环大小平均值",
         "芳香环大小方差",
         "芳香环比例（芳香环数/总环数）",
         "平均 pKa",
         "pKa 方差",
         "平均 TPSA",
         "TPSA 方差",
         "平均重原子数（每个分子）",
         "重原子数方差"
     ]


def compute_pKa(mol):
    """
    使用简单规则，根据分子中是否包含目标功能团计算pKa。
    如果检测到羧酸，认为pKa约为4.8；
    如果检测到苯酚，认为pKa约为10.0；
    如果检测到胺（非酰胺），认为pKa约为10.5；
    如果没有检测到任何，则返回默认7.0；
    如果有多个匹配，则返回平均值。
    """
    pKa_candidates = []

    # 羧酸：SMARTS '[CX3](=O)[OX2H1]'
    carboxylic_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if mol.HasSubstructMatch(carboxylic_pattern):
        matches = mol.GetSubstructMatches(carboxylic_pattern)
        pKa_candidates.extend([4.8] * len(matches))

    # 苯酚：SMARTS 'c1ccc(O)cc1'
    phenol_pattern = Chem.MolFromSmarts('c1ccc(O)cc1')
    if mol.HasSubstructMatch(phenol_pattern):
        matches = mol.GetSubstructMatches(phenol_pattern)
        pKa_candidates.extend([10.0] * len(matches))

    # 胺：SMARTS '[NX3;H2,H1;!$(NC=O)]'
    amine_pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
    if mol.HasSubstructMatch(amine_pattern):
        matches = mol.GetSubstructMatches(amine_pattern)
        pKa_candidates.extend([10.5] * len(matches))

    if pKa_candidates:
        return np.mean(pKa_candidates)
    else:
        return 7.0


def generate_global_vector(reaction_mixture, analyte, functional_groups):
    """
    输入：
      - reaction_mixture: 多个SMILES组成的列表，代表整个反应体系的反应物
      - analyte: 当前分析的反应物SMILES
      - functional_groups: 官能团字典（SMARTS模式）
    输出：
      一个全局向量，维度为31：
      前17维为官能团计数（排除当前分析物），后14维为其它统计特征：
        [total_conjugated_bonds,
         total_ring_count, avg_ring_size, var_ring_size,
         total_aromatic_ring_count, avg_aromatic_ring_size, var_aromatic_ring_size, aromatic_ring_ratio,
         avg_pKa, var_pKa,
         avg_TPSA, var_TPSA,
         avg_atom_count, var_atom_count]
    """
    # 将 analyte 转换为规范SMILES，用于比较
    analyte_mol = Chem.MolFromSmiles(analyte)
    analyte_canon = Chem.MolToSmiles(analyte_mol) if analyte_mol is not None else None

    # 初始化官能团计数字典
    fg_counts = {fg: 0 for fg in functional_groups}
    # 预先转换SMARTS为RDKit分子对象
    fg_patterns = {fg: Chem.MolFromSmarts(pattern) for fg, pattern in functional_groups.items()}

    # 其它全局统计
    pKa_list = []
    TPSA_list = []
    atom_count_list = []
    conjugated_bonds_list = []
    ring_sizes_all = []  # 全部环的尺寸
    aromatic_ring_sizes_all = []  # 全部芳香环尺寸

    total_ring_count = 0
    total_aromatic_ring_count = 0
    total_conjugated_bonds = 0

    # 遍历 reaction_mixture 中的每个 SMILES
    for smi in reaction_mixture:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        # 排除当前分析物
        if analyte_canon is not None and canon == analyte_canon:
            continue

        # 先处理官能团：对于 'C(=O)O' 和 'C(=O)'特殊处理
        carboxylic_pattern = Chem.MolFromSmarts('C(=O)O')
        carbonyl_pattern = Chem.MolFromSmarts('C(=O)')
        carboxylic_matches = mol.GetSubstructMatches(carboxylic_pattern)
        carbonyl_matches = mol.GetSubstructMatches(carbonyl_pattern)
        count_carboxylic = len(carboxylic_matches)
        count_carbonyl = len(carbonyl_matches)
        adjusted_carbonyl = max(0, count_carbonyl - count_carboxylic)
        fg_counts['C(=O)O'] += count_carboxylic
        fg_counts['C(=O)'] += adjusted_carbonyl

        # 对于其它官能团，直接匹配
        for fg, pattern in fg_patterns.items():
            if fg in ['C(=O)O', 'C(=O)']:
                continue
            matches = mol.GetSubstructMatches(pattern)
            fg_counts[fg] += len(matches)

        # 共轭键计数
        conjugated_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsConjugated())
        total_conjugated_bonds += conjugated_bonds
        conjugated_bonds_list.append(conjugated_bonds)

        # 环信息
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()  # 返回所有环（原子索引元组）
        num_rings = len(rings)
        total_ring_count += num_rings
        for ring in rings:
            ring_sizes_all.append(len(ring))

        # 芳香环信息
        num_aromatic_rings = 0
        for ring in rings:
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                num_aromatic_rings += 1
                aromatic_ring_sizes_all.append(len(ring))
        total_aromatic_ring_count += num_aromatic_rings

        # pKa：使用改进的规则计算
        pKa_value = compute_pKa(mol)
        pKa_list.append(pKa_value)

        # TPSA：极性表面积
        tpsa_value = Descriptors.TPSA(mol)
        TPSA_list.append(tpsa_value)

        # 分子大小（仅重原子数）
        atom_count = mol.GetNumAtoms()
        atom_count_list.append(atom_count)

    # 辅助函数：计算平均值和方差
    def avg_var(lst):
        if len(lst) == 0:
            return 0.0, 0.0
        arr = np.array(lst, dtype=float)
        return float(np.mean(arr)), float(np.var(arr))

    avg_pKa, var_pKa = avg_var(pKa_list)
    avg_TPSA, var_TPSA = avg_var(TPSA_list)
    avg_atom_count, var_atom_count = avg_var(atom_count_list)

    # 环尺寸统计
    if len(ring_sizes_all) > 0:
        avg_ring_size = float(np.mean(ring_sizes_all))
        var_ring_size = float(np.var(ring_sizes_all))
    else:
        avg_ring_size, var_ring_size = 0.0, 0.0

    if len(aromatic_ring_sizes_all) > 0:
        avg_aromatic_ring_size = float(np.mean(aromatic_ring_sizes_all))
        var_aromatic_ring_size = float(np.var(aromatic_ring_sizes_all))
    else:
        avg_aromatic_ring_size, var_aromatic_ring_size = 0.0, 0.0

    aromatic_ring_ratio = (total_aromatic_ring_count / total_ring_count) if total_ring_count > 0 else 0.0

    # 构造全局向量
    # 官能团部分：顺序按照 functional_groups 的顺序（17维）
    fg_vector = np.array([fg_counts[fg] for fg in functional_groups])

    # 其它统计特征（14维）：
    other_features = np.array([
        total_conjugated_bonds,
        total_ring_count, avg_ring_size, var_ring_size,
        total_aromatic_ring_count, avg_aromatic_ring_size, var_aromatic_ring_size, aromatic_ring_ratio,
        avg_pKa, var_pKa,
        avg_TPSA, var_TPSA,
        avg_atom_count, var_atom_count
    ])

    # 拼接为一个全局向量（维度 17+14 = 31）
    global_vector = np.concatenate([fg_vector, other_features])
    return global_vector

def Global_Information_Vector_test(global_vector):
    print("全局信息向量：")
    for i, (label, value) in enumerate(zip(labels, global_vector), start=1):
        print(f"维度 {i:2d}: {value:12.8f}   --> {label}")


if __name__ == "__main__":
     # 示例测试
     reaction_mixture = ['CCCl', 'CCCBr', 'C1=CC=CC=C1', 'CC(=O)O']  # 反应物集合
     analyte = 'CCCBr'  # 当前分析的反应物

     global_vector = generate_global_vector(reaction_mixture, analyte, functional_groups)
     Global_Information_Vector_test(global_vector)




