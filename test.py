"""
SN1 / SN2 机理判断原型
---------------------------------------
* 仅做演示用途：依赖 RDKit (2023.09 及以上)
* 输入反应需 **带原子映射的反应 SMILES**，否则无法准确跟踪键变。
* 判断逻辑：
  1. 利用原子映射对比反应物/产物的键邻接表，找出
     - 断开的键 (central_C ↠ LG)
     - 新形成的键 (central_C ↠ Nu)
  2. 如果断/成键模式符合 SN1/SN2 拟合：
        central_C 为 sp3 C，且 仅有一对 (break, form)
  3. 根据 central_C 在反应物中的取代度 (primary/secondary/tertiary)
        * 三级或带共轭稳定 → SN1
        * 其余 → SN2

后续可在 classify_reaction 中插入更精细的立体化学检查、速率常数筛选等。
"""

from rdkit import Chem
from typing import List, Tuple, Dict, Set

# ----------------------- 工具函数 -----------------------

def parse_reaction_smiles(rxn_smiles: str) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
    """把带映射的反应 SMILES 拆分成 reactant / product 分子列表"""
    try:
        reactant_part, product_part = rxn_smiles.split(">>")
    except ValueError:
        raise ValueError("反应 SMILES 必须包含 '>>' 分隔符")
    reactants = [Chem.MolFromSmiles(smi) for smi in reactant_part.split('.') if smi]
    products  = [Chem.MolFromSmiles(smi) for smi in product_part.split('.') if smi]
    if None in reactants + products:
        raise ValueError("SMILES 解析失败，检查映射或语法错误")
    return reactants, products


def build_neighbor_map(mols: List[Chem.Mol]) -> Dict[int, Set[Tuple[int, int]]]:
    """返回 {atom_map : {(nbr_map, bond_order)}}，跨所有分子合并"""
    neighbor_map: Dict[int, Set[Tuple[int, int]]] = {}

    for mol in mols:
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            if not (m1 and m2):
                continue  # 未映射原子跳过
            order = int(bond.GetBondTypeAsDouble())  # 单键=1, 双键=2...
            neighbor_map.setdefault(m1, set()).add((m2, order))
            neighbor_map.setdefault(m2, set()).add((m1, order))
    return neighbor_map


def carbon_substitution_degree(atom: Chem.Atom) -> int:
    """统计碳原子在反应物中的取代度 (连了几个非氢原子)"""
    return sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() != 'H')

# ------------------ 机理判定核心 ------------------

LEAVING_GROUPS = {"Cl", "Br", "I", "F"}


def classify_reaction(rxn_smiles: str) -> str:
    """粗略判断反应属于 SN1 / SN2 / Unknown"""
    reactants, products = parse_reaction_smiles(rxn_smiles)

    nbr_map_R = build_neighbor_map(reactants)
    nbr_map_P = build_neighbor_map(products)

    # 找到邻接表发生变化的原子
    changed_atoms = [map_idx for map_idx in set(nbr_map_R) | set(nbr_map_P)
                     if nbr_map_R.get(map_idx, set()) != nbr_map_P.get(map_idx, set())]

    # SN1/SN2 模式只考虑恰有三个关键原子：central_C、LG、Nu
    if len(changed_atoms) != 3:
        return "Unknown"

    # 识别 central_C：它同时在 R 中失去一个键且在 P 中获得一个键
    central_candidates = []
    for idx in changed_atoms:
        lost = nbr_map_R.get(idx, set()) - nbr_map_P.get(idx, set())
        gained = nbr_map_P.get(idx, set()) - nbr_map_R.get(idx, set())
        if lost and gained:  # 同时断又成
            central_candidates.append(idx)
    if len(central_candidates) != 1:
        return "Unknown"
    c_center = central_candidates[0]

    # 确定 LG 与 Nu
    lost_partner = next(iter(nbr_map_R[c_center] - nbr_map_P.get(c_center, set())))[0]
    gained_partner = next(iter(nbr_map_P[c_center] - nbr_map_R.get(c_center, set())))[0]

    # 取出对应 RDKit Atom 以判断元素
    map2atom_R: Dict[int, Chem.Atom] = {atom.GetAtomMapNum(): atom for mol in reactants for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
    map2atom_P: Dict[int, Chem.Atom] = {atom.GetAtomMapNum(): atom for mol in products  for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
    atom_center = map2atom_R[c_center]
    atom_LG     = map2atom_R.get(lost_partner, map2atom_P.get(lost_partner))
    atom_Nu     = map2atom_P.get(gained_partner, map2atom_R.get(gained_partner))

    if atom_center.GetSymbol() != 'C':
        return "Unknown"  # 这里只演示典型碳中心

    # Leaving group 必须是卤素或类似伞状负离子
    if atom_LG.GetSymbol() not in LEAVING_GROUPS:
        return "Unknown"

    # 判断 SN1 / SN2：简单用取代度 >=3 判 SN1，否则 SN2
    degree = carbon_substitution_degree(atom_center)
    return "SN1" if degree >= 3 else "SN2"

# ----------------------- 简易测试 -----------------------

if __name__ == "__main__":
    # 甲基氯 + 氢氧根 → SN2
    rxn_sn2 = "[CH3:2][Cl:1].[OH-:3]>>[CH3:2][OH:3].[Cl-:1]"
    print("SN2 示例判断:", classify_reaction(rxn_sn2))

    # 三甲基碘 + 水 → SN1
    rxn_sn1 = "[CH3:4][C:2]([CH3:5])([CH3:6])[I:1].[OH2:3]>>[CH3:4][C:2]([CH3:5])([CH3:6])[OH:3].[I-:1]"
    print("SN1 示例判断:", classify_reaction(rxn_sn1))

    reaction= "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][NH:11][CH2:16][CH2:17]1.[CH3:18][OH:19].[CH2:12]([CH:13]1[CH2:14][O:15]1)[Cl:20]>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][NH+:11]([CH2:12][CH:13]2[CH2:14][O:15]2)[CH2:16][CH2:17]1.[CH3:18][OH:19].[Cl-:20]"
    print("示例判断:", classify_reaction(reaction))


