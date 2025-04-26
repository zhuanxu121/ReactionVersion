from typing import List, Dict, Any, Tuple
from chem_mech.core.loader import InternalGraph
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from chem_mech.tools.graph_utils import graph_to_rdkit

class SN2Plugin:
    @staticmethod
    def apply_step(graph: InternalGraph,
                   core_idxs: Tuple[int, int, int]
                   ) -> List[Dict[str, Any]]:
        """
        给定 (Nu, C, LG) 三元组，依次生成：
          1) 结合中间体：Nu-C-LG 并保留所有原键
          2) 离去基脱离中间体：断 C-LG，C 带正电
          3) 最终产物：去除多余 H，恢复电中性
        返回：
          [
            {'smiles': "I1", 'intermediates': []},
            {'smiles': "I2", 'intermediates': []},
            {'smiles': "P", 'intermediates': []}
          ]
        实际实现中可调用 RDKit 构建 RWMol、AddBond/RemoveBond、SetFormalCharge 等。
        """
        nu_idx, c_idx, lg_idx = core_idxs
        # 将 InternalGraph 转回 RDKit Mol 的工具函数
        mol = graph_to_rdkit(graph)
        intermediates = []
        # 1) 结合中间体
        mol1 = Chem.RWMol(mol)
        mol1.AddBond(nu_idx, c_idx, Chem.BondType.SINGLE)
        smiles1 = Chem.MolToSmiles(mol1)
        intermediates.append(smiles1)
        # 2) 离去基脱除
        mol2 = Chem.RWMol(mol1)
        mol2.RemoveBond(c_idx, lg_idx)
        atom_c = mol2.GetAtomWithIdx(c_idx)
        atom_c.SetFormalCharge(1)
        smiles2 = Chem.MolToSmiles(mol2)
        intermediates.append(smiles2)
        # 3) 最终产物
        mol3 = Chem.RWMol(mol2)
        # 去掉一个 H
        for nbr in mol3.GetAtomWithIdx(nu_idx).GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                mol3.RemoveAtom(nbr.GetIdx())
                break
        atom_c.SetFormalCharge(0)
        smiles3 = Chem.MolToSmiles(mol3)
        return [
            {'smiles': smiles1, 'intermediates': []},
            {'smiles': smiles2, 'intermediates': []},
            {'smiles': smiles3, 'intermediates': []}
        ]