# data_preprocessing.py

import os
import csv
from rdkit import Chem
from rdkit.Chem import Draw
import math
import numpy as np
from Atom_Descriptors import get_molecule_feature_matrix,functional_groups,Atom_Descriptors_test
from Bond_Descriptors import generate_adjacency_and_bond_descriptor_matrix,Bond_Descriptors_test
from Global_Information_Vector import generate_global_vector,Global_Information_Vector_test


def load_and_preprocess_csv(input_file='Dataset/47083204_Trans_G2S_val.csv', batch_size=32):
    """
    读取 CSV 文件，解析其中的 rxn_smiles 字段，依据反应步骤构成完整反应序列，
    反应序列的每一步格式为 "SMILES1 >> SMILES2"。当遇到某一步中反应物与产物相同时，
    认为该反应序列结束。

    解析规则：
      - 每个步骤使用 ">>" 分隔左右。
      - 将连续步骤组成一个反应序列。
      - 整体反应物：取序列第一步的 left 部分。
      - 整体产物：取序列最后一步的 right 部分。
      - 中间体：取序列中除最后一步外每一步的 right 部分，
        但过滤掉在整体反应物或整体产物中出现的物质。

    :param input_file: CSV 文件路径
    :param batch_size: int, 后续批处理用（当前未用到）
    :return: 一个列表，每个元素为字典，包含 'reactants', 'intermediates', 'products'
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist!")

    reactions = []  # 存储所有完整反应序列
    current_chain = []  # 存储当前反应序列的每一步，格式为 {'left': [...], 'right': [...]}

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rxn_smiles = row['rxn_smiles'].strip()
            if ">>" not in rxn_smiles:
                print("Reaction SMILES missing '>>', skipping:", rxn_smiles)
                continue

            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                print("Unexpected reaction SMILES format, skipping:", rxn_smiles)
                continue

            left_str = parts[0].strip()
            right_str = parts[1].strip()

            # 用 '.' 分割各个分子，并过滤空字符串
            left_mols = [s.strip() for s in left_str.split('.') if s.strip()]
            right_mols = [s.strip() for s in right_str.split('.') if s.strip()]

            step = {'left': left_mols, 'right': right_mols}
            current_chain.append(step)

            # 如果本步反应物与产物完全相同，则认为反应序列结束
            if left_mols == right_mols:
                chain_data = process_chain(current_chain)
                reactions.append(chain_data)
                current_chain = []

    # 若文件结束时仍有未结束的反应序列，也可以加入（根据需要处理）
    if current_chain:
        chain_data = process_chain(current_chain)
        reactions.append(chain_data)

    # test_visualization(reactions)
    return reactions


def process_chain(chain):
    """
    根据一个反应序列（chain）构建整体反应数据，返回包含 'reactants', 'intermediates', 'products' 的字典。

    规则：
      - 整体反应物：取序列第一步的 left 部分。
      - 整体产物：取序列最后一步的 right 部分。
      - 中间体：取序列中除最后一步外所有步骤的 right 部分，
        然后过滤掉出现在反应物或产物中的物质。
    """
    if not chain:
        return None

    # 整体反应物
    overall_reactants = chain[0]['left']
    # 整体产物
    overall_products = chain[-1]['right']

    intermediates = []
    # 收集中间体：序列中除最后一步外所有步骤的 right 部分
    for step in chain[:-1]:
        intermediates.extend(step['right'])

    # 去重，但保留顺序
    intermediates = list(dict.fromkeys(intermediates))

    # 过滤：如果一个物质在反应物或产物中出现，则不作为中间体
    intermediates = [mol for mol in intermediates if (mol not in overall_reactants) and (mol not in overall_products)]

    return {
        'reactants': overall_reactants,
        'intermediates': intermediates,
        'products': overall_products
    }


def test_visualization(reactions, output_dir='mol_images'):
    """
    利用 RDKit 的 MolToFile 将 reactants、intermediates 和 products 分别绘制成 PNG 文件，
    保存在 output_dir 中，便于检查解析是否正确。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, reaction_data in enumerate(reactions):
        print(f"=== Reaction Sequence {idx + 1} ===")

        # 绘制反应物
        print("Reactants:", reaction_data['reactants'])
        for i, smi in enumerate(reaction_data['reactants']):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                png_path = os.path.join(output_dir, f"reaction{idx + 1}_reactant_{i}.png")
                Draw.MolToFile(mol, png_path, size=(300, 300))
                print(f"  -> Reactant image saved to {png_path}")
            else:
                print(f"  -> Failed to create molecule from reactant SMILES: {smi}")

        # 绘制中间体
        print("Intermediates:", reaction_data['intermediates'])
        for i, smi in enumerate(reaction_data['intermediates']):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                png_path = os.path.join(output_dir, f"reaction{idx + 1}_intermediate_{i}.png")
                Draw.MolToFile(mol, png_path, size=(300, 300))
                print(f"  -> Intermediate image saved to {png_path}")
            else:
                print(f"  -> Failed to create molecule from intermediate SMILES: {smi}")

        # 绘制产物
        print("Products:", reaction_data['products'])
        for i, smi in enumerate(reaction_data['products']):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                png_path = os.path.join(output_dir, f"reaction{idx + 1}_product_{i}.png")
                Draw.MolToFile(mol, png_path, size=(300, 300))
                print(f"  -> Product image saved to {png_path}")
            else:
                print(f"  -> Failed to create molecule from product SMILES: {smi}")

        print("====================================\n")


# 在 data_preprocessing.py 中添加如下代码

class ReactionBatchReader:
    """
    用于按批次读取 reactions 数据中的分子（这里取 reactants 内的 SMILES），
    并转换为分子图表示（包含原子描述矩阵、邻接矩阵、键描述矩阵、全局向量）。

    该类会记住当前的读取位置，每次调用 next_batch() 时返回一个 batch_size 的批次，
    直到数据读完（此时返回 None）。
    """

    def __init__(self, reactions, batch_size=16, functional_groups=None):
        """
        :param reactions: 反应序列列表，每个元素为字典 {'reactants', 'intermediates', 'products'}（通常由 load_and_preprocess_csv 得到）
        :param batch_size: 每批处理的分子数
        :param functional_groups: 官能团字典，供特征工程使用
        """
        self.reactions = reactions
        self.batch_size = batch_size
        self.functional_groups = functional_groups
        self.tasks = []
        # 此处仅以反应物作为处理对象，可根据需要扩展到中间体或产物
        for r in reactions:
            for s in r['reactants']:
                # 此处 reaction_mixture 与 analyte 作为示例参数，可根据实际情况调整
                self.tasks.append((s, ['CCCl', 'CCCBr', 'C1=CC=CC=C1', 'CC(=O)O'], 'CCCBr'))
        self.current_index = 0
        self.num_tasks = len(self.tasks)

    def next_batch(self):
        """
        返回下一个 batch 的分子图数据，每个元素为字典：
          {
              'smiles': str,
              'atom_matrix': np.ndarray or None,
              'bond_matrix': np.ndarray or None,
              'adjacency': np.ndarray or None,
              'global_vector': np.ndarray or None
          }
        如果已读取完所有数据，则返回 None。
        """
        if self.current_index >= self.num_tasks:
            return None  # 或者可选择重置 self.current_index=0，实现循环读取

        batch = self.tasks[self.current_index: self.current_index + self.batch_size]
        self.current_index += self.batch_size
        batch_results = []
        for (smi, reaction_mixture, analyte) in batch:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                batch_results.append({
                    'smiles': smi,
                    'atom_matrix': None,
                    'bond_matrix': None,
                    'adjacency': None,
                    'global_vector': None
                })
                continue
            # 调用你已实现的特征工程函数
            atom_matrix = get_molecule_feature_matrix(smi, self.functional_groups, max_distance=5)
            adjacency_matrix, bond_matrix = generate_adjacency_and_bond_descriptor_matrix(smi)
            global_vector = generate_global_vector(reaction_mixture, analyte, self.functional_groups)
            batch_results.append({
                'smiles': smi,
                'atom_matrix': atom_matrix,
                'bond_matrix': bond_matrix,
                'adjacency': adjacency_matrix,
                'global_vector': global_vector
            })
        return batch_results


if __name__ == "__main__":
    # 读取并分类反应数据
    parsed_reactions = load_and_preprocess_csv()
    print("Parsed reaction sequences count:", len(parsed_reactions))

    # 初始化批量读取器，取反应物中的 SMILES
    reader = ReactionBatchReader(parsed_reactions, batch_size=16, functional_groups=functional_groups)
    batch = reader.next_batch()
    if batch is None:
        print("没有更多数据")
    else:
        print(f"返回了一个批次，共包含 {len(batch)} 个分子图数据")
        for idx, item in enumerate(batch, start=1):
            print(f"\n----------- 分子 {idx}: {item['smiles']} -----------")

            # 调用全局向量测试函数
            if item['global_vector'] is not None:
                Global_Information_Vector_test(item['global_vector'])
            else:
                print("全局信息向量: None")

            # 调用邻接矩阵和键描述测试函数
            if (item['adjacency'] is not None) and (item['bond_matrix'] is not None):
                Bond_Descriptors_test(item['adjacency'], item['bond_matrix'])
            else:
                print("邻接矩阵/键描述矩阵: None")

            # 调用原子描述测试函数
            if item['atom_matrix'] is not None:
                Atom_Descriptors_test(item['atom_matrix'])
            else:
                print("原子描述矩阵: None")
        print("========== 测试结束 ==========")

# # 测试读取数据并分类反应物、中间体、生成物的部分
# if __name__ == "__main__":
#     reactions = load_and_preprocess_csv()
#     print("Parsed reaction sequences count:", len(reactions))
