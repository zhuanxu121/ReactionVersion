import numpy as np
from Feature_Engineering.data_preprocessing import load_and_preprocess_csv, ReactionBatchReader,add_random_features
from Feature_Engineering.Atom_Descriptors import functional_groups


def merge_adjacency_matrices(adj_matrices, atom_counts, atom_feature_matrices):
    """
    合并邻接矩阵并将其转换为标准邻接矩阵（原子数 * 原子数的二维矩阵）。
    同时根据原子特征矩阵的后13个特征决定是否新增连接。

    :param adj_matrices: 反应物的邻接矩阵列表，每个元素为一个稀疏邻接矩阵 [(i, j), ...]
    :param atom_counts: 每个反应物的原子数列表，用于调整原子序号
    :param atom_feature_matrices: 每个反应物的原子特征矩阵列表，每个元素为一个原子特征矩阵
    :return: 转换后的完整邻接矩阵（二维矩阵）
    """
    total_atoms = sum(atom_counts)  # 总原子数
    adjacency_matrix = np.zeros((total_atoms, total_atoms))  # 初始化一个全零的邻接矩阵

    current_offset = 0  # 当前反应物的偏移量，用于更新原子序号

    # 遍历每个反应物的邻接矩阵
    for i, adj_matrix in enumerate(adj_matrices):
        # 对于每个边，调整原子序号
        for (start, end) in adj_matrix:
            # 调整原子序号
            adjusted_start = start + current_offset
            adjusted_end = end + current_offset
            # 设置邻接矩阵中的值为 1，表示存在边
            adjacency_matrix[adjusted_start, adjusted_end] = 1
            adjacency_matrix[adjusted_end, adjusted_start] = 1  # 假设邻接矩阵是对称的

        # 遍历原子特征矩阵的后13个特征，检查是否需要增加边
        atom_features = atom_feature_matrices[i]  # 获取当前反应物的原子特征矩阵
        num_atoms = atom_features.shape[0]  # 当前反应物的原子数

        for a in range(num_atoms):
            for b in range(a + 1, num_atoms):  # 只检查一半，避免重复
                # 获取后13个特征
                feature_a = atom_features[a, -13:]
                feature_b = atom_features[b, -13:]

                # 如果两个原子的后13个特征中有相同位置的特征非零，则连接
                if np.any(feature_a * feature_b != 0):  # 特征不为零且有匹配的地方
                    adjusted_a = a + current_offset
                    adjusted_b = b + current_offset
                    adjacency_matrix[adjusted_a, adjusted_b] = 1
                    adjacency_matrix[adjusted_b, adjusted_a] = 1  # 对称矩阵

        # 更新偏移量
        current_offset += atom_counts[i]

    return adjacency_matrix


def build_bond_feature_matrix(adj_matrices, bond_feature_matrices, atom_counts, atom_feature_matrices):
    """
    根据邻接矩阵和键特征矩阵构建新的键特征矩阵，并添加一个维度：是否实际连接。

    :param adj_matrices: 反应物的邻接矩阵列表，每个元素为一个稀疏邻接矩阵 [(i, j), ...]
    :param bond_feature_matrices: 反应物的键特征矩阵列表
    :param atom_counts: 每个反应物的原子数列表，用于调整原子序号
    :param atom_feature_matrices: 每个反应物的原子特征矩阵列表
    :return: 新的键特征矩阵（二维矩阵，包含是否实际连接的维度）
    """
    total_atoms = sum(atom_counts)  # 总原子数
    total_bonds = 0  # 总键数

    # 计算总的键数
    for adj_matrix in adj_matrices:
        total_bonds += len(adj_matrix)  # 统计当前反应物的边数

    # 初始化键特征矩阵（包含是否实际连接维度）
    new_bond_feature_matrix = np.zeros((total_bonds, len(bond_feature_matrices[0][0]) + 1))  # 增加一个维度来表示是否实际连接
    current_bond_index = 0  # 当前键的索引

    current_offset = 0  # 当前反应物的偏移量，用于更新原子序号

    # 遍历每个反应物的邻接矩阵
    for i, adj_matrix in enumerate(adj_matrices):
        # 遍历当前反应物的所有键
        bond_index=0
        for (start, end) in adj_matrix:
            adjusted_start = start + current_offset
            adjusted_end = end + current_offset
            bond_features = bond_feature_matrices[i][bond_index]

            # 判断是否是新添加的键（根据原子特征矩阵的后13个特征决定连接）
            atom_features_a = atom_feature_matrices[i][start, -13:]  # 当前原子特征
            atom_features_b = atom_feature_matrices[i][end, -13:]  # 目标原子特征

            # 如果是根据后13个特征决定的连接，则所有维度全为0
            if np.any(atom_features_a * atom_features_b != 0):
                new_bond_feature_matrix[current_bond_index, :-1] = 0  # 其余维度为0
                new_bond_feature_matrix[current_bond_index, -1] = 0  # 连接标记为0
            else:
                # 如果原本就有这个键，则在最后一维标记为1
                new_bond_feature_matrix[current_bond_index, :-1] = bond_features  # 保留原有的键特征
                new_bond_feature_matrix[current_bond_index, -1] = 1  # 连接标记为1

            # 更新键的索引
            current_bond_index += 1
            bond_index+=1

        # 更新偏移量
        current_offset += atom_counts[i]

    return new_bond_feature_matrix


def graph_processing(reaction):
    """
    处理单个 reaction 的反应物图列表，合成该 reaction 的综合图，
    包括整合后的原子矩阵、邻接矩阵、键特征矩阵以及全局向量。

    :param reaction: 一个反应方程式的反应物图列表，每个图是一个字典，
                     包含 'atom_matrix'、'adjacency_matrix'、'bond_matrix'、'global_vector'、'smiles' 等键
    :return: 处理后的 reaction 字典，包含合并后的 'smiles'、'atom_matrix'、
             'adjacency_matrix'、'bond_feature_matrix'、'global_vector'
    """
    Atom_matrix = []
    Adjacency_matrix = []  # 用于存储每个反应物的邻接矩阵
    atom_counts = []  # 用于记录每个反应物的原子数
    atom_feature_matrices = []  # 用于记录每个反应物的原子特征矩阵
    bond_feature_matrices = []  # 用于存储每个反应物的键特征矩阵
    global_vectors = []  # 用于存储全局向量
    result_smi = ""  # 初始化反应物的 SMILES 拼接字符串

    # 遍历当前 reaction 内的每个反应物图
    for reactant in reaction:
        Atom_matrix.append(reactant["atom_matrix"])
        Adjacency_matrix.append(reactant["adjacency_matrix"])  # 获取每个反应物的邻接矩阵
        atom_counts.append(reactant["atom_matrix"].shape[0])  # 获取每个反应物的原子数
        atom_feature_matrices.append(reactant["atom_matrix"])  # 获取每个反应物的原子特征矩阵
        bond_feature_matrices.append(reactant["bond_matrix"])  # 获取每个反应物的键特征矩阵

        # 获取全局向量，假设全局向量存储在每个反应物的 'global_vector' 键中
        if not global_vectors:  # 只保留第一个反应物的全局向量
            global_vectors.append(reactant["global_vector"])

        # 拼接每个反应物的 SMILES 表示
        if result_smi:  # 如果已经有 SMILES，则用 '.' 连接
            result_smi += "." + reactant["smiles"]
        else:
            result_smi = reactant["smiles"]

    # 合并所有反应物的原子矩阵（上下拼接）
    result_Atom_matrix = np.vstack(Atom_matrix)

    # 合并邻接矩阵并转换为标准邻接矩阵（原子数×原子数的二维矩阵）
    result_Adjacency_matrix = merge_adjacency_matrices(Adjacency_matrix, atom_counts, atom_feature_matrices)

    # 根据邻接矩阵和键特征矩阵构建新的键特征矩阵
    result_Bond_feature_matrix = build_bond_feature_matrix(Adjacency_matrix, bond_feature_matrices, atom_counts,
                                                           atom_feature_matrices)

    # 假设各个反应物的全局向量一致，直接取第一个反应物的全局向量
    result_Global_vector = global_vectors[0]

    # 构建处理后的 reaction 结果字典
    result_reaction = {
        'smiles': result_smi,
        'atom_matrix': result_Atom_matrix,
        'adjacency_matrix': result_Adjacency_matrix,
        'bond_feature_matrix': result_Bond_feature_matrix,
        'global_vector': result_Global_vector
    }

    return result_reaction


def Graph_Processing_test():
    parsed_reactions = load_and_preprocess_csv()
    print("Parsed reaction sequences count:", len(parsed_reactions))

    # 初始化批量读取器，取反应物中的 SMILES
    reader = ReactionBatchReader(parsed_reactions, batch_size=1, functional_groups=functional_groups)
    batch = reader.next_batch()
    if batch is None:
        print("没有更多数据")
    else:
        print(f"返回了一个批次，共包含 {len(batch)} 个反应方程式数据")

        # 遍历批次中的每个反应方程式
        for batch_idx, reaction_data in enumerate(batch, start=1):

            # 遍历当前反应方程式中的每个反应物
            for reactant_idx, item in enumerate(reaction_data, start=1):

                # 添加新的13维随机特征
                if item['atom_matrix'] is not None:
                    updated_atom_matrix = add_random_features(item['atom_matrix'])
                    item['atom_matrix'] = updated_atom_matrix
                    print(f"Updated Atom Matrix: {updated_atom_matrix.shape}")



        print("====================================\n")
    processed_batch = graph_processing(batch)
    print("Processed batch:")
    print(processed_batch)

    # 测试功能：输出每个 atom_matrix 和 adjacency_matrix 的形状
    for i, reaction in enumerate(processed_batch):
        for j, result in enumerate(reaction):
            atom_matrix = result['atom_matrix']
            adjacency_matrix = result['adjacency_matrix']
            bond_feature_matrix = result['bond_feature_matrix']  # 获取键特征矩阵
            global_vector= result['global_vector']

            # 获取并打印 atom_matrix 和 adjacency_matrix 的形状
            rows, cols = atom_matrix.shape
            adj_rows, adj_cols = adjacency_matrix.shape
            bond_rows, bond_cols = bond_feature_matrix.shape  # 获取键特征矩阵的形状

            print(f"Reaction {i + 1}, Result {j + 1}:")
            print(f"  SMILES: {result['smiles']}")
            print(f"  Atom Matrix shape: {rows} rows, {cols} columns")
            print(f"  Adjacency Matrix shape: {adj_rows} rows, {adj_cols} columns")
            print(f"  Bond Feature Matrix shape: {bond_rows} rows, {bond_cols} columns")  # 打印键特征矩阵的形状
            print(f"  Bond Feature Matrix content:\n{bond_feature_matrix}")  # 打印键特征矩阵的内容
            print(f"  Global Vector content:{global_vector}")


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    Graph_Processing_test()