from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
from rdkit import Chem



def sn1_transformation(reactant, reactant_site, leaving_group, nucleophile, nuc_site):
    """
    模拟 SN1 反应的图变换过程，接受以下5个参数：
      - reactant: RDKit 的反应物分子（Mol对象）
      - reactant_site: 整个反应物中，带有离去基团的碳原子索引
      - leaving_group: 离去基团（Mol对象，通常只含1个原子，如 [Br]、[Cl] 等）
      - nucleophile: 亲核试剂（Mol对象）
      - nuc_site: 亲核试剂分子中参与进攻的原子索引

    反应过程分为三个主要步骤：
      1. 离去基团从反应物离去，形成碳正离子中间体。
      2. 亲核试剂进攻碳正离子，生成带正电的中间体。
      3. 脱质子，生成最终产物。

    返回一个列表 [中间体1, 中间体2, 最终产物]。
    """
    results = []

    # =============== 第一步：离去基团离去 -> 生成碳正离子 ===============
    # 复制反应物以便修改
    cation_rwmol = Chem.RWMol(reactant)
    # 获取将要变为碳正离子的那个原子
    try:
        cation_atom = cation_rwmol.GetAtomWithIdx(reactant_site)
    except Exception as e:
        print("错误：反应物中找不到指定的反应位点。", e)
        return None

    # leaving_group 假设只含有 1 个原子
    lg_atom = leaving_group.GetAtomWithIdx(0)
    lg_atomic_num = lg_atom.GetAtomicNum()

    # 在反应位点邻域中，找到与离去基团匹配的原子（原子序数相同）
    leaving_idx_in_reactant = None
    for nbr in cation_atom.GetNeighbors():
        if nbr.GetAtomicNum() == lg_atomic_num:
            leaving_idx_in_reactant = nbr.GetIdx()
            break

    if leaving_idx_in_reactant is None:
        print("在反应物中未找到与离去基团匹配的原子。请检查参数。")
        return None

    # 断开这条键
    cation_rwmol.RemoveBond(reactant_site, leaving_idx_in_reactant)
    # 删除离去基团原子
    cation_rwmol.RemoveAtom(leaving_idx_in_reactant)

    # 将该碳原子设为 +1 电荷，模拟碳正离子
    cation_rwmol.GetAtomWithIdx(reactant_site).SetFormalCharge(+1)

    # 生成碳正离子中间体
    carbocation = cation_rwmol.GetMol()
    AllChem.SanitizeMol(carbocation)
    results.append(carbocation)

    # =============== 第二步：亲核试剂进攻 -> 带正电中间体 ===============
    # 将碳正离子与亲核试剂组合
    combined = Chem.CombineMols(carbocation, nucleophile)
    rw_comb = Chem.RWMol(combined)

    # 计算 nucleophile 在组合分子中的偏移
    shift = carbocation.GetNumAtoms()
    combined_nuc_site = nuc_site + shift

    # 在组合分子中添加键：碳正离子（reactant_site）与亲核试剂位点（combined_nuc_site）
    rw_comb.AddBond(reactant_site, combined_nuc_site, order=Chem.rdchem.BondType.SINGLE)

    # 设亲核试剂原子带正电
    rw_comb.GetAtomWithIdx(combined_nuc_site).SetFormalCharge(+1)

    intermediate2 = rw_comb.GetMol()
    AllChem.SanitizeMol(intermediate2)
    results.append(intermediate2)

    # =============== 第三步：脱质子 -> 最终产物 ===============
    rw_final = Chem.RWMol(intermediate2)
    nuc_atom = rw_final.GetAtomWithIdx(combined_nuc_site)

    # 简单遍历亲核试剂原子周围的氢原子，移除一个
    hydrogen_to_remove = None
    for nbr in nuc_atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            hydrogen_to_remove = nbr.GetIdx()
            break

    if hydrogen_to_remove is not None:
        rw_final.RemoveAtom(hydrogen_to_remove)
    # 将亲核试剂原子电荷设为中性
    rw_final.GetAtomWithIdx(combined_nuc_site).SetFormalCharge(0)

    final_product = rw_final.GetMol()
    AllChem.SanitizeMol(final_product)
    results.append(final_product)

    return results


def analyze_sn1_reaction(reaction_sequence):
    """
    分析一长串 SMILES 化学方程式，用于判断反应的各关键步骤及位点。
    假设输入 reaction_sequence 的格式为：
       "reactant_smiles >> intermediate1_smiles >> intermediate2_smiles >> final_product_smiles"
    分别代表：
       - 反应物（带有离去基团）
       - 离去基团离去生成碳正离子的中间体
       - 亲核试剂进攻后生成的不稳定中间体
       - 脱质子后的最终产物

    分析步骤：
      1. 将整个反应序列分割成四个阶段；
      2. 利用图的变化确定：
           - 可能的反应物及其反应位点：这里简单假设反应物中第一个卤素（Cl, Br, I）的相邻原子为反应位点；
           - 可能的离去基团：即上述卤素对应的原子（构造成单原子分子）；
           - 可能的亲核试剂及其反应位点：这里采用简单启发式，如果在中间体2中观察到反应物反应位点新增了氧原子，则视该氧为亲核试剂，且其唯一原子（索引0）为反应位点。
      3. 调用 sn1_transformation 函数模拟 SN1 反应变换，生成的最终产物与输入的最终产物比对，
         如果一致则认为该反应符合 SN1 机理。
    """
    segments = reaction_sequence.split(">>")
    if len(segments) != 4:
        print("输入的反应序列必须包含4个阶段，用 '>>' 分隔（反应物、intermediate1、intermediate2、最终产物）。")
        return

    reactant_smiles = segments[0].strip()
    inter1_smiles = segments[1].strip()
    inter2_smiles = segments[2].strip()
    final_smiles = segments[3].strip()

    reactant = Chem.MolFromSmiles(reactant_smiles)
    inter1 = Chem.MolFromSmiles(inter1_smiles)
    inter2 = Chem.MolFromSmiles(inter2_smiles)
    final_product = Chem.MolFromSmiles(final_smiles)

    # --------- 步骤1：判断离去基团与反应位点 ---------
    # 简单策略：在反应物中查找第一个卤素原子（Cl、Br、I），认为其相邻原子为反应位点
    halogen_nums = [17, 35, 53]
    leaving_group = None
    reactant_site = None
    for atom in reactant.GetAtoms():
        if atom.GetAtomicNum() in halogen_nums:
            # 构造离去基团（单原子），利用该原子的符号生成
            leaving_group = Chem.MolFromSmiles(atom.GetSymbol())
            # 假设其相邻原子中第一个为反应位点
            neighbors = atom.GetNeighbors()
            if neighbors:
                reactant_site = neighbors[0].GetIdx()
            break
    if leaving_group is None or reactant_site is None:
        print("无法识别离去基团或反应物的反应位点。")
        return

    # --------- 步骤2：判断亲核试剂及其反应位点 ---------
    # 简单策略：比较 intermediate1 与 intermediate2 的变化，
    # 如果在 intermediate2 中，反应位点处出现了新的氧原子，则认为该氧原子来自亲核试剂，
    # 这里我们构造亲核试剂为单氧原子 "O" ，其反应位点索引为0。
    nucleophile = None
    nuc_site = None
    # 获取反应物（或 intermediate1）中反应位点原子
    react_atom_inter1 = inter1.GetAtomWithIdx(reactant_site)
    # 在 intermediate2 中，查看 reactant_site 原子的邻居是否包含氧
    react_atom_inter2 = inter2.GetAtomWithIdx(reactant_site)
    new_neighbor_found = False
    for neighbor in react_atom_inter2.GetNeighbors():
        if neighbor.GetAtomicNum() == 8:
            # 此处简单认为该氧为亲核试剂的进攻原子
            nucleophile = Chem.MolFromSmiles("O")
            nuc_site = 0  # "O"仅有一个原子
            new_neighbor_found = True
            break
    if not new_neighbor_found or nucleophile is None:
        print("无法从反应中识别出明显的亲核试剂。")
        return

    # --------- 步骤3：调用 SN1 变换函数进行验证 ---------
    intermediates = sn1_transformation(reactant, reactant_site, leaving_group, nucleophile, nuc_site)
    if intermediates is None:
        print("SN1 变换过程未能顺利执行。")
        return

    # 最后一步生成的产物
    simulated_final = intermediates[-1]
    simulated_final_smiles = Chem.MolToSmiles(simulated_final)
    given_final_smiles = Chem.MolToSmiles(final_product)

    print("----- SN1 反应模拟结果 -----")
    print("原始反应物 SMILES：", reactant_smiles)
    print("模拟生成的最终产物 SMILES：", simulated_final_smiles)
    print("输入的最终产物 SMILES：", given_final_smiles)
    if simulated_final_smiles == given_final_smiles:
        print("验证通过：该反应符合 SN1 机理。")
    else:
        print("验证失败：模拟结果与输入产物不匹配。")


if __name__ == "__main__":
    # 构造测试分子：
    # 反应物：乙基溴 "CCBr"，假设反应位点为索引1（第二个碳）
    reactant_smiles = "CCBr"
    reactant = Chem.MolFromSmiles(reactant_smiles)
    AllChem.Compute2DCoords(reactant)

    # 离去基团：溴原子 "Br"
    leaving_group = Chem.MolFromSmiles("[Br]")
    # 亲核试剂：单氧原子 "O"，其反应位点索引为 0
    nucleophile = Chem.MolFromSmiles("[O]")
    nuc_site = 0

    # 设定反应位点索引为 1（反应物中第二个碳）
    reactant_site = 1

    # 调用 SN1 变换函数
    intermediates = sn1_transformation(
        reactant=reactant,
        reactant_site=reactant_site,
        leaving_group=leaving_group,
        nucleophile=nucleophile,
        nuc_site=nuc_site
    )

    # 若变换成功，绘制每一步生成的分子图像
    if intermediates is not None:
        fig, axs = plt.subplots(1, len(intermediates), figsize=(4 * len(intermediates), 4))
        if len(intermediates) == 1:
            axs = [axs]  # 兼容单张图的情况
        for i, mol in enumerate(intermediates, start=1):
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(300,300))
            axs[i-1].imshow(img)
            axs[i-1].set_title(f"步骤 {i}")
            axs[i-1].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("SN1 反应变换过程失败。")