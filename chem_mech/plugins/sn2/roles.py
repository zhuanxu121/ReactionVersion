from typing import Dict, List, Tuple
from chem_mech.core.loader import InternalGraph

class SN2Plugin:
    name = 'sn2'
    # 角色权重，用于 log_prob 计算
    WEIGHTS = {'Nu': 1.0, 'C': 0.5, 'LG': 1.0}

    @staticmethod
    def assign_roles(graph: InternalGraph,
                     batch_probs: List[List[Dict[str, float]]]
                     ) -> Dict[str, List[Tuple[int, float]]]:
        """
        识别并返回角色集合：
          - LG: 底物中典型离去基原子 (Cl, Br, I)
          - C : 与 LG 直连的碳中心
          - Nu: 试剂分子中常见亲核原子 (O, S, N)
        利用 batch_probs 作为初步打分，并结合元素信息做化学先验筛选。
        返回：{'Nu':[(idx,score),...],'C':[(idx,score),...],'LG':[(idx,score),...]}。
        """
        roles = {'Nu': [], 'C': [], 'LG': []}
        # 假设前两个子图：第一个是底物，第二是试剂
        # 计算子图原子偏移
        sub_n = len(batch_probs[0])
        # 1) 底物：寻找 LG 与 C
        for i, scores in enumerate(batch_probs[0]):
            score = scores.get('sn2', 0.0)
            atomic_num = int(graph.x[i][0])  # 假设 x[:,0] 存原子序数
            # 离去基：卤素
            if atomic_num in (17, 35, 53):
                roles['LG'].append((i, score))
                # 找相连的碳
                for j in range(graph.n_atoms):
                    if graph.adj[i, j] == 1 and int(graph.x[j][0]) == 6:
                        roles['C'].append((j, score))
        # 2) 试剂：亲核原子
        offset = sub_n
        for k, scores in enumerate(batch_probs[1]):
            idx = offset + k
            score = scores.get('sn2', 0.0)
            atomic_num = int(graph.x[idx][0])
            if atomic_num in (8, 16, 7):  # O, S, N
                roles['Nu'].append((idx, score))
        return roles
