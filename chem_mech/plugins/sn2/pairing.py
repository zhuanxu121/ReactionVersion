from typing import List, Tuple, Dict, Any
from chem_mech.core.loader import InternalGraph

class SN2Plugin:
    @staticmethod
    def gen_pairs(graph: InternalGraph,
                  roles: Dict[str, List[Tuple[int, float]]]
                  ) -> List[Tuple[Tuple[int, int, int], Dict[str, float]]]:
        """
        生成三元组 (Nu_idx, C_idx, LG_idx) 及对应角色分数字典。
        返回：[((nu,c,lg),{'Nu':p_nu,'C':p_c,'LG':p_lg}), ...]
        """
        pairs = []
        for nu_idx, p_nu in roles['Nu']:
            for c_idx, p_c in roles['C']:
                for lg_idx, p_lg in roles['LG']:
                    # 确保 LG 与 C 相连
                    if graph.adj[lg_idx, c_idx] == 1:
                        pairs.append(
                            ((nu_idx, c_idx, lg_idx),
                             {'Nu': p_nu, 'C': p_c, 'LG': p_lg})
                        )
        return pairs

