import math
import numpy as np
from typing import List, Tuple, Dict, Any

def log_prob(role_score: Dict[str, float], weights: Dict[str, float]) -> float:
    # 计算各角色 log 概率加权和
    return sum(weights.get(role, 1.0) * math.log(score + 1e-9)
               for role, score in role_score.items())


def softmax_and_pack(path_pool: List[Tuple[str, float, Dict[str, Any]]]) -> List[List[Any]]:
    # path_pool: [(mech_name, log_score, prod_dict), ...]
    logs = np.array([p[1] for p in path_pool])
    exps = np.exp(logs - logs.max())
    probs = exps / exps.sum()
    out = []
    for (mech, _, prod_dict), p in zip(path_pool, probs):
        row = [mech, float(p)]
        # 插件返回的中间体
        row.extend(prod_dict.get('intermediates', []))
        row.append(prod_dict.get('smiles'))
        out.append(row)
    # 按概率降序
    return sorted(out, key=lambda x: -x[1])