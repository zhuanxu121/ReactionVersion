from typing import List, Dict, Any
from .loader import to_internal
from .composer import compose
from .path_score import log_prob, softmax_and_pack

# MECH_REGISTRY 在插件 __init__.py 中动态注入
MECH_REGISTRY = {}


def mechanism_engine(graph_dicts: List[Dict[str, Any]],
                     batch_probs: List[List[Dict[str, float]]],
                     top_k: int = 10) -> List[List[Any]]:
    # 1. 转 InternalGraph
    graphs = [to_internal(d) for d in graph_dicts]
    # 2. 合并多分子
    composite = compose(graphs)
    # 3. 插件循环收集路径
    path_pool = []
    for mech_name, Plugin in MECH_REGISTRY.items():
        plugin = Plugin()
        # 3.1 角色映射
        roles = plugin.assign_roles(composite, batch_probs)
        # 3.2 核心原子对生成
        pairs = plugin.gen_pairs(composite, roles)
        # 3.3 应用模板并得分
        for core_idxs, role_score in pairs:
            prod_list = plugin.apply_step(composite, core_idxs)
            lp = log_prob(role_score, plugin.WEIGHTS)
            for prod_dict in prod_list:
                path_pool.append((mech_name, lp, prod_dict))
    # 4. 归一化 & 打包
    results = softmax_and_pack(path_pool)[:top_k]
    return results