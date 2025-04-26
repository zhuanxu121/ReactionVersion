import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any
from collections import defaultdict
from Feature_Engineering.data_preprocessing import (
    load_and_preprocess_csv,
    ReactionBatchReader,
    functional_groups,
)

# 确保项目根在 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ReactionBatchProcessor:
    @staticmethod
    def reactions_to_batch(reactions: List[List[Dict[str, Any]]]) -> Batch:
        flat_graphs: List[Data] = []
        # 先找一个有边的图来推断 edge_feat_dim
        edge_feat_dim = next(
            (
                np.array(g["bond_matrix"]).shape[1]
                for rxn in reactions
                for g in rxn
                if np.array(g["bond_matrix"]).ndim == 2 and np.array(g["bond_matrix"]).shape[0] > 0
            ),
            None,
        )
        if edge_feat_dim is None:
            raise ValueError("所有图都没有边，无法推断边特征维度")

        for rid, rxn in enumerate(reactions):
            for g in rxn:
                # 节点特征
                x = torch.tensor(g["atom_matrix"], dtype=torch.float)  # [n_atoms, atom_feat_dim]

                # edge_index
                adj = np.array(g["adjacency_matrix"])
                if adj.size == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    # 原始可能是 [n_edges,2]，也可能已经是 [2,n_edges]
                    if adj.ndim == 2 and adj.shape[1] == 2:
                        edge_index = torch.tensor(adj.T, dtype=torch.long)
                    else:
                        edge_index = torch.tensor(adj, dtype=torch.long)

                # edge_attr
                bfm = np.array(g["bond_matrix"])
                if bfm.ndim != 2 or bfm.size == 0:
                    bfm = bfm.reshape(0, edge_feat_dim)
                edge_attr = torch.tensor(bfm, dtype=torch.float)  # [n_edges, edge_feat_dim]

                # global_vector (保持 2D: [1, global_feat_dim]，以便 Batch 拼接成 [num_graphs, dim])
                gv = torch.tensor(g["global_vector"], dtype=torch.float).unsqueeze(0)

                # reaction_id 和 smiles
                reaction_id = torch.tensor([rid], dtype=torch.long)
                smiles = g.get("smiles", "")

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    global_vector=gv,
                    reaction_id=reaction_id,
                    smiles=smiles,
                )
                flat_graphs.append(data)

        return Batch.from_data_list(flat_graphs)

    @staticmethod
    def unpack_batch_results(
        batch: Batch, results: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        rids = batch.reaction_id.tolist()
        grouped = defaultdict(list)
        for rid, res in zip(rids, results):
            grouped[rid].append(res)
        max_rid = max(rids) if rids else -1
        return [grouped[rid] for rid in range(max_rid + 1)]


if __name__ == "__main__":
    # 加载并取一个 batch
    dataset = load_and_preprocess_csv()
    reader = ReactionBatchReader(dataset, batch_size=8, functional_groups=functional_groups)
    raw_reactions = reader.next_batch()

    # 打印原始结构
    print("Number of reactions in batch:", len(raw_reactions))
    for i, rxn in enumerate(raw_reactions):
        print(f" Reaction {i}: {len(rxn)} graphs")
        for j, g in enumerate(rxn):
            print(f"   Graph {j} keys: {list(g.keys())}")

    # pack
    processor = ReactionBatchProcessor()
    batch = processor.reactions_to_batch(raw_reactions)
    print("Batch.x shape:", batch.x.shape)
    print("Batch.edge_index shape:", batch.edge_index.shape)
    print("Batch.edge_attr shape:", batch.edge_attr.shape)
    print("Batch.global_vector shape:", batch.global_vector.shape)
    print("Batch.batch shape:", batch.batch.shape)
    print("Batch.reaction_id:", batch.reaction_id.tolist())

    # 检查 to_data_list() 能拿到 smiles
    for idx, data in enumerate(batch.to_data_list()):
        print(f"Data {idx} smiles: {data.smiles}")

    # # 测试 unpack
    # total = batch.reaction_id.size(0)
    # dummy = torch.arange(total)
    # restored = processor.unpack_batch_results(batch, dummy)
    # print("Restored structure lengths:", [len(r) for r in restored])
    # print("restored is",type(restored))
    # for reaction in restored:
    #     for reactant in reaction:
    #         print(reactant["smi"])
