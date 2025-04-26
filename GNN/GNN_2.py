import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import random
from Utils.Graph_Processing import graph_processing

# 设置随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()


# 初始化权重函数
def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


#  MLP模块
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2, dropout=0.1, activation=nn.ReLU()):
        super().__init__()
        if layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            layer_list = []
            for i in range(layers - 1):
                layer_list.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                layer_list.append(activation)
                layer_list.append(nn.LayerNorm(hidden_dim))
                layer_list.append(nn.Dropout(dropout))
            layer_list.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*layer_list)

        self.layers.apply(init_weight)

    def forward(self, x):
        return self.layers(x)


# 🧠 GAT模型(输出的反应机理还没确定有哪些)
class Single_Molecule_GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, mlp_hidden=64, output_dim=13, heads=4, mlp_layers=2, dropout=0.1):
        super(Single_Molecule_GATNet, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=mlp_hidden, output_dim=output_dim,
                       layers=mlp_layers, dropout=dropout, activation=nn.ReLU())

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.gat1(x, edge_index, edge_attr=edge_attr)  # 使用边特征
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)  # 使用边特征
        x = F.elu(x)
        x = self.gat3(x, edge_index, edge_attr=edge_attr)  # 使用边特征
        x = F.elu(x)

        return self.mlp(x)
#
# class Multi_Molecule_GATNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, mlp_hidden=64, output_dim=3, heads=4, mlp_layers=2, dropout=0.1):
#         super(Multi_Molecule_GATNet, self).__init__()
#
#         # GAT层，用于边的特征变换
#         self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
#         self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
#         self.gat3 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
#
#         # 边变换模块，用于结合原子特征和边特征
#         self.edge_transform = nn.Sequential(
#             nn.Linear(input_dim * 2, hidden_dim),  # 将原子信息与边信息拼接在一起
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         # 最终的MLP层，用于输出结果
#         self.mlp = MLP(input_dim=hidden_dim, hidden_dim=mlp_hidden, output_dim=output_dim,
#                        layers=mlp_layers, dropout=dropout, activation=nn.ReLU())
#
#     def forward(self, x, edge_index, batch, edge_attr=None):
#         # 基于原子特征拼接边特征
#         edge_attr = self.edge_transform(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1))
#
#         # 通过GAT层进行分子间的交互
#         x = self.gat1(x, edge_index, edge_attr=edge_attr)  # 使用变换后的边特征
#         x = F.elu(x)
#         x = self.gat2(x, edge_index, edge_attr=edge_attr)  # 使用变换后的边特征
#         x = F.elu(x)
#         x = self.gat3(x, edge_index, edge_attr=edge_attr)  # 使用变换后的边特征
#         x = F.elu(x)
#
#         # 通过MLP层输出最终的结果
#         return self.mlp(x)
#
#     class ReactionPipeline(nn.Module):
#         def __init__(self, single_model, multi_model, graph_connector_fn):
#             """
#             single_model: 用来处理单个反应物图（计算分子反应倾向），例如 Single_Molecule_GATNet
#             multi_model: 用于处理大图（计算键的反应倾向），例如 Multi_Molecule_GATNet
#             graph_connector_fn: 图连接函数，将经过单分子模型处理后的各图合成一张大图
#             """
#             super(ReactionPipeline, self).__init__()
#             self.single_model = single_model
#             self.multi_model = multi_model
#             self.graph_connector_fn = graph_connector_fn
#
#         def forward(self, reactant_graphs):
#             """
#             reactant_graphs: 列表，每个元素为一个 torch_geometric.data.Data 对象，代表一个反应物图，
#                              包含属性 x、edge_index、batch、edge_attr（如果有）等。
#             """
#             processed_graphs = []
#             # 对每个反应物图分别使用单分子模型，计算分子反应倾向
#             for data in reactant_graphs:
#                 # 单分子模型会更新节点特征，表示每个反应物中原子的分子反应倾向
#                 # 请注意，这里不改变原始图的边结构，只更新节点属性（x）
#                 molecule_tendency = self.single_model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
#
#                 # 复制原图，并将计算后的分子反应倾向覆盖原有节点特征
#                 # 这里假设 data 是支持 clone() 的 torch_geometric.data.Data 对象
#                 data_updated = data.clone()
#                 data_updated.x = molecule_tendency
#                 processed_graphs.append(data_updated)
#
#             # 使用你自定义的图连接算法，将各个反应物图“连接”成一张大图
#             big_graph = self.graph_connector_fn(processed_graphs)
#
#             # 将大图输入到多分子模型中，计算键的反应倾向
#             # 要求大图至少包含 x, edge_index, batch 以及（可选的） edge_attr 属性
#             bond_tendency = self.multi_model(big_graph.x, big_graph.edge_index, big_graph.batch,
#                                              edge_attr=big_graph.edge_attr)
#             return bond_tendency
#
#
# class ReactionPipeline(nn.Module):
#     def __init__(self, single_model, multi_model, graph_connector_fn):
#         """
#         single_model: 用来处理单个反应物图（计算分子反应倾向），例如 Single_Molecule_GATNet
#         multi_model: 用于处理大图（计算键的反应倾向），例如 Multi_Molecule_GATNet
#         graph_connector_fn: 图连接函数，将经过单分子模型处理后的各图合成一张大图
#         """
#         super(ReactionPipeline, self).__init__()
#         self.single_model = single_model
#         self.multi_model = multi_model
#         self.graph_connector_fn = graph_processing
#
#     def forward(self, reactant_graphs):
#         """
#         reactant_graphs: 列表，每个元素为一个 torch_geometric.data.Data 对象，代表一个反应物图，
#                          包含属性 x、edge_index、batch、edge_attr（如果有）等。
#         """
#         processed_graphs = []
#         # 对每个反应物图分别使用单分子模型，计算分子反应倾向
#         for data in reactant_graphs:
#             # 单分子模型会更新节点特征，表示每个反应物中原子的分子反应倾向
#             # 请注意，这里不改变原始图的边结构，只更新节点属性（x）
#             molecule_tendency = self.single_model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
#
#             # 复制原图，并将计算后的分子反应倾向覆盖原有节点特征
#             # 这里假设 data 是支持 clone() 的 torch_geometric.data.Data 对象
#             data_updated = data.clone()
#             data_updated.x = molecule_tendency
#             processed_graphs.append(data_updated)
#
#         # 使用你自定义的图连接算法，将各个反应物图“连接”成一张大图
#         big_graph = self.graph_connector_fn(processed_graphs)
#
#         # 将大图输入到多分子模型中，计算键的反应倾向
#         # 要求大图至少包含 x, edge_index, batch 以及（可选的） edge_attr 属性
#         bond_tendency = self.multi_model(big_graph.x, big_graph.edge_index, big_graph.batch,
#                                          edge_attr=big_graph.edge_attr)
#         return bond_tendency
#
