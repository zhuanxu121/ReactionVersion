import torch
import torch.optim as optim
from sklearn.metrics import r2_score
import torch.nn.functional as F
from Feature_Engineering.data_preprocessing import load_and_preprocess_csv, ReactionBatchReader, functional_groups
# 假定你的模型类已经定义好并导入，比如：
from GNN.GNN_2 import Single_Molecule_GATNet
from Utils.Graph_Processing import graph_processing


# 数据预处理与批量加载器初始化
dataset=load_and_preprocess_csv()
print("Parsed reaction sequences count:", len(dataset))

train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = ReactionBatchReader(train_dataset, batch_size=32, functional_groups=functional_groups)
val_loader = ReactionBatchReader(val_dataset, batch_size=32, functional_groups=functional_groups)



# 模型参数与实例化

# 需要根据你的数据设置输入特征维度（input_dim），例如原子特征的长度
input_dim = 76  # 示例：根据你的数据实际情况修改

# 定义模型的超参数
hidden_dim = 64
mlp_hidden = 64
output_dim = 3   # 示例：预测类别数
heads = 4
mlp_layers = 2
dropout = 0.1

# 实例化你的模型
model = Single_Molecule_GATNet(input_dim, hidden_dim, mlp_hidden, output_dim, heads, mlp_layers, dropout)


# 设备设置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义优化器和损失函数

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# 如果你的任务是分类问题，使用交叉熵损失函数；如是回归任务请使用 MSELoss 等
loss_fn = torch.nn.CrossEntropyLoss()

# 训练函数
def train():
    model.train()
    total_loss = 0
    while True:
        try:
            batch = train_loader.next_batch()
            batch = batch.to(device)
            optimizer.zero_grad()
            #batch 是一个数组，无法直接放进模型进行计算，需要写一个函数来拼接
            outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)  # 传递边特征

            # 连接output与batch，output是一个大图，要先切分再连接
            # 还没写。。。。。。。。。。。。。。。。。。。。。。。
            #output_reactions就是连接后的图数组
            output_reactions = []

            processed_batch = []
            for reaction in output_reactions:
                processed_reaction = graph_processing(reaction)
                processed_batch.append(processed_reaction)

            #该loss得自定义一个，传入处理后的图数组
            loss = loss_fn(processed_batch, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        except StopIteration:
                # 当数据集迭代完毕时，退出循环
            break
    return total_loss / len(train_dataset)

# 验证函数
def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        while True:
            try:
                batch = loader.next_batch()
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)  # 传递边特征
                preds.append(pred.cpu())
                labels.append(batch.y.cpu())
            except StopIteration:
            # 当数据集迭代完毕时，退出循环
                break

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    #这里的计算mse与r2的函数是有问题的，可能需要我自己写
    mse = F.mse_loss(preds, labels).item()
    r2 = r2_score(labels.numpy(), preds.numpy())
    return mse, r2

# ⏳ 训练主循环 + 保存特征
train_losses = []
val_losses = []
best_val_r2 = -float("inf")
best_epoch = 0  # 记录最佳模型的epoch


for epoch in range(1, 51):
    train_loss = train()
    val_mse, val_r2 = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_mse)

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_epoch = epoch  # 更新最佳模型的epoch
        torch.save(model.state_dict(), "best_LogP_model.pt")

    # # 保存本轮训练的图表示 + 标签
    # global_feats, labels = extract_global_features(train_loader)
    # np.save(os.path.join(save_dir, f"features_epoch_{epoch:03d}.npy"), global_feats)
    # np.save(os.path.join(save_dir, f"labels_epoch_{epoch:03d}.npy"), labels)

    if epoch % 1 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val R2: {val_r2:.4f}")

print(f"训练完成 ✅ 最佳验证集 R²: {best_val_r2:.4f} (在第 {best_epoch} 轮)")



# # 训练循环
#
# num_epochs = 150  # 可根据需要调整epoch数量
#
# for epoch in range(num_epochs):
#
#
#     # 如果 ReactionBatchReader 是迭代器或者需要反复调用 next_batch()，则用下面这种方式迭代
#     while True:
#         try:
#             batch = reader.next_batch()
#         except StopIteration:
#             # 当数据集迭代完毕时，退出循环
#             break
#
#         # 将batch中的数据移动到选定设备上。
#         # 这里假设batch是一个类似PyG的Data对象，包含 x, edge_index, batch, edge_attr 以及 y（标签）
#         batch = batch.to(device)
#
#         optimizer.zero_grad()  # 清空梯度
#
#         # 前向传播：输入节点特征 x、边关系 edge_index、图 batch 信息，以及边特征 edge_attr
#         outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
#
#         #连接output与batch
#         #还没写。。。。。。。。。。。。。。。。。。。。。。。
#         output_reactions=[]
#
#         processed_batch=[]
#         for reaction in output_reactions:
#             processed_reaction=graph_processing(reaction)
#             processed_batch.append(processed_reaction)
#
#
#
#         # 计算损失，注意：这里假定标签存储在 batch.y 中，如果你的数据结构不一样，请调整此处代码
#         loss = criterion(outputs, batch.y)
#
#         # 反向传播和梯度更新
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     # 如果 ReactionBatchReader 提供总批次数量，可以计算平均损失；如果没有，你可以自行记录批次数
#     if hasattr(reader, "num_batches"):
#         avg_loss = total_loss / reader.num_batches
#     else:
#         avg_loss = total_loss  # 或者记录每个 epoch 内的 batch 数量，再求平均
#
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
#
# # -----------------------------
# # 保存训练好的模型（可选）
# # -----------------------------
# torch.save(model.state_dict(), "multi_molecule_gatnet.pth")
# print("模型已保存！")
