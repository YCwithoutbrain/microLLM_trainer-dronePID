import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from torch.utils.data import DataLoader, TensorDataset

# 1. 配置设备 (优先使用 GPU 以加速训练)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载数据并加入特征标准化 (对 Transformer 收敛极其重要)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_base_dir = os.path.join(project_root, "生成数据集")

# 获取最新生成的文件夹
subdirs = [os.path.join(dataset_base_dir, d) for d in os.listdir(dataset_base_dir) if os.path.isdir(os.path.join(dataset_base_dir, d))]
if not subdirs:
    raise FileNotFoundError(f"在 {dataset_base_dir} 下未找到任何数据集文件夹，请先生成数据集。")
latest_subdir = max(subdirs, key=os.path.getmtime)
print(f"正在从最新数据集目录加载：{latest_subdir}")

X_np = np.load(os.path.join(latest_subdir, "X_data.npy"))
Y_np = np.load(os.path.join(latest_subdir, "Y_data.npy"))

# 切分训练集(80%)和验证集(20%)
np.random.seed(42)
indices = np.random.permutation(len(X_np))
split_idx = int(0.8 * len(X_np))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train, Y_train = X_np[train_idx], Y_np[train_idx]
X_val, Y_val = X_np[val_idx], Y_np[val_idx]

# 计算均值和标准差，并保存供树莓派推理使用
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(project_root, "完成模型", timestamp)
onnx_dir = os.path.join(save_dir, "onnx")
pth_dir = os.path.join(save_dir, "Pytorch权重字典")
os.makedirs(onnx_dir, exist_ok=True)
os.makedirs(pth_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X_scaler_mean.npy"), X_mean)
np.save(os.path.join(save_dir, "X_scaler_std.npy"), X_std)

X_train_scaled = (X_train - X_mean) / (X_std + 1e-8)
X_val_scaled = (X_val - X_mean) / (X_std + 1e-8)

# 构建 DataLoader 使用 Mini-batch
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                              torch.tensor(Y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), 
                            torch.tensor(Y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# 模型：Transformer + 投影
class PIDTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(9, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.out = nn.Linear(64, 9)

    def forward(self, x):
        x = self.in_proj(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return self.out(x)

model = PIDTransformer()

# LoRA配置
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["in_proj", "out"]
)
model = get_peft_model(model, lora_cfg)
model.to(device)

# 训练
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 100  # 使用 Mini-batch 后往往不需要那么多 epoch
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        y_hat = model(batch_x)
        loss = criterion(y_hat, batch_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()

    if epoch % 10 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_hat = model(batch_x)
                val_loss += criterion(y_hat, batch_y).item()
        print(f"Epoch {epoch:03d}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

# 3. 推理部署优化：将 LoRA 权重合并回原模型中 (Merge and Unload)
# 合并后模型变为常规的 PIDTransformer，将其导出为 ONNX 格式，无须安装 PyTorch 即可在树莓派上使用 ONNX Runtime 进行推理。
merged_model = model.merge_and_unload()
merged_model.cpu()
merged_model.eval()

# 导出 Pytorch 原版权重字典
torch.save(merged_model.state_dict(), os.path.join(pth_dir, "pid_transformer.pth"))

# 创建一个虚拟输入用于追踪整个图的结构
dummy_input = torch.randn(1, 9, dtype=torch.float32)

# 导出为 ONNX
torch.onnx.export(
    merged_model, 
    dummy_input, 
    os.path.join(onnx_dir, "pid_transformer_deploy.onnx"), 
    export_params=True, 
    opset_version=14, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"\n✅ 模型训练与LoRA合并完成！")
print(f"💡 PyTorch权重已保存至：{os.path.join(pth_dir, 'pid_transformer.pth')}")
print(f"💡 ONNX模型已保存至：{os.path.join(onnx_dir, 'pid_transformer_deploy.onnx')}")
print(f"💡 树莓派部署准备完毕：拷贝上述 onnx 文件以及 {save_dir} 内记录的标量 mean/std 至树莓派使用。")