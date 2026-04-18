import os
import glob
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset

# 从同级脚本导入获取真实目标 PID 的方法
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gen_dataset_fromVOID import get_best_pid

# 1. 配置设备 (优先使用 GPU 以加速训练)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 定位各类目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
real_data_dir = os.path.join(project_root, "实飞数据收集_树莓派内使用")
model_base_dir = os.path.join(project_root, "完成模型")

# 3. 找到 "./完成模型/" 下的最新文件夹作为基础模型与 Scaler 的目标
subdirs = [os.path.join(model_base_dir, d) for d in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, d))]
if not subdirs:
    raise FileNotFoundError(f"在 {model_base_dir} 下未找到任何基础模型文件夹。")
latest_model_dir = max(subdirs, key=os.path.getmtime)
print(f"👉 找到最新完成模型目录：{latest_model_dir}")

# 4. 读取多个实飞 .csv 文件合并并打标
csv_files = glob.glob(os.path.join(real_data_dir, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"在 {real_data_dir} 下未找到任何实飞数据 .csv 文件。")

print("🔄 开始读取并合并实飞 .csv 数据...")
all_X = []
all_Y = []

feature_cols = ['roll', 'pitch', 'yaw', 'rollspeed', 'pitchspeed', 'yawspeed', 
                'accelerationx', 'accelerationy', 'accelerationz']

for f in csv_files:
    try:
        df = pd.read_csv(f)
        # 验证必需特征列是否存在
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"⚠️ 文件 {os.path.basename(f)} 缺少列 {missing}，已跳过。")
            continue
        
        # 提取输入特征
        X_part = df[feature_cols].values
        
        # 利用原有评价基准来计算每个实飞状态理应分配的最优 PID（以此作为目标Y）
        Y_part = np.array([get_best_pid(*row) for row in X_part])
        
        all_X.append(X_part)
        all_Y.append(Y_part)
    except Exception as e:
        print(f"⚠️ 读取 {os.path.basename(f)} 时发生错误: {e}，跳过")

if not all_X:
    raise ValueError("未能从任何 csv 中提取到有效数据，请检查列名格式！")

X_np = np.vstack(all_X)
Y_np = np.vstack(all_Y)
print(f"📊 共合并了 {len(csv_files)} 个文件，实飞数据总量: {len(X_np)} 条")

# 5. 加载上一阶段生成的归一化权重对实飞特征统一标准化（必须用旧的标准否则推断失效）
mean_path = os.path.join(latest_model_dir, "X_scaler_mean.npy")
std_path = os.path.join(latest_model_dir, "X_scaler_std.npy")

if not os.path.exists(mean_path) or not os.path.exists(std_path):
    raise FileNotFoundError(f"环境缺失，{latest_model_dir} 中未找到 X_scaler_mean/std.npy。")

X_mean = np.load(mean_path)
X_std = np.load(std_path)
X_scaled = (X_np - X_mean) / (X_std + 1e-8)

# 切分训练集(80%)和测试集(20%)
np.random.seed(42)
indices = np.random.permutation(len(X_np))
split_idx = int(0.8 * len(X_np))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train, Y_train = X_scaled[train_idx], Y_np[train_idx]
X_val, Y_val = X_scaled[val_idx], Y_np[val_idx]

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                              torch.tensor(Y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                            torch.tensor(Y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# 6. 构建相同的 Transformer 模型
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

# 适应新的目录结构，寻找旧模型的 pth 权重
pth_path = os.path.join(latest_model_dir, "Pytorch权重字典", "pid_transformer.pth")
if not os.path.exists(pth_path): # 兼容未分级的就模型或前代增量模型
    pth_path = os.path.join(latest_model_dir, "pid_transformer.pth")

if os.path.exists(pth_path):
    print(f"🔄 找到旧版本 PyTorch 权重，正在加载上一阶段模型进行增量: {os.path.basename(pth_path)}")
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
else:
    print(f"⚠️ 在 {latest_model_dir} 中似乎未发现 pid_transformer.pth。")
    print("   之前仅导出了 ONNX 格式。若需完美增量训练，请在之前的训练阶段保存 torch state_dict。")
    print("   本次将初始化新权重并在此批实飞数据上进行 LoRA 微调！")

# 7. 应用 LoRA 包装
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["in_proj", "out"]
)
model = get_peft_model(model, lora_cfg)
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 8. 执行增量训练
epochs = 100
print("\n🚀 开始针对实飞数据的 Transformer-LoRA 增量训练...")
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
        print(f"Epoch {epoch:03d}, 训练集 Loss: {train_loss/len(train_loader):.6f}, 验证集 Loss: {val_loss/len(val_loader):.6f}")

# 9. 部署输出重组：合并 LoRA，建立子文件夹并保存模型
merged_model = model.merge_and_unload()
merged_model.cpu()
merged_model.eval()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
new_save_dir = os.path.join(latest_model_dir, f"{timestamp}_已实飞数据增量")
onnx_dir = os.path.join(new_save_dir, "onnx")
pth_dir = os.path.join(new_save_dir, "Pytorch权重字典")
os.makedirs(onnx_dir, exist_ok=True)
os.makedirs(pth_dir, exist_ok=True)

# 【重要补丁】为之后可以继续增量，本次导出时不仅保存ONNX还会储存 .pth 权重
torch.save(merged_model.state_dict(), os.path.join(pth_dir, "pid_transformer.pth"))

dummy_input = torch.randn(1, 9, dtype=torch.float32)
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

# 复制或再次持久化统计标量，以便于树莓派在调用 ONNX 时依旧有对应比例进行映射
np.save(os.path.join(new_save_dir, "X_scaler_mean.npy"), X_mean)
np.save(os.path.join(new_save_dir, "X_scaler_std.npy"), X_std)

print(f"\n✅ 增量训练完成！模型合并完毕。")
print(f"💡 包含 PyTorch 权重、ONNX 部署包、及 Scaler 均已存储在: {new_save_dir}")
