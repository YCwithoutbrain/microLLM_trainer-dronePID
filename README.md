# MicroLLM Trainer - Drone PID

**基于轻量级 Transformer 与 LoRA 的无人机 AI PID 实时自适应调节系统**

## 📖 项目简介
本项目旨在利用深度学习中的 Transformer 架构处理时间序列/状态特征，为无人机（特别是7寸四轴、搭载 Pixhawk 5X 等飞控的机型）提供**实时、动态的最优 PID 参数调节**。本项目跑通了从数据生成、模型微调训练（使用 LoRA 降低算力要求）到边缘端（如树莓派）跨平台快速部署的全流程。

通过读取飞控回传的 9 维状态数据（Roll、Pitch、Yaw 角误差、角速度以及 X、Y、Z 轴加速度），AI 模型能够在树莓派内以 ONNX Runtime 极低延迟推理出当前姿态下的最佳角速率 PID (P, I, D) 参数，并通过 MAVLink 协议将其毫秒级下发回 Pixhawk 飞控，从而增强无人机在各种复杂工况下的抗扰动和姿态保持能力。

## 📂 核心目录与文件说明
- `py_scripts/`：包含所有的核心 Python 脚本代码。
  - `gen_dataset_fromVOID.py`：合成/生成适配飞控特征的 9 维虚拟飞行数据集。
  - `train_transformer_lora.py`：基于高精度数据集训练轻量级 Transformer，合并 LoRA 权重，并导出为无框架依赖的 `ONNX` 模型。
  - `pid_deploy_final_ultra*.py`：部署在树莓派端的推理反馈脚本，通过串口与飞控进行双向 MAVLink 通信。
- `实飞数据收集_树莓派内使用/`：
  - `real_flight_data_collector.py`：用于在树莓派中采集实飞的姿态和 IMU 数据供后续微调使用。
- `生成数据集/`：存放生成用于训练的 CSV 和 NPY 格式高维数据集。
- `完成模型/`：用于存储训练完成导出的 ONNX 部署模型模型及数据标准化配置参数。

## 🚀 如何使用

### 1. 环境准备 (PC 端训练)
建议使用 Anaconda 创建虚拟环境并安装由于训练可能依赖的包（PyTorch 等）：
```bash
pip install torch torchvision numpy pandas peft onnx onnxruntime pymavlink
```

### 2. 生成训练数据
在 PC 端运行数据生成脚本，生成带有 9 维度飞行器状态和最优 PID 对应标签的数据集：
```bash
python py_scripts/gen_dataset_fromVOID.py
```
*数据集将自动保存在根据时间戳创建的 `./生成数据集/` 文件夹下。*

### 3. 模型训练与导出
直接在 PC 端运行训练代码。代码会自动搜寻最新生成的数据集目录，完成基线 Transformer 模型的 LoRA 训练，计算 StandardScaler 标准化参数，并最终导出到 ONNX 格式以便边缘设备部署：
```bash
python py_scripts/train_transformer_lora.py
```
*模型 `.onnx` 及对应的状态标准化配置文件 `.npy` 将输出至 `./完成模型/` 内。*

### 4. 实飞数据收集 (可选但推荐)
你可以使用树莓派连接飞控串口，在真实试飞中收集状态数据：
1. 将 `实飞数据收集_树莓派内使用/real_flight_data_collector.py` 放进树莓派。
2. 指定 `/dev/serial0` 串口并运行脚本采集实飞验证数据。

### 5. 跨平台边缘端部署 (树莓派端)
在树莓派 (Raspberry Pi/Jetson Nano 等伴机电脑) 端进行如下操作：
1. **无需安装 PyTorch**，只需安装轻量级推理引擎和串口库：
   ```bash
   pip install onnxruntime numpy pymavlink
   ```
2. 将 `完成模型/` 目录新生成的 `pid_transformer_deploy.onnx`、`X_scaler_mean.npy`、`X_scaler_std.npy` 以及 `py_scripts/pid_deploy_final_ultra_feedback.py` 拷贝至树莓派同一个目录下。
3. 确保树莓派的 `/dev/serial0` (或其它有效串口)连接至飞控的 TELEM 口。
4. 运行部署脚本开启 AI PID 自适应接管：
   ```bash
   python pid_deploy_final_ultra_feedback.py
   ```
   *(为保证串口读写权限，可能需要用到 sudo 或者提前 `sudo chmod a+rw /dev/serial0`)*

## ⚠️ 注意事项与安全提示
- **带桨测试风险较高**。请务必在移除螺旋桨的桌面台架测试中或者安全系留飞行平台上验证串口参数修改正常、通信无延迟、读取闭环正常后再尝试挂桨外场试飞。
- 代码中内置了 `np.clip` 等安全软限幅措施，防止模型推理出错给出极限范围导致炸机。具体限幅基准参数可根据飞控品牌及实际机架尺寸进行二次调整。