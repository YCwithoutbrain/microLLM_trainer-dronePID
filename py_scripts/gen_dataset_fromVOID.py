import os
from datetime import datetime
import numpy as np
import pandas as pd

# ======================
# 生成 10000 条真实分布数据 (适配 7寸4轴 Pixhawk 5X 飞控数据特征)
# ======================
np.random.seed(42)
n = 10000

# 角度误差 (Roll, Pitch, Yaw) 单位: 弧度 (rad)。 7寸机飞行通常姿态误差范围较小
roll_err = np.random.uniform(-0.5, 0.5, n)
pitch_err = np.random.uniform(-0.5, 0.5, n)
yaw_err = np.random.uniform(-0.8, 0.8, n)

# 角速度 (Roll, Pitch, Yaw) 单位: rad/s。 7寸机机动性较5寸低，最大角速率通常在 3-5 rad/s
roll_rate = np.random.uniform(-4.0, 4.0, n)
pitch_rate = np.random.uniform(-4.0, 4.0, n)
yaw_rate = np.random.uniform(-2.5, 2.5, n)

# 加速度 (X, Y, Z) 单位: m/s^2。通常 Z 轴在 9.8 左右浮动，X, Y 在 0 附近浮动
acc_x = np.random.uniform(-2.0, 2.0, n)
acc_y = np.random.uniform(-2.0, 2.0, n)
acc_z = np.random.uniform(7.8, 11.8, n)

X = np.stack([roll_err, pitch_err, yaw_err, roll_rate, pitch_rate, yaw_rate, acc_x, acc_y, acc_z], axis=1)

# ======================
# 【针对 7寸四轴 & Pixhawk 5X 的最优PID计算模型】
# 7寸机架特点：惯量大，低频谐振多，因此 P和D 相对较小，需要更高的 I 来保持姿态。
# PX4 速率环典型参考值（7寸）：P: 0.04-0.12, I: 0.1-0.25, D: 0.001-0.004
# ======================
def get_best_pid(e_r, e_p, e_y, vr, vp, vy, ax, ay, az):
    ar, ap, ay = abs(e_r), abs(e_p), abs(e_y)

    # ROLL
    P = 0.05 + ar*0.04 + abs(vr)*0.005
    I = 0.12 + ar*0.05
    D = 0.0015 + abs(vr)*0.0004
    P = min(0.12, P)
    I = min(0.25, I)
    D = min(0.004, D)

    # PITCH (通常 Pitch 轴惯量稍大，参数与 Roll 类似或略高)
    Pp = 0.05 + ap*0.04 + abs(vp)*0.005
    Ip = 0.12 + ap*0.05
    Dp = 0.0015 + abs(vp)*0.0004
    Pp = min(0.12, Pp)
    Ip = min(0.25, Ip)
    Dp = min(0.004, Dp)

    # YAW (偏航通常主要靠 P 和 I 进行控制，D 往往非常小或为 0)
    Py = 0.15 + ay*0.05 + abs(vy)*0.02
    Iy = 0.10 + ay*0.03
    Dy = 0.0001 + abs(vy)*0.0002  # 增加极微小的 D 项变化，防止该特征列全为 0
    Py = min(0.25, Py)
    Iy = min(0.20, Iy)
    Dy = min(0.001, Dy)  # 限制在极小范围内

    return [P,I,D, Pp,Ip,Dp, Py,Iy,Dy]

# 生成标签
Y = np.array([get_best_pid(*x) for x in X])

# 创建以时间戳命名的新文件夹
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(project_root, "生成数据集", timestamp)
os.makedirs(save_dir, exist_ok=True)

# 保存
np.save(os.path.join(save_dir, "X_data.npy"), X)
np.save(os.path.join(save_dir, "Y_data.npy"), Y)
df = pd.DataFrame(np.hstack([X,Y]), columns=[
    "er","ep","ey","vr","vp","vy","ax","ay","az",
    "rp","ri","rd","pp","pi","pd","yp","yi","yd"
])
df.to_csv(os.path.join(save_dir, "pid_dataset_10000.csv"), index=False)
print(f"✅ 10000条数据集生成完成，已保存至：{save_dir}")