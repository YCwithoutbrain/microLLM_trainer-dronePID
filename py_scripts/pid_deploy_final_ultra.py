import numpy as np
import onnxruntime as ort
from pymavlink import mavutil
import time
import os
import sys

# ==================
# 模型加载
# ==================
# 直接使用 onnxruntime 加载 ONNX 模型，无需 torch 环境
session = ort.InferenceSession("pid_transformer_deploy.onnx")

# 加载数据标准化参数
X_mean = np.load("X_scaler_mean.npy")
X_std = np.load("X_scaler_std.npy")

# ==================
# 飞控
# ==================
serial_port = "/dev/serial0"

# 如果串口存在，但没有读写权限，尝试通过 sudo 自动提权
if os.path.exists(serial_port) and not os.access(serial_port, os.R_OK | os.W_OK):
    print(f"⚠️ 检测到对 {serial_port} 没有读写权限，正在尝试通过 sudo 自动获取...")
    
    # 尝试临时放开串口的权限
    os.system(f"sudo chmod a+rw {serial_port}")
    time.sleep(0.5) # 等待文件系统权限生效
    
    # 再次检查权限
    if not os.access(serial_port, os.R_OK | os.W_OK):
        print("❌ 自动获取串口权限失败！")
        print("💡 请尝试使用以下两种方法之一：")
        print("1. 使用 sudo 运行: sudo python pid_deploy_final_ultra.py")
        print("2. 授予永久权限: sudo usermod -a -G dialout $USER (随后需要重新登录终端生效)")
        sys.exit(1)
    else:
        print(f"✅ 成功自动获取 {serial_port} 权限！")

master = mavutil.mavlink_connection(serial_port, baud=115200)
master.wait_heartbeat()

def get_state():
    msg = master.recv_match(type='ATTITUDE', blocking=True)
    # 获取加速度计数据
    imu_msg = master.messages.get('HIGHRES_IMU', master.messages.get('RAW_IMU'))
    xacc = imu_msg.xacc if imu_msg else 0.0
    yacc = imu_msg.yacc if imu_msg else 0.0
    zacc = imu_msg.zacc if imu_msg else 9.8
    return [msg.roll, msg.pitch, msg.yaw, msg.rollspeed, msg.pitchspeed, msg.yawspeed, xacc, yacc, zacc]

def send(axis,p,i,d):
    axis_map = {0: "ROLLRATE", 1: "PITCHRATE", 2: "YAWRATE"}
    axis_name = axis_map[axis]
    param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    master.mav.param_set_send(master.target_system, master.target_component, f"MC_{axis_name}_P".encode('ascii')[:16].ljust(16, b'\x00'), float(p), param_type)
    master.mav.param_set_send(master.target_system, master.target_component, f"MC_{axis_name}_I".encode('ascii')[:16].ljust(16, b'\x00'), float(i), param_type)
    master.mav.param_set_send(master.target_system, master.target_component, f"MC_{axis_name}_D".encode('ascii')[:16].ljust(16, b'\x00'), float(d), param_type)

# ==================
# 主循环
# ==================
print("🚀 AI PID (ONNX 版本) 已启动")
while True:
    # 获取当前状态并转换为 numpy 数组
    x_raw = np.array([get_state()])

    # 按照训练时的均值和标准差进行标准化归一化
    x_scaled = (x_raw - X_mean) / (X_std + 1e-8)
    
    # 转换为模型接受的输入数据类型
    x_input = x_scaled.astype(np.float32)

    # 运行 ONNX 推理
    ort_inputs = {session.get_inputs()[0].name: x_input}
    out = session.run(None, ort_inputs)[0][0]

    rp,ri,rd, pp,pi,pd, yp,yi,yd = out
    
    # === 增加安全限幅处理 ===
    # Roll/Pitch 基准: P=0.15, I=0.03, D=0.005
    # Yaw 基准: P=0.20, I=0.05, D=0.00
    rp, ri, rd = np.clip(rp, 0.05, 0.25), np.clip(ri, 0.01, 0.10), np.clip(rd, 0.001, 0.01)
    pp, pi, pd = np.clip(pp, 0.05, 0.25), np.clip(pi, 0.01, 0.10), np.clip(pd, 0.001, 0.01)
    yp, yi, yd = np.clip(yp, 0.10, 0.30), np.clip(yi, 0.01, 0.15), np.clip(yd, 0.0, 0.005)

    send(0,rp,ri,rd)
    send(1,pp,pi,pd)
    send(2,yp,yi,yd)
    print(f"ROLL [P:{rp:.3f} I:{ri:.3f} D:{rd:.5f}] | PITCH [P:{pp:.3f} I:{pi:.3f} D:{pd:.5f}] | YAW [P:{yp:.3f} I:{yi:.3f} D:{yd:.5f}]")
    time.sleep(0.2)

