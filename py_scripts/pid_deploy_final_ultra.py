import numpy as np
import onnxruntime as ort
import time
import os
import sys
import socket
import json

# ==================
# 模型加载
# ==================
# 直接使用 onnxruntime 加载 ONNX 模型，无需 torch 环境
session = ort.InferenceSession("pid_transformer_deploy.onnx")

# 加载数据标准化参数
X_mean = np.load("X_scaler_mean.npy")
X_std = np.load("X_scaler_std.npy")

# ==================
# UDP Client 初始化
# ==================
UDP_SERVER_IP = "127.0.0.1"
UDP_SERVER_PORT = 5005
client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_sock.settimeout(2.0)

def get_state():
    try:
        req = {"cmd": "GET_STATE"}
        client_sock.sendto(json.dumps(req).encode('utf-8'), (UDP_SERVER_IP, UDP_SERVER_PORT))
        data, _ = client_sock.recvfrom(1024)
        resp = json.loads(data.decode('utf-8'))
        return resp.get("state", [0.0]*9)
    except Exception as e:
        print(f"Error getting state: {e}")
        return [0.0]*9

def send(axis, p, i, d):
    try:
        req = {
            "cmd": "SET_PID",
            "axis": axis,
            "p": float(p),
            "i": float(i),
            "d": float(d)
        }
        client_sock.sendto(json.dumps(req).encode('utf-8'), (UDP_SERVER_IP, UDP_SERVER_PORT))
        data, _ = client_sock.recvfrom(1024)
    except Exception as e:
        print(f"Error sending PID: {e}")

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

