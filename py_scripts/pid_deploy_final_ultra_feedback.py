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

master = None
print("🔌 开始尝试连接飞控...")
for baud in [57600, 115200]:
    try:
        print(f"尝试使用波特率 {baud} 连接...")
        m = mavutil.mavlink_connection(serial_port, baud=baud)
        m.wait_heartbeat(timeout=3)
        if m.target_system > 0:
            master = m
            print(f"✅ 成功连接飞控！实际握手波特率: {baud}, 系统ID: {m.target_system}")
            break
    except Exception as e:
        print(f"⚠️ 使用波特率 {baud} 连接遇到错误: {e}")

if master is None:
    print("❌ 无法连接飞控，请检查接线和端口权限！")
    sys.exit(1)

def get_state():
    msg = master.recv_match(type='ATTITUDE', blocking=True)
    # 获取加速度计数据
    imu_msg = master.messages.get('HIGHRES_IMU', master.messages.get('RAW_IMU'))
    xacc = imu_msg.xacc if imu_msg else 0.0
    yacc = imu_msg.yacc if imu_msg else 0.0
    zacc = imu_msg.zacc if imu_msg else 9.8
    return [msg.roll, msg.pitch, msg.yaw, msg.rollspeed, msg.pitchspeed, msg.yawspeed, xacc, yacc, zacc]

def send(axis, p, i, d, retries=3):
    # 转换为 PX4 角速率 PID 映射逻辑
    axis_map = {0: "ROLLRATE", 1: "PITCHRATE", 2: "YAWRATE"}
    axis_name = axis_map[axis]

    # 构造参数名列表，强转为16字节，并用 \x00 填充
    param_ids = [
        f"MC_{axis_name}_P".encode('ascii')[:16].ljust(16, b'\x00'),
        f"MC_{axis_name}_I".encode('ascii')[:16].ljust(16, b'\x00'),
        f"MC_{axis_name}_D".encode('ascii')[:16].ljust(16, b'\x00')
    ]
    values = [p, i, d]
    param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32

    # 关键修复点1: 在发送写入前，清空输入缓冲区，防止旧的ACK消息干扰
    # master.flush() # 如果使用串口原生flush，或者利用mavutil的机制

    # 发送写入请求
    for param_id, value in zip(param_ids, values):
        target_param_id = param_id.decode('utf-8').rstrip('\x00')

        for attempt in range(retries):
            print(f"📤 [尝试 {attempt+1}/{retries}] 尝试写入: {target_param_id} = {value}")
            master.mav.param_set_send(
                master.target_system, 
                master.target_component, 
                param_id, 
                float(value), 
                param_type
            )

            # 关键修复点2: 等待写入确认 (PARAM_VALUE)
            # ArduPilot 在写入成功后会回传一个 PARAM_VALUE
            # 我们等待这个消息来确认写入是否被接收
            timeout = time.time() + 1.0 # 1秒超时
            ack_received = False
            while time.time() < timeout and not ack_received:
                # 非阻塞读取，直到收到对应的参数回复
                msg = master.recv_match(type='PARAM_VALUE', blocking=False)
                if msg:
                    # 检查是否是针对我们刚刚发送参数的回复
                    received_param_id = msg.param_id
                    if isinstance(received_param_id, bytes):
                        received_param_id = received_param_id.decode('utf-8')
                    received_param_id = received_param_id.rstrip('\x00')

                    if received_param_id == target_param_id:
                        # 根据文档，如果写入成功，这里回传的应该是新值
                        # 如果写入失败，这里可能先回传新值，然后回传旧值
                        print(f"📨 收到回执: {received_param_id} = {msg.param_value}")
                        if abs(msg.param_value - value) < 0.001:
                            print(f"✅ 参数 {received_param_id} 写入成功并验证")
                        else:
                            print(f"⚠️  参数 {received_param_id} 回传值不匹配，可能写入失败或被覆盖")
                        ack_received = True

            if ack_received:
                break

        if not ack_received:
            print(f"❌ 写入超时失败: {target_param_id}")

def read_pid_param(axis):
    """
    读取飞控当前的 P/I/D
    修复了逻辑：直接读取，不包含写入逻辑
    """
    # 转换为 PX4 角速率 PID 映射逻辑
    axis_map = {0: "ROLLRATE", 1: "PITCHRATE", 2: "YAWRATE"}
    axis_name = axis_map[axis]

    # 重要修正：ArduPilot 文档指出，参数名可能因子系统启用状态而变化
    # 且文档[1]中的读取函数 read_single 内部逻辑有误（它在循环内定义，且逻辑混乱）
    # 这里使用更直接的方法：请求读取，然后等待回复

    param_names = [
        f"MC_{axis_name}_P",
        f"MC_{axis_name}_I", 
        f"MC_{axis_name}_D"
    ]
    results = []

    for param_name in param_names:
        # 请求读取该参数
        # 关键点：param_id 同样需转换为满 16 字节
        param_id_bytes = param_name.encode('ascii')[:16].ljust(16, b'\x00')
        master.mav.param_request_read_send(
            master.target_system, 
            master.target_component, 
            param_id_bytes, 
            -1 
        )

        # 读取返回值
        tstart = time.time()
        while time.time() - tstart < 1.0:
            msg = master.recv_match(type='PARAM_VALUE', blocking=False)
            if msg:
                # 严格匹配参数名
                pid = msg.param_id
                if isinstance(pid, bytes):
                    pid = pid.decode('utf-8')
                pid = pid.rstrip('\x00')
                if pid == param_name:
                    results.append(round(msg.param_value, 4))
                    break
        else:
            # 超时处理
            print(f"❌ 读取超时: {param_name}")
            results.append(None)

    # 确保返回三个值
    while len(results) < 3:
        results.append(None)

    return results[0], results[1], results[2]

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
    
    # === 增加安全限幅处理 (基于最新的 PX4 基准) ===
    # Roll/Pitch 基准附近: P=0.15, I=0.03, D=0.005
    # Yaw 基准附近: P=0.20, I=0.05, D=0.00
    rp, ri, rd = np.clip(rp, 0.05, 0.25), np.clip(ri, 0.01, 0.10), np.clip(rd, 0.001, 0.01)
    pp, pi, pd = np.clip(pp, 0.05, 0.25), np.clip(pi, 0.01, 0.10), np.clip(pd, 0.001, 0.01)
    yp, yi, yd = np.clip(yp, 0.10, 0.30), np.clip(yi, 0.01, 0.15), np.clip(yd, 0.0, 0.005)

    send(0,rp,ri,rd)

    # 下发后立刻读取校验
    read_p, read_i, read_d = read_pid_param(0)

    if read_p is not None and read_i is not None and read_d is not None:
        p_ok = abs(rp - read_p) < 0.001
        i_ok = abs(ri - read_i) < 0.001
        d_ok = abs(rd - read_d) < 0.001

        print(f"🔍 验证 ROLL 轴:")
        print(f"  [写入] P={rp:.4f}, I={ri:.4f}, D={rd:.4f}")
        print(f"  [读取] P={read_p:.4f}, I={read_i:.4f}, D={read_d:.4f}")

        if p_ok and i_ok and d_ok:
            print("✅ ROLL轴 PID 已全部成功写入飞控！")
        else:
            print(f"⚠️ ROLL轴 部分写入失败：P={p_ok}, I={i_ok}, D={d_ok}")
    else:
        print("❌ 读取校验超时或失败，无法验证写入结果！")

    send(1,pp,pi,pd)
    send(2,yp,yi,yd)
    print(f"ROLL [P:{rp:.3f} I:{ri:.3f} D:{rd:.5f}] | PITCH [P:{pp:.3f} I:{pi:.3f} D:{pd:.5f}] | YAW [P:{yp:.3f} I:{yi:.3f} D:{yd:.5f}]")
    time.sleep(0.2)

