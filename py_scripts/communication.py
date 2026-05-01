import socket
import json
import time
import os
import sys
from pymavlink import mavutil
import threading

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 115200
BIND_PORT = 5005

# 检查权限
if os.path.exists(SERIAL_PORT) and not os.access(SERIAL_PORT, os.R_OK | os.W_OK):
    print(f"⚠️ 检测到对 {SERIAL_PORT} 没有读写权限，正在尝试通过 sudo 自动获取...")
    os.system(f"sudo chmod a+rw {SERIAL_PORT}")
    time.sleep(0.5)
    if not os.access(SERIAL_PORT, os.R_OK | os.W_OK):
        print("❌ 自动获取串口权限失败！")
        sys.exit(1)

print("Connecting to MAVLink...")
master = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
master.wait_heartbeat()
print("MAVLink connected.")

# 缓存最新的飞控状态
current_state = [0.0]*9 
current_velocity = [0.0, 0.0, 0.0]  # vx, vy, vz (m/s)
current_altitude = 0.0              # 相对高度 (m)

def mavlink_receive_loop():
    global current_state, current_velocity, current_altitude
    while True:
        msg = master.recv_match(blocking=True)
        if not msg:
            continue
        
        msg_type = msg.get_type()
        if msg_type == 'ATTITUDE':
            current_state[0] = msg.roll
            current_state[1] = msg.pitch
            current_state[2] = msg.yaw
            current_state[3] = msg.rollspeed
            current_state[4] = msg.pitchspeed
            current_state[5] = msg.yawspeed
        elif msg_type in ['HIGHRES_IMU', 'RAW_IMU']:
            current_state[6] = getattr(msg, 'xacc', 0.0)
            current_state[7] = getattr(msg, 'yacc', 0.0)
            current_state[8] = getattr(msg, 'zacc', 9.8)
        elif msg_type == 'LOCAL_POSITION_NED':
            current_velocity[0] = msg.vx
            current_velocity[1] = msg.vy
            current_velocity[2] = msg.vz
            current_altitude = -msg.z # NED坐标系下，Z轴向下，所以高度为-Z
        elif msg_type == 'GLOBAL_POSITION_INT':
            current_velocity[0] = msg.vx / 100.0
            current_velocity[1] = msg.vy / 100.0
            current_velocity[2] = msg.vz / 100.0
            current_altitude = msg.relative_alt / 1000.0
        elif msg_type == 'VFR_HUD':
            # 只在没有收到更精确的全局本地定位时，作为备选高度更新
            # 但 VFR_HUD 的 alt 是海拔，如果需要相对高度可用其他方式，此处一并收录
            pass # 也可以根据需要开启: current_altitude = msg.alt

# 启动 MAVLink 接收线程
threading.Thread(target=mavlink_receive_loop, daemon=True).start()

# 设置 UDP Server
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", BIND_PORT))
print(f"UDP Server listening on port {BIND_PORT}")

while True:
    try:
        data, addr = sock.recvfrom(1024)
        msg_str = data.decode('utf-8')
        request = json.loads(msg_str)
        
        cmd = request.get("cmd")
        
        if cmd == "GET_STATE":
            response = {
                "status": "ok", 
                "state": current_state,
                "acceleration": [current_state[6], current_state[7], current_state[8]],
                "velocity": current_velocity,
                "altitude": current_altitude
            }
            sock.sendto(json.dumps(response).encode('utf-8'), addr)
            
        elif cmd == "SET_PID":
            axis = request.get("axis")
            p = request.get("p")
            i = request.get("i")
            d = request.get("d")
            
            axis_map = {0: "ROLLRATE", 1: "PITCHRATE", 2: "YAWRATE"}
            if axis in axis_map:
                axis_name = axis_map[axis]
                param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                master.mav.param_set_send(master.target_system, master.target_component, f"MC_{axis_name}_P".encode('ascii')[:16].ljust(16, b'\x00'), float(p), param_type)
                master.mav.param_set_send(master.target_system, master.target_component, f"MC_{axis_name}_I".encode('ascii')[:16].ljust(16, b'\x00'), float(i), param_type)
                master.mav.param_set_send(master.target_system, master.target_component, f"MC_{axis_name}_D".encode('ascii')[:16].ljust(16, b'\x00'), float(d), param_type)
            
            response = {"status": "ok"}
            sock.sendto(json.dumps(response).encode('utf-8'), addr)
            
    except Exception as e:
        print(f"Error handling UDP datagram: {e}")
