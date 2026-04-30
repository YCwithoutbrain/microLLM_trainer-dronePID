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

def mavlink_receive_loop():
    global current_state
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
            response = {"status": "ok", "state": current_state}
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
