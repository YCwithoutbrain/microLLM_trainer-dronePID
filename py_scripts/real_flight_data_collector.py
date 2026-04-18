#树莓派内运行
#!/usr/bin/env python3
import os
import time
import csv
import argparse
from pymavlink import mavutil
from datetime import datetime

# 解析命令行参数
parser = argparse.ArgumentParser(description="实飞数据采集脚本")
parser.add_argument("-t", "--time", type=int, default=10, help="指定采集时间(秒)，未指定时默认为10秒")
args = parser.parse_args()

# 配置和飞控一致
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 115200
COLLECT_TIME_SEC = args.time

print("🔐 尝试自动获取串口读写权限...")
os.system(f"sudo chmod a+rw {SERIAL_PORT}")

print("🔌 正在连接飞控...")
try:
    vehicle = mavutil.mavlink_connection(
        SERIAL_PORT,
        baud=BAUD_RATE,
        autoreconnect=True
    )
    vehicle.wait_heartbeat()
    print("✅ 飞控连接成功！")
except Exception as e:
    print(f"❌ 连接失败：{e}")
    exit(1)

def collect_data(collect_time=10):
    output_csv = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_real_data.csv"
    data = []
    start = time.time()
    print(f"\n📡 开始采集 {collect_time} 秒实飞数据...")
    while time.time() - start < collect_time:
        try:
            # 非阻塞读取，避免崩溃
            msg = vehicle.recv_match(type='ATTITUDE', blocking=False, timeout=0.1)
            if msg:
                # 从 pymavlink 的缓存中获取与此时间点最接近的加速度计数据（优先 HIGHRES_IMU，其次 RAW_IMU）
                imu_msg = vehicle.messages.get('HIGHRES_IMU', vehicle.messages.get('RAW_IMU'))
                xacc = imu_msg.xacc if imu_msg else 0.0
                yacc = imu_msg.yacc if imu_msg else 0.0
                zacc = imu_msg.zacc if imu_msg else 0.0

                data.append([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    msg.roll,
                    msg.pitch,
                    msg.yaw,
                    msg.rollspeed,
                    msg.pitchspeed,
                    msg.yawspeed,
                    xacc,
                    yacc,
                    zacc
                ])
        except Exception:
            pass  # 忽略读取错误，不中断采集
        time.sleep(0.05)
    # 保存CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed", "accelerationx", "accelerationy", "accelerationz"])
        writer.writerows(data)
    print(f"\n✅ 采集完成！共 {len(data)} 条数据，已保存到 {output_csv}")

if __name__ == "__main__":
    try:
        collect_data(COLLECT_TIME_SEC)
    except KeyboardInterrupt:
        print("\n🔌 手动退出采集")
    finally:
        vehicle.close()