#python real_flight_data_collector.pyimport socket
import concurrent.futures
import ipaddress
import os
import sys
import platform

# ================= 配置区 =================
USERNAME = "yc"                # 你的树莓派账号
PASSWORD = "123456"     # 你的树莓派密码
PORT = 22                      # SSH端口，默认22
# ==========================================

def get_local_ip():
    """获取本机局域网IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def check_port(ip, port=22, timeout=0.5):
    """检测指定IP的端口是否开放"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False

def scan_and_find_pi():
    local_ip = get_local_ip()
    print(f"[*] 当前设备局域网IP: {local_ip}")
    if local_ip == "127.0.0.1":
        print("[-] 无法获取局域网IP，请检查网络连接。")
        return None

    # 计算出 C类(/24) 子网网段，能适用大部分路由器或手机热点环境
    network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    print(f"[*] 开始扫描当前网段: {network}")

    alive_hosts = []
    print("[*] 正在并行扫描开放22端口的主机，可能需要数秒钟...")
    
    # 步骤1：找到所有开了 SSH 服务的主机
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        future_to_ip = {executor.submit(check_port, str(ip), PORT): str(ip) for ip in network.hosts()}
        for future in concurrent.futures.as_completed(future_to_ip):
            ip = future_to_ip[future]
            if future.result():
                if ip != local_ip:
                    alive_hosts.append(ip)

    if not alive_hosts:
        print("[-] 当前网段未发现开放SSH(22)端口的主机。")
        return None

    print(f"[*] 发现疑似设备: {', '.join(alive_hosts)}")

    # 取第一个开放22端口的设备
    valid_pi_ip = alive_hosts[0]
    print(f"[+] 成功定位到目标树莓派！IP地址为: {valid_pi_ip}")

    return valid_pi_ip

def auto_connect(ip):
    """根据系统环境选择最佳命令实现自动或半自动连接"""
    print(f"[*] 准备连接到 {USERNAME}@{ip} ...")
    system_type = platform.system().lower()
    
    if "windows" in system_type:
        print("[!] 提醒: Windows 系统下通常不支持自动传入SSH密码。")
        try:
            # 尝试将密码推入剪贴板方便用户迅速右键粘贴
            os.system(f"echo|set /p=\"{PASSWORD}\"| clip")
            print("[+] 密码已自动保存到剪贴板，当出现密码提示时请直接「右键粘贴」并按回车！")
        except:
            pass
        os.system(f"ssh -o StrictHostKeyChecking=no {USERNAME}@{ip}")
        
    else:
        # 类 Unix 系统 (Linux / Termux)
        # 判断能否使用 sshpass 工具自动传入密码
        if os.system("command -v sshpass > /dev/null 2>&1") == 0:
            print("[*] 成功检测到 sshpass，正在执行全自动登录...")
            os.environ['SSHPASS'] = PASSWORD
            os.system(f"sshpass -e ssh -o StrictHostKeyChecking=no {USERNAME}@{ip}")
        else:
            print("[!] 未能在环境中检测到 sshpass 工具！")
            print(f"[!] 如果你想在手机端(Termux)实现全自动免密连接，请执行下面指令安装工具: pkg install sshpass")
            print(f"\n[!] 正在直接调起 ssh，稍后请手动输入密码: {PASSWORD}")
            os.system(f"ssh -o StrictHostKeyChecking=no {USERNAME}@{ip}")

if __name__ == "__main__":
    try:
        pi_ip = scan_and_find_pi()
        if pi_ip:
            auto_connect(pi_ip)
        else:
            print("[-] 未能在网络中找到配置了该账号的树莓派。")
    except KeyboardInterrupt:
        print("\n[-] 连接终止。")

