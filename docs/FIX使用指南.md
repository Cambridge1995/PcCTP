# FIX穿透式监管数据采集使用指南

## 概述

`PcCTP.fix` 模块提供了CTP穿透式监管数据采集库的Python接口，主要用于**中继模式**下采集终端系统信息。

## 导入方式

```python
from PcCTP import fix
```

## 可用方法

### 1. `fix.get_fix_api_version()`

获取FIX采集库的版本信息。

```python
from PcCTP import fix

version = fix.get_fix_api_version()
print(f"FIX采集库版本: {version}")
```

### 2. `fix.collect_system_info()`

采集当前终端的系统信息。

**返回值**：`bytes` 对象，包含采集的系统信息（至少270字节）

**注意**：
- 采集的信息是**二进制数据**，不是字符串
- 采集库**不是线程安全的**，多线程调用时需要加锁
- 如果采集失败，会抛出 `RuntimeError` 异常

```python
from PcCTP import fix

system_info = fix.collect_system_info()
print(f"采集到 {len(system_info)} 字节的系统信息")
```

## 使用场景

### 场景1：直连模式（个人交易者、量化私募）

**不需要调用FIX采集库！**

CTP会自动采集并上报终端信息：

```python
from PcCTP.trade import TradeApi

class MySpi:
    def on_front_connected(self):
        # 直连模式：直接登录即可，CTP自动采集
        api.req_authenticate({
            'broker_id': '9999',
            'user_id': '123456',
            'app_id': 'MY_APP_1.0',
            'auth_code': 'AUTH_CODE_HERE'
        })

    def on_rsp_authenticate(self, rsp_authenticate, rsp_info, request_id, is_last):
        if rsp_info and rsp_info['error_id'] == 0:
            api.req_user_login({...})

api = TradeApi.create_ftdc_trader_api()
api.register_front('tcp://180.168.146.187:10201')
api.register_spi(MySpi())
api.init()
```

### 场景2：中继模式 - 客户端（软件公司的用户）

客户端需要采集系统信息并发送给中继服务器：

```python
from PcCTP import fix
import socket

# 1. 采集系统信息
system_info = fix.collect_system_info()

# 2. 发送给中继服务器
def send_to_relay(system_info, app_id):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('relay.example.com', 8888))

    # 发送采集信息和中继登录请求
    login_request = {
        'broker_id': '9999',
        'user_id': 'client_001',
        'app_id': app_id,
        'system_info': system_info.hex()  # 转为十六进制字符串传输
    }

    sock.sendall(json.dumps(login_request).encode())
    response = sock.recv(4096)
    sock.close()
    return response

# 使用
send_to_relay(system_info, 'CLIENT_APP_1.0')
```

### 场景3：中继模式 - 中继服务器（软件公司）

中继服务器接收客户端的采集信息，注册到CTP：

```python
from PcCTP.trade import TradeApi
import json
import socket

class RelayServer:
    def __init__(self):
        self.api = TradeApi.create_ftdc_trader_api()
        self.api.register_front('tcp://180.168.146.187:10201')
        self.api.register_spi(self)
        self.api.init()

    def on_front_connected(self):
        # 中继服务器认证
        self.api.req_authenticate({
            'broker_id': '9999',
            'user_id': 'relay_server',
            'app_id': 'RELAY_APP_1.0',
            'auth_code': 'RELAY_AUTH_CODE'
        })

    def on_rsp_authenticate(self, rsp_authenticate, rsp_info, request_id, is_last):
        if rsp_info and rsp_info['error_id'] == 0:
            # 认证成功，使用操作员登录
            self.api.req_user_login({
                'broker_id': '9999',
                'user_id': 'operator_001',
                'password': 'password',
                'client_ip_address': '127.0.0.1',
                'client_ip_port': 8888
            })

    def handle_client_login(self, client_data):
        """处理客户端登录请求"""
        # 解析客户端发来的采集信息
        system_info = bytes.fromhex(client_data['system_info'])
        client_app_id = client_data['app_id']
        client_id = client_data['user_id']

        # 注册客户端系统信息（多对多模式）
        self.api.register_user_system_info({
            'broker_id': '9999',
            'user_id': client_id,
            'client_system_info': system_info,
            'client_system_info_len': len(system_info),
            'client_public_ip': '192.168.1.100',
            'client_ip_port': 8888,
            'client_app_id': client_app_id
        })

        # 或一对多模式：使用SubmitUserSystemInfo
        # self.api.submit_user_system_info({...})
```

### 场景4：量化团队机房托管

**不需要FIX采集库！** 只是物理位置更近，仍然是直连模式：

```python
from PcCTP.trade import TradeApi

# 程序部署在期货公司机房，直连CTP柜台
api = TradeApi.create_ftdc_trader_api()
api.register_front('tcp://10.0.0.5:47005')  # 内网地址，超低延迟
api.register_spi(MySpi())
api.init()

# 和在本地运行一模一样！
```

## 错误处理

### 采集失败

如果采集失败，会抛出 `RuntimeError` 异常，包含详细的错误信息：

```python
from PcCTP import fix

try:
    system_info = fix.collect_system_info()
except RuntimeError as e:
    print(f"采集失败: {e}")
    # 输出示例：
    # 采集失败: Failed to collect system info. Error code: 7
    # Details:
    #   - Hard disk serial number not collected
    #   - CPU serial number not collected
```

### 错误代码说明

| 错误位 | 含义 |
|--------|------|
| 0 | 终端类型未采集到 |
| 1 | 信息采集时间获取异常 |
| 2 | IP地址获取失败 |
| 3 | MAC地址获取失败 |
| 4 | 设备名获取失败 |
| 5 | 操作系统版本获取失败 |
| 6 | 硬盘序列号获取失败 |
| 7 | CPU序列号获取失败 |
| 8 | BIOS获取失败 |
| 9 | 系统盘分区信息获取失败（仅Windows） |

## 线程安全

**采集库不是线程安全的！** 如果在多线程环境中使用，需要加锁：

```python
from PcCTP import fix
import threading

fix_lock = threading.Lock()

def collect_in_thread():
    with fix_lock:
        system_info = fix.collect_system_info()
        return system_info

# 使用
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(collect_in_thread)
    system_info = future.result()
```

## 常见问题

### Q1: 为什么有时候采集失败？

**A:** 可能的原因：
- 权限不足（Linux需要u+s权限）
- 缺少系统组件（Windows缺少WMI等服务）
- 使用了虚拟机或容器（某些信息采集不到）

解决方法：
- 下载[穿透式采集自检工具](https://www.simnow.com.cn/DocumentDown/api_3/5_2_8/tool250107.zip)检查
- 确保有足够的系统权限
- 在真实的物理机上运行

### Q2: 采集到的信息可以直接用吗？

**A:** 不可以！这是二进制数据，不是字符串。正确用法：

```python
# 正确 ✅
system_info = fix.collect_system_info()
api.register_user_system_info({
    'client_system_info': system_info,  # 直接传递bytes
    'client_system_info_len': len(system_info)
})

# 错误 ❌
system_info = fix.collect_system_info()
print(system_info.decode('utf-8'))  # 不要试图解码为字符串！
```

### Q3: 直连模式需要调用FIX采集吗？

**A:** 不需要！直连模式下CTP会自动采集并上报终端信息，你只需要正常登录即可。

### Q4: 如何判断我需要使用中继模式吗？

**A:** 简单判断：

| 你的身份 | 是否需要FIX采集 |
|----------|----------------|
| 个人交易者（单/多账户） | ❌ 不需要 |
| 量化交易团队 | ❌ 不需要 |
| 量化私募 | ❌ 不需要 |
| 程序化交易者 | ❌ 不需要 |
| 软件公司（向用户提供SaaS） | ✅ 需要（客户端采集） |
| 软件公司（中继服务器） | ❌ 不需要（接收客户端采集的信息） |

## 完整示例：中继模式客户端

```python
#!/usr/bin/env python3
"""
中继模式客户端示例
软件公司的用户端程序
"""

from PcCTP import fix
import json
import socket

def login_to_relay_server():
    """登录到中继服务器"""

    # 1. 采集系统信息（关键步骤！）
    print("正在采集系统信息...")
    try:
        system_info = fix.collect_system_info()
        print(f"采集成功！信息长度: {len(system_info)} 字节")
    except RuntimeError as e:
        print(f"采集失败: {e}")
        return

    # 2. 准备登录数据
    login_data = {
        'broker_id': '9999',
        'user_id': 'user_001',
        'password': 'password123',
        'app_id': 'CLIENT_APP_1.0',
        'auth_code': 'CLIENT_AUTH_CODE',
        'system_info': system_info.hex()  # 转为十六进制便于传输
    }

    # 3. 连接中继服务器
    print("连接中继服务器...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('relay.example.com', 8888))

        # 4. 发送登录请求
        sock.sendall(json.dumps(login_data).encode())

        # 5. 接收响应
        response = sock.recv(4096)
        result = json.loads(response.decode())

        if result.get('success'):
            print("登录成功！")
            return True
        else:
            print(f"登录失败: {result.get('message')}")
            return False

    finally:
        sock.close()

if __name__ == '__main__':
    # 先检查FIX版本
    print(f"FIX采集库版本: {fix.get_fix_api_version()}")
    print()

    # 登录中继服务器
    login_to_relay_server()
```

## 参考文档

- [CTP看穿式监管数据采集说明](https://ctpdoc.jedore.top/6.7.11/CTSJGSJCJJK/_CTSJGSJCJJK/)
- [CTP_GetSystemInfo详解](https://ctpdoc.jedore.top/6.7.11/CTSJGSJCJJK/CTP-GETSYSTEMINFO/)
- [常见FAQ](https://ctpdoc.jedore.top/6.7.11/CTSJGSJCJJK/CJFAQ/)
