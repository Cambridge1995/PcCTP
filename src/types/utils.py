"""
PcCTP 工具模块
"""

import platform


def get_os_type():
    """
    获取操作系统类型:
    - win64: Windows (仅支持64位)
    - linux: Linux
    - macos: macOS
    - other: 其他系统
    """
    system = platform.system().lower()

    if system == "windows":
        # Windows 只支持 64 位
        return 'win64'
    elif system == "linux":
        return 'linux'
    elif system == "darwin":
        return 'macos'
    else:
        return 'other'

# 调用方法获取当前系统类型
env = get_os_type()