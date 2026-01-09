# PcCTP - win64 平台特定模块
# 自动生成文件，请勿手动修改
# 运行平台
__platform__ = 'win64'

# 导入模块的所有内容（包含 MdApi 等）
from .PcCTP import *

# 定义__all__列表
__all__ = [
    # PyCapsule 辅助函数
    "pycapsule_check_exact", "pycapsule_get_pointer", "pycapsule_new",
    # 字符串池监控和清理函数
    "cleanup_temporal_pools", "cleanup_instruments",
    "check_instrument_pool_size", "get_pool_sizes",
    # 核心类
    "MdApi","TradeApi","Fix",
]
