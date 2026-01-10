#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PcCTP Python 文件生成脚本

职责：
1. 读取版本号文件（从 src/version.txt, ctp/PC/version.txt, ctp/fix/version.txt）
2. 复制 src 下的文件到 PcCTP 目录
3. 生成 __init__.py 文件（平台特定和主入口）

支持命令行参数：
  --platform     目标平台 (win64, linux)
  --stage        执行阶段
                 - prepare: 编译前准备（复制文件，生成 __init__.py）
                 - finalize: 编译后整理（更新 __init__.py）
                 - all: 执行所有步骤
  --use-nanobind 是否使用 nanobind 绑定层（默认：OFF，使用 Python C API）

使用示例：
  python generate_init.py --platform win64 --stage prepare
  python generate_init.py --platform win64 --stage finalize
  python generate_init.py --platform win64 --stage all
"""

import sys
import re
import shutil
import argparse
from pathlib import Path


# =============================================================================
# 版本号读取
# =============================================================================

def read_version_file(file_path: Path) -> str:
    """读取版本号文件，去除首尾空白"""
    if file_path.exists():
        try:
            content = file_path.read_text(encoding='utf-8')
            return content.strip()
        except Exception as e:
            print(f"[警告] 无法读取版本文件 {file_path}: {e}")
    return "unknown"


def get_all_versions(project_root: Path) -> dict:
    """读取所有版本号

    Returns:
        dict: 包含 pcctp_version, ctp_version, fix_version 的字典
    """
    versions = {
        'pcctp_version': read_version_file(project_root / 'src' / 'version.txt'),
        'ctp_version': read_version_file(project_root / 'ctp' / 'PC' / 'version.txt'),
        'fix_version': read_version_file(project_root / 'ctp' / 'fix' / 'version.txt'),
    }
    return versions


# =============================================================================
# 文件复制
# =============================================================================

def copy_python_files(project_root: Path, target_platform: str) -> bool:
    """复制 src 下的 Python 文件到 PcCTP 目录

    Args:
        project_root: 项目根目录
        target_platform: 目标平台 (win64, win32, linux)

    Returns:
        bool: 是否成功
    """
    src_dir = project_root / 'src'
    pcctp_dir = project_root / 'PcCTP'

    # 需要复制的文件和目录
    items_to_copy = [
        ('enums.py', 'enums.py'),
        ('interface.py', 'interface.py'),
        ('types', 'types'),  # 整个目录
    ]

    print("[复制] Python 文件...")
    for src_name, dst_name in items_to_copy:
        src_path = src_dir / src_name
        dst_path = pcctp_dir / dst_name

        if not src_path.exists():
            print(f"[警告] 源文件不存在: {src_path}")
            continue

        try:
            if src_path.is_dir():
                # 复制目录
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                print(f"  [目录] {src_name} -> {dst_name}")
            else:
                # 复制文件
                shutil.copy2(src_path, dst_path)
                print(f"  [文件] {src_name} -> {dst_name}")
        except Exception as e:
            print(f"[错误] 复制失败 {src_name}: {e}")
            return False

    return True


def copy_type_stub_files(project_root: Path) -> bool:
    """复制 PcCTP.pyi 类型存根文件到所有平台文件夹

    Args:
        project_root: 项目根目录

    Returns:
        bool: 是否成功
    """
    src_stub = project_root / 'src' / 'PcCTP.pyi'
    pcctp_dir = project_root / 'PcCTP'

    if not src_stub.exists():
        print(f"[警告] 类型存根文件不存在: {src_stub}")
        return True  # 不是致命错误

    platforms = ['win64', 'linux']

    print("[复制] 类型存根文件...")
    for platform in platforms:
        dst_stub = pcctp_dir / platform / 'PcCTP.pyi'
        try:
            shutil.copy2(src_stub, dst_stub)
            print(f"  [文件] PcCTP.pyi -> {platform}/PcCTP.pyi")
        except Exception as e:
            print(f"[错误] 复制类型存根文件失败 ({platform}): {e}")
            return False

    return True


# =============================================================================
# 模板定义
# =============================================================================

# Python C API 平台特定模板
PLATFORM_INIT_TEMPLATE_PYTHON_C_API = '''# {module_name} - {platform} 平台特定模块
# 自动生成文件，请勿手动修改
# 运行平台
__platform__ = '{platform}'

# 导入模块的所有内容（包含 MdApi 等）
from .{module_name} import *

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
'''

# Python C API 主入口模板
MAIN_INIT_TEMPLATE_PYTHON_C_API = '''# {module_name} - CTP API Python 绑定
# 自动生成文件，请勿手动修改
# 模块常量
# =============================================================================

# PcCTP 模块版本号
__version__ = 'v{pcctp_version}'
# CTP 接口版本号
__ctp_version__ = '{ctp_version}'
# 采集库fix版本号
__fix_version__ = '{fix_version}'
# 版本类型
__version_type__ = '{version_type}'

__full_version__ = '{module_name} v{pcctp_version} | CTP {ctp_version} | FIX {fix_version} | {version_type} '
import platform
import os
from PcCTP.interface import *
# 获取操作系统类型
from PcCTP.types import *

# 通过系统类型导入对应的接口
if env == 'win64':  # windows64位
    from PcCTP.win64 import *
elif env == 'linux':  # linux
    from PcCTP.linux import *
else:  # 其他系统 暂不支持,如：macOS
    raise EnvironmentError('本CTP版本与当前系统不匹配')

# 导入所有枚举类（支持 from win64 import Direction, OffsetFlag 等）
from PcCTP.enums import *

# 定义__all__列表，包含所有导出的名称
__all__ = [
    "env",
    # 枚举类
{enum_names},
    # 方法
    "validate_direction", "validate_offset_flag",
    "validate_order_price_type", "get_direction_name",
    "get_offset_flag_name",
    # 错误码映射
    "reason_map",
    # PyCapsule 辅助函数
    "pycapsule_check_exact", "pycapsule_get_pointer", "pycapsule_new",
    # 字符串池监控和清理函数
    "cleanup_temporal_pools", "cleanup_instruments",
    "check_instrument_pool_size", "get_pool_sizes",
    # 核心类
    "PyMdSpi","MdApi","PyTradeSpi","TradeApi","Fix",
    # TypedDict（{type_count}个）
{type_names},
]
'''


# =============================================================================
# 代码生成辅助函数
# =============================================================================

def extract_enum_names(enums_file: Path) -> list:
    """从 enums.py 文件中提取所有枚举类名"""
    content = enums_file.read_text(encoding='utf-8')
    # 匹配 class XxxName(StrEnum): 或 class XxxName(IntEnum):
    pattern = r'^class (\w+)\((?:StrEnum|IntEnum)\):'
    matches = re.findall(pattern, content, flags=re.MULTILINE)
    return sorted(matches)


def extract_type_names(types_file: Path) -> list:
    """从 types/__init__.py 文件中提取所有类名"""
    content = types_file.read_text(encoding='utf-8')
    # 匹配 __all__ 中的类名，格式: "ClassName"
    pattern = r'^\s+"(\w+)"'
    matches = re.findall(pattern, content, flags=re.MULTILINE)
    return sorted(matches)


def format_list(items: list, indent: str = '    ') -> str:
    """将类名列表格式化为 __all__ 列表格式"""
    if not items:
        return ''

    lines = []
    for i, name in enumerate(items):
        if i == len(items) - 1:
            # 最后一个不加逗号
            lines.append(f'{indent}"{name}"')
        else:
            lines.append(f'{indent}"{name}",')
    return '\n'.join(lines)


# =============================================================================
# __init__.py 文件生成
# =============================================================================

def generate_platform_init(
    project_root: Path,
    target_platform: str,
    versions: dict,
    module_name: str = 'PcCTP'
) -> bool:
    """生成平台特定的 __init__.py 文件（PcCTP/{platform}/__init__.py）

    Args:
        project_root: 项目根目录
        target_platform: 目标平台 (win64, win32, linux)
        versions: 版本号字典
        module_name: 模块名称

    Returns:
        bool: 是否成功
    """
    output_dir = project_root / 'PcCTP' / target_platform
    output_file = output_dir / '__init__.py'

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用模板生成内容
    content = PLATFORM_INIT_TEMPLATE_PYTHON_C_API.format(
        module_name=module_name,
        platform=target_platform,
    )

    try:
        output_file.write_text(content, encoding='utf-8')
        print(f"[生成] {output_file.relative_to(project_root)}")
        return True
    except Exception as e:
        print(f"[错误] 生成文件失败: {e}")
        return False


def generate_main_init(
    project_root: Path,
    target_platform: str,
    versions: dict,
    module_name: str = 'PcCTP'
) -> bool:
    """生成主入口的 __init__.py 文件（PcCTP/__init__.py）

    Args:
        project_root: 项目根目录
        target_platform: 目标平台
        versions: 版本号字典
        module_name: 模块名称

    Returns:
        bool: 是否成功
    """
    output_file = project_root / 'PcCTP' / '__init__.py'

    # 提取枚举类名
    enums_file = project_root / 'src' / 'enums.py'
    if enums_file.exists():
        enum_names = extract_enum_names(enums_file)
        print(f"[提取] 从 enums.py 提取到 {len(enum_names)} 个枚举类")
    else:
        print(f"[警告] 找不到 {enums_file}，使用默认枚举列表")
        enum_names = []

    # 提取类型类名
    types_file = project_root / 'src' / 'types' / '__init__.py'
    if types_file.exists():
        type_names = extract_type_names(types_file)
        print(f"[提取] 从 types/__init__.py 提取到 {len(type_names)} 个类型类")
    else:
        print(f"[警告] 找不到 {types_file}，类型列表将为空")
        type_names = []

    # 格式化列表
    enum_names_str = format_list(enum_names)
    type_names_str = format_list(type_names)

    # 使用模板生成内容
    content = MAIN_INIT_TEMPLATE_PYTHON_C_API.format(
        module_name=module_name,
        version=versions['ctp_version'],
        platform=target_platform,
        version_type='PC',
        pcctp_version=versions['pcctp_version'],
        ctp_version=versions['ctp_version'],
        fix_version=versions['fix_version'],
        ctp_platform=target_platform,
        enum_names=enum_names_str if enum_names_str else '    # (无枚举类)',
        type_count=len(type_names),
        type_names=type_names_str if type_names_str else '    # (无类型类)',
    )

    try:
        output_file.write_text(content, encoding='utf-8')
        print(f"[生成] {output_file.relative_to(project_root)}")
        return True
    except Exception as e:
        print(f"[错误] 生成文件失败: {e}")
        return False


# =============================================================================
# 主函数
# =============================================================================

def run_prepare_stage(project_root: Path, target_platform: str) -> bool:
    """执行编译前准备阶段

    1. 复制 src 下的 Python 文件到 PcCTP 目录
    2. 复制类型存根文件
    3. 生成平台特定的 __init__.py 文件
    4. 生成主入口的 __init__.py 文件

    Args:
        project_root: 项目根目录
        target_platform: 目标平台

    Returns:
        bool: 是否成功
    """
    print("\n" + "=" * 60)
    print("编译前准备阶段")
    print("=" * 60)

    # 读取版本号
    versions = get_all_versions(project_root)
    print(f"[版本] PcCTP: {versions['pcctp_version']}")
    print(f"[版本] CTP: {versions['ctp_version']}")
    print(f"[版本] FIX: {versions['fix_version']}")

    # 复制 Python 文件
    if not copy_python_files(project_root, target_platform):
        return False

    # 复制类型存根文件
    if not copy_type_stub_files(project_root):
        return False

    # 生成平台特定的 __init__.py
    if not generate_platform_init(project_root, target_platform, versions):
        return False

    # 生成主入口的 __init__.py
    if not generate_main_init(project_root, target_platform, versions):
        return False

    return True


def run_finalize_stage(project_root: Path, target_platform: str) -> bool:
    """执行编译后整理阶段

    重新生成 __init__.py 文件以确保内容正确

    Args:
        project_root: 项目根目录
        target_platform: 目标平台

    Returns:
        bool: 是否成功
    """
    print("\n" + "=" * 60)
    print("编译后整理阶段")
    print("=" * 60)

    # 读取版本号
    versions = get_all_versions(project_root)

    # 重新生成平台特定的 __init__.py
    if not generate_platform_init(project_root, target_platform, versions):
        return False

    # 重新生成主入口的 __init__.py
    if not generate_main_init(project_root, target_platform, versions):
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='PcCTP Python 文件生成脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--platform',
        required=True,
        choices=['win64', 'linux'],
        help='目标平台'
    )
    parser.add_argument(
        '--stage',
        default='all',
        choices=['prepare', 'finalize', 'all'],
        help='执行阶段 (默认: all)'
    )
    parser.add_argument(
        '--use-nanobind',
        default='OFF',
        help='是否使用 nanobind 绑定层（默认：OFF）'
    )

    args = parser.parse_args()

    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    print(f"[信息] 目标平台: {args.platform}")
    print(f"[信息] 执行阶段: {args.stage}")

    # 根据阶段执行相应操作
    if args.stage == 'prepare':
        success = run_prepare_stage(project_root, args.platform)
    elif args.stage == 'finalize':
        success = run_finalize_stage(project_root, args.platform)
    elif args.stage == 'all':
        success = run_prepare_stage(project_root, args.platform)
        if success:
            success = run_finalize_stage(project_root, args.platform)
    else:
        print(f"[错误] 未知的阶段: {args.stage}")
        return 1

    if success:
        print("\n[完成] 所有操作成功")
        return 0
    else:
        print("\n[错误] 操作失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
