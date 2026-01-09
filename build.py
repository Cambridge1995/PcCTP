#!/usr/bin/env python3
"""
PcCTP 一键构建脚本

职责：
1. 创建各级文件夹
2. 协调 generate_init.py 进行 Python 文件生成和复制
3. 调用 CMake 进行 C++ 编译
4. 编译完成后，调用 generate_init.py 进行最后的整理

使用示例：
  python build.py                    # 自动检测当前平台并编译
  python build.py --platform win64   # 编译 Windows 64位版本
  python build.py --platform win32   # 编译 Windows 32位版本
  python build.py --platform linux   # 编译 Linux 版本
"""

import os
import sys
import subprocess
import argparse
import shutil
import platform
from pathlib import Path


def get_current_platform():
    """检测当前运行平台（只支持 win64 和 linux）"""
    system = platform.system()
    if system == "Windows":
        # Windows 只支持 64 位
        print(f"[调试] 检测到平台: win64")
        return "win64"
    elif system == "Linux":
        print(f"[调试] 检测到平台: linux")
        return "linux"
    else:
        return "unknown"


def validate_platform_environment(target_platform):
    """验证当前环境是否支持编译目标平台"""
    current_system = platform.system()

    if target_platform == "linux":
        if current_system != "Linux":
            print(f"[错误] 编译 Linux 版本需要在 Linux 系统上进行")
            print(f"[提示] 当前系统: {current_system}")
            return False
    elif target_platform == "win64":
        if current_system != "Windows":
            print(f"[错误] 编译 Windows 版本需要在 Windows 系统上进行")
            print(f"[提示] 当前系统: {current_system}")
            return False
        # 检查 Python 是否为 64 位
        if sys.maxsize <= 2**32:
            print(f"[错误] Windows 版本需要64位Python")
            print(f"[提示] 当前Python是32位，请使用64位Python")
            return False
    else:
        print(f"[错误] 不支持的平台: {target_platform}")
        print(f"[提示] 支持的平台: win64, linux")
        return False

    return True


def get_cmake_generator(args):
    """获取 CMake 生成器"""
    if args.generator:
        return args.generator

    # 自动选择生成器
    system = platform.system()
    if system == "Windows":
        try:
            result = subprocess.run(
                ["cmake", "--help"],
                capture_output=True,
                timeout=10
            )

            # 使用 bytes 解码，避免编码问题
            content = result.stdout.decode('utf-8', errors='ignore')
            lines = content.split('\n')

            in_generators = False
            available_generators = []

            for line in lines:
                line_stripped = line.strip()
                if line_stripped == "Generators":
                    in_generators = True
                    continue
                if in_generators:
                    if not line_stripped:
                        break
                    # 生成器行的特征：包含 = 且包含已知的关键字
                    if '=' in line_stripped and ('Visual Studio' in line_stripped or 'Borland' in line_stripped or 'NMake' in line_stripped or 'MinGW' in line_stripped or 'Ninja' in line_stripped or 'Unix' in line_stripped):
                        # 提取等号前的部分作为生成器名称
                        parts = line_stripped.split('=', 1)
                        gen_part = parts[0].strip()
                        # 移除可能的 * 前缀（表示默认）
                        if gen_part.startswith('*'):
                            gen_part = gen_part[1:].strip()
                        if gen_part:
                            available_generators.append(gen_part)

            if available_generators:
                print("[信息] 检测到可用的 CMake 生成器:")
                for gen in available_generators[:15]:  # 只显示前15个
                    print(f"  - {gen}")
                if len(available_generators) > 15:
                    print(f"  ... 还有 {len(available_generators) - 15} 个生成器")

                # 按优先级匹配生成器
                preferred_generators = [
                    "Visual Studio 17 2022",
                    "Visual Studio 16 2019",
                    "Visual Studio 15 2017",
                ]

                for preferred in preferred_generators:
                    if preferred in available_generators:
                        print(f"[选择] 使用生成器: {preferred}")
                        return preferred

                # 如果没有找到首选生成器，使用任何可用的 Visual Studio 生成器
                for gen in available_generators:
                    if "Visual Studio" in gen:
                        print(f"[选择] 使用生成器: {gen}")
                        return gen

        except Exception as e:
            print(f"[警告] 无法检测可用的生成器: {e}")

        # 默认生成器
        print("[选择] 使用默认生成器")
        return ""  # 让 CMake 使用默认生成器
    else:
        return "Unix Makefiles"


def create_directories(target_platform):
    """创建必要的目录结构"""
    project_root = Path(__file__).parent
    directories = [
        project_root / "build" / target_platform,
        project_root / "PcCTP" / "win64",
        project_root / "PcCTP" / "linux",
        project_root / "PcCTP" / "types",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[创建] 目录: {directory.relative_to(project_root)}")


def run_generate_init(target_platform, stage="all"):
    """运行 generate_init.py 脚本

    Args:
        target_platform: 目标平台 (win64, win32, linux)
        stage: 执行阶段
            - "prepare": 编译前准备（复制 src 文件，生成 __init__.py）
            - "finalize": 编译后整理（更新 __init__.py）
            - "all": 执行所有步骤
    """
    project_root = Path(__file__).parent
    script_path = project_root / "scripts" / "generate_init.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--platform", target_platform,
        "--stage", stage
    ]

    print(f"[执行] generate_init.py (stage: {stage})...")
    print(f"[命令] {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[错误] generate_init.py 执行失败 (stage: {stage})")
        return False

    return True


def run_cmake_build(target_platform, build_type, generator):
    """运行 CMake 配置和编译

    Args:
        target_platform: 目标平台 (win64, win32, linux)
        build_type: 构建类型 (Release, Debug, RelWithDebInfo)
        generator: CMake 生成器
    """
    build_dir = f"build/{target_platform}"

    # 准备 CMake 配置命令
    cmake_config_args = ["cmake", "-S", ".", "-B", build_dir]

    # 添加构建类型
    cmake_config_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

    # 添加平台参数
    cmake_config_args.append(f"-DCTP_PLATFORM={target_platform}")

    # 使用 Python C API 绑定层
    print("[配置] 使用 Python C API 绑定层")

    # 添加生成器和架构参数（Windows）
    if platform.system() == "Windows":
        # 只有当生成器不为空时才添加 -G 参数
        if generator:
            cmake_config_args.extend(["-G", generator])
            print(f"[配置] 使用生成器: {generator}")

        # Windows只支持64位，指定x64架构
        if target_platform == "win64":
            cmake_config_args.extend(["-A", "x64"])
            print("[配置] 目标架构: x64 (64位)")

    # 配置 CMake
    print(f"[步骤1] 配置 CMake (目标: {target_platform})...")
    print(f"[命令] {' '.join(cmake_config_args)}")

    result = subprocess.run(cmake_config_args)
    if result.returncode != 0:
        print("[错误] CMake 配置失败")
        return False

    # 构建命令
    cmake_build_args = [
        "cmake",
        "--build", build_dir,
        "--config", build_type,
        "--",  # 分隔符，之后的参数传递给原生构建工具
    ]

    # 添加并行编译参数
    if platform.system() == "Linux":
        # Linux/Unix系统使用 -j 参数启用并行编译
        import os
        # 使用CPU核心数，最多4个并行作业
        jobs = min(os.cpu_count() or 1, 4)
        cmake_build_args.extend(["-j", str(jobs)])

    # 构建
    print(f"[步骤2] 编译项目 ({build_type}, {target_platform})...")
    print(f"[命令] {' '.join(cmake_build_args)}")

    result = subprocess.run(cmake_build_args)
    if result.returncode != 0:
        print("[错误] 编译失败")
        return False

    return True


def setup_linux_symlinks():
    """为 Linux 平台创建 CTP 库的符号链接（带 lib 前缀）"""
    if platform.system() != "Linux":
        return

    ctp_lib_dir = "ctp/PC/linux"
    if not os.path.exists(ctp_lib_dir):
        print(f"[警告] CTP 库目录不存在: {ctp_lib_dir}")
        return

    # 需要创建符号链接的库文件
    libraries = [
        "thostmduserapi_se.so",
        "thosttraderapi_se.so"
    ]

    for lib in libraries:
        src = os.path.join(ctp_lib_dir, lib)
        dst = os.path.join(ctp_lib_dir, f"lib{lib}")

        # 如果符号链接已存在且有效，跳过
        if os.path.islink(dst):
            if os.path.exists(dst):
                print(f"[存在] 符号链接: {dst}")
                continue
            else:
                # 删除无效的符号链接
                os.remove(dst)

        # 如果目标文件已存在（不是符号链接），跳过避免覆盖
        if os.path.exists(dst):
            print(f"[跳过] 目标已存在: {dst}")
            continue

        # 创建符号链接
        try:
            os.symlink(lib, dst)
            print(f"[创建] 符号链接: {dst} -> {lib}")
        except OSError as e:
            print(f"[错误] 无法创建符号链接 {dst}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="PcCTP 一键构建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--platform",
        choices=["win64", "linux", "auto"],
        default="auto",
        help="目标平台 (默认: auto - 自动检测当前平台)"
    )
    parser.add_argument(
        "--generator",
        help="CMake 生成器 (如 'Visual Studio 17 2022')"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Release 模式编译（默认）"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug 模式编译"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理构建目录后重新构建"
    )

    args = parser.parse_args()

    # 确定目标平台
    if args.platform == "auto":
        target_platform = get_current_platform()
        print(f"[检测] 自动检测平台: {target_platform}")
    else:
        target_platform = args.platform
        print(f"[指定] 目标平台: {target_platform}")

    # 验证环境
    if not validate_platform_environment(target_platform):
        return 1

    # 确定构建类型
    build_type = "Release" if args.release else ("Debug" if args.debug else "RelWithDebInfo")

    # 清理构建目录（如果需要）
    if args.clean:
        build_dir = Path("build") / target_platform
        if build_dir.exists():
            print(f"[清理] 删除构建目录: {build_dir}")
            shutil.rmtree(build_dir)

    # Linux 平台：创建 CTP 库符号链接
    if target_platform == "linux":
        setup_linux_symlinks()

    # 获取 CMake 生成器
    generator = get_cmake_generator(args)

    print("\n" + "=" * 60)
    print("开始构建流程")
    print("=" * 60)

    # 步骤1: 创建目录结构
    print("\n[步骤0] 创建目录结构...")
    create_directories(target_platform)

    # 步骤2: 编译前准备 - 复制 src 文件，生成初始 __init__.py
    print("\n[步骤1] 编译前准备...")
    if not run_generate_init(target_platform, "prepare"):
        return 1

    # 步骤3: CMake 编译
    print("\n[步骤2] C++ 编译...")
    if not run_cmake_build(target_platform, build_type, generator):
        return 1

    # 步骤4: 编译后整理 - 更新 __init__.py
    print("\n[步骤3] 编译后整理...")
    if not run_generate_init(target_platform, "finalize"):
        return 1

    # 成功
    print("\n" + "=" * 60)
    print("[完成] 构建成功！")
    print(f"[输出] PcCTP/{target_platform}/")
    print(f"[提示] 可以将 PcCTP/ 目录作为 Python 包使用")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
