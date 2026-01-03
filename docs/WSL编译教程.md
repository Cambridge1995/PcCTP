# PcCTP WSL Ubuntu 编译完整教程

## 目录
1. [环境准备](#环境准备)
2. [安装编译工具](#安装编译工具)
3. [安装Python和依赖](#安装python和依赖)
4. [准备CTP库文件](#准备ctp库文件)
5. [编译项目](#编译项目)
6. [常见问题](#常见问题)

---

## 环境准备

### 1. 更新WSL和Ubuntu系统

```bash
# 更新软件包列表
sudo apt update

# 升级已安装的软件包
sudo apt upgrade -y
```

### 2. 检查系统版本

```bash
# 查看Ubuntu版本
lsb_release -a

# 查看内核版本
uname -r
```

---

## 安装编译工具

### 1. 安装基础编译工具

```bash
# 安装build-essential (包含GCC, G++, make等)
sudo apt install -y build-essential

# 安装CMake (需要3.15+版本)
sudo apt install -y cmake

# 验证版本
gcc --version    # 应该显示 9.0+ 或更高
g++ --version    # 应该显示 9.0+ 或更高
cmake --version  # 应该显示 3.15+ 或更高
```

### 2. 如果CMake版本过低，手动安装最新版

```bash
# 下载最新版CMake (以3.30.0为例)
wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.tar.gz

# 解压到/opt目录
sudo tar -zxvf cmake-3.30.0-linux-x86_64.tar.gz -C /opt

# 创建软链接
sudo ln -sf /opt/cmake-3.30.0-linux-x86_64/bin/cmake /usr/local/bin/cmake
sudo ln -sf /opt/cmake-3.30.0-linux-x86_64/bin/ctest /usr/local/bin/ctest

# 验证版本
cmake --version
```

---

## 安装Python和依赖

### 推荐方式: 使用 Miniconda (版本管理)

Miniconda 是轻量级的 conda 安装器，推荐用于管理 Python 版本和依赖。

#### 1. 安装 Miniconda

```bash
# 下载 Miniconda 安装脚本 (Linux 64位)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 运行安装脚本
bash Miniconda3-latest-Linux-x86_64.sh

# 安装过程说明:
# - 按回车查看许可协议
# - 输入 yes 同意协议
# - 按回车使用默认安装位置 ~/miniconda3
# - 输入 yes 初始化 conda (自动配置环境变量)

# 重新加载shell配置
source ~/.bashrc
```

#### 2. 验证安装

```bash
# 检查conda版本
conda --version

# 检查Python版本
python --version
```
如果接受使用conda默认python环境,可以直接跳到第4步,若是想要隔离环境,则进行第3步创建新环境,事后可以使用命令`conda env remove --name <环境名称>`删除环境。
#### 3. 创建 PcCTP 专用环境

```bash
# 创建名为 pcctp 的环境，指定Python版本
conda create -n pcctp python=3.10 -y

# 激活环境
conda activate pcctp

# 提示符会变为: (pcctp) username@hostname:~$
```

#### 4. 安装项目依赖

```bash
# 安装编译所需的核心包
pip install numpy

# 安装nanobind (使用pip，conda可能没有最新版)
pip install nanobind

# 验证安装
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import nanobind; print('nanobind: OK')"
```

#### 5. Conda 常用命令

```bash
# 查看所有环境
conda env list

# 激活环境
conda activate pcctp

# 退出环境
conda deactivate

# 删除环境
conda remove -n pcctp --all

# 搜索可用包
conda search python
```

---

### 备用方式: 系统原生 Python (不推荐)

如果不想使用 conda，可以直接使用系统 Python (但版本管理会受限)：

```bash
# 安装Python 3和开发头文件
sudo apt install -y python3 python3-pip python3-dev

# 安装依赖
python3 -m pip install --upgrade pip
python3 -m pip install numpy nanobind
```

---

## 准备CTP库文件

### 1. 将项目文件复制到WSL

有两种方式：

#### 方式A: 使用Windows路径访问 (推荐)

WSL可以直接访问Windows文件系统，你的项目位于：
```bash
cd /mnt/d/project/PcCTP
```

#### 方式B: 复制到WSL本地文件系统

```bash
# 创建项目目录
mkdir -p ~/projects
cd ~/projects

# 从Windows复制项目
cp -r /mnt/d/project/PcCTP .

# 进入项目目录
cd PcCTP
```

### 2. 检查CTP库文件

确保Linux版本的CTP库文件已就位：

```bash
# 检查PC接口库文件
ls -la ctp/PC/linux/

# 应该包含以下文件：
# - libthostmduserapi_se.so (行情API)
```

**如果缺少Linux版本的CTP库文件，需要从SimNow官网下载：**
1. 访问：http://www.simnow.com.cn/
2. 注册并下载Linux版本的CTP API
3. 将库文件放置到 `ctp/PC/linux/` 目录

---

## 编译项目

### 1. 激活 conda 环境

```bash
# 激活之前创建的 pcctp 环境
conda activate pcctp

# 确认环境已激活 (提示符应显示 (pcctp))
```

### 2. 进入项目目录

```bash
cd /mnt/d/project/PcCTP
# 或 cd ~/projects/PcCTP (如果已复制到本地)
```

### 3. 执行编译

```bash
# 编译Linux版本 (conda环境中使用 python 命令)
python build.py --platform linux

# 或者使用默认自动检测
python build.py
```

### 4. 编译选项

```bash
# 查看帮助
python build.py --help

# 使用nanobind绑定层 (可选)
python build.py --platform linux --nanobind

# Release模式编译
python build.py --platform linux --release

# 清理后重新编译
python build.py --platform linux --clean
```

### 5. 编译成功输出

```
==================================================
[完成] 构建成功！
[输出] PcCTP/linux/
[提示] 可以将 PcCTP/ 目录作为 Python 包使用
==================================================
```

---

## 常见问题

### Q1: CMake版本过低

```bash
error: CMake 3.15 or higher is required
```

**解决:** 按照"安装编译工具"章节中的步骤手动安装最新版CMake。

### Q2: 找不到Python开发头文件

```bash
fatal error: Python.h: No such file or directory
```

**解决:** 安装python3-dev
```bash
sudo apt install -y python3-dev
```

### Q3: numpy模块未找到

```bash
ModuleNotFoundError: No module named 'numpy'
```

**解决:** 安装numpy
```bash
python3 -m pip install numpy
```

### Q4: 权限问题

```bash
permission denied: ./build.py
```

**解决:** 给脚本添加执行权限
```bash
chmod +x build.py
python3 build.py --platform linux
```

### Q5: 找不到CTP库文件

```bash
error: cannot find -lthostmduserapi_se
```

**解决:** 确保CTP Linux库文件已正确放置
```bash
ls -la ctp/PC/linux/libthostmduserapi_se.so
```

### Q6: 编译64位但需要32位

Linux下本项目默认编译64位版本。如需32位，需要安装32位编译工具链：

```bash
# 安装32位库
sudo apt install -y gcc-multilib g++-multilib

# 修改CMakeLists.txt添加-m32编译选项
```

---

## 快速一键安装脚本

将以下内容保存为 `setup_wsl_conda.sh` 并执行：

```bash
#!/bin/bash

echo "==================================="
echo "PcCTP WSL编译环境一键安装脚本"
echo "使用 Miniconda 进行Python版本管理"
echo "==================================="

# 更新系统
echo "[1/7] 更新系统..."
sudo apt update && sudo apt upgrade -y

# 安装编译工具
echo "[2/7] 安装编译工具..."
sudo apt install -y build-essential cmake wget

# 下载并安装 Miniconda
echo "[3/7] 安装 Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
    rm ~/miniconda.sh

    # 初始化 conda
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
else
    echo "Miniconda 已安装，跳过..."
fi

# 创建 conda 环境
echo "[4/7] 创建 PcCTP 专用环境..."
source ~/.bashrc
conda create -n pcctp python=3.10 -y
conda activate pcctp

# 安装 Python 依赖
echo "[5/7] 安装 Python 依赖..."
conda install -y numpy
pip install nanobind

# 验证安装
echo "[6/7] 验证安装..."
echo "=== 编译工具 ==="
gcc --version | head -n1
cmake --version
echo ""
echo "=== Python 环境 ==="
python --version
echo ""
echo "=== Python 包 ==="
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import nanobind; print('nanobind: OK')"

echo ""
echo "[7/7] 检查CTP库文件..."
if [ -f "ctp/PC/linux/libthostmduserapi_se.so" ]; then
    echo "✓ CTP Linux库文件: 存在"
else
    echo "⚠ 警告: CTP Linux库文件不存在，请从SimNow官网下载"
fi

echo ""
echo "==================================="
echo "安装完成！"
echo "==================================="
echo "请执行以下命令编译项目："
echo "  conda activate pcctp"
echo "  cd /mnt/d/project/PcCTP"
echo "  python build.py --platform linux"
echo "==================================="
```

使用方法：
```bash
chmod +x setup_wsl_conda.sh
./setup_wsl_conda.sh
```

---

## 验证编译结果

编译成功后，验证输出：

```bash
# 查看输出目录
ls -la PcCTP/linux/

# 应该包含：
# - PcCTP.cpython-3xx-x86_64-linux-gnu.so (编译的模块)
# - libthostmduserapi_se.so (CTP API库)
# - __init__.py (Python包初始化文件)
# - enums.py, types.py, interface.py (辅助模块)
```

---

## 联系支持

如遇到其他问题，请提供以下信息：

1. WSL版本: `wsl --list --verbose`
2. Ubuntu版本: `lsb_release -a`
3. GCC版本: `gcc --version`
4. CMake版本: `cmake --version`
5. Python版本: `python3 --version`
6. 完整错误信息
