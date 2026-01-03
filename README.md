# PcCTP 傻瓜式教程

## 1.项目结构



## 2. 开箱即用

如果想省去编译构建环境、pip工具库等复杂操作，希望开箱即用的人，可以直接下载本项目后，将[PcCTP](PcCTP)文件夹移入自己的项目中，[PcCTP](PcCTP)是一个可迁移的完整的python包，可以根据系统环境，自动选择使用对应版本，目前支持 `win64` ， `win32` ， `linux` 三个版本，不必担心系统差异导致的问题。

如果不介意构建自己的环境，想自己编译，则从 [2. 参数选择](#2. 参数选择) 开始阅读。

操作如下：

### 2.1 下载方式

#### 2.1.1 git命令

可以使用git命令下载，如下：

```bash
# 1. github
git clone https://github.com/
# 2. gitee
git clone https://gitee.com/
```

#### 2.1.2  zip

`github`：在项目页面点击 `<> code` 按钮，点击 `Download ZIP` 下载到本地。

`gitee`：在项目页面点击 `克隆/下载` 按钮，点击 `下载ZIP` 下载到本地。

#### 2.1.3 pip命令（待完善）

```bash
pip install PcCTP
```

### 2.2 使用方式

**第一步：** 若是使用的git命令或zip下载方式，先将项目压缩包解压。若是pip命令下载，直接跳到第三步。

**第二步：** 进入解压后的项目目录中，可以见到项目的目录结构如下：

```
PcCTP/
├── bind/          # 绑定C++代码与python代码的绑定层代码，不可删除
├── build/          # 编译时临时文件，编译成功后可删除，一般被不会上传，有可能弄错被误上传到仓库
├── ctp/          # ctp接口的cpp源文件，不可删除
├── docs/          # 各类使用文档
├── flows/          # simple_test.py的MdApi.create_ftdc_md_api('./flows/')的缓存文件夹
├── PcCTP/          # 开箱即用的PcCTP包
├── scripts/          # 构建 __init__.py 文件的脚本，不可删除
├── src/          # 构建 PcCTP包 所需py文件，不可删除
├── build.py          # 构建脚本，不可删除
├── CMakeLists.txt  # 编译脚本，不可删除
├── README.md      # 项目介绍，可删除
└── simple_test.py # 测试用例
```

将项目中，`PcCTP/  # 开箱即用的PcCTP包` 包迁移到自己项目中。

**第三步：** 使用如下方式导入使用：

```python
from PcCTP import MdApi,PyMdSpi,TraderApi,TraderSpi
```

### 2.2 测试用例

#### 2.2.1 Terminal

直接使用 `simple_test.py` 进行测试：

```
# 在cmd/powershell或linux Terminal终端中，进入PcCTP文件夹中，执行以下命令
python simple_test.py
```

#### 2.2.2 IDE编辑器

直接启动 `simple_test.py` 进行测试。

## 2. 参数选择

### 2.1 `--nanobind`

**目前支持使用两种库进行绑定编译：**1.  `nanobind`   2. `Python C API`

**使用方法：** 使用 `--nanobind` 参数可以选择 `nanobind` 库进行构建，若是不加参数，则默认编译 `Python C API` 版本。

**使用两种方式的原因：** 刚开始打算省事，跟AI沟通后，发现 `nanobind` 库算是众多绑定库中运行速度比较快的，编码方式比较简洁，开发难度低。所以就选用了 `nanobind` ，结果在运行时，出现退出码为  `-1073741819 (0xC0000005)` 的诡异情况，虽然影响不大，但我这人有强迫症，非要解决，折腾了几天，解决不了，本能的以为是 `nanobind` 库自带的问题，就彻底重构，改用了 `Python C API` 。

结果没想到 `Python C API` 也出现同样的问题，就一点一点的排查，最后发现与 `nanobind` 库无关，而是我自定义的 `zero_copy_uitls.h` 库有bug，所以又回头修改了 `nanobind` 版。

也就留下了两个版本。

**区别：** 至于二者有何区别，`Python C API` 编码更接近C，编程量大些，代码量更多、更复杂； `nanobind` 更接近C++，语法更简洁，代码量更少；其他区别我就不谈了，估计最关心的还是执行效率问题，AI分析的理论对比图如下：

性能差异量化（高频回调场景）

对于深度行情回调（最关键的性能路径）：

| 操作         | nanobind 开销 | Python C API 开销 |
| ------------ | ------------- | ----------------- |
| GIL 获取     | ~20-30 ns     | ~10-15 ns         |
| 属性方法查找 | ~50-100 ns    | ~20-30 ns         |
| 字典构建     | ~200-300 ns   | ~150-200 ns       |
| 回调调用     | ~100-150 ns   | ~50-80 ns         |
| 单次回调总计 | ~370-580 ns   | ~230-325 ns       |
Python C API 版本大约快 30-50%（在高频回调场景下）

总结：

| 维度       | nanobind               | Python C API       |
| ---------- | ---------------------- | ------------------ |
| 执行速度   | 较慢（抽象开销）       | 更快（直接调用）   |
| 开发效率   | 高（现代 C++）         | 低（手动管理）     |
| 代码可读性 | 好（简洁）             | 差（冗长）         |
| 维护性     | 好（类型安全）         | 差（易错）         |
| 内存安全   | 好（自动 RAII）        | 差（手动管理）     |
| 适合场景   | 通用应用、开发效率优先 | 高频交易、性能关键 |























## 快速开始

### 1.1 一键编译（推荐）

使用 Python 构建脚本，自动检测平台并编译：

```bash
# 自动检测当前平台并编译
python build.py
```

### 1.2 指定平台编译

```bash
# Windows 64位版本
python build.py --platform win64

# Windows 32位版本
python build.py --platform win32

# Linux 版本
python build.py --platform linux
```

### 1.3 更多选项

```bash
# Release 模式编译（默认）
python build.py --platform win64 --release

# Debug 模式编译
python build.py --platform win64 --debug

# 使用 nanobind 绑定层
python build.py --platform win64 --nanobind

# 清理后重新编译
python build.py --platform win64 --clean

# 指定 CMake 生成器
python build.py --platform win32 --generator "Visual Studio 17 2022"

# 查看帮助
python build.py --help
```

---

## 2. 编译环境要求

### 2.1 Windows 环境

**Windows 64位编译**：
- Windows 10/11 (64位)
- Visual Studio 2017/2019/2022（含 C++ 构建工具）
- CMake 3.15+
- Python 3.8+ (64位)
- numpy
- nanobind

**Windows 32位编译**：
- 选项1：Windows 32位系统 + 32位 Python + 32位编译器
- 选项2：Windows 64位系统 + 64位 Python + Visual Studio（使用 `-A Win32` 参数）

### 2.2 Linux 环境

- Ubuntu 20.04+ / CentOS 7+ / Debian 10+
- GCC 7+ 或 Clang 6+
- CMake 3.15+
- Python 3.8+ 开发头文件
- numpy
- nanobind

安装依赖：
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev python3-numpy
```

---

## 3. 手动编译（不使用 build.py）

如果您不想使用 `build.py`，可以直接使用 CMake 命令：

### 3.1 Windows 64位

```powershell
cmake -S . -B build/win64 -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build/win64 --config Release
```

### 3.2 Windows 32位

```powershell
cmake -S . -B build/win32 -G "Visual Studio 17 2022" -A Win32 -DCMAKE_BUILD_TYPE=Release
cmake --build build/win32 --config Release
```

### 3.3 Linux

```bash
cmake -S . -B build/linux -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux --config Release
```

---

## 4. 输出目录

编译成功后，输出文件位于 `PcCTP/` 目录：

```
PcCTP/
├── win32/          # Windows 32位版本
│   ├── PcCTP.cp313-win32.pyd
│   ├── enums.py
│   ├── types.py
│   ├── interface.py
│   └── ...
├── win64/          # Windows 64位版本
│   ├── PcCTP.cp313-win_amd64.pyd
│   ├── enums.py
│   ├── types.py
│   ├── interface.py
│   └── ...
└── linux/          # Linux 版本
    ├── PcCTP.cpython-313-x86_64-linux-gnu.so
    ├── enums.py
    ├── types.py
    ├── interface.py
    └── ...

build/              # 构建缓存目录（可删除）
├── win32/          # win32 构建缓存
├── win64/          # win64 构建缓存
└── linux/          # linux 构建缓存
```

---

## 5. 使用方法

将编译好的 `PcCTP/` 目录复制到您的项目中，即可作为 Python 包使用：

```python
# 导入 PcCTP 模块
from PcCTP.win64 import PcCTP, enums

# 或直接添加到路径
import sys
sys.path.insert(0, '/path/to/PcCTP/win64')
import PcCTP
```

---

## 6. 常见问题

### Q1: 在 Windows 64位系统上编译 32位版本失败？
**A**: 确保安装了 Visual Studio 的 C++ 构建工具，并使用：
```bash
python build.py --platform win32 --generator "Visual Studio 17 2022"
```

### Q2: Linux 编译提示找不到 Python.h？
**A**: 安装 Python 开发头文件：
```bash
sudo apt-get install python3-dev
```

### Q3: 如何清理构建缓存？
**A**: 使用 `--clean` 参数：
```bash
python build.py --platform win64 --clean
```

---

**特别感谢:**

- @上期技术公司 提供的 [CTP](https://www.simnow.com.cn/) 接口
- @shizhuolin 提供的项目 [PyCTP](https://github.com/shizhuolin/PyCTP) 作为参考
