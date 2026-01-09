# PcCTP 傻瓜式教程
(注:由于无win32机器,在win64上无论如何修改,都无法成功执行,无法进行测试,因此不再支持win32版本,仅支持win64与linux版本)
## 目录

- [项目结构](#1. 项目结构)
- [快速开始](#2. 快速开始，开箱即用)
  - [下载方式](#2.1 下载方式)
    - [git命令](#2.1.1 git命令)
    - [ZIP](#2.1.2 zip)
    - [pip](#2.1.3 pip命令)
  - [使用方式](#2.2 使用方式)
  - [测试用例](#2.3 测试用例)
    - [Terminal](#2.3.1 Terminal)
    - [IDE编辑器](#2.3.2 IDE编辑器)
- [编译环境要求](#3. 编译环境要求)
  - [Windows](#3.1 Windows 环境)
  - [Linux](#3.2 Linux 环境)
- [编译](#4. 编译)
  - [一键编译（推荐）](#4.1 一键编译（推荐）)
  - [platfrom](#4.3 --platform)
    - [win64](#4.3.1 win64)
    - [linux](#4.3.3 linux)
  - [更多选项](#4.4 更多选项)
- [输出目录](#5. 输出目录)
- [常见问题](#6. 常见问题)

(注：目录因为序号与标题之间有空格，在github上无法正常显示和使用，若需要目录，请下载后，使用Typora阅读使用 `Ctrl+左键` 可正常跳转，若用其他markdown编辑器无法正常使用，将目录中框号中的空格删去试试。)

------



## 1. 项目结构

```bash
PcCTP/
├── bind/          # 绑定C++代码与python代码的绑定层代码，否则会报错，找不到文件位置
├── build/          # 编译时临时文件，编译成功后可删除，一般被.ignore文件标注，被不会上传，但也有可能弄错被误上传到仓库
├── ctp/          # ctp接口的cpp源文件，不可删除，否则会报错，找不到文件位置
├── docs/          # 各类使用参考文档，可删除
├── flows/          # simple_test.py的MdApi.create_ftdc_md_api('./flows/')的缓存文件夹，一般被.ignore文件标注，被不会上传，可删除
├── PcCTP/          # 开箱即用的PcCTP包
│	├── __init__.py
│	├── enums.py
│	├── types.py
│	├── interface.py    # nanobind版可以删除它，Python C API版本不能删！
│	├── win64/          # Windows 64位版本
│	│   ├── __init__.py
│	│   ├── PcCTP.pyi    # 存根文件，不建议删除，删除后IDE会提示找不到PcCTP位置
│	│   ├── PcCTP.cp313-win_amd64.pyd
│	│   └── ...
│	└── linux/ # Linux 版本
│	    ├── __init__.py
│	    ├── PcCTP.pyi    # 存根文件，不建议删除，删除后IDE会提示找不到PcCTP位置
│	    ├── PcCTP.cpython-313-x86_64-linux-gnu.so
│	    └── ...
│
├── scripts/           # 存放构建 __init__.py 文件的脚本，文件夹及其文件不可删除
│	└── generate_init.py # 构建 __init__.py 文件的脚本，不可删除，否则会报错，找不到文件位置
├── src/          # 构建 PcCTP包 所需py文件，文件夹及其文件不可删除，否则会报错，找不到文件位置
│	├── enums.py      # PcCTP使用时用到的enums类，编译完后，会被复制到对应位置，不可删除，否则会报错，找不到文件位置
│	├── interface.py  # 接口，编译完后，会被复制到对应位置（Python C API才会用到，但nanobind版也不能删除，否则会报错，找不到文件位置）
│	├── PcCTP.pyi	  # PcCTP使用时用到的存根文件，编译完后，会被复制到对应位置，不可删除，否则会报错，找不到文件位置
│	├── types.py      # 类型文件，编译完后，会被复制到对应位置，不可删除，否则会报错，找不到文件位置
│	└── version.txt	  # 版本信息，编译时会读取该文件，不可删除，否则会报错，找不到文件位置
├── .gitattributes # 大文件上传github标注文件，可删除
├── .ignore			# 上传忽略文件标注文件，可删除
├── build.py          # 构建脚本，不可删除
├── CMakeLists.txt  # 编译脚本，不可删除
├── README.md      # 项目介绍，可删除
├── requirements.txt # 依赖库，如果已经按要求准备好环境和依赖，可以删除
└── simple_test.py # 测试用例
```



------



## 2. 快速开始，开箱即用

如果想省去编译构建环境、pip工具库等复杂操作，希望开箱即用的人，可以直接下载本项目后，将[PcCTP](PcCTP)文件夹移入自己的项目中，[PcCTP](PcCTP)是一个可迁移的完整的python包，可以根据系统环境，自动选择使用对应版本，目前支持 `win64` ，  `linux` 三个版本，不必担心系统差异导致的问题。

如果不介意构建自己的环境，想自己编译，则从 [3. 编译环境要求](#3. 编译环境要求) 开始阅读。

操作如下：

### 2.1 下载方式

#### 2.1.1 git命令

可以使用git命令下载，如下：

```bash
# 1.1 github
git clone https://github.com/Cambridge1995/PcCTP.git
# 1.2 使用SSH，速度更快（但需要你自己已经配置好你的公钥，详情步骤自己问AI）
git clone git@github.com:Cambridge1995/PcCTP.git
```

#### 2.1.2 zip

`github`：在项目页面点击 `<> code` 按钮，点击 `Download ZIP` 下载到本地。

#### 2.1.3 pip命令（待完善）

```bash
pip install PcCTP
```

### 2.2 使用方式

**第一步：** 若是使用的git命令或zip下载方式，先将项目压缩包解压。若是pip命令下载，直接跳到第三步。

**第二步：** 进入解压后的项目目录中，可以见到项目的目录结构如下：

```bash
PcCTP/
├── bind/          # 绑定C++代码与python代码的绑定层代码，不可删除
├── build/          # 编译时临时文件，编译成功后可删除，一般被.ignore文件标注，被不会上传，但也有可能弄错被误上传到仓库
├── ctp/          # ctp接口的cpp源文件，不可删除
├── docs/          # 各类使用文档
├── flows/          # simple_test.py的MdApi.create_ftdc_md_api('./flows/')的缓存文件夹，一般被.ignore文件标注，被不会上传
├── PcCTP/          # 开箱即用的PcCTP包
├── scripts/          # 构建 __init__.py 文件的脚本，不可删除
├── src/          # 构建 PcCTP包 所需py文件，不可删除
├── .gitattributes # 大文件上传github标注文件，可删除
├── .ignore			# 上传忽略文件标注文件，可删除
├── build.py          # 构建脚本，不可删除
├── CMakeLists.txt  # 编译脚本，不可删除
├── README.md      # 项目介绍，可删除
└── simple_test.py # 测试用例
```

将项目中的 `PcCTP/  # 开箱即用的PcCTP包` 包迁移到自己项目中。

**第三步：** 使用如下方式导入使用：

```python
from PcCTP import MdApi,PyMdSpi,TraderApi,PyTraderSpi
```

### 2.3 测试用例

#### 2.3.1 Terminal

直接使用 `simple_test.py` 进行测试：

```bash
# 在cmd/powershell或linux Terminal终端中，进入PcCTP文件夹中，执行以下命令
python simple_test.py
```

#### 2.3.2 IDE编辑器

直接启动 `simple_test.py` 进行测试。



------



## 3. 编译环境要求

### 3.1 Windows 环境

**Windows 64位编译**：

- Windows 10/11 (64位)
- Visual Studio 2017/2019/2022（含 C++ 构建工具）
- CMake 3.15+
- `requirement.txt` 中配备的依赖库


### 3.2 Linux 环境

- Ubuntu 20.04+ / CentOS 7+ / Debian 10+
- GCC 7+ 或 Clang 6+
- CMake 3.15+
- `requirement.txt` 中配备的依赖库

可以使用微软的`WSL2`进行 `linux` 环境的搭建，具体步骤见文档：[WSL编译教程.md](docs/WSL%E7%BC%96%E8%AF%91%E6%95%99%E7%A8%8B.md)



------



## 4. 编译

### 4.1 一键编译（推荐）

使用 Python 构建脚本，自动检测平台并编译：

```bash
# 自动检测当前平台并编译对应版本
python build.py
```

### 4.2 `--platform`

 **编译平台：** 现在支持两个平台`win64`、`linux`。指定平台编译：

```bash
# Windows 64位版本
python build.py --platform win64



# Linux 版本
python build.py --platform linux 
```

**注意：** 一般情况下，用不到该参数，因为已经配置了可以根据系统环境，自动选择编译版本。

#### 4.2.1 `win64`

```bash
# 编译（自动识别编译器，并编译64位）
python build.py
# 也可以指定参数（但没卵用，我都给你配好了，傻瓜式操作不做，非要脱裤子放屁干嘛）
python build.py --platform win64
```


#### 4.2.2 `linux`

```bash
# 编译（自动识别编译器，并编译linux版本）
python build.py
# 也可以指定参数（但没卵用，我都给你配好了，傻瓜式操作不做，非要脱裤子放屁干嘛）
python build.py --platform linux
```

linux需要配备linux相关环境，详情见文档： [WSL编译教程.md](docs/WSL%E7%BC%96%E8%AF%91%E6%95%99%E7%A8%8B.md)

### 4.3 更多选项

```bash
# Release 模式编译（默认）
python build.py --platform win64 --release

# Debug 模式编译
python build.py --platform win64 --debug

# 使用 nanobind 绑定层
python build.py --platform win64 --nanobind

# 清理后重新编译
python build.py --platform win64 --clean

# 查看帮助
python build.py --help
```



---



## 5. 输出目录

编译成功后，输出文件位于 `PcCTP/` 目录：

```bash
PcCTP/
├── __init__.py
├── enums.py
├── types.py
├── interface.py
├── win64/          # Windows 64位版本
│   ├── __init__.py
│   ├── PcCTP.pyi
│   ├── PcCTP.cp313-win_amd64.pyd
│   └── ...
├── linux/ # Linux 版本
│   ├── __init__.py
│   ├── PcCTP.pyi
│   ├── PcCTP.cpython-313-x86_64-linux-gnu.so
│   └── ...
│
build/              # 构建缓存目录（可删除）
├── win64/          # win64 构建缓存
└── linux/          # linux 构建缓存
```



---



## 6. 常见问题

### Q1: Linux 编译提示找不到 Python.h？
**A**: 安装 Python 开发头文件：

```bash
sudo apt-get install python3-dev
```

### Q2: 如何清理构建缓存？
**A**: 使用 `--clean` 参数：

```bash
python build.py --platform win64 --clean
```

---

**特别感谢:**

- [@上海期货信息技术有限公司](https://www.simnow.com.cn/) 提供的 [CTP](https://www.simnow.com.cn/static/apiDownload.action) 接口
