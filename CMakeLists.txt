cmake_minimum_required(VERSION 3.15)
project(PyCTP LANGUAGES CXX)

# 设置策略以消除 CMP0144 警告
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

# ==============================================================================
# 版本配置
# ==============================================================================
# 读取 CTP 版本号
if(EXISTS "${CMAKE_SOURCE_DIR}/ctp/PC/version.txt")
    file(READ "${CMAKE_SOURCE_DIR}/ctp/PC/version.txt" CTP_VERSION_PC)
    string(STRIP "${CTP_VERSION_PC}" CTP_VERSION_PC)
endif()

# 读取 PcCTP 版本号
if(EXISTS "${CMAKE_SOURCE_DIR}/src/version.txt")
    file(READ "${CMAKE_SOURCE_DIR}/src/version.txt" PcCTP_VERSION)
    string(STRIP "${PcCTP_VERSION}" PcCTP_VERSION)
else()
    set(PcCTP_VERSION "1.0.0")
endif()

# 读取 FIX 库版本号
if(EXISTS "${CMAKE_SOURCE_DIR}/ctp/fix/version.txt")
    file(READ "${CMAKE_SOURCE_DIR}/ctp/fix/version.txt" FIX_VERSION)
    string(STRIP "${FIX_VERSION}" FIX_VERSION)
else()
    set(FIX_VERSION "unknown")
endif()

# ==============================================================================
# 平台检测
# ==============================================================================
# 从命令行获取目标平台
if(NOT DEFINED CTP_PLATFORM)
    # 如果未指定，自动检测
    if(WIN32)
        # Windows 只支持 64 位
        set(CTP_PLATFORM "win64")
    elseif(UNIX AND NOT APPLE)
        set(CTP_PLATFORM "linux")
    else()
        message(FATAL_ERROR "不支持的平台: ${CMAKE_SYSTEM_NAME}")
    endif()
endif()

message(STATUS "平台: ${CTP_PLATFORM}")

# ==============================================================================
# C++ 标准设置
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# 编译选项
# ==============================================================================
if(MSVC)
    # MSVC 编译选项
    add_compile_options(/W4 /utf-8)
    # 性能优化
    add_compile_options(/O2 /GL /Ob2 /Oi /Ot /Oy /GT)
    # 禁用增量链接（与 /LTCG 冲突）
    add_link_options(/LTCG /INCREMENTAL:NO)
    # 禁用特定警告
    add_compile_options(/wd4100 /wd4267 /wd4244 /wd4800)
    # 运行时库设置为多线程 DLL
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang 编译选项
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    # Python C API 相关警告禁用
    add_compile_options(-Wno-write-strings)
    add_compile_options(-Wno-cast-function-type)
    add_compile_options(-Wno-missing-field-initializers)

    # 性能优化
    add_compile_options(-O3 -march=native -flto)
    add_link_options(-flto)
endif()

# ==============================================================================
# 查找 Python
# ==============================================================================
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# ==============================================================================
# 查找 numpy (用于数组零拷贝)
# ==============================================================================
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "numpy include dir: ${NUMPY_INCLUDE_DIR}")

# ==============================================================================
# CTP 头文件和库文件路径
# ==============================================================================
set(CTP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ctp/PC/${CTP_PLATFORM}")
set(CTP_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/ctp/PC/${CTP_PLATFORM}")

# FIX 模块路径
set(CTP_FIX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ctp/fix")
set(CTP_FIX_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/ctp/fix/${CTP_PLATFORM}")

# 库文件名称
if(WIN32)
    set(CTP_MD_LIBRARY "thostmduserapi_se")
    set(CTP_TRADE_LIBRARY "thosttraderapi_se")
    set(CTP_FIX_LIBRARY "FixWinDataCollect")
elseif(UNIX)
    set(CTP_MD_LIBRARY "thostmduserapi_se")
    set(CTP_TRADE_LIBRARY "thosttraderapi_se")
    set(CTP_FIX_LIBRARY "FixLinuxDataCollect")
endif()

# ==============================================================================
# 创建 Python C API 模块
# ==============================================================================
# 输出目录
set(CTP_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/PcCTP/${CTP_PLATFORM}")
set(CTP_BASE_DIR "${CMAKE_SOURCE_DIR}/PcCTP")

# 模块名称
set(CTP_MODULE_NAME "PcCTP")

# 源文件列表（行情接口 MdApi + 交易接口 TradeApi + FIX 采集库）
set(PYTHONC_SOURCE_FILES
    bind/PythonC/common.h
    bind/PythonC/util.h
    bind/PythonC/util.cpp
    bind/PythonC/md.h
    bind/PythonC/md_api.cpp
    bind/PythonC/md_spi.cpp
    bind/PythonC/trade.h
    bind/PythonC/trade_api.cpp
    bind/PythonC/trade_spi.cpp
    bind/PythonC/fix.cpp
)

# 创建 Python 模块（使用 Python C API）
if(WIN32)
    # Windows: 创建共享库（.pyd）
    Python3_add_library(${CTP_MODULE_NAME} MODULE ${PYTHONC_SOURCE_FILES})

    # 设置输出名称为 PcCTP.pyd
    set_target_properties(${CTP_MODULE_NAME} PROPERTIES OUTPUT_NAME "PcCTP" SUFFIX ".pyd")
elseif(UNIX)
    # Linux: 创建共享库（.so）
    Python3_add_library(${CTP_MODULE_NAME} MODULE ${PYTHONC_SOURCE_FILES})

    # 设置输出名称
    set_target_properties(${CTP_MODULE_NAME} PROPERTIES OUTPUT_NAME "PcCTP" PREFIX "")
endif()

# 包含目录
target_include_directories(${CTP_MODULE_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CTP_INCLUDE_DIR}
    ${CTP_FIX_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
)

# 链接库（CTP 行情库 + CTP 交易库 + FIX 采集库）
if(WIN32)
    # 添加库目录到链接路径
    target_link_directories(${CTP_MODULE_NAME} PRIVATE
        ${CTP_LIBRARY_DIR}
        ${CTP_FIX_LIBRARY_DIR}
    )
    target_link_libraries(${CTP_MODULE_NAME} PRIVATE
        "${CTP_MD_LIBRARY}.lib"
        "${CTP_TRADE_LIBRARY}.lib"
        "${CTP_FIX_LIBRARY}.lib"
    )
elseif(UNIX)
    target_link_libraries(${CTP_MODULE_NAME} PRIVATE
        "${CTP_LIBRARY_DIR}/lib${CTP_MD_LIBRARY}.so"
        "${CTP_LIBRARY_DIR}/lib${CTP_TRADE_LIBRARY}.so"
        "${CTP_FIX_LIBRARY_DIR}/lib${CTP_FIX_LIBRARY}.so"
    )
endif()

# Windows 特定设置
if(WIN32)
    target_compile_definitions(${CTP_MODULE_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

# ==============================================================================
# 输出和复制
# ==============================================================================
# 复制编译后的模块文件到输出目录
add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CTP_OUTPUT_DIR}"
    COMMENT "Creating output directory: ${CTP_OUTPUT_DIR}"
)

add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:${CTP_MODULE_NAME}>"
        "${CTP_OUTPUT_DIR}/$<TARGET_FILE_NAME:${CTP_MODULE_NAME}>"
    COMMENT "Copying ${CTP_MODULE_NAME} module to ${CTP_OUTPUT_DIR}"
)

# 复制 CTP MD API DLL/SO
if(WIN32)
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CTP_LIBRARY_DIR}/${CTP_MD_LIBRARY}.dll"
            "${CTP_OUTPUT_DIR}/"
        COMMENT "Copying CTP MD API DLL"
    )
elseif(UNIX)
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CTP_LIBRARY_DIR}/${CTP_MD_LIBRARY}.so"
            "${CTP_OUTPUT_DIR}/"
        COMMENT "Copying CTP MD API SO"
    )
endif()

# 复制 CTP 交易 API DLL/SO
if(WIN32)
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CTP_LIBRARY_DIR}/${CTP_TRADE_LIBRARY}.dll"
            "${CTP_OUTPUT_DIR}/"
        COMMENT "Copying CTP Trade API DLL"
    )
elseif(UNIX)
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CTP_LIBRARY_DIR}/${CTP_TRADE_LIBRARY}.so"
            "${CTP_OUTPUT_DIR}/"
        COMMENT "Copying CTP Trade API SO"
    )
endif()

# 复制 FIX 采集库 DLL/SO
if(WIN32)
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CTP_FIX_LIBRARY_DIR}/${CTP_FIX_LIBRARY}.dll"
            "${CTP_OUTPUT_DIR}/"
        COMMENT "Copying FIX Data Collection DLL"
    )
elseif(UNIX)
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CTP_FIX_LIBRARY_DIR}/lib${CTP_FIX_LIBRARY}.so"
            "${CTP_OUTPUT_DIR}/"
        COMMENT "Copying FIX Data Collection SO"
    )
endif()

# ==============================================================================
# 显示构建信息
# ==============================================================================
message(STATUS "==================================================")
message(STATUS "PyCTP 构建配置")
message(STATUS "==================================================")
message(STATUS "目标平台: ${CTP_PLATFORM}")
message(STATUS "输出目录: ${CTP_OUTPUT_DIR}")
message(STATUS "绑定层: Python C API")
message(STATUS "当前编译: 行情 (MdApi) + 交易 (TradeApi) + FIX (Fix)")
message(STATUS "必须依赖:")
message(STATUS "  - CTP 行情 API: ${CTP_MD_LIBRARY}")
message(STATUS "  - CTP 交易 API: ${CTP_TRADE_LIBRARY}")
message(STATUS "  - FIX 采集库: ${CTP_FIX_LIBRARY}")
message(STATUS "源文件:")
foreach(SOURCE_FILE ${PYTHONC_SOURCE_FILES})
    message(STATUS "  - ${SOURCE_FILE}")
endforeach()
message(STATUS "==================================================")
