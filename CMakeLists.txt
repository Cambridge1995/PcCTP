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
# 构建选项
# ==============================================================================
option(BUILD_PC "Build PC version" ON)
option(USE_NANOBIND "Use nanobind binding (default: Python C API)" OFF)

# ==============================================================================
# 平台检测
# ==============================================================================
# 检测当前平台和架构
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(CTP_PLATFORM "win32")
        set(CTP_PLATFORM_SHORT "win32")
        set(CTP_ARCH_32BIT TRUE)
    else()
        set(CTP_PLATFORM "win64")
        set(CTP_PLATFORM_SHORT "win64")
        set(CTP_ARCH_32BIT FALSE)
    endif()
elseif(UNIX AND NOT APPLE)
    set(CTP_PLATFORM "linux")
    set(CTP_PLATFORM_SHORT "linux")
    set(CTP_ARCH_32BIT FALSE)
else()
    message(FATAL_ERROR "不支持的平台: ${CMAKE_SYSTEM_NAME}")
endif()

message(STATUS "平台: ${CTP_PLATFORM}")
message(STATUS "编译版本: PC=${BUILD_PC}")

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
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# ==============================================================================
# 自动查找 nanobind
# ==============================================================================
if(NOT nanobind_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import nanobind, os; print(os.path.dirname(nanobind.__file__))"
        OUTPUT_VARIABLE NANOBIND_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(EXISTS "${NANOBIND_ROOT}/cmake/nanobind-config.cmake")
        set(nanobind_DIR "${NANOBIND_ROOT}/cmake")
    endif()
endif()

find_package(nanobind CONFIG REQUIRED)

# ==============================================================================
# 查找 numpy (用于数组零拷贝)
# ==============================================================================
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "numpy include dir: ${NUMPY_INCLUDE_DIR}")

# ==============================================================================
# FIX 模块路径（共用）
# ==============================================================================
set(CTP_FIX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ctp/fix")
set(CTP_FIX_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/ctp/fix/${CTP_PLATFORM}")

# ==============================================================================
# 辅助函数：创建 PyCTP 模块
# ==============================================================================
function(create_pyctp_module CTP_VERSION CTP_ROOT_DIR MODULE_SUFFIX SOURCE_FILE)
    # 版本类型标识
    set(VERSION_TYPE_DISPLAY "PC")

    # CTP 头文件和库文件路径
    set(CTP_MD_INCLUDE_DIR "${CTP_ROOT_DIR}/${CTP_PLATFORM}")
    set(CTP_MD_LIBRARY_DIR "${CTP_ROOT_DIR}/${CTP_PLATFORM}")

    # 库文件名称
    if(WIN32)
        set(CTP_MD_LIBRARY "thostmduserapi_se")
        set(CTP_FIX_LIBRARY "FixWinDataCollect")
    elseif(UNIX)
        set(CTP_MD_LIBRARY "thostmduserapi_se")
        set(CTP_FIX_LIBRARY "FixLinuxDataCollect")
    endif()

    # 输出目录
    set(CTP_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/PcCTP/${CTP_PLATFORM}")
    # 基础输出目录（用于公共文件）
    set(CTP_BASE_DIR "${CMAKE_SOURCE_DIR}/PcCTP")

    # 模块名称
    set(CTP_MODULE_NAME "PcCTP")
    set(CTP_MODULE_FILE_NAME "${CTP_MODULE_NAME}${MODULE_SUFFIX}")

    # 创建 nanobind 模块（使用指定的源文件）
    nanobind_add_module(${CTP_MODULE_NAME} ${SOURCE_FILE})

    # 包含目录
    target_include_directories(${CTP_MODULE_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CTP_MD_INCLUDE_DIR}
        ${CTP_FIX_INCLUDE_DIR}
        ${NUMPY_INCLUDE_DIR}
    )

    # 链接库
    if(WIN32)
        target_link_libraries(${CTP_MODULE_NAME} PRIVATE
            "${CTP_MD_LIBRARY_DIR}/${CTP_MD_LIBRARY}.lib"
        )
    elseif(UNIX)
        target_link_libraries(${CTP_MODULE_NAME} PRIVATE
            "${CTP_MD_LIBRARY_DIR}/${CTP_MD_LIBRARY}.so"
        )
    endif()

    # Windows 特定设置
    if(WIN32)
        target_compile_definitions(${CTP_MODULE_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()

    # 设置模块输出名称
    set_target_properties(${CTP_MODULE_NAME} PROPERTIES OUTPUT_NAME "${CTP_MODULE_FILE_NAME}")

    # 复制文件到输出目录
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CTP_OUTPUT_DIR}"
        COMMENT "Creating output directory: ${CTP_OUTPUT_DIR}"
    )

    # 复制 Python 模块文件
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:${CTP_MODULE_NAME}>"
            "${CTP_OUTPUT_DIR}/${CTP_MODULE_FILE_NAME}"
        COMMENT "Copying ${CTP_MODULE_NAME} module to ${CTP_OUTPUT_DIR}"
    )

    # 复制 CTP MD API DLL/SO
    if(WIN32)
        add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CTP_MD_LIBRARY_DIR}/${CTP_MD_LIBRARY}.dll"
                "${CTP_OUTPUT_DIR}/"
            COMMENT "Copying CTP MD API DLL"
        )
    elseif(UNIX)
        add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CTP_MD_LIBRARY_DIR}/${CTP_MD_LIBRARY}.so"
                "${CTP_OUTPUT_DIR}/"
            COMMENT "Copying CTP MD API SO"
        )
    endif()

    # 复制 FIX DLL/SO
    if(EXISTS "${CTP_FIX_LIBRARY_DIR}")
        if(WIN32)
            if(EXISTS "${CTP_FIX_LIBRARY_DIR}/${CTP_FIX_LIBRARY}.dll")
                add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CTP_FIX_LIBRARY_DIR}/${CTP_FIX_LIBRARY}.dll"
                        "${CTP_OUTPUT_DIR}/"
                    COMMENT "Copying CTP FIX DLL"
                )
            endif()
        elseif(UNIX)
            if(EXISTS "${CTP_FIX_LIBRARY_DIR}/lib${CTP_FIX_LIBRARY}.so")
                add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CTP_FIX_LIBRARY_DIR}/lib${CTP_FIX_LIBRARY}.so"
                        "${CTP_OUTPUT_DIR}/"
                    COMMENT "Copying CTP FIX SO"
                )
            endif()
        endif()
    endif()

    # 复制 src/enums.py 模块文件（枚举实现）到基础目录
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/enums.py"
            "${CTP_BASE_DIR}/enums.py"
        COMMENT "Copying enums module file to ${CTP_BASE_DIR}"
    )

    # 复制 types.py 模块文件（类型定义）到基础目录
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/types.py"
            "${CTP_BASE_DIR}/types.py"
        COMMENT "Copying types module file to ${CTP_BASE_DIR}"
    )

    # 复制 interface.py 模块文件（接口定义）到基础目录
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/interface.py"
            "${CTP_BASE_DIR}/interface.py"
        COMMENT "Copying interface module file to ${CTP_BASE_DIR}"
    )

    # 确保所有平台目录都存在
    foreach(PLAT win32 win64 linux)
        add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CTP_BASE_DIR}/${PLAT}"
            COMMENT "Ensuring platform directory exists: ${CTP_BASE_DIR}/${PLAT}/"
        )
    endforeach()

    # 复制 PcCTP.pyi 类型存根文件到所有平台文件夹
    foreach(PLAT win32 win64 linux)
        add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/src/PcCTP.pyi"
                "${CTP_BASE_DIR}/${PLAT}/PcCTP.pyi"
            COMMENT "Copying PcCTP.pyi to ${CTP_BASE_DIR}/${PLAT}/"
        )
    endforeach()

    # 生成平台特定的 __init__.py 文件（使用 Python 脚本）
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env
        "CTP_OUTPUT_DIR=${CTP_OUTPUT_DIR}"
        "CTP_BASE_DIR=${CTP_BASE_DIR}"
        "CTP_VERSION=${CTP_VERSION}"
        "CTP_PLATFORM=${CTP_PLATFORM}"
        "VERSION_TYPE_DISPLAY=${VERSION_TYPE_DISPLAY}"
        "PCCTP_VERSION=${PcCTP_VERSION}"
        "FIX_VERSION=${FIX_VERSION}"
        "MODULE_NAME=${CTP_MODULE_NAME}"
        "USE_NANOBIND=${USE_NANOBIND}"
        "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/generate_init.py"
        COMMENT "Generating platform-specific __init__.py for ${VERSION_TYPE_DISPLAY}"
    )

    # 生成主 PcCTP/__init__.py 文件（使用 Python 脚本）
    add_custom_command(TARGET ${CTP_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env
        "CTP_BASE_DIR=${CTP_BASE_DIR}"
        "CTP_VERSION=${CTP_VERSION}"
        "CTP_PLATFORM=${CTP_PLATFORM}"
        "VERSION_TYPE_DISPLAY=${VERSION_TYPE_DISPLAY}"
        "PCCTP_VERSION=${PcCTP_VERSION}"
        "FIX_VERSION=${FIX_VERSION}"
        "MODULE_NAME=${CTP_MODULE_NAME}"
        "GENERATE_MAIN_INIT=1"
        "USE_NANOBIND=${USE_NANOBIND}"
        "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/generate_init.py"
        COMMENT "Generating main PcCTP/__init__.py"
    )

    message(STATUS "配置 ${VERSION_TYPE_DISPLAY} 版本 -> ${CTP_OUTPUT_DIR}")
endfunction()

# ==============================================================================
# 创建 PC 版本
# ==============================================================================
if(BUILD_PC)
    if(WIN32)
        set(PC_MODULE_SUFFIX ".cp313-win_amd64.pyd")
    elseif(UNIX)
        set(PC_MODULE_SUFFIX ".cpython-313-x86_64-linux-gnu.so")
    endif()

    # 根据选项选择绑定层
    if(USE_NANOBIND)
        set(BIND_SOURCE_FILE "bind/nanobind/md.cpp")
        set(BIND_TYPE "nanobind")
    else()
        set(BIND_SOURCE_FILE "bind/PythonC/md.cpp")
        set(BIND_TYPE "Python C API")
    endif()

    create_pyctp_module("${CTP_VERSION_PC}" "${CMAKE_SOURCE_DIR}/ctp/PC" "${PC_MODULE_SUFFIX}" "${BIND_SOURCE_FILE}")
endif()

# ==============================================================================
# 显示构建信息
# ==============================================================================
message(STATUS "==================================================")
message(STATUS "PyCTP 构建配置")
message(STATUS "==================================================")
if(BUILD_PC)
    message(STATUS "PC 版本: ON -> PcCTP/${CTP_PLATFORM}/")
    message(STATUS "绑定层: ${BIND_TYPE}")
    message(STATUS "源文件: ${BIND_SOURCE_FILE}")
endif()
message(STATUS "目标平台: ${CTP_PLATFORM}")
message(STATUS "==================================================")
