/**
 * PcCTP - FIX穿透式监管数据采集模块
 *
 * 本模块提供FIX数据采集库的Python接口
 * 主要用于中继模式下采集终端系统信息
 *
 * 使用示例：
 *   from PcCTP import Fix
 *   system_info = Fix.collect_system_info()
 *   version = Fix.get_fix_api_version()
 */

#include <Python.h>
#include <cstring>
#include <string>

// FIX数据采集库头文件
// 注意：链接库在 CMakeLists.txt 中配置
#include "ctp/fix/FixDataCollect.h"

// =============================================================================
// 模块级方法实现
// =============================================================================

/**
 * @brief 采集系统信息
 *
 * 返回：bytes对象，包含采集的系统信息（至少270字节）
 *       用于中继模式下传递给RegisterUserSystemInfo或SubmitUserSystemInfo
 *
 * 注意：
 * 1. 采集库不是线程安全的，多线程调用时需要加锁
 * 2. 采集的信息是二进制数据，不是字符串
 * 3. 直连模式不需要调用此函数，CTP会自动采集
 *
 * @return PyObject* bytes对象或NULL（失败时）
 */
static PyObject* fix_collect_system_info(PyObject* self, PyObject* args) {
    (void)self;  // 未使用
    (void)args;  // 无参数

    // 分配缓冲区（至少344字节，确保安全）
    char buffer[344];
    int len = 0;

    // 调用FIX采集库
    int result = CTP_GetSystemInfo(buffer, len);

    // 检查采集结果
    if (result != 0) {
        // 采集失败，返回详细错误信息
        std::string error_msg = "Failed to collect system info. Error code: " +
                                std::to_string(result) + "\nDetails:\n";

        // Windows/Linux的错误位定义
        if (result & (0x01 << 0)) error_msg += "  - Terminal type not collected\n";
        if (result & (0x01 << 1)) error_msg += "  - Collection time error\n";
        if (result & (0x01 << 2)) error_msg += "  - IP address not collected\n";
        if (result & (0x01 << 3)) error_msg += "  - MAC address not collected\n";
        if (result & (0x01 << 4)) error_msg += "  - Device name not collected\n";
        if (result & (0x01 << 5)) error_msg += "  - OS version not collected\n";
        if (result & (0x01 << 6)) error_msg += "  - Hard disk serial number not collected\n";
        if (result & (0x01 << 7)) error_msg += "  - CPU serial number not collected\n";
        if (result & (0x01 << 8)) error_msg += "  - BIOS not collected\n";

        #if defined(_WIN32) || defined(_WIN64)
        if (result & (0x01 << 9)) error_msg += "  - Disk partition info not collected\n";
        #endif

        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    }

    // 验证采集长度
    if (len <= 0 || len > 344) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid system info length collected");
        return NULL;
    }

    // 返回bytes对象（不是字符串！）
    return PyBytes_FromStringAndSize(buffer, len);
}

/**
 * @brief 获取FIX采集库版本
 *
 * @return PyObject* 版本字符串
 */
static PyObject* fix_get_fix_api_version(PyObject* self, PyObject* args) {
    (void)self;  // 未使用
    (void)args;  // 无参数

    const char* version = CTP_GetDataCollectApiVersion();
    return PyUnicode_FromString(version);
}

// =============================================================================
// FIX 模块方法表
// =============================================================================

static PyMethodDef fix_module_methods[] = {
    {"collect_system_info", fix_collect_system_info, METH_NOARGS,
     "Collect system information for relay mode. Returns bytes object."},
    {"get_fix_api_version", fix_get_fix_api_version, METH_NOARGS,
     "Get FIX data collection API version."},
    {NULL}
};

// =============================================================================
// 模块定义
// =============================================================================

static PyModuleDef PcCTP_fix_module = {
    PyModuleDef_HEAD_INIT,
    "PcCTP.Fix",  // 模块名
    "CTP FIX Data Collection Module (穿透式监管数据采集模块)\n\n"
    "Usage:\n"
    "  from PcCTP import Fix\n"
    "  system_info = Fix.collect_system_info()\n"
    "  version = Fix.get_fix_api_version()\n\n"
    "Note: This module is only needed for relay mode clients.\n"
    "Direct connection mode doesn't need to call collect_system_info(),\n"
    "CTP will automatically collect and report terminal information.",
    -1,
    fix_module_methods,
};

// =============================================================================
// 创建 FIX 子模块的函数（在主模块初始化时调用）
// =============================================================================

/**
 * @brief 创建并初始化 FIX 子模块
 *
 * 此函数在主模块 PcCTP 初始化时调用，创建 Fix 子模块
 * 使用方式：import PcCTP.Fix 或 from PcCTP import Fix
 *
 * @param parent_module 父模块（PcCTP）
 * @return int 成功返回 0，失败返回 -1
 */
extern "C" int initialize_fix_submodule(PyObject* parent_module) {
    // 创建 FIX 子模块
    PyObject* fix_module = PyModule_Create(&PcCTP_fix_module);
    if (!fix_module) {
        return -1;
    }

    // 将子模块添加到父模块
    if (PyModule_AddObject(parent_module, "Fix", fix_module) < 0) {
        Py_DECREF(fix_module);
        return -1;
    }

    // 同时将子模块添加到 sys.modules，使得 import PcCTP.Fix 可以工作
    PyObject* sys = PyImport_ImportModule("sys");
    if (sys) {
        PyObject* modules = PyObject_GetAttrString(sys, "modules");
        if (modules) {
            PyDict_SetItemString(modules, "PcCTP.Fix", fix_module);
            Py_DECREF(modules);
        }
        Py_DECREF(sys);
    }

    return 0;
}
