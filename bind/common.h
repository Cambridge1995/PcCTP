/**
 * PcCTP - 通用基类和宏定义
 *
 * 用途：
 * - 抽象 MdApi 和 TradeApi 的共同代码
 * - 提供 SPI 回调的通用宏
 * - 减少 50-70% 的重复代码
 */

#pragma once

#include <Python.h>
#include <structmember.h>
#include "util.h"

// =============================================================================
// Python 版本兼容性宏
// =============================================================================
// Py_IsFinalizing() 在 Python 3.13+ 才可用
#if PY_VERSION_HEX >= 0x030D0000
    #define PCCTP_PY_IS_FINALIZING() Py_IsFinalizing()
#else
    // 旧版本Python中，假设不在终止阶段
    #define PCCTP_PY_IS_FINALIZING() (0)
#endif

// =============================================================================
// Capsule 辅助宏
// =============================================================================

/**
 * @brief 创建 Capsule 或返回 Py_None
 *
 * 用途：简化 Capsule 创建和引用计数处理
 *
 * Args:
 *   ptr: C 结构体指针
 *   name: Capsule 名称
 *
 * Returns:
 *   Capsule 对象或 Py_None（引用计数已正确处理）
 */
#define CREATE_CAPSULE_OR_NONE(ptr, name) \
    ((ptr) ? (Py_INCREF(PyCapsule_New(ptr, name, nullptr)), PyCapsule_New(ptr, name, nullptr)) \
           : (Py_INCREF(Py_None), Py_None))

/**
 * @brief 创建 Capsule 或返回 Py_None（优化版本，避免重复调用）
 *
 * Args:
 *   ptr: C 结构体指针
 *   name: Capsule 名称
 *
 * Returns:
 *   Capsule 对象或 Py_None（引用计数已正确处理）
 */
#define CREATE_CAPSULE_OR_NONE_OPT(ptr, name) \
    [&]() -> PyObject* { \
        if (ptr) { \
            PyObject* capsule = PyCapsule_New(ptr, name, nullptr); \
            if (capsule) Py_INCREF(capsule); \
            return capsule; \
        } \
        Py_INCREF(Py_None); \
        return Py_None; \
    }()

/**
 * @brief 清理 Capsule 对象
 */
#define CLEANUP_CAPSULE(obj) \
    do { \
        if ((obj) != Py_None) { \
            Py_XDECREF(obj); \
        } \
    } while(0)

// =============================================================================
// SPI 回调通用宏
// =============================================================================

/**
 * @brief 0 参数回调宏（使用 Python snake_case 命名）
 *
 * 用法：CALLBACK_0_PARAM(OnFrontConnected, "on_front_connected")
 */
#define CALLBACK_0_PARAM(method_name, py_name) \
    FORCE_INLINE_MEMBER void method_name() override { \
        if (!m_api || !m_api->py_spi) return; \
        PyGILStateKeeper gil; \
        PyObject* result = PyObject_CallMethod(m_api->py_spi, py_name, NULL); \
        if (!result) PyErr_Print(); \
        Py_XDECREF(result); \
    }

/**
 * @brief 1 参数回调宏（整数类型）（使用 Python snake_case 命名）
 *
 * 用法：CALLBACK_1_PARAM_INT(OnFrontDisconnected, nReason, "on_front_disconnected")
 */
#define CALLBACK_1_PARAM_INT(method_name, param_name, py_name) \
    FORCE_INLINE_MEMBER void method_name(int param_name) override { \
        if (!m_api || !m_api->py_spi) return; \
        PyGILStateKeeper gil; \
        PyObject* py_##param_name = PyLong_FromLong(param_name); \
        PyObject* result = PyObject_CallMethod(m_api->py_spi, py_name, "(O)", py_##param_name); \
        Py_XDECREF(py_##param_name); \
        if (!result) PyErr_Print(); \
        Py_XDECREF(result); \
    }

/**
 * @brief Capsule + RspInfo 回调宏（标准响应）- 直接传递 Capsule（零开销）
 *
 * 用法：CAPSULE_CALLBACK_RSP(OnRspUserLogin, CThostFtdcRspUserLoginField, "RspUserLogin", "on_rsp_user_login")
 */
#define CAPSULE_CALLBACK_RSP(method_name, struct_type, struct_name, py_name) \
    FORCE_INLINE_MEMBER void method_name(struct_type* pStruct, \
                                        CThostFtdcRspInfoField* pRspInfo, \
                                        int nRequestID, bool bIsLast) override { \
        if (!m_api || !m_api->py_spi) return; \
        PyGILStateKeeper gil; \
        PyObject* capsule_struct = CREATE_CAPSULE_OR_NONE_OPT(pStruct, struct_name); \
        PyObject* capsule_rsp_info = CREATE_CAPSULE_OR_NONE_OPT(pRspInfo, "RspInfo"); \
        /* 直接传递原始 Capsule，由 Python 端处理 */ \
        PyObject* result = PyObject_CallMethod(m_api->py_spi, py_name, "OOii", \
            capsule_struct, capsule_rsp_info, nRequestID, bIsLast); \
        Py_XDECREF(capsule_struct); \
        Py_XDECREF(capsule_rsp_info); \
        if (!result) PyErr_Print(); \
        Py_XDECREF(result); \
    }

/**
 * @brief 单 Capsule 回调宏（行情/成交回报）- 直接传递 Capsule（零开销）
 *
 * 用法：CAPSULE_CALLBACK_SINGLE(OnRtnDepthMarketData, CThostFtdcDepthMarketDataField, "DepthMarketData", "on_rtn_depth_market_data")
 */
#define CAPSULE_CALLBACK_SINGLE(method_name, struct_type, struct_name, py_name) \
    FORCE_INLINE_MEMBER void method_name(struct_type* pStruct) override { \
        if (!m_api || !m_api->py_spi) return; \
        PyGILStateKeeper gil; \
        if (!pStruct) { \
            PyObject* result = PyObject_CallMethod(m_api->py_spi, py_name, "O", Py_None); \
            if (!result) PyErr_Print(); \
            Py_XDECREF(result); \
            return; \
        } \
        PyObject* capsule = PyCapsule_New(pStruct, struct_name, nullptr); \
        if (!capsule) { \
            PyErr_Print(); \
            return; \
        } \
        /* 直接传递原始 Capsule */ \
        PyObject* result = PyObject_CallMethod(m_api->py_spi, py_name, "O", capsule); \
        Py_XDECREF(capsule); \
        if (!result) PyErr_Print(); \
        Py_XDECREF(result); \
    }

/**
 * @brief 错误回报回调宏 - 直接传递 Capsule（零开销）
 *
 * 用法：CAPSULE_CALLBACK_ERROR(OnErrRtnOrderInsert, CThostFtdcInputOrderField, "InputOrder", "on_err_rtn_order_insert")
 */
#define CAPSULE_CALLBACK_ERROR(method_name, struct_type, struct_name, py_name) \
    FORCE_INLINE_MEMBER void method_name(struct_type* pStruct, \
                                        CThostFtdcRspInfoField* pRspInfo) override { \
        if (!m_api || !m_api->py_spi) return; \
        PyGILStateKeeper gil; \
        PyObject* capsule_struct = CREATE_CAPSULE_OR_NONE_OPT(pStruct, struct_name); \
        PyObject* capsule_rsp_info = CREATE_CAPSULE_OR_NONE_OPT(pRspInfo, "RspInfo"); \
        /* 直接传递原始 Capsule */ \
        PyObject* result = PyObject_CallMethod(m_api->py_spi, py_name, "OO", \
            capsule_struct, capsule_rsp_info); \
        Py_XDECREF(capsule_struct); \
        Py_XDECREF(capsule_rsp_info); \
        if (!result) PyErr_Print(); \
        Py_XDECREF(result); \
    }

// =============================================================================
// 宏展开辅助工具（解决 ## 阻止宏展开的问题）
// =============================================================================

/**
 * @brief 强制宏展开的辅助宏
 *
 * 使用两级宏展开来确保在使用 ## 连接符前，宏参数已经被完全展开。
 *
 * 原理：
 * - C++ 预处理器在使用 ## 时不会展开宏参数
 * - 通过两级宏调用，第一级传入参数，第二级使用 CONCAT 展开参数
 *
 * 用法：
 *   DEF_API_INIT(MdApi, MdApiObject)
 *   展开为：
 *   DEF_API_INIT_IMPL(MdApi, MdApiObject)
 *   进一步展开为：
 *   static PyObject* MdApi_init(PyObject* self, PyObject* args) { ... }
 */
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)

// =============================================================================
// API 基础方法宏（使用两级展开）
// =============================================================================

/**
 * @brief 定义 get_api_version 静态方法（实现层）
 */
#define DEF_API_GET_VERSION_IMPL(api_class, ctp_api_class) \
    static PyObject* api_class##_get_api_version(PyObject* self, PyObject* args) { \
        const char* version = ctp_api_class::GetApiVersion(); \
        return PyUnicode_FromString(version); \
    }

/**
 * @brief 定义 get_api_version 静态方法（接口层）
 */
#define DEF_API_GET_VERSION(api_class, ctp_api_class) \
    DEF_API_GET_VERSION_IMPL(api_class, ctp_api_class)

/**
 * @brief 定义 init 方法（实现层）
 */
#define DEF_API_INIT_IMPL(api_class, api_obj_type) \
    static PyObject* api_class##_init(PyObject* self, PyObject* args) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (!obj->api) { \
            PyErr_SetString(PyExc_RuntimeError, #api_class " not initialized"); \
            return NULL; \
        } \
        obj->api->Init(); \
        Py_RETURN_NONE; \
    }

/**
 * @brief 定义 init 方法（接口层）
 */
#define DEF_API_INIT(api_class, api_obj_type) \
    DEF_API_INIT_IMPL(api_class, api_obj_type)

/**
 * @brief 定义 join 方法（实现层）
 */
#define DEF_API_JOIN_IMPL(api_class, api_obj_type) \
    static PyObject* api_class##_join(PyObject* self, PyObject* args) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (!obj->api) { \
            PyErr_SetString(PyExc_RuntimeError, #api_class " not initialized"); \
            return NULL; \
        } \
        int result = obj->api->Join(); \
        return PyLong_FromLong(result); \
    }

/**
 * @brief 定义 join 方法（接口层）
 */
#define DEF_API_JOIN(api_class, api_obj_type) \
    DEF_API_JOIN_IMPL(api_class, api_obj_type)

/**
 * @brief 定义 get_trading_day 方法（实现层）
 */
#define DEF_API_GET_TRADING_DAY_IMPL(api_class, api_obj_type) \
    static PyObject* api_class##_get_trading_day(PyObject* self, PyObject* args) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (!obj->api) { \
            PyErr_SetString(PyExc_RuntimeError, #api_class " not initialized"); \
            return NULL; \
        } \
        const char* trading_day = obj->api->GetTradingDay(); \
        return PyUnicode_FromString(trading_day); \
    }

/**
 * @brief 定义 get_trading_day 方法（接口层）
 */
#define DEF_API_GET_TRADING_DAY(api_class, api_obj_type) \
    DEF_API_GET_TRADING_DAY_IMPL(api_class, api_obj_type)

/**
 * @brief 定义 register_front 方法（实现层）
 *
 * ⚠️ **必须兼容三种输入类型**：
 * 1. str - 单个地址字符串，直接调用 C++ 接口注册
 * 2. list[str] - Python 列表，遍历每个元素调用 C++ 接口注册
 * 3. numpy.ndarray - NumPy 字符串数组，遍历每个元素调用 C++ 接口注册（支持零拷贝）
 */
#define DEF_API_REGISTER_FRONT_IMPL(api_class, api_obj_type) \
    static PyObject* api_class##_register_front(PyObject* self, PyObject* args) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (!obj->api) { \
            PyErr_SetString(PyExc_RuntimeError, #api_class " not initialized"); \
            return NULL; \
        } \
        PyObject* addresses_obj; \
        if (!PyArg_ParseTuple(args, "O", &addresses_obj)) { \
            return NULL; \
        } \
        /* 字符串路径 - 单个地址 */ \
        if (PyUnicode_Check(addresses_obj)) { \
            const char* front_address = PyUnicode_AsUTF8(addresses_obj); \
            if (!front_address) { \
                PyErr_SetString(PyExc_TypeError, "Failed to convert string to UTF-8"); \
                return NULL; \
            } \
            obj->api->RegisterFront((char*)front_address); \
            Py_RETURN_NONE; \
        } \
        /* NumPy 数组路径 */ \
        if (PyArray_Check(addresses_obj)) { \
            PyArrayObject* array = reinterpret_cast<PyArrayObject*>(addresses_obj); \
            PyArray_Descr* dtype = PyArray_DESCR(array); \
            if (dtype->type_num != NPY_UNICODE && dtype->type_num != NPY_STRING) { \
                PyErr_SetString(PyExc_TypeError, "NumPy array must be string dtype (U or S)"); \
                return NULL; \
            } \
            if (PyArray_NDIM(array) != 1) { \
                PyErr_SetString(PyExc_ValueError, "NumPy array must be 1-dimensional"); \
                return NULL; \
            } \
            char* data = static_cast<char*>(PyArray_DATA(array)); \
            npy_intp size = PyArray_SIZE(array); \
            npy_intp stride = PyArray_STRIDE(array, 0); \
            for (npy_intp i = 0; i < size; ++i) { \
                obj->api->RegisterFront(data + i * stride); \
            } \
            Py_RETURN_NONE; \
        } \
        /* List 路径 */ \
        if (PyList_Check(addresses_obj)) { \
            Py_ssize_t count = PyList_Size(addresses_obj); \
            for (Py_ssize_t i = 0; i < count; ++i) { \
                PyObject* item = PyList_GetItem(addresses_obj, i); \
                if (!PyUnicode_Check(item)) { \
                    PyErr_SetString(PyExc_TypeError, "All addresses in list must be strings"); \
                    return NULL; \
                } \
                const char* front_address = PyUnicode_AsUTF8(item); \
                if (!front_address) { \
                    PyErr_SetString(PyExc_TypeError, "Failed to convert string to UTF-8"); \
                    return NULL; \
                } \
                obj->api->RegisterFront((char*)front_address); \
            } \
            Py_RETURN_NONE; \
        } \
        PyErr_SetString(PyExc_TypeError, "addresses must be str, list[str] or numpy.ndarray"); \
        return NULL; \
    }

/**
 * @brief 定义 register_front 方法（接口层）
 */
#define DEF_API_REGISTER_FRONT(api_class, api_obj_type) \
    DEF_API_REGISTER_FRONT_IMPL(api_class, api_obj_type)

/**
 * @brief 定义 register_name_server 方法（实现层）
 *
 * ⚠️ **必须兼容三种输入类型**：
 * 1. str - 单个地址字符串，直接调用 C++ 接口注册
 * 2. list[str] - Python 列表，遍历每个元素调用 C++ 接口注册
 * 3. numpy.ndarray - NumPy 字符串数组，遍历每个元素调用 C++ 接口注册（支持零拷贝）
 */
#define DEF_API_REGISTER_NAME_SERVER_IMPL(api_class, api_obj_type) \
    static PyObject* api_class##_register_name_server(PyObject* self, PyObject* args) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (!obj->api) { \
            PyErr_SetString(PyExc_RuntimeError, #api_class " not initialized"); \
            return NULL; \
        } \
        PyObject* addresses_obj; \
        if (!PyArg_ParseTuple(args, "O", &addresses_obj)) { \
            return NULL; \
        } \
        /* 字符串路径 - 单个地址 */ \
        if (PyUnicode_Check(addresses_obj)) { \
            const char* ns_address = PyUnicode_AsUTF8(addresses_obj); \
            if (!ns_address) { \
                PyErr_SetString(PyExc_TypeError, "Failed to convert string to UTF-8"); \
                return NULL; \
            } \
            obj->api->RegisterNameServer((char*)ns_address); \
            Py_RETURN_NONE; \
        } \
        /* NumPy 数组路径 */ \
        if (PyArray_Check(addresses_obj)) { \
            PyArrayObject* array = reinterpret_cast<PyArrayObject*>(addresses_obj); \
            PyArray_Descr* dtype = PyArray_DESCR(array); \
            if (dtype->type_num != NPY_UNICODE && dtype->type_num != NPY_STRING) { \
                PyErr_SetString(PyExc_TypeError, "NumPy array must be string dtype (U or S)"); \
                return NULL; \
            } \
            if (PyArray_NDIM(array) != 1) { \
                PyErr_SetString(PyExc_ValueError, "NumPy array must be 1-dimensional"); \
                return NULL; \
            } \
            char* data = static_cast<char*>(PyArray_DATA(array)); \
            npy_intp size = PyArray_SIZE(array); \
            npy_intp stride = PyArray_STRIDE(array, 0); \
            for (npy_intp i = 0; i < size; ++i) { \
                obj->api->RegisterNameServer(data + i * stride); \
            } \
            Py_RETURN_NONE; \
        } \
        /* List 路径 */ \
        if (PyList_Check(addresses_obj)) { \
            Py_ssize_t count = PyList_Size(addresses_obj); \
            for (Py_ssize_t i = 0; i < count; ++i) { \
                PyObject* item = PyList_GetItem(addresses_obj, i); \
                if (!PyUnicode_Check(item)) { \
                    PyErr_SetString(PyExc_TypeError, "All addresses in list must be strings"); \
                    return NULL; \
                } \
                const char* ns_address = PyUnicode_AsUTF8(item); \
                if (!ns_address) { \
                    PyErr_SetString(PyExc_TypeError, "Failed to convert string to UTF-8"); \
                    return NULL; \
                } \
                obj->api->RegisterNameServer((char*)ns_address); \
            } \
            Py_RETURN_NONE; \
        } \
        PyErr_SetString(PyExc_TypeError, "addresses must be str, list[str] or numpy.ndarray"); \
        return NULL; \
    }

/**
 * @brief 定义 register_name_server 方法（接口层）
 */
#define DEF_API_REGISTER_NAME_SERVER(api_class, api_obj_type) \
    DEF_API_REGISTER_NAME_SERVER_IMPL(api_class, api_obj_type)

/**
 * @brief 定义 register_spi 方法（实现层）
 */
#define DEF_API_REGISTER_SPI_IMPL(api_class, api_obj_type, spi_class) \
    static PyObject* api_class##_register_spi(PyObject* self, PyObject* args) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (!obj->api) { \
            PyErr_SetString(PyExc_RuntimeError, #api_class " not initialized"); \
            return NULL; \
        } \
        PyObject* py_spi; \
        if (!PyArg_ParseTuple(args, "O", &py_spi)) { \
            return NULL; \
        } \
        Py_XDECREF(obj->py_spi); \
        obj->py_spi = py_spi; \
        Py_INCREF(obj->py_spi); \
        if (obj->spi) { \
            delete dynamic_cast<spi_class*>(obj->spi); \
        } \
        obj->spi = new spi_class(obj); \
        obj->api->RegisterSpi(obj->spi); \
        Py_RETURN_NONE; \
    }

/**
 * @brief 定义 register_spi 方法（接口层）
 */
#define DEF_API_REGISTER_SPI(api_class, api_obj_type, spi_class) \
    DEF_API_REGISTER_SPI_IMPL(api_class, api_obj_type, spi_class)

/**
 * @brief 定义 release 方法（实现层）
 */
#define DEF_API_RELEASE_IMPL(api_class, api_obj_type, spi_class) \
    static PyObject* api_class##_release(PyObject* self, PyObject* args) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (obj->api) { \
            obj->api->RegisterSpi(NULL); \
            obj->api->Release(); \
            obj->api = NULL; \
        } \
        if (obj->spi) { \
            delete dynamic_cast<spi_class*>(obj->spi); \
            obj->spi = NULL; \
        } \
        Py_XDECREF(obj->py_spi); \
        obj->py_spi = NULL; \
        Py_RETURN_NONE; \
    }

/**
 * @brief 定义 release 方法（接口层）
 */
#define DEF_API_RELEASE(api_class, api_obj_type, spi_class) \
    DEF_API_RELEASE_IMPL(api_class, api_obj_type, spi_class)

/**
 * @brief 定义 tp_dealloc 函数（实现层）
 */
#define DEF_API_DEALLOC_IMPL(api_obj_type, api_class, spi_class) \
    static void api_class##_dealloc(PyObject* self) { \
        api_obj_type* obj = (api_obj_type*)self; \
        if (!Py_IsInitialized()) { \
            if (obj->spi) { \
                delete dynamic_cast<spi_class*>(obj->spi); \
                obj->spi = NULL; \
            } \
            if (obj->api) { \
                obj->api->RegisterSpi(NULL); \
                obj->api->Release(); \
                obj->api = NULL; \
            } \
            Py_TYPE(self)->tp_free(self); \
            return; \
        } \
        /* 运行时检查是否在 Python 3.13+ 中且正在终止 */ \
        if (PCCTP_PY_IS_FINALIZING()) { \
            if (obj->spi) { \
                delete dynamic_cast<spi_class*>(obj->spi); \
                obj->spi = NULL; \
            } \
            if (obj->api) { \
                obj->api->RegisterSpi(NULL); \
                obj->api->Release(); \
                obj->api = NULL; \
            } \
            Py_TYPE(self)->tp_free(self); \
            return; \
        } \
        PyObject* result = api_class##_release(self, NULL); \
        Py_XDECREF(result); \
        Py_TYPE(self)->tp_free(self); \
    }

/**
 * @brief 定义 tp_dealloc 函数（接口层）
 */
#define DEF_API_DEALLOC(api_obj_type, api_class, spi_class) \
    DEF_API_DEALLOC_IMPL(api_obj_type, api_class, spi_class)

/**
 * @brief 定义 tp_new 和 tp_init 函数（实现层）
 */
#define DEF_API_NEW_INIT_IMPL(api_class, api_obj_type) \
    static PyObject* api_class##_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) { \
        api_obj_type* self = (api_obj_type*)type->tp_alloc(type, 0); \
        if (self) { \
            self->api = nullptr; \
            self->spi = nullptr; \
            self->py_spi = nullptr; \
        } \
        return (PyObject*)self; \
    } \
    static int api_class##_tp_init(PyObject* self, PyObject* args, PyObject* kwargs) { \
        PyErr_SetString(PyExc_TypeError, "Cannot create " #api_class " directly, use " #api_class ".create_ftdc_xxx_api() instead"); \
        return -1; \
    }

/**
 * @brief 定义 tp_new 和 tp_init 函数（接口层）
 */
#define DEF_API_NEW_INIT(api_class, api_obj_type) \
    DEF_API_NEW_INIT_IMPL(api_class, api_obj_type)

// =============================================================================
// 通用 PyCapsule 包装函数（可直接在模块方法表中使用）
// =============================================================================

/**
 * @brief PyCapsule_CheckExact 的 Python 可调用版本
 */
extern "C" inline PyObject* PyCapsule_CheckExact_Wrapper_Common(PyObject* self, PyObject* obj) {
    return PyLong_FromLong(PyCapsule_CheckExact(obj));
}

/**
 * @brief PyCapsule_GetPointer 的 Python 可调用版本
 */
extern "C" inline PyObject* PyCapsule_GetPointer_Wrapper_Common(PyObject* self, PyObject* args) {
    PyObject* capsule;
    const char* name;

    if (!PyArg_ParseTuple(args, "Os", &capsule, &name)) {
        return NULL;
    }

    void* ptr = PyCapsule_GetPointer(capsule, name);
    if (!ptr) {
        return NULL;
    }

    return PyLong_FromVoidPtr(ptr);
}

/**
 * @brief PyCapsule_New 的 Python 可调用版本
 */
extern "C" inline PyObject* PyCapsule_New_Wrapper_Common(PyObject* self, PyObject* args) {
    PyObject* ptr_obj;
    const char* name;
    PyObject* destructor_obj;

    if (!PyArg_ParseTuple(args, "OsO", &ptr_obj, &name, &destructor_obj)) {
        return NULL;
    }

    void* ptr = PyLong_AsVoidPtr(ptr_obj);
    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyCapsule_New(ptr, name, NULL);
}

/**
 * @brief 添加 PyCapsule 方法到模块方法表的宏
 */
#define ADD_PYCAPSULE_METHODS() \
    {"pycapsule_check_exact", PyCapsule_CheckExact_Wrapper_Common, METH_O, \
     "Check if object is a Capsule"}, \
    {"pycapsule_get_pointer", PyCapsule_GetPointer_Wrapper_Common, METH_VARARGS, \
     "Get pointer from Capsule"}, \
    {"pycapsule_new", PyCapsule_New_Wrapper_Common, METH_VARARGS, \
     "Create a new Capsule from pointer address"},

/**
 * @brief 通用请求方法宏（优先使用零拷贝 Capsule）
 *
 * 用法：DEF_REQ_METHOD_CAPSULE(MdApi, req_user_login, ReqUserLogin, CThostFtdcReqUserLoginField)
 *
 * 参数说明：
 * - api_class: API 类名 (如 MdApi, TradeApi)
 * - method_name: Python 方法名 (snake_case，如 req_user_login)
 * - ctp_method: CTP API 方法名 (驼峰命名，如 ReqUserLogin)
 * - ctp_req_type: CTP 请求结构体类型 (如 CThostFtdcReqUserLoginField)
 *
 * 零拷贝优化：
 * 1. 优先检查 _capsule 属性（零拷贝路径）
 * 2. 如果 _capsule 不存在或无效，尝试从 _struct 创建 capsule（兼容路径）
 */
#define DEF_REQ_METHOD_CAPSULE(api_class, method_name, ctp_method, ctp_req_type) \
    static PyObject* api_class##_##method_name(PyObject* self, PyObject* args, PyObject* kwargs) { \
        api_class##Object* obj = (api_class##Object*)self; \
        if (!obj->api) { \
            PyErr_SetString(PyExc_RuntimeError, #api_class " not initialized"); \
            return NULL; \
        } \
        PyObject* req_obj; \
        int request_id; \
        static char* kwlist[] = {#ctp_method, "request_id", NULL}; \
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &req_obj, &request_id)) { \
            return NULL; \
        } \
        ctp_req_type* req_ptr = nullptr; \
        /* 优先使用 _capsule（零拷贝路径） */ \
        PyObject* capsule = PyObject_GetAttrString(req_obj, "_capsule"); \
        if (capsule && capsule != Py_None && PyCapsule_CheckExact(capsule)) { \
            req_ptr = static_cast<ctp_req_type*>( \
                PyCapsule_GetPointer(capsule, #ctp_method)); \
            Py_DECREF(capsule); \
        } \
        /* 兼容路径：从 _struct 创建 capsule（有拷贝） */ \
        else { \
            Py_XDECREF(capsule); \
            /* 尝试调用 to_capsule() 方法 */ \
            PyObject* new_capsule = PyObject_CallMethod(req_obj, "to_capsule", NULL); \
            if (new_capsule && new_capsule != Py_None && PyCapsule_CheckExact(new_capsule)) { \
                req_ptr = static_cast<ctp_req_type*>( \
                    PyCapsule_GetPointer(new_capsule, #ctp_method)); \
                Py_DECREF(new_capsule); \
            } else { \
                Py_XDECREF(new_capsule); \
            } \
        } \
        if (!req_ptr) { \
            PyErr_SetString(PyExc_TypeError, \
                #ctp_method " must be a CapsuleStruct object with to_capsule() method"); \
            return NULL; \
        } \
        int result = obj->api->ctp_method(req_ptr, request_id); \
        return PyLong_FromLong(result); \
    }

