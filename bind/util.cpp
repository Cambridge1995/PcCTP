/**
 * PcCTP - 工具函数实现文件
 *
 * 本文件包含：
 * - 全局字符串池实例的定义
 * - NumPy C API 的符号导出（唯一定义的位置）
 * - 需要完整 NumPy 头的函数实现
 * - Python 模块方法的实现（防止 C4505 警告）
 */

// 先包含 Python 和 NumPy
#define PCCTP_UTIL_CPP_DEFINE  // 标记这是 util.cpp，用于导出 NumPy C API 符号
#include <Python.h>

// NumPy C API - 在此文件中定义符号（必须在 util.h 之前）
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PcCTP_CAPI_ARRAY_API
#include <numpy/ndarrayobject.h>

// 再包含 util.h（此时 PyArrayObject 已定义）
#include "util.h"

// =============================================================================
// Python 版本兼容性宏
// =============================================================================
// Py_IsFinalizing() 在 Python 3.13+ 才可用
#if PY_VERSION_HEX >= 0x030D0000
    #define PCCTP_PY_IS_FINALIZING() Py_IsFinalizing()
#else
    // 旧版本Python中，假设不在终止阶段（因为通常在模块清理时调用）
    #define PCCTP_PY_IS_FINALIZING() (0)
#endif

// =============================================================================
// 全局字符串池实例定义
// =============================================================================

namespace GlobalStringPools {
    StringPool ExchangeCodes;
    StringPool Dates;
    StringPool Times;
    StringPool Instruments;
    StringPool UserIds;
    StringPool BrokerIds;
}

// =============================================================================
// StringPool 方法实现
// =============================================================================

PyObject* StringPool::intern(const char* str, size_t max_len) {
    size_t len = strnlen(str, max_len);
    std::string_view sv(str, len);

    std::lock_guard<std::mutex> lock(mutex_);

    std::string key(sv);
    auto it = pool_.find(key);
    if (it != pool_.end()) {
        if (Py_IsInitialized() && !PCCTP_PY_IS_FINALIZING()) {
            Py_INCREF(it->second);
        }
        return it->second;
    }

    PyObject* result = nullptr;
    if (Py_IsInitialized() && !PCCTP_PY_IS_FINALIZING()) {
        result = PyUnicode_DecodeASCII(str, len, nullptr);
        if (result) {
            Py_INCREF(result);
        }
    }
    if (result) {
        pool_.emplace(std::move(key), result);
    }
    return result;
}

void StringPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (Py_IsInitialized() && !PCCTP_PY_IS_FINALIZING()) {
        for (auto& pair : pool_) {
            Py_XDECREF(pair.second);
        }
    }
    pool_.clear();
}

size_t StringPool::size() const {
    return pool_.size();
}

void StringPool::reserve(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.reserve(count);
}

// =============================================================================
// NumPy 工具函数实现
// =============================================================================

bool is_numpy_array(PyObject* obj) {
    return PyArray_Check(obj) != 0;
}

bool is_string_array(void* array_ptr) {
    PyArrayObject* array = static_cast<PyArrayObject*>(array_ptr);
    PyArray_Descr* dtype = PyArray_DESCR(array);
    return dtype->type_num == NPY_UNICODE || dtype->type_num == NPY_STRING;
}

bool is_1d_array(void* array_ptr) {
    PyArrayObject* array = static_cast<PyArrayObject*>(array_ptr);
    return PyArray_NDIM(array) == 1;
}

bool is_c_contiguous(void* array_ptr) {
    PyArrayObject* array = static_cast<PyArrayObject*>(array_ptr);
    return PyArray_IS_C_CONTIGUOUS(array);
}

int extract_instrument_ids(
    PyObject* instrument_ids,
    std::vector<char*>& out_ids,
    std::vector<std::string>& out_strs
) {
    // 情况1：Python list
    if (PyList_Check(instrument_ids)) {
        Py_ssize_t size = PyList_Size(instrument_ids);
        out_ids.reserve(size);
        out_strs.reserve(size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(instrument_ids, i);
            if (!PyUnicode_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Instrument IDs must be strings");
                return -1;
            }

            const char* str = PyUnicode_AsUTF8(item);
            if (!str) return -1;

            out_strs.emplace_back(str);
            out_ids.push_back(const_cast<char*>(out_strs.back().c_str()));
        }
        return static_cast<int>(size);
    }

    // 情况2：NumPy 数组
    if (is_numpy_array(instrument_ids)) {
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(instrument_ids);

        if (!is_1d_array(array)) {
            PyErr_SetString(PyExc_ValueError, "Instrument IDs must be a 1D array");
            return -1;
        }

        if (!is_string_array(array)) {
            PyErr_SetString(PyExc_TypeError, "Instrument IDs array must be string type");
            return -1;
        }

        Py_ssize_t size = PyArray_SIZE(array);
        out_ids.reserve(size);
        out_strs.reserve(size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyArray_GETITEM(array, static_cast<const char*>(PyArray_GETPTR1(array, i)));
            if (!item) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to get array item");
                return -1;
            }

            const char* str = nullptr;
            // 处理 Unicode 字符串（str 对象）
            if (PyUnicode_Check(item)) {
                str = PyUnicode_AsUTF8(item);
                Py_DECREF(item);
                if (!str) return -1;
            }
            // 处理字节字符串（bytes 对象）
            else if (PyBytes_Check(item)) {
                char* bytes_str = nullptr;
                Py_ssize_t len = 0;
                if (PyBytes_AsStringAndSize(item, &bytes_str, &len) == 0) {
                    // 找到第一个 null 终止符或使用长度
                    size_t actual_len = strnlen(bytes_str, len);
                    out_strs.emplace_back(bytes_str, actual_len);
                    Py_DECREF(item);
                    out_ids.push_back(const_cast<char*>(out_strs.back().c_str()));
                    continue;
                }
                Py_DECREF(item);
                return -1;
            }
            else {
                Py_DECREF(item);
                PyErr_SetString(PyExc_TypeError, "Array elements must be strings or bytes");
                return -1;
            }

            out_strs.emplace_back(str);
            out_ids.push_back(const_cast<char*>(out_strs.back().c_str()));
        }
        return static_cast<int>(size);
    }

    PyErr_SetString(PyExc_TypeError, "Instrument IDs must be a list or NumPy array");
    return -1;
}

// =============================================================================
// Python 模块方法实现（防止 C4505 警告）
// =============================================================================

PyObject* pcctp_clear_dates(PyObject* self, PyObject* args) {
    GlobalStringPools::Dates.clear();
    Py_RETURN_NONE;
}

PyObject* pcctp_clear_times(PyObject* self, PyObject* args) {
    GlobalStringPools::Times.clear();
    Py_RETURN_NONE;
}

PyObject* pcctp_clear_instruments(PyObject* self, PyObject* args) {
    GlobalStringPools::Instruments.clear();
    Py_RETURN_NONE;
}

PyObject* pcctp_pool_stats(PyObject* self, PyObject* args) {
    PyObject* result = PyDict_New();
    if (!result) return NULL;

    PyDict_SetItemString(result, "exchange_codes", PyLong_FromSize_t(GlobalStringPools::ExchangeCodes.size()));
    PyDict_SetItemString(result, "dates", PyLong_FromSize_t(GlobalStringPools::Dates.size()));
    PyDict_SetItemString(result, "times", PyLong_FromSize_t(GlobalStringPools::Times.size()));
    PyDict_SetItemString(result, "instruments", PyLong_FromSize_t(GlobalStringPools::Instruments.size()));
    PyDict_SetItemString(result, "user_ids", PyLong_FromSize_t(GlobalStringPools::UserIds.size()));
    PyDict_SetItemString(result, "broker_ids", PyLong_FromSize_t(GlobalStringPools::BrokerIds.size()));

    return result;
}

// =============================================================================
// 新增 Python 模块方法（使用 Python 期望的名称）
// =============================================================================

/**
 * cleanup_temporal_pools()
 *
 * 清除临时字符串池（日期和时间）
 */
PyObject* cleanup_temporal_pools(PyObject* self, PyObject* args) {
    GlobalStringPools::Dates.clear();
    GlobalStringPools::Times.clear();
    Py_RETURN_NONE;
}

/**
 * cleanup_instruments()
 *
 * 清除合约字符串池
 */
PyObject* cleanup_instruments(PyObject* self, PyObject* args) {
    GlobalStringPools::Instruments.clear();
    Py_RETURN_NONE;
}

/**
 * check_instrument_pool_size()
 *
 * 返回合约字符串池大小
 */
PyObject* check_instrument_pool_size(PyObject* self, PyObject* args) {
    return PyLong_FromSize_t(GlobalStringPools::Instruments.size());
}

/**
 * get_pool_sizes()
 *
 * 返回所有字符串池的大小（字典形式）
 */
PyObject* get_pool_sizes(PyObject* self, PyObject* args) {
    PyObject* result = PyDict_New();
    if (!result) return NULL;

    PyDict_SetItemString(result, "exchange_codes", PyLong_FromSize_t(GlobalStringPools::ExchangeCodes.size()));
    PyDict_SetItemString(result, "dates", PyLong_FromSize_t(GlobalStringPools::Dates.size()));
    PyDict_SetItemString(result, "times", PyLong_FromSize_t(GlobalStringPools::Times.size()));
    PyDict_SetItemString(result, "instruments", PyLong_FromSize_t(GlobalStringPools::Instruments.size()));
    PyDict_SetItemString(result, "user_ids", PyLong_FromSize_t(GlobalStringPools::UserIds.size()));
    PyDict_SetItemString(result, "broker_ids", PyLong_FromSize_t(GlobalStringPools::BrokerIds.size()));

    return result;
}

// =============================================================================
// 初始化函数实现
// =============================================================================

int initialize_util() {
    // 导入 NumPy C API
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "Failed to import numpy C API");
        return -1;
    }

    // 初始化字符串池
    GlobalStringPools::initialize();

    return 0;
}
