/**
 * PcCTP - 工具函数头文件
 *
 * 声明工具函数和类，实现在 util.cpp 中
 */

#pragma once

#include <Python.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

// NumPy C API 符号名称（供各 .cpp 文件使用）
// 注意：在 util.cpp 中定义 PY_ARRAY_UNIQUE_SYMBOL
// 在其他文件中使用 NO_IMPORT_ARRAY 导入符号
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PcCTP_CAPI_ARRAY_API

// 如果不是在 util.cpp 中定义，则使用 NO_IMPORT_ARRAY
#ifndef PCCTP_UTIL_CPP_DEFINE
    #define NO_IMPORT_ARRAY
#endif

// =============================================================================
// 1. 平台宏和常量
// =============================================================================

#if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
    #define FORCE_INLINE_MEMBER __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
    #define FORCE_INLINE_MEMBER __attribute__((always_inline)) inline
#else
    #define FORCE_INLINE inline
    #define FORCE_INLINE_MEMBER inline
#endif

constexpr size_t MAX_EXCHANGE_ID_LEN = 9;
constexpr size_t MAX_DATE_LEN = 9;
constexpr size_t MAX_TIME_LEN = 9;
constexpr size_t MAX_INSTRUMENT_ID_LEN = 31;
constexpr size_t MAX_INVESTOR_ID_LEN = 15;
constexpr size_t MAX_BROKER_ID_LEN = 11;
constexpr size_t MAX_USER_ID_LEN = 16;
constexpr size_t MAX_PASSWORD_LEN = 41;
constexpr size_t MAX_SESSION_ID_LEN = 13;
constexpr size_t MAX_ORDER_REF_LEN = 13;
constexpr size_t MAX_ERROR_MSG_LEN = 81;

// =============================================================================
// 2. GIL 管理
// =============================================================================

class PyGILStateKeeper {
public:
    PyGILStateKeeper() : state(PyGILState_Ensure()) {}
    ~PyGILStateKeeper() { PyGILState_Release(state); }

    PyGILStateKeeper(const PyGILStateKeeper&) = delete;
    PyGILStateKeeper& operator=(const PyGILStateKeeper&) = delete;

private:
    PyGILState_STATE state;
};

// =============================================================================
// 3. 字符串池类
// =============================================================================

class StringPool {
private:
    struct StringHash {
        size_t operator()(const std::string_view& sv) const {
            return std::hash<std::string_view>{}(sv);
        }
    };

    std::unordered_map<std::string, PyObject*, StringHash> pool_;
    std::mutex mutex_;

public:
    ~StringPool() { clear(); }

    PyObject* intern(const char* str, size_t max_len);
    void clear();
    size_t size() const;
    void reserve(size_t count);
};

// =============================================================================
// 4. 全局字符串池（声明，定义在 util.cpp）
// =============================================================================

namespace GlobalStringPools {
    extern StringPool ExchangeCodes;
    extern StringPool Dates;
    extern StringPool Times;
    extern StringPool Instruments;
    extern StringPool UserIds;
    extern StringPool BrokerIds;

    inline void initialize() {
        ExchangeCodes.reserve(20);
        Dates.reserve(15);
        Times.reserve(36000);
        Instruments.reserve(1000);
        UserIds.reserve(40);
        BrokerIds.reserve(25);
    }

    inline void cleanup_temporal_pools() {
        Dates.clear();
        Times.clear();
    }

    inline void cleanup_instruments() {
        Instruments.clear();
    }
}

// =============================================================================
// 5. NumPy 工具函数（声明）
// =============================================================================

// NumPy 辅助函数声明
// 注意：这些函数需要在包含 <numpy/ndarrayobject.h> 后才能使用
// 为避免与 NumPy 类型定义冲突，这里使用 void* 作为参数类型
bool is_numpy_array(PyObject* obj);
bool is_string_array(void* array);
bool is_1d_array(void* array);
bool is_c_contiguous(void* array);

int extract_instrument_ids(
    PyObject* instrument_ids,
    std::vector<char*>& out_ids,
    std::vector<std::string>& out_strs
);

// =============================================================================
// 6. Capsule 辅助函数
// =============================================================================

inline void* capsule_to_ptr(PyObject* capsule, const char* name) {
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_Format(PyExc_TypeError, "Expected Capsule, got %s", Py_TYPE(capsule)->tp_name);
        return nullptr;
    }

    void* ptr = PyCapsule_GetPointer(capsule, name);
    if (!ptr) {
        PyErr_Format(PyExc_ValueError, "Invalid Capsule type: expected '%s'", name);
        return nullptr;
    }

    return ptr;
}

inline PyObject* ptr_to_capsule(void* ptr, const char* name) {
    if (!ptr) {
        PyErr_Format(PyExc_ValueError, "Cannot create Capsule from null pointer");
        return nullptr;
    }

    PyObject* capsule = PyCapsule_New(ptr, name, nullptr);
    if (!capsule) {
        PyErr_Format(PyExc_RuntimeError, "Failed to create Capsule for '%s'", name);
        return nullptr;
    }

    return capsule;
}

template<typename T>
inline void capsule_destructor(PyObject* capsule) {
    T* ptr = static_cast<T*>(PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule)));
    if (ptr) {
        delete ptr;
    }
}

// =============================================================================
// 7. 初始化函数（声明）
// =============================================================================

int initialize_util();

// =============================================================================
// 8. 宏定义
// =============================================================================

#define DEFINE_STRING_POOLS() \
    namespace GlobalStringPools { \
        extern StringPool ExchangeCodes; \
        extern StringPool Dates; \
        extern StringPool Times; \
        extern StringPool Instruments; \
        extern StringPool UserIds; \
        extern StringPool BrokerIds; \
    }

// Python 模块方法声明（实现在 util.cpp 中）
PyObject* pcctp_clear_dates(PyObject* self, PyObject* args);
PyObject* pcctp_clear_times(PyObject* self, PyObject* args);
PyObject* pcctp_clear_instruments(PyObject* self, PyObject* args);
PyObject* pcctp_pool_stats(PyObject* self, PyObject* args);

// 新增：Python 期望的函数声明
PyObject* cleanup_temporal_pools(PyObject* self, PyObject* args);
PyObject* cleanup_instruments(PyObject* self, PyObject* args);
PyObject* check_instrument_pool_size(PyObject* self, PyObject* args);
PyObject* get_pool_sizes(PyObject* self, PyObject* args);

#define PCCTP_POOL_METHODS() \
    {"clear_dates", pcctp_clear_dates, METH_NOARGS, "Clear date string pool"}, \
    {"clear_times", pcctp_clear_times, METH_NOARGS, "Clear time string pool"}, \
    {"clear_instruments", pcctp_clear_instruments, METH_NOARGS, "Clear instrument string pool"}, \
    {"pool_stats", pcctp_pool_stats, METH_NOARGS, "Get string pool statistics"}

// 新增：使用 Python 期望命名的模块方法宏
#define PCCTP_POOL_CLEANUP_METHODS() \
    {"cleanup_temporal_pools", cleanup_temporal_pools, METH_NOARGS, "Clear date and time string pools"}, \
    {"cleanup_instruments", cleanup_instruments, METH_NOARGS, "Clear instrument string pool"}, \
    {"check_instrument_pool_size", check_instrument_pool_size, METH_NOARGS, "Get instrument pool size"}, \
    {"get_pool_sizes", get_pool_sizes, METH_NOARGS, "Get all string pool sizes"}
