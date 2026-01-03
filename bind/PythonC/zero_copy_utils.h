/**
 * 零拷贝优化工具函数（纯 Python C API 版本）
 *
 * 提供以下优化：
 * 1. 优化的字符串构造（非零拷贝，但开销最小）
 * 2. 字符串池（零拷贝复用，适用于有限集合）
 * 3. numpy 类型安全检查
 * 4. numpy 零拷贝访问（直接访问底层数据指针）
 *
 * 不依赖 nanobind，完全使用 Python C API 实现
 */

#pragma once

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <cstring>
#include <unordered_map>
#include <string_view>
#include <mutex>
#include <vector>

// Fallback for Py_IsFinalizing (Python 3.13+)
// Python 3.13 之前的版本无法可靠检测解释器关闭状态
#if PY_VERSION_HEX >= 0x030D0000
    #ifndef Py_IsFinalizing
        // 使用内部API，如果不存在则定义fallback
        #ifdef _Py_IsFinalizing
            #define Py_IsFinalizing() _Py_IsFinalizing()
        #else
            // Fallback: 假设不在finalization阶段
            static inline int Py_IsFinalizing_Fallback() { return 0; }
            #define Py_IsFinalizing() Py_IsFinalizing_Fallback()
        #endif
    #endif
#endif

// =============================================================================
// 优化的字符串构造
// =============================================================================

/**
 * @brief 安全的字符串长度计算（强制内联）
 *
 * @param str C 字符串指针
 * @param max_len 最大长度
 * @return size_t 实际字符串长度
 *
 * @note 使用 strnlen 确保不会读取超出 max_len 的内存
 */
inline size_t safe_strlen(const char* str, size_t max_len) {
    return strnlen(str, max_len);
}

/**
 * @brief 从 C 字符串创建 Python str（强制内联优化版本）
 *
 * @param cstr C 字符串指针
 * @param max_len 最大长度（包括空字符）
 * @return PyObject* Python 字符串对象（新引用）
 *
 * @note 优化点：
 * 1. 使用 safe_strlen (strnlen) 计算实际长度（去除尾部空字符）
 * 2. 直接调用 PyUnicode_FromStringAndSize
 * 3. 强制内联确保高频路径零函数调用开销
 * 4. 仍需拷贝，但开销最小（Python str 需要 UTF-8 编码）
 */
inline PyObject* create_py_string(const char* cstr, size_t max_len) {
    size_t len = safe_strlen(cstr, max_len);
    return PyUnicode_FromStringAndSize(cstr, len);
}

/**
 * @brief 从 C 字符串创建 Python str（已知长度，内联优化）
 */
inline PyObject* create_py_string_with_length(const char* cstr, size_t len) {
    return PyUnicode_FromStringAndSize(cstr, len);
}

// =============================================================================
// RAII 风格的 GIL 管理（仿照 nanobind 的 gil_scoped_acquire）
// =============================================================================

/**
 * @brief RAII 风格的 GIL 获取管理类
 *
 * 构造时自动获取 GIL，析构时自动释放 GIL，确保异常安全。
 * 适用于需要在 C++ 代码中调用 Python C API 的场景。
 *
 * 特性：
 * - 构造时调用 PyGILState_Ensure() 获取 GIL
 * - 析构时调用 PyGILState_Release() 释放 GIL
 * - 禁止拷贝（防止重复释放）
 * - 支持移动语义（允许转移所有权）
 * - 异常安全：任何退出路径都会自动释放 GIL
 *
 * 使用示例：
 * @code
 *   virtual void OnFrontConnected() override {
 *       if (!m_api || !m_api->py_spi) return;
 *       PyGILStateKeeper gil;  // 构造时获取 GIL
 *       PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_front_connected", NULL);
 *       if (!result) PyErr_Print();
 *       Py_XDECREF(result);
 *   }  // 析构时自动释放 GIL，任何退出路径都安全
 * @endcode
 *
 * 对比手动管理：
 * @code
 *   // 手动管理（容易出错）
 *   PyGILState_STATE gstate = PyGILState_Ensure();
 *   // ... 操作 ...
 *   if (error) {
 *       PyGILState_Release(gstate);  // 需要记得释放
 *       return;
 *   }
 *   PyGILState_Release(gstate);  // 容易忘记或放错位置
 *
 *   // RAII 管理（安全简洁）
 *   PyGILStateKeeper gil;  // 自动管理
 *   // ... 操作 ...
 *   if (error) return;  // 自动释放
 * @endcode
 *
 * @note 性能与手动管理完全相同（编译器内联优化）
 * @note 线程安全：PyGILState_Ensure/Release 本身是线程安全的
 */
class PyGILStateKeeper {
private:
    PyGILState_STATE state_;
    bool owns_;  // 独立标志跟踪是否拥有 GIL

public:
    /**
     * @brief 构造函数：获取 GIL
     *
     * 调用 PyGILState_Ensure() 获取全局解释器锁。
     * 如果当前线程已持有 GIL，此操作是无操作（不会死锁）。
     */
    PyGILStateKeeper() : state_(PyGILState_Ensure()), owns_(true) {}

    /**
     * @brief 拷贝构造函数（已删除）
     *
     * 禁止拷贝以防止重复释放 GIL。
     */
    PyGILStateKeeper(const PyGILStateKeeper&) = delete;

    /**
     * @brief 拷贝赋值运算符（已删除）
     *
     * 禁止拷贝赋值以防止重复释放 GIL。
     */
    PyGILStateKeeper& operator=(const PyGILStateKeeper&) = delete;

    /**
     * @brief 移动构造函数
     *
     * 允许转移 GIL 管理权。
     * 移动后，源对象的 owns_ 被置为 false，析构时不会释放 GIL。
     */
    PyGILStateKeeper(PyGILStateKeeper&& other) noexcept
        : state_(other.state_), owns_(other.owns_) {
        other.owns_ = false;  // 清空源对象的标志，防止重复释放
    }

    /**
     * @brief 移动赋值运算符
     *
     * 允许转移 GIL 管理权。
     * 移动前会先释放当前对象持有的 GIL（如果有）。
     */
    PyGILStateKeeper& operator=(PyGILStateKeeper&& other) noexcept {
        if (this != &other) {
            // 释放当前对象持有的 GIL
            if (owns_) {
                PyGILState_Release(state_);
            }
            // 转移所有权
            state_ = other.state_;
            owns_ = other.owns_;
            other.owns_ = false;
        }
        return *this;
    }

    /**
     * @brief 析构函数：释放 GIL
     *
     * 调用 PyGILState_Release() 释放全局解释器锁。
     * 如果 owns_ 为 false，则不执行任何操作。
     *
     * @note 析构函数是 noexcept 的，不会抛出异常
     */
    ~PyGILStateKeeper() noexcept {
        if (owns_) {  // 检查状态是否有效
            PyGILState_Release(state_);
        }
    }

    /**
     * @brief 获取底层的 GIL 状态（高级用法）
     *
     * @return PyGILState_STATE 底层 GIL 状态对象
     *
     * @warning 谨慎使用，修改状态可能导致未定义行为
     */
    PyGILState_STATE get() const noexcept {
        return state_;
    }

    /**
     * @brief 检查是否持有有效的 GIL 状态
     *
     * @return true 如果持有有效状态，false 如果已被移动或置零
     */
    bool valid() const noexcept {
        return owns_;
    }
};

// =============================================================================
// 字符串池（零拷贝复用）- 纯 Python C API 版本
// =============================================================================

/**
 * @brief 字符串池（线程安全，纯 Python C API 版本）
 *
 * 用于驻留重复出现的字符串，实现零拷贝复用。
 * 适用场景：
 * - 交易所代码（SHFE, DCE, CZCE 等，有限集合）
 * - 日期字符串（每个交易日固定，有限集合）
 * - 时间字符串（HH:MM:SS 格式，有限集合）
 * - 合约代码（订阅的合约有限，需要定期清理）
 * - 用户代码、经纪公司代码（有限集合）
 */
class StringPool {
private:
    struct StringHash {
        size_t operator()(const std::string_view& sv) const {
            return std::hash<std::string_view>{}(sv);
        }
    };

    // 存储 PyObject*（已增加引用计数）
    std::unordered_map<std::string, PyObject*, StringHash> pool_;
    std::mutex mutex_;

public:
    /**
     * @brief 析构函数：释放所有缓存的 Python 对象
     *
     * @note 检查 Python 解释器状态，仅在解释器仍在运行时释放 Python 对象
     */
    ~StringPool() {
        clear();
    }

    /**
     * @brief 驻留字符串（返回 Python str，新引用）
     *
     * @param str C 字符串指针
     * @param max_len 最大长度
     * @return PyObject* Python 字符串对象（新引用，如果已存在则增加引用计数并返回）
     */
    PyObject* intern(const char* str, size_t max_len) {
        size_t len = strnlen(str, max_len);
        std::string_view sv(str, len);

        std::lock_guard<std::mutex> lock(mutex_);

        // 将 string_view 转换为 string 用于查找
        std::string key(sv);
        auto it = pool_.find(key);
        if (it != pool_.end()) {
            // 返回已缓存的 Python str（零拷贝复用，增加引用计数）
            // 检查 Python 解释器状态
            if (Py_IsInitialized()) {
                Py_INCREF(it->second);
            }
            return it->second;
        }

        // 创建并缓存（增加引用计数以保持引用）
        PyObject* result = nullptr;
        if (Py_IsInitialized()) {
            result = create_py_string_with_length(str, len);
            if (result) {
                Py_INCREF(result);  // 增加引用计数，pool 持有一份
            }
        }
        if (result) {
            pool_.emplace(std::move(key), result);
        }
        return result;
    }

    /**
     * @brief 清空字符串池（释放所有 Python 对象）
     *
     * @note 检查 Python 解释器状态，仅在解释器仍在运行时调用 Py_XDECREF
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        // 仅在 Python 解释器仍在运行时释放 Python 对象
        // 程序退出时，解释器关闭后会自动回收所有对象
        if (Py_IsInitialized()) {
            #if PY_VERSION_HEX >= 0x030D0000
            // Python 3.13+: 精确判断是否在关闭阶段
            if (!Py_IsFinalizing()) {
                for (auto& pair : pool_) {
                    Py_XDECREF(pair.second);
                }
            }
            // 否则跳过释放，避免 0xC0000005 崩溃
            #else
            // Python 3.8-3.12: 无法精确判断，采用保守策略
            // 跳过释放以避免潜在的 0xC0000005 崩溃
            // 操作系统会在进程退出时回收内存
            // 不做任何操作，pool_.clear() 会清空容器
            #endif
        }
        pool_.clear();
    }

    /**
     * @brief 获取字符串池大小
     */
    size_t size() const {
        return pool_.size();
    }

    /**
     * @brief 预留空间（避免 rehash）
     */
    void reserve(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.reserve(count);
    }
};

// =============================================================================
// 全局字符串池（按用途分类）
// =============================================================================

namespace GlobalStringPools {
    // 交易所代码池（固定约 10 个值）
    extern StringPool ExchangeCodes;

    // 日期池（每个交易日约 1 个新值，可定期清理）
    extern StringPool Dates;

    // 时间字符串池（HH:MM:SS 格式，有限集合）
    extern StringPool Times;

    // 合约代码池（订阅的合约有限，使用字符串池）
    extern StringPool Instruments;

    // 用户代码池（登录用户有限）
    extern StringPool UserIds;

    // 经纪公司代码池（经纪公司有限）
    extern StringPool BrokerIds;

    /**
     * @brief 初始化所有字符串池
     */
    inline void initialize() {
        // 预分配空间（避免 rehash）
        ExchangeCodes.reserve(20);     // 约 10 个交易所
        Dates.reserve(15);             // 两周交易日（考虑跨周末）
        Times.reserve(36000);          // 完整交易时段
        Instruments.reserve(1000);     // 订阅合约数量
        UserIds.reserve(40);           // 用户数量（30交易员 + 技术账号）
        BrokerIds.reserve(25);         // 经纪公司数量
    }

    /**
     * @brief 清理日期和时间池（定期调用以释放内存）
     */
    inline void cleanup_temporal_pools() {
        Dates.clear();
        Times.clear();
    }

    /**
     * @brief 清理合约代码池（切换交易日或重新订阅时调用）
     */
    inline void cleanup_instruments() {
        Instruments.clear();
    }

    /**
     * @brief 检查合约池大小并警告（超过 950 时警告）
     * @return 当前合约池大小
     */
    inline size_t check_instrument_pool_size() {
        size_t size = Instruments.size();
        if (size >= 950) {
            PyErr_WarnFormat(
                PyExc_RuntimeWarning,
                1,
                "合约池接近上限: %zu / 1000 (%.1f%%)，建议调用 cleanup_instruments() 清理或增加预留容量",
                size, size * 100.0 / 1000
            );
        }
        return size;
    }

    /**
     * @brief 获取各字符串池大小（用于监控）
     */
    inline void get_pool_sizes(size_t& exchanges, size_t& dates, size_t& times,
                              size_t& instruments, size_t& users, size_t& brokers) {
        exchanges = ExchangeCodes.size();
        dates = Dates.size();
        times = Times.size();
        instruments = Instruments.size();
        users = UserIds.size();
        brokers = BrokerIds.size();
    }

    /**
     * @brief 合约代码 intern 并自动检查池大小（超过 950 时警告）
     * @param str C 字符串指针
     * @param max_len 最大长度
     * @return PyObject* Python 字符串对象（新引用）
     */
    inline PyObject* intern_instrument(const char* str, size_t max_len) {
        PyObject* result = Instruments.intern(str, max_len);
        // 每次添加后检查（性能影响极小：仅当 size >= 950 时才执行警告逻辑）
        size_t size = Instruments.size();
        if (size >= 950 && size <= 1000) {  // 避免重复警告
            PyErr_WarnFormat(
                PyExc_RuntimeWarning,
                1,
                "合约池接近上限: %zu / 1000 (%.1f%%)，建议调用 cleanup_instruments() 清理或增加预留容量",
                size, size * 100.0 / 1000
            );
        }
        return result;
    }
}

// =============================================================================
// numpy 类型检查工具（纯 Python C API 版本）
// =============================================================================

/**
 * @brief 检查对象是否为 numpy 数组
 */
inline bool is_numpy_array(PyObject* obj) {
    return PyArray_Check(obj) != 0;
}

/**
 * @brief 检查 numpy 数组是否为字符串类型
 */
inline bool is_string_array(PyArrayObject* array) {
    PyArray_Descr* dtype = PyArray_DESCR(array);
    return dtype->type_num == NPY_UNICODE || dtype->type_num == NPY_STRING;
}

/**
 * @brief 检查 numpy 数组是否为一维
 */
inline bool is_1d_array(PyArrayObject* array) {
    return PyArray_NDIM(array) == 1;
}

/**
 * @brief 检查 numpy 数组是否为 C 连续
 */
inline bool is_c_contiguous(PyArrayObject* array) {
    return PyArray_IS_C_CONTIGUOUS(array);
}

// =============================================================================
// 辅助宏（简化代码编写）- Python C API 版本
// =============================================================================

/**
 * @brief 使用字符串池转换字段并添加到 dict
 *
 * @param dict 目标 PyObject* dict
 * @param c_field C 结构体字段名
 * @param pField C 结构体指针
 * @param pool 使用的字符串池
 *
 * 示例：
 *   FIELD_STRING_POOL_PY(data, ExchangeID, pDepthMarketData, GlobalStringPools::ExchangeCodes);
 */
#define FIELD_STRING_POOL_PY(dict, c_field, pField, pool) \
    do { \
        PyObject* _str_val = (pool).intern((pField)->c_field, sizeof((pField)->c_field)); \
        if (_str_val) { \
            PyDict_SetItemString(dict, #c_field, _str_val); \
            Py_DECREF(_str_val);  /* SetItemString 增加引用，我们需要减少 */ \
        } \
    } while(0)

/**
 * @brief 使用优化构造转换字段并添加到 dict
 *
 * @param dict 目标 PyObject* dict
 * @param c_field C 结构体字段名
 * @param pField C 结构体指针
 *
 * 示例：
 *   FIELD_STRING_OPTIMIZED_PY(data, ExchangeInstID, pDepthMarketData);
 */
#define FIELD_STRING_OPTIMIZED_PY(dict, c_field, pField) \
    do { \
        PyObject* _str_val = create_py_string((pField)->c_field, sizeof((pField)->c_field)); \
        if (_str_val) { \
            PyDict_SetItemString(dict, #c_field, _str_val); \
            Py_DECREF(_str_val); \
        } \
    } while(0)

/**
 * @brief 转换数值字段（零拷贝）并添加到 dict
 *
 * @param dict 目标 PyObject* dict
 * @param py_key Python 字典键名（下划线命名，如 "last_price"）
 * @param c_field C 结构体字段名（大驼峰命名，如 LastPrice）
 * @param pField C 结构体指针
 * @param py_value_func 转换为 Python 对象的函数（如 PyLong_FromLong）
 *
 * 示例：
 *   FIELD_NUMBER_PY_KEY(data, last_price, LastPrice, pDepthMarketData, PyFloat_FromDouble);
 */
#define FIELD_NUMBER_PY_KEY(dict, py_key, c_field, pField, py_value_func) \
    do { \
        PyObject* _num_val = py_value_func((pField)->c_field); \
        if (_num_val) { \
            PyDict_SetItemString(dict, #py_key, _num_val); \
            Py_DECREF(_num_val); \
        } \
    } while(0)

/**
 * @brief 转换整数字段（零拷贝，使用 PyLong_FromLong）
 * @param dict 目标字典
 * @param py_key Python 键名（下划线命名）
 * @param c_field C 结构体字段名（大驼峰命名）
 * @param pField C 结构体指针
 */
#define FIELD_INT_PY(dict, py_key, c_field, pField) \
    FIELD_NUMBER_PY_KEY(dict, py_key, c_field, pField, PyLong_FromLong)

/**
 * @brief 转换浮点数字段（零拷贝，使用 PyFloat_FromDouble）
 * @param dict 目标字典
 * @param py_key Python 键名（下划线命名）
 * @param c_field C 结构体字段名（大驼峰命名）
 * @param pField C 结构体指针
 */
#define FIELD_DOUBLE_PY(dict, py_key, c_field, pField) \
    FIELD_NUMBER_PY_KEY(dict, py_key, c_field, pField, PyFloat_FromDouble)

// =============================================================================
// 初始化辅助函数
// =============================================================================

/**
 * @brief 初始化零拷贝工具（在模块初始化时调用）
 *
 * @return 0 成功，-1 失败
 */
inline int initialize_zero_copy_utils() {
    // 初始化 numpy C API（如果尚未初始化）
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "Failed to import numpy C API");
        return -1;
    }

    // 初始化全局字符串池
    GlobalStringPools::initialize();

    return 0;
}

// =============================================================================
// 全局字符串池实例定义
// =============================================================================

// 在 .cpp 文件中定义这些实例
#define DEFINE_STRING_POOLS() \
    namespace GlobalStringPools { \
        StringPool ExchangeCodes; \
        StringPool Dates; \
        StringPool Times; \
        StringPool Instruments; \
        StringPool UserIds; \
        StringPool BrokerIds; \
    }
