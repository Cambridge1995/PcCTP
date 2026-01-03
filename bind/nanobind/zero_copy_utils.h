/**
 * 性能优化工具函数
 *
 * 提供以下优化：
 * 1. 优化的字符串构造（非零拷贝，但开销最小）
 * 2. 字符串池（零拷贝复用，适用于有限集合）
 * 3. numpy 类型安全检查
 * 4. numpy 零拷贝访问（直接访问底层数据指针）
 */

#pragma once

#include <nanobind/nanobind.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <cstring>
#include <unordered_map>
#include <string_view>
#include <mutex>

namespace nb = nanobind;

// =============================================================================
// 优化的字符串构造
// =============================================================================

/**
 * @brief 从 C 字符串创建 Python str（优化版本）
 *
 * @param cstr C 字符串指针
 * @param max_len 最大长度（包括空字符）
 * @return nb::str Python 字符串对象
 *
 * @note 优化点：
 * 1. 使用 strnlen 计算实际长度（去除尾部空字符）
 * 2. 直接调用 PyUnicode_FromStringAndSize（比 nb::str() 更快）
 * 3. 仍需拷贝，但开销最小（无法避免，Python str 需要 UTF-8 编码）
 */
inline nb::str optimized_str(const char* cstr, size_t max_len) {
    // 计算实际长度（去除尾部空字符）
    size_t len = strnlen(cstr, max_len);

    // 直接从 C 字符串创建 Python str（仍需拷贝，但开销最小）
    // nb::steal 返回 nb::object，需要显式转换为 nb::str
    return nb::str(nb::steal(PyUnicode_FromStringAndSize(cstr, len)));
}

/**
 * @brief 从 C 字符串创建 Python str（已知长度）
 */
inline nb::str optimized_str_with_length(const char* cstr, size_t len) {
    // nb::steal 返回 nb::object，需要显式转换为 nb::str
    return nb::str(nb::steal(PyUnicode_FromStringAndSize(cstr, len)));
}

// =============================================================================
// 字符串池（零拷贝复用）
// =============================================================================

/**
 * @brief 字符串池（线程安全）
 *
 * 用于驻留重复出现的字符串，实现零拷贝复用。
 * 适用场景：
 * - 交易所代码（SHFE, DCE, CZCE 等，有限集合）
 * - 日期字符串（每个交易日固定，有限集合）
 * - 其他有限集合的枚举值
 *
 * 不适用场景：
 * - 合约代码（可能无限增长，需要定期清理）
 *
 * @note 使用 PyObject* 存储以更好地控制析构时机
 */
class StringPool {
private:
    struct StringHash {
        size_t operator()(const std::string_view& sv) const {
            return std::hash<std::string_view>{}(sv);
        }
    };

    // 使用 PyObject* 而不是 nb::str，以便手动控制引用计数
    std::unordered_map<std::string, PyObject*, StringHash> pool_;
    std::mutex mutex_;

public:
    /**
     * @brief 析构函数
     *
     * @note 检查 Python 解释器状态，仅在安全时释放 Python 对象
     */
    ~StringPool() {
        // 检查 Python 解释器状态
        // 保守策略：仅当能确认 Python 稳定运行时才释放对象
        bool python_active = false;
        if (Py_IsInitialized()) {
            #if PY_VERSION_HEX >= 0x030D0000
            // Python 3.13+: 精确判断是否在关闭阶段
            python_active = !Py_IsFinalizing();
            #else
            // Python 3.8-3.12: 无法精确判断，采用保守策略
            // 跳过释放以避免潜在的 0xC0000005 崩溃
            // 操作系统会在进程退出时回收内存
            python_active = false;
            #endif
        }

        // 仅在解释器仍在运行时释放 Python 对象
        if (python_active) {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& pair : pool_) {
                Py_XDECREF(pair.second);
            }
            pool_.clear();
        }
        // 如果解释器已关闭，不做任何操作
        // 程序退出时操作系统会回收内存
    }

    /**
     * @brief 驻留字符串（返回 Python str）
     *
     * @param str C 字符串指针
     * @param max_len 最大长度
     * @return nb::str Python 字符串对象（如果已存在则复用，否则创建并缓存）
     *
     * @note 检查 Python 解释器状态，仅在解释器仍在运行时创建新字符串
     */
    nb::str intern(const char* str, size_t max_len) {
        size_t len = strnlen(str, max_len);
        std::string_view sv(str, len);

        std::lock_guard<std::mutex> lock(mutex_);

        // 将 string_view 转换为 string 用于查找
        std::string key(sv);
        auto it = pool_.find(key);
        if (it != pool_.end()) {
            // 返回已缓存的 Python str（零拷贝复用）
            // 使用 nb::borrow 标记为 borrowed 引用，nb::str 会增加引用计数
            Py_INCREF(it->second);
            return nb::str(nb::borrow(it->second));
        }

        // 创建并缓存新字符串
        // 仅在 Python 解释器仍在运行时创建
        nb::str result;
        if (Py_IsInitialized()) {
            #if PY_VERSION_HEX >= 0x030D0000
            // Python 3.13+: 精确判断是否在关闭阶段
            if (!Py_IsFinalizing()) {
                result = optimized_str_with_length(str, len);
            }
            #else
            // Python 3.8-3.12: 保守策略，假设可以创建字符串
            // (创建新字符串通常比释放对象更安全)
            result = optimized_str_with_length(str, len);
            #endif
        }

        // 缓存 PyObject*（增加引用计数以保持所有权）
        if (result.ptr() != nullptr) {
            Py_INCREF(result.ptr());  // pool 持有一份引用
            pool_.emplace(std::move(key), result.ptr());
            return result;
        }
        // 返回空字符串作为后备
        return nb::str("");
    }

    /**
     * @brief 清空字符串池
     *
     * @note 检查 Python 解释器状态，仅在安全时释放 Python 对象
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);

        // 检查 Python 解释器状态
        // 保守策略：仅当能确认 Python 稳定运行时才释放对象
        bool python_active = false;
        if (Py_IsInitialized()) {
            #if PY_VERSION_HEX >= 0x030D0000
            // Python 3.13+: 精确判断是否在关闭阶段
            python_active = !Py_IsFinalizing();
            #else
            // Python 3.8-3.12: 无法精确判断，采用保守策略
            // 跳过释放以避免潜在的 0xC0000005 崩溃
            // 操作系统会在进程退出时回收内存
            python_active = false;
            #endif
        }

        // 仅在解释器仍在运行时释放 Python 对象
        if (python_active) {
            for (auto& pair : pool_) {
                Py_XDECREF(pair.second);
            }
        }
        // 无论是否释放，都清空容器
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
    static StringPool ExchangeCodes;

    // 日期池（每个交易日约 1 个新值，可定期清理）
    static StringPool Dates;

    // 时间字符串池（HH:MM:SS 格式，有限集合）
    static StringPool Times;

    // 合约代码池（订阅的合约有限，使用字符串池）
    static StringPool Instruments;

    // 用户代码池（登录用户有限）
    static StringPool UserIds;

    // 经纪公司代码池（经纪公司有限）
    static StringPool BrokerIds;

    // 枚举值池（OrderStatus, Direction 等，有限集合）
    static StringPool EnumValues;

    /**
     * @brief 初始化所有字符串池
     */
    inline void initialize() {
        // 预分配空间（避免 rehash）
        ExchangeCodes.reserve(20);     // 约 10 个交易所
        Dates.reserve(15);             // 两周交易日（考虑跨周末）
        Times.reserve(36000);          // 完整交易时段：08:55-10:15, 10:30-11:30, 13:30-15:00, 20:55-02:30（夜盘跨日）
        Instruments.reserve(1000);     // 订阅合约数量
        UserIds.reserve(40);           // 用户数量（30交易员 + 技术账号）
        BrokerIds.reserve(25);         // 经纪公司数量
        EnumValues.reserve(100);       // 约 50-100 个枚举值
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
            // 使用 Python API 输出警告
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
     * @return nb::str Python 字符串对象
     */
    inline nb::str intern_instrument(const char* str, size_t max_len) {
        nb::str result = Instruments.intern(str, max_len);
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
// numpy 类型检查工具
// =============================================================================

/**
 * @brief 检查对象是否为 numpy 数组
 */
inline bool is_numpy_array(nb::object obj) {
    return PyArray_Check(obj.ptr()) != 0;
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
// 辅助宏（简化代码编写）
// =============================================================================

/**
 * @brief 使用字符串池转换字段
 *
 * @param dict 目标 dict
 * @param c_field C 结构体字段名
 * @param pool 使用的字符串池
 *
 * 示例：
 *   FIELD_STRING_POOL(data, ExchangeID, GlobalStringPools::ExchangeCodes);
 */
#define FIELD_STRING_POOL(dict, c_field, pool) \
    dict[#c_field] = (pool).intern(pField->c_field, sizeof(pField->c_field))

/**
 * @brief 使用优化构造转换字段
 *
 * @param dict 目标 dict
 * @param c_field C 结构体字段名
 *
 * 示例：
 *   FIELD_STRING_OPTIMIZED(data, TradingDay);
 */
#define FIELD_STRING_OPTIMIZED(dict, c_field) \
    dict[#c_field] = optimized_str(pField->c_field, sizeof(pField->c_field))

/**
 * @brief 转换数值字段（零拷贝）
 *
 * @param dict 目标 dict
 * @param c_field C 结构体字段名
 *
 * 示例：
 *   FIELD_NUMBER(data, LastPrice);
 */
#define FIELD_NUMBER(dict, c_field) \
    dict[#c_field] = pField->c_field

/**
 * @brief 转换字符串数组字段
 *
 * @param dict 目标 dict
 * @param c_field C 结构体字段名
 * @param count 数组元素个数
 *
 * 示例：
 *   FIELD_STRING_ARRAY(data, BrokerID, 1);
 */
#define FIELD_STRING_ARRAY(dict, c_field, count) \
    do { \
        nb::list arr; \
        for (size_t i = 0; i < (count); ++i) { \
            arr.append(optimized_str(pField->c_field[i], sizeof(pField->c_field[i]))); \
        } \
        dict[#c_field] = arr; \
    } while(0)

// =============================================================================
// 零拷贝 numpy 数组访问（类型安全）
// =============================================================================

/**
 * @brief numpy 数组访问助手（类型安全）
 */
class NumpyStringArrayAccessor {
private:
    PyArrayObject* array_;
    char* data_;
    npy_intp size_;
    npy_intp stride_;
    bool is_valid_;

public:
    /**
     * @brief 构造函数（进行类型检查）
     */
    explicit NumpyStringArrayAccessor(nb::object obj)
        : array_(nullptr), data_(nullptr), size_(0), stride_(0), is_valid_(false)
    {
        // 检查是否为 numpy 数组
        if (!is_numpy_array(obj)) {
            PyErr_SetString(PyExc_TypeError, "Expected numpy.ndarray");
            return;
        }

        array_ = reinterpret_cast<PyArrayObject*>(obj.ptr());

        // 检查是否为字符串类型
        if (!is_string_array(array_)) {
            PyErr_SetString(PyExc_TypeError, "Array must be string dtype (U or S)");
            return;
        }

        // 检查是否为一维
        if (!is_1d_array(array_)) {
            PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
            return;
        }

        // 获取数组信息
        data_ = static_cast<char*>(PyArray_DATA(array_));
        size_ = PyArray_SIZE(array_);
        stride_ = PyArray_STRIDE(array_, 0);
        is_valid_ = true;
    }

    /**
     * @brief 是否有效
     */
    bool is_valid() const {
        return is_valid_;
    }

    /**
     * @brief 获取数组大小
     */
    npy_intp size() const {
        return size_;
    }

    /**
     * @brief 获取元素指针（零拷贝访问）
     */
    char* operator[](npy_intp index) const {
        if (index < 0 || index >= size_) {
            PyErr_SetString(PyExc_IndexError, "Array index out of range");
            return nullptr;
        }
        return data_ + index * stride_;
    }

    /**
     * @brief 转换为 CTP API 所需的 char* 数组
     */
    std::vector<char*> to_char_ptr_vector() const {
        std::vector<char*> result;
        result.reserve(size_);
        for (npy_intp i = 0; i < size_; ++i) {
            result.push_back((*this)[i]);
        }
        return result;
    }
};

// =============================================================================
// 初始化辅助函数
// =============================================================================

/**
 * @brief 初始化零拷贝工具（在模块初始化时调用）
 */
inline void initialize_zero_copy_utils() {
    // 初始化 numpy C API（如果尚未初始化）
    if (_import_array() < 0) {
        throw std::runtime_error("Failed to import numpy C API");
    }

    // 初始化全局字符串池
    GlobalStringPools::initialize();
}
