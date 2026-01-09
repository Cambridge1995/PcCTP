/**
 * PcCTP - CTP PC版行情API Python绑定 (纯Python C API + 零拷贝优化 + gil优化 + 强内联版本)
 *
 * 核心特性：
 * - 使用 tp_dealloc 控制析构
 * - 使用 PyGILState_Ensure/Release 管理 GIL
 * - 回调函数使用下划线命名 (如 on_front_connected)
 * - 与PyCTP相同的资源管理机制
 *
 * 性能优化：
 * - 字符串池零拷贝复用（交易所代码、日期、时间、合约代码）
 * - 数值类型直接传递（零拷贝）
 *
 * 内联优化（本版本新增）：
 * - 强制内联关键路径代码
 * - 编译器特定内联属性（__forceinline / __attribute__((always_inline))）
 * - 优化宏展开减少函数调用
 * - constexpr 编译期常量
 */

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <vector>
#include <string>
#include <cstring>

// 通用工具库（包含字符串池等）
// NumPy 由各 .cpp 文件直接包含
#include "util.h"

// 通用基类和宏（新增）
#include "common.h"

// PC版 CTP 行情 API 头文件
#if defined(_WIN64)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win64/ThostFtdcMdApi.h"
    #pragma warning(pop)
#else
    #include "ctp/PC/linux/ThostFtdcMdApi.h"
#endif

// =============================================================================
// 定义全局字符串池实例
// =============================================================================

DEFINE_STRING_POOLS()

// =============================================================================
// Python对象结构定义 (参考PyCTP)
// =============================================================================

/**
 * @brief MdApi Python对象结构
 *
 * 对应 PyCTP 的 CTP_THOST_FTDC_MD_API 结构
 */
typedef struct {
    PyObject_HEAD           // Python对象头
    CThostFtdcMdApi* api;   // CTP行情API指针
    CThostFtdcMdSpi* spi;   // CTP回调SPI指针
    PyObject* py_spi;       // Python回调对象
} MdApiObject;

// =============================================================================
// SPI回调类声明 (使用下划线命名法 + 内联优化)
// =============================================================================

/**
 * @brief CTP SPI回调类 (C++实现，内联优化)
 *
 * 将CTP的回调转发到Python对象，使用下划线命名法
 * 关键路径代码使用强制内联优化
 */
class PyMdSpi : public CThostFtdcMdSpi {
private:
    MdApiObject* m_api;      // 持有MdApi对象指针

public:
    FORCE_INLINE PyMdSpi(MdApiObject* api) : m_api(api) {}

    // 虚析构函数：确保通过基类指针删除时正确调用派生类析构函数
    virtual ~PyMdSpi() = default;

    // ------------------------------------------------------------------------
    // 连接相关回调 (使用通用宏)
    // ------------------------------------------------------------------------
    CALLBACK_0_PARAM(OnFrontConnected, "on_front_connected")
    CALLBACK_1_PARAM_INT(OnFrontDisconnected, nReason, "on_front_disconnected")
    CALLBACK_1_PARAM_INT(OnHeartBeatWarning, nTimeLapse, "on_heart_beat_warning")

    // ------------------------------------------------------------------------
    // 登录相关回调 (使用 Capsule 宏)
    // ------------------------------------------------------------------------
    CAPSULE_CALLBACK_RSP(OnRspUserLogin, CThostFtdcRspUserLoginField, "RspUserLogin", "on_rsp_user_login")
    CAPSULE_CALLBACK_RSP(OnRspUserLogout, CThostFtdcUserLogoutField, "UserLogout", "on_rsp_user_logout")

    // OnRspError 特殊处理 - 标准错误回调（3个参数：pRspInfo, nRequestID, bIsLast）
    // 虚函数不需要内联
    void OnRspError(CThostFtdcRspInfoField* pRspInfo, int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;
        PyObject* capsule_rsp_info = CREATE_CAPSULE_OR_NONE_OPT(pRspInfo, "RspInfo");
        // 直接传递原始 Capsule
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_error", "Oii",
            capsule_rsp_info, nRequestID, bIsLast);
        Py_XDECREF(capsule_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // ------------------------------------------------------------------------
    // 订阅相关回调 (使用 Capsule + 字符串池，实现在 md_spi.cpp)
    // 注意：虚函数不能真正内联，因此不使用 FORCE_INLINE_MEMBER
    // ------------------------------------------------------------------------
    void OnRspSubMarketData(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                           CThostFtdcRspInfoField* pRspInfo,
                           int nRequestID, bool bIsLast) override;
    void OnRspUnSubMarketData(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                             CThostFtdcRspInfoField* pRspInfo,
                             int nRequestID, bool bIsLast) override;

    // ------------------------------------------------------------------------
    // 询价相关回调 (实现在 md_spi.cpp)
    // ------------------------------------------------------------------------
    void OnRspSubForQuoteRsp(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                            CThostFtdcRspInfoField* pRspInfo,
                            int nRequestID, bool bIsLast) override;
    void OnRspUnSubForQuoteRsp(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                              CThostFtdcRspInfoField* pRspInfo,
                              int nRequestID, bool bIsLast) override;
    CAPSULE_CALLBACK_SINGLE(OnRtnForQuoteRsp, CThostFtdcForQuoteRspField, "ForQuoteRsp", "on_rtn_for_quote_rsp")

    // ------------------------------------------------------------------------
    // 查询 multicast 相关回调
    // 注意：响应使用 CThostFtdcMulticastInstrumentField（不带 Qry 前缀）
    // ------------------------------------------------------------------------
    CAPSULE_CALLBACK_RSP(OnRspQryMulticastInstrument, CThostFtdcMulticastInstrumentField, "MulticastInstrument", "on_rsp_qry_multicast_instrument")

    // ------------------------------------------------------------------------
    // 深度行情回调（高频路径，最强内联优化）
    // ------------------------------------------------------------------------
    CAPSULE_CALLBACK_SINGLE(OnRtnDepthMarketData, CThostFtdcDepthMarketDataField, "DepthMarketData", "on_rtn_depth_market_data")
};

// =============================================================================
// Python类型声明
// =============================================================================

extern PyTypeObject MdApiType;

// =============================================================================
// 模块初始化函数
// =============================================================================

PyMODINIT_FUNC PyInit_PcCTP(void);
