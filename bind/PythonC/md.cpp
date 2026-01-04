/**
 * PcCTP - CTP PC版行情API Python绑定 (纯Python C API + 零拷贝优化 + gil优化 + 强内联版本)
 *
 * 完全仿照PyCTP的设计，使用纯Python C API实现
 * 所有函数名使用Python下划线命名法 (snake_case)
 *
 * 核心特性：
 * - 使用 tp_dealloc 控制析构
 * - 使用 PyGILState_Ensure/Release 管理 GIL
 * - 回调函数使用下划线命名 (如 on_front_connected)
 * - 与PyCTP相同的资源管理机制
 *
 * 性能优化：
 * - 字符串池零拷贝复用（交易所代码、日期、时间、合约代码）
 * - NumPy 数组零拷贝支持（订阅行情）
 * - 数值类型直接传递（零拷贝）
 * - 优化字符串构造（动态字符串）
 *
 * 内联优化（本版本新增）：
 * - 强制内联关键路径代码
 * - 编译器特定内联属性（__forceinline / __attribute__((always_inline))）
 * - 优化宏展开减少函数调用
 * - constexpr 编译期常量
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <vector>
#include <string>
#include <cstring>

// NumPy C API（宏展开，零开销，直接结构体访问）
#define PY_ARRAY_UNIQUE_SYMBOL PcCTP_CAPI_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

// 通用工具库（包含平台特定宏、编译期常量、字典设置函数等）
#include "utils.h"

// PC版 CTP 行情 API 头文件
#if defined(_WIN64)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win64/ThostFtdcMdApi.h"
    #pragma warning(pop)
#elif defined(_WIN32)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win32/ThostFtdcMdApi.h"
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
// SPI回调类实现 (使用下划线命名法 + 内联优化)
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

    virtual ~PyMdSpi() {
        // 析构时检查Python状态
        if (!Py_IsInitialized()) {
            return;
        }
        #if PY_VERSION_HEX >= 0x030D0000
        if (Py_IsFinalizing()) {
            return;
        }
        #endif
    }

    // ------------------------------------------------------------------------
    // 连接相关回调 (使用下划线命名 + 内联优化)
    // ------------------------------------------------------------------------

    FORCE_INLINE_MEMBER void OnFrontConnected() override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_front_connected", NULL);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnFrontDisconnected(int nReason) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;
        PyObject* py_reason = PyLong_FromLong(nReason);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_front_disconnected", "(O)", py_reason);
        Py_XDECREF(py_reason);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnHeartBeatWarning(int nTimeLapse) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;
        PyObject* py_time_lapse = PyLong_FromLong(nTimeLapse);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_heart_beat_warning", "(O)", py_time_lapse);
        Py_XDECREF(py_time_lapse);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // ------------------------------------------------------------------------
    // 登录相关回调 (使用下划线命名 + 内联优化)
    // ------------------------------------------------------------------------

    /**
     * @brief 构建登录响应字典（内联优化）
     */
    FORCE_INLINE_MEMBER PyObject* build_login_dict(CThostFtdcRspUserLoginField* pRspUserLogin) {
        if (!pRspUserLogin) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* login_dict = PyDict_New();
        if (!login_dict) return nullptr;

        // 使用优化的字符串池（零拷贝）
        dict_set_pooled_string(login_dict, "trading_day",
            pRspUserLogin->TradingDay, sizeof(pRspUserLogin->TradingDay), GlobalStringPools::Dates);
        dict_set_pooled_string(login_dict, "login_time",
            pRspUserLogin->LoginTime, sizeof(pRspUserLogin->LoginTime), GlobalStringPools::Times);
        dict_set_pooled_string(login_dict, "broker_id",
            pRspUserLogin->BrokerID, sizeof(pRspUserLogin->BrokerID), GlobalStringPools::BrokerIds);
        dict_set_pooled_string(login_dict, "user_id",
            pRspUserLogin->UserID, sizeof(pRspUserLogin->UserID), GlobalStringPools::UserIds);
        dict_set_pooled_string(login_dict, "shfe_time",
            pRspUserLogin->SHFETime, sizeof(pRspUserLogin->SHFETime), GlobalStringPools::Times);
        dict_set_pooled_string(login_dict, "dce_time",
            pRspUserLogin->DCETime, sizeof(pRspUserLogin->DCETime), GlobalStringPools::Times);
        dict_set_pooled_string(login_dict, "czce_time",
            pRspUserLogin->CZCETime, sizeof(pRspUserLogin->CZCETime), GlobalStringPools::Times);
        dict_set_pooled_string(login_dict, "ffex_time",
            pRspUserLogin->FFEXTime, sizeof(pRspUserLogin->FFEXTime), GlobalStringPools::Times);
        dict_set_pooled_string(login_dict, "ine_time",
            pRspUserLogin->INETime, sizeof(pRspUserLogin->INETime), GlobalStringPools::Times);
        dict_set_pooled_string(login_dict, "gfex_time",
            pRspUserLogin->GFEXTime, sizeof(pRspUserLogin->GFEXTime), GlobalStringPools::Times);

        // 动态字符串（优化构造）
        dict_set_string(login_dict, "system_name",
            pRspUserLogin->SystemName, sizeof(pRspUserLogin->SystemName));
        dict_set_string(login_dict, "max_order_ref",
            pRspUserLogin->MaxOrderRef, sizeof(pRspUserLogin->MaxOrderRef));

        // 数值（零拷贝）
        dict_set_long(login_dict, "front_id", pRspUserLogin->FrontID);
        dict_set_long(login_dict, "session_id", pRspUserLogin->SessionID);

        return login_dict;
    }

    /**
     * @brief 构建错误信息字典（内联优化）
     */
    FORCE_INLINE_MEMBER PyObject* build_error_dict(CThostFtdcRspInfoField* pRspInfo) {
        if (!pRspInfo) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* error_dict = PyDict_New();
        if (!error_dict) return nullptr;

        dict_set_long(error_dict, "error_id", pRspInfo->ErrorID);
        // CTP API 的错误消息使用 GBK 编码，需要使用 dict_set_gbk_string
        dict_set_gbk_string(error_dict, "error_msg",
            pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));

        return error_dict;
    }

    FORCE_INLINE_MEMBER void OnRspUserLogin(CThostFtdcRspUserLoginField* pRspUserLogin,
                                            CThostFtdcRspInfoField* pRspInfo,
                                            int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rsp_user_login = build_login_dict(pRspUserLogin);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_user_login", "(OOii)",
            py_rsp_user_login, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rsp_user_login);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspUserLogout(CThostFtdcUserLogoutField* pUserLogout,
                                             CThostFtdcRspInfoField* pRspInfo,
                                             int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_user_logout = Py_None;
        Py_INCREF(Py_None);
        if (pUserLogout) {
            py_user_logout = PyDict_New();
            dict_set_pooled_string(py_user_logout, "broker_id",
                pUserLogout->BrokerID, sizeof(pUserLogout->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_user_logout, "user_id",
                pUserLogout->UserID, sizeof(pUserLogout->UserID), GlobalStringPools::UserIds);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_user_logout", "(OOii)",
            py_user_logout, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_user_logout);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspError(CThostFtdcRspInfoField* pRspInfo, int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_error", "(Oii)",
            py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // ------------------------------------------------------------------------
    // 订阅相关回调 (使用下划线命名 + 内联优化)
    // ------------------------------------------------------------------------

    FORCE_INLINE_MEMBER void OnRspSubMarketData(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                               CThostFtdcRspInfoField* pRspInfo,
                                               int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_instrument_id = Py_None;
        Py_INCREF(Py_None);
        if (pSpecificInstrument) {
            py_instrument_id = GlobalStringPools::intern_instrument(
                pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_sub_market_data", "(OOii)",
            py_instrument_id, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_instrument_id);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspUnSubMarketData(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                                  CThostFtdcRspInfoField* pRspInfo,
                                                  int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_instrument_id = Py_None;
        Py_INCREF(Py_None);
        if (pSpecificInstrument) {
            py_instrument_id = GlobalStringPools::intern_instrument(
                pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_un_sub_market_data", "(OOii)",
            py_instrument_id, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_instrument_id);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // ------------------------------------------------------------------------
    // 询价相关回调 (使用下划线命名 + 内联优化)
    // ------------------------------------------------------------------------

    FORCE_INLINE_MEMBER void OnRspSubForQuoteRsp(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                                 CThostFtdcRspInfoField* pRspInfo,
                                                 int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_instrument_id = Py_None;
        Py_INCREF(Py_None);
        if (pSpecificInstrument) {
            py_instrument_id = GlobalStringPools::intern_instrument(
                pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_sub_for_quote_rsp", "(OOii)",
            py_instrument_id, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_instrument_id);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspUnSubForQuoteRsp(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                                   CThostFtdcRspInfoField* pRspInfo,
                                                   int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_instrument_id = Py_None;
        Py_INCREF(Py_None);
        if (pSpecificInstrument) {
            py_instrument_id = GlobalStringPools::intern_instrument(
                pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_un_sub_for_quote_rsp", "(OOii)",
            py_instrument_id, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_instrument_id);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 询价通知回调（内联优化）
     */
    FORCE_INLINE_MEMBER void OnRtnForQuoteRsp(CThostFtdcForQuoteRspField* pForQuoteRsp) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        if (!pForQuoteRsp) {
            PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_for_quote_rsp", "(O)", Py_None);
            if (!result) PyErr_Print();
            Py_XDECREF(result);
            return;
        }

        PyObject* for_quote_dict = PyDict_New();
        if (!for_quote_dict) return;

        // 使用字符串池优化
        dict_set_pooled_string(for_quote_dict, "trading_day",
            pForQuoteRsp->TradingDay, sizeof(pForQuoteRsp->TradingDay), GlobalStringPools::Dates);
        dict_set_pooled_string(for_quote_dict, "action_day",
            pForQuoteRsp->ActionDay, sizeof(pForQuoteRsp->ActionDay), GlobalStringPools::Dates);
        dict_set_pooled_string(for_quote_dict, "instrument_id",
            pForQuoteRsp->InstrumentID, sizeof(pForQuoteRsp->InstrumentID), GlobalStringPools::Instruments);

        // 动态字符串
        dict_set_string(for_quote_dict, "exchange_id",
            pForQuoteRsp->ExchangeID, sizeof(pForQuoteRsp->ExchangeID));
        dict_set_string(for_quote_dict, "for_quote_sys_id",
            pForQuoteRsp->ForQuoteSysID, sizeof(pForQuoteRsp->ForQuoteSysID));
        dict_set_string(for_quote_dict, "for_quote_time",
            pForQuoteRsp->ForQuoteTime, sizeof(pForQuoteRsp->ForQuoteTime));

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_for_quote_rsp", "(O)", for_quote_dict);

        Py_XDECREF(for_quote_dict);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // ------------------------------------------------------------------------
    // 查询 multicast 相关回调 (使用下划线命名 + 内联优化)
    // ------------------------------------------------------------------------

    /**
     * @brief 查询 multicast 响应回调（内联优化）
     */
    FORCE_INLINE_MEMBER void OnRspQryMulticastInstrument(CThostFtdcMulticastInstrumentField* pMulticastInstrument,
                                                         CThostFtdcRspInfoField* pRspInfo,
                                                         int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* multicast_dict = Py_None;
        Py_INCREF(Py_None);
        if (pMulticastInstrument) {
            multicast_dict = PyDict_New();

            // 数值字段
            dict_set_long(multicast_dict, "topic_id", pMulticastInstrument->TopicID);

            // 动态字符串
            dict_set_string(multicast_dict, "instrument_id",
                pMulticastInstrument->InstrumentID, sizeof(pMulticastInstrument->InstrumentID));

            // 数值字段
            dict_set_long(multicast_dict, "instrument_no", pMulticastInstrument->InstrumentNo);
            dict_set_double(multicast_dict, "code_price", pMulticastInstrument->CodePrice);
            dict_set_long(multicast_dict, "volume_multiple", pMulticastInstrument->VolumeMultiple);
            dict_set_double(multicast_dict, "price_tick", pMulticastInstrument->PriceTick);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_multicast_instrument", "(OOii)",
            multicast_dict, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(multicast_dict);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // ------------------------------------------------------------------------
    // 深度行情回调 (高频路径，最强内联优化)
    // ------------------------------------------------------------------------

    /**
     * @brief 优化的深度行情字典构建（强制内联，高频路径）
     */
    FORCE_INLINE_MEMBER PyObject* build_market_data_dict(CThostFtdcDepthMarketDataField* pDepthMarketData) {
        if (!pDepthMarketData) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* data = PyDict_New();
        if (!data) return nullptr;

        // ========== 字符串池零拷贝优化（有限集合） ==========
        dict_set_pooled_string(data, "exchange_id",
            pDepthMarketData->ExchangeID, sizeof(pDepthMarketData->ExchangeID), GlobalStringPools::ExchangeCodes);
        dict_set_pooled_string(data, "trading_day",
            pDepthMarketData->TradingDay, sizeof(pDepthMarketData->TradingDay), GlobalStringPools::Dates);
        dict_set_pooled_string(data, "action_day",
            pDepthMarketData->ActionDay, sizeof(pDepthMarketData->ActionDay), GlobalStringPools::Dates);
        dict_set_pooled_string(data, "update_time",
            pDepthMarketData->UpdateTime, sizeof(pDepthMarketData->UpdateTime), GlobalStringPools::Times);
        dict_set_pooled_string(data, "instrument_id",
            pDepthMarketData->InstrumentID, sizeof(pDepthMarketData->InstrumentID), GlobalStringPools::Instruments);

        // ========== 动态字符串（优化构造） ==========
        dict_set_string(data, "exchange_inst_id",
            pDepthMarketData->ExchangeInstID, sizeof(pDepthMarketData->ExchangeInstID));

        // ========== 数值字段（零拷贝） ==========
        dict_set_double(data, "last_price", pDepthMarketData->LastPrice);
        dict_set_double(data, "pre_settlement_price", pDepthMarketData->PreSettlementPrice);
        dict_set_double(data, "pre_close_price", pDepthMarketData->PreClosePrice);
        dict_set_double(data, "pre_open_interest", pDepthMarketData->PreOpenInterest);
        dict_set_double(data, "open_price", pDepthMarketData->OpenPrice);
        dict_set_double(data, "highest_price", pDepthMarketData->HighestPrice);
        dict_set_double(data, "lowest_price", pDepthMarketData->LowestPrice);
        dict_set_long(data, "volume", pDepthMarketData->Volume);
        dict_set_double(data, "turnover", pDepthMarketData->Turnover);
        dict_set_double(data, "open_interest", pDepthMarketData->OpenInterest);
        dict_set_double(data, "close_price", pDepthMarketData->ClosePrice);
        dict_set_double(data, "settlement_price", pDepthMarketData->SettlementPrice);
        dict_set_double(data, "upper_limit_price", pDepthMarketData->UpperLimitPrice);
        dict_set_double(data, "lower_limit_price", pDepthMarketData->LowerLimitPrice);
        dict_set_double(data, "pre_delta", pDepthMarketData->PreDelta);
        dict_set_double(data, "curr_delta", pDepthMarketData->CurrDelta);
        dict_set_long(data, "update_millisec", pDepthMarketData->UpdateMillisec);

        // ========== 五档买卖价（零拷贝） ==========
        dict_set_double(data, "bid_price1", pDepthMarketData->BidPrice1);
        dict_set_long(data, "bid_volume1", pDepthMarketData->BidVolume1);
        dict_set_double(data, "ask_price1", pDepthMarketData->AskPrice1);
        dict_set_long(data, "ask_volume1", pDepthMarketData->AskVolume1);

        dict_set_double(data, "bid_price2", pDepthMarketData->BidPrice2);
        dict_set_long(data, "bid_volume2", pDepthMarketData->BidVolume2);
        dict_set_double(data, "ask_price2", pDepthMarketData->AskPrice2);
        dict_set_long(data, "ask_volume2", pDepthMarketData->AskVolume2);

        dict_set_double(data, "bid_price3", pDepthMarketData->BidPrice3);
        dict_set_long(data, "bid_volume3", pDepthMarketData->BidVolume3);
        dict_set_double(data, "ask_price3", pDepthMarketData->AskPrice3);
        dict_set_long(data, "ask_volume3", pDepthMarketData->AskVolume3);

        dict_set_double(data, "bid_price4", pDepthMarketData->BidPrice4);
        dict_set_long(data, "bid_volume4", pDepthMarketData->BidVolume4);
        dict_set_double(data, "ask_price4", pDepthMarketData->AskPrice4);
        dict_set_long(data, "ask_volume4", pDepthMarketData->AskVolume4);

        dict_set_double(data, "bid_price5", pDepthMarketData->BidPrice5);
        dict_set_long(data, "bid_volume5", pDepthMarketData->BidVolume5);
        dict_set_double(data, "ask_price5", pDepthMarketData->AskPrice5);
        dict_set_long(data, "ask_volume5", pDepthMarketData->AskVolume5);

        dict_set_double(data, "average_price", pDepthMarketData->AveragePrice);

        return data;
    }

    /**
     * @brief 深度行情回调（高频路径，最强内联优化）
     */
    FORCE_INLINE_MEMBER void OnRtnDepthMarketData(CThostFtdcDepthMarketDataField* pDepthMarketData) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        if (!pDepthMarketData) {
            PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_depth_market_data", "(O)", Py_None);
            if (!result) PyErr_Print();
            Py_XDECREF(result);
            return;
        }

        // 使用优化的字典构建函数
        PyObject* data = build_market_data_dict(pDepthMarketData);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_depth_market_data", "(O)", data);

        Py_XDECREF(data);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }
};

// =============================================================================
// MdApi 方法实现 (使用下划线命名法 + 内联优化)
// =============================================================================

/**
 * @brief 创建MdApi实例 (静态方法)
 * Python命名: create_ftdc_md_api
 */
static PyObject* MdApi_create_ftdc_md_api(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* flow_path = "";
    int is_using_udp = 0;
    int is_multicast = 0;

    static char* kwlist[] = {"flow_path", "is_using_udp", "is_multicast", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sii", kwlist, &flow_path, &is_using_udp, &is_multicast)) {
        return NULL;
    }

    PyTypeObject* type = (PyTypeObject*)self;
    MdApiObject* obj = (MdApiObject*)type->tp_alloc(type, 0);
    if (!obj) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate MdApi object");
        return NULL;
    }

    obj->api = CThostFtdcMdApi::CreateFtdcMdApi(flow_path, is_using_udp != 0, is_multicast != 0);
    obj->spi = nullptr;
    obj->py_spi = nullptr;

    if (!obj->api) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create CTP MdApi");
        return NULL;
    }

    return (PyObject*)obj;
}

/**
 * @brief 获取API版本 (静态方法)
 * Python命名: get_api_version
 */
static PyObject* MdApi_get_api_version(PyObject* self, PyObject* args) {
    const char* version = CThostFtdcMdApi::GetApiVersion();
    return PyUnicode_FromString(version);
}

/**
 * @brief 初始化
 * Python命名: init
 */
static PyObject* MdApi_init(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    obj->api->Init();
    Py_RETURN_NONE;
}

/**
 * @brief 等待线程结束
 * Python命名: join
 */
static PyObject* MdApi_join(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    int result = obj->api->Join();
    return PyLong_FromLong(result);
}

/**
 * @brief 获取交易日
 * Python命名: get_trading_day
 */
static PyObject* MdApi_get_trading_day(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    const char* trading_day = obj->api->GetTradingDay();
    return PyUnicode_FromString(trading_day);
}

/**
 * @brief 注册前台地址
 * Python命名: register_front
 */
static PyObject* MdApi_register_front(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    const char* front_address;
    if (!PyArg_ParseTuple(args, "s", &front_address)) {
        return NULL;
    }

    obj->api->RegisterFront((char*)front_address);
    Py_RETURN_NONE;
}

/**
 * @brief 注册名称服务器
 * Python命名: register_name_server
 */
static PyObject* MdApi_register_name_server(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    const char* ns_address;
    if (!PyArg_ParseTuple(args, "s", &ns_address)) {
        return NULL;
    }

    obj->api->RegisterNameServer((char*)ns_address);
    Py_RETURN_NONE;
}

/**
 * @brief 注册SPI回调
 * Python命名: register_spi
 */
static PyObject* MdApi_register_spi(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* py_spi;
    if (!PyArg_ParseTuple(args, "O", &py_spi)) {
        return NULL;
    }

    Py_XDECREF(obj->py_spi);
    obj->py_spi = py_spi;
    Py_INCREF(obj->py_spi);

    if (obj->spi) {
        delete dynamic_cast<PyMdSpi*>(obj->spi);
    }
    obj->spi = new PyMdSpi(obj);

    obj->api->RegisterSpi(obj->spi);

    Py_RETURN_NONE;
}

/**
 * @brief 订阅行情（支持 NumPy 零拷贝）
 * Python命名: subscribe_market_data
 */
static PyObject* MdApi_subscribe_market_data(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // NumPy 零拷贝路径
    if (is_numpy_array(instrument_ids)) {
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(instrument_ids);

        if (!is_string_array(array)) {
            PyErr_SetString(PyExc_TypeError, "NumPy array must be string dtype (U or S)");
            return NULL;
        }
        if (!is_1d_array(array)) {
            PyErr_SetString(PyExc_ValueError, "NumPy array must be 1-dimensional");
            return NULL;
        }

        char* data = static_cast<char*>(PyArray_DATA(array));
        npy_intp size = PyArray_SIZE(array);
        npy_intp stride = PyArray_STRIDE(array, 0);

        std::vector<char*> ids;
        ids.reserve(size);
        for (npy_intp i = 0; i < size; ++i) {
            ids.push_back(data + i * stride);
        }

        int result = obj->api->SubscribeMarketData(ids.data(), static_cast<int>(size));
        return PyLong_FromLong(result);
    }

    // List 回退路径
    if (PyList_Check(instrument_ids)) {
        Py_ssize_t count = PyList_Size(instrument_ids);
        std::vector<char*> ids;
        std::vector<std::string> strs;
        ids.reserve(count);
        strs.reserve(count);

        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* item = PyList_GetItem(instrument_ids, i);
            if (!PyUnicode_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "instrument_ids must be strings");
                return NULL;
            }
            const char* str = PyUnicode_AsUTF8(item);
            strs.push_back(str);
            ids.push_back(const_cast<char*>(strs.back().c_str()));
        }

        int result = obj->api->SubscribeMarketData(ids.data(), static_cast<int>(count));
        return PyLong_FromLong(result);
    }

    PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
    return NULL;
}

/**
 * @brief 取消订阅行情（支持 NumPy 零拷贝）
 * Python命名: un_subscribe_market_data
 */
static PyObject* MdApi_un_subscribe_market_data(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // NumPy 零拷贝路径
    if (is_numpy_array(instrument_ids)) {
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(instrument_ids);

        if (!is_string_array(array)) {
            PyErr_SetString(PyExc_TypeError, "NumPy array must be string dtype (U or S)");
            return NULL;
        }
        if (!is_1d_array(array)) {
            PyErr_SetString(PyExc_ValueError, "NumPy array must be 1-dimensional");
            return NULL;
        }

        char* data = static_cast<char*>(PyArray_DATA(array));
        npy_intp size = PyArray_SIZE(array);
        npy_intp stride = PyArray_STRIDE(array, 0);

        std::vector<char*> ids;
        ids.reserve(size);
        for (npy_intp i = 0; i < size; ++i) {
            ids.push_back(data + i * stride);
        }

        int result = obj->api->UnSubscribeMarketData(ids.data(), static_cast<int>(size));
        return PyLong_FromLong(result);
    }

    // List 回退路径
    if (PyList_Check(instrument_ids)) {
        Py_ssize_t count = PyList_Size(instrument_ids);
        std::vector<char*> ids;
        std::vector<std::string> strs;
        ids.reserve(count);
        strs.reserve(count);

        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* item = PyList_GetItem(instrument_ids, i);
            if (!PyUnicode_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "instrument_ids must be strings");
                return NULL;
            }
            const char* str = PyUnicode_AsUTF8(item);
            strs.push_back(str);
            ids.push_back(const_cast<char*>(strs.back().c_str()));
        }

        int result = obj->api->UnSubscribeMarketData(ids.data(), static_cast<int>(count));
        return PyLong_FromLong(result);
    }

    PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
    return NULL;
}

/**
 * @brief 订阅询价（支持 NumPy 零拷贝）
 * Python命名: subscribe_for_quote_rsp
 */
static PyObject* MdApi_subscribe_for_quote_rsp(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // NumPy 零拷贝路径
    if (is_numpy_array(instrument_ids)) {
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(instrument_ids);

        if (!is_string_array(array)) {
            PyErr_SetString(PyExc_TypeError, "NumPy array must be string dtype (U or S)");
            return NULL;
        }
        if (!is_1d_array(array)) {
            PyErr_SetString(PyExc_ValueError, "NumPy array must be 1-dimensional");
            return NULL;
        }

        char* data = static_cast<char*>(PyArray_DATA(array));
        npy_intp size = PyArray_SIZE(array);
        npy_intp stride = PyArray_STRIDE(array, 0);

        std::vector<char*> ids;
        ids.reserve(size);
        for (npy_intp i = 0; i < size; ++i) {
            ids.push_back(data + i * stride);
        }

        int result = obj->api->SubscribeForQuoteRsp(ids.data(), static_cast<int>(size));
        return PyLong_FromLong(result);
    }

    // List 回退路径
    if (PyList_Check(instrument_ids)) {
        Py_ssize_t count = PyList_Size(instrument_ids);
        std::vector<char*> ids;
        std::vector<std::string> strs;
        ids.reserve(count);
        strs.reserve(count);

        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* item = PyList_GetItem(instrument_ids, i);
            if (!PyUnicode_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "instrument_ids must be strings");
                return NULL;
            }
            const char* str = PyUnicode_AsUTF8(item);
            strs.push_back(str);
            ids.push_back(const_cast<char*>(strs.back().c_str()));
        }

        int result = obj->api->SubscribeForQuoteRsp(ids.data(), static_cast<int>(count));
        return PyLong_FromLong(result);
    }

    PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
    return NULL;
}

/**
 * @brief 取消订阅询价（支持 NumPy 零拷贝）
 * Python命名: un_subscribe_for_quote_rsp
 */
static PyObject* MdApi_un_subscribe_for_quote_rsp(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // NumPy 零拷贝路径
    if (is_numpy_array(instrument_ids)) {
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(instrument_ids);

        if (!is_string_array(array)) {
            PyErr_SetString(PyExc_TypeError, "NumPy array must be string dtype (U or S)");
            return NULL;
        }
        if (!is_1d_array(array)) {
            PyErr_SetString(PyExc_ValueError, "NumPy array must be 1-dimensional");
            return NULL;
        }

        char* data = static_cast<char*>(PyArray_DATA(array));
        npy_intp size = PyArray_SIZE(array);
        npy_intp stride = PyArray_STRIDE(array, 0);

        std::vector<char*> ids;
        ids.reserve(size);
        for (npy_intp i = 0; i < size; ++i) {
            ids.push_back(data + i * stride);
        }

        int result = obj->api->UnSubscribeForQuoteRsp(ids.data(), static_cast<int>(size));
        return PyLong_FromLong(result);
    }

    // List 回退路径
    if (PyList_Check(instrument_ids)) {
        Py_ssize_t count = PyList_Size(instrument_ids);
        std::vector<char*> ids;
        std::vector<std::string> strs;
        ids.reserve(count);
        strs.reserve(count);

        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* item = PyList_GetItem(instrument_ids, i);
            if (!PyUnicode_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "instrument_ids must be strings");
                return NULL;
            }
            const char* str = PyUnicode_AsUTF8(item);
            strs.push_back(str);
            ids.push_back(const_cast<char*>(strs.back().c_str()));
        }

        int result = obj->api->UnSubscribeForQuoteRsp(ids.data(), static_cast<int>(count));
        return PyLong_FromLong(result);
    }

    PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
    return NULL;
}

/**
 * @brief 注册用户信息
 * Python命名: register_fens_user_info
 */
static PyObject* MdApi_register_fens_user_info(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* fens_user_info_dict;
    if (!PyArg_ParseTuple(args, "O", &fens_user_info_dict)) {
        return NULL;
    }

    if (!PyDict_Check(fens_user_info_dict)) {
        PyErr_SetString(PyExc_TypeError, "fens_user_info must be a dict");
        return NULL;
    }

    CThostFtdcFensUserInfoField fens_user_info = {};

    PyObject* value;
    if ((value = PyDict_GetItemString(fens_user_info_dict, "broker_id")) && PyUnicode_Check(value)) {
        strncpy(fens_user_info.BrokerID, PyUnicode_AsUTF8(value), sizeof(fens_user_info.BrokerID) - 1);
    }
    if ((value = PyDict_GetItemString(fens_user_info_dict, "user_id")) && PyUnicode_Check(value)) {
        strncpy(fens_user_info.UserID, PyUnicode_AsUTF8(value), sizeof(fens_user_info.UserID) - 1);
    }

    obj->api->RegisterFensUserInfo(&fens_user_info);

    Py_RETURN_NONE;
}

/**
 * @brief 查询组播合约请求
 * Python命名: req_qry_multicast_instrument
 */
static PyObject* MdApi_req_qry_multicast_instrument(PyObject* self, PyObject* args, PyObject* kwargs) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* req_dict;
    int request_id;

    static char* kwlist[] = {"qry_multicast_instrument", "request_id", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &req_dict, &request_id)) {
        return NULL;
    }

    if (!PyDict_Check(req_dict)) {
        PyErr_SetString(PyExc_TypeError, "qry_multicast_instrument must be a dict");
        return NULL;
    }

    CThostFtdcQryMulticastInstrumentField req = {};

    PyObject* value;
    if ((value = PyDict_GetItemString(req_dict, "topic_id")) && PyLong_Check(value)) {
        req.TopicID = static_cast<int>(PyLong_AsLong(value));
    }
    if ((value = PyDict_GetItemString(req_dict, "instrument_id")) && PyUnicode_Check(value)) {
        strncpy(req.InstrumentID, PyUnicode_AsUTF8(value), sizeof(req.InstrumentID) - 1);
    }

    int result = obj->api->ReqQryMulticastInstrument(&req, request_id);
    return PyLong_FromLong(result);
}

/**
 * @brief 用户登录请求
 * Python命名: req_user_login
 */
static PyObject* MdApi_req_user_login(PyObject* self, PyObject* args, PyObject* kwargs) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* req_dict;
    int request_id;

    static char* kwlist[] = {"req_user_login", "request_id", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &req_dict, &request_id)) {
        return NULL;
    }

    if (!PyDict_Check(req_dict)) {
        PyErr_SetString(PyExc_TypeError, "req_user_login must be a dict");
        return NULL;
    }

    CThostFtdcReqUserLoginField req = {};

    PyObject* value;
    if ((value = PyDict_GetItemString(req_dict, "broker_id")) && PyUnicode_Check(value)) {
        strncpy(req.BrokerID, PyUnicode_AsUTF8(value), sizeof(req.BrokerID) - 1);
    }
    if ((value = PyDict_GetItemString(req_dict, "user_id")) && PyUnicode_Check(value)) {
        strncpy(req.UserID, PyUnicode_AsUTF8(value), sizeof(req.UserID) - 1);
    }
    if ((value = PyDict_GetItemString(req_dict, "password")) && PyUnicode_Check(value)) {
        strncpy(req.Password, PyUnicode_AsUTF8(value), sizeof(req.Password) - 1);
    }

    int result = obj->api->ReqUserLogin(&req, request_id);
    return PyLong_FromLong(result);
}

/**
 * @brief 用户登出请求
 * Python命名: req_user_logout
 */
static PyObject* MdApi_req_user_logout(PyObject* self, PyObject* args, PyObject* kwargs) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* req_dict;
    int request_id;

    static char* kwlist[] = {"user_logout", "request_id", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &req_dict, &request_id)) {
        return NULL;
    }

    if (!PyDict_Check(req_dict)) {
        PyErr_SetString(PyExc_TypeError, "user_logout must be a dict");
        return NULL;
    }

    CThostFtdcUserLogoutField req = {};

    PyObject* value;
    if ((value = PyDict_GetItemString(req_dict, "broker_id")) && PyUnicode_Check(value)) {
        strncpy(req.BrokerID, PyUnicode_AsUTF8(value), sizeof(req.BrokerID) - 1);
    }
    if ((value = PyDict_GetItemString(req_dict, "user_id")) && PyUnicode_Check(value)) {
        strncpy(req.UserID, PyUnicode_AsUTF8(value), sizeof(req.UserID) - 1);
    }

    int result = obj->api->ReqUserLogout(&req, request_id);
    return PyLong_FromLong(result);
}

/**
 * @brief 释放资源
 * Python命名: release
 */
static PyObject* MdApi_release(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;

    if (obj->api) {
        obj->api->RegisterSpi(NULL);
        obj->api->Release();
        obj->api = NULL;
    }

    if (obj->spi) {
        delete dynamic_cast<PyMdSpi*>(obj->spi);
        obj->spi = NULL;
    }

    Py_XDECREF(obj->py_spi);
    obj->py_spi = NULL;

    Py_RETURN_NONE;
}

/**
 * @brief tp_dealloc函数
 */
static void MdApi_dealloc(PyObject* self) {
    MdApiObject* obj = (MdApiObject*)self;

    if (!Py_IsInitialized()) {
        if (obj->spi) {
            delete dynamic_cast<PyMdSpi*>(obj->spi);
            obj->spi = NULL;
        }
        if (obj->api) {
            obj->api->RegisterSpi(NULL);
            obj->api->Release();
            obj->api = NULL;
        }
        Py_TYPE(self)->tp_free(self);
        return;
    }

    #if PY_VERSION_HEX >= 0x030D0000
    if (Py_IsFinalizing()) {
        if (obj->spi) {
            delete dynamic_cast<PyMdSpi*>(obj->spi);
            obj->spi = NULL;
        }
        if (obj->api) {
            obj->api->RegisterSpi(NULL);
            obj->api->Release();
            obj->api = NULL;
        }
        Py_TYPE(self)->tp_free(self);
        return;
    }
    #endif

    PyObject* result = MdApi_release(self, NULL);
    Py_XDECREF(result);
    Py_TYPE(self)->tp_free(self);
}

/**
 * @brief tp_new函数
 */
static PyObject* MdApi_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    MdApiObject* self = (MdApiObject*)type->tp_alloc(type, 0);
    if (self) {
        self->api = nullptr;
        self->spi = nullptr;
        self->py_spi = nullptr;
    }
    return (PyObject*)self;
}

/**
 * @brief tp_init函数
 */
static int MdApi_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyErr_SetString(PyExc_TypeError, "Cannot create MdApi directly, use MdApi.create_ftdc_md_api() instead");
    return -1;
}

// =============================================================================
// Python类型定义
// =============================================================================

static PyMethodDef MdApi_methods[] = {
    {"create_ftdc_md_api", (PyCFunction)MdApi_create_ftdc_md_api, METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     "Create CTP MdApi instance"},
    {"get_api_version", MdApi_get_api_version, METH_NOARGS | METH_STATIC,
     "Get CTP API version"},
    {"init", MdApi_init, METH_NOARGS, "Initialize MdApi"},
    {"join", MdApi_join, METH_NOARGS, "Join thread"},
    {"get_trading_day", MdApi_get_trading_day, METH_NOARGS, "Get trading day"},
    {"register_front", MdApi_register_front, METH_VARARGS, "Register front address"},
    {"register_name_server", MdApi_register_name_server, METH_VARARGS, "Register name server"},
    {"register_spi", MdApi_register_spi, METH_VARARGS, "Register SPI callback"},
    {"register_fens_user_info", MdApi_register_fens_user_info, METH_VARARGS, "Register fens user info"},
    {"subscribe_market_data", MdApi_subscribe_market_data, METH_VARARGS, "Subscribe market data"},
    {"un_subscribe_market_data", MdApi_un_subscribe_market_data, METH_VARARGS, "Unsubscribe market data"},
    {"subscribe_for_quote_rsp", MdApi_subscribe_for_quote_rsp, METH_VARARGS, "Subscribe for quote response"},
    {"un_subscribe_for_quote_rsp", MdApi_un_subscribe_for_quote_rsp, METH_VARARGS, "Unsubscribe for quote response"},
    {"req_user_login", (PyCFunction)MdApi_req_user_login, METH_VARARGS | METH_KEYWORDS, "Request user login"},
    {"req_user_logout", (PyCFunction)MdApi_req_user_logout, METH_VARARGS | METH_KEYWORDS, "Request user logout"},
    {"req_qry_multicast_instrument", (PyCFunction)MdApi_req_qry_multicast_instrument, METH_VARARGS | METH_KEYWORDS, "Request query multicast instrument"},
    {"release", MdApi_release, METH_NOARGS, "Release resources"},
    {NULL}
};

static PyTypeObject MdApiType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "PcCTP.MdApi",                          /* tp_name */
    sizeof(MdApiObject),                    /* tp_basicsize */
    0,                                      /* tp_itemsize */
    (destructor)MdApi_dealloc,              /* tp_dealloc */
    0,                                      /* tp_print */
    0,                                      /* tp_getattr */
    0,                                      /* tp_setattr */
    0,                                      /* tp_reserved */
    0,                                      /* tp_repr */
    0,                                      /* tp_as_number */
    0,                                      /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash */
    0,                                      /* tp_call */
    0,                                      /* tp_str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "CTP PC Market Data API (Optimized with Force Inline)",  /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    MdApi_methods,                          /* tp_methods */
    0,                                      /* tp_members */
    0,                                      /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    0,                                      /* tp_descr_get */
    0,                                      /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)MdApi_init,                   /* tp_init */
    0,                                      /* tp_alloc */
    MdApi_new,                              /* tp_new */
};

// =============================================================================
// 模块级方法
// =============================================================================

static PyObject* cleanup_temporal_pools(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    GlobalStringPools::cleanup_temporal_pools();
    Py_RETURN_NONE;
}

static PyObject* cleanup_instruments(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    GlobalStringPools::cleanup_instruments();
    Py_RETURN_NONE;
}

static PyObject* check_instrument_pool_size(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    size_t size = GlobalStringPools::check_instrument_pool_size();
    return PyLong_FromSize_t(size);
}

static PyObject* get_pool_sizes(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;

    size_t exchanges, dates, times, instruments, users, brokers;
    GlobalStringPools::get_pool_sizes(exchanges, dates, times, instruments, users, brokers);

    PyObject* result = PyDict_New();
    PyDict_SetItemString(result, "exchanges", PyLong_FromSize_t(exchanges));
    PyDict_SetItemString(result, "dates", PyLong_FromSize_t(dates));
    PyDict_SetItemString(result, "times", PyLong_FromSize_t(times));
    PyDict_SetItemString(result, "instruments", PyLong_FromSize_t(instruments));
    PyDict_SetItemString(result, "users", PyLong_FromSize_t(users));
    PyDict_SetItemString(result, "brokers", PyLong_FromSize_t(brokers));

    return result;
}

static PyMethodDef pcctp_module_methods[] = {
    {"cleanup_temporal_pools", cleanup_temporal_pools, METH_NOARGS,
     "清理日期和时间字符串池（交易日收盘后调用）"},
    {"cleanup_instruments", cleanup_instruments, METH_NOARGS,
     "清理合约代码字符串池（切换交易日或重新订阅时调用）"},
    {"check_instrument_pool_size", check_instrument_pool_size, METH_NOARGS,
     "检查合约池大小并返回当前值（超过 950 时自动警告）"},
    {"get_pool_sizes", get_pool_sizes, METH_NOARGS,
     "获取所有字符串池的大小统计"},
    {NULL}
};

static PyModuleDef PcCTP_module = {
    PyModuleDef_HEAD_INIT,
    "PcCTP",
    "CTP PC Market Data API Bindings (Pure Python C API + Force Inline Optimization)",
    -1,
    pcctp_module_methods,
};

PyMODINIT_FUNC PyInit_PcCTP(void) {
    if (initialize_utils() < 0) {
        return NULL;
    }

    PyObject* module = PyModule_Create(&PcCTP_module);
    if (!module) {
        return NULL;
    }

    if (PyType_Ready(&MdApiType) < 0) {
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&MdApiType);
    if (PyModule_AddObject(module, "MdApi", (PyObject*)&MdApiType) < 0) {
        Py_DECREF(&MdApiType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
