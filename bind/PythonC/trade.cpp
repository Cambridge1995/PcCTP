/**
 * PcCTP - CTP PC版交易API Python绑定 (纯Python C API + 零拷贝优化 + gil优化 + 强内联版本)
 *
 * 核心特性：
 * - 使用 tp_dealloc 控制析构
 * - 使用 PyGILState_Ensure/Release 管理 GIL
 * - 回调函数使用下划线命名 (如 on_front_connected)
 * - 与 PyMdApi 相同的资源管理机制
 *
 * 性能优化：
 * - 字符串池零拷贝复用（交易所代码、日期、时间、合约代码等）
 * - 数值类型直接传递（零拷贝）
 * - 优化字符串构造（动态字符串）
 *
 * 内联优化：
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
#define PY_ARRAY_UNIQUE_SYMBOL PcCTP_CAPI_TRADE_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

// 通用工具库（包含平台特定宏、编译期常量、字典设置函数等）
#include "utils.h"

// PC版 CTP 交易 API 头文件
#if defined(_WIN64)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win64/ThostFtdcTraderApi.h"
    #pragma warning(pop)
#elif defined(_WIN32)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win32/ThostFtdcTraderApi.h"
    #pragma warning(pop)
#else
    #include "ctp/PC/linux/ThostFtdcTraderApi.h"
#endif

// =============================================================================
// 定义全局字符串池实例
// =============================================================================

DEFINE_STRING_POOLS()

// =============================================================================
// Python对象结构定义
// =============================================================================

/**
 * @brief TradeApi Python对象结构
 *
 * 对应 CTP 的 CThostFtdcTraderApi
 */
typedef struct {
    PyObject_HEAD                // Python对象头
    CThostFtdcTraderApi* api;    // CTP交易API指针
    CThostFtdcTraderSpi* spi;    // CTP回调SPI指针
    PyObject* py_spi;            // Python回调对象
} TradeApiObject;

// =============================================================================
// SPI回调类实现 (使用下划线命名法 + 内联优化)
// =============================================================================

/**
 * @brief CTP 交易 SPI回调类 (C++实现，内联优化)
 *
 * 将CTP的交易回调转发到Python对象，使用下划线命名法
 * 关键路径代码使用强制内联优化
 */
class PyTradeSpi : public CThostFtdcTraderSpi {
private:
    TradeApiObject* m_api;      // 持有 TradeApi 对象指针

public:
    FORCE_INLINE PyTradeSpi(TradeApiObject* api) : m_api(api) {}

    virtual ~PyTradeSpi() {
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

    // //////////////////////////////////////////////////////////////////////////
    // 连接相关回调 (使用下划线命名 + 内联优化)
    // //////////////////////////////////////////////////////////////////////////

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

    // //////////////////////////////////////////////////////////////////////////
    // 认证登录相关回调 (使用下划线命名 + 内联优化)
    // //////////////////////////////////////////////////////////////////////////

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

    /**
     * @brief 客户端认证响应
     */
    FORCE_INLINE_MEMBER void OnRspAuthenticate(CThostFtdcRspAuthenticateField* pRspAuthenticateField,
                                              CThostFtdcRspInfoField* pRspInfo,
                                              int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rsp_authenticate = Py_None;
        Py_INCREF(Py_None);
        if (pRspAuthenticateField) {
            py_rsp_authenticate = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_rsp_authenticate, "broker_id",
                pRspAuthenticateField->BrokerID, sizeof(pRspAuthenticateField->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_rsp_authenticate, "user_id",
                pRspAuthenticateField->UserID, sizeof(pRspAuthenticateField->UserID), GlobalStringPools::Users);

            // 动态字符串
            dict_set_string(py_rsp_authenticate, "user_product_info",
                pRspAuthenticateField->UserProductInfo, sizeof(pRspAuthenticateField->UserProductInfo));
            dict_set_string(py_rsp_authenticate, "app_id",
                pRspAuthenticateField->AppID, sizeof(pRspAuthenticateField->AppID));
            dict_set_string(py_rsp_authenticate, "app_type",
                pRspAuthenticateField->AppType, sizeof(pRspAuthenticateField->AppType));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_authenticate", "(OOii)",
            py_rsp_authenticate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rsp_authenticate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 登录请求响应
     */
    FORCE_INLINE_MEMBER void OnRspUserLogin(CThostFtdcRspUserLoginField* pRspUserLogin,
                                           CThostFtdcRspInfoField* pRspInfo,
                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rsp_user_login = Py_None;
        Py_INCREF(Py_None);
        if (pRspUserLogin) {
            py_rsp_user_login = PyDict_New();

            // 使用字符串池优化
            dict_set_pooled_string(py_rsp_user_login, "trading_day",
                pRspUserLogin->TradingDay, sizeof(pRspUserLogin->TradingDay), GlobalStringPools::Dates);
            dict_set_pooled_string(py_rsp_user_login, "login_time",
                pRspUserLogin->LoginTime, sizeof(pRspUserLogin->LoginTime), GlobalStringPools::Times);
            dict_set_pooled_string(py_rsp_user_login, "broker_id",
                pRspUserLogin->BrokerID, sizeof(pRspUserLogin->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_rsp_user_login, "user_id",
                pRspUserLogin->UserID, sizeof(pRspUserLogin->UserID), GlobalStringPools::UserIds);

            // 动态字符串
            dict_set_string(py_rsp_user_login, "system_name",
                pRspUserLogin->SystemName, sizeof(pRspUserLogin->SystemName));
            dict_set_string(py_rsp_user_login, "max_order_ref",
                pRspUserLogin->MaxOrderRef, sizeof(pRspUserLogin->MaxOrderRef));
            dict_set_string(py_rsp_user_login, "front_address",
                pRspUserLogin->FrontAddress, sizeof(pRspUserLogin->FrontAddress));

            // 各交易所时间
            dict_set_string(py_rsp_user_login, "shfe_time",
                pRspUserLogin->SHFETime, sizeof(pRspUserLogin->SHFETime));
            dict_set_string(py_rsp_user_login, "dce_time",
                pRspUserLogin->DCETime, sizeof(pRspUserLogin->DCETime));
            dict_set_string(py_rsp_user_login, "czce_time",
                pRspUserLogin->CZCETime, sizeof(pRspUserLogin->CZCETime));
            dict_set_string(py_rsp_user_login, "ffex_time",
                pRspUserLogin->FFEXTime, sizeof(pRspUserLogin->FFEXTime));
            dict_set_string(py_rsp_user_login, "ine_time",
                pRspUserLogin->INETime, sizeof(pRspUserLogin->INETime));
            dict_set_string(py_rsp_user_login, "gfex_time",
                pRspUserLogin->GFEXTime, sizeof(pRspUserLogin->GFEXTime));

            // 数值（零拷贝）
            dict_set_long(py_rsp_user_login, "front_id", pRspUserLogin->FrontID);
            dict_set_long(py_rsp_user_login, "session_id", pRspUserLogin->SessionID);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_user_login", "(OOii)",
            py_rsp_user_login, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rsp_user_login);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 登出请求响应
     */
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

    /**
     * @brief 用户口令更新请求响应
     */
    FORCE_INLINE_MEMBER void OnRspUserPasswordUpdate(CThostFtdcUserPasswordUpdateField* pUserPasswordUpdate,
                                                     CThostFtdcRspInfoField* pRspInfo,
                                                     int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_user_password_update = Py_None;
        Py_INCREF(Py_None);
        if (pUserPasswordUpdate) {
            py_user_password_update = PyDict_New();
            dict_set_pooled_string(py_user_password_update, "user_id",
                pUserPasswordUpdate->UserID, sizeof(pUserPasswordUpdate->UserID), GlobalStringPools::UserIds);
            dict_set_string(py_user_password_update, "old_password",
                pUserPasswordUpdate->OldPassword, sizeof(pUserPasswordUpdate->OldPassword));
            dict_set_string(py_user_password_update, "new_password",
                pUserPasswordUpdate->NewPassword, sizeof(pUserPasswordUpdate->NewPassword));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_user_password_update", "(OOii)",
            py_user_password_update, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_user_password_update);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 资金账户口令更新请求响应
     */
    FORCE_INLINE_MEMBER void OnRspTradingAccountPasswordUpdate(CThostFtdcTradingAccountPasswordUpdateField* pTradingAccountPasswordUpdate,
                                                              CThostFtdcRspInfoField* pRspInfo,
                                                              int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_trading_account_password_update = Py_None;
        Py_INCREF(Py_None);
        if (pTradingAccountPasswordUpdate) {
            py_trading_account_password_update = PyDict_New();
            dict_set_pooled_string(py_trading_account_password_update, "account_id",
                pTradingAccountPasswordUpdate->AccountID, sizeof(pTradingAccountPasswordUpdate->AccountID), GlobalStringPools::UserIds);
            dict_set_string(py_trading_account_password_update, "old_password",
                pTradingAccountPasswordUpdate->OldPassword, sizeof(pTradingAccountPasswordUpdate->OldPassword));
            dict_set_string(py_trading_account_password_update, "new_password",
                pTradingAccountPasswordUpdate->NewPassword, sizeof(pTradingAccountPasswordUpdate->NewPassword));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_trading_account_password_update", "(OOii)",
            py_trading_account_password_update, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_trading_account_password_update);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // //////////////////////////////////////////////////////////////////////////
    // 报单相关回调 (使用下划线命名 + 内联优化)
    // //////////////////////////////////////////////////////////////////////////

    /**
     * @brief 报单录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspOrderInsert(CThostFtdcInputOrderField* pInputOrder,
                                              CThostFtdcRspInfoField* pRspInfo,
                                              int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_input_order = Py_None;
        Py_INCREF(Py_None);
        if (pInputOrder) {
            py_input_order = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_input_order, "broker_id",
                pInputOrder->BrokerID, sizeof(pInputOrder->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_input_order, "investor_id",
                pInputOrder->InvestorID, sizeof(pInputOrder->InvestorID), GlobalStringPools::UserIds);
            dict_set_pooled_string(py_input_order, "user_id",
                pInputOrder->UserID, sizeof(pInputOrder->UserID), GlobalStringPools::UserIds);
            dict_set_pooled_string(py_input_order, "instrument_id",
                pInputOrder->InstrumentID, sizeof(pInputOrder->InstrumentID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_input_order, "exchange_id",
                pInputOrder->ExchangeID, sizeof(pInputOrder->ExchangeID), GlobalStringPools::ExchangeCodes);

            // 动态字符串
            dict_set_string(py_input_order, "order_ref",
                pInputOrder->OrderRef, sizeof(pInputOrder->OrderRef));
            dict_set_string(py_input_order, "business_unit",
                pInputOrder->BusinessUnit, sizeof(pInputOrder->BusinessUnit));
            dict_set_string(py_input_order, "invest_unit_id",
                pInputOrder->InvestUnitID, sizeof(pInputOrder->InvestUnitID));
            dict_set_string(py_input_order, "account_id",
                pInputOrder->AccountID, sizeof(pInputOrder->AccountID));
            dict_set_string(py_input_order, "currency_id",
                pInputOrder->CurrencyID, sizeof(pInputOrder->CurrencyID));
            dict_set_string(py_input_order, "client_id",
                pInputOrder->ClientID, sizeof(pInputOrder->ClientID));
            dict_set_string(py_input_order, "ip_address",
                pInputOrder->IPAddress, sizeof(pInputOrder->IPAddress));
            dict_set_string(py_input_order, "mac_address",
                pInputOrder->MacAddress, sizeof(pInputOrder->MacAddress));
            dict_set_string(py_input_order, "order_memo",
                pInputOrder->OrderMemo, sizeof(pInputOrder->OrderMemo));

            // 数值字段
            dict_set_long(py_input_order, "order_price_type", pInputOrder->OrderPriceType);
            dict_set_long(py_input_order, "direction", pInputOrder->Direction);
            dict_set_string(py_input_order, "comb_offset_flag",
                pInputOrder->CombOffsetFlag, sizeof(pInputOrder->CombOffsetFlag));
            dict_set_string(py_input_order, "comb_hedge_flag",
                pInputOrder->CombHedgeFlag, sizeof(pInputOrder->CombHedgeFlag));
            dict_set_double(py_input_order, "limit_price", pInputOrder->LimitPrice);
            dict_set_long(py_input_order, "volume_total_original", pInputOrder->VolumeTotalOriginal);
            dict_set_long(py_input_order, "time_condition", pInputOrder->TimeCondition);
            dict_set_string(py_input_order, "gtd_date",
                pInputOrder->GTDDate, sizeof(pInputOrder->GTDDate));
            dict_set_long(py_input_order, "volume_condition", pInputOrder->VolumeCondition);
            dict_set_long(py_input_order, "min_volume", pInputOrder->MinVolume);
            dict_set_long(py_input_order, "contingent_condition", pInputOrder->ContingentCondition);
            dict_set_double(py_input_order, "stop_price", pInputOrder->StopPrice);
            dict_set_long(py_input_order, "force_close_reason", pInputOrder->ForceCloseReason);
            dict_set_long(py_input_order, "is_auto_suspend", pInputOrder->IsAutoSuspend);
            dict_set_long(py_input_order, "user_force_close", pInputOrder->UserForceClose);
            dict_set_long(py_input_order, "is_swap_order", pInputOrder->IsSwapOrder);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_order_insert", "(OOii)",
            py_input_order, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_input_order);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 预埋单录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspParkedOrderInsert(CThostFtdcParkedOrderField* pParkedOrder,
                                                    CThostFtdcRspInfoField* pRspInfo,
                                                    int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_parked_order = Py_None;
        Py_INCREF(Py_None);
        if (pParkedOrder) {
            py_parked_order = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_parked_order, "broker_id",
                pParkedOrder->BrokerID, sizeof(pParkedOrder->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_parked_order, "investor_id",
                pParkedOrder->InvestorID, sizeof(pParkedOrder->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_parked_order, "user_id",
                pParkedOrder->UserID, sizeof(pParkedOrder->UserID), GlobalStringPools::Users);
            dict_set_pooled_string(py_parked_order, "exchange_id",
                pParkedOrder->ExchangeID, sizeof(pParkedOrder->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_parked_order, "instrument_id",
                pParkedOrder->InstrumentID, sizeof(pParkedOrder->InstrumentID), GlobalStringPools::Instruments);

            // 动态字符串
            dict_set_string(py_parked_order, "order_ref",
                pParkedOrder->OrderRef, sizeof(pParkedOrder->OrderRef));
            dict_set_string(py_parked_order, "order_price_type",
                pParkedOrder->OrderPriceType, sizeof(pParkedOrder->OrderPriceType));
            dict_set_string(py_parked_order, "direction",
                pParkedOrder->Direction, sizeof(pParkedOrder->Direction));
            dict_set_string(py_parked_order, "comb_offset_flag",
                pParkedOrder->CombOffsetFlag, sizeof(pParkedOrder->CombOffsetFlag));
            dict_set_string(py_parked_order, "comb_hedge_flag",
                pParkedOrder->CombHedgeFlag, sizeof(pParkedOrder->CombHedgeFlag));
            dict_set_string(py_parked_order, "time_condition",
                pParkedOrder->TimeCondition, sizeof(pParkedOrder->TimeCondition));
            dict_set_pooled_string(py_parked_order, "gtd_date",
                pParkedOrder->GTDDate, sizeof(pParkedOrder->GTDDate), GlobalStringPools::Dates);
            dict_set_string(py_parked_order, "volume_condition",
                pParkedOrder->VolumeCondition, sizeof(pParkedOrder->VolumeCondition));
            dict_set_string(py_parked_order, "contingent_condition",
                pParkedOrder->ContingentCondition, sizeof(pParkedOrder->ContingentCondition));
            dict_set_string(py_parked_order, "force_close_reason",
                pParkedOrder->ForceCloseReason, sizeof(pParkedOrder->ForceCloseReason));
            dict_set_string(py_parked_order, "business_unit",
                pParkedOrder->BusinessUnit, sizeof(pParkedOrder->BusinessUnit));
            dict_set_string(py_parked_order, "parked_order_id",
                pParkedOrder->ParkedOrderID, sizeof(pParkedOrder->ParkedOrderID));
            dict_set_string(py_parked_order, "user_type",
                pParkedOrder->UserType, sizeof(pParkedOrder->UserType));
            dict_set_string(py_parked_order, "status",
                pParkedOrder->Status, sizeof(pParkedOrder->Status));
            dict_set_string(py_parked_order, "account_id",
                pParkedOrder->AccountID, sizeof(pParkedOrder->AccountID));
            dict_set_string(py_parked_order, "currency_id",
                pParkedOrder->CurrencyID, sizeof(pParkedOrder->CurrencyID));
            dict_set_string(py_parked_order, "client_id",
                pParkedOrder->ClientID, sizeof(pParkedOrder->ClientID));
            dict_set_string(py_parked_order, "invest_unit_id",
                pParkedOrder->InvestUnitID, sizeof(pParkedOrder->InvestUnitID));
            dict_set_string(py_parked_order, "mac_address",
                pParkedOrder->MacAddress, sizeof(pParkedOrder->MacAddress));
            dict_set_string(py_parked_order, "ip_address",
                pParkedOrder->IPAddress, sizeof(pParkedOrder->IPAddress));

            // 数值字段
            dict_set_double(py_parked_order, "limit_price", pParkedOrder->LimitPrice);
            dict_set_int(py_parked_order, "volume_total_original", pParkedOrder->VolumeTotalOriginal);
            dict_set_int(py_parked_order, "min_volume", pParkedOrder->MinVolume);
            dict_set_double(py_parked_order, "stop_price", pParkedOrder->StopPrice);
            dict_set_int(py_parked_order, "request_id", pParkedOrder->RequestID);
            dict_set_int(py_parked_order, "error_id", pParkedOrder->ErrorID);

            // 布尔字段
            PyDict_SetItemString(py_parked_order, "is_auto_suspend", PyBool_FromLong(pParkedOrder->IsAutoSuspend != 0));
            PyDict_SetItemString(py_parked_order, "user_force_close", PyBool_FromLong(pParkedOrder->UserForceClose != 0));
            PyDict_SetItemString(py_parked_order, "is_swap_order", PyBool_FromLong(pParkedOrder->IsSwapOrder != 0));

            // GBK编码的错误消息
            dict_set_gbk_string(py_parked_order, "error_msg", pParkedOrder->ErrorMsg, sizeof(pParkedOrder->ErrorMsg));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_parked_order_insert", "(OOii)",
            py_parked_order, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_parked_order);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 预埋撤单录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspParkedOrderAction(CThostFtdcParkedOrderActionField* pParkedOrderAction,
                                                    CThostFtdcRspInfoField* pRspInfo,
                                                    int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_parked_order_action = Py_None;
        Py_INCREF(Py_None);
        if (pParkedOrderAction) {
            py_parked_order_action = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_parked_order_action, "broker_id",
                pParkedOrderAction->BrokerID, sizeof(pParkedOrderAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_parked_order_action, "investor_id",
                pParkedOrderAction->InvestorID, sizeof(pParkedOrderAction->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_parked_order_action, "user_id",
                pParkedOrderAction->UserID, sizeof(pParkedOrderAction->UserID), GlobalStringPools::Users);
            dict_set_pooled_string(py_parked_order_action, "exchange_id",
                pParkedOrderAction->ExchangeID, sizeof(pParkedOrderAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_parked_order_action, "instrument_id",
                pParkedOrderAction->InstrumentID, sizeof(pParkedOrderAction->InstrumentID), GlobalStringPools::Instruments);

            // 动态字符串
            dict_set_string(py_parked_order_action, "order_action_ref",
                pParkedOrderAction->OrderActionRef, sizeof(pParkedOrderAction->OrderActionRef));
            dict_set_string(py_parked_order_action, "order_ref",
                pParkedOrderAction->OrderRef, sizeof(pParkedOrderAction->OrderRef));
            dict_set_string(py_parked_order_action, "order_sys_id",
                pParkedOrderAction->OrderSysID, sizeof(pParkedOrderAction->OrderSysID));
            dict_set_string(py_parked_order_action, "action_flag",
                pParkedOrderAction->ActionFlag, sizeof(pParkedOrderAction->ActionFlag));
            dict_set_string(py_parked_order_action, "parked_order_action_id",
                pParkedOrderAction->ParkedOrderActionID, sizeof(pParkedOrderAction->ParkedOrderActionID));
            dict_set_string(py_parked_order_action, "user_type",
                pParkedOrderAction->UserType, sizeof(pParkedOrderAction->UserType));
            dict_set_string(py_parked_order_action, "status",
                pParkedOrderAction->Status, sizeof(pParkedOrderAction->Status));
            dict_set_string(py_parked_order_action, "invest_unit_id",
                pParkedOrderAction->InvestUnitID, sizeof(pParkedOrderAction->InvestUnitID));
            dict_set_string(py_parked_order_action, "mac_address",
                pParkedOrderAction->MacAddress, sizeof(pParkedOrderAction->MacAddress));
            dict_set_string(py_parked_order_action, "ip_address",
                pParkedOrderAction->IPAddress, sizeof(pParkedOrderAction->IPAddress));

            // 数值字段
            dict_set_int(py_parked_order_action, "request_id", pParkedOrderAction->RequestID);
            dict_set_int(py_parked_order_action, "front_id", pParkedOrderAction->FrontID);
            dict_set_int(py_parked_order_action, "session_id", pParkedOrderAction->SessionID);
            dict_set_double(py_parked_order_action, "limit_price", pParkedOrderAction->LimitPrice);
            dict_set_int(py_parked_order_action, "volume_change", pParkedOrderAction->VolumeChange);
            dict_set_int(py_parked_order_action, "error_id", pParkedOrderAction->ErrorID);

            // GBK编码的错误消息
            dict_set_gbk_string(py_parked_order_action, "error_msg", pParkedOrderAction->ErrorMsg, sizeof(pParkedOrderAction->ErrorMsg));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_parked_order_action", "(OOii)",
            py_parked_order_action, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_parked_order_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报单操作请求响应
     */
    FORCE_INLINE_MEMBER void OnRspOrderAction(CThostFtdcInputOrderActionField* pInputOrderAction,
                                             CThostFtdcRspInfoField* pRspInfo,
                                             int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_input_order_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputOrderAction) {
            py_input_order_action = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_input_order_action, "broker_id",
                pInputOrderAction->BrokerID, sizeof(pInputOrderAction->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_input_order_action, "investor_id",
                pInputOrderAction->InvestorID, sizeof(pInputOrderAction->InvestorID), GlobalStringPools::UserIds);
            dict_set_pooled_string(py_input_order_action, "user_id",
                pInputOrderAction->UserID, sizeof(pInputOrderAction->UserID), GlobalStringPools::UserIds);
            dict_set_pooled_string(py_input_order_action, "instrument_id",
                pInputOrderAction->InstrumentID, sizeof(pInputOrderAction->InstrumentID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_input_order_action, "exchange_id",
                pInputOrderAction->ExchangeID, sizeof(pInputOrderAction->ExchangeID), GlobalStringPools::ExchangeCodes);

            // 动态字符串
            dict_set_string(py_input_order_action, "order_action_ref",
                pInputOrderAction->OrderActionRef, sizeof(pInputOrderAction->OrderActionRef));
            dict_set_string(py_input_order_action, "order_ref",
                pInputOrderAction->OrderRef, sizeof(pInputOrderAction->OrderRef));
            dict_set_string(py_input_order_action, "order_sys_id",
                pInputOrderAction->OrderSysID, sizeof(pInputOrderAction->OrderSysID));
            dict_set_string(py_input_order_action, "invest_unit_id",
                pInputOrderAction->InvestUnitID, sizeof(pInputOrderAction->InvestUnitID));
            dict_set_string(py_input_order_action, "ip_address",
                pInputOrderAction->IPAddress, sizeof(pInputOrderAction->IPAddress));
            dict_set_string(py_input_order_action, "mac_address",
                pInputOrderAction->MacAddress, sizeof(pInputOrderAction->MacAddress));
            dict_set_string(py_input_order_action, "order_memo",
                pInputOrderAction->OrderMemo, sizeof(pInputOrderAction->OrderMemo));

            // 数值字段
            dict_set_long(py_input_order_action, "front_id", pInputOrderAction->FrontID);
            dict_set_long(py_input_order_action, "session_id", pInputOrderAction->SessionID);
            dict_set_long(py_input_order_action, "action_flag", pInputOrderAction->ActionFlag);
            dict_set_double(py_input_order_action, "limit_price", pInputOrderAction->LimitPrice);
            dict_set_long(py_input_order_action, "volume_change", pInputOrderAction->VolumeChange);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_order_action", "(OOii)",
            py_input_order_action, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_input_order_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报单通知（高频回调，需要最强内联优化）
     */
    FORCE_INLINE_MEMBER void OnRtnOrder(CThostFtdcOrderField* pOrder) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        if (!pOrder) {
            PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_order", "(O)", Py_None);
            if (!result) PyErr_Print();
            Py_XDECREF(result);
            return;
        }

        PyObject* order_dict = PyDict_New();
        if (!order_dict) return;

        // 字符串池优化
        dict_set_pooled_string(order_dict, "broker_id",
            pOrder->BrokerID, sizeof(pOrder->BrokerID), GlobalStringPools::BrokerIds);
        dict_set_pooled_string(order_dict, "investor_id",
            pOrder->InvestorID, sizeof(pOrder->InvestorID), GlobalStringPools::UserIds);
        dict_set_pooled_string(order_dict, "user_id",
            pOrder->UserID, sizeof(pOrder->UserID), GlobalStringPools::UserIds);
        dict_set_pooled_string(order_dict, "instrument_id",
            pOrder->InstrumentID, sizeof(pOrder->InstrumentID), GlobalStringPools::Instruments);
        dict_set_pooled_string(order_dict, "exchange_id",
            pOrder->ExchangeID, sizeof(pOrder->ExchangeID), GlobalStringPools::ExchangeCodes);
        dict_set_pooled_string(order_dict, "trading_day",
            pOrder->TradingDay, sizeof(pOrder->TradingDay), GlobalStringPools::Dates);

        // 动态字符串
        dict_set_string(order_dict, "order_ref",
            pOrder->OrderRef, sizeof(pOrder->OrderRef));
        dict_set_string(order_dict, "order_local_id",
            pOrder->OrderLocalID, sizeof(pOrder->OrderLocalID));
        dict_set_string(order_dict, "participant_id",
            pOrder->ParticipantID, sizeof(pOrder->ParticipantID));
        dict_set_string(order_dict, "client_id",
            pOrder->ClientID, sizeof(pOrder->ClientID));
        dict_set_string(order_dict, "exchange_inst_id",
            pOrder->ExchangeInstID, sizeof(pOrder->ExchangeInstID));
        dict_set_string(order_dict, "trader_id",
            pOrder->TraderID, sizeof(pOrder->TraderID));
        dict_set_string(order_dict, "order_sys_id",
            pOrder->OrderSysID, sizeof(pOrder->OrderSysID));
        dict_set_string(order_dict, "insert_date",
            pOrder->InsertDate, sizeof(pOrder->InsertDate));
        dict_set_string(order_dict, "insert_time",
            pOrder->InsertTime, sizeof(pOrder->InsertTime));
        dict_set_string(order_dict, "active_time",
            pOrder->ActiveTime, sizeof(pOrder->ActiveTime));
        dict_set_string(order_dict, "suspend_time",
            pOrder->SuspendTime, sizeof(pOrder->SuspendTime));
        dict_set_string(order_dict, "update_time",
            pOrder->UpdateTime, sizeof(pOrder->UpdateTime));
        dict_set_string(order_dict, "cancel_time",
            pOrder->CancelTime, sizeof(pOrder->CancelTime));
        dict_set_string(order_dict, "active_trader_id",
            pOrder->ActiveTraderID, sizeof(pOrder->ActiveTraderID));
        dict_set_string(order_dict, "clearing_part_id",
            pOrder->ClearingPartID, sizeof(pOrder->ClearingPartID));
        dict_set_string(order_dict, "branch_id",
            pOrder->BranchID, sizeof(pOrder->BranchID));
        dict_set_string(order_dict, "invest_unit_id",
            pOrder->InvestUnitID, sizeof(pOrder->InvestUnitID));
        dict_set_string(order_dict, "account_id",
            pOrder->AccountID, sizeof(pOrder->AccountID));
        dict_set_string(order_dict, "currency_id",
            pOrder->CurrencyID, sizeof(pOrder->CurrencyID));
        dict_set_string(order_dict, "ip_address",
            pOrder->IPAddress, sizeof(pOrder->IPAddress));
        dict_set_string(order_dict, "mac_address",
            pOrder->MacAddress, sizeof(pOrder->MacAddress));
        dict_set_gbk_string(order_dict, "status_msg",
            pOrder->StatusMsg, sizeof(pOrder->StatusMsg));
        dict_set_string(order_dict, "order_memo",
            pOrder->OrderMemo, sizeof(pOrder->OrderMemo));

        // 数值字段
        dict_set_long(order_dict, "order_price_type", pOrder->OrderPriceType);
        dict_set_long(order_dict, "direction", pOrder->Direction);
        dict_set_string(order_dict, "comb_offset_flag",
            pOrder->CombOffsetFlag, sizeof(pOrder->CombOffsetFlag));
        dict_set_string(order_dict, "comb_hedge_flag",
            pOrder->CombHedgeFlag, sizeof(pOrder->CombHedgeFlag));
        dict_set_double(order_dict, "limit_price", pOrder->LimitPrice);
        dict_set_long(order_dict, "volume_total_original", pOrder->VolumeTotalOriginal);
        dict_set_long(order_dict, "time_condition", pOrder->TimeCondition);
        dict_set_string(order_dict, "gtd_date",
            pOrder->GTDDate, sizeof(pOrder->GTDDate));
        dict_set_long(order_dict, "volume_condition", pOrder->VolumeCondition);
        dict_set_long(order_dict, "min_volume", pOrder->MinVolume);
        dict_set_long(order_dict, "contingent_condition", pOrder->ContingentCondition);
        dict_set_double(order_dict, "stop_price", pOrder->StopPrice);
        dict_set_long(order_dict, "force_close_reason", pOrder->ForceCloseReason);
        dict_set_long(order_dict, "is_auto_suspend", pOrder->IsAutoSuspend);
        dict_set_long(order_dict, "user_force_close", pOrder->UserForceClose);
        dict_set_long(order_dict, "is_swap_order", pOrder->IsSwapOrder);
        dict_set_long(order_dict, "front_id", pOrder->FrontID);
        dict_set_long(order_dict, "session_id", pOrder->SessionID);
        dict_set_long(order_dict, "order_submit_status", pOrder->OrderSubmitStatus);
        dict_set_long(order_dict, "notify_sequence", pOrder->NotifySequence);
        dict_set_long(order_dict, "settlement_id", pOrder->SettlementID);
        dict_set_long(order_dict, "order_source", pOrder->OrderSource);
        dict_set_long(order_dict, "order_status", pOrder->OrderStatus);
        dict_set_long(order_dict, "order_type", pOrder->OrderType);
        dict_set_long(order_dict, "volume_traded", pOrder->VolumeTraded);
        dict_set_long(order_dict, "volume_total", pOrder->VolumeTotal);
        dict_set_long(order_dict, "sequence_no", pOrder->SequenceNo);
        dict_set_long(order_dict, "broker_order_seq", pOrder->BrokerOrderSeq);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_order", "(O)", order_dict);

        Py_XDECREF(order_dict);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 成交通知（高频回调，需要最强内联优化）
     */
    FORCE_INLINE_MEMBER void OnRtnTrade(CThostFtdcTradeField* pTrade) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        if (!pTrade) {
            PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_trade", "(O)", Py_None);
            if (!result) PyErr_Print();
            Py_XDECREF(result);
            return;
        }

        PyObject* trade_dict = PyDict_New();
        if (!trade_dict) return;

        // 字符串池优化
        dict_set_pooled_string(trade_dict, "broker_id",
            pTrade->BrokerID, sizeof(pTrade->BrokerID), GlobalStringPools::BrokerIds);
        dict_set_pooled_string(trade_dict, "investor_id",
            pTrade->InvestorID, sizeof(pTrade->InvestorID), GlobalStringPools::UserIds);
        dict_set_pooled_string(trade_dict, "instrument_id",
            pTrade->InstrumentID, sizeof(pTrade->InstrumentID), GlobalStringPools::Instruments);
        dict_set_pooled_string(trade_dict, "exchange_id",
            pTrade->ExchangeID, sizeof(pTrade->ExchangeID), GlobalStringPools::ExchangeCodes);
        dict_set_pooled_string(trade_dict, "trading_day",
            pTrade->TradingDay, sizeof(pTrade->TradingDay), GlobalStringPools::Dates);

        // 动态字符串
        dict_set_string(trade_dict, "order_ref",
            pTrade->OrderRef, sizeof(pTrade->OrderRef));
        dict_set_string(trade_dict, "order_local_id",
            pTrade->OrderLocalID, sizeof(pTrade->OrderLocalID));
        dict_set_string(trade_dict, "participant_id",
            pTrade->ParticipantID, sizeof(pTrade->ParticipantID));
        dict_set_string(trade_dict, "client_id",
            pTrade->ClientID, sizeof(pTrade->ClientID));
        dict_set_string(trade_dict, "exchange_inst_id",
            pTrade->ExchangeInstID, sizeof(pTrade->ExchangeInstID));
        dict_set_string(trade_dict, "trader_id",
            pTrade->TraderID, sizeof(pTrade->TraderID));
        dict_set_string(trade_dict, "order_sys_id",
            pTrade->OrderSysID, sizeof(pTrade->OrderSysID));
        dict_set_string(trade_dict, "trade_id",
            pTrade->TradeID, sizeof(pTrade->TradeID));
        dict_set_string(trade_dict, "trade_time",
            pTrade->TradeTime, sizeof(pTrade->TradeTime));
        dict_set_string(trade_dict, "branch_id",
            pTrade->BranchID, sizeof(pTrade->BranchID));
        dict_set_string(trade_dict, "invest_unit_id",
            pTrade->InvestUnitID, sizeof(pTrade->InvestUnitID));
        dict_set_string(trade_dict, "account_id",
            pTrade->AccountID, sizeof(pTrade->AccountID));
        dict_set_string(trade_dict, "currency_id",
            pTrade->CurrencyID, sizeof(pTrade->CurrencyID));
        dict_set_string(trade_dict, "ip_address",
            pTrade->IPAddress, sizeof(pTrade->IPAddress));
        dict_set_string(trade_dict, "mac_address",
            pTrade->MacAddress, sizeof(pTrade->MacAddress));

        // 数值字段
        dict_set_long(trade_dict, "direction", pTrade->Direction);
        dict_set_string(trade_dict, "offset_flag",
            pTrade->OffsetFlag, sizeof(pTrade->OffsetFlag));
        dict_set_string(trade_dict, "hedge_flag",
            pTrade->HedgeFlag, sizeof(pTrade->HedgeFlag));
        dict_set_double(trade_dict, "price", pTrade->Price);
        dict_set_long(trade_dict, "volume", pTrade->Volume);
        dict_set_long(trade_dict, "trade_date", pTrade->TradeDate);
        dict_set_long(trade_dict, "settlement_id", pTrade->SettlementID);
        dict_set_long(trade_dict, "broker_order_seq", pTrade->BrokerOrderSeq);
        dict_set_long(trade_dict, "trade_source", pTrade->TradeSource);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_trade", "(O)", trade_dict);

        Py_XDECREF(trade_dict);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报单录入错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnOrderInsert(CThostFtdcInputOrderField* pInputOrder,
                                                CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_input_order = Py_None;
        Py_INCREF(Py_None);
        if (pInputOrder) {
            py_input_order = PyDict_New();
            // 简化的字段映射，与 OnRspOrderInsert 类似
            dict_set_pooled_string(py_input_order, "broker_id",
                pInputOrder->BrokerID, sizeof(pInputOrder->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_input_order, "instrument_id",
                pInputOrder->InstrumentID, sizeof(pInputOrder->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_input_order, "order_ref",
                pInputOrder->OrderRef, sizeof(pInputOrder->OrderRef));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_order_insert", "(OO)",
            py_input_order, py_rsp_info);

        Py_XDECREF(py_input_order);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报单操作错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnOrderAction(CThostFtdcOrderActionField* pOrderAction,
                                                CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_order_action = Py_None;
        Py_INCREF(Py_None);
        if (pOrderAction) {
            py_order_action = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_order_action, "broker_id",
                pOrderAction->BrokerID, sizeof(pOrderAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_order_action, "investor_id",
                pOrderAction->InvestorID, sizeof(pOrderAction->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_order_action, "user_id",
                pOrderAction->UserID, sizeof(pOrderAction->UserID), GlobalStringPools::Users);
            dict_set_pooled_string(py_order_action, "exchange_id",
                pOrderAction->ExchangeID, sizeof(pOrderAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_order_action, "instrument_id",
                pOrderAction->InstrumentID, sizeof(pOrderAction->InstrumentID), GlobalStringPools::Instruments);

            // 动态字符串
            dict_set_string(py_order_action, "order_action_ref",
                pOrderAction->OrderActionRef, sizeof(pOrderAction->OrderActionRef));
            dict_set_string(py_order_action, "order_ref",
                pOrderAction->OrderRef, sizeof(pOrderAction->OrderRef));
            dict_set_string(py_order_action, "order_sys_id",
                pOrderAction->OrderSysID, sizeof(pOrderAction->OrderSysID));
            dict_set_string(py_order_action, "action_flag",
                pOrderAction->ActionFlag, sizeof(pOrderAction->ActionFlag));
            dict_set_pooled_string(py_order_action, "action_date",
                pOrderAction->ActionDate, sizeof(pOrderAction->ActionDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_order_action, "action_time",
                pOrderAction->ActionTime, sizeof(pOrderAction->ActionTime), GlobalStringPools::Times);
            dict_set_string(py_order_action, "trader_id",
                pOrderAction->TraderID, sizeof(pOrderAction->TraderID));
            dict_set_string(py_order_action, "order_local_id",
                pOrderAction->OrderLocalID, sizeof(pOrderAction->OrderLocalID));
            dict_set_string(py_order_action, "action_local_id",
                pOrderAction->ActionLocalID, sizeof(pOrderAction->ActionLocalID));
            dict_set_string(py_order_action, "participant_id",
                pOrderAction->ParticipantID, sizeof(pOrderAction->ParticipantID));
            dict_set_string(py_order_action, "client_id",
                pOrderAction->ClientID, sizeof(pOrderAction->ClientID));
            dict_set_string(py_order_action, "business_unit",
                pOrderAction->BusinessUnit, sizeof(pOrderAction->BusinessUnit));
            dict_set_string(py_order_action, "order_action_status",
                pOrderAction->OrderActionStatus, sizeof(pOrderAction->OrderActionStatus));
            dict_set_gbk_string(py_order_action, "status_msg",
                pOrderAction->StatusMsg, sizeof(pOrderAction->StatusMsg));
            dict_set_string(py_order_action, "branch_id",
                pOrderAction->BranchID, sizeof(pOrderAction->BranchID));
            dict_set_string(py_order_action, "invest_unit_id",
                pOrderAction->InvestUnitID, sizeof(pOrderAction->InvestUnitID));
            dict_set_string(py_order_action, "mac_address",
                pOrderAction->MacAddress, sizeof(pOrderAction->MacAddress));
            dict_set_string(py_order_action, "ip_address",
                pOrderAction->IPAddress, sizeof(pOrderAction->IPAddress));

            // 数值字段
            dict_set_int(py_order_action, "request_id", pOrderAction->RequestID);
            dict_set_int(py_order_action, "front_id", pOrderAction->FrontID);
            dict_set_int(py_order_action, "session_id", pOrderAction->SessionID);
            dict_set_double(py_order_action, "limit_price", pOrderAction->LimitPrice);
            dict_set_int(py_order_action, "volume_change", pOrderAction->VolumeChange);
            dict_set_int(py_order_action, "install_id", pOrderAction->InstallID);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_order_action", "(OO)",
            py_order_action, py_rsp_info);

        Py_XDECREF(py_order_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // //////////////////////////////////////////////////////////////////////////
    // 查询相关回调 (使用下划线命名 + 内联优化)
    // //////////////////////////////////////////////////////////////////////////

    /**
     * @brief 请求查询报单响应
     */
    FORCE_INLINE_MEMBER void OnRspQryOrder(CThostFtdcOrderField* pOrder,
                                           CThostFtdcRspInfoField* pRspInfo,
                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_order = Py_None;
        Py_INCREF(Py_None);
        if (pOrder) {
            py_order = PyDict_New();
            // 字段映射与 OnRtnOrder 类似
            dict_set_pooled_string(py_order, "broker_id",
                pOrder->BrokerID, sizeof(pOrder->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_order, "instrument_id",
                pOrder->InstrumentID, sizeof(pOrder->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_order, "order_sys_id",
                pOrder->OrderSysID, sizeof(pOrder->OrderSysID));
            dict_set_string(py_order, "order_status",
                pOrder->OrderStatus, sizeof(pOrder->OrderStatus));
            dict_set_long(py_order, "volume_traded", pOrder->VolumeTraded);
            dict_set_long(py_order, "volume_total", pOrder->VolumeTotal);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_order", "(OOii)",
            py_order, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_order);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询成交响应
     */
    FORCE_INLINE_MEMBER void OnRspQryTrade(CThostFtdcTradeField* pTrade,
                                           CThostFtdcRspInfoField* pRspInfo,
                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_trade = Py_None;
        Py_INCREF(Py_None);
        if (pTrade) {
            py_trade = PyDict_New();
            // 字段映射与 OnRtnTrade 类似
            dict_set_pooled_string(py_trade, "broker_id",
                pTrade->BrokerID, sizeof(pTrade->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_trade, "instrument_id",
                pTrade->InstrumentID, sizeof(pTrade->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_trade, "trade_id",
                pTrade->TradeID, sizeof(pTrade->TradeID));
            dict_set_string(py_trade, "trade_time",
                pTrade->TradeTime, sizeof(pTrade->TradeTime));
            dict_set_double(py_trade, "price", pTrade->Price);
            dict_set_long(py_trade, "volume", pTrade->Volume);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_trade", "(OOii)",
            py_trade, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_trade);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询投资者持仓响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInvestorPosition(CThostFtdcInvestorPositionField* pInvestorPosition,
                                                       CThostFtdcRspInfoField* pRspInfo,
                                                       int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_position = Py_None;
        Py_INCREF(Py_None);
        if (pInvestorPosition) {
            py_position = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_position, "broker_id",
                pInvestorPosition->BrokerID, sizeof(pInvestorPosition->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_position, "investor_id",
                pInvestorPosition->InvestorID, sizeof(pInvestorPosition->InvestorID), GlobalStringPools::UserIds);
            dict_set_pooled_string(py_position, "instrument_id",
                pInvestorPosition->InstrumentID, sizeof(pInvestorPosition->InstrumentID), GlobalStringPools::Instruments);

            // 动态字符串
            dict_set_string(py_position, "invest_unit_id",
                pInvestorPosition->InvestUnitID, sizeof(pInvestorPosition->InvestUnitId));
            dict_set_string(py_position, "trading_day",
                pInvestorPosition->TradingDay, sizeof(pInvestorPosition->TradingDay));
            dict_set_string(py_position, "open_date",
                pInvestorPosition->OpenDate, sizeof(pInvestorPosition->OpenDate));
            dict_set_string(py_position, "exchange_id",
                pInvestorPosition->ExchangeID, sizeof(pInvestorPosition->ExchangeID));

            // 数值字段
            dict_set_long(py_position, "position_date", pInvestorPosition->PositionDate);
            dict_set_long(py_position, "direction", pInvestorPosition->Direction);
            dict_set_string(py_position, "hedge_flag",
                pInvestorPosition->HedgeFlag, sizeof(pInvestorPosition->HedgeFlag));
            dict_set_long(py_position, "position", pInvestorPosition->Position);
            dict_set_long(py_position, "long frozen", pInvestorPosition->LongFrozen);
            dict_set_long(py_position, "short_frozen", pInvestorPosition->ShortFrozen);
            dict_set_long(py_position, "yd_position", pInvestorPosition->YdPosition);
            dict_set_long(py_position, "td_position", pInvestorPosition->TdPosition);
            dict_set_double(py_position, "open_price", pInvestorPosition->OpenPrice);
            dict_set_double(py_position, "settlement_price", pInvestorPosition->SettlementPrice);
            dict_set_double(py_position, "position_cost", pInvestorPosition->PositionCost);
            dict_set_double(py_position, "margin", pInvestorPosition->Margin);
            dict_set_long(py_position, "position_profit", pInvestorPosition->PositionProfit);
            dict_set_long(py_position, "pre_settlement_price", pInvestorPosition->PreSettlementPrice);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_investor_position", "(OOii)",
            py_position, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_position);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询资金账户响应
     */
    FORCE_INLINE_MEMBER void OnRspQryTradingAccount(CThostFtdcTradingAccountField* pTradingAccount,
                                                    CThostFtdcRspInfoField* pRspInfo,
                                                    int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_account = Py_None;
        Py_INCREF(Py_None);
        if (pTradingAccount) {
            py_account = PyDict_New();

            // 字符串池优化
            dict_set_pooled_string(py_account, "broker_id",
                pTradingAccount->BrokerID, sizeof(pTradingAccount->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_account, "account_id",
                pTradingAccount->AccountID, sizeof(pTradingAccount->AccountID), GlobalStringPools::UserIds);

            // 动态字符串
            dict_set_string(py_account, "trading_day",
                pTradingAccount->TradingDay, sizeof(pTradingAccount->TradingDay));
            dict_set_string(py_account, "currency_id",
                pTradingAccount->CurrencyID, sizeof(pTradingAccount->CurrencyID));

            // 数值字段
            dict_set_double(py_account, "pre_mortgage", pTradingAccount->PreMortgage);
            dict_set_double(py_account, "pre_credit", pTradingAccount->PreCredit);
            dict_set_double(py_account, "pre_deposit", pTradingAccount->PreDeposit);
            dict_set_double(py_account, "pre_balance", pTradingAccount->PreBalance);
            dict_set_double(py_account, "pre_margin", pTradingAccount->PreMargin);
            dict_set_double(py_account, "interest_base", pTradingAccount->InterestBase);
            dict_set_double(py_account, "interest", pTradingAccount->Interest);
            dict_set_double(py_account, "deposit", pTradingAccount->Deposit);
            dict_set_double(py_account, "withdraw", pTradingAccount->Withdraw);
            dict_set_double(py_account, "frozen_margin", pTradingAccount->FrozenMargin);
            dict_set_double(py_account, "frozen_cash", pTradingAccount->FrozenCash);
            dict_set_double(py_account, "frozen_commission", pTradingAccount->FrozenCommission);
            dict_set_double(py_account, "curr_margin", pTradingAccount->CurrMargin);
            dict_set_double(py_account, "cash_in", pTradingAccount->CashIn);
            dict_set_double(py_account, "commission", pTradingAccount->Commission);
            dict_set_double(py_account, "balance", pTradingAccount->Balance);
            dict_set_double(py_account, "available", pTradingAccount->Available);
            dict_set_double(py_account, "withdraw_quota", pTradingAccount->WithdrawQuota);
            dict_set_double(py_account, "trn_time", pTradingAccount->TrnTime);
            dict_set_long(py_account, "settlement_id", pTradingAccount->SettlementID);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_trading_account", "(OOii)",
            py_account, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_account);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询投资者响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInvestor(CThostFtdcInvestorField* pInvestor,
                                              CThostFtdcRspInfoField* pRspInfo,
                                              int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_investor = Py_None;
        Py_INCREF(Py_None);
        if (pInvestor) {
            py_investor = PyDict_New();
            dict_set_pooled_string(py_investor, "broker_id",
                pInvestor->BrokerID, sizeof(pInvestor->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_investor, "investor_id",
                pInvestor->InvestorID, sizeof(pInvestor->InvestorID), GlobalStringPools::UserIds);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_investor", "(OOii)",
            py_investor, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_investor);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询交易编码响应
     */
    FORCE_INLINE_MEMBER void OnRspQryTradingCode(CThostFtdcTradingCodeField* pTradingCode,
                                                 CThostFtdcRspInfoField* pRspInfo,
                                                 int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_trading_code = Py_None;
        Py_INCREF(Py_None);
        if (pTradingCode) {
            py_trading_code = PyDict_New();
            dict_set_pooled_string(py_trading_code, "broker_id",
                pTradingCode->BrokerID, sizeof(pTradingCode->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_trading_code, "investor_id",
                pTradingCode->InvestorID, sizeof(pTradingCode->InvestorID), GlobalStringPools::UserIds);
            dict_set_string(py_trading_code, "exchange_id",
                pTradingCode->ExchangeID, sizeof(pTradingCode->ExchangeID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_trading_code", "(OOii)",
            py_trading_code, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_trading_code);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询合约保证金率响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInstrumentMarginRate(CThostFtdcInstrumentMarginRateField* pInstrumentMarginRate,
                                                          CThostFtdcRspInfoField* pRspInfo,
                                                          int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_margin_rate = Py_None;
        Py_INCREF(Py_None);
        if (pInstrumentMarginRate) {
            py_margin_rate = PyDict_New();
            dict_set_pooled_string(py_margin_rate, "broker_id",
                pInstrumentMarginRate->BrokerID, sizeof(pInstrumentMarginRate->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_margin_rate, "instrument_id",
                pInstrumentMarginRate->InstrumentID, sizeof(pInstrumentMarginRate->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_margin_rate, "exchange_id",
                pInstrumentMarginRate->ExchangeID, sizeof(pInstrumentMarginRate->ExchangeID));
            dict_set_string(py_margin_rate, "hedge_flag",
                pInstrumentMarginRate->HedgeFlag, sizeof(pInstrumentMarginRate->HedgeFlag));
            dict_set_double(py_margin_rate, "long_margin_ratio_by_money", pInstrumentMarginRate->LongMarginRatioByMoney);
            dict_set_double(py_margin_rate, "long_margin_ratio_by_volume", pInstrumentMarginRate->LongMarginRatioByVolume);
            dict_set_double(py_margin_rate, "short_margin_ratio_by_money", pInstrumentMarginRate->ShortMarginRatioByMoney);
            dict_set_double(py_margin_rate, "short_margin_ratio_by_volume", pInstrumentMarginRate->ShortMarginRatioByVolume);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_instrument_margin_rate", "(OOii)",
            py_margin_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_margin_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询合约手续费率响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInstrumentCommissionRate(CThostFtdcInstrumentCommissionRateField* pInstrumentCommissionRate,
                                                              CThostFtdcRspInfoField* pRspInfo,
                                                              int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_commission_rate = Py_None;
        Py_INCREF(Py_None);
        if (pInstrumentCommissionRate) {
            py_commission_rate = PyDict_New();
            dict_set_pooled_string(py_commission_rate, "broker_id",
                pInstrumentCommissionRate->BrokerID, sizeof(pInstrumentCommissionRate->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_commission_rate, "instrument_id",
                pInstrumentCommissionRate->InstrumentID, sizeof(pInstrumentCommissionRate->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_commission_rate, "exchange_id",
                pInstrumentCommissionRate->ExchangeID, sizeof(pInstrumentCommissionRate->ExchangeID));
            dict_set_double(py_commission_rate, "open_ratio_by_money", pInstrumentCommissionRate->OpenRatioByMoney);
            dict_set_double(py_commission_rate, "open_ratio_by_volume", pInstrumentCommissionRate->OpenRatioByVolume);
            dict_set_double(py_commission_rate, "close_ratio_by_money", pInstrumentCommissionRate->CloseRatioByMoney);
            dict_set_double(py_commission_rate, "close_ratio_by_volume", pInstrumentCommissionRate->CloseRatioByVolume);
            dict_set_double(py_commission_rate, "close_today_ratio_by_money", pInstrumentCommissionRate->CloseTodayRatioByMoney);
            dict_set_double(py_commission_rate, "close_today_ratio_by_volume", pInstrumentCommissionRate->CloseTodayRatioByVolume);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_instrument_commission_rate", "(OOii)",
            py_commission_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_commission_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询合约响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInstrument(CThostFtdcInstrumentField* pInstrument,
                                               CThostFtdcRspInfoField* pRspInfo,
                                               int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_instrument = Py_None;
        Py_INCREF(Py_None);
        if (pInstrument) {
            py_instrument = PyDict_New();
            dict_set_pooled_string(py_instrument, "instrument_id",
                pInstrument->InstrumentID, sizeof(pInstrument->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_instrument, "exchange_id",
                pInstrument->ExchangeID, sizeof(pInstrument->ExchangeID));
            dict_set_string(py_instrument, "instrument_name",
                pInstrument->InstrumentName, sizeof(pInstrument->InstrumentName));
            dict_set_string(py_instrument, "product_class",
                pInstrument->ProductClass, sizeof(pInstrument->ProductClass));
            dict_set_long(py_instrument, "volume_multiple", pInstrument->VolumeMultiple);
            dict_set_double(py_instrument, "price_tick", pInstrument->PriceTick);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_instrument", "(OOii)",
            py_instrument, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_instrument);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询投资者结算结果响应
     */
    FORCE_INLINE_MEMBER void OnRspQrySettlementInfo(CThostFtdcSettlementInfoField* pSettlementInfo,
                                                    CThostFtdcRspInfoField* pRspInfo,
                                                    int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_settlement_info = Py_None;
        Py_INCREF(Py_None);
        if (pSettlementInfo) {
            py_settlement_info = PyDict_New();
            dict_set_pooled_string(py_settlement_info, "broker_id",
                pSettlementInfo->BrokerID, sizeof(pSettlementInfo->BrokerID), GlobalStringPools::BrokerIds);
            dict_set_pooled_string(py_settlement_info, "investor_id",
                pSettlementInfo->InvestorID, sizeof(pSettlementInfo->InvestorID), GlobalStringPools::UserIds);
            dict_set_string(py_settlement_info, "trading_day",
                pSettlementInfo->TradingDay, sizeof(pSettlementInfo->TradingDay));
            dict_set_gbk_string(py_settlement_info, "settlement_info",
                pSettlementInfo->SettlementInfo, sizeof(pSettlementInfo->SettlementInfo));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_settlement_info", "(OOii)",
            py_settlement_info, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_settlement_info);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // =============================================================================
    // 第1批回调方法 (20个高优先级方法)
    // =============================================================================

    /**
     * @brief 查询用户当前支持的认证模式的回复
     */
    FORCE_INLINE_MEMBER void OnRspUserAuthMethod(CThostFtdcRspUserAuthMethodField* pRspUserAuthMethod,
                                                  CThostFtdcRspInfoField* pRspInfo,
                                                  int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        // 直接返回字符串，不使用字典
        PyObject* py_auth_method = Py_None;
        Py_INCREF(Py_None);
        if (pRspUserAuthMethod) {
            py_auth_method = GlobalStringPools::ExchangeCodes.intern(
                pRspUserAuthMethod->UsableAuthMethod,
                sizeof(pRspUserAuthMethod->UsableAuthMethod));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_user_auth_method", "(OOii)",
            py_auth_method, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_auth_method);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 获取图形验证码请求的回复
     */
    FORCE_INLINE_MEMBER void OnRspGenUserCaptcha(CThostFtdcRspGenUserCaptchaField* pRspGenUserCaptcha,
                                                 CThostFtdcRspInfoField* pRspInfo,
                                                 int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_captcha = Py_None;
        Py_INCREF(Py_None);
        if (pRspGenUserCaptcha) {
            py_captcha = PyDict_New();
            dict_set_pooled_string(py_captcha, "broker_id",
                pRspGenUserCaptcha->BrokerID, sizeof(pRspGenUserCaptcha->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_captcha, "user_id",
                pRspGenUserCaptcha->UserID, sizeof(pRspGenUserCaptcha->UserID), GlobalStringPools::Users);
            dict_set_long(py_captcha, "captcha_info_len", pRspGenUserCaptcha->CaptchaInfoLen);
            // CaptchaInfo is binary data, use bytes
            if (pRspGenUserCaptcha->CaptchaInfoLen > 0) {
                dict_set_bytes(py_captcha, "captcha_info",
                    pRspGenUserCaptcha->CaptchaInfo, pRspGenUserCaptcha->CaptchaInfoLen);
            }
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_gen_user_captcha", "(OOii)",
            py_captcha, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_captcha);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 获取短信验证码请求的回复
     */
    FORCE_INLINE_MEMBER void OnRspGenUserText(CThostFtdcRspGenUserTextField* pRspGenUserText,
                                             CThostFtdcRspInfoField* pRspInfo,
                                             int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_text = Py_None;
        Py_INCREF(Py_None);
        if (pRspGenUserText) {
            py_text = PyDict_New();
            dict_set_long(py_text, "user_text_seq", pRspGenUserText->UserTextSeq);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_gen_user_text", "(OOii)",
            py_text, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_text);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 投资者结算结果确认响应
     */
    FORCE_INLINE_MEMBER void OnRspSettlementInfoConfirm(CThostFtdcSettlementInfoConfirmField* pSettlementInfoConfirm,
                                                       CThostFtdcRspInfoField* pRspInfo,
                                                       int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_confirm = Py_None;
        Py_INCREF(Py_None);
        if (pSettlementInfoConfirm) {
            py_confirm = PyDict_New();
            dict_set_pooled_string(py_confirm, "broker_id",
                pSettlementInfoConfirm->BrokerID, sizeof(pSettlementInfoConfirm->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_confirm, "investor_id",
                pSettlementInfoConfirm->InvestorID, sizeof(pSettlementInfoConfirm->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_confirm, "confirm_date",
                pSettlementInfoConfirm->ConfirmDate, sizeof(pSettlementInfoConfirm->ConfirmDate));
            dict_set_string(py_confirm, "confirm_time",
                pSettlementInfoConfirm->ConfirmTime, sizeof(pSettlementInfoConfirm->ConfirmTime));
            dict_set_long(py_confirm, "settlement_id", pSettlementInfoConfirm->SettlementID);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_settlement_info_confirm", "(OOii)",
            py_confirm, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_confirm);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询交易所响应
     */
    FORCE_INLINE_MEMBER void OnRspQryExchange(CThostFtdcExchangeField* pExchange,
                                             CThostFtdcRspInfoField* pRspInfo,
                                             int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_exchange = Py_None;
        Py_INCREF(Py_None);
        if (pExchange) {
            py_exchange = PyDict_New();
            dict_set_pooled_string(py_exchange, "exchange_id",
                pExchange->ExchangeID, sizeof(pExchange->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_exchange, "exchange_name",
                pExchange->ExchangeName, sizeof(pExchange->ExchangeName));
            dict_set_pooled_string(py_exchange, "exchange_property",
                pExchange->ExchangeProperty, sizeof(pExchange->ExchangeProperty), GlobalStringPools::ExchangeCodes);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_exchange", "(OOii)",
            py_exchange, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_exchange);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询产品响应
     */
    FORCE_INLINE_MEMBER void OnRspQryProduct(CThostFtdcProductField* pProduct,
                                            CThostFtdcRspInfoField* pRspInfo,
                                            int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_product = Py_None;
        Py_INCREF(Py_None);
        if (pProduct) {
            py_product = PyDict_New();
            dict_set_string(py_product, "product_name",
                pProduct->ProductName, sizeof(pProduct->ProductName));
            dict_set_pooled_string(py_product, "exchange_id",
                pProduct->ExchangeID, sizeof(pProduct->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_product, "product_class",
                pProduct->ProductClass, sizeof(pProduct->ProductClass), GlobalStringPools::ExchangeCodes);
            dict_set_long(py_product, "volume_multiple", pProduct->VolumeMultiple);
            dict_set_double(py_product, "price_tick", pProduct->PriceTick);
            dict_set_long(py_product, "max_market_order_volume", pProduct->MaxMarketOrderVolume);
            dict_set_long(py_product, "min_market_order_volume", pProduct->MinMarketOrderVolume);
            dict_set_long(py_product, "max_limit_order_volume", pProduct->MaxLimitOrderVolume);
            dict_set_long(py_product, "min_limit_order_volume", pProduct->MinLimitOrderVolume);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_product", "(OOii)",
            py_product, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_product);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询行情响应
     */
    FORCE_INLINE_MEMBER void OnRspQryDepthMarketData(CThostFtdcDepthMarketDataField* pDepthMarketData,
                                                     CThostFtdcRspInfoField* pRspInfo,
                                                     int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_data = Py_None;
        Py_INCREF(Py_None);
        if (pDepthMarketData) {
            py_data = PyDict_New();
            dict_set_pooled_string(py_data, "trading_day",
                pDepthMarketData->TradingDay, sizeof(pDepthMarketData->TradingDay), GlobalStringPools::Dates);
            dict_set_pooled_string(py_data, "instrument_id",
                pDepthMarketData->InstrumentID, sizeof(pDepthMarketData->InstrumentID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_data, "exchange_id",
                pDepthMarketData->ExchangeID, sizeof(pDepthMarketData->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_data, "exchange_inst_id",
                pDepthMarketData->ExchangeInstID, sizeof(pDepthMarketData->ExchangeInstID), GlobalStringPools::Instruments);
            dict_set_string(py_data, "update_time",
                pDepthMarketData->UpdateTime, sizeof(pDepthMarketData->UpdateTime));
            dict_set_pooled_string(py_data, "action_day",
                pDepthMarketData->ActionDay, sizeof(pDepthMarketData->ActionDay), GlobalStringPools::Dates);
            dict_set_double(py_data, "last_price", pDepthMarketData->LastPrice);
            dict_set_double(py_data, "pre_settlement_price", pDepthMarketData->PreSettlementPrice);
            dict_set_double(py_data, "pre_close_price", pDepthMarketData->PreClosePrice);
            dict_set_double(py_data, "pre_open_interest", pDepthMarketData->PreOpenInterest);
            dict_set_double(py_data, "open_price", pDepthMarketData->OpenPrice);
            dict_set_double(py_data, "highest_price", pDepthMarketData->HighestPrice);
            dict_set_double(py_data, "lowest_price", pDepthMarketData->LowestPrice);
            dict_set_double(py_data, "close_price", pDepthMarketData->ClosePrice);
            dict_set_double(py_data, "settlement_price", pDepthMarketData->SettlementPrice);
            dict_set_double(py_data, "upper_limit_price", pDepthMarketData->UpperLimitPrice);
            dict_set_double(py_data, "lower_limit_price", pDepthMarketData->LowerLimitPrice);
            dict_set_double(py_data, "pre_delta", pDepthMarketData->PreDelta);
            dict_set_double(py_data, "curr_delta", pDepthMarketData->CurrDelta);
            dict_set_double(py_data, "average_price", pDepthMarketData->AveragePrice);
            dict_set_long(py_data, "volume", pDepthMarketData->Volume);
            dict_set_double(py_data, "turnover", pDepthMarketData->Turnover);
            dict_set_double(py_data, "open_interest", pDepthMarketData->OpenInterest);
            dict_set_long(py_data, "update_millisec", pDepthMarketData->UpdateMillisec);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_depth_market_data", "(OOii)",
            py_data, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_data);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询转帐银行响应
     */
    FORCE_INLINE_MEMBER void OnRspQryTransferBank(CThostFtdcTransferBankField* pTransferBank,
                                                 CThostFtdcRspInfoField* pRspInfo,
                                                 int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_bank = Py_None;
        Py_INCREF(Py_None);
        if (pTransferBank) {
            py_bank = PyDict_New();
            dict_set_pooled_string(py_bank, "bank_id",
                pTransferBank->BankID, sizeof(pTransferBank->BankID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_bank, "bank_brch_id",
                pTransferBank->BankBrchID, sizeof(pTransferBank->BankBrchID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_bank, "bank_name",
                pTransferBank->BankName, sizeof(pTransferBank->BankName));
            dict_set_long(py_bank, "is_active", pTransferBank->IsActive);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_transfer_bank", "(OOii)",
            py_bank, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_bank);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询投资者持仓明细响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInvestorPositionDetail(CThostFtdcInvestorPositionDetailField* pInvestorPositionDetail,
                                                          CThostFtdcRspInfoField* pRspInfo,
                                                          int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_position = Py_None;
        Py_INCREF(Py_None);
        if (pInvestorPositionDetail) {
            py_position = PyDict_New();
            dict_set_pooled_string(py_position, "broker_id",
                pInvestorPositionDetail->BrokerID, sizeof(pInvestorPositionDetail->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_position, "investor_id",
                pInvestorPositionDetail->InvestorID, sizeof(pInvestorPositionDetail->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_position, "hedge_flag",
                pInvestorPositionDetail->HedgeFlag, sizeof(pInvestorPositionDetail->HedgeFlag), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_position, "direction",
                pInvestorPositionDetail->Direction, sizeof(pInvestorPositionDetail->Direction), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_position, "open_date",
                pInvestorPositionDetail->OpenDate, sizeof(pInvestorPositionDetail->OpenDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_position, "trade_id",
                pInvestorPositionDetail->TradeID, sizeof(pInvestorPositionDetail->TradeID), GlobalStringPools::ExchangeCodes);
            dict_set_long(py_position, "volume", pInvestorPositionDetail->Volume);
            dict_set_double(py_position, "open_price", pInvestorPositionDetail->OpenPrice);
            dict_set_pooled_string(py_position, "trading_day",
                pInvestorPositionDetail->TradingDay, sizeof(pInvestorPositionDetail->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_position, "settlement_id", pInvestorPositionDetail->SettlementID);
            dict_set_pooled_string(py_position, "instrument_id",
                pInvestorPositionDetail->InstrumentID, sizeof(pInvestorPositionDetail->InstrumentID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_position, "exchange_id",
                pInvestorPositionDetail->ExchangeID, sizeof(pInvestorPositionDetail->ExchangeID), GlobalStringPools::ExchangeCodes);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_investor_position_detail", "(OOii)",
            py_position, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_position);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询客户通知响应
     */
    FORCE_INLINE_MEMBER void OnRspQryNotice(CThostFtdcNoticeField* pNotice,
                                           CThostFtdcRspInfoField* pRspInfo,
                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_notice = Py_None;
        Py_INCREF(Py_None);
        if (pNotice) {
            py_notice = PyDict_New();
            dict_set_pooled_string(py_notice, "broker_id",
                pNotice->BrokerID, sizeof(pNotice->BrokerID), GlobalStringPools::Brokers);
            dict_set_string(py_notice, "content",
                pNotice->Content, sizeof(pNotice->Content));
            dict_set_pooled_string(py_notice, "sequence_label",
                pNotice->SequenceLabel, sizeof(pNotice->SequenceLabel), GlobalStringPools::ExchangeCodes);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_notice", "(OOii)",
            py_notice, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_notice);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询结算信息确认响应
     */
    FORCE_INLINE_MEMBER void OnRspQrySettlementInfoConfirm(CThostFtdcSettlementInfoConfirmField* pSettlementInfoConfirm,
                                                           CThostFtdcRspInfoField* pRspInfo,
                                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_confirm = Py_None;
        Py_INCREF(Py_None);
        if (pSettlementInfoConfirm) {
            py_confirm = PyDict_New();
            dict_set_pooled_string(py_confirm, "broker_id",
                pSettlementInfoConfirm->BrokerID, sizeof(pSettlementInfoConfirm->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_confirm, "investor_id",
                pSettlementInfoConfirm->InvestorID, sizeof(pSettlementInfoConfirm->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_confirm, "confirm_date",
                pSettlementInfoConfirm->ConfirmDate, sizeof(pSettlementInfoConfirm->ConfirmDate));
            dict_set_string(py_confirm, "confirm_time",
                pSettlementInfoConfirm->ConfirmTime, sizeof(pSettlementInfoConfirm->ConfirmTime));
            dict_set_long(py_confirm, "settlement_id", pSettlementInfoConfirm->SettlementID);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_settlement_info_confirm", "(OOii)",
            py_confirm, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_confirm);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 交易通知
     */
    FORCE_INLINE_MEMBER void OnRtnTradingNotice(CThostFtdcTradingNoticeInfoField* pTradingNoticeInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_notice = Py_None;
        Py_INCREF(Py_None);
        if (pTradingNoticeInfo) {
            py_notice = PyDict_New();
            dict_set_pooled_string(py_notice, "broker_id",
                pTradingNoticeInfo->BrokerID, sizeof(pTradingNoticeInfo->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_notice, "investor_range",
                pTradingNoticeInfo->InvestorRange, sizeof(pTradingNoticeInfo->InvestorRange), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_notice, "investor_id",
                pTradingNoticeInfo->InvestorID, sizeof(pTradingNoticeInfo->InvestorID), GlobalStringPools::Users);
            dict_set_long(py_notice, "sequence_series", pTradingNoticeInfo->SequenceSeries);
            dict_set_pooled_string(py_notice, "user_id",
                pTradingNoticeInfo->UserID, sizeof(pTradingNoticeInfo->UserID), GlobalStringPools::Users);
            dict_set_string(py_notice, "send_time",
                pTradingNoticeInfo->SendTime, sizeof(pTradingNoticeInfo->SendTime));
            dict_set_long(py_notice, "sequence_no", pTradingNoticeInfo->SequenceNo);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_trading_notice", "(O)", py_notice);

        Py_XDECREF(py_notice);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 交易所公告通知
     */
    FORCE_INLINE_MEMBER void OnRtnBulletin(CThostFtdcBulletinField* pBulletin) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_bulletin = Py_None;
        Py_INCREF(Py_None);
        if (pBulletin) {
            py_bulletin = PyDict_New();
            dict_set_pooled_string(py_bulletin, "exchange_id",
                pBulletin->ExchangeID, sizeof(pBulletin->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_bulletin, "trading_day",
                pBulletin->TradingDay, sizeof(pBulletin->TradingDay), GlobalStringPools::Dates);
            dict_set_pooled_string(py_bulletin, "bulletin_id",
                pBulletin->BulletinID, sizeof(pBulletin->BulletinID), GlobalStringPools::ExchangeCodes);
            dict_set_long(py_bulletin, "sequence_no", pBulletin->SequenceNo);
            dict_set_pooled_string(py_bulletin, "news_type",
                pBulletin->NewsType, sizeof(pBulletin->NewsType), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_bulletin, "news_urgency",
                pBulletin->NewsUrgency, sizeof(pBulletin->NewsUrgency), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_bulletin, "send_time",
                pBulletin->SendTime, sizeof(pBulletin->SendTime));
            dict_set_string(py_bulletin, "abstract",
                pBulletin->Abstract, sizeof(pBulletin->Abstract));
            // Content and SourceFile are binary data
            dict_set_bytes(py_bulletin, "content",
                pBulletin->Content, strlen(pBulletin->Content));
            dict_set_bytes(py_bulletin, "source_file",
                pBulletin->SourceFile, strlen(pBulletin->SourceFile));
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_bulletin", "(O)", py_bulletin);

        Py_XDECREF(py_bulletin);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 提示条件单校验错误
     */
    FORCE_INLINE_MEMBER void OnRtnErrorConditionalOrder(CThostFtdcErrorConditionalOrderField* pErrorConditionalOrder) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_order = Py_None;
        Py_INCREF(Py_None);
        if (pErrorConditionalOrder) {
            py_order = PyDict_New();
            dict_set_pooled_string(py_order, "broker_id",
                pErrorConditionalOrder->BrokerID, sizeof(pErrorConditionalOrder->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_order, "investor_id",
                pErrorConditionalOrder->InvestorID, sizeof(pErrorConditionalOrder->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_order, "order_ref",
                pErrorConditionalOrder->OrderRef, sizeof(pErrorConditionalOrder->OrderRef), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_order, "user_id",
                pErrorConditionalOrder->UserID, sizeof(pErrorConditionalOrder->UserID), GlobalStringPools::Users);
            dict_set_pooled_string(py_order, "order_price_type",
                pErrorConditionalOrder->OrderPriceType, sizeof(pErrorConditionalOrder->OrderPriceType), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_order, "direction",
                pErrorConditionalOrder->Direction, sizeof(pErrorConditionalOrder->Direction), GlobalStringPools::ExchangeCodes);
            dict_set_double(py_order, "limit_price", pErrorConditionalOrder->LimitPrice);
            dict_set_long(py_order, "volume_total_original", pErrorConditionalOrder->VolumeTotalOriginal);
            dict_set_string(py_order, "time_condition",
                pErrorConditionalOrder->TimeCondition, sizeof(pErrorConditionalOrder->TimeCondition));
            dict_set_pooled_string(py_order, "gtd_date",
                pErrorConditionalOrder->GTDDate, sizeof(pErrorConditionalOrder->GTDDate), GlobalStringPools::Dates);
            dict_set_string(py_order, "volume_condition",
                pErrorConditionalOrder->VolumeCondition, sizeof(pErrorConditionalOrder->VolumeCondition));
            dict_set_long(py_order, "min_volume", pErrorConditionalOrder->MinVolume);
            dict_set_string(py_order, "contingent_condition",
                pErrorConditionalOrder->ContingentCondition, sizeof(pErrorConditionalOrder->ContingentCondition));
            dict_set_double(py_order, "stop_price", pErrorConditionalOrder->StopPrice);
            dict_set_pooled_string(py_order, "force_close_reason",
                pErrorConditionalOrder->ForceCloseReason, sizeof(pErrorConditionalOrder->ForceCloseReason), GlobalStringPools::ExchangeCodes);
            dict_set_long(py_order, "is_auto_suspend", pErrorConditionalOrder->IsAutoSuspend);
            dict_set_string(py_order, "business_unit",
                pErrorConditionalOrder->BusinessUnit, sizeof(pErrorConditionalOrder->BusinessUnit));
            dict_set_long(py_order, "user_force_close", pErrorConditionalOrder->UserForceClose);
            dict_set_pooled_string(py_order, "error_code",
                pErrorConditionalOrder->ErrorCode, sizeof(pErrorConditionalOrder->ErrorCode), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_order, "error_msg",
                pErrorConditionalOrder->ErrorMsg, sizeof(pErrorConditionalOrder->ErrorMsg));
            dict_set_pooled_string(py_order, "instrument_id",
                pErrorConditionalOrder->InstrumentID, sizeof(pErrorConditionalOrder->InstrumentID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_order, "exchange_id",
                pErrorConditionalOrder->ExchangeID, sizeof(pErrorConditionalOrder->ExchangeID), GlobalStringPools::ExchangeCodes);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_error_conditional_order", "(O)", py_order);

        Py_XDECREF(py_order);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询签约银行响应
     */
    FORCE_INLINE_MEMBER void OnRspQryContractBank(CThostFtdcContractBankField* pContractBank,
                                                 CThostFtdcRspInfoField* pRspInfo,
                                                 int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_bank = Py_None;
        Py_INCREF(Py_None);
        if (pContractBank) {
            py_bank = PyDict_New();
            dict_set_pooled_string(py_bank, "broker_id",
                pContractBank->BrokerID, sizeof(pContractBank->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_bank, "bank_id",
                pContractBank->BankID, sizeof(pContractBank->BankID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_bank, "bank_brch_id",
                pContractBank->BankBrchID, sizeof(pContractBank->BankBrchID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_bank, "bank_name",
                pContractBank->BankName, sizeof(pContractBank->BankName));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_contract_bank", "(OOii)",
            py_bank, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_bank);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询交易通知响应
     */
    FORCE_INLINE_MEMBER void OnRspQryTradingNotice(CThostFtdcTradingNoticeField* pTradingNotice,
                                                   CThostFtdcRspInfoField* pRspInfo,
                                                   int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_notice = Py_None;
        Py_INCREF(Py_None);
        if (pTradingNotice) {
            py_notice = PyDict_New();
            dict_set_pooled_string(py_notice, "broker_id",
                pTradingNotice->BrokerID, sizeof(pTradingNotice->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_notice, "investor_range",
                pTradingNotice->InvestorRange, sizeof(pTradingNotice->InvestorRange), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_notice, "investor_id",
                pTradingNotice->InvestorID, sizeof(pTradingNotice->InvestorID), GlobalStringPools::Users);
            dict_set_long(py_notice, "sequence_series", pTradingNotice->SequenceSeries);
            dict_set_pooled_string(py_notice, "user_id",
                pTradingNotice->UserID, sizeof(pTradingNotice->UserID), GlobalStringPools::Users);
            dict_set_string(py_notice, "send_time",
                pTradingNotice->SendTime, sizeof(pTradingNotice->SendTime));
            dict_set_long(py_notice, "sequence_no", pTradingNotice->SequenceNo);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_trading_notice", "(OOii)",
            py_notice, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_notice);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 请求查询监控中心用户令牌
     */
    FORCE_INLINE_MEMBER void OnRspQueryCFMMCTradingAccountToken(CThostFtdcQueryCFMMCTradingAccountTokenField* pQueryCFMMCTradingAccountToken,
                                                           CThostFtdcRspInfoField* pRspInfo,
                                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_token = Py_None;
        Py_INCREF(Py_None);
        if (pQueryCFMMCTradingAccountToken) {
            py_token = PyDict_New();
            dict_set_pooled_string(py_token, "broker_id",
                pQueryCFMMCTradingAccountToken->BrokerID, sizeof(pQueryCFMMCTradingAccountToken->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_token, "investor_id",
                pQueryCFMMCTradingAccountToken->InvestorID, sizeof(pQueryCFMMCTradingAccountToken->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_token, "invest_unit_id",
                pQueryCFMMCTradingAccountToken->InvestUnitID, sizeof(pQueryCFMMCTradingAccountToken->InvestUnitID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_query_c_f_m_m_c_trading_account_token", "(OOii)",
            py_token, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_token);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 保证金监控中心用户令牌
     */
    FORCE_INLINE_MEMBER void OnRtnCFMMCTradingAccountToken(CThostFtdcCFMMCTradingAccountTokenField* pCFMMCTradingAccountToken) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_token = Py_None;
        Py_INCREF(Py_None);
        if (pCFMMCTradingAccountToken) {
            py_token = PyDict_New();
            dict_set_pooled_string(py_token, "broker_id",
                pCFMMCTradingAccountToken->BrokerID, sizeof(pCFMMCTradingAccountToken->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_token, "participant_id",
                pCFMMCTradingAccountToken->ParticipantID, sizeof(pCFMMCTradingAccountToken->ParticipantID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_token, "account_id",
                pCFMMCTradingAccountToken->AccountID, sizeof(pCFMMCTradingAccountToken->AccountID), GlobalStringPools::Instruments);
            dict_set_long(py_token, "key_id", pCFMMCTradingAccountToken->KeyID);
            dict_set_pooled_string(py_token, "token",
                pCFMMCTradingAccountToken->Token, sizeof(pCFMMCTradingAccountToken->Token), GlobalStringPools::Instruments);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_c_f_m_m_c_trading_account_token", "(O)", py_token);

        Py_XDECREF(py_token);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 错误应答
     */
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

    /**
     * @brief 合约交易状态通知
     */
    FORCE_INLINE_MEMBER void OnRtnInstrumentStatus(CThostFtdcInstrumentStatusField* pInstrumentStatus) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_status = Py_None;
        Py_INCREF(Py_None);
        if (pInstrumentStatus) {
            py_status = PyDict_New();
            dict_set_pooled_string(py_status, "exchange_id",
                pInstrumentStatus->ExchangeID, sizeof(pInstrumentStatus->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_status, "instrument_id",
                pInstrumentStatus->InstrumentID, sizeof(pInstrumentStatus->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_status, "trading_day",
                pInstrumentStatus->TradingDay, sizeof(pInstrumentStatus->TradingDay));
            dict_set_string(py_status, "enter_time",
                pInstrumentStatus->EnterTime, sizeof(pInstrumentStatus->EnterTime));
            dict_set_long(py_status, "instrument_status", pInstrumentStatus->InstrumentStatus);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_instrument_status", "(O)", py_status);

        Py_XDECREF(py_status);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // =============================================================================
    // 第2批回调方法 (执行宣告+询价报价+批量操作)
    // =============================================================================

    /**
     * @brief 执行宣告录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspExecOrderInsert(CThostFtdcInputExecOrderField* pInputExecOrder,
                                                   CThostFtdcRspInfoField* pRspInfo,
                                                   int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_exec_order = Py_None;
        Py_INCREF(Py_None);
        if (pInputExecOrder) {
            py_exec_order = PyDict_New();
            dict_set_pooled_string(py_exec_order, "broker_id",
                pInputExecOrder->BrokerID, sizeof(pInputExecOrder->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_exec_order, "investor_id",
                pInputExecOrder->InvestorID, sizeof(pInputExecOrder->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_exec_order, "exec_order_ref",
                pInputExecOrder->ExecOrderRef, sizeof(pInputExecOrder->ExecOrderRef));
            dict_set_pooled_string(py_exec_order, "user_id",
                pInputExecOrder->UserID, sizeof(pInputExecOrder->UserID), GlobalStringPools::Users);
            dict_set_long(py_exec_order, "volume", pInputExecOrder->Volume);
            dict_set_long(py_exec_order, "request_id", pInputExecOrder->RequestID);
            dict_set_string(py_exec_order, "business_unit",
                pInputExecOrder->BusinessUnit, sizeof(pInputExecOrder->BusinessUnit));
            dict_set_string(py_exec_order, "offset_flag",
                pInputExecOrder->OffsetFlag, sizeof(pInputExecOrder->OffsetFlag));
            dict_set_string(py_exec_order, "hedge_flag",
                pInputExecOrder->HedgeFlag, sizeof(pInputExecOrder->HedgeFlag));
            dict_set_string(py_exec_order, "action_type",
                pInputExecOrder->ActionType, sizeof(pInputExecOrder->ActionType));
            dict_set_string(py_exec_order, "posi_direction",
                pInputExecOrder->PosiDirection, sizeof(pInputExecOrder->PosiDirection));
            dict_set_string(py_exec_order, "reserve_position_flag",
                pInputExecOrder->ReservePositionFlag, sizeof(pInputExecOrder->ReservePositionFlag));
            dict_set_string(py_exec_order, "close_flag",
                pInputExecOrder->CloseFlag, sizeof(pInputExecOrder->CloseFlag));
            dict_set_pooled_string(py_exec_order, "exchange_id",
                pInputExecOrder->ExchangeID, sizeof(pInputExecOrder->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_exec_order, "invest_unit_id",
                pInputExecOrder->InvestUnitID, sizeof(pInputExecOrder->InvestUnitID));
            dict_set_string(py_exec_order, "account_id",
                pInputExecOrder->AccountID, sizeof(pInputExecOrder->AccountID));
            dict_set_string(py_exec_order, "currency_id",
                pInputExecOrder->CurrencyID, sizeof(pInputExecOrder->CurrencyID));
            dict_set_pooled_string(py_exec_order, "instrument_id",
                pInputExecOrder->InstrumentID, sizeof(pInputExecOrder->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_exec_order_insert", "(OOii)",
            py_exec_order, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_exec_order);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 执行宣告操作请求响应
     */
    FORCE_INLINE_MEMBER void OnRspExecOrderAction(CThostFtdcInputExecOrderActionField* pInputExecOrderAction,
                                                   CThostFtdcRspInfoField* pRspInfo,
                                                   int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputExecOrderAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pInputExecOrderAction->BrokerID, sizeof(pInputExecOrderAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pInputExecOrderAction->InvestorID, sizeof(pInputExecOrderAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "exec_order_action_ref",
                pInputExecOrderAction->ExecOrderActionRef, sizeof(pInputExecOrderAction->ExecOrderActionRef));
            dict_set_string(py_action, "exec_order_ref",
                pInputExecOrderAction->ExecOrderRef, sizeof(pInputExecOrderAction->ExecOrderRef));
            dict_set_long(py_action, "request_id", pInputExecOrderAction->RequestID);
            dict_set_long(py_action, "front_id", pInputExecOrderAction->FrontID);
            dict_set_long(py_action, "session_id", pInputExecOrderAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pInputExecOrderAction->ExchangeID, sizeof(pInputExecOrderAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_action, "exec_order_sys_id",
                pInputExecOrderAction->ExecOrderSysID, sizeof(pInputExecOrderAction->ExecOrderSysID));
            dict_set_string(py_action, "action_flag",
                pInputExecOrderAction->ActionFlag, sizeof(pInputExecOrderAction->ActionFlag));
            dict_set_pooled_string(py_action, "user_id",
                pInputExecOrderAction->UserID, sizeof(pInputExecOrderAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_action, "invest_unit_id",
                pInputExecOrderAction->InvestUnitID, sizeof(pInputExecOrderAction->InvestUnitID));
            dict_set_pooled_string(py_action, "instrument_id",
                pInputExecOrderAction->InstrumentID, sizeof(pInputExecOrderAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_exec_order_action", "(OOii)",
            py_action, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 执行宣告通知
     */
    FORCE_INLINE_MEMBER void OnRtnExecOrder(CThostFtdcExecOrderField* pExecOrder) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_exec_order = Py_None;
        Py_INCREF(Py_None);
        if (pExecOrder) {
            py_exec_order = PyDict_New();
            dict_set_pooled_string(py_exec_order, "broker_id",
                pExecOrder->BrokerID, sizeof(pExecOrder->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_exec_order, "investor_id",
                pExecOrder->InvestorID, sizeof(pExecOrder->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_exec_order, "exec_order_ref",
                pExecOrder->ExecOrderRef, sizeof(pExecOrder->ExecOrderRef));
            dict_set_pooled_string(py_exec_order, "user_id",
                pExecOrder->UserID, sizeof(pExecOrder->UserID), GlobalStringPools::Users);
            dict_set_long(py_exec_order, "volume", pExecOrder->Volume);
            dict_set_long(py_exec_order, "request_id", pExecOrder->RequestID);
            dict_set_string(py_exec_order, "business_unit",
                pExecOrder->BusinessUnit, sizeof(pExecOrder->BusinessUnit));
            dict_set_string(py_exec_order, "offset_flag",
                pExecOrder->OffsetFlag, sizeof(pExecOrder->OffsetFlag));
            dict_set_string(py_exec_order, "hedge_flag",
                pExecOrder->HedgeFlag, sizeof(pExecOrder->HedgeFlag));
            dict_set_string(py_exec_order, "action_type",
                pExecOrder->ActionType, sizeof(pExecOrder->ActionType));
            dict_set_string(py_exec_order, "posi_direction",
                pExecOrder->PosiDirection, sizeof(pExecOrder->PosiDirection));
            dict_set_string(py_exec_order, "reserve_position_flag",
                pExecOrder->ReservePositionFlag, sizeof(pExecOrder->ReservePositionFlag));
            dict_set_string(py_exec_order, "close_flag",
                pExecOrder->CloseFlag, sizeof(pExecOrder->CloseFlag));
            dict_set_string(py_exec_order, "exec_order_local_id",
                pExecOrder->ExecOrderLocalID, sizeof(pExecOrder->ExecOrderLocalID));
            dict_set_pooled_string(py_exec_order, "exchange_id",
                pExecOrder->ExchangeID, sizeof(pExecOrder->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_exec_order, "trading_day",
                pExecOrder->TradingDay, sizeof(pExecOrder->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_exec_order, "settlement_id", pExecOrder->SettlementID);
            dict_set_string(py_exec_order, "exec_order_sys_id",
                pExecOrder->ExecOrderSysID, sizeof(pExecOrder->ExecOrderSysID));
            dict_set_pooled_string(py_exec_order, "insert_date",
                pExecOrder->InsertDate, sizeof(pExecOrder->InsertDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_exec_order, "insert_time",
                pExecOrder->InsertTime, sizeof(pExecOrder->InsertTime), GlobalStringPools::Times);
            dict_set_pooled_string(py_exec_order, "cancel_time",
                pExecOrder->CancelTime, sizeof(pExecOrder->CancelTime), GlobalStringPools::Times);
            dict_set_string(py_exec_order, "exec_result",
                pExecOrder->ExecResult, sizeof(pExecOrder->ExecResult));
            dict_set_long(py_exec_order, "sequence_no", pExecOrder->SequenceNo);
            dict_set_long(py_exec_order, "front_id", pExecOrder->FrontID);
            dict_set_long(py_exec_order, "session_id", pExecOrder->SessionID);
            dict_set_string(py_exec_order, "user_product_info",
                pExecOrder->UserProductInfo, sizeof(pExecOrder->UserProductInfo));
            dict_set_string(py_exec_order, "status_msg",
                pExecOrder->StatusMsg, sizeof(pExecOrder->StatusMsg));
            dict_set_pooled_string(py_exec_order, "active_user_id",
                pExecOrder->ActiveUserID, sizeof(pExecOrder->ActiveUserID), GlobalStringPools::Users);
            dict_set_long(py_exec_order, "broker_exec_order_seq", pExecOrder->BrokerExecOrderSeq);
            dict_set_string(py_exec_order, "branch_id",
                pExecOrder->BranchID, sizeof(pExecOrder->BranchID));
            dict_set_string(py_exec_order, "invest_unit_id",
                pExecOrder->InvestUnitID, sizeof(pExecOrder->InvestUnitID));
            dict_set_string(py_exec_order, "account_id",
                pExecOrder->AccountID, sizeof(pExecOrder->AccountID));
            dict_set_string(py_exec_order, "currency_id",
                pExecOrder->CurrencyID, sizeof(pExecOrder->CurrencyID));
            dict_set_pooled_string(py_exec_order, "instrument_id",
                pExecOrder->InstrumentID, sizeof(pExecOrder->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_exec_order", "(O)", py_exec_order);

        Py_XDECREF(py_exec_order);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 执行宣告错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnExecOrderInsert(CThostFtdcInputExecOrderField* pInputExecOrder,
                                                      CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_exec_order = Py_None;
        Py_INCREF(Py_None);
        if (pInputExecOrder) {
            py_exec_order = PyDict_New();
            dict_set_pooled_string(py_exec_order, "broker_id",
                pInputExecOrder->BrokerID, sizeof(pInputExecOrder->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_exec_order, "investor_id",
                pInputExecOrder->InvestorID, sizeof(pInputExecOrder->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_exec_order, "exec_order_ref",
                pInputExecOrder->ExecOrderRef, sizeof(pInputExecOrder->ExecOrderRef));
            dict_set_pooled_string(py_exec_order, "user_id",
                pInputExecOrder->UserID, sizeof(pInputExecOrder->UserID), GlobalStringPools::Users);
            dict_set_long(py_exec_order, "volume", pInputExecOrder->Volume);
            dict_set_pooled_string(py_exec_order, "exchange_id",
                pInputExecOrder->ExchangeID, sizeof(pInputExecOrder->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_exec_order, "instrument_id",
                pInputExecOrder->InstrumentID, sizeof(pInputExecOrder->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_exec_order_insert", "(OO)",
            py_exec_order, py_rsp_info);

        Py_XDECREF(py_exec_order);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 执行宣告操作错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnExecOrderAction(CThostFtdcInputExecOrderActionField* pInputExecOrderAction,
                                                      CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputExecOrderAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pInputExecOrderAction->BrokerID, sizeof(pInputExecOrderAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pInputExecOrderAction->InvestorID, sizeof(pInputExecOrderAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "exec_order_ref",
                pInputExecOrderAction->ExecOrderRef, sizeof(pInputExecOrderAction->ExecOrderRef));
            dict_set_long(py_action, "front_id", pInputExecOrderAction->FrontID);
            dict_set_long(py_action, "session_id", pInputExecOrderAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pInputExecOrderAction->ExchangeID, sizeof(pInputExecOrderAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_action, "exec_order_sys_id",
                pInputExecOrderAction->ExecOrderSysID, sizeof(pInputExecOrderAction->ExecOrderSysID));
            dict_set_pooled_string(py_action, "instrument_id",
                pInputExecOrderAction->InstrumentID, sizeof(pInputExecOrderAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_exec_order_action", "(OO)",
            py_action, py_rsp_info);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询执行宣告响应
     */
    FORCE_INLINE_MEMBER void OnRspQryExecOrder(CThostFtdcExecOrderField* pExecOrder,
                                                CThostFtdcRspInfoField* pRspInfo,
                                                int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_exec_order = Py_None;
        Py_INCREF(Py_None);
        if (pExecOrder) {
            py_exec_order = PyDict_New();
            dict_set_pooled_string(py_exec_order, "broker_id",
                pExecOrder->BrokerID, sizeof(pExecOrder->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_exec_order, "investor_id",
                pExecOrder->InvestorID, sizeof(pExecOrder->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_exec_order, "exec_order_ref",
                pExecOrder->ExecOrderRef, sizeof(pExecOrder->ExecOrderRef));
            dict_set_long(py_exec_order, "volume", pExecOrder->Volume);
            dict_set_long(py_exec_order, "request_id", pExecOrder->RequestID);
            dict_set_string(py_exec_order, "offset_flag",
                pExecOrder->OffsetFlag, sizeof(pExecOrder->OffsetFlag));
            dict_set_string(py_exec_order, "hedge_flag",
                pExecOrder->HedgeFlag, sizeof(pExecOrder->HedgeFlag));
            dict_set_string(py_exec_order, "action_type",
                pExecOrder->ActionType, sizeof(pExecOrder->ActionType));
            dict_set_pooled_string(py_exec_order, "exchange_id",
                pExecOrder->ExchangeID, sizeof(pExecOrder->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_exec_order, "trading_day",
                pExecOrder->TradingDay, sizeof(pExecOrder->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_exec_order, "settlement_id", pExecOrder->SettlementID);
            dict_set_string(py_exec_order, "exec_order_sys_id",
                pExecOrder->ExecOrderSysID, sizeof(pExecOrder->ExecOrderSysID));
            dict_set_pooled_string(py_exec_order, "insert_date",
                pExecOrder->InsertDate, sizeof(pExecOrder->InsertDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_exec_order, "insert_time",
                pExecOrder->InsertTime, sizeof(pExecOrder->InsertTime), GlobalStringPools::Times);
            dict_set_string(py_exec_order, "exec_result",
                pExecOrder->ExecResult, sizeof(pExecOrder->ExecResult));
            dict_set_long(py_exec_order, "front_id", pExecOrder->FrontID);
            dict_set_long(py_exec_order, "session_id", pExecOrder->SessionID);
            dict_set_string(py_exec_order, "status_msg",
                pExecOrder->StatusMsg, sizeof(pExecOrder->StatusMsg));
            dict_set_pooled_string(py_exec_order, "instrument_id",
                pExecOrder->InstrumentID, sizeof(pExecOrder->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_exec_order", "(OOii)",
            py_exec_order, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_exec_order);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 询价录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspForQuoteInsert(CThostFtdcInputForQuoteField* pInputForQuote,
                                                  CThostFtdcRspInfoField* pRspInfo,
                                                  int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_for_quote = Py_None;
        Py_INCREF(Py_None);
        if (pInputForQuote) {
            py_for_quote = PyDict_New();
            dict_set_pooled_string(py_for_quote, "broker_id",
                pInputForQuote->BrokerID, sizeof(pInputForQuote->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_for_quote, "investor_id",
                pInputForQuote->InvestorID, sizeof(pInputForQuote->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_for_quote, "for_quote_ref",
                pInputForQuote->ForQuoteRef, sizeof(pInputForQuote->ForQuoteRef));
            dict_set_pooled_string(py_for_quote, "user_id",
                pInputForQuote->UserID, sizeof(pInputForQuote->UserID), GlobalStringPools::Users);
            dict_set_pooled_string(py_for_quote, "exchange_id",
                pInputForQuote->ExchangeID, sizeof(pInputForQuote->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_for_quote, "invest_unit_id",
                pInputForQuote->InvestUnitID, sizeof(pInputForQuote->InvestUnitID));
            dict_set_pooled_string(py_for_quote, "instrument_id",
                pInputForQuote->InstrumentID, sizeof(pInputForQuote->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_for_quote_insert", "(OOii)",
            py_for_quote, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_for_quote);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报价录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspQuoteInsert(CThostFtdcInputQuoteField* pInputQuote,
                                               CThostFtdcRspInfoField* pRspInfo,
                                               int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_quote = Py_None;
        Py_INCREF(Py_None);
        if (pInputQuote) {
            py_quote = PyDict_New();
            dict_set_pooled_string(py_quote, "broker_id",
                pInputQuote->BrokerID, sizeof(pInputQuote->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_quote, "investor_id",
                pInputQuote->InvestorID, sizeof(pInputQuote->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_quote, "quote_ref",
                pInputQuote->QuoteRef, sizeof(pInputQuote->QuoteRef));
            dict_set_pooled_string(py_quote, "user_id",
                pInputQuote->UserID, sizeof(pInputQuote->UserID), GlobalStringPools::Users);
            dict_set_double(py_quote, "ask_price", pInputQuote->AskPrice);
            dict_set_double(py_quote, "bid_price", pInputQuote->BidPrice);
            dict_set_long(py_quote, "ask_volume", pInputQuote->AskVolume);
            dict_set_long(py_quote, "bid_volume", pInputQuote->BidVolume);
            dict_set_long(py_quote, "request_id", pInputQuote->RequestID);
            dict_set_string(py_quote, "business_unit",
                pInputQuote->BusinessUnit, sizeof(pInputQuote->BusinessUnit));
            dict_set_string(py_quote, "ask_offset_flag",
                pInputQuote->AskOffsetFlag, sizeof(pInputQuote->AskOffsetFlag));
            dict_set_string(py_quote, "bid_offset_flag",
                pInputQuote->BidOffsetFlag, sizeof(pInputQuote->BidOffsetFlag));
            dict_set_string(py_quote, "ask_hedge_flag",
                pInputQuote->AskHedgeFlag, sizeof(pInputQuote->AskHedgeFlag));
            dict_set_string(py_quote, "bid_hedge_flag",
                pInputQuote->BidHedgeFlag, sizeof(pInputQuote->BidHedgeFlag));
            dict_set_string(py_quote, "ask_order_ref",
                pInputQuote->AskOrderRef, sizeof(pInputQuote->AskOrderRef));
            dict_set_string(py_quote, "bid_order_ref",
                pInputQuote->BidOrderRef, sizeof(pInputQuote->BidOrderRef));
            dict_set_string(py_quote, "for_quote_sys_id",
                pInputQuote->ForQuoteSysID, sizeof(pInputQuote->ForQuoteSysID));
            dict_set_pooled_string(py_quote, "exchange_id",
                pInputQuote->ExchangeID, sizeof(pInputQuote->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_quote, "invest_unit_id",
                pInputQuote->InvestUnitID, sizeof(pInputQuote->InvestUnitID));
            dict_set_string(py_quote, "client_id",
                pInputQuote->ClientID, sizeof(pInputQuote->ClientID));
            dict_set_pooled_string(py_quote, "instrument_id",
                pInputQuote->InstrumentID, sizeof(pInputQuote->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_quote, "replace_sys_id",
                pInputQuote->ReplaceSysID, sizeof(pInputQuote->ReplaceSysID));
            dict_set_string(py_quote, "time_condition",
                pInputQuote->TimeCondition, sizeof(pInputQuote->TimeCondition));
            dict_set_string(py_quote, "order_memo",
                pInputQuote->OrderMemo, sizeof(pInputQuote->OrderMemo));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_quote_insert", "(OOii)",
            py_quote, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_quote);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报价操作请求响应
     */
    FORCE_INLINE_MEMBER void OnRspQuoteAction(CThostFtdcInputQuoteActionField* pInputQuoteAction,
                                               CThostFtdcRspInfoField* pRspInfo,
                                               int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputQuoteAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pInputQuoteAction->BrokerID, sizeof(pInputQuoteAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pInputQuoteAction->InvestorID, sizeof(pInputQuoteAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "quote_action_ref",
                pInputQuoteAction->QuoteActionRef, sizeof(pInputQuoteAction->QuoteActionRef));
            dict_set_string(py_action, "quote_ref",
                pInputQuoteAction->QuoteRef, sizeof(pInputQuoteAction->QuoteRef));
            dict_set_long(py_action, "request_id", pInputQuoteAction->RequestID);
            dict_set_long(py_action, "front_id", pInputQuoteAction->FrontID);
            dict_set_long(py_action, "session_id", pInputQuoteAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pInputQuoteAction->ExchangeID, sizeof(pInputQuoteAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_action, "quote_sys_id",
                pInputQuoteAction->QuoteSysID, sizeof(pInputQuoteAction->QuoteSysID));
            dict_set_string(py_action, "action_flag",
                pInputQuoteAction->ActionFlag, sizeof(pInputQuoteAction->ActionFlag));
            dict_set_pooled_string(py_action, "user_id",
                pInputQuoteAction->UserID, sizeof(pInputQuoteAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_action, "invest_unit_id",
                pInputQuoteAction->InvestUnitID, sizeof(pInputQuoteAction->InvestUnitID));
            dict_set_string(py_action, "client_id",
                pInputQuoteAction->ClientID, sizeof(pInputQuoteAction->ClientID));
            dict_set_pooled_string(py_action, "instrument_id",
                pInputQuoteAction->InstrumentID, sizeof(pInputQuoteAction->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_action, "order_memo",
                pInputQuoteAction->OrderMemo, sizeof(pInputQuoteAction->OrderMemo));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_quote_action", "(OOii)",
            py_action, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 询价通知
     */
    FORCE_INLINE_MEMBER void OnRtnForQuoteRsp(CThostFtdcForQuoteRspField* pForQuoteRsp) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rsp = Py_None;
        Py_INCREF(Py_None);
        if (pForQuoteRsp) {
            py_rsp = PyDict_New();
            dict_set_pooled_string(py_rsp, "trading_day",
                pForQuoteRsp->TradingDay, sizeof(pForQuoteRsp->TradingDay), GlobalStringPools::Dates);
            dict_set_string(py_rsp, "for_quote_sys_id",
                pForQuoteRsp->ForQuoteSysID, sizeof(pForQuoteRsp->ForQuoteSysID));
            dict_set_pooled_string(py_rsp, "for_quote_time",
                pForQuoteRsp->ForQuoteTime, sizeof(pForQuoteRsp->ForQuoteTime), GlobalStringPools::Times);
            dict_set_pooled_string(py_rsp, "action_day",
                pForQuoteRsp->ActionDay, sizeof(pForQuoteRsp->ActionDay), GlobalStringPools::Dates);
            dict_set_pooled_string(py_rsp, "exchange_id",
                pForQuoteRsp->ExchangeID, sizeof(pForQuoteRsp->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_rsp, "instrument_id",
                pForQuoteRsp->InstrumentID, sizeof(pForQuoteRsp->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_for_quote_rsp", "(O)", py_rsp);

        Py_XDECREF(py_rsp);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报价通知
     */
    FORCE_INLINE_MEMBER void OnRtnQuote(CThostFtdcQuoteField* pQuote) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_quote = Py_None;
        Py_INCREF(Py_None);
        if (pQuote) {
            py_quote = PyDict_New();
            dict_set_pooled_string(py_quote, "broker_id",
                pQuote->BrokerID, sizeof(pQuote->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_quote, "investor_id",
                pQuote->InvestorID, sizeof(pQuote->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_quote, "quote_ref",
                pQuote->QuoteRef, sizeof(pQuote->QuoteRef));
            dict_set_pooled_string(py_quote, "user_id",
                pQuote->UserID, sizeof(pQuote->UserID), GlobalStringPools::Users);
            dict_set_double(py_quote, "ask_price", pQuote->AskPrice);
            dict_set_double(py_quote, "bid_price", pQuote->BidPrice);
            dict_set_long(py_quote, "ask_volume", pQuote->AskVolume);
            dict_set_long(py_quote, "bid_volume", pQuote->BidVolume);
            dict_set_long(py_quote, "request_id", pQuote->RequestID);
            dict_set_string(py_quote, "business_unit",
                pQuote->BusinessUnit, sizeof(pQuote->BusinessUnit));
            dict_set_string(py_quote, "ask_offset_flag",
                pQuote->AskOffsetFlag, sizeof(pQuote->AskOffsetFlag));
            dict_set_string(py_quote, "bid_offset_flag",
                pQuote->BidOffsetFlag, sizeof(pQuote->BidOffsetFlag));
            dict_set_string(py_quote, "ask_hedge_flag",
                pQuote->AskHedgeFlag, sizeof(pQuote->AskHedgeFlag));
            dict_set_string(py_quote, "bid_hedge_flag",
                pQuote->BidHedgeFlag, sizeof(pQuote->BidHedgeFlag));
            dict_set_string(py_quote, "ask_order_ref",
                pQuote->AskOrderRef, sizeof(pQuote->AskOrderRef));
            dict_set_string(py_quote, "bid_order_ref",
                pQuote->BidOrderRef, sizeof(pQuote->BidOrderRef));
            dict_set_string(py_quote, "for_quote_sys_id",
                pQuote->ForQuoteSysID, sizeof(pQuote->ForQuoteSysID));
            dict_set_pooled_string(py_quote, "exchange_id",
                pQuote->ExchangeID, sizeof(pQuote->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_quote, "quote_local_id",
                pQuote->QuoteLocalID, sizeof(pQuote->QuoteLocalID));
            dict_set_string(py_quote, "quote_sys_id",
                pQuote->QuoteSysID, sizeof(pQuote->QuoteSysID));
            dict_set_pooled_string(py_quote, "trading_day",
                pQuote->TradingDay, sizeof(pQuote->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_quote, "settlement_id", pQuote->SettlementID);
            dict_set_string(py_quote, "quote_status",
                pQuote->QuoteStatus, sizeof(pQuote->QuoteStatus));
            dict_set_long(py_quote, "front_id", pQuote->FrontID);
            dict_set_long(py_quote, "session_id", pQuote->SessionID);
            dict_set_string(py_quote, "status_msg",
                pQuote->StatusMsg, sizeof(pQuote->StatusMsg));
            dict_set_pooled_string(py_quote, "active_user_id",
                pQuote->ActiveUserID, sizeof(pQuote->ActiveUserID), GlobalStringPools::Users);
            dict_set_long(py_quote, "broker_quote_seq", pQuote->BrokerQuoteSeq);
            dict_set_string(py_quote, "invest_unit_id",
                pQuote->InvestUnitID, sizeof(pQuote->InvestUnitID));
            dict_set_pooled_string(py_quote, "instrument_id",
                pQuote->InstrumentID, sizeof(pQuote->InstrumentID), GlobalStringPools::Instruments);
            dict_set_string(py_quote, "exchange_inst_id",
                pQuote->ExchangeInstID, sizeof(pQuote->ExchangeInstID));
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_quote", "(O)", py_quote);

        Py_XDECREF(py_quote);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 询价录入错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnForQuoteInsert(CThostFtdcInputForQuoteField* pInputForQuote,
                                                     CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_for_quote = Py_None;
        Py_INCREF(Py_None);
        if (pInputForQuote) {
            py_for_quote = PyDict_New();
            dict_set_pooled_string(py_for_quote, "broker_id",
                pInputForQuote->BrokerID, sizeof(pInputForQuote->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_for_quote, "investor_id",
                pInputForQuote->InvestorID, sizeof(pInputForQuote->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_for_quote, "for_quote_ref",
                pInputForQuote->ForQuoteRef, sizeof(pInputForQuote->ForQuoteRef));
            dict_set_pooled_string(py_for_quote, "exchange_id",
                pInputForQuote->ExchangeID, sizeof(pInputForQuote->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_for_quote, "instrument_id",
                pInputForQuote->InstrumentID, sizeof(pInputForQuote->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_for_quote_insert", "(OO)",
            py_for_quote, py_rsp_info);

        Py_XDECREF(py_for_quote);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报价录入错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnQuoteInsert(CThostFtdcInputQuoteField* pInputQuote,
                                                  CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_quote = Py_None;
        Py_INCREF(Py_None);
        if (pInputQuote) {
            py_quote = PyDict_New();
            dict_set_pooled_string(py_quote, "broker_id",
                pInputQuote->BrokerID, sizeof(pInputQuote->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_quote, "investor_id",
                pInputQuote->InvestorID, sizeof(pInputQuote->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_quote, "quote_ref",
                pInputQuote->QuoteRef, sizeof(pInputQuote->QuoteRef));
            dict_set_pooled_string(py_quote, "exchange_id",
                pInputQuote->ExchangeID, sizeof(pInputQuote->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_quote, "instrument_id",
                pInputQuote->InstrumentID, sizeof(pInputQuote->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_quote_insert", "(OO)",
            py_quote, py_rsp_info);

        Py_XDECREF(py_quote);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 报价操作错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnQuoteAction(CThostFtdcInputQuoteActionField* pInputQuoteAction,
                                                  CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputQuoteAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pInputQuoteAction->BrokerID, sizeof(pInputQuoteAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pInputQuoteAction->InvestorID, sizeof(pInputQuoteAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "quote_ref",
                pInputQuoteAction->QuoteRef, sizeof(pInputQuoteAction->QuoteRef));
            dict_set_long(py_action, "front_id", pInputQuoteAction->FrontID);
            dict_set_long(py_action, "session_id", pInputQuoteAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pInputQuoteAction->ExchangeID, sizeof(pInputQuoteAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_action, "quote_sys_id",
                pInputQuoteAction->QuoteSysID, sizeof(pInputQuoteAction->QuoteSysID));
            dict_set_pooled_string(py_action, "instrument_id",
                pInputQuoteAction->InstrumentID, sizeof(pInputQuoteAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_quote_action", "(OO)",
            py_action, py_rsp_info);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询询价响应
     */
    FORCE_INLINE_MEMBER void OnRspQryForQuote(CThostFtdcForQuoteField* pForQuote,
                                               CThostFtdcRspInfoField* pRspInfo,
                                               int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_for_quote = Py_None;
        Py_INCREF(Py_None);
        if (pForQuote) {
            py_for_quote = PyDict_New();
            dict_set_pooled_string(py_for_quote, "broker_id",
                pForQuote->BrokerID, sizeof(pForQuote->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_for_quote, "investor_id",
                pForQuote->InvestorID, sizeof(pForQuote->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_for_quote, "for_quote_ref",
                pForQuote->ForQuoteRef, sizeof(pForQuote->ForQuoteRef));
            dict_set_pooled_string(py_for_quote, "user_id",
                pForQuote->UserID, sizeof(pForQuote->UserID), GlobalStringPools::Users);
            dict_set_string(py_for_quote, "for_quote_local_id",
                pForQuote->ForQuoteLocalID, sizeof(pForQuote->ForQuoteLocalID));
            dict_set_pooled_string(py_for_quote, "exchange_id",
                pForQuote->ExchangeID, sizeof(pForQuote->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_for_quote, "insert_date",
                pForQuote->InsertDate, sizeof(pForQuote->InsertDate));
            dict_set_pooled_string(py_for_quote, "insert_time",
                pForQuote->InsertTime, sizeof(pForQuote->InsertTime), GlobalStringPools::Times);
            dict_set_string(py_for_quote, "for_quote_status",
                pForQuote->ForQuoteStatus, sizeof(pForQuote->ForQuoteStatus));
            dict_set_long(py_for_quote, "front_id", pForQuote->FrontID);
            dict_set_long(py_for_quote, "session_id", pForQuote->SessionID);
            dict_set_string(py_for_quote, "status_msg",
                pForQuote->StatusMsg, sizeof(pForQuote->StatusMsg));
            dict_set_pooled_string(py_for_quote, "active_user_id",
                pForQuote->ActiveUserID, sizeof(pForQuote->ActiveUserID), GlobalStringPools::Users);
            dict_set_pooled_string(py_for_quote, "instrument_id",
                pForQuote->InstrumentID, sizeof(pForQuote->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_for_quote", "(OOii)",
            py_for_quote, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_for_quote);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询报价响应
     */
    FORCE_INLINE_MEMBER void OnRspQryQuote(CThostFtdcQuoteField* pQuote,
                                            CThostFtdcRspInfoField* pRspInfo,
                                            int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_quote = Py_None;
        Py_INCREF(Py_None);
        if (pQuote) {
            py_quote = PyDict_New();
            dict_set_pooled_string(py_quote, "broker_id",
                pQuote->BrokerID, sizeof(pQuote->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_quote, "investor_id",
                pQuote->InvestorID, sizeof(pQuote->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_quote, "quote_ref",
                pQuote->QuoteRef, sizeof(pQuote->QuoteRef));
            dict_set_double(py_quote, "ask_price", pQuote->AskPrice);
            dict_set_double(py_quote, "bid_price", pQuote->BidPrice);
            dict_set_long(py_quote, "ask_volume", pQuote->AskVolume);
            dict_set_long(py_quote, "bid_volume", pQuote->BidVolume);
            dict_set_string(py_quote, "ask_offset_flag",
                pQuote->AskOffsetFlag, sizeof(pQuote->AskOffsetFlag));
            dict_set_string(py_quote, "bid_offset_flag",
                pQuote->BidOffsetFlag, sizeof(pQuote->BidOffsetFlag));
            dict_set_pooled_string(py_quote, "exchange_id",
                pQuote->ExchangeID, sizeof(pQuote->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_quote, "quote_sys_id",
                pQuote->QuoteSysID, sizeof(pQuote->QuoteSysID));
            dict_set_pooled_string(py_quote, "trading_day",
                pQuote->TradingDay, sizeof(pQuote->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_quote, "settlement_id", pQuote->SettlementID);
            dict_set_string(py_quote, "quote_status",
                pQuote->QuoteStatus, sizeof(pQuote->QuoteStatus));
            dict_set_long(py_quote, "front_id", pQuote->FrontID);
            dict_set_long(py_quote, "session_id", pQuote->SessionID);
            dict_set_string(py_quote, "status_msg",
                pQuote->StatusMsg, sizeof(pQuote->StatusMsg));
            dict_set_pooled_string(py_quote, "instrument_id",
                pQuote->InstrumentID, sizeof(pQuote->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_quote", "(OOii)",
            py_quote, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_quote);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询最大报单数量响应
     */
    FORCE_INLINE_MEMBER void OnRspQueryMaxOrderVolume(CThostFtdcQryMaxOrderVolumeField* pQryMaxOrderVolume,
                                                       CThostFtdcRspInfoField* pRspInfo,
                                                       int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_result = Py_None;
        Py_INCREF(Py_None);
        if (pQryMaxOrderVolume) {
            py_result = PyDict_New();
            dict_set_pooled_string(py_result, "broker_id",
                pQryMaxOrderVolume->BrokerID, sizeof(pQryMaxOrderVolume->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_result, "investor_id",
                pQryMaxOrderVolume->InvestorID, sizeof(pQryMaxOrderVolume->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_result, "direction",
                pQryMaxOrderVolume->Direction, sizeof(pQryMaxOrderVolume->Direction));
            dict_set_string(py_result, "offset_flag",
                pQryMaxOrderVolume->OffsetFlag, sizeof(pQryMaxOrderVolume->OffsetFlag));
            dict_set_string(py_result, "hedge_flag",
                pQryMaxOrderVolume->HedgeFlag, sizeof(pQryMaxOrderVolume->HedgeFlag));
            dict_set_long(py_result, "max_volume", pQryMaxOrderVolume->MaxVolume);
            dict_set_pooled_string(py_result, "exchange_id",
                pQryMaxOrderVolume->ExchangeID, sizeof(pQryMaxOrderVolume->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_result, "invest_unit_id",
                pQryMaxOrderVolume->InvestUnitID, sizeof(pQryMaxOrderVolume->InvestUnitID));
            dict_set_pooled_string(py_result, "instrument_id",
                pQryMaxOrderVolume->InstrumentID, sizeof(pQryMaxOrderVolume->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_query_max_order_volume", "(OOii)",
            py_result, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_result);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 批量报单操作请求响应
     */
    FORCE_INLINE_MEMBER void OnRspBatchOrderAction(CThostFtdcInputBatchOrderActionField* pInputBatchOrderAction,
                                                    CThostFtdcRspInfoField* pRspInfo,
                                                    int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputBatchOrderAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pInputBatchOrderAction->BrokerID, sizeof(pInputBatchOrderAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pInputBatchOrderAction->InvestorID, sizeof(pInputBatchOrderAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "order_action_ref",
                pInputBatchOrderAction->OrderActionRef, sizeof(pInputBatchOrderAction->OrderActionRef));
            dict_set_long(py_action, "request_id", pInputBatchOrderAction->RequestID);
            dict_set_long(py_action, "front_id", pInputBatchOrderAction->FrontID);
            dict_set_long(py_action, "session_id", pInputBatchOrderAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pInputBatchOrderAction->ExchangeID, sizeof(pInputBatchOrderAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_action, "user_id",
                pInputBatchOrderAction->UserID, sizeof(pInputBatchOrderAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_action, "invest_unit_id",
                pInputBatchOrderAction->InvestUnitID, sizeof(pInputBatchOrderAction->InvestUnitID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_batch_order_action", "(OOii)",
            py_action, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 批量报单操作错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnBatchOrderAction(CThostFtdcBatchOrderActionField* pBatchOrderAction,
                                                       CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pBatchOrderAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pBatchOrderAction->BrokerID, sizeof(pBatchOrderAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pBatchOrderAction->InvestorID, sizeof(pBatchOrderAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "order_action_ref",
                pBatchOrderAction->OrderActionRef, sizeof(pBatchOrderAction->OrderActionRef));
            dict_set_long(py_action, "request_id", pBatchOrderAction->RequestID);
            dict_set_long(py_action, "front_id", pBatchOrderAction->FrontID);
            dict_set_long(py_action, "session_id", pBatchOrderAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pBatchOrderAction->ExchangeID, sizeof(pBatchOrderAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_action, "action_date",
                pBatchOrderAction->ActionDate, sizeof(pBatchOrderAction->ActionDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_action, "action_time",
                pBatchOrderAction->ActionTime, sizeof(pBatchOrderAction->ActionTime), GlobalStringPools::Times);
            dict_set_string(py_action, "action_local_id",
                pBatchOrderAction->ActionLocalID, sizeof(pBatchOrderAction->ActionLocalID));
            dict_set_string(py_action, "order_action_status",
                pBatchOrderAction->OrderActionStatus, sizeof(pBatchOrderAction->OrderActionStatus));
            dict_set_pooled_string(py_action, "user_id",
                pBatchOrderAction->UserID, sizeof(pBatchOrderAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_action, "status_msg",
                pBatchOrderAction->StatusMsg, sizeof(pBatchOrderAction->StatusMsg));
            dict_set_string(py_action, "invest_unit_id",
                pBatchOrderAction->InvestUnitID, sizeof(pBatchOrderAction->InvestUnitID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_batch_order_action", "(OO)",
            py_action, py_rsp_info);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // =============================================================================
    // 第3批回调方法 (期权自对冲+组合合约)
    // =============================================================================

    /**
     * @brief 期权自对冲录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspOptionSelfCloseInsert(CThostFtdcInputOptionSelfCloseField* pInputOptionSelfClose,
                                                          CThostFtdcRspInfoField* pRspInfo,
                                                          int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_option = Py_None;
        Py_INCREF(Py_None);
        if (pInputOptionSelfClose) {
            py_option = PyDict_New();
            dict_set_pooled_string(py_option, "broker_id",
                pInputOptionSelfClose->BrokerID, sizeof(pInputOptionSelfClose->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_option, "investor_id",
                pInputOptionSelfClose->InvestorID, sizeof(pInputOptionSelfClose->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_option, "option_self_close_ref",
                pInputOptionSelfClose->OptionSelfCloseRef, sizeof(pInputOptionSelfClose->OptionSelfCloseRef));
            dict_set_pooled_string(py_option, "user_id",
                pInputOptionSelfClose->UserID, sizeof(pInputOptionSelfClose->UserID), GlobalStringPools::Users);
            dict_set_long(py_option, "volume", pInputOptionSelfClose->Volume);
            dict_set_long(py_option, "request_id", pInputOptionSelfClose->RequestID);
            dict_set_string(py_option, "business_unit",
                pInputOptionSelfClose->BusinessUnit, sizeof(pInputOptionSelfClose->BusinessUnit));
            dict_set_string(py_option, "hedge_flag",
                pInputOptionSelfClose->HedgeFlag, sizeof(pInputOptionSelfClose->HedgeFlag));
            dict_set_string(py_option, "opt_self_close_flag",
                pInputOptionSelfClose->OptSelfCloseFlag, sizeof(pInputOptionSelfClose->OptSelfCloseFlag));
            dict_set_pooled_string(py_option, "exchange_id",
                pInputOptionSelfClose->ExchangeID, sizeof(pInputOptionSelfClose->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_option, "invest_unit_id",
                pInputOptionSelfClose->InvestUnitID, sizeof(pInputOptionSelfClose->InvestUnitID));
            dict_set_string(py_option, "account_id",
                pInputOptionSelfClose->AccountID, sizeof(pInputOptionSelfClose->AccountID));
            dict_set_string(py_option, "currency_id",
                pInputOptionSelfClose->CurrencyID, sizeof(pInputOptionSelfClose->CurrencyID));
            dict_set_string(py_option, "client_id",
                pInputOptionSelfClose->ClientID, sizeof(pInputOptionSelfClose->ClientID));
            dict_set_pooled_string(py_option, "instrument_id",
                pInputOptionSelfClose->InstrumentID, sizeof(pInputOptionSelfClose->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_option_self_close_insert", "(OOii)",
            py_option, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_option);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 期权自对冲操作请求响应
     */
    FORCE_INLINE_MEMBER void OnRspOptionSelfCloseAction(CThostFtdcInputOptionSelfCloseActionField* pInputOptionSelfCloseAction,
                                                          CThostFtdcRspInfoField* pRspInfo,
                                                          int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputOptionSelfCloseAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pInputOptionSelfCloseAction->BrokerID, sizeof(pInputOptionSelfCloseAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pInputOptionSelfCloseAction->InvestorID, sizeof(pInputOptionSelfCloseAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "option_self_close_action_ref",
                pInputOptionSelfCloseAction->OptionSelfCloseActionRef, sizeof(pInputOptionSelfCloseAction->OptionSelfCloseActionRef));
            dict_set_string(py_action, "option_self_close_ref",
                pInputOptionSelfCloseAction->OptionSelfCloseRef, sizeof(pInputOptionSelfCloseAction->OptionSelfCloseRef));
            dict_set_long(py_action, "request_id", pInputOptionSelfCloseAction->RequestID);
            dict_set_long(py_action, "front_id", pInputOptionSelfCloseAction->FrontID);
            dict_set_long(py_action, "session_id", pInputOptionSelfCloseAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pInputOptionSelfCloseAction->ExchangeID, sizeof(pInputOptionSelfCloseAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_action, "option_self_close_sys_id",
                pInputOptionSelfCloseAction->OptionSelfCloseSysID, sizeof(pInputOptionSelfCloseAction->OptionSelfCloseSysID));
            dict_set_string(py_action, "action_flag",
                pInputOptionSelfCloseAction->ActionFlag, sizeof(pInputOptionSelfCloseAction->ActionFlag));
            dict_set_pooled_string(py_action, "user_id",
                pInputOptionSelfCloseAction->UserID, sizeof(pInputOptionSelfCloseAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_action, "invest_unit_id",
                pInputOptionSelfCloseAction->InvestUnitID, sizeof(pInputOptionSelfCloseAction->InvestUnitID));
            dict_set_pooled_string(py_action, "instrument_id",
                pInputOptionSelfCloseAction->InstrumentID, sizeof(pInputOptionSelfCloseAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_option_self_close_action", "(OOii)",
            py_action, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 期权自对冲通知
     */
    FORCE_INLINE_MEMBER void OnRtnOptionSelfClose(CThostFtdcOptionSelfCloseField* pOptionSelfClose) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_option = Py_None;
        Py_INCREF(Py_None);
        if (pOptionSelfClose) {
            py_option = PyDict_New();
            dict_set_pooled_string(py_option, "broker_id",
                pOptionSelfClose->BrokerID, sizeof(pOptionSelfClose->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_option, "investor_id",
                pOptionSelfClose->InvestorID, sizeof(pOptionSelfClose->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_option, "option_self_close_ref",
                pOptionSelfClose->OptionSelfCloseRef, sizeof(pOptionSelfClose->OptionSelfCloseRef));
            dict_set_pooled_string(py_option, "user_id",
                pOptionSelfClose->UserID, sizeof(pOptionSelfClose->UserID), GlobalStringPools::Users);
            dict_set_long(py_option, "volume", pOptionSelfClose->Volume);
            dict_set_long(py_option, "request_id", pOptionSelfClose->RequestID);
            dict_set_string(py_option, "business_unit",
                pOptionSelfClose->BusinessUnit, sizeof(pOptionSelfClose->BusinessUnit));
            dict_set_string(py_option, "hedge_flag",
                pOptionSelfClose->HedgeFlag, sizeof(pOptionSelfClose->HedgeFlag));
            dict_set_string(py_option, "opt_self_close_flag",
                pOptionSelfClose->OptSelfCloseFlag, sizeof(pOptionSelfClose->OptSelfCloseFlag));
            dict_set_string(py_option, "option_self_close_local_id",
                pOptionSelfClose->OptionSelfCloseLocalID, sizeof(pOptionSelfClose->OptionSelfCloseLocalID));
            dict_set_pooled_string(py_option, "exchange_id",
                pOptionSelfClose->ExchangeID, sizeof(pOptionSelfClose->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_option, "trading_day",
                pOptionSelfClose->TradingDay, sizeof(pOptionSelfClose->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_option, "settlement_id", pOptionSelfClose->SettlementID);
            dict_set_string(py_option, "option_self_close_sys_id",
                pOptionSelfClose->OptionSelfCloseSysID, sizeof(pOptionSelfClose->OptionSelfCloseSysID));
            dict_set_pooled_string(py_option, "insert_date",
                pOptionSelfClose->InsertDate, sizeof(pOptionSelfClose->InsertDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_option, "insert_time",
                pOptionSelfClose->InsertTime, sizeof(pOptionSelfClose->InsertTime), GlobalStringPools::Times);
            dict_set_pooled_string(py_option, "cancel_time",
                pOptionSelfClose->CancelTime, sizeof(pOptionSelfClose->CancelTime), GlobalStringPools::Times);
            dict_set_string(py_option, "exec_result",
                pOptionSelfClose->ExecResult, sizeof(pOptionSelfClose->ExecResult));
            dict_set_long(py_option, "sequence_no", pOptionSelfClose->SequenceNo);
            dict_set_long(py_option, "front_id", pOptionSelfClose->FrontID);
            dict_set_long(py_option, "session_id", pOptionSelfClose->SessionID);
            dict_set_string(py_option, "user_product_info",
                pOptionSelfClose->UserProductInfo, sizeof(pOptionSelfClose->UserProductInfo));
            dict_set_string(py_option, "status_msg",
                pOptionSelfClose->StatusMsg, sizeof(pOptionSelfClose->StatusMsg));
            dict_set_pooled_string(py_option, "active_user_id",
                pOptionSelfClose->ActiveUserID, sizeof(pOptionSelfClose->ActiveUserID), GlobalStringPools::Users);
            dict_set_long(py_option, "broker_option_self_close_seq", pOptionSelfClose->BrokerOptionSelfCloseSeq);
            dict_set_string(py_option, "branch_id",
                pOptionSelfClose->BranchID, sizeof(pOptionSelfClose->BranchID));
            dict_set_string(py_option, "invest_unit_id",
                pOptionSelfClose->InvestUnitID, sizeof(pOptionSelfClose->InvestUnitID));
            dict_set_string(py_option, "account_id",
                pOptionSelfClose->AccountID, sizeof(pOptionSelfClose->AccountID));
            dict_set_string(py_option, "currency_id",
                pOptionSelfClose->CurrencyID, sizeof(pOptionSelfClose->CurrencyID));
            dict_set_pooled_string(py_option, "instrument_id",
                pOptionSelfClose->InstrumentID, sizeof(pOptionSelfClose->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_option_self_close", "(O)", py_option);

        Py_XDECREF(py_option);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 期权自对冲错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnOptionSelfCloseInsert(CThostFtdcInputOptionSelfCloseField* pInputOptionSelfClose,
                                                             CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_option = Py_None;
        Py_INCREF(Py_None);
        if (pInputOptionSelfClose) {
            py_option = PyDict_New();
            dict_set_pooled_string(py_option, "broker_id",
                pInputOptionSelfClose->BrokerID, sizeof(pInputOptionSelfClose->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_option, "investor_id",
                pInputOptionSelfClose->InvestorID, sizeof(pInputOptionSelfClose->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_option, "option_self_close_ref",
                pInputOptionSelfClose->OptionSelfCloseRef, sizeof(pInputOptionSelfClose->OptionSelfCloseRef));
            dict_set_pooled_string(py_option, "user_id",
                pInputOptionSelfClose->UserID, sizeof(pInputOptionSelfClose->UserID), GlobalStringPools::Users);
            dict_set_long(py_option, "volume", pInputOptionSelfClose->Volume);
            dict_set_pooled_string(py_option, "exchange_id",
                pInputOptionSelfClose->ExchangeID, sizeof(pInputOptionSelfClose->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_option, "instrument_id",
                pInputOptionSelfClose->InstrumentID, sizeof(pInputOptionSelfClose->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_option_self_close_insert", "(OO)",
            py_option, py_rsp_info);

        Py_XDECREF(py_option);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 期权自对冲操作错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnOptionSelfCloseAction(CThostFtdcInputOptionSelfCloseActionField* pInputOptionSelfCloseAction,
                                                             CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_action = Py_None;
        Py_INCREF(Py_None);
        if (pInputOptionSelfCloseAction) {
            py_action = PyDict_New();
            dict_set_pooled_string(py_action, "broker_id",
                pInputOptionSelfCloseAction->BrokerID, sizeof(pInputOptionSelfCloseAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_action, "investor_id",
                pInputOptionSelfCloseAction->InvestorID, sizeof(pInputOptionSelfCloseAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_action, "option_self_close_ref",
                pInputOptionSelfCloseAction->OptionSelfCloseRef, sizeof(pInputOptionSelfCloseAction->OptionSelfCloseRef));
            dict_set_long(py_action, "front_id", pInputOptionSelfCloseAction->FrontID);
            dict_set_long(py_action, "session_id", pInputOptionSelfCloseAction->SessionID);
            dict_set_pooled_string(py_action, "exchange_id",
                pInputOptionSelfCloseAction->ExchangeID, sizeof(pInputOptionSelfCloseAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_action, "option_self_close_sys_id",
                pInputOptionSelfCloseAction->OptionSelfCloseSysID, sizeof(pInputOptionSelfCloseAction->OptionSelfCloseSysID));
            dict_set_pooled_string(py_action, "instrument_id",
                pInputOptionSelfCloseAction->InstrumentID, sizeof(pInputOptionSelfCloseAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_option_self_close_action", "(OO)",
            py_action, py_rsp_info);

        Py_XDECREF(py_action);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询期权自对冲响应
     */
    FORCE_INLINE_MEMBER void OnRspQryOptionSelfClose(CThostFtdcOptionSelfCloseField* pOptionSelfClose,
                                                        CThostFtdcRspInfoField* pRspInfo,
                                                        int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_option = Py_None;
        Py_INCREF(Py_None);
        if (pOptionSelfClose) {
            py_option = PyDict_New();
            dict_set_pooled_string(py_option, "broker_id",
                pOptionSelfClose->BrokerID, sizeof(pOptionSelfClose->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_option, "investor_id",
                pOptionSelfClose->InvestorID, sizeof(pOptionSelfClose->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_option, "option_self_close_ref",
                pOptionSelfClose->OptionSelfCloseRef, sizeof(pOptionSelfClose->OptionSelfCloseRef));
            dict_set_long(py_option, "volume", pOptionSelfClose->Volume);
            dict_set_long(py_option, "request_id", pOptionSelfClose->RequestID);
            dict_set_string(py_option, "hedge_flag",
                pOptionSelfClose->HedgeFlag, sizeof(pOptionSelfClose->HedgeFlag));
            dict_set_string(py_option, "opt_self_close_flag",
                pOptionSelfClose->OptSelfCloseFlag, sizeof(pOptionSelfClose->OptSelfCloseFlag));
            dict_set_pooled_string(py_option, "exchange_id",
                pOptionSelfClose->ExchangeID, sizeof(pOptionSelfClose->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_option, "trading_day",
                pOptionSelfClose->TradingDay, sizeof(pOptionSelfClose->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_option, "settlement_id", pOptionSelfClose->SettlementID);
            dict_set_string(py_option, "option_self_close_sys_id",
                pOptionSelfClose->OptionSelfCloseSysID, sizeof(pOptionSelfClose->OptionSelfCloseSysID));
            dict_set_pooled_string(py_option, "insert_date",
                pOptionSelfClose->InsertDate, sizeof(pOptionSelfClose->InsertDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_option, "insert_time",
                pOptionSelfClose->InsertTime, sizeof(pOptionSelfClose->InsertTime), GlobalStringPools::Times);
            dict_set_string(py_option, "exec_result",
                pOptionSelfClose->ExecResult, sizeof(pOptionSelfClose->ExecResult));
            dict_set_long(py_option, "front_id", pOptionSelfClose->FrontID);
            dict_set_long(py_option, "session_id", pOptionSelfClose->SessionID);
            dict_set_string(py_option, "status_msg",
                pOptionSelfClose->StatusMsg, sizeof(pOptionSelfClose->StatusMsg));
            dict_set_pooled_string(py_option, "instrument_id",
                pOptionSelfClose->InstrumentID, sizeof(pOptionSelfClose->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_option_self_close", "(OOii)",
            py_option, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_option);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 申请组合录入请求响应
     */
    FORCE_INLINE_MEMBER void OnRspCombActionInsert(CThostFtdcInputCombActionField* pInputCombAction,
                                                     CThostFtdcRspInfoField* pRspInfo,
                                                     int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_comb = Py_None;
        Py_INCREF(Py_None);
        if (pInputCombAction) {
            py_comb = PyDict_New();
            dict_set_pooled_string(py_comb, "broker_id",
                pInputCombAction->BrokerID, sizeof(pInputCombAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_comb, "investor_id",
                pInputCombAction->InvestorID, sizeof(pInputCombAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_comb, "comb_action_ref",
                pInputCombAction->CombActionRef, sizeof(pInputCombAction->CombActionRef));
            dict_set_pooled_string(py_comb, "user_id",
                pInputCombAction->UserID, sizeof(pInputCombAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_comb, "direction",
                pInputCombAction->Direction, sizeof(pInputCombAction->Direction));
            dict_set_long(py_comb, "volume", pInputCombAction->Volume);
            dict_set_string(py_comb, "comb_direction",
                pInputCombAction->CombDirection, sizeof(pInputCombAction->CombDirection));
            dict_set_string(py_comb, "hedge_flag",
                pInputCombAction->HedgeFlag, sizeof(pInputCombAction->HedgeFlag));
            dict_set_pooled_string(py_comb, "exchange_id",
                pInputCombAction->ExchangeID, sizeof(pInputCombAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_comb, "invest_unit_id",
                pInputCombAction->InvestUnitID, sizeof(pInputCombAction->InvestUnitID));
            dict_set_pooled_string(py_comb, "instrument_id",
                pInputCombAction->InstrumentID, sizeof(pInputCombAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_comb_action_insert", "(OOii)",
            py_comb, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_comb);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 申请组合通知
     */
    FORCE_INLINE_MEMBER void OnRtnCombAction(CThostFtdcCombActionField* pCombAction) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_comb = Py_None;
        Py_INCREF(Py_None);
        if (pCombAction) {
            py_comb = PyDict_New();
            dict_set_pooled_string(py_comb, "broker_id",
                pCombAction->BrokerID, sizeof(pCombAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_comb, "investor_id",
                pCombAction->InvestorID, sizeof(pCombAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_comb, "comb_action_ref",
                pCombAction->CombActionRef, sizeof(pCombAction->CombActionRef));
            dict_set_pooled_string(py_comb, "user_id",
                pCombAction->UserID, sizeof(pCombAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_comb, "direction",
                pCombAction->Direction, sizeof(pCombAction->Direction));
            dict_set_long(py_comb, "volume", pCombAction->Volume);
            dict_set_string(py_comb, "comb_direction",
                pCombAction->CombDirection, sizeof(pCombAction->CombDirection));
            dict_set_string(py_comb, "hedge_flag",
                pCombAction->HedgeFlag, sizeof(pCombAction->HedgeFlag));
            dict_set_string(py_comb, "action_local_id",
                pCombAction->ActionLocalID, sizeof(pCombAction->ActionLocalID));
            dict_set_pooled_string(py_comb, "exchange_id",
                pCombAction->ExchangeID, sizeof(pCombAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_comb, "trading_day",
                pCombAction->TradingDay, sizeof(pCombAction->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_comb, "settlement_id", pCombAction->SettlementID);
            dict_set_long(py_comb, "sequence_no", pCombAction->SequenceNo);
            dict_set_long(py_comb, "front_id", pCombAction->FrontID);
            dict_set_long(py_comb, "session_id", pCombAction->SessionID);
            dict_set_string(py_comb, "user_product_info",
                pCombAction->UserProductInfo, sizeof(pCombAction->UserProductInfo));
            dict_set_string(py_comb, "status_msg",
                pCombAction->StatusMsg, sizeof(pCombAction->StatusMsg));
            dict_set_string(py_comb, "com_trade_id",
                pCombAction->ComTradeID, sizeof(pCombAction->ComTradeID));
            dict_set_string(py_comb, "branch_id",
                pCombAction->BranchID, sizeof(pCombAction->BranchID));
            dict_set_string(py_comb, "invest_unit_id",
                pCombAction->InvestUnitID, sizeof(pCombAction->InvestUnitID));
            dict_set_pooled_string(py_comb, "instrument_id",
                pCombAction->InstrumentID, sizeof(pCombAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_comb_action", "(O)", py_comb);

        Py_XDECREF(py_comb);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 申请组合错误回报
     */
    FORCE_INLINE_MEMBER void OnErrRtnCombActionInsert(CThostFtdcInputCombActionField* pInputCombAction,
                                                         CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_comb = Py_None;
        Py_INCREF(Py_None);
        if (pInputCombAction) {
            py_comb = PyDict_New();
            dict_set_pooled_string(py_comb, "broker_id",
                pInputCombAction->BrokerID, sizeof(pInputCombAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_comb, "investor_id",
                pInputCombAction->InvestorID, sizeof(pInputCombAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_comb, "comb_action_ref",
                pInputCombAction->CombActionRef, sizeof(pInputCombAction->CombActionRef));
            dict_set_pooled_string(py_comb, "user_id",
                pInputCombAction->UserID, sizeof(pInputCombAction->UserID), GlobalStringPools::Users);
            dict_set_string(py_comb, "direction",
                pInputCombAction->Direction, sizeof(pInputCombAction->Direction));
            dict_set_pooled_string(py_comb, "exchange_id",
                pInputCombAction->ExchangeID, sizeof(pInputCombAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_comb, "instrument_id",
                pInputCombAction->InstrumentID, sizeof(pInputCombAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_comb_action_insert", "(OO)",
            py_comb, py_rsp_info);

        Py_XDECREF(py_comb);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询申请组合响应
     */
    FORCE_INLINE_MEMBER void OnRspQryCombAction(CThostFtdcCombActionField* pCombAction,
                                                  CThostFtdcRspInfoField* pRspInfo,
                                                  int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_comb = Py_None;
        Py_INCREF(Py_None);
        if (pCombAction) {
            py_comb = PyDict_New();
            dict_set_pooled_string(py_comb, "broker_id",
                pCombAction->BrokerID, sizeof(pCombAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_comb, "investor_id",
                pCombAction->InvestorID, sizeof(pCombAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_comb, "comb_action_ref",
                pCombAction->CombActionRef, sizeof(pCombAction->CombActionRef));
            dict_set_string(py_comb, "direction",
                pCombAction->Direction, sizeof(pCombAction->Direction));
            dict_set_long(py_comb, "volume", pCombAction->Volume);
            dict_set_string(py_comb, "comb_direction",
                pCombAction->CombDirection, sizeof(pCombAction->CombDirection));
            dict_set_pooled_string(py_comb, "exchange_id",
                pCombAction->ExchangeID, sizeof(pCombAction->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_comb, "action_local_id",
                pCombAction->ActionLocalID, sizeof(pCombAction->ActionLocalID));
            dict_set_pooled_string(py_comb, "trading_day",
                pCombAction->TradingDay, sizeof(pCombAction->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_comb, "settlement_id", pCombAction->SettlementID);
            dict_set_long(py_comb, "front_id", pCombAction->FrontID);
            dict_set_long(py_comb, "session_id", pCombAction->SessionID);
            dict_set_string(py_comb, "status_msg",
                pCombAction->StatusMsg, sizeof(pCombAction->StatusMsg));
            dict_set_pooled_string(py_comb, "instrument_id",
                pCombAction->InstrumentID, sizeof(pCombAction->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_comb_action", "(OOii)",
            py_comb, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_comb);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询组合合约安全系数响应
     */
    FORCE_INLINE_MEMBER void OnRspQryCombInstrumentGuard(CThostFtdcCombInstrumentGuardField* pCombInstrumentGuard,
                                                           CThostFtdcRspInfoField* pRspInfo,
                                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_guard = Py_None;
        Py_INCREF(Py_None);
        if (pCombInstrumentGuard) {
            py_guard = PyDict_New();
            dict_set_pooled_string(py_guard, "broker_id",
                pCombInstrumentGuard->BrokerID, sizeof(pCombInstrumentGuard->BrokerID), GlobalStringPools::Brokers);
            dict_set_long(py_guard, "guarant_ratio", pCombInstrumentGuard->GuarantRatio);
            dict_set_pooled_string(py_guard, "exchange_id",
                pCombInstrumentGuard->ExchangeID, sizeof(pCombInstrumentGuard->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_guard, "instrument_id",
                pCombInstrumentGuard->InstrumentID, sizeof(pCombInstrumentGuard->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_comb_instrument_guard", "(OOii)",
            py_guard, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_guard);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询投资者持仓组合明细响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInvestorPositionCombineDetail(CThostFtdcInvestorPositionCombineDetailField* pInvestorPositionCombineDetail,
                                                                     CThostFtdcRspInfoField* pRspInfo,
                                                                     int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_detail = Py_None;
        Py_INCREF(Py_None);
        if (pInvestorPositionCombineDetail) {
            py_detail = PyDict_New();
            dict_set_pooled_string(py_detail, "trading_day",
                pInvestorPositionCombineDetail->TradingDay, sizeof(pInvestorPositionCombineDetail->TradingDay), GlobalStringPools::Dates);
            dict_set_pooled_string(py_detail, "open_date",
                pInvestorPositionCombineDetail->OpenDate, sizeof(pInvestorPositionCombineDetail->OpenDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_detail, "exchange_id",
                pInvestorPositionCombineDetail->ExchangeID, sizeof(pInvestorPositionCombineDetail->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_long(py_detail, "settlement_id", pInvestorPositionCombineDetail->SettlementID);
            dict_set_pooled_string(py_detail, "broker_id",
                pInvestorPositionCombineDetail->BrokerID, sizeof(pInvestorPositionCombineDetail->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_detail, "investor_id",
                pInvestorPositionCombineDetail->InvestorID, sizeof(pInvestorPositionCombineDetail->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_detail, "com_trade_id",
                pInvestorPositionCombineDetail->ComTradeID, sizeof(pInvestorPositionCombineDetail->ComTradeID));
            dict_set_string(py_detail, "trade_id",
                pInvestorPositionCombineDetail->TradeID, sizeof(pInvestorPositionCombineDetail->TradeID));
            dict_set_string(py_detail, "hedge_flag",
                pInvestorPositionCombineDetail->HedgeFlag, sizeof(pInvestorPositionCombineDetail->HedgeFlag));
            dict_set_string(py_detail, "direction",
                pInvestorPositionCombineDetail->Direction, sizeof(pInvestorPositionCombineDetail->Direction));
            dict_set_long(py_detail, "total_amt", pInvestorPositionCombineDetail->TotalAmt);
            dict_set_double(py_detail, "margin", pInvestorPositionCombineDetail->Margin);
            dict_set_double(py_detail, "exch_margin", pInvestorPositionCombineDetail->ExchMargin);
            dict_set_double(py_detail, "margin_rate_by_money", pInvestorPositionCombineDetail->MarginRateByMoney);
            dict_set_double(py_detail, "margin_rate_by_volume", pInvestorPositionCombineDetail->MarginRateByVolume);
            dict_set_string(py_detail, "leg_id",
                pInvestorPositionCombineDetail->LegID, sizeof(pInvestorPositionCombineDetail->LegID));
            dict_set_long(py_detail, "leg_multiple", pInvestorPositionCombineDetail->LegMultiple);
            dict_set_string(py_detail, "trade_group_id",
                pInvestorPositionCombineDetail->TradeGroupID, sizeof(pInvestorPositionCombineDetail->TradeGroupID));
            dict_set_string(py_detail, "invest_unit_id",
                pInvestorPositionCombineDetail->InvestUnitID, sizeof(pInvestorPositionCombineDetail->InvestUnitID));
            dict_set_pooled_string(py_detail, "instrument_id",
                pInvestorPositionCombineDetail->InstrumentID, sizeof(pInvestorPositionCombineDetail->InstrumentID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_detail, "comb_instrument_id",
                pInvestorPositionCombineDetail->CombInstrumentID, sizeof(pInvestorPositionCombineDetail->CombInstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_investor_position_combine_detail", "(OOii)",
            py_detail, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_detail);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询仓单折抵信息响应
     */
    FORCE_INLINE_MEMBER void OnRspQryEWarrantOffset(CThostFtdcEWarrantOffsetField* pEWarrantOffset,
                                                      CThostFtdcRspInfoField* pRspInfo,
                                                      int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_offset = Py_None;
        Py_INCREF(Py_None);
        if (pEWarrantOffset) {
            py_offset = PyDict_New();
            dict_set_pooled_string(py_offset, "trading_day",
                pEWarrantOffset->TradingDay, sizeof(pEWarrantOffset->TradingDay), GlobalStringPools::Dates);
            dict_set_pooled_string(py_offset, "broker_id",
                pEWarrantOffset->BrokerID, sizeof(pEWarrantOffset->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_offset, "investor_id",
                pEWarrantOffset->InvestorID, sizeof(pEWarrantOffset->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_offset, "exchange_id",
                pEWarrantOffset->ExchangeID, sizeof(pEWarrantOffset->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_offset, "direction",
                pEWarrantOffset->Direction, sizeof(pEWarrantOffset->Direction));
            dict_set_string(py_offset, "hedge_flag",
                pEWarrantOffset->HedgeFlag, sizeof(pEWarrantOffset->HedgeFlag));
            dict_set_long(py_offset, "volume", pEWarrantOffset->Volume);
            dict_set_string(py_offset, "invest_unit_id",
                pEWarrantOffset->InvestUnitID, sizeof(pEWarrantOffset->InvestUnitID));
            dict_set_pooled_string(py_offset, "instrument_id",
                pEWarrantOffset->InstrumentID, sizeof(pEWarrantOffset->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_e_warrant_offset", "(OOii)",
            py_offset, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_offset);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询投资者品种/跨品种保证金响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInvestorProductGroupMargin(CThostFtdcInvestorProductGroupMarginField* pInvestorProductGroupMargin,
                                                                  CThostFtdcRspInfoField* pRspInfo,
                                                                  int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_margin = Py_None;
        Py_INCREF(Py_None);
        if (pInvestorProductGroupMargin) {
            py_margin = PyDict_New();
            dict_set_pooled_string(py_margin, "broker_id",
                pInvestorProductGroupMargin->BrokerID, sizeof(pInvestorProductGroupMargin->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_margin, "investor_id",
                pInvestorProductGroupMargin->InvestorID, sizeof(pInvestorProductGroupMargin->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_margin, "trading_day",
                pInvestorProductGroupMargin->TradingDay, sizeof(pInvestorProductGroupMargin->TradingDay), GlobalStringPools::Dates);
            dict_set_long(py_margin, "settlement_id", pInvestorProductGroupMargin->SettlementID);
            dict_set_double(py_margin, "frozen_margin", pInvestorProductGroupMargin->FrozenMargin);
            dict_set_double(py_margin, "long_frozen_margin", pInvestorProductGroupMargin->LongFrozenMargin);
            dict_set_double(py_margin, "short_frozen_margin", pInvestorProductGroupMargin->ShortFrozenMargin);
            dict_set_double(py_margin, "use_margin", pInvestorProductGroupMargin->UseMargin);
            dict_set_double(py_margin, "long_use_margin", pInvestorProductGroupMargin->LongUseMargin);
            dict_set_double(py_margin, "short_use_margin", pInvestorProductGroupMargin->ShortUseMargin);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_investor_product_group_margin", "(OOii)",
            py_margin, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_margin);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询转账流水响应
     */
    FORCE_INLINE_MEMBER void OnRspQryTransferSerial(CThostFtdcTransferSerialField* pTransferSerial,
                                                      CThostFtdcRspInfoField* pRspInfo,
                                                      int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_serial = Py_None;
        Py_INCREF(Py_None);
        if (pTransferSerial) {
            py_serial = PyDict_New();
            dict_set_long(py_serial, "plate_serial", pTransferSerial->PlateSerial);
            dict_set_pooled_string(py_serial, "trade_date",
                pTransferSerial->TradeDate, sizeof(pTransferSerial->TradeDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_serial, "trading_day",
                pTransferSerial->TradingDay, sizeof(pTransferSerial->TradingDay), GlobalStringPools::Dates);
            dict_set_pooled_string(py_serial, "trade_time",
                pTransferSerial->TradeTime, sizeof(pTransferSerial->TradeTime), GlobalStringPools::Times);
            dict_set_string(py_serial, "trade_code",
                pTransferSerial->TradeCode, sizeof(pTransferSerial->TradeCode));
            dict_set_long(py_serial, "session_id", pTransferSerial->SessionID);
            dict_set_string(py_serial, "bank_id",
                pTransferSerial->BankID, sizeof(pTransferSerial->BankID));
            dict_set_string(py_serial, "bank_account",
                pTransferSerial->BankAccount, sizeof(pTransferSerial->BankAccount));
            dict_set_pooled_string(py_serial, "broker_id",
                pTransferSerial->BrokerID, sizeof(pTransferSerial->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_serial, "investor_id",
                pTransferSerial->InvestorID, sizeof(pTransferSerial->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_serial, "account_id",
                pTransferSerial->AccountID, sizeof(pTransferSerial->AccountID));
            dict_set_string(py_serial, "currency_id",
                pTransferSerial->CurrencyID, sizeof(pTransferSerial->CurrencyID));
            dict_set_double(py_serial, "trade_amount", pTransferSerial->TradeAmount);
            dict_set_double(py_serial, "cust_fee", pTransferSerial->CustFee);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_transfer_serial", "(OOii)",
            py_serial, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_serial);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询银期签约响应
     */
    FORCE_INLINE_MEMBER void OnRspQryAccountregister(CThostFtdcAccountregisterField* pAccountregister,
                                                       CThostFtdcRspInfoField* pRspInfo,
                                                       int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_register = Py_None;
        Py_INCREF(Py_None);
        if (pAccountregister) {
            py_register = PyDict_New();
            dict_set_pooled_string(py_register, "trade_day",
                pAccountregister->TradeDay, sizeof(pAccountregister->TradeDay), GlobalStringPools::Dates);
            dict_set_string(py_register, "bank_id",
                pAccountregister->BankID, sizeof(pAccountregister->BankID));
            dict_set_string(py_register, "bank_account",
                pAccountregister->BankAccount, sizeof(pAccountregister->BankAccount));
            dict_set_pooled_string(py_register, "broker_id",
                pAccountregister->BrokerID, sizeof(pAccountregister->BrokerID), GlobalStringPools::Brokers);
            dict_set_string(py_register, "account_id",
                pAccountregister->AccountID, sizeof(pAccountregister->AccountID));
            dict_set_string(py_register, "customer_name",
                pAccountregister->CustomerName, sizeof(pAccountregister->CustomerName));
            dict_set_string(py_register, "currency_id",
                pAccountregister->CurrencyID, sizeof(pAccountregister->CurrencyID));
            dict_set_pooled_string(py_register, "reg_date",
                pAccountregister->RegDate, sizeof(pAccountregister->RegDate), GlobalStringPools::Dates);
            dict_set_pooled_string(py_register, "out_date",
                pAccountregister->OutDate, sizeof(pAccountregister->OutDate), GlobalStringPools::Dates);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_accountregister", "(OOii)",
            py_register, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_register);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询交易所保证金率响应
     */
    FORCE_INLINE_MEMBER void OnRspQryExchangeMarginRate(CThostFtdcExchangeMarginRateField* pExchangeMarginRate,
                                                         CThostFtdcRspInfoField* pRspInfo,
                                                         int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_margin = Py_None;
        Py_INCREF(Py_None);
        if (pExchangeMarginRate) {
            py_margin = PyDict_New();
            dict_set_pooled_string(py_margin, "broker_id",
                pExchangeMarginRate->BrokerID, sizeof(pExchangeMarginRate->BrokerID), GlobalStringPools::Brokers);
            dict_set_string(py_margin, "hedge_flag",
                pExchangeMarginRate->HedgeFlag, sizeof(pExchangeMarginRate->HedgeFlag));
            dict_set_double(py_margin, "long_margin_ratio_by_money", pExchangeMarginRate->LongMarginRatioByMoney);
            dict_set_double(py_margin, "long_margin_ratio_by_volume", pExchangeMarginRate->LongMarginRatioByVolume);
            dict_set_double(py_margin, "short_margin_ratio_by_money", pExchangeMarginRate->ShortMarginRatioByMoney);
            dict_set_double(py_margin, "short_margin_ratio_by_volume", pExchangeMarginRate->ShortMarginRatioByVolume);
            dict_set_pooled_string(py_margin, "exchange_id",
                pExchangeMarginRate->ExchangeID, sizeof(pExchangeMarginRate->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_margin, "instrument_id",
                pExchangeMarginRate->InstrumentID, sizeof(pExchangeMarginRate->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_exchange_margin_rate", "(OOii)",
            py_margin, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_margin);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询交易所保证金率调整响应
     */
    FORCE_INLINE_MEMBER void OnRspQryExchangeMarginRateAdjust(CThostFtdcExchangeMarginRateAdjustField* pExchangeMarginRateAdjust,
                                                                CThostFtdcRspInfoField* pRspInfo,
                                                                int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_adjust = Py_None;
        Py_INCREF(Py_None);
        if (pExchangeMarginRateAdjust) {
            py_adjust = PyDict_New();
            dict_set_pooled_string(py_adjust, "broker_id",
                pExchangeMarginRateAdjust->BrokerID, sizeof(pExchangeMarginRateAdjust->BrokerID), GlobalStringPools::Brokers);
            dict_set_string(py_adjust, "hedge_flag",
                pExchangeMarginRateAdjust->HedgeFlag, sizeof(pExchangeMarginRateAdjust->HedgeFlag));
            dict_set_double(py_adjust, "long_margin_ratio_by_money", pExchangeMarginRateAdjust->LongMarginRatioByMoney);
            dict_set_double(py_adjust, "long_margin_ratio_by_volume", pExchangeMarginRateAdjust->LongMarginRatioByVolume);
            dict_set_double(py_adjust, "short_margin_ratio_by_money", pExchangeMarginRateAdjust->ShortMarginRatioByMoney);
            dict_set_double(py_adjust, "short_margin_ratio_by_volume", pExchangeMarginRateAdjust->ShortMarginRatioByVolume);
            dict_set_pooled_string(py_adjust, "exchange_id",
                pExchangeMarginRateAdjust->ExchangeID, sizeof(pExchangeMarginRateAdjust->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_adjust, "instrument_id",
                pExchangeMarginRateAdjust->InstrumentID, sizeof(pExchangeMarginRateAdjust->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_exchange_margin_rate_adjust", "(OOii)",
            py_adjust, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_adjust);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询汇率响应
     */
    FORCE_INLINE_MEMBER void OnRspQryExchangeRate(CThostFtdcExchangeRateField* pExchangeRate,
                                                    CThostFtdcRspInfoField* pRspInfo,
                                                    int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rate = Py_None;
        Py_INCREF(Py_None);
        if (pExchangeRate) {
            py_rate = PyDict_New();
            dict_set_pooled_string(py_rate, "broker_id",
                pExchangeRate->BrokerID, sizeof(pExchangeRate->BrokerID), GlobalStringPools::Brokers);
            dict_set_string(py_rate, "from_currency_id",
                pExchangeRate->FromCurrencyID, sizeof(pExchangeRate->FromCurrencyID));
            dict_set_long(py_rate, "from_currency_unit", pExchangeRate->FromCurrencyUnit);
            dict_set_string(py_rate, "to_currency_id",
                pExchangeRate->ToCurrencyID, sizeof(pExchangeRate->ToCurrencyID));
            dict_set_double(py_rate, "exchange_rate", pExchangeRate->ExchangeRate);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_exchange_rate", "(OOii)",
            py_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    /**
     * @brief 查询投资单元响应
     */
    FORCE_INLINE_MEMBER void OnRspQryInvestUnit(CThostFtdcInvestUnitField* pInvestUnit,
                                                  CThostFtdcRspInfoField* pRspInfo,
                                                  int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_unit = Py_None;
        Py_INCREF(Py_None);
        if (pInvestUnit) {
            py_unit = PyDict_New();
            dict_set_pooled_string(py_unit, "broker_id",
                pInvestUnit->BrokerID, sizeof(pInvestUnit->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_unit, "investor_id",
                pInvestUnit->InvestorID, sizeof(pInvestUnit->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_unit, "invest_unit_id",
                pInvestUnit->InvestUnitID, sizeof(pInvestUnit->InvestUnitID));
            dict_set_string(py_unit, "investor_unit_name",
                pInvestUnit->InvestorUnitName, sizeof(pInvestUnit->InvestorUnitName));
            dict_set_string(py_unit, "investor_group_id",
                pInvestUnit->InvestorGroupID, sizeof(pInvestUnit->InvestorGroupID));
            dict_set_string(py_unit, "comm_model_id",
                pInvestUnit->CommModelID, sizeof(pInvestUnit->CommModelID));
            dict_set_string(py_unit, "margin_model_id",
                pInvestUnit->MarginModelID, sizeof(pInvestUnit->MarginModelID));
            dict_set_string(py_unit, "account_id",
                pInvestUnit->AccountID, sizeof(pInvestUnit->AccountID));
            dict_set_string(py_unit, "currency_id",
                pInvestUnit->CurrencyID, sizeof(pInvestUnit->CurrencyID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);

        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_invest_unit", "(OOii)",
            py_unit, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_unit);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }



    // =============================================================================
    // 第4批：二级代理+期权成本+其他查询 (13个方法)
    // =============================================================================

    FORCE_INLINE_MEMBER void OnRspQrySecAgentACIDMap(CThostFtdcSecAgentACIDMapField* pSecAgentACIDMap,
                                                       CThostFtdcRspInfoField* pRspInfo,
                                                       int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_map = Py_None;
        Py_INCREF(Py_None);
        if (pSecAgentACIDMap) {
            py_map = PyDict_New();
            dict_set_pooled_string(py_map, "broker_id",
                pSecAgentACIDMap->BrokerID, sizeof(pSecAgentACIDMap->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_map, "user_id",
                pSecAgentACIDMap->UserID, sizeof(pSecAgentACIDMap->UserID), GlobalStringPools::Users);
            dict_set_string(py_map, "account_id",
                pSecAgentACIDMap->AccountID, sizeof(pSecAgentACIDMap->AccountID));
            dict_set_string(py_map, "currency_id",
                pSecAgentACIDMap->CurrencyID, sizeof(pSecAgentACIDMap->CurrencyID));
            dict_set_string(py_map, "broker_sec_agent_id",
                pSecAgentACIDMap->BrokerSecAgentID, sizeof(pSecAgentACIDMap->BrokerSecAgentID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_sec_agent_acid_map", "(OOii)",
            py_map, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_map);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryProductExchRate(CThostFtdcProductExchRateField* pProductExchRate,
                                                      CThostFtdcRspInfoField* pRspInfo,
                                                      int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rate = Py_None;
        Py_INCREF(Py_None);
        if (pProductExchRate) {
            py_rate = PyDict_New();
            dict_set_string(py_rate, "quote_currency_id",
                pProductExchRate->QuoteCurrencyID, sizeof(pProductExchRate->QuoteCurrencyID));
            dict_set_double(py_rate, "exchange_rate", pProductExchRate->ExchangeRate);
            dict_set_pooled_string(py_rate, "exchange_id",
                pProductExchRate->ExchangeID, sizeof(pProductExchRate->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_rate, "product_id",
                pProductExchRate->ProductID, sizeof(pProductExchRate->ProductID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_product_exch_rate", "(OOii)",
            py_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryProductGroup(CThostFtdcProductGroupField* pProductGroup,
                                                   CThostFtdcRspInfoField* pRspInfo,
                                                   int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_group = Py_None;
        Py_INCREF(Py_None);
        if (pProductGroup) {
            py_group = PyDict_New();
            dict_set_pooled_string(py_group, "exchange_id",
                pProductGroup->ExchangeID, sizeof(pProductGroup->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_pooled_string(py_group, "product_id",
                pProductGroup->ProductID, sizeof(pProductGroup->ProductID), GlobalStringPools::Instruments);
            dict_set_pooled_string(py_group, "product_group_id",
                pProductGroup->ProductGroupID, sizeof(pProductGroup->ProductGroupID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_product_group", "(OOii)",
            py_group, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_group);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryMMInstrumentCommissionRate(CThostFtdcMMInstrumentCommissionRateField* pMMInstrumentCommissionRate,
                                                                 CThostFtdcRspInfoField* pRspInfo,
                                                                 int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rate = Py_None;
        Py_INCREF(Py_None);
        if (pMMInstrumentCommissionRate) {
            py_rate = PyDict_New();
            dict_set_pooled_string(py_rate, "broker_id",
                pMMInstrumentCommissionRate->BrokerID, sizeof(pMMInstrumentCommissionRate->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_rate, "investor_id",
                pMMInstrumentCommissionRate->InvestorID, sizeof(pMMInstrumentCommissionRate->InvestorID), GlobalStringPools::Users);
            dict_set_double(py_rate, "open_ratio_by_money", pMMInstrumentCommissionRate->OpenRatioByMoney);
            dict_set_double(py_rate, "open_ratio_by_volume", pMMInstrumentCommissionRate->OpenRatioByVolume);
            dict_set_double(py_rate, "close_ratio_by_money", pMMInstrumentCommissionRate->CloseRatioByMoney);
            dict_set_double(py_rate, "close_ratio_by_volume", pMMInstrumentCommissionRate->CloseRatioByVolume);
            dict_set_double(py_rate, "close_today_ratio_by_money", pMMInstrumentCommissionRate->CloseTodayRatioByMoney);
            dict_set_double(py_rate, "close_today_ratio_by_volume", pMMInstrumentCommissionRate->CloseTodayRatioByVolume);
            dict_set_pooled_string(py_rate, "instrument_id",
                pMMInstrumentCommissionRate->InstrumentID, sizeof(pMMInstrumentCommissionRate->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_mm_instrument_commission_rate", "(OOii)",
            py_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryMMOptionInstrCommRate(CThostFtdcMMOptionInstrCommRateField* pMMOptionInstrCommRate,
                                                            CThostFtdcRspInfoField* pRspInfo,
                                                            int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rate = Py_None;
        Py_INCREF(Py_None);
        if (pMMOptionInstrCommRate) {
            py_rate = PyDict_New();
            dict_set_pooled_string(py_rate, "broker_id",
                pMMOptionInstrCommRate->BrokerID, sizeof(pMMOptionInstrCommRate->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_rate, "investor_id",
                pMMOptionInstrCommRate->InvestorID, sizeof(pMMOptionInstrCommRate->InvestorID), GlobalStringPools::Users);
            dict_set_double(py_rate, "open_ratio_by_money", pMMOptionInstrCommRate->OpenRatioByMoney);
            dict_set_double(py_rate, "open_ratio_by_volume", pMMOptionInstrCommRate->OpenRatioByVolume);
            dict_set_double(py_rate, "close_ratio_by_money", pMMOptionInstrCommRate->CloseRatioByMoney);
            dict_set_double(py_rate, "close_ratio_by_volume", pMMOptionInstrCommRate->CloseRatioByVolume);
            dict_set_double(py_rate, "close_today_ratio_by_money", pMMOptionInstrCommRate->CloseTodayRatioByMoney);
            dict_set_double(py_rate, "close_today_ratio_by_volume", pMMOptionInstrCommRate->CloseTodayRatioByVolume);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_mm_option_instr_comm_rate", "(OOii)",
            py_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryInstrumentOrderCommRate(CThostFtdcInstrumentOrderCommRateField* pInstrumentOrderCommRate,
                                                              CThostFtdcRspInfoField* pRspInfo,
                                                              int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rate = Py_None;
        Py_INCREF(Py_None);
        if (pInstrumentOrderCommRate) {
            py_rate = PyDict_New();
            dict_set_pooled_string(py_rate, "broker_id",
                pInstrumentOrderCommRate->BrokerID, sizeof(pInstrumentOrderCommRate->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_rate, "investor_id",
                pInstrumentOrderCommRate->InvestorID, sizeof(pInstrumentOrderCommRate->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_rate, "hedge_flag",
                pInstrumentOrderCommRate->HedgeFlag, sizeof(pInstrumentOrderCommRate->HedgeFlag));
            dict_set_double(py_rate, "order_comm_by_volume", pInstrumentOrderCommRate->OrderCommByVolume);
            dict_set_double(py_rate, "order_action_comm_by_volume", pInstrumentOrderCommRate->OrderActionCommByVolume);
            dict_set_pooled_string(py_rate, "exchange_id",
                pInstrumentOrderCommRate->ExchangeID, sizeof(pInstrumentOrderCommRate->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_rate, "invest_unit_id",
                pInstrumentOrderCommRate->InvestUnitID, sizeof(pInstrumentOrderCommRate->InvestUnitID));
            dict_set_pooled_string(py_rate, "instrument_id",
                pInstrumentOrderCommRate->InstrumentID, sizeof(pInstrumentOrderCommRate->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_instrument_order_comm_rate", "(OOii)",
            py_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQrySecAgentCheckMode(CThostFtdcSecAgentCheckModeField* pSecAgentCheckMode,
                                                        CThostFtdcRspInfoField* pRspInfo,
                                                        int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_mode = Py_None;
        Py_INCREF(Py_None);
        if (pSecAgentCheckMode) {
            py_mode = PyDict_New();
            dict_set_pooled_string(py_mode, "investor_id",
                pSecAgentCheckMode->InvestorID, sizeof(pSecAgentCheckMode->InvestorID), GlobalStringPools::Users);
            dict_set_pooled_string(py_mode, "broker_id",
                pSecAgentCheckMode->BrokerID, sizeof(pSecAgentCheckMode->BrokerID), GlobalStringPools::Brokers);
            dict_set_string(py_mode, "currency_id",
                pSecAgentCheckMode->CurrencyID, sizeof(pSecAgentCheckMode->CurrencyID));
            dict_set_string(py_mode, "broker_sec_agent_id",
                pSecAgentCheckMode->BrokerSecAgentID, sizeof(pSecAgentCheckMode->BrokerSecAgentID));
            PyDict_SetItemString(py_mode, "check_self_account", PyBool_FromLong(pSecAgentCheckMode->CheckSelfAccount != 0));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_sec_agent_check_mode", "(OOii)",
            py_mode, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_mode);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQrySecAgentTradeInfo(CThostFtdcSecAgentTradeInfoField* pSecAgentTradeInfo,
                                                        CThostFtdcRspInfoField* pRspInfo,
                                                        int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_info = Py_None;
        Py_INCREF(Py_None);
        if (pSecAgentTradeInfo) {
            py_info = PyDict_New();
            dict_set_pooled_string(py_info, "broker_id",
                pSecAgentTradeInfo->BrokerID, sizeof(pSecAgentTradeInfo->BrokerID), GlobalStringPools::Brokers);
            dict_set_string(py_info, "broker_sec_agent_id",
                pSecAgentTradeInfo->BrokerSecAgentID, sizeof(pSecAgentTradeInfo->BrokerSecAgentID));
            dict_set_pooled_string(py_info, "investor_id",
                pSecAgentTradeInfo->InvestorID, sizeof(pSecAgentTradeInfo->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_info, "long_customer_name",
                pSecAgentTradeInfo->LongCustomerName, sizeof(pSecAgentTradeInfo->LongCustomerName));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_sec_agent_trade_info", "(OOii)",
            py_info, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_info);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryOptionInstrTradeCost(CThostFtdcOptionInstrTradeCostField* pOptionInstrTradeCost,
                                                           CThostFtdcRspInfoField* pRspInfo,
                                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_cost = Py_None;
        Py_INCREF(Py_None);
        if (pOptionInstrTradeCost) {
            py_cost = PyDict_New();
            dict_set_pooled_string(py_cost, "broker_id",
                pOptionInstrTradeCost->BrokerID, sizeof(pOptionInstrTradeCost->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_cost, "investor_id",
                pOptionInstrTradeCost->InvestorID, sizeof(pOptionInstrTradeCost->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_cost, "hedge_flag",
                pOptionInstrTradeCost->HedgeFlag, sizeof(pOptionInstrTradeCost->HedgeFlag));
            dict_set_double(py_cost, "fixed_margin", pOptionInstrTradeCost->FixedMargin);
            dict_set_double(py_cost, "mini_margin", pOptionInstrTradeCost->MiniMargin);
            dict_set_double(py_cost, "royalty", pOptionInstrTradeCost->Royalty);
            dict_set_double(py_cost, "exch_fixed_margin", pOptionInstrTradeCost->ExchFixedMargin);
            dict_set_double(py_cost, "exch_mini_margin", pOptionInstrTradeCost->ExchMiniMargin);
            dict_set_pooled_string(py_cost, "exchange_id",
                pOptionInstrTradeCost->ExchangeID, sizeof(pOptionInstrTradeCost->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_cost, "invest_unit_id",
                pOptionInstrTradeCost->InvestUnitID, sizeof(pOptionInstrTradeCost->InvestUnitID));
            dict_set_pooled_string(py_cost, "instrument_id",
                pOptionInstrTradeCost->InstrumentID, sizeof(pOptionInstrTradeCost->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_option_instr_trade_cost", "(OOii)",
            py_cost, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_cost);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryOptionInstrCommRate(CThostFtdcOptionInstrCommRateField* pOptionInstrCommRate,
                                                          CThostFtdcRspInfoField* pRspInfo,
                                                          int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_rate = Py_None;
        Py_INCREF(Py_None);
        if (pOptionInstrCommRate) {
            py_rate = PyDict_New();
            dict_set_pooled_string(py_rate, "broker_id",
                pOptionInstrCommRate->BrokerID, sizeof(pOptionInstrCommRate->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_rate, "investor_id",
                pOptionInstrCommRate->InvestorID, sizeof(pOptionInstrCommRate->InvestorID), GlobalStringPools::Users);
            dict_set_double(py_rate, "open_ratio_by_money", pOptionInstrCommRate->OpenRatioByMoney);
            dict_set_double(py_rate, "open_ratio_by_volume", pOptionInstrCommRate->OpenRatioByVolume);
            dict_set_double(py_rate, "close_ratio_by_money", pOptionInstrCommRate->CloseRatioByMoney);
            dict_set_double(py_rate, "close_ratio_by_volume", pOptionInstrCommRate->CloseRatioByVolume);
            dict_set_double(py_rate, "close_today_ratio_by_money", pOptionInstrCommRate->CloseTodayRatioByMoney);
            dict_set_double(py_rate, "close_today_ratio_by_volume", pOptionInstrCommRate->CloseTodayRatioByVolume);
            dict_set_double(py_rate, "strike_ratio_by_money", pOptionInstrCommRate->StrikeRatioByMoney);
            dict_set_double(py_rate, "strike_ratio_by_volume", pOptionInstrCommRate->StrikeRatioByVolume);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_option_instr_comm_rate", "(OOii)",
            py_rate, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_rate);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryBrokerTradingParams(CThostFtdcBrokerTradingParamsField* pBrokerTradingParams,
                                                          CThostFtdcRspInfoField* pRspInfo,
                                                          int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_params = Py_None;
        Py_INCREF(Py_None);
        if (pBrokerTradingParams) {
            py_params = PyDict_New();
            dict_set_pooled_string(py_params, "broker_id",
                pBrokerTradingParams->BrokerID, sizeof(pBrokerTradingParams->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_params, "investor_id",
                pBrokerTradingParams->InvestorID, sizeof(pBrokerTradingParams->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_params, "margin_price_type",
                pBrokerTradingParams->MarginPriceType, sizeof(pBrokerTradingParams->MarginPriceType));
            dict_set_string(py_params, "algorithm",
                pBrokerTradingParams->Algorithm, sizeof(pBrokerTradingParams->Algorithm));
            dict_set_string(py_params, "avail_include_close_profit",
                pBrokerTradingParams->AvailIncludeCloseProfit, sizeof(pBrokerTradingParams->AvailIncludeCloseProfit));
            dict_set_string(py_params, "currency_id",
                pBrokerTradingParams->CurrencyID, sizeof(pBrokerTradingParams->CurrencyID));
            dict_set_string(py_params, "option_royalty_price_type",
                pBrokerTradingParams->OptionRoyaltyPriceType, sizeof(pBrokerTradingParams->OptionRoyaltyPriceType));
            dict_set_string(py_params, "account_id",
                pBrokerTradingParams->AccountID, sizeof(pBrokerTradingParams->AccountID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_broker_trading_params", "(OOii)",
            py_params, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_params);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQryBrokerTradingAlgos(CThostFtdcBrokerTradingAlgosField* pBrokerTradingAlgos,
                                                         CThostFtdcRspInfoField* pRspInfo,
                                                         int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_algos = Py_None;
        Py_INCREF(Py_None);
        if (pBrokerTradingAlgos) {
            py_algos = PyDict_New();
            dict_set_pooled_string(py_algos, "broker_id",
                pBrokerTradingAlgos->BrokerID, sizeof(pBrokerTradingAlgos->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_algos, "exchange_id",
                pBrokerTradingAlgos->ExchangeID, sizeof(pBrokerTradingAlgos->ExchangeID), GlobalStringPools::ExchangeCodes);
            dict_set_string(py_algos, "handle_position_algo_id",
                pBrokerTradingAlgos->HandlePositionAlgoID, sizeof(pBrokerTradingAlgos->HandlePositionAlgoID));
            dict_set_string(py_algos, "find_margin_rate_algo_id",
                pBrokerTradingAlgos->FindMarginRateAlgoID, sizeof(pBrokerTradingAlgos->FindMarginRateAlgoID));
            dict_set_string(py_algos, "handle_trading_account_algo_id",
                pBrokerTradingAlgos->HandleTradingAccountAlgoID, sizeof(pBrokerTradingAlgos->HandleTradingAccountAlgoID));
            dict_set_pooled_string(py_algos, "instrument_id",
                pBrokerTradingAlgos->InstrumentID, sizeof(pBrokerTradingAlgos->InstrumentID), GlobalStringPools::Instruments);
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_qry_broker_trading_algos", "(OOii)",
            py_algos, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_algos);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspRemoveParkedOrder(CThostFtdcRemoveParkedOrderField* pRemoveParkedOrder,
                                                    CThostFtdcRspInfoField* pRspInfo,
                                                    int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_remove = Py_None;
        Py_INCREF(Py_None);
        if (pRemoveParkedOrder) {
            py_remove = PyDict_New();
            dict_set_pooled_string(py_remove, "broker_id",
                pRemoveParkedOrder->BrokerID, sizeof(pRemoveParkedOrder->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_remove, "investor_id",
                pRemoveParkedOrder->InvestorID, sizeof(pRemoveParkedOrder->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_remove, "parked_order_id",
                pRemoveParkedOrder->ParkedOrderID, sizeof(pRemoveParkedOrder->ParkedOrderID));
            dict_set_string(py_remove, "invest_unit_id",
                pRemoveParkedOrder->InvestUnitID, sizeof(pRemoveParkedOrder->InvestUnitID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_remove_parked_order", "(OOii)",
            py_remove, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_remove);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspRemoveParkedOrderAction(CThostFtdcRemoveParkedOrderActionField* pRemoveParkedOrderAction,
                                                          CThostFtdcRspInfoField* pRspInfo,
                                                          int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_remove = Py_None;
        Py_INCREF(Py_None);
        if (pRemoveParkedOrderAction) {
            py_remove = PyDict_New();
            dict_set_pooled_string(py_remove, "broker_id",
                pRemoveParkedOrderAction->BrokerID, sizeof(pRemoveParkedOrderAction->BrokerID), GlobalStringPools::Brokers);
            dict_set_pooled_string(py_remove, "investor_id",
                pRemoveParkedOrderAction->InvestorID, sizeof(pRemoveParkedOrderAction->InvestorID), GlobalStringPools::Users);
            dict_set_string(py_remove, "parked_order_action_id",
                pRemoveParkedOrderAction->ParkedOrderActionID, sizeof(pRemoveParkedOrderAction->ParkedOrderActionID));
            dict_set_string(py_remove, "invest_unit_id",
                pRemoveParkedOrderAction->InvestUnitID, sizeof(pRemoveParkedOrderAction->InvestUnitID));
        }

        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_remove_parked_order_action", "(OOii)",
            py_remove, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_remove);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // =============================================================================
    // 第5批：银行转账相关 (22个方法)
    // 注意：由于银行转账结构体字段过多(40+)，此处只映射核心字段
    // =============================================================================

    FORCE_INLINE_MEMBER void OnRtnFromBankToFutureByBank(CThostFtdcRspTransferField* pRspTransfer) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_transfer_dict(pRspTransfer);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_from_bank_to_future_by_bank", "(O)", py_transfer);

        Py_XDECREF(py_transfer);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnFromFutureToBankByBank(CThostFtdcRspTransferField* pRspTransfer) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_transfer_dict(pRspTransfer);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_from_future_to_bank_by_bank", "(O)", py_transfer);

        Py_XDECREF(py_transfer);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnRepealFromBankToFutureByBank(CThostFtdcRspRepealField* pRspRepeal) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_repeal_dict(pRspRepeal);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_repeal_from_bank_to_future_by_bank", "(O)", py_repeal);

        Py_XDECREF(py_repeal);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnRepealFromFutureToBankByBank(CThostFtdcRspRepealField* pRspRepeal) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_repeal_dict(pRspRepeal);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_repeal_from_future_to_bank_by_bank", "(O)", py_repeal);

        Py_XDECREF(py_repeal);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnFromBankToFutureByFuture(CThostFtdcRspTransferField* pRspTransfer) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_transfer_dict(pRspTransfer);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_from_bank_to_future_by_future", "(O)", py_transfer);

        Py_XDECREF(py_transfer);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnFromFutureToBankByFuture(CThostFtdcRspTransferField* pRspTransfer) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_transfer_dict(pRspTransfer);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_from_future_to_bank_by_future", "(O)", py_transfer);

        Py_XDECREF(py_transfer);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnRepealFromBankToFutureByFutureManual(CThostFtdcRspRepealField* pRspRepeal) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_repeal_dict(pRspRepeal);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_repeal_from_bank_to_future_by_future_manual", "(O)", py_repeal);

        Py_XDECREF(py_repeal);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnRepealFromFutureToBankByFutureManual(CThostFtdcRspRepealField* pRspRepeal) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_repeal_dict(pRspRepeal);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_repeal_from_future_to_bank_by_future_manual", "(O)", py_repeal);

        Py_XDECREF(py_repeal);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnQueryBankBalanceByFuture(CThostFtdcNotifyQueryAccountField* pNotifyQueryAccount) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_query = build_query_account_dict(pNotifyQueryAccount);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_query_bank_balance_by_future", "(O)", py_query);

        Py_XDECREF(py_query);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnErrRtnBankToFutureByFuture(CThostFtdcReqTransferField* pReqTransfer,
                                                           CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_req_transfer_dict(pReqTransfer);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_bank_to_future_by_future", "(OO)",
            py_transfer, py_rsp_info);

        Py_XDECREF(py_transfer);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnErrRtnFutureToBankByFuture(CThostFtdcReqTransferField* pReqTransfer,
                                                           CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_req_transfer_dict(pReqTransfer);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_future_to_bank_by_future", "(OO)",
            py_transfer, py_rsp_info);

        Py_XDECREF(py_transfer);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnErrRtnRepealBankToFutureByFutureManual(CThostFtdcReqRepealField* pReqRepeal,
                                                                      CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_req_repeal_dict(pReqRepeal);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_repeal_bank_to_future_by_future_manual", "(OO)",
            py_repeal, py_rsp_info);

        Py_XDECREF(py_repeal);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnErrRtnRepealFutureToBankByFutureManual(CThostFtdcReqRepealField* pReqRepeal,
                                                                      CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_req_repeal_dict(pReqRepeal);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_repeal_future_to_bank_by_future_manual", "(OO)",
            py_repeal, py_rsp_info);

        Py_XDECREF(py_repeal);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnErrRtnQueryBankBalanceByFuture(CThostFtdcReqQueryAccountField* pReqQueryAccount,
                                                              CThostFtdcRspInfoField* pRspInfo) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_query = build_req_query_account_dict(pReqQueryAccount);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_err_rtn_query_bank_balance_by_future", "(OO)",
            py_query, py_rsp_info);

        Py_XDECREF(py_query);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnRepealFromBankToFutureByFuture(CThostFtdcRspRepealField* pRspRepeal) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_repeal_dict(pRspRepeal);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_repeal_from_bank_to_future_by_future", "(O)", py_repeal);

        Py_XDECREF(py_repeal);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnRepealFromFutureToBankByFuture(CThostFtdcRspRepealField* pRspRepeal) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_repeal = build_repeal_dict(pRspRepeal);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_repeal_from_future_to_bank_by_future", "(O)", py_repeal);

        Py_XDECREF(py_repeal);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspFromBankToFutureByFuture(CThostFtdcReqTransferField* pReqTransfer,
                                                           CThostFtdcRspInfoField* pRspInfo,
                                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_req_transfer_dict(pReqTransfer);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_from_bank_to_future_by_future", "(OOii)",
            py_transfer, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_transfer);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspFromFutureToBankByFuture(CThostFtdcReqTransferField* pReqTransfer,
                                                           CThostFtdcRspInfoField* pRspInfo,
                                                           int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_transfer = build_req_transfer_dict(pReqTransfer);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_from_future_to_bank_by_future", "(OOii)",
            py_transfer, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_transfer);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRspQueryBankAccountMoneyByFuture(CThostFtdcReqQueryAccountField* pReqQueryAccount,
                                                                CThostFtdcRspInfoField* pRspInfo,
                                                                int nRequestID, bool bIsLast) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_query = build_req_query_account_dict(pReqQueryAccount);
        PyObject* py_rsp_info = build_error_dict(pRspInfo);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_query_bank_account_money_by_future", "(OOii)",
            py_query, py_rsp_info, nRequestID, bIsLast);

        Py_XDECREF(py_query);
        Py_XDECREF(py_rsp_info);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnOpenAccountByBank(CThostFtdcOpenAccountField* pOpenAccount) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_account = build_open_account_dict(pOpenAccount);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_open_account_by_bank", "(O)", py_account);

        Py_XDECREF(py_account);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnCancelAccountByBank(CThostFtdcCancelAccountField* pCancelAccount) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_account = build_cancel_account_dict(pCancelAccount);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_cancel_account_by_bank", "(O)", py_account);

        Py_XDECREF(py_account);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    FORCE_INLINE_MEMBER void OnRtnChangeAccountByBank(CThostFtdcChangeAccountField* pChangeAccount) override {
        if (!m_api || !m_api->py_spi) return;
        PyGILStateKeeper gil;

        PyObject* py_account = build_change_account_dict(pChangeAccount);
        PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rtn_change_account_by_bank", "(O)", py_account);

        Py_XDECREF(py_account);
        if (!result) PyErr_Print();
        Py_XDECREF(result);
    }

    // =============================================================================
    // 辅助函数：银行转账相关字典构建
    // =============================================================================

private:
    // 构建转账响应字典 (核心字段)
    static PyObject* build_transfer_dict(CThostFtdcRspTransferField* pTransfer) {
        if (!pTransfer) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_string(result, "trade_code", pTransfer->TradeCode, sizeof(pTransfer->TradeCode));
        dict_set_string(result, "bank_id", pTransfer->BankID, sizeof(pTransfer->BankID));
        dict_set_string(result, "bank_branch_id", pTransfer->BankBranchID, sizeof(pTransfer->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pTransfer->BrokerID, sizeof(pTransfer->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pTransfer->FutureBranchID, sizeof(pTransfer->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pTransfer->TradeDate, sizeof(pTransfer->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pTransfer->TradeTime, sizeof(pTransfer->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pTransfer->BankSerial, sizeof(pTransfer->BankSerial));
        dict_set_pooled_string(result, "trading_day", pTransfer->TradingDay, sizeof(pTransfer->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pTransfer->PlateSerial);
        dict_set_int(result, "session_id", pTransfer->SessionID);
        dict_set_string(result, "customer_name", pTransfer->CustomerName, sizeof(pTransfer->CustomerName));
        dict_set_string(result, "id_card_type", pTransfer->IdCardType, sizeof(pTransfer->IdCardType));
        dict_set_string(result, "identified_card_no", pTransfer->IdentifiedCardNo, sizeof(pTransfer->IdentifiedCardNo));
        dict_set_string(result, "cust_type", pTransfer->CustType, sizeof(pTransfer->CustType));
        dict_set_string(result, "bank_account", pTransfer->BankAccount, sizeof(pTransfer->BankAccount));
        dict_set_string(result, "account_id", pTransfer->AccountID, sizeof(pTransfer->AccountID));
        dict_set_pooled_string(result, "user_id", pTransfer->UserID, sizeof(pTransfer->UserID), GlobalStringPools::Users);
        dict_set_string(result, "currency_id", pTransfer->CurrencyID, sizeof(pTransfer->CurrencyID));
        dict_set_double(result, "trade_amount", pTransfer->TradeAmount);
        dict_set_double(result, "future_fetch_amount", pTransfer->FutureFetchAmount);
        dict_set_string(result, "fee_pay_flag", pTransfer->FeePayFlag, sizeof(pTransfer->FeePayFlag));
        dict_set_double(result, "cust_fee", pTransfer->CustFee);
        dict_set_double(result, "broker_fee", pTransfer->BrokerFee);
        dict_set_string(result, "message", pTransfer->Message, sizeof(pTransfer->Message));
        dict_set_int(result, "request_id", pTransfer->RequestID);
        dict_set_string(result, "tid", pTransfer->TID, sizeof(pTransfer->TID));
        dict_set_string(result, "transfer_status", pTransfer->TransferStatus, sizeof(pTransfer->TransferStatus));
        dict_set_int(result, "error_id", pTransfer->ErrorID);
        dict_set_gbk_string(result, "error_msg", pTransfer->ErrorMsg, sizeof(pTransfer->ErrorMsg));
        return result;
    }

    // 构建冲正响应字典 (核心字段)
    static PyObject* build_repeal_dict(CThostFtdcRspRepealField* pRepeal) {
        if (!pRepeal) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_int(result, "repeal_time_interval", pRepeal->RepealTimeInterval);
        dict_set_int(result, "repealed_times", pRepeal->RepealedTimes);
        dict_set_string(result, "bank_repeal_flag", pRepeal->BankRepealFlag, sizeof(pRepeal->BankRepealFlag));
        dict_set_string(result, "broker_repeal_flag", pRepeal->BrokerRepealFlag, sizeof(pRepeal->BrokerRepealFlag));
        dict_set_int(result, "plate_repeal_serial", pRepeal->PlateRepealSerial);
        dict_set_string(result, "bank_repeal_serial", pRepeal->BankRepealSerial, sizeof(pRepeal->BankRepealSerial));
        dict_set_int(result, "future_repeal_serial", pRepeal->FutureRepealSerial);
        dict_set_string(result, "trade_code", pRepeal->TradeCode, sizeof(pRepeal->TradeCode));
        dict_set_string(result, "bank_id", pRepeal->BankID, sizeof(pRepeal->BankID));
        dict_set_string(result, "bank_branch_id", pRepeal->BankBranchID, sizeof(pRepeal->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pRepeal->BrokerID, sizeof(pRepeal->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pRepeal->FutureBranchID, sizeof(pRepeal->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pRepeal->TradeDate, sizeof(pRepeal->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pRepeal->TradeTime, sizeof(pRepeal->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pRepeal->BankSerial, sizeof(pRepeal->BankSerial));
        dict_set_pooled_string(result, "trading_day", pRepeal->TradingDay, sizeof(pRepeal->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pRepeal->PlateSerial);
        dict_set_int(result, "session_id", pRepeal->SessionID);
        dict_set_string(result, "customer_name", pRepeal->CustomerName, sizeof(pRepeal->CustomerName));
        dict_set_string(result, "account_id", pRepeal->AccountID, sizeof(pRepeal->AccountID));
        dict_set_pooled_string(result, "user_id", pRepeal->UserID, sizeof(pRepeal->UserID), GlobalStringPools::Users);
        dict_set_string(result, "currency_id", pRepeal->CurrencyID, sizeof(pRepeal->CurrencyID));
        dict_set_double(result, "trade_amount", pRepeal->TradeAmount);
        dict_set_string(result, "message", pRepeal->Message, sizeof(pRepeal->Message));
        dict_set_int(result, "request_id", pRepeal->RequestID);
        dict_set_string(result, "tid", pRepeal->TID, sizeof(pRepeal->TID));
        dict_set_int(result, "error_id", pRepeal->ErrorID);
        dict_set_gbk_string(result, "error_msg", pRepeal->ErrorMsg, sizeof(pRepeal->ErrorMsg));
        return result;
    }

    // 构建查询账户通知字典 (核心字段)
    static PyObject* build_query_account_dict(CThostFtdcNotifyQueryAccountField* pQuery) {
        if (!pQuery) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_string(result, "trade_code", pQuery->TradeCode, sizeof(pQuery->TradeCode));
        dict_set_string(result, "bank_id", pQuery->BankID, sizeof(pQuery->BankID));
        dict_set_string(result, "bank_branch_id", pQuery->BankBranchID, sizeof(pQuery->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pQuery->BrokerID, sizeof(pQuery->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pQuery->FutureBranchID, sizeof(pQuery->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pQuery->TradeDate, sizeof(pQuery->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pQuery->TradeTime, sizeof(pQuery->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pQuery->BankSerial, sizeof(pQuery->BankSerial));
        dict_set_pooled_string(result, "trading_day", pQuery->TradingDay, sizeof(pQuery->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pQuery->PlateSerial);
        dict_set_int(result, "session_id", pQuery->SessionID);
        dict_set_string(result, "customer_name", pQuery->CustomerName, sizeof(pQuery->CustomerName));
        dict_set_string(result, "bank_account", pQuery->BankAccount, sizeof(pQuery->BankAccount));
        dict_set_string(result, "account_id", pQuery->AccountID, sizeof(pQuery->AccountID));
        dict_set_pooled_string(result, "user_id", pQuery->UserID, sizeof(pQuery->UserID), GlobalStringPools::Users);
        dict_set_string(result, "currency_id", pQuery->CurrencyID, sizeof(pQuery->CurrencyID));
        dict_set_double(result, "bank_balance", pQuery->BankBalance);
        dict_set_double(result, "future_fetch_amount", pQuery->FutureFetchAmount);
        dict_set_int(result, "error_id", pQuery->ErrorID);
        dict_set_gbk_string(result, "error_msg", pQuery->ErrorMsg, sizeof(pQuery->ErrorMsg));
        return result;
    }

    // 构建转账请求字典 (核心字段)
    static PyObject* build_req_transfer_dict(CThostFtdcReqTransferField* pTransfer) {
        if (!pTransfer) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_string(result, "trade_code", pTransfer->TradeCode, sizeof(pTransfer->TradeCode));
        dict_set_string(result, "bank_id", pTransfer->BankID, sizeof(pTransfer->BankID));
        dict_set_string(result, "bank_branch_id", pTransfer->BankBranchID, sizeof(pTransfer->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pTransfer->BrokerID, sizeof(pTransfer->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pTransfer->FutureBranchID, sizeof(pTransfer->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pTransfer->TradeDate, sizeof(pTransfer->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pTransfer->TradeTime, sizeof(pTransfer->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pTransfer->BankSerial, sizeof(pTransfer->BankSerial));
        dict_set_pooled_string(result, "trading_day", pTransfer->TradingDay, sizeof(pTransfer->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pTransfer->PlateSerial);
        dict_set_int(result, "session_id", pTransfer->SessionID);
        dict_set_string(result, "customer_name", pTransfer->CustomerName, sizeof(pTransfer->CustomerName));
        dict_set_string(result, "bank_account", pTransfer->BankAccount, sizeof(pTransfer->BankAccount));
        dict_set_string(result, "account_id", pTransfer->AccountID, sizeof(pTransfer->AccountID));
        dict_set_pooled_string(result, "user_id", pTransfer->UserID, sizeof(pTransfer->UserID), GlobalStringPools::Users);
        dict_set_string(result, "currency_id", pTransfer->CurrencyID, sizeof(pTransfer->CurrencyID));
        dict_set_double(result, "trade_amount", pTransfer->TradeAmount);
        return result;
    }

    // 构建冲正请求字典 (核心字段)
    static PyObject* build_req_repeal_dict(CThostFtdcReqRepealField* pRepeal) {
        if (!pRepeal) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_int(result, "repeal_time_interval", pRepeal->RepealTimeInterval);
        dict_set_int(result, "repealed_times", pRepeal->RepealedTimes);
        dict_set_string(result, "bank_repeal_flag", pRepeal->BankRepealFlag, sizeof(pRepeal->BankRepealFlag));
        dict_set_string(result, "broker_repeal_flag", pRepeal->BrokerRepealFlag, sizeof(pRepeal->BrokerRepealFlag));
        dict_set_int(result, "plate_repeal_serial", pRepeal->PlateRepealSerial);
        dict_set_string(result, "bank_repeal_serial", pRepeal->BankRepealSerial, sizeof(pRepeal->BankRepealSerial));
        dict_set_int(result, "future_repeal_serial", pRepeal->FutureRepealSerial);
        dict_set_string(result, "trade_code", pRepeal->TradeCode, sizeof(pRepeal->TradeCode));
        dict_set_string(result, "bank_id", pRepeal->BankID, sizeof(pRepeal->BankID));
        dict_set_string(result, "bank_branch_id", pRepeal->BankBranchID, sizeof(pRepeal->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pRepeal->BrokerID, sizeof(pRepeal->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pRepeal->FutureBranchID, sizeof(pRepeal->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pRepeal->TradeDate, sizeof(pRepeal->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pRepeal->TradeTime, sizeof(pRepeal->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pRepeal->BankSerial, sizeof(pRepeal->BankSerial));
        dict_set_pooled_string(result, "trading_day", pRepeal->TradingDay, sizeof(pRepeal->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pRepeal->PlateSerial);
        dict_set_int(result, "session_id", pRepeal->SessionID);
        dict_set_string(result, "bank_account", pRepeal->BankAccount, sizeof(pRepeal->BankAccount));
        dict_set_string(result, "account_id", pRepeal->AccountID, sizeof(pRepeal->AccountID));
        dict_set_pooled_string(result, "user_id", pRepeal->UserID, sizeof(pRepeal->UserID), GlobalStringPools::Users);
        dict_set_string(result, "currency_id", pRepeal->CurrencyID, sizeof(pRepeal->CurrencyID));
        dict_set_double(result, "trade_amount", pRepeal->TradeAmount);
        return result;
    }

    // 构建查询账户请求字典 (核心字段)
    static PyObject* build_req_query_account_dict(CThostFtdcReqQueryAccountField* pQuery) {
        if (!pQuery) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_string(result, "trade_code", pQuery->TradeCode, sizeof(pQuery->TradeCode));
        dict_set_string(result, "bank_id", pQuery->BankID, sizeof(pQuery->BankID));
        dict_set_string(result, "bank_branch_id", pQuery->BankBranchID, sizeof(pQuery->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pQuery->BrokerID, sizeof(pQuery->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pQuery->FutureBranchID, sizeof(pQuery->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pQuery->TradeDate, sizeof(pQuery->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pQuery->TradeTime, sizeof(pQuery->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pQuery->BankSerial, sizeof(pQuery->BankSerial));
        dict_set_pooled_string(result, "trading_day", pQuery->TradingDay, sizeof(pQuery->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pQuery->PlateSerial);
        dict_set_int(result, "session_id", pQuery->SessionID);
        dict_set_string(result, "bank_account", pQuery->BankAccount, sizeof(pQuery->BankAccount));
        dict_set_string(result, "account_id", pQuery->AccountID, sizeof(pQuery->AccountID));
        dict_set_pooled_string(result, "user_id", pQuery->UserID, sizeof(pQuery->UserID), GlobalStringPools::Users);
        dict_set_string(result, "currency_id", pQuery->CurrencyID, sizeof(pQuery->CurrencyID));
        return result;
    }

    // 构建开户字典 (核心字段)
    static PyObject* build_open_account_dict(CThostFtdcOpenAccountField* pAccount) {
        if (!pAccount) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_string(result, "trade_code", pAccount->TradeCode, sizeof(pAccount->TradeCode));
        dict_set_string(result, "bank_id", pAccount->BankID, sizeof(pAccount->BankID));
        dict_set_string(result, "bank_branch_id", pAccount->BankBranchID, sizeof(pAccount->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pAccount->BrokerID, sizeof(pAccount->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pAccount->FutureBranchID, sizeof(pAccount->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pAccount->TradeDate, sizeof(pAccount->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pAccount->TradeTime, sizeof(pAccount->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pAccount->BankSerial, sizeof(pAccount->BankSerial));
        dict_set_pooled_string(result, "trading_day", pAccount->TradingDay, sizeof(pAccount->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pAccount->PlateSerial);
        dict_set_int(result, "session_id", pAccount->SessionID);
        dict_set_string(result, "customer_name", pAccount->CustomerName, sizeof(pAccount->CustomerName));
        dict_set_string(result, "bank_account", pAccount->BankAccount, sizeof(pAccount->BankAccount));
        dict_set_string(result, "account_id", pAccount->AccountID, sizeof(pAccount->AccountID));
        dict_set_pooled_string(result, "user_id", pAccount->UserID, sizeof(pAccount->UserID), GlobalStringPools::Users);
        dict_set_int(result, "error_id", pAccount->ErrorID);
        dict_set_gbk_string(result, "error_msg", pAccount->ErrorMsg, sizeof(pAccount->ErrorMsg));
        return result;
    }

    // 构建销户字典 (核心字段)
    static PyObject* build_cancel_account_dict(CThostFtdcCancelAccountField* pAccount) {
        if (!pAccount) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_string(result, "trade_code", pAccount->TradeCode, sizeof(pAccount->TradeCode));
        dict_set_string(result, "bank_id", pAccount->BankID, sizeof(pAccount->BankID));
        dict_set_string(result, "bank_branch_id", pAccount->BankBranchID, sizeof(pAccount->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pAccount->BrokerID, sizeof(pAccount->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pAccount->FutureBranchID, sizeof(pAccount->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pAccount->TradeDate, sizeof(pAccount->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pAccount->TradeTime, sizeof(pAccount->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pAccount->BankSerial, sizeof(pAccount->BankSerial));
        dict_set_pooled_string(result, "trading_day", pAccount->TradingDay, sizeof(pAccount->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pAccount->PlateSerial);
        dict_set_int(result, "session_id", pAccount->SessionID);
        dict_set_string(result, "customer_name", pAccount->CustomerName, sizeof(pAccount->CustomerName));
        dict_set_string(result, "bank_account", pAccount->BankAccount, sizeof(pAccount->BankAccount));
        dict_set_string(result, "account_id", pAccount->AccountID, sizeof(pAccount->AccountID));
        dict_set_pooled_string(result, "user_id", pAccount->UserID, sizeof(pAccount->UserID), GlobalStringPools::Users);
        dict_set_int(result, "error_id", pAccount->ErrorID);
        dict_set_gbk_string(result, "error_msg", pAccount->ErrorMsg, sizeof(pAccount->ErrorMsg));
        return result;
    }

    // 构建变更账户字典 (核心字段)
    static PyObject* build_change_account_dict(CThostFtdcChangeAccountField* pAccount) {
        if (!pAccount) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject* result = PyDict_New();
        dict_set_string(result, "trade_code", pAccount->TradeCode, sizeof(pAccount->TradeCode));
        dict_set_string(result, "bank_id", pAccount->BankID, sizeof(pAccount->BankID));
        dict_set_string(result, "bank_branch_id", pAccount->BankBranchID, sizeof(pAccount->BankBranchID));
        dict_set_pooled_string(result, "broker_id", pAccount->BrokerID, sizeof(pAccount->BrokerID), GlobalStringPools::Brokers);
        dict_set_string(result, "future_branch_id", pAccount->FutureBranchID, sizeof(pAccount->FutureBranchID));
        dict_set_pooled_string(result, "trade_date", pAccount->TradeDate, sizeof(pAccount->TradeDate), GlobalStringPools::Dates);
        dict_set_pooled_string(result, "trade_time", pAccount->TradeTime, sizeof(pAccount->TradeTime), GlobalStringPools::Times);
        dict_set_string(result, "bank_serial", pAccount->BankSerial, sizeof(pAccount->BankSerial));
        dict_set_pooled_string(result, "trading_day", pAccount->TradingDay, sizeof(pAccount->TradingDay), GlobalStringPools::Dates);
        dict_set_int(result, "plate_serial", pAccount->PlateSerial);
        dict_set_int(result, "session_id", pAccount->SessionID);
        dict_set_string(result, "customer_name", pAccount->CustomerName, sizeof(pAccount->CustomerName));
        dict_set_string(result, "bank_account", pAccount->BankAccount, sizeof(pAccount->BankAccount));
        dict_set_string(result, "account_id", pAccount->AccountID, sizeof(pAccount->AccountID));
        dict_set_pooled_string(result, "user_id", pAccount->UserID, sizeof(pAccount->UserID), GlobalStringPools::Users);
        dict_set_int(result, "error_id", pAccount->ErrorID);
        dict_set_gbk_string(result, "error_msg", pAccount->ErrorMsg, sizeof(pAccount->ErrorMsg));
        return result;
    }
};

// =============================================================================
// 模块初始化函数
// =============================================================================

static PyObject* cleanup_temporal_pools_trade(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    GlobalStringPools::cleanup_temporal_pools();
    Py_RETURN_NONE;
}

static PyObject* cleanup_instruments_trade(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    GlobalStringPools::cleanup_instruments();
    Py_RETURN_NONE;
}

static PyObject* check_instrument_pool_size_trade(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    size_t size = GlobalStringPools::check_instrument_pool_size();
    return PyLong_FromSize_t(size);
}

static PyObject* get_pool_sizes_trade(PyObject* self, PyObject* args) {
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

static PyMethodDef pcctp_trade_module_methods[] = {
    {"cleanup_temporal_pools", cleanup_temporal_pools_trade, METH_NOARGS,
     "清理日期和时间字符串池（交易日收盘后调用）"},
    {"cleanup_instruments", cleanup_instruments_trade, METH_NOARGS,
     "清理合约代码字符串池（切换交易日或重新订阅时调用）"},
    {"check_instrument_pool_size", check_instrument_pool_size_trade, METH_NOARGS,
     "检查合约池大小并返回当前值（超过 950 时自动警告）"},
    {"get_pool_sizes", get_pool_sizes_trade, METH_NOARGS,
     "获取所有字符串池的大小统计"},
    {NULL}
};

static PyModuleDef PcCTP_trade_module = {
    PyModuleDef_HEAD_INIT,
    "PcCTP.trade",
    "CTP PC Trading API Bindings (Pure Python C API + Force Inline Optimization)",
    -1,
    pcctp_trade_module_methods,
};

PyMODINIT_FUNC PyInit_PcCTP_trade(void) {
    if (initialize_utils() < 0) {
        return NULL;
    }

    PyObject* module = PyModule_Create(&PcCTP_trade_module);
    if (!module) {
        return NULL;
    }

    // TODO: 在实现完 TradeApi 类后，添加类型定义
    // if (PyType_Ready(&TradeApiType) < 0) {
    //     Py_DECREF(module);
    //     return NULL;
    // }

    return module;
}
