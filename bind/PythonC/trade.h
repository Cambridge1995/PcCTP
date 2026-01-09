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

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <vector>
#include <string>
#include <cstring>

// 通用工具库（包含 PY_ARRAY_UNIQUE_SYMBOL 定义）
// 必须在 NumPy 之前包含
#include "util.h"

// NumPy C API（使用 util.h 中定义的 PY_ARRAY_UNIQUE_SYMBOL）
#include <numpy/ndarrayobject.h>

// 通用基类和宏（新增）
#include "common.h"

// PC版 CTP 交易 API 头文件
#if defined(_WIN64)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win64/ThostFtdcTraderApi.h"
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
// SPI回调类声明 (使用下划线命名法 + 内联优化)
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

    // 虚析构函数：确保通过基类指针删除时正确调用派生类析构函数
    virtual ~PyTradeSpi() = default;

    // //////////////////////////////////////////////////////////////////////////
    // 连接相关回调 (使用通用宏)
    // //////////////////////////////////////////////////////////////////////////
    CALLBACK_0_PARAM(OnFrontConnected, "on_front_connected")
    CALLBACK_1_PARAM_INT(OnFrontDisconnected, nReason, "on_front_disconnected")
    CALLBACK_1_PARAM_INT(OnHeartBeatWarning, nTimeLapse, "on_heart_beat_warning")

    // //////////////////////////////////////////////////////////////////////////
    // 认证登录相关回调 (使用 Capsule 宏)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspAuthenticate, CThostFtdcRspAuthenticateField, "RspAuthenticate", "on_rsp_authenticate")
    CAPSULE_CALLBACK_RSP(OnRspUserLogin, CThostFtdcRspUserLoginField, "RspUserLogin", "on_rsp_user_login")
    CAPSULE_CALLBACK_RSP(OnRspUserLogout, CThostFtdcUserLogoutField, "UserLogout", "on_rsp_user_logout")
    CAPSULE_CALLBACK_RSP(OnRspUserPasswordUpdate, CThostFtdcUserPasswordUpdateField, "UserPasswordUpdate", "on_rsp_user_password_update")
    CAPSULE_CALLBACK_RSP(OnRspTradingAccountPasswordUpdate, CThostFtdcTradingAccountPasswordUpdateField, "TradingAccountPasswordUpdate", "on_rsp_trading_account_password_update")
    // OnRspError 签名特殊：只有 pRspInfo, nRequestID, bIsLast，没有 pStruct 参数
    FORCE_INLINE_MEMBER void OnRspError(CThostFtdcRspInfoField* pRspInfo, int nRequestID, bool bIsLast) override {
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

    // //////////////////////////////////////////////////////////////////////////
    // 报单相关回调 (使用 Capsule 宏)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspOrderInsert, CThostFtdcInputOrderField, "InputOrder", "on_rsp_order_insert")
    CAPSULE_CALLBACK_RSP(OnRspParkedOrderInsert, CThostFtdcParkedOrderField, "ParkedOrder", "on_rsp_parked_order_insert")
    CAPSULE_CALLBACK_RSP(OnRspParkedOrderAction, CThostFtdcParkedOrderActionField, "ParkedOrderAction", "on_rsp_parked_order_action")
    CAPSULE_CALLBACK_RSP(OnRspOrderAction, CThostFtdcInputOrderActionField, "InputOrderAction", "on_rsp_order_action")
    CAPSULE_CALLBACK_SINGLE(OnRtnOrder, CThostFtdcOrderField, "Order", "on_rtn_order")
    CAPSULE_CALLBACK_SINGLE(OnRtnTrade, CThostFtdcTradeField, "Trade", "on_rtn_trade")
    CAPSULE_CALLBACK_ERROR(OnErrRtnOrderInsert, CThostFtdcInputOrderField, "InputOrder", "on_err_rtn_order_insert")
    CAPSULE_CALLBACK_ERROR(OnErrRtnOrderAction, CThostFtdcOrderActionField, "OrderAction", "on_err_rtn_order_action")

    // //////////////////////////////////////////////////////////////////////////
    // 查询相关回调 (使用 Capsule 宏 - 主要方法)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQryOrder, CThostFtdcOrderField, "Order", "on_rsp_qry_order")
    CAPSULE_CALLBACK_RSP(OnRspQryTrade, CThostFtdcTradeField, "Trade", "on_rsp_qry_trade")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorPosition, CThostFtdcInvestorPositionField, "InvestorPosition", "on_rsp_qry_investor_position")
    CAPSULE_CALLBACK_RSP(OnRspQryTradingAccount, CThostFtdcTradingAccountField, "TradingAccount", "on_rsp_qry_trading_account")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestor, CThostFtdcInvestorField, "Investor", "on_rsp_qry_investor")
    CAPSULE_CALLBACK_RSP(OnRspQryTradingCode, CThostFtdcTradingCodeField, "TradingCode", "on_rsp_qry_trading_code")
    CAPSULE_CALLBACK_RSP(OnRspQryInstrumentMarginRate, CThostFtdcInstrumentMarginRateField, "InstrumentMarginRate", "on_rsp_qry_instrument_margin_rate")
    CAPSULE_CALLBACK_RSP(OnRspQryInstrumentCommissionRate, CThostFtdcInstrumentCommissionRateField, "InstrumentCommissionRate", "on_rsp_qry_instrument_commission_rate")
    CAPSULE_CALLBACK_RSP(OnRspQryInstrument, CThostFtdcInstrumentField, "Instrument", "on_rsp_qry_instrument")
    CAPSULE_CALLBACK_RSP(OnRspQrySettlementInfo, CThostFtdcSettlementInfoField, "SettlementInfo", "on_rsp_qry_settlement_info")
    CAPSULE_CALLBACK_RSP(OnRspSettlementInfoConfirm, CThostFtdcSettlementInfoConfirmField, "SettlementInfoConfirm", "on_rsp_settlement_info_confirm")
    CAPSULE_CALLBACK_RSP(OnRspQrySettlementInfoConfirm, CThostFtdcSettlementInfoConfirmField, "SettlementInfoConfirm", "on_rsp_qry_settlement_info_confirm")  // ⚠️ 查询版本
    CAPSULE_CALLBACK_RSP(OnRspQryExchange, CThostFtdcExchangeField, "Exchange", "on_rsp_qry_exchange")
    CAPSULE_CALLBACK_RSP(OnRspQryProduct, CThostFtdcProductField, "Product", "on_rsp_qry_product")
    CAPSULE_CALLBACK_RSP(OnRspQryDepthMarketData, CThostFtdcDepthMarketDataField, "DepthMarketData", "on_rsp_qry_depth_market_data")

    // //////////////////////////////////////////////////////////////////////////
    // 通知相关回调
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_SINGLE(OnRtnTradingNotice, CThostFtdcTradingNoticeInfoField, "TradingNoticeInfo", "on_rtn_trading_notice")
    CAPSULE_CALLBACK_SINGLE(OnRtnBulletin, CThostFtdcBulletinField, "Bulletin", "on_rtn_bulletin")
    CAPSULE_CALLBACK_SINGLE(OnRtnInstrumentStatus, CThostFtdcInstrumentStatusField, "InstrumentStatus", "on_rtn_instrument_status")

    // //////////////////////////////////////////////////////////////////////////
    // 高级认证相关回调 (SPI-02)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspUserAuthMethod, CThostFtdcRspUserAuthMethodField, "RspUserAuthMethod", "on_rsp_user_auth_method")
    CAPSULE_CALLBACK_RSP(OnRspGenUserCaptcha, CThostFtdcRspGenUserCaptchaField, "RspGenUserCaptcha", "on_rsp_gen_user_captcha")
    CAPSULE_CALLBACK_RSP(OnRspGenUserText, CThostFtdcRspGenUserTextField, "RspGenUserText", "on_rsp_gen_user_text")

    // //////////////////////////////////////////////////////////////////////////
    // 报单查询相关回调 (SPI-02)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQryMaxOrderVolume, CThostFtdcQryMaxOrderVolumeField, "QryMaxOrderVolume", "on_rsp_qry_max_order_volume")
    CAPSULE_CALLBACK_RSP(OnRspRemoveParkedOrder, CThostFtdcRemoveParkedOrderField, "RemoveParkedOrder", "on_rsp_remove_parked_order")
    CAPSULE_CALLBACK_RSP(OnRspRemoveParkedOrderAction, CThostFtdcRemoveParkedOrderActionField, "RemoveParkedOrderAction", "on_rsp_remove_parked_order_action")

    // //////////////////////////////////////////////////////////////////////////
    // 组合/期权相关回调 (SPI-02)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspExecOrderInsert, CThostFtdcInputExecOrderField, "InputExecOrder", "on_rsp_exec_order_insert")
    CAPSULE_CALLBACK_RSP(OnRspExecOrderAction, CThostFtdcInputExecOrderActionField, "InputExecOrderAction", "on_rsp_exec_order_action")
    CAPSULE_CALLBACK_RSP(OnRspForQuoteInsert, CThostFtdcInputForQuoteField, "InputForQuote", "on_rsp_for_quote_insert")
    CAPSULE_CALLBACK_RSP(OnRspQuoteInsert, CThostFtdcInputQuoteField, "InputQuote", "on_rsp_quote_insert")
    CAPSULE_CALLBACK_RSP(OnRspQuoteAction, CThostFtdcInputQuoteActionField, "InputQuoteAction", "on_rsp_quote_action")
    CAPSULE_CALLBACK_RSP(OnRspBatchOrderAction, CThostFtdcInputBatchOrderActionField, "InputBatchOrderAction", "on_rsp_batch_order_action")
    CAPSULE_CALLBACK_RSP(OnRspOptionSelfCloseInsert, CThostFtdcInputOptionSelfCloseField, "InputOptionSelfClose", "on_rsp_option_self_close_insert")
    CAPSULE_CALLBACK_RSP(OnRspOptionSelfCloseAction, CThostFtdcInputOptionSelfCloseActionField, "InputOptionSelfCloseAction", "on_rsp_option_self_close_action")
    CAPSULE_CALLBACK_RSP(OnRspCombActionInsert, CThostFtdcInputCombActionField, "InputCombAction", "on_rsp_comb_action_insert")

    // //////////////////////////////////////////////////////////////////////////
    // 组合/期权回报相关回调 (SPI-02)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_SINGLE(OnRtnExecOrder, CThostFtdcExecOrderField, "ExecOrder", "on_rtn_exec_order")
    CAPSULE_CALLBACK_ERROR(OnErrRtnExecOrderInsert, CThostFtdcInputExecOrderField, "InputExecOrder", "on_err_rtn_exec_order_insert")
    CAPSULE_CALLBACK_ERROR(OnErrRtnExecOrderAction, CThostFtdcExecOrderActionField, "ExecOrderAction", "on_err_rtn_exec_order_action")
    CAPSULE_CALLBACK_ERROR(OnErrRtnForQuoteInsert, CThostFtdcInputForQuoteField, "InputForQuote", "on_err_rtn_for_quote_insert")
    CAPSULE_CALLBACK_SINGLE(OnRtnQuote, CThostFtdcQuoteField, "Quote", "on_rtn_quote")
    CAPSULE_CALLBACK_ERROR(OnErrRtnQuoteInsert, CThostFtdcInputQuoteField, "InputQuote", "on_err_rtn_quote_insert")
    CAPSULE_CALLBACK_ERROR(OnErrRtnQuoteAction, CThostFtdcQuoteActionField, "QuoteAction", "on_err_rtn_quote_action")
    CAPSULE_CALLBACK_SINGLE(OnRtnForQuoteRsp, CThostFtdcForQuoteRspField, "ForQuoteRsp", "on_rtn_for_quote_rsp")
    CAPSULE_CALLBACK_SINGLE(OnRtnCFMMCTradingAccountToken, CThostFtdcCFMMCTradingAccountTokenField, "CFMMCTradingAccountToken", "on_rtn_cfmmc_trading_account_token")
    CAPSULE_CALLBACK_ERROR(OnErrRtnBatchOrderAction, CThostFtdcBatchOrderActionField, "BatchOrderAction", "on_err_rtn_batch_order_action")
    CAPSULE_CALLBACK_SINGLE(OnRtnOptionSelfClose, CThostFtdcOptionSelfCloseField, "OptionSelfClose", "on_rtn_option_self_close")
    CAPSULE_CALLBACK_ERROR(OnErrRtnOptionSelfCloseInsert, CThostFtdcInputOptionSelfCloseField, "InputOptionSelfClose", "on_err_rtn_option_self_close_insert")
    CAPSULE_CALLBACK_ERROR(OnErrRtnOptionSelfCloseAction, CThostFtdcOptionSelfCloseActionField, "OptionSelfCloseAction", "on_err_rtn_option_self_close_action")
    CAPSULE_CALLBACK_SINGLE(OnRtnCombAction, CThostFtdcCombActionField, "CombAction", "on_rtn_comb_action")
    CAPSULE_CALLBACK_ERROR(OnErrRtnCombActionInsert, CThostFtdcInputCombActionField, "InputCombAction", "on_err_rtn_comb_action_insert")

    // //////////////////////////////////////////////////////////////////////////
    // 查询回调第一批 (SPI-03)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQryUserSession, CThostFtdcUserSessionField, "UserSession", "on_rsp_qry_user_session")
    CAPSULE_CALLBACK_RSP(OnRspQryExchangeMarginRate, CThostFtdcExchangeMarginRateField, "ExchangeMarginRate", "on_rsp_qry_exchange_margin_rate")
    CAPSULE_CALLBACK_RSP(OnRspQryExchangeMarginRateAdjust, CThostFtdcExchangeMarginRateAdjustField, "ExchangeMarginRateAdjust", "on_rsp_qry_exchange_margin_rate_adjust")
    CAPSULE_CALLBACK_RSP(OnRspQryExchangeRate, CThostFtdcExchangeRateField, "ExchangeRate", "on_rsp_qry_exchange_rate")
    CAPSULE_CALLBACK_RSP(OnRspQrySecAgentACIDMap, CThostFtdcSecAgentACIDMapField, "SecAgentACIDMap", "on_rsp_qry_sec_agent_acid_map")
    CAPSULE_CALLBACK_RSP(OnRspQryProductExchRate, CThostFtdcProductExchRateField, "ProductExchRate", "on_rsp_qry_product_exch_rate")
    CAPSULE_CALLBACK_RSP(OnRspQryProductGroup, CThostFtdcProductGroupField, "ProductGroup", "on_rsp_qry_product_group")
    CAPSULE_CALLBACK_RSP(OnRspQryMMInstrumentCommissionRate, CThostFtdcMMInstrumentCommissionRateField, "MMInstrumentCommissionRate", "on_rsp_qry_mm_instrument_commission_rate")
    CAPSULE_CALLBACK_RSP(OnRspQryMMOptionInstrCommRate, CThostFtdcMMOptionInstrCommRateField, "MMOptionInstrCommRate", "on_rsp_qry_mm_option_instr_comm_rate")
    CAPSULE_CALLBACK_RSP(OnRspQryInstrumentOrderCommRate, CThostFtdcInstrumentOrderCommRateField, "InstrumentOrderCommRate", "on_rsp_qry_instrument_order_comm_rate")
    CAPSULE_CALLBACK_RSP(OnRspQrySecAgentTradingAccount, CThostFtdcTradingAccountField, "TradingAccount", "on_rsp_qry_sec_agent_trading_account")
    CAPSULE_CALLBACK_RSP(OnRspQrySecAgentCheckMode, CThostFtdcSecAgentCheckModeField, "SecAgentCheckMode", "on_rsp_qry_sec_agent_check_mode")
    CAPSULE_CALLBACK_RSP(OnRspQrySecAgentTradeInfo, CThostFtdcSecAgentTradeInfoField, "SecAgentTradeInfo", "on_rsp_qry_sec_agent_trade_info")
    CAPSULE_CALLBACK_RSP(OnRspQryOptionInstrTradeCost, CThostFtdcOptionInstrTradeCostField, "OptionInstrTradeCost", "on_rsp_qry_option_instr_trade_cost")
    CAPSULE_CALLBACK_RSP(OnRspQryOptionInstrCommRate, CThostFtdcOptionInstrCommRateField, "OptionInstrCommRate", "on_rsp_qry_option_instr_comm_rate")
    CAPSULE_CALLBACK_RSP(OnRspQryExecOrder, CThostFtdcExecOrderField, "ExecOrder", "on_rsp_qry_exec_order")
    CAPSULE_CALLBACK_RSP(OnRspQryForQuote, CThostFtdcForQuoteField, "ForQuote", "on_rsp_qry_for_quote")
    CAPSULE_CALLBACK_RSP(OnRspQryQuote, CThostFtdcQuoteField, "Quote", "on_rsp_qry_quote")
    CAPSULE_CALLBACK_RSP(OnRspQryOptionSelfClose, CThostFtdcOptionSelfCloseField, "OptionSelfClose", "on_rsp_qry_option_self_close")

    // //////////////////////////////////////////////////////////////////////////
    // 查询回调第二批 (SPI-04)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQryInvestUnit, CThostFtdcInvestUnitField, "InvestUnit", "on_rsp_qry_invest_unit")
    CAPSULE_CALLBACK_RSP(OnRspQryCombInstrumentGuard, CThostFtdcCombInstrumentGuardField, "CombInstrumentGuard", "on_rsp_qry_comb_instrument_guard")
    CAPSULE_CALLBACK_RSP(OnRspQryCombAction, CThostFtdcCombActionField, "CombAction", "on_rsp_qry_comb_action")
    CAPSULE_CALLBACK_RSP(OnRspQryTransferSerial, CThostFtdcTransferSerialField, "TransferSerial", "on_rsp_qry_transfer_serial")
    CAPSULE_CALLBACK_RSP(OnRspQryAccountregister, CThostFtdcAccountregisterField, "AccountRegister", "on_rsp_qry_account_register")
    CAPSULE_CALLBACK_RSP(OnRspQryContractBank, CThostFtdcContractBankField, "ContractBank", "on_rsp_qry_contract_bank")
    CAPSULE_CALLBACK_RSP(OnRspQryParkedOrder, CThostFtdcParkedOrderField, "ParkedOrder", "on_rsp_qry_parked_order")
    CAPSULE_CALLBACK_RSP(OnRspQryParkedOrderAction, CThostFtdcParkedOrderActionField, "ParkedOrderAction", "on_rsp_qry_parked_order_action")
    CAPSULE_CALLBACK_RSP(OnRspQryTradingNotice, CThostFtdcTradingNoticeField, "TradingNotice", "on_rsp_qry_trading_notice")
    CAPSULE_CALLBACK_RSP(OnRspQryBrokerTradingParams, CThostFtdcBrokerTradingParamsField, "BrokerTradingParams", "on_rsp_qry_broker_trading_params")
    CAPSULE_CALLBACK_RSP(OnRspQryBrokerTradingAlgos, CThostFtdcBrokerTradingAlgosField, "BrokerTradingAlgos", "on_rsp_qry_broker_trading_algos")
    CAPSULE_CALLBACK_RSP(OnRspQueryCFMMCTradingAccountToken, CThostFtdcQueryCFMMCTradingAccountTokenField, "QueryCFMMCTradingAccountToken", "on_rsp_query_cfmmc_trading_account_token")
    CAPSULE_CALLBACK_RSP(OnRspQryClassifiedInstrument, CThostFtdcInstrumentField, "Instrument", "on_rsp_qry_classified_instrument")
    CAPSULE_CALLBACK_RSP(OnRspQryCombPromotionParam, CThostFtdcCombPromotionParamField, "CombPromotionParam", "on_rsp_qry_comb_promotion_param")
    CAPSULE_CALLBACK_RSP(OnRspOffsetSetting, CThostFtdcInputOffsetSettingField, "InputOffsetSetting", "on_rsp_offset_setting")  // ⚠️ 设置版本（非查询）
    CAPSULE_CALLBACK_RSP(OnRspQryOffsetSetting, CThostFtdcOffsetSettingField, "OffsetSetting", "on_rsp_qry_offset_setting")
    CAPSULE_CALLBACK_RSP(OnRspCancelOffsetSetting, CThostFtdcInputOffsetSettingField, "InputOffsetSetting", "on_rsp_cancel_offset_setting")
    CAPSULE_CALLBACK_SINGLE(OnRtnOffsetSetting, CThostFtdcOffsetSettingField, "OffsetSetting", "on_rtn_offset_setting")
    CAPSULE_CALLBACK_ERROR(OnErrRtnOffsetSetting, CThostFtdcInputOffsetSettingField, "InputOffsetSetting", "on_err_rtn_offset_setting")
    CAPSULE_CALLBACK_ERROR(OnErrRtnCancelOffsetSetting, CThostFtdcCancelOffsetSettingField, "CancelOffsetSetting", "on_err_rtn_cancel_offset_setting")

    // //////////////////////////////////////////////////////////////////////////
    // 查询回调第三批 + 银期转账相关 (SPI-05)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorPositionDetail, CThostFtdcInvestorPositionDetailField, "InvestorPositionDetail", "on_rsp_qry_investor_position_detail")
    CAPSULE_CALLBACK_RSP(OnRspQryNotice, CThostFtdcNoticeField, "Notice", "on_rsp_qry_notice")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorPositionCombineDetail, CThostFtdcInvestorPositionCombineDetailField, "InvestorPositionCombineDetail", "on_rsp_qry_investor_position_combine_detail")
    CAPSULE_CALLBACK_RSP(OnRspQryCFMMCTradingAccountKey, CThostFtdcCFMMCTradingAccountKeyField, "CFMMCTradingAccountKey", "on_rsp_qry_cfmmc_trading_account_key")
    CAPSULE_CALLBACK_RSP(OnRspQryEWarrantOffset, CThostFtdcEWarrantOffsetField, "EWarrantOffset", "on_rsp_qry_ewarrant_offset")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorProductGroupMargin, CThostFtdcInvestorProductGroupMarginField, "InvestorProductGroupMargin", "on_rsp_qry_investor_product_group_margin")
    CAPSULE_CALLBACK_RSP(OnRspQryTransferBank, CThostFtdcTransferBankField, "TransferBank", "on_rsp_qry_transfer_bank")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorInfoCommRec, CThostFtdcInvestorInfoCommRecField, "InvestorInfoCommRec", "on_rsp_qry_investor_info_comm_rec")
    CAPSULE_CALLBACK_RSP(OnRspQryCombLeg, CThostFtdcCombLegField, "CombLeg", "on_rsp_qry_comb_leg")
    CAPSULE_CALLBACK_RSP(OnRspFromBankToFutureByFuture, CThostFtdcReqTransferField, "ReqTransfer", "on_rsp_from_bank_to_future_by_future")
    CAPSULE_CALLBACK_RSP(OnRspFromFutureToBankByFuture, CThostFtdcReqTransferField, "ReqTransfer", "on_rsp_from_future_to_bank_by_future")
    CAPSULE_CALLBACK_RSP(OnRspQueryBankAccountMoneyByFuture, CThostFtdcReqQueryAccountField, "ReqQueryAccount", "on_rsp_query_bank_account_money_by_future")
    CAPSULE_CALLBACK_SINGLE(OnRtnFromBankToFutureByBank, CThostFtdcRspTransferField, "RspTransfer", "on_rtn_from_bank_to_future_by_bank")
    CAPSULE_CALLBACK_SINGLE(OnRtnFromFutureToBankByBank, CThostFtdcRspTransferField, "RspTransfer", "on_rtn_from_future_to_bank_by_bank")
    CAPSULE_CALLBACK_SINGLE(OnRtnRepealFromBankToFutureByBank, CThostFtdcRspRepealField, "RspRepeal", "on_rtn_repeal_from_bank_to_future_by_bank")
    CAPSULE_CALLBACK_SINGLE(OnRtnRepealFromFutureToBankByBank, CThostFtdcRspRepealField, "RspRepeal", "on_rtn_repeal_from_future_to_bank_by_bank")
    CAPSULE_CALLBACK_SINGLE(OnRtnFromBankToFutureByFuture, CThostFtdcRspTransferField, "RspTransfer", "on_rtn_from_bank_to_future_by_future")
    CAPSULE_CALLBACK_SINGLE(OnRtnFromFutureToBankByFuture, CThostFtdcRspTransferField, "RspTransfer", "on_rtn_from_future_to_bank_by_future")
    CAPSULE_CALLBACK_SINGLE(OnRtnRepealFromBankToFutureByFutureManual, CThostFtdcRspRepealField, "RspRepeal", "on_rtn_repeal_from_bank_to_future_by_future_manual")
    CAPSULE_CALLBACK_SINGLE(OnRtnRepealFromFutureToBankByFutureManual, CThostFtdcRspRepealField, "RspRepeal", "on_rtn_repeal_from_future_to_bank_by_future_manual")

    // //////////////////////////////////////////////////////////////////////////
    // 银期转账相关 (SPI-06)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_SINGLE(OnRtnQueryBankBalanceByFuture, CThostFtdcNotifyQueryAccountField, "NotifyQueryAccount", "on_rtn_query_bank_balance_by_future")
    CAPSULE_CALLBACK_ERROR(OnErrRtnBankToFutureByFuture, CThostFtdcReqTransferField, "ReqTransfer", "on_err_rtn_bank_to_future_by_future")
    CAPSULE_CALLBACK_ERROR(OnErrRtnFutureToBankByFuture, CThostFtdcReqTransferField, "ReqTransfer", "on_err_rtn_future_to_bank_by_future")
    CAPSULE_CALLBACK_ERROR(OnErrRtnRepealBankToFutureByFutureManual, CThostFtdcReqRepealField, "ReqRepeal", "on_err_rtn_repeal_bank_to_future_by_future_manual")
    CAPSULE_CALLBACK_ERROR(OnErrRtnRepealFutureToBankByFutureManual, CThostFtdcReqRepealField, "ReqRepeal", "on_err_rtn_repeal_future_to_bank_by_future_manual")
    CAPSULE_CALLBACK_ERROR(OnErrRtnQueryBankBalanceByFuture, CThostFtdcReqQueryAccountField, "ReqQueryAccount", "on_err_rtn_query_bank_balance_by_future")
    CAPSULE_CALLBACK_SINGLE(OnRtnRepealFromBankToFutureByFuture, CThostFtdcRspRepealField, "RspRepeal", "on_rtn_repeal_from_bank_to_future_by_future")
    CAPSULE_CALLBACK_SINGLE(OnRtnRepealFromFutureToBankByFuture, CThostFtdcRspRepealField, "RspRepeal", "on_rtn_repeal_from_future_to_bank_by_future")
    CAPSULE_CALLBACK_SINGLE(OnRtnOpenAccountByBank, CThostFtdcOpenAccountField, "OpenAccount", "on_rtn_open_account_by_bank")
    CAPSULE_CALLBACK_SINGLE(OnRtnCancelAccountByBank, CThostFtdcCancelAccountField, "CancelAccount", "on_rtn_cancel_account_by_bank")
    CAPSULE_CALLBACK_SINGLE(OnRtnChangeAccountByBank, CThostFtdcChangeAccountField, "ChangeAccount", "on_rtn_change_account_by_bank")
    CAPSULE_CALLBACK_SINGLE(OnRtnErrorConditionalOrder, CThostFtdcErrorConditionalOrderField, "ErrorConditionalOrder", "on_rtn_error_conditional_order")

    // //////////////////////////////////////////////////////////////////////////
    // SPBM 风控回调 (SPI-07)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQrySPBMFutureParameter, CThostFtdcSPBMFutureParameterField, "SPBMFutureParameter", "on_rsp_qry_spbm_future_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQrySPBMOptionParameter, CThostFtdcSPBMOptionParameterField, "SPBMOptionParameter", "on_rsp_qry_spbm_option_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQrySPBMIntraParameter, CThostFtdcSPBMIntraParameterField, "SPBMIntraParameter", "on_rsp_qry_spbm_intra_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQrySPBMInterParameter, CThostFtdcSPBMInterParameterField, "SPBMInterParameter", "on_rsp_qry_spbm_inter_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQrySPBMPortfDefinition, CThostFtdcSPBMPortfDefinitionField, "SPBMPortfDefinition", "on_rsp_qry_spbm_portf_definition")
    CAPSULE_CALLBACK_RSP(OnRspQrySPBMInvestorPortfDef, CThostFtdcSPBMInvestorPortfDefField, "SPBMInvestorPortfDef", "on_rsp_qry_spbm_investor_portf_def")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorPortfMarginRatio, CThostFtdcInvestorPortfMarginRatioField, "InvestorPortfMarginRatio", "on_rsp_qry_investor_portf_margin_ratio")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorProdSPBMDetail, CThostFtdcInvestorProdSPBMDetailField, "InvestorProdSPBMDetail", "on_rsp_qry_investor_prod_spbm_detail")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorCommoditySPMMMargin, CThostFtdcInvestorCommoditySPMMMarginField, "InvestorCommoditySPMMMargin", "on_rsp_qry_investor_commodity_spmm_margin")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorCommodityGroupSPMMMargin, CThostFtdcInvestorCommodityGroupSPMMMarginField, "InvestorCommodityGroupSPMMMargin", "on_rsp_qry_investor_commodity_group_spmm_margin")
    CAPSULE_CALLBACK_RSP(OnRspQrySPMMInstParam, CThostFtdcSPMMInstParamField, "SPMMInstParam", "on_rsp_qry_spmm_inst_param")
    CAPSULE_CALLBACK_RSP(OnRspQrySPMMProductParam, CThostFtdcSPMMProductParamField, "SPMMProductParam", "on_rsp_qry_spmm_product_param")
    CAPSULE_CALLBACK_RSP(OnRspQrySPBMAddOnInterParameter, CThostFtdcSPBMAddOnInterParameterField, "SPBMAddOnInterParameter", "on_rsp_qry_spbm_add_on_inter_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQryRCAMSCombProductInfo, CThostFtdcRCAMSCombProductInfoField, "RCAMSCombProductInfo", "on_rsp_qry_rcams_comb_product_info")
    CAPSULE_CALLBACK_RSP(OnRspQryRCAMSInstrParameter, CThostFtdcRCAMSInstrParameterField, "RCAMSInstrParameter", "on_rsp_qry_rcams_instr_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQryRCAMSIntraParameter, CThostFtdcRCAMSIntraParameterField, "RCAMSIntraParameter", "on_rsp_qry_rcams_intra_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQryRCAMSInterParameter, CThostFtdcRCAMSInterParameterField, "RCAMSInterParameter", "on_rsp_qry_rcams_inter_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQryRCAMSShortOptAdjustParam, CThostFtdcRCAMSShortOptAdjustParamField, "RCAMSShortOptAdjustParam", "on_rsp_qry_rcams_short_opt_adjust_param")
    CAPSULE_CALLBACK_RSP(OnRspQryRCAMSInvestorCombPosition, CThostFtdcRCAMSInvestorCombPositionField, "RCAMSInvestorCombPosition", "on_rsp_qry_rcams_investor_comb_position")

    // //////////////////////////////////////////////////////////////////////////
    // RCAMS/RULE 风控回调 (SPI-08)
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorProdRCAMSMargin, CThostFtdcInvestorProdRCAMSMarginField, "InvestorProdRCAMSMargin", "on_rsp_qry_investor_prod_rcams_margin")
    CAPSULE_CALLBACK_RSP(OnRspQryRULEInstrParameter, CThostFtdcRULEInstrParameterField, "RULEInstrParameter", "on_rsp_qry_rule_instr_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQryRULEIntraParameter, CThostFtdcRULEIntraParameterField, "RULEIntraParameter", "on_rsp_qry_rule_intra_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQryRULEInterParameter, CThostFtdcRULEInterParameterField, "RULEInterParameter", "on_rsp_qry_rule_inter_parameter")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorProdRULEMargin, CThostFtdcInvestorProdRULEMarginField, "InvestorProdRULEMargin", "on_rsp_qry_investor_prod_rule_margin")
    CAPSULE_CALLBACK_RSP(OnRspQryInvestorPortfSetting, CThostFtdcInvestorPortfSettingField, "InvestorPortfSetting", "on_rsp_qry_investor_portf_setting")

    // //////////////////////////////////////////////////////////////////////////
    // 风险结算回调 (SPI-09) - 含命名转换特例
    // 注意：OnRspQryInvestorInfoCommRec 和 OnRspQryCombLeg 已在 SPI-05 中实现，此处不重复定义
    // //////////////////////////////////////////////////////////////////////////
    CAPSULE_CALLBACK_RSP(OnRspQryTraderOffer, CThostFtdcTraderOfferField, "TraderOffer", "on_rsp_qry_trader_offer")
    CAPSULE_CALLBACK_RSP(OnRspQryRiskSettleProductStatus, CThostFtdcRiskSettleProductStatusField, "QryRiskSettleProductStatus", "on_rsp_qry_risk_settle_product_status")
    CAPSULE_CALLBACK_RSP(OnRspQryRiskSettleInvstPosition, CThostFtdcRiskSettleInvstPositionField, "QryRiskSettleInvestPosition", "on_rsp_qry_risk_settle_invest_position")  // ⚠️ 特例：Invst → Invest

    // 注意：其他 100+ 个回调方法在 trade_spi.cpp 中使用宏批量实现
};

// =============================================================================
// Python类型声明
// =============================================================================

extern PyTypeObject TradeApiType;

// =============================================================================
// 模块初始化函数
// =============================================================================

PyMODINIT_FUNC PyInit_PcCTP_trade(void);
