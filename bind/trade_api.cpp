/**
 * PcCTP - CTP PC版交易API Python绑定 - API方法实现
 *
 * 本文件包含 TradeApi 的所有 Python 方法实现和模块初始化代码
 * 使用 common.h 中的宏来简化重复代码
 */

#include "trade.h"

// =============================================================================
// TradeApi 方法实现 (使用下划线命名法 + 内联优化)
// =============================================================================

/**
 * @brief 创建TradeApi实例 (静态方法)
 * Python命名: create_ftdc_trader_api
 */
static PyObject* TradeApi_create_ftdc_trader_api(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* flow_path = "";
    int is_production_mode = 1;

    static char* kwlist[] = {"flow_path", "is_production_mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|si", kwlist, &flow_path, &is_production_mode)) {
        return NULL;
    }

    PyTypeObject* type = (PyTypeObject*)self;
    TradeApiObject* obj = (TradeApiObject*)type->tp_alloc(type, 0);
    if (!obj) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate TradeApi object");
        return NULL;
    }

    obj->api = CThostFtdcTraderApi::CreateFtdcTraderApi(flow_path, is_production_mode != 0);
    obj->spi = nullptr;
    obj->py_spi = nullptr;

    if (!obj->api) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create CTP TradeApi");
        return NULL;
    }

    return (PyObject*)obj;
}

// 使用 common.h 中的宏定义基础方法
// 直接传递类名，不使用中间宏（避免宏展开问题）

DEF_API_GET_VERSION(TradeApi, CThostFtdcTraderApi)
DEF_API_INIT(TradeApi, TradeApiObject)
DEF_API_JOIN(TradeApi, TradeApiObject)
DEF_API_GET_TRADING_DAY(TradeApi, TradeApiObject)
DEF_API_REGISTER_FRONT(TradeApi, TradeApiObject)
DEF_API_REGISTER_NAME_SERVER(TradeApi, TradeApiObject)
DEF_API_REGISTER_SPI(TradeApi, TradeApiObject, PyTradeSpi)
DEF_API_RELEASE(TradeApi, TradeApiObject, PyTradeSpi)
DEF_API_DEALLOC(TradeApiObject, TradeApi, PyTradeSpi)
DEF_API_NEW_INIT(TradeApi, TradeApiObject)

// =============================================================================
// TradeApi 特有方法（简化版 - 只包含核心方法）
// =============================================================================

/**
 * @brief 获取前置信息
 */
static PyObject* TradeApi_get_front_info(PyObject* self, PyObject* args) {
    TradeApiObject* obj = (TradeApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "TradeApi not initialized");
        return NULL;
    }

    PyErr_SetString(PyExc_NotImplementedError, "get_front_id is not implemented in CTP API");
    return NULL;
}

/**
 * @brief 注册用户信息（零拷贝优化）
 * Python命名: register_fens_user_info
 *
 * 零拷贝实现：
 * - 优先使用 _capsule（零拷贝路径）
 * - 如果 _capsule 不存在，降级到 _struct + memcpy（兼容路径）
 */
static PyObject* TradeApi_register_fens_user_info(PyObject* self, PyObject* args) {
    TradeApiObject* obj = (TradeApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "TradeApi not initialized");
        return NULL;
    }

    PyObject* fens_user_info_obj;
    if (!PyArg_ParseTuple(args, "O", &fens_user_info_obj)) {
        return NULL;
    }

    CThostFtdcFensUserInfoField* fens_user_info_ptr = nullptr;

    // 优先使用 _capsule（零拷贝路径）
    PyObject* capsule = PyObject_GetAttrString(fens_user_info_obj, "_capsule");
    if (capsule && PyCapsule_CheckExact(capsule)) {
        fens_user_info_ptr = static_cast<CThostFtdcFensUserInfoField*>(
            PyCapsule_GetPointer(capsule, "FensUserInfo"));
        Py_DECREF(capsule);
    }
    // 降级到 _struct（兼容路径，有拷贝）
    else {
        Py_XDECREF(capsule);
        PyObject* struct_obj = PyObject_GetAttrString(fens_user_info_obj, "_struct");
        if (struct_obj) {
            PyObject* buffer = PyObject_CallMethod(struct_obj, "__bytes__", NULL);
            Py_DECREF(struct_obj);
            if (buffer) {
                static CThostFtdcFensUserInfoField temp_struct;
                char* buffer_ptr;
                Py_ssize_t buffer_len;
                if (PyBytes_AsStringAndSize(buffer, &buffer_ptr, &buffer_len) == 0 &&
                    static_cast<size_t>(buffer_len) >= sizeof(CThostFtdcFensUserInfoField)) {
                    memcpy(&temp_struct, buffer_ptr, sizeof(CThostFtdcFensUserInfoField));
                    fens_user_info_ptr = &temp_struct;
                }
                Py_DECREF(buffer);
            }
        }
    }

    if (!fens_user_info_ptr) {
        PyErr_SetString(PyExc_TypeError,
            "fens_user_info must be a FensUserInfo object with _capsule or _struct attribute");
        return NULL;
    }

    obj->api->RegisterFensUserInfo(fens_user_info_ptr);
    Py_RETURN_NONE;
}

/**
 * @brief 订阅私有主题
 * Python命名: subscribe_private_topic
 */
static PyObject* TradeApi_subscribe_private_topic(PyObject* self, PyObject* args) {
    TradeApiObject* obj = (TradeApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "TradeApi not initialized");
        return NULL;
    }

    int resume_type;
    if (!PyArg_ParseTuple(args, "i", &resume_type)) {
        return NULL;
    }

    obj->api->SubscribePrivateTopic((THOST_TE_RESUME_TYPE)resume_type);
    Py_RETURN_NONE;
}

/**
 * @brief 订阅公共主题
 * Python命名: subscribe_public_topic
 */
static PyObject* TradeApi_subscribe_public_topic(PyObject* self, PyObject* args) {
    TradeApiObject* obj = (TradeApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "TradeApi not initialized");
        return NULL;
    }

    int resume_type;
    if (!PyArg_ParseTuple(args, "i", &resume_type)) {
        return NULL;
    }

    obj->api->SubscribePublicTopic((THOST_TE_RESUME_TYPE)resume_type);
    Py_RETURN_NONE;
}

// 使用通用请求方法宏定义主要的请求方法
// 直接传递类名，不使用中间宏（避免宏展开问题）

// ============================================================================
// API-01: 基础方法 + 核心请求方法（已实现）
// ============================================================================

// 认证和登录相关
DEF_REQ_METHOD_CAPSULE(TradeApi, req_authenticate, ReqAuthenticate, CThostFtdcReqAuthenticateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_user_login, ReqUserLogin, CThostFtdcReqUserLoginField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_user_logout, ReqUserLogout, CThostFtdcUserLogoutField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_user_password_update, ReqUserPasswordUpdate, CThostFtdcUserPasswordUpdateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_trading_account_password_update, ReqTradingAccountPasswordUpdate, CThostFtdcTradingAccountPasswordUpdateField)

// 用户系统信息相关（注意：这些方法不接受 request_id 参数，需要手动实现）
// DEF_REQ_METHOD_CAPSULE 不适用于这些方法，因为它们只有 1 个参数

// 报单相关
DEF_REQ_METHOD_CAPSULE(TradeApi, req_order_insert, ReqOrderInsert, CThostFtdcInputOrderField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_order_action, ReqOrderAction, CThostFtdcInputOrderActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_parked_order_insert, ReqParkedOrderInsert, CThostFtdcParkedOrderField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_parked_order_action, ReqParkedOrderAction, CThostFtdcParkedOrderActionField)

// 查询相关
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_order, ReqQryOrder, CThostFtdcQryOrderField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_trade, ReqQryTrade, CThostFtdcQryTradeField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_position, ReqQryInvestorPosition, CThostFtdcQryInvestorPositionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_trading_account, ReqQryTradingAccount, CThostFtdcQryTradingAccountField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor, ReqQryInvestor, CThostFtdcQryInvestorField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_trading_code, ReqQryTradingCode, CThostFtdcQryTradingCodeField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_instrument, ReqQryInstrument, CThostFtdcQryInstrumentField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_settlement_info, ReqQrySettlementInfo, CThostFtdcQrySettlementInfoField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_depth_market_data, ReqQryDepthMarketData, CThostFtdcQryDepthMarketDataField)

// 结算相关
DEF_REQ_METHOD_CAPSULE(TradeApi, req_settlement_info_confirm, ReqSettlementInfoConfirm, CThostFtdcSettlementInfoConfirmField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_settlement_info_confirm, ReqQrySettlementInfoConfirm, CThostFtdcQrySettlementInfoConfirmField)

// ============================================================================
// API-02: 高级认证 + 报单请求方法（新增）
// ============================================================================

// 高级认证相关（注意：微信系统信息注册方法不接受 request_id 参数）
// DEF_REQ_METHOD_CAPSULE 不适用于这些方法

// 报单查询和管理
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_max_order_volume, ReqQryMaxOrderVolume, CThostFtdcQryMaxOrderVolumeField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_remove_parked_order, ReqRemoveParkedOrder, CThostFtdcRemoveParkedOrderField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_remove_parked_order_action, ReqRemoveParkedOrderAction, CThostFtdcRemoveParkedOrderActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_batch_order_action, ReqBatchOrderAction, CThostFtdcInputBatchOrderActionField)

// ============================================================================
// API-03: 组合/期权相关方法
// ============================================================================
DEF_REQ_METHOD_CAPSULE(TradeApi, req_exec_order_insert, ReqExecOrderInsert, CThostFtdcInputExecOrderField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_exec_order_action, ReqExecOrderAction, CThostFtdcInputExecOrderActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_for_quote_insert, ReqForQuoteInsert, CThostFtdcInputForQuoteField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_quote_insert, ReqQuoteInsert, CThostFtdcInputQuoteField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_quote_action, ReqQuoteAction, CThostFtdcInputQuoteActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_option_self_close_insert, ReqOptionSelfCloseInsert, CThostFtdcInputOptionSelfCloseField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_option_self_close_action, ReqOptionSelfCloseAction, CThostFtdcInputOptionSelfCloseActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_comb_action_insert, ReqCombActionInsert, CThostFtdcInputCombActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_offset_setting, ReqOffsetSetting, CThostFtdcInputOffsetSettingField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_cancel_offset_setting, ReqCancelOffsetSetting, CThostFtdcInputOffsetSettingField)

// ============================================================================
// API-04: 查询方法（交易所、产品、合约等）
// ============================================================================
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_exchange, ReqQryExchange, CThostFtdcQryExchangeField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_product, ReqQryProduct, CThostFtdcQryProductField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_instrument_margin_rate, ReqQryInstrumentMarginRate, CThostFtdcQryInstrumentMarginRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_instrument_commission_rate, ReqQryInstrumentCommissionRate, CThostFtdcQryInstrumentCommissionRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_user_session, ReqQryUserSession, CThostFtdcQryUserSessionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_exchange_margin_rate, ReqQryExchangeMarginRate, CThostFtdcQryExchangeMarginRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_exchange_margin_rate_adjust, ReqQryExchangeMarginRateAdjust, CThostFtdcQryExchangeMarginRateAdjustField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_exchange_rate, ReqQryExchangeRate, CThostFtdcQryExchangeRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_sec_agent_acid_map, ReqQrySecAgentACIDMap, CThostFtdcQrySecAgentACIDMapField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_product_exch_rate, ReqQryProductExchRate, CThostFtdcQryProductExchRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_product_group, ReqQryProductGroup, CThostFtdcQryProductGroupField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_mm_instrument_commission_rate, ReqQryMMInstrumentCommissionRate, CThostFtdcQryMMInstrumentCommissionRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_mm_option_instr_comm_rate, ReqQryMMOptionInstrCommRate, CThostFtdcQryMMOptionInstrCommRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_instrument_order_comm_rate, ReqQryInstrumentOrderCommRate, CThostFtdcQryInstrumentOrderCommRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_sec_agent_trading_account, ReqQrySecAgentTradingAccount, CThostFtdcQryTradingAccountField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_sec_agent_check_mode, ReqQrySecAgentCheckMode, CThostFtdcQrySecAgentCheckModeField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_sec_agent_trade_info, ReqQrySecAgentTradeInfo, CThostFtdcQrySecAgentTradeInfoField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_option_instr_trade_cost, ReqQryOptionInstrTradeCost, CThostFtdcQryOptionInstrTradeCostField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_option_instr_comm_rate, ReqQryOptionInstrCommRate, CThostFtdcQryOptionInstrCommRateField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_exec_order, ReqQryExecOrder, CThostFtdcQryExecOrderField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_for_quote, ReqQryForQuote, CThostFtdcQryForQuoteField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_quote, ReqQryQuote, CThostFtdcQryQuoteField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_option_self_close, ReqQryOptionSelfClose, CThostFtdcQryOptionSelfCloseField)

// ============================================================================
// API-05: 查询方法（投资者单位、组合等）
// ============================================================================
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_invest_unit, ReqQryInvestUnit, CThostFtdcQryInvestUnitField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_comb_instrument_guard, ReqQryCombInstrumentGuard, CThostFtdcQryCombInstrumentGuardField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_comb_action, ReqQryCombAction, CThostFtdcQryCombActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_transfer_serial, ReqQryTransferSerial, CThostFtdcQryTransferSerialField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_contract_bank, ReqQryContractBank, CThostFtdcQryContractBankField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_parked_order, ReqQryParkedOrder, CThostFtdcQryParkedOrderField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_parked_order_action, ReqQryParkedOrderAction, CThostFtdcQryParkedOrderActionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_trading_notice, ReqQryTradingNotice, CThostFtdcQryTradingNoticeField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_broker_trading_params, ReqQryBrokerTradingParams, CThostFtdcQryBrokerTradingParamsField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_broker_trading_algos, ReqQryBrokerTradingAlgos, CThostFtdcQryBrokerTradingAlgosField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_query_cfmmc_trading_account_token, ReqQueryCFMMCTradingAccountToken, CThostFtdcQueryCFMMCTradingAccountTokenField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_classified_instrument, ReqQryClassifiedInstrument, CThostFtdcQryClassifiedInstrumentField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_comb_promotion_param, ReqQryCombPromotionParam, CThostFtdcQryCombPromotionParamField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_offset_setting, ReqQryOffsetSetting, CThostFtdcQryOffsetSettingField)

// ============================================================================
// API-06: 查询方法（结算、持仓详情、转账等）
// ============================================================================
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_transfer_bank, ReqQryTransferBank, CThostFtdcQryTransferBankField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_position_detail, ReqQryInvestorPositionDetail, CThostFtdcQryInvestorPositionDetailField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_notice, ReqQryNotice, CThostFtdcQryNoticeField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_position_combine_detail, ReqQryInvestorPositionCombineDetail, CThostFtdcQryInvestorPositionCombineDetailField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_cfmmc_trading_account_key, ReqQryCFMMCTradingAccountKey, CThostFtdcQryCFMMCTradingAccountKeyField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_ewarrant_offset, ReqQryEWarrantOffset, CThostFtdcQryEWarrantOffsetField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_product_group_margin, ReqQryInvestorProductGroupMargin, CThostFtdcQryInvestorProductGroupMarginField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_from_bank_to_future_by_future, ReqFromBankToFutureByFuture, CThostFtdcReqTransferField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_from_future_to_bank_by_future, ReqFromFutureToBankByFuture, CThostFtdcReqTransferField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_query_bank_account_money_by_future, ReqQueryBankAccountMoneyByFuture, CThostFtdcReqQueryAccountField)

// ============================================================================
// API-07: SPBM/SPMM 查询方法
// ============================================================================
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spbm_future_parameter, ReqQrySPBMFutureParameter, CThostFtdcQrySPBMFutureParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spbm_option_parameter, ReqQrySPBMOptionParameter, CThostFtdcQrySPBMOptionParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spbm_intra_parameter, ReqQrySPBMIntraParameter, CThostFtdcQrySPBMIntraParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spbm_inter_parameter, ReqQrySPBMInterParameter, CThostFtdcQrySPBMInterParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spbm_portf_definition, ReqQrySPBMPortfDefinition, CThostFtdcQrySPBMPortfDefinitionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spbm_investor_portf_def, ReqQrySPBMInvestorPortfDef, CThostFtdcQrySPBMInvestorPortfDefField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_portf_margin_ratio, ReqQryInvestorPortfMarginRatio, CThostFtdcQryInvestorPortfMarginRatioField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_prod_spbm_detail, ReqQryInvestorProdSPBMDetail, CThostFtdcQryInvestorProdSPBMDetailField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_commodity_spmm_margin, ReqQryInvestorCommoditySPMMMargin, CThostFtdcQryInvestorCommoditySPMMMarginField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_commodity_group_spmm_margin, ReqQryInvestorCommodityGroupSPMMMargin, CThostFtdcQryInvestorCommodityGroupSPMMMarginField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spmm_inst_param, ReqQrySPMMInstParam, CThostFtdcQrySPMMInstParamField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spmm_product_param, ReqQrySPMMProductParam, CThostFtdcQrySPMMProductParamField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_spbm_add_on_inter_parameter, ReqQrySPBMAddOnInterParameter, CThostFtdcQrySPBMAddOnInterParameterField)

// ============================================================================
// API-08: RCAMS/RULE 查询方法
// ============================================================================
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rcams_comb_product_info, ReqQryRCAMSCombProductInfo, CThostFtdcQryRCAMSCombProductInfoField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rcams_instr_parameter, ReqQryRCAMSInstrParameter, CThostFtdcQryRCAMSInstrParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rcams_intra_parameter, ReqQryRCAMSIntraParameter, CThostFtdcQryRCAMSIntraParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rcams_inter_parameter, ReqQryRCAMSInterParameter, CThostFtdcQryRCAMSInterParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rcams_short_opt_adjust_param, ReqQryRCAMSShortOptAdjustParam, CThostFtdcQryRCAMSShortOptAdjustParamField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rcams_investor_comb_position, ReqQryRCAMSInvestorCombPosition, CThostFtdcQryRCAMSInvestorCombPositionField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_prod_rcams_margin, ReqQryInvestorProdRCAMSMargin, CThostFtdcQryInvestorProdRCAMSMarginField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rule_instr_parameter, ReqQryRULEInstrParameter, CThostFtdcQryRULEInstrParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rule_intra_parameter, ReqQryRULEIntraParameter, CThostFtdcQryRULEIntraParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_rule_inter_parameter, ReqQryRULEInterParameter, CThostFtdcQryRULEInterParameterField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_prod_rule_margin, ReqQryInvestorProdRULEMargin, CThostFtdcQryInvestorProdRULEMarginField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_portf_setting, ReqQryInvestorPortfSetting, CThostFtdcQryInvestorPortfSettingField)

// ============================================================================
// API-09: 其他查询方法（通信记录、组合腿、银期转账、风险结算）
// ============================================================================
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_investor_info_comm_rec, ReqQryInvestorInfoCommRec, CThostFtdcQryInvestorInfoCommRecField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_comb_leg, ReqQryCombLeg, CThostFtdcQryCombLegField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_trader_offer, ReqQryTraderOffer, CThostFtdcQryTraderOfferField)
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_risk_settle_product_status, ReqQryRiskSettleProductStatus, CThostFtdcQryRiskSettleProductStatusField)
// ⚠️ 特例：register → Register
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_account_register, ReqQryAccountregister, CThostFtdcQryAccountregisterField)
// ⚠️ 特例：Invst → Invest
DEF_REQ_METHOD_CAPSULE(TradeApi, req_qry_risk_settle_invest_position, ReqQryRiskSettleInvstPosition, CThostFtdcQryRiskSettleInvstPositionField)

// =============================================================================
// Python类型定义
// =============================================================================

static PyMethodDef TradeApi_methods[] = {
    // 基础方法
    {"create_ftdc_trader_api", (PyCFunction)TradeApi_create_ftdc_trader_api, METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     "Create CTP TradeApi instance"},
    {"get_api_version", TradeApi_get_api_version, METH_NOARGS | METH_STATIC,
     "Get CTP API version"},
    {"init", TradeApi_init, METH_NOARGS, "Initialize TradeApi"},
    {"join", TradeApi_join, METH_NOARGS, "Join thread"},
    {"get_trading_day", TradeApi_get_trading_day, METH_NOARGS, "Get trading day"},
    {"get_front_info", TradeApi_get_front_info, METH_NOARGS, "Get front info"},
    {"register_front", TradeApi_register_front, METH_VARARGS, "Register front address"},
    {"register_name_server", TradeApi_register_name_server, METH_VARARGS, "Register name server"},
    {"register_spi", TradeApi_register_spi, METH_VARARGS, "Register SPI callback"},
    {"register_fens_user_info", TradeApi_register_fens_user_info, METH_VARARGS, "Register fens user info"},
    {"subscribe_private_topic", TradeApi_subscribe_private_topic, METH_VARARGS, "Subscribe private topic"},
    {"subscribe_public_topic", TradeApi_subscribe_public_topic, METH_VARARGS, "Subscribe public topic"},
    {"release", TradeApi_release, METH_NOARGS, "Release resources"},

    // API-01: 认证和登录
    {"req_authenticate", (PyCFunction)TradeApi_req_authenticate, METH_VARARGS | METH_KEYWORDS, "Request authenticate"},
    {"req_user_login", (PyCFunction)TradeApi_req_user_login, METH_VARARGS | METH_KEYWORDS, "Request user login"},
    {"req_user_logout", (PyCFunction)TradeApi_req_user_logout, METH_VARARGS | METH_KEYWORDS, "Request user logout"},
    {"req_user_password_update", (PyCFunction)TradeApi_req_user_password_update, METH_VARARGS | METH_KEYWORDS, "Request user password update"},
    {"req_trading_account_password_update", (PyCFunction)TradeApi_req_trading_account_password_update, METH_VARARGS | METH_KEYWORDS, "Request trading account password update"},
    // {"register_user_system_info", (PyCFunction)TradeApi_register_user_system_info, METH_VARARGS | METH_KEYWORDS, "Register user system info"},
    // {"submit_user_system_info", (PyCFunction)TradeApi_submit_user_system_info, METH_VARARGS | METH_KEYWORDS, "Submit user system info"},

    // API-01: 报单相关
    {"req_order_insert", (PyCFunction)TradeApi_req_order_insert, METH_VARARGS | METH_KEYWORDS, "Request order insert"},
    {"req_order_action", (PyCFunction)TradeApi_req_order_action, METH_VARARGS | METH_KEYWORDS, "Request order action"},
    {"req_parked_order_insert", (PyCFunction)TradeApi_req_parked_order_insert, METH_VARARGS | METH_KEYWORDS, "Request parked order insert"},
    {"req_parked_order_action", (PyCFunction)TradeApi_req_parked_order_action, METH_VARARGS | METH_KEYWORDS, "Request parked order action"},

    // API-01: 查询相关
    {"req_qry_order", (PyCFunction)TradeApi_req_qry_order, METH_VARARGS | METH_KEYWORDS, "Request query order"},
    {"req_qry_trade", (PyCFunction)TradeApi_req_qry_trade, METH_VARARGS | METH_KEYWORDS, "Request query trade"},
    {"req_qry_investor_position", (PyCFunction)TradeApi_req_qry_investor_position, METH_VARARGS | METH_KEYWORDS, "Request query investor position"},
    {"req_qry_trading_account", (PyCFunction)TradeApi_req_qry_trading_account, METH_VARARGS | METH_KEYWORDS, "Request query trading account"},
    {"req_qry_investor", (PyCFunction)TradeApi_req_qry_investor, METH_VARARGS | METH_KEYWORDS, "Request query investor"},
    {"req_qry_trading_code", (PyCFunction)TradeApi_req_qry_trading_code, METH_VARARGS | METH_KEYWORDS, "Request query trading code"},
    {"req_qry_instrument", (PyCFunction)TradeApi_req_qry_instrument, METH_VARARGS | METH_KEYWORDS, "Request query instrument"},
    {"req_qry_settlement_info", (PyCFunction)TradeApi_req_qry_settlement_info, METH_VARARGS | METH_KEYWORDS, "Request query settlement info"},
    {"req_qry_depth_market_data", (PyCFunction)TradeApi_req_qry_depth_market_data, METH_VARARGS | METH_KEYWORDS, "Request query depth market data"},

    // API-01: 结算相关
    {"req_settlement_info_confirm", (PyCFunction)TradeApi_req_settlement_info_confirm, METH_VARARGS | METH_KEYWORDS, "Request settlement info confirm"},
    {"req_qry_settlement_info_confirm", (PyCFunction)TradeApi_req_qry_settlement_info_confirm, METH_VARARGS | METH_KEYWORDS, "Request query settlement info confirm"},

    // API-02: 高级认证相关
    // {"req_user_auth_method", (PyCFunction)TradeApi_req_user_auth_method, METH_VARARGS | METH_KEYWORDS, "Request user auth method"},
    // {"req_gen_user_captcha", (PyCFunction)TradeApi_req_gen_user_captcha, METH_VARARGS | METH_KEYWORDS, "Request generate user captcha"},
    // {"req_gen_user_text", (PyCFunction)TradeApi_req_gen_user_text, METH_VARARGS | METH_KEYWORDS, "Request generate user text"},
    // {"req_user_login_with_captcha", (PyCFunction)TradeApi_req_user_login_with_captcha, METH_VARARGS | METH_KEYWORDS, "Request user login with captcha"},
    // {"req_user_login_with_text", (PyCFunction)TradeApi_req_user_login_with_text, METH_VARARGS | METH_KEYWORDS, "Request user login with text"},
    // {"req_user_login_with_otp", (PyCFunction)TradeApi_req_user_login_with_otp, METH_VARARGS | METH_KEYWORDS, "Request user login with OTP"},
    // {"register_wechat_user_system_info", (PyCFunction)TradeApi_register_wechat_user_system_info, METH_VARARGS | METH_KEYWORDS, "Register wechat user system info"},
    // {"submit_wechat_user_system_info", (PyCFunction)TradeApi_submit_wechat_user_system_info, METH_VARARGS | METH_KEYWORDS, "Submit wechat user system info"},

    // API-02: 报单查询和管理
    {"req_qry_max_order_volume", (PyCFunction)TradeApi_req_qry_max_order_volume, METH_VARARGS | METH_KEYWORDS, "Request query max order volume"},
    {"req_remove_parked_order", (PyCFunction)TradeApi_req_remove_parked_order, METH_VARARGS | METH_KEYWORDS, "Request remove parked order"},
    {"req_remove_parked_order_action", (PyCFunction)TradeApi_req_remove_parked_order_action, METH_VARARGS | METH_KEYWORDS, "Request remove parked order action"},
    {"req_batch_order_action", (PyCFunction)TradeApi_req_batch_order_action, METH_VARARGS | METH_KEYWORDS, "Request batch order action"},

    // API-03: 组合/期权相关方法
    {"req_exec_order_insert", (PyCFunction)TradeApi_req_exec_order_insert, METH_VARARGS | METH_KEYWORDS, "Request exec order insert"},
    {"req_exec_order_action", (PyCFunction)TradeApi_req_exec_order_action, METH_VARARGS | METH_KEYWORDS, "Request exec order action"},
    {"req_for_quote_insert", (PyCFunction)TradeApi_req_for_quote_insert, METH_VARARGS | METH_KEYWORDS, "Request for quote insert"},
    {"req_quote_insert", (PyCFunction)TradeApi_req_quote_insert, METH_VARARGS | METH_KEYWORDS, "Request quote insert"},
    {"req_quote_action", (PyCFunction)TradeApi_req_quote_action, METH_VARARGS | METH_KEYWORDS, "Request quote action"},
    {"req_option_self_close_insert", (PyCFunction)TradeApi_req_option_self_close_insert, METH_VARARGS | METH_KEYWORDS, "Request option self close insert"},
    {"req_option_self_close_action", (PyCFunction)TradeApi_req_option_self_close_action, METH_VARARGS | METH_KEYWORDS, "Request option self close action"},
    {"req_comb_action_insert", (PyCFunction)TradeApi_req_comb_action_insert, METH_VARARGS | METH_KEYWORDS, "Request comb action insert"},
    {"req_offset_setting", (PyCFunction)TradeApi_req_offset_setting, METH_VARARGS | METH_KEYWORDS, "Request offset setting"},
    {"req_cancel_offset_setting", (PyCFunction)TradeApi_req_cancel_offset_setting, METH_VARARGS | METH_KEYWORDS, "Request cancel offset setting"},

    // API-04: 查询方法（交易所、产品、合约等）
    {"req_qry_exchange", (PyCFunction)TradeApi_req_qry_exchange, METH_VARARGS | METH_KEYWORDS, "Request query exchange"},
    {"req_qry_product", (PyCFunction)TradeApi_req_qry_product, METH_VARARGS | METH_KEYWORDS, "Request query product"},
    {"req_qry_instrument_margin_rate", (PyCFunction)TradeApi_req_qry_instrument_margin_rate, METH_VARARGS | METH_KEYWORDS, "Request query instrument margin rate"},
    {"req_qry_instrument_commission_rate", (PyCFunction)TradeApi_req_qry_instrument_commission_rate, METH_VARARGS | METH_KEYWORDS, "Request query instrument commission rate"},
    {"req_qry_user_session", (PyCFunction)TradeApi_req_qry_user_session, METH_VARARGS | METH_KEYWORDS, "Request query user session"},
    {"req_qry_exchange_margin_rate", (PyCFunction)TradeApi_req_qry_exchange_margin_rate, METH_VARARGS | METH_KEYWORDS, "Request query exchange margin rate"},
    {"req_qry_exchange_margin_rate_adjust", (PyCFunction)TradeApi_req_qry_exchange_margin_rate_adjust, METH_VARARGS | METH_KEYWORDS, "Request query exchange margin rate adjust"},
    {"req_qry_exchange_rate", (PyCFunction)TradeApi_req_qry_exchange_rate, METH_VARARGS | METH_KEYWORDS, "Request query exchange rate"},
    {"req_qry_sec_agent_acid_map", (PyCFunction)TradeApi_req_qry_sec_agent_acid_map, METH_VARARGS | METH_KEYWORDS, "Request query sec agent acid map"},
    {"req_qry_product_exch_rate", (PyCFunction)TradeApi_req_qry_product_exch_rate, METH_VARARGS | METH_KEYWORDS, "Request query product exch rate"},
    {"req_qry_product_group", (PyCFunction)TradeApi_req_qry_product_group, METH_VARARGS | METH_KEYWORDS, "Request query product group"},
    {"req_qry_mm_instrument_commission_rate", (PyCFunction)TradeApi_req_qry_mm_instrument_commission_rate, METH_VARARGS | METH_KEYWORDS, "Request query mm instrument commission rate"},
    {"req_qry_mm_option_instr_comm_rate", (PyCFunction)TradeApi_req_qry_mm_option_instr_comm_rate, METH_VARARGS | METH_KEYWORDS, "Request query mm option instr comm rate"},
    {"req_qry_instrument_order_comm_rate", (PyCFunction)TradeApi_req_qry_instrument_order_comm_rate, METH_VARARGS | METH_KEYWORDS, "Request query instrument order comm rate"},
    {"req_qry_sec_agent_trading_account", (PyCFunction)TradeApi_req_qry_sec_agent_trading_account, METH_VARARGS | METH_KEYWORDS, "Request query sec agent trading account"},
    {"req_qry_sec_agent_check_mode", (PyCFunction)TradeApi_req_qry_sec_agent_check_mode, METH_VARARGS | METH_KEYWORDS, "Request query sec agent check mode"},
    {"req_qry_sec_agent_trade_info", (PyCFunction)TradeApi_req_qry_sec_agent_trade_info, METH_VARARGS | METH_KEYWORDS, "Request query sec agent trade info"},
    {"req_qry_option_instr_trade_cost", (PyCFunction)TradeApi_req_qry_option_instr_trade_cost, METH_VARARGS | METH_KEYWORDS, "Request query option instr trade cost"},
    {"req_qry_option_instr_comm_rate", (PyCFunction)TradeApi_req_qry_option_instr_comm_rate, METH_VARARGS | METH_KEYWORDS, "Request query option instr comm rate"},
    {"req_qry_exec_order", (PyCFunction)TradeApi_req_qry_exec_order, METH_VARARGS | METH_KEYWORDS, "Request query exec order"},
    {"req_qry_for_quote", (PyCFunction)TradeApi_req_qry_for_quote, METH_VARARGS | METH_KEYWORDS, "Request query for quote"},
    {"req_qry_quote", (PyCFunction)TradeApi_req_qry_quote, METH_VARARGS | METH_KEYWORDS, "Request query quote"},
    {"req_qry_option_self_close", (PyCFunction)TradeApi_req_qry_option_self_close, METH_VARARGS | METH_KEYWORDS, "Request query option self close"},

    // API-05: 查询方法（投资者单位、组合等）
    {"req_qry_invest_unit", (PyCFunction)TradeApi_req_qry_invest_unit, METH_VARARGS | METH_KEYWORDS, "Request query invest unit"},
    {"req_qry_comb_instrument_guard", (PyCFunction)TradeApi_req_qry_comb_instrument_guard, METH_VARARGS | METH_KEYWORDS, "Request query comb instrument guard"},
    {"req_qry_comb_action", (PyCFunction)TradeApi_req_qry_comb_action, METH_VARARGS | METH_KEYWORDS, "Request query comb action"},
    {"req_qry_transfer_serial", (PyCFunction)TradeApi_req_qry_transfer_serial, METH_VARARGS | METH_KEYWORDS, "Request query transfer serial"},
    {"req_qry_contract_bank", (PyCFunction)TradeApi_req_qry_contract_bank, METH_VARARGS | METH_KEYWORDS, "Request query contract bank"},
    {"req_qry_parked_order", (PyCFunction)TradeApi_req_qry_parked_order, METH_VARARGS | METH_KEYWORDS, "Request query parked order"},
    {"req_qry_parked_order_action", (PyCFunction)TradeApi_req_qry_parked_order_action, METH_VARARGS | METH_KEYWORDS, "Request query parked order action"},
    {"req_qry_trading_notice", (PyCFunction)TradeApi_req_qry_trading_notice, METH_VARARGS | METH_KEYWORDS, "Request query trading notice"},
    {"req_qry_broker_trading_params", (PyCFunction)TradeApi_req_qry_broker_trading_params, METH_VARARGS | METH_KEYWORDS, "Request query broker trading params"},
    {"req_qry_broker_trading_algos", (PyCFunction)TradeApi_req_qry_broker_trading_algos, METH_VARARGS | METH_KEYWORDS, "Request query broker trading algos"},
    {"req_query_cfmmc_trading_account_token", (PyCFunction)TradeApi_req_query_cfmmc_trading_account_token, METH_VARARGS | METH_KEYWORDS, "Request query cfmmc trading account token"},
    {"req_qry_classified_instrument", (PyCFunction)TradeApi_req_qry_classified_instrument, METH_VARARGS | METH_KEYWORDS, "Request query classified instrument"},
    {"req_qry_comb_promotion_param", (PyCFunction)TradeApi_req_qry_comb_promotion_param, METH_VARARGS | METH_KEYWORDS, "Request query comb promotion param"},
    {"req_qry_offset_setting", (PyCFunction)TradeApi_req_qry_offset_setting, METH_VARARGS | METH_KEYWORDS, "Request query offset setting"},

    // API-06: 查询方法（结算、持仓详情、转账等）
    {"req_qry_investor_position_detail", (PyCFunction)TradeApi_req_qry_investor_position_detail, METH_VARARGS | METH_KEYWORDS, "Request query investor position detail"},
    {"req_qry_notice", (PyCFunction)TradeApi_req_qry_notice, METH_VARARGS | METH_KEYWORDS, "Request query notice"},
    {"req_qry_investor_position_combine_detail", (PyCFunction)TradeApi_req_qry_investor_position_combine_detail, METH_VARARGS | METH_KEYWORDS, "Request query investor position combine detail"},
    {"req_qry_cfmmc_trading_account_key", (PyCFunction)TradeApi_req_qry_cfmmc_trading_account_key, METH_VARARGS | METH_KEYWORDS, "Request query cfmmc trading account key"},
    {"req_qry_ewarrant_offset", (PyCFunction)TradeApi_req_qry_ewarrant_offset, METH_VARARGS | METH_KEYWORDS, "Request query ewarrant offset"},
    {"req_qry_investor_product_group_margin", (PyCFunction)TradeApi_req_qry_investor_product_group_margin, METH_VARARGS | METH_KEYWORDS, "Request query investor product group margin"},
    {"req_qry_transfer_bank", (PyCFunction)TradeApi_req_qry_transfer_bank, METH_VARARGS | METH_KEYWORDS, "Request query transfer bank"},
    {"req_from_bank_to_future_by_future", (PyCFunction)TradeApi_req_from_bank_to_future_by_future, METH_VARARGS | METH_KEYWORDS, "Request from bank to future by future"},
    {"req_from_future_to_bank_by_future", (PyCFunction)TradeApi_req_from_future_to_bank_by_future, METH_VARARGS | METH_KEYWORDS, "Request from future to bank by future"},
    {"req_query_bank_account_money_by_future", (PyCFunction)TradeApi_req_query_bank_account_money_by_future, METH_VARARGS | METH_KEYWORDS, "Request query bank account money by future"},

    // API-07: SPBM/SPMM 查询方法
    {"req_qry_spbm_future_parameter", (PyCFunction)TradeApi_req_qry_spbm_future_parameter, METH_VARARGS | METH_KEYWORDS, "Request query spbm future parameter"},
    {"req_qry_spbm_option_parameter", (PyCFunction)TradeApi_req_qry_spbm_option_parameter, METH_VARARGS | METH_KEYWORDS, "Request query spbm option parameter"},
    {"req_qry_spbm_intra_parameter", (PyCFunction)TradeApi_req_qry_spbm_intra_parameter, METH_VARARGS | METH_KEYWORDS, "Request query spbm intra parameter"},
    {"req_qry_spbm_inter_parameter", (PyCFunction)TradeApi_req_qry_spbm_inter_parameter, METH_VARARGS | METH_KEYWORDS, "Request query spbm inter parameter"},
    {"req_qry_spbm_portf_definition", (PyCFunction)TradeApi_req_qry_spbm_portf_definition, METH_VARARGS | METH_KEYWORDS, "Request query spbm portf definition"},
    {"req_qry_spbm_investor_portf_def", (PyCFunction)TradeApi_req_qry_spbm_investor_portf_def, METH_VARARGS | METH_KEYWORDS, "Request query spbm investor portf def"},
    {"req_qry_investor_portf_margin_ratio", (PyCFunction)TradeApi_req_qry_investor_portf_margin_ratio, METH_VARARGS | METH_KEYWORDS, "Request query investor portf margin ratio"},
    {"req_qry_investor_prod_spbm_detail", (PyCFunction)TradeApi_req_qry_investor_prod_spbm_detail, METH_VARARGS | METH_KEYWORDS, "Request query investor prod spbm detail"},
    {"req_qry_investor_commodity_spmm_margin", (PyCFunction)TradeApi_req_qry_investor_commodity_spmm_margin, METH_VARARGS | METH_KEYWORDS, "Request query investor commodity spmm margin"},
    {"req_qry_investor_commodity_group_spmm_margin", (PyCFunction)TradeApi_req_qry_investor_commodity_group_spmm_margin, METH_VARARGS | METH_KEYWORDS, "Request query investor commodity group spmm margin"},
    {"req_qry_spmm_inst_param", (PyCFunction)TradeApi_req_qry_spmm_inst_param, METH_VARARGS | METH_KEYWORDS, "Request query spmm inst param"},
    {"req_qry_spmm_product_param", (PyCFunction)TradeApi_req_qry_spmm_product_param, METH_VARARGS | METH_KEYWORDS, "Request query spmm product param"},
    {"req_qry_spbm_add_on_inter_parameter", (PyCFunction)TradeApi_req_qry_spbm_add_on_inter_parameter, METH_VARARGS | METH_KEYWORDS, "Request query spbm add on inter parameter"},

    // API-08: RCAMS/RULE 查询方法
    {"req_qry_rcams_comb_product_info", (PyCFunction)TradeApi_req_qry_rcams_comb_product_info, METH_VARARGS | METH_KEYWORDS, "Request query rcams comb product info"},
    {"req_qry_rcams_instr_parameter", (PyCFunction)TradeApi_req_qry_rcams_instr_parameter, METH_VARARGS | METH_KEYWORDS, "Request query rcams instr parameter"},
    {"req_qry_rcams_intra_parameter", (PyCFunction)TradeApi_req_qry_rcams_intra_parameter, METH_VARARGS | METH_KEYWORDS, "Request query rcams intra parameter"},
    {"req_qry_rcams_inter_parameter", (PyCFunction)TradeApi_req_qry_rcams_inter_parameter, METH_VARARGS | METH_KEYWORDS, "Request query rcams inter parameter"},
    {"req_qry_rcams_short_opt_adjust_param", (PyCFunction)TradeApi_req_qry_rcams_short_opt_adjust_param, METH_VARARGS | METH_KEYWORDS, "Request query rcams short opt adjust param"},
    {"req_qry_rcams_investor_comb_position", (PyCFunction)TradeApi_req_qry_rcams_investor_comb_position, METH_VARARGS | METH_KEYWORDS, "Request query rcams investor comb position"},
    {"req_qry_investor_prod_rcams_margin", (PyCFunction)TradeApi_req_qry_investor_prod_rcams_margin, METH_VARARGS | METH_KEYWORDS, "Request query investor prod rcams margin"},
    {"req_qry_rule_instr_parameter", (PyCFunction)TradeApi_req_qry_rule_instr_parameter, METH_VARARGS | METH_KEYWORDS, "Request query rule instr parameter"},
    {"req_qry_rule_intra_parameter", (PyCFunction)TradeApi_req_qry_rule_intra_parameter, METH_VARARGS | METH_KEYWORDS, "Request query rule intra parameter"},
    {"req_qry_rule_inter_parameter", (PyCFunction)TradeApi_req_qry_rule_inter_parameter, METH_VARARGS | METH_KEYWORDS, "Request query rule inter parameter"},
    {"req_qry_investor_prod_rule_margin", (PyCFunction)TradeApi_req_qry_investor_prod_rule_margin, METH_VARARGS | METH_KEYWORDS, "Request query investor prod rule margin"},
    {"req_qry_investor_portf_setting", (PyCFunction)TradeApi_req_qry_investor_portf_setting, METH_VARARGS | METH_KEYWORDS, "Request query investor portf setting"},

    // API-09: 其他查询方法（通信记录、组合腿、银期转账、风险结算）
    {"req_qry_investor_info_comm_rec", (PyCFunction)TradeApi_req_qry_investor_info_comm_rec, METH_VARARGS | METH_KEYWORDS, "Request query investor info comm rec"},
    {"req_qry_comb_leg", (PyCFunction)TradeApi_req_qry_comb_leg, METH_VARARGS | METH_KEYWORDS, "Request query comb leg"},
    {"req_qry_trader_offer", (PyCFunction)TradeApi_req_qry_trader_offer, METH_VARARGS | METH_KEYWORDS, "Request query trader offer"},
    {"req_qry_risk_settle_product_status", (PyCFunction)TradeApi_req_qry_risk_settle_product_status, METH_VARARGS | METH_KEYWORDS, "Request query risk settle product status"},
    // 特例：register → Register
    {"req_qry_account_register", (PyCFunction)TradeApi_req_qry_account_register, METH_VARARGS | METH_KEYWORDS, "Request query account register"},
    // 特例：Invst → Invest
    {"req_qry_risk_settle_invest_position", (PyCFunction)TradeApi_req_qry_risk_settle_invest_position, METH_VARARGS | METH_KEYWORDS, "Request query risk settle invest position"},

    {NULL}
};

PyTypeObject TradeApiType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "PcCTP.TradeApi",                       /* tp_name */
    sizeof(TradeApiObject),                 /* tp_basicsize */
    0,                                       /* tp_itemsize */
    (destructor)TradeApi_dealloc,           /* tp_dealloc */
    0,                                       /* tp_print */
    0,                                       /* tp_getattr */
    0,                                       /* tp_setattr */
    0,                                       /* tp_reserved */
    0,                                       /* tp_repr */
    0,                                       /* tp_as_number */
    0,                                       /* tp_as_sequence */
    0,                                       /* tp_as_mapping */
    0,                                       /* tp_hash */
    0,                                       /* tp_call */
    0,                                       /* tp_str */
    0,                                       /* tp_getattro */
    0,                                       /* tp_setattro */
    0,                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "CTP PC Trade API (Optimized with Force Inline)",  /* tp_doc */
    0,                                       /* tp_traverse */
    0,                                       /* tp_clear */
    0,                                       /* tp_richcompare */
    0,                                       /* tp_weaklistoffset */
    0,                                       /* tp_iter */
    0,                                       /* tp_iternext */
    TradeApi_methods,                       /* tp_methods */
    0,                                       /* tp_members */
    0,                                       /* tp_getset */
    0,                                       /* tp_base */
    0,                                       /* tp_dict */
    0,                                       /* tp_descr_get */
    0,                                       /* tp_descr_set */
    0,                                       /* tp_dictoffset */
    (initproc)TradeApi_tp_init,             /* tp_init */
    0,                                       /* tp_alloc */
    TradeApi_new,                           /* tp_new */
};

// =============================================================================
// 模块级方法
// =============================================================================

static PyMethodDef pcctp_trade_module_methods[] = {
    PCCTP_POOL_METHODS(),
    PCCTP_POOL_CLEANUP_METHODS(),  // 新增：字符串池清理函数
    ADD_PYCAPSULE_METHODS()
    {NULL}
};

static PyModuleDef PcCTP_trade_module = {
    PyModuleDef_HEAD_INIT,
    "PcCTP.trade",
    "CTP PC Trade API Bindings (Pure Python C API + Force Inline Optimization)",
    -1,
    pcctp_trade_module_methods,
};

PyMODINIT_FUNC PyInit_PcCTP_trade(void) {
    if (initialize_util() < 0) {
        return NULL;
    }

    PyObject* module = PyModule_Create(&PcCTP_trade_module);
    if (!module) {
        return NULL;
    }

    if (PyType_Ready(&TradeApiType) < 0) {
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&TradeApiType);
    if (PyModule_AddObject(module, "TradeApi", (PyObject*)&TradeApiType) < 0) {
        Py_DECREF(&TradeApiType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
