from abc import ABC
from enum import IntEnum
from typing import  Union, List, Dict, Optional
import numpy as np



class ResumeType(IntEnum):
    ...
class CapsuleStruct:
    ...

class DepthMarketData(CapsuleStruct):
    ...

class RspInfo(CapsuleStruct):
    ...

class RspUserLogin(CapsuleStruct):
    ...

class UserLogout(CapsuleStruct):
    ...

class ReqUserLogin(CapsuleStruct):
    ...

class ForQuoteRsp(CapsuleStruct):
    ...

class FensUserInfo(CapsuleStruct):
    ...

class MulticastInstrument(CapsuleStruct):
    ...

class QryMulticastInstrument(CapsuleStruct):
    ...

class RspAuthenticate(CapsuleStruct):
    ...

class UserPasswordUpdate(CapsuleStruct):
    ...

class TradingAccountPasswordUpdate(CapsuleStruct):
    ...

class RspGenUserCaptcha(CapsuleStruct):
    ...

class InputOrder(CapsuleStruct):
    ...

class InputExecOrderAction(CapsuleStruct):
    ...

class InputForQuote(CapsuleStruct):
    ...

class InputQuote(CapsuleStruct):
    ...

class InputQuoteAction(CapsuleStruct):
    ...

class InputBatchOrderAction(CapsuleStruct):
    ...

class InputOptionSelfClose(CapsuleStruct):
    ...

class InputOptionSelfCloseAction(CapsuleStruct):
    ...

class InputCombAction(CapsuleStruct):
    ...

class Order(CapsuleStruct):
    ...

class Trade(CapsuleStruct):
    ...

class InvestorPosition(CapsuleStruct):
    ...

class TradingAccount(CapsuleStruct):
    ...

class Investor(CapsuleStruct):
    ...

class TradingCode(CapsuleStruct):
    ...

class InstrumentMarginRate(CapsuleStruct):
    ...

class InstrumentCommissionRate(CapsuleStruct):
    ...

class UserSession(CapsuleStruct):
    ...

class Exchange(CapsuleStruct):
    ...

class Product(CapsuleStruct):
    ...

class Instrument(CapsuleStruct):
    ...

class SettlementInfo(CapsuleStruct):
    ...

class TransferBank(CapsuleStruct):
    ...

class InvestorPositionDetail(CapsuleStruct):
    ...

class Notice(CapsuleStruct):
    ...

class SettlementInfoConfirm(CapsuleStruct):
    ...

class InvestorPositionCombineDetail(CapsuleStruct):
    ...

class CFMMCTradingAccountKey(CapsuleStruct):
    ...

class EWarrantOffset(CapsuleStruct):
    ...

class InvestorProductGroupMargin(CapsuleStruct):
    ...

class ExchangeMarginRate(CapsuleStruct):
    ...

class ExchangeMarginRateAdjust(CapsuleStruct):
    ...

class ExchangeRate(CapsuleStruct):
    ...

class SecAgentACIDMap(CapsuleStruct):
    ...

class ProductExchRate(CapsuleStruct):
    ...

class ProductGroup(CapsuleStruct):
    ...

class MMInstrumentCommissionRate(CapsuleStruct):
    ...

class MMOptionInstrCommRate(CapsuleStruct):
    ...

class InstrumentOrderCommRate(CapsuleStruct):
    ...

class TradingNoticeInfo(CapsuleStruct):
    ...

class CFMMCTradingAccountToken(CapsuleStruct):
    ...

class ContractBank(CapsuleStruct):
    ...

class InputExecOrder(CapsuleStruct):
    ...

class ExecOrderAction(CapsuleStruct):
    ...

class QuoteAction(CapsuleStruct):
    ...

class BatchOrderAction(CapsuleStruct):
    ...

class OptionSelfCloseAction(CapsuleStruct):
    ...

class ParkedOrder(CapsuleStruct):
    ...

class ParkedOrderAction(CapsuleStruct):
    ...

class ErrorConditionalOrder(CapsuleStruct):
    ...

class SecAgentCheckMode(CapsuleStruct):
    ...

class SecAgentTradeInfo(CapsuleStruct):
    ...

class OptionInstrTradeCost(CapsuleStruct):
    ...

class OptionInstrCommRate(CapsuleStruct):
    ...

class ExecOrder(CapsuleStruct):
    ...

class ForQuote(CapsuleStruct):
    ...

class Quote(CapsuleStruct):
    ...

class OptionSelfClose(CapsuleStruct):
    ...

class InvestUnit(CapsuleStruct):
    ...

class CombInstrumentGuard(CapsuleStruct):
    ...

class CombAction(CapsuleStruct):
    ...

class TransferSerial(CapsuleStruct):
    ...

class OrderAction(CapsuleStruct):
    ...

class InstrumentStatus(CapsuleStruct):
    ...

class Bulletin(CapsuleStruct):
    ...

class TradingNotice(CapsuleStruct):
    ...

class BrokerTradingParams(CapsuleStruct):
    ...

class BrokerTradingAlgos(CapsuleStruct):
    ...

class QueryCFMMCTradingAccountToken(CapsuleStruct):
    ...

class ReqTransfer(CapsuleStruct):
    ...

class RspTransfer(CapsuleStruct):
    ...

class RspRepeal(CapsuleStruct):
    ...

class ReqRepeal(CapsuleStruct):
    ...

class ReqQueryAccount(CapsuleStruct):
    ...

class NotifyQueryAccount(CapsuleStruct):
    ...

class AccountRegister(CapsuleStruct):
    ...

class OffsetSetting(CapsuleStruct):
    ...

class CancelOffsetSetting(CapsuleStruct):
    ...

class InputOffsetSetting(CapsuleStruct):
    ...

class OpenAccount(CapsuleStruct):
    ...

class CancelAccount(CapsuleStruct):
    ...

class ChangeAccount(CapsuleStruct):
    ...

class CombPromotionParam(CapsuleStruct):
    ...

class SPBMFutureParameter(CapsuleStruct):
    ...

class SPBMOptionParameter(CapsuleStruct):
    ...

class SPBMIntraParameter(CapsuleStruct):
    ...

class SPBMInterParameter(CapsuleStruct):
    ...

class SPBMPortfDefinition(CapsuleStruct):
    ...

class SPBMInvestorPortfDef(CapsuleStruct):
    ...

class InvestorPortfMarginRatio(CapsuleStruct):
    ...

class InvestorProdSPBMDetail(CapsuleStruct):
    ...

class InvestorCommoditySPMMMargin(CapsuleStruct):
    ...

class InvestorCommodityGroupSPMMMargin(CapsuleStruct):
    ...

class SPMMInstParam(CapsuleStruct):
    ...

class CombLeg(CapsuleStruct):
    ...

class InvestorInfoCommRec(CapsuleStruct):
    ...

class InvestorPortfSetting(CapsuleStruct):
    ...

class InvestorProdRULEMargin(CapsuleStruct):
    ...

class RULEInterParameter(CapsuleStruct):
    ...

class RULEIntraParameter(CapsuleStruct):
    ...

class RULEInstrParameter(CapsuleStruct):
    ...

class InvestorProdRCAMSMargin(CapsuleStruct):
    ...

class RCAMSInvestorCombPosition(CapsuleStruct):
    ...

class RCAMSShortOptAdjustParam(CapsuleStruct):
    ...

class RCAMSInterParameter(CapsuleStruct):
    ...

class RCAMSIntraParameter(CapsuleStruct):
    ...

class RCAMSInstrParameter(CapsuleStruct):
    ...

class SPBMAddOnInterParameter(CapsuleStruct):
    ...

class RCAMSCombProductInfo(CapsuleStruct):
    ...

class RspGenUserText(CapsuleStruct):
    ...

class RspUserAuthMethod(CapsuleStruct):
    ...

class TraderOffer(CapsuleStruct):
    ...

class RiskSettleInvestPosition(CapsuleStruct):
    ...

class RiskSettleProductStatus(CapsuleStruct):
    ...

class ReqAuthenticate(CapsuleStruct):
    ...

class UserSystemInfo(CapsuleStruct):
    ...

class WechatUserSystemInfo(CapsuleStruct):
    ...

class ReqUserAuthMethod(CapsuleStruct):
    ...

class ReqGenUserCaptcha(CapsuleStruct):
    ...

class ReqGenUserText(CapsuleStruct):
    ...

class ReqUserLoginWithCaptcha(CapsuleStruct):
    ...

class ReqUserLoginWithText(CapsuleStruct):
    ...

class ReqUserLoginWithOTP(CapsuleStruct):
    ...

class InputOrderAction(CapsuleStruct):
    ...

class QryMaxOrderVolume(CapsuleStruct):
    ...

class RemoveParkedOrder(CapsuleStruct):
    ...

class RemoveParkedOrderAction(CapsuleStruct):
    ...

class QryOrder(CapsuleStruct):
    ...

class QryTrade(CapsuleStruct):
    ...

class QryInvestorPosition(CapsuleStruct):
    ...

class QryTradingAccount(CapsuleStruct):
    ...

class QryInvestor(CapsuleStruct):
    ...

class QryTradingCode(CapsuleStruct):
    ...

class QryInstrumentMarginRate(CapsuleStruct):
    ...

class QryInstrumentCommissionRate(CapsuleStruct):
    ...

class QryUserSession(CapsuleStruct):
    ...

class QryExchange(CapsuleStruct):
    ...

class QryProduct(CapsuleStruct):
    ...

class QryInstrument(CapsuleStruct):
    ...

class QryDepthMarketData(CapsuleStruct):
    ...

class QrySettlementInfo(CapsuleStruct):
    ...

class QryTransferBank(CapsuleStruct):
    ...

class QryInvestorPositionDetail(CapsuleStruct):
    ...

class QryNotice(CapsuleStruct):
    ...

class QrySettlementInfoConfirm(CapsuleStruct):
    ...

class QryInvestorPositionCombineDetail(CapsuleStruct):
    ...

class QryCFMMCTradingAccountKey(CapsuleStruct):
    ...

class QryEWarrantOffset(CapsuleStruct):
    ...

class QryInvestorProductGroupMargin(CapsuleStruct):
    ...

class QryExchangeMarginRate(CapsuleStruct):
    ...

class QryExchangeMarginRateAdjust(CapsuleStruct):
    ...

class QryExchangeRate(CapsuleStruct):
    ...

class QrySecAgentACIDMap(CapsuleStruct):
    ...

class QryProductExchRate(CapsuleStruct):
    ...

class QryProductGroup(CapsuleStruct):
    ...

class QryMMInstrumentCommissionRate(CapsuleStruct):
    ...

class QryMMOptionInstrCommRate(CapsuleStruct):
    ...

class QryInstrumentOrderCommRate(CapsuleStruct):
    ...

class QrySecAgentCheckMode(CapsuleStruct):
    ...

class QrySecAgentTradeInfo(CapsuleStruct):
    ...

class QryOptionInstrTradeCost(CapsuleStruct):
    ...

class QryOptionInstrCommRate(CapsuleStruct):
    ...

class QryExecOrder(CapsuleStruct):
    ...

class QryForQuote(CapsuleStruct):
    ...

class QryQuote(CapsuleStruct):
    ...

class QryOptionSelfClose(CapsuleStruct):
    ...

class QryInvestUnit(CapsuleStruct):
    ...

class QryCombInstrumentGuard(CapsuleStruct):
    ...

class QryCombAction(CapsuleStruct):
    ...

class QryTransferSerial(CapsuleStruct):
    ...

class QryAccountRegister(CapsuleStruct):
    ...

class QryContractBank(CapsuleStruct):
    ...

class QryParkedOrder(CapsuleStruct):
    ...

class QryParkedOrderAction(CapsuleStruct):
    ...

class QryTradingNotice(CapsuleStruct):
    ...

class QryBrokerTradingParams(CapsuleStruct):
    ...

class QryBrokerTradingAlgos(CapsuleStruct):
    ...

class QryClassifiedInstrument(CapsuleStruct):
    ...

class QryCombPromotionParam(CapsuleStruct):
    ...

class QryOffsetSetting(CapsuleStruct):
    ...

class QrySPBMFutureParameter(CapsuleStruct):
    ...

class QrySPBMOptionParameter(CapsuleStruct):
    ...

class QrySPBMIntraParameter(CapsuleStruct):
    ...

class QrySPBMInterParameter(CapsuleStruct):
    ...

class QrySPBMPortfDefinition(CapsuleStruct):
    ...

class QrySPBMInvestorPortfDef(CapsuleStruct):
    ...

class QryInvestorPortfMarginRatio(CapsuleStruct):
    ...

class QryInvestorProdSPBMDetail(CapsuleStruct):
    ...

class QryInvestorCommoditySPMMMargin(CapsuleStruct):
    ...

class QryInvestorCommodityGroupSPMMMargin(CapsuleStruct):
    ...

class QrySPMMInstParam(CapsuleStruct):
    ...

class QrySPMMProductParam(CapsuleStruct):
    ...

class QrySPBMAddOnInterParameter(CapsuleStruct):
    ...

class QryRCAMSCombProductInfo(CapsuleStruct):
    ...

class QryRCAMSInstrParameter(CapsuleStruct):
    ...

class QryRCAMSIntraParameter(CapsuleStruct):
    ...

class QryRCAMSInterParameter(CapsuleStruct):
    ...

class QryRCAMSShortOptAdjustParam(CapsuleStruct):
    ...

class QryRCAMSInvestorCombPosition(CapsuleStruct):
    ...

class QryInvestorProdRCAMSMargin(CapsuleStruct):
    ...

class QryRULEInstrParameter(CapsuleStruct):
    ...

class QryRULEIntraParameter(CapsuleStruct):
    ...

class QryRULEInterParameter(CapsuleStruct):
    ...

class QryInvestorProdRULEMargin(CapsuleStruct):
    ...

class QryInvestorPortfSetting(CapsuleStruct):
    ...

class QryInvestorInfoCommRec(CapsuleStruct):
    ...

class QryCombLeg(CapsuleStruct):
    ...

class QryTraderOffer(CapsuleStruct):
    ...

class QryRiskSettleInvestPosition(CapsuleStruct):
    ...

class QryRiskSettleProductStatus(CapsuleStruct):
    ...

class SPMMProductParam(CapsuleStruct):
    ...
# =============================================================================
# PyMdSpi 和 MdApi 类定义（仅在类型检查时可用）
# 这些类由 C++ 模块提供，这里仅定义类型提示用于 IDE 和 mypy
# =============================================================================

class PyMdSpi(ABC):
    """
    CTP PC版行情回调接口协议

    使用 Protocol 的优势：
    - 可以继承此类获得类型提示和 IDE 自动完成
    - 也可以不继承，只需实现相应方法即可（鸭子类型）
    - IDE 类型检查更准确

    使用示例：
        # 方式1：继承（推荐，获得完整类型提示）
        class MySpi(PyMdSpi):
            def on_front_connected(self) -> None: ...
            # 实现其他方法...

        # 方式2：不继承（鸭子类型）
        class MySpi:
            def on_front_connected(self) -> None: ...
            # 实现其他方法...

        api.register_spi(MySpi())  # 两种方式类型检查都通过
    """
    def __init__(self) -> None: ...

    def on_front_connected(self) -> None:
        pass
    def on_front_disconnected(self, reason: int) -> None:
        pass
    def on_heart_beat_warning(self, time_lapse: int) -> None:
        pass
    def on_rsp_user_login(self, rsp_user_login: RspUserLogin, rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_user_logout(self, user_logout: UserLogout, rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_error(self, rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_sub_market_data(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_un_sub_market_data(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_sub_for_quote_rsp(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_un_sub_for_quote_rsp(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rtn_depth_market_data(self, depth_market_data: DepthMarketData) -> None:
        pass
    def on_rtn_for_quote_rsp(self, for_quote_rsp: ForQuoteRsp) -> None:
        pass
    def on_rsp_qry_multicast_instrument(self, multicast_instrument: Optional[MulticastInstrument], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass


class PyTradeSpi(ABC):
    """
    CTP PC版交易回调接口协议

    使用示例：
        # 方式1：继承（推荐，获得完整类型提示）
        class MyTradeSpi(PyTradeSpi):
            def on_front_connected(self) -> None: ...
            # 实现其他方法...

        # 方式2：不继承（鸭子类型）
        class MyTradeSpi:
            def on_front_connected(self) -> None: ...
            # 实现其他方法...

        api.register_spi(MyTradeSpi())  # 两种方式类型检查都通过
    """

    # 连接相关回调

    def on_front_connected(self) -> None:
        pass
    def on_front_disconnected(self, reason: int) -> None:
        pass
    def on_heart_beat_warning(self, time_lapse: int) -> None:
        pass
    def on_rsp_authenticate(self, rsp_authenticate: Optional[RspAuthenticate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_user_login(self, rsp_user_login: Optional[RspUserLogin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_user_logout(self, user_logout: Optional[UserLogout], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_user_password_update(self, user_password_update: Optional[UserPasswordUpdate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_trading_account_password_update(self, trading_account_password_update: Optional[TradingAccountPasswordUpdate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_user_auth_method(self, rsp_user_auth_method: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_gen_user_captcha(self, rsp_gen_user_captcha: Optional[RspGenUserCaptcha], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_gen_user_text(self, rsp_gen_user_text: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_order_insert(self, input_order: Optional[InputOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_parked_order_insert(self, parked_order: Optional[ParkedOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_parked_order_action(self, parked_order_action: Optional[ParkedOrderAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_order_action(self, input_order_action: Optional[InputOrderAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_max_order_volume(self, qry_max_order_volume: Optional[QryMaxOrderVolume], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_settlement_info_confirm(self, settlement_info_confirm: Optional[SettlementInfoConfirm], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_remove_parked_order(self, remove_parked_order: Optional[RemoveParkedOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_remove_parked_order_action(self, remove_parked_order_action: Optional[RemoveParkedOrderAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_exec_order_insert(self, input_exec_order: Optional[InputExecOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_exec_order_action(self, input_exec_order_action: Optional[InputExecOrderAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_for_quote_insert(self, input_for_quote: Optional[InputForQuote], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_quote_insert(self, input_quote: Optional[InputQuote], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_quote_action(self, input_quote_action: Optional[InputQuoteAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_batch_order_action(self, input_batch_order_action: Optional[InputBatchOrderAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_option_self_close_insert(self, input_option_self_close: Optional[InputOptionSelfClose], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_option_self_close_action(self, input_option_self_close_action: Optional[InputOptionSelfCloseAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_comb_action_insert(self, input_comb_action: Optional[InputCombAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_order(self, order: Optional[Order], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_trade(self, trade: Optional[Trade], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_position(self, investor_position: Optional[InvestorPosition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_trading_account(self, trading_account: Optional[TradingAccount], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor(self, investor: Optional[Investor], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_trading_code(self, trading_code: Optional[TradingCode], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_instrument_margin_rate(self, instrument_margin_rate: Optional[InstrumentMarginRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_instrument_commission_rate(self, instrument_commission_rate: Optional[InstrumentCommissionRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_user_session(self, user_session: Optional[UserSession], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_exchange(self, exchange: Optional[Exchange], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_product(self, product: Optional[Product], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_instrument(self, instrument: Optional[Instrument], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_depth_market_data(self, depth_market_data: Optional[DepthMarketData], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_settlement_info(self, settlement_info: Optional[SettlementInfo], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_transfer_bank(self, transfer_bank: Optional[TransferBank], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_position_detail(self, investor_position_detail: Optional[InvestorPositionDetail], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_notice(self, notice: Optional[Notice], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_settlement_info_confirm(self, settlement_info_confirm: Optional[SettlementInfoConfirm], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_position_combine_detail(self, investor_position_combine_detail: Optional[InvestorPositionCombineDetail], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_cfmmc_trading_account_key(self, cfmmc_trading_account_key: Optional[CFMMCTradingAccountKey], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_ewarrant_offset(self, ewarrant_offset: Optional[EWarrantOffset], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_product_group_margin(self, investor_product_group_margin: Optional[InvestorProductGroupMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_exchange_margin_rate(self, exchange_margin_rate: Optional[ExchangeMarginRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_exchange_margin_rate_adjust(self, exchange_margin_rate_adjust: Optional[ExchangeMarginRateAdjust], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_exchange_rate(self, exchange_rate: Optional[ExchangeRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_sec_agent_acid_map(self, sec_agent_acid_map: Optional[SecAgentACIDMap], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_product_exch_rate(self, product_exch_rate: Optional[ProductExchRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_product_group(self, product_group: Optional[ProductGroup], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_mm_instrument_commission_rate(self, mm_instrument_commission_rate: Optional[MMInstrumentCommissionRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_mm_option_instr_comm_rate(self, mm_option_instr_comm_rate: Optional[MMOptionInstrCommRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_instrument_order_comm_rate(self, instrument_order_comm_rate: Optional[InstrumentOrderCommRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_sec_agent_trading_account(self, trading_account: Optional[TradingAccount], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_sec_agent_check_mode(self, sec_agent_check_mode: Optional[SecAgentCheckMode], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_sec_agent_trade_info(self, sec_agent_trade_info: Optional[SecAgentTradeInfo], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_option_instr_trade_cost(self, option_instr_trade_cost: Optional[OptionInstrTradeCost], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_option_instr_comm_rate(self, option_instr_comm_rate: Optional[OptionInstrCommRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_exec_order(self, exec_order: Optional[ExecOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_for_quote(self, for_quote: Optional[ForQuote], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_quote(self, quote: Optional[Quote], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_option_self_close(self, option_self_close: Optional[OptionSelfClose], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_invest_unit(self, invest_unit: Optional[InvestUnit], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_comb_instrument_guard(self, comb_instrument_guard: Optional[CombInstrumentGuard], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_comb_action(self, comb_action: Optional[CombAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_transfer_serial(self, transfer_serial: Optional[TransferSerial], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_account_register(self, account_register: Optional[AccountRegister], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_error(self, rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rtn_order(self, order: Order) -> None:
        pass
    def on_rtn_trade(self, trade: Trade) -> None:
        pass
    def on_err_rtn_order_insert(self, input_order: Optional[InputOrder], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_order_action(self, order_action: Optional[OrderAction], rsp_info: RspInfo) -> None:
        pass
    def on_rtn_instrument_status(self, instrument_status: InstrumentStatus) -> None:
        pass
    def on_rtn_bulletin(self, bulletin: Bulletin) -> None:
        pass
    def on_rtn_trading_notice(self, trading_notice_info: TradingNoticeInfo) -> None:
        pass
    def on_rtn_error_conditional_order(self, error_conditional_order: ErrorConditionalOrder) -> None:
        pass
    def on_rtn_exec_order(self, exec_order: ExecOrder) -> None:
        pass
    def on_err_rtn_exec_order_insert(self, input_exec_order: Optional[InputExecOrder], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_exec_order_action(self, exec_order_action: Optional[ExecOrderAction], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_for_quote_insert(self, input_for_quote: Optional[InputForQuote], rsp_info: RspInfo) -> None:
        pass
    def on_rtn_quote(self, quote: Quote) -> None:
        pass
    def on_err_rtn_quote_insert(self, input_quote: Optional[InputQuote], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_quote_action(self, quote_action: Optional[QuoteAction], rsp_info: RspInfo) -> None:
        pass
    def on_rtn_for_quote_rsp(self, for_quote_rsp: ForQuoteRsp) -> None:
        pass
    def on_rtn_cfmmc_trading_account_token(self, cfmmc_trading_account_token: CFMMCTradingAccountToken) -> None:
        pass
    def on_err_rtn_batch_order_action(self, batch_order_action: Optional[BatchOrderAction], rsp_info: RspInfo) -> None:
        pass
    def on_rtn_option_self_close(self, option_self_close: OptionSelfClose) -> None:
        pass
    def on_err_rtn_option_self_close_insert(self, input_option_self_close: Optional[InputOptionSelfClose], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_option_self_close_action(self, option_self_close_action: Optional[OptionSelfCloseAction], rsp_info: RspInfo) -> None:
        pass
    def on_rtn_comb_action(self, comb_action: CombAction) -> None:
        pass
    def on_err_rtn_comb_action_insert(self, input_comb_action: Optional[InputCombAction], rsp_info: RspInfo) -> None:
        pass
    def on_rsp_qry_contract_bank(self, contract_bank: Optional[ContractBank], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_parked_order(self, parked_order: Optional[ParkedOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_parked_order_action(self, parked_order_action: Optional[ParkedOrderAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_trading_notice(self, trading_notice: Optional[TradingNotice], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_broker_trading_params(self, broker_trading_params: Optional[BrokerTradingParams], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_broker_trading_algos(self, broker_trading_algos: Optional[BrokerTradingAlgos], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_query_cfmmc_trading_account_token(self, query_cfmmc_trading_account_token: Optional[QueryCFMMCTradingAccountToken], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rtn_from_bank_to_future_by_bank(self, rsp_transfer: RspTransfer) -> None:
        pass
    def on_rtn_from_future_to_bank_by_bank(self, rsp_transfer: RspTransfer) -> None:
        pass
    def on_rtn_repeal_from_bank_to_future_by_bank(self, rsp_repeal: RspRepeal) -> None:
        pass
    def on_rtn_repeal_from_future_to_bank_by_bank(self, rsp_repeal: RspRepeal) -> None:
        pass
    def on_rtn_from_bank_to_future_by_future(self, rsp_transfer: RspTransfer) -> None:
        pass
    def on_rtn_from_future_to_bank_by_future(self, rsp_transfer: RspTransfer) -> None:
        pass
    def on_rtn_repeal_from_bank_to_future_by_future_manual(self, rsp_repeal: RspRepeal) -> None:
        pass
    def on_rtn_repeal_from_future_to_bank_by_future_manual(self, rsp_repeal: RspRepeal) -> None:
        pass
    def on_rtn_query_bank_balance_by_future(self, notify_query_account: NotifyQueryAccount) -> None:
        pass
    def on_err_rtn_bank_to_future_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_future_to_bank_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_repeal_bank_to_future_by_future_manual(self, req_repeal: Optional[ReqRepeal], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_repeal_future_to_bank_by_future_manual(self, req_repeal: Optional[ReqRepeal], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_query_bank_balance_by_future(self, req_query_account: Optional[ReqQueryAccount], rsp_info: RspInfo) -> None:
        pass
    def on_rtn_repeal_from_bank_to_future_by_future(self, rsp_repeal: RspRepeal) -> None:
        pass
    def on_rtn_repeal_from_future_to_bank_by_future(self, rsp_repeal: RspRepeal) -> None:
        pass
    def on_rsp_from_bank_to_future_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_from_future_to_bank_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_query_bank_account_money_by_future(self, req_query_account: Optional[ReqQueryAccount], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rtn_open_account_by_bank(self, open_account: OpenAccount) -> None:
        pass
    def on_rtn_cancel_account_by_bank(self, cancel_account: CancelAccount) -> None:
        pass
    def on_rtn_change_account_by_bank(self, change_account: ChangeAccount) -> None:
        pass
    def on_rsp_qry_classified_instrument(self, instrument: Optional[Instrument], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_comb_promotion_param(self, comb_promotion_param: Optional[CombPromotionParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spbm_future_parameter(self, spbm_future_parameter: Optional[SPBMFutureParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spbm_option_parameter(self, spbm_option_parameter: Optional[SPBMOptionParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spbm_intra_parameter(self, spbm_intra_parameter: Optional[SPBMIntraParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spbm_inter_parameter(self, spbm_inter_parameter: Optional[SPBMInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spbm_portf_definition(self, spbm_portf_definition: Optional[SPBMPortfDefinition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spbm_investor_portf_def(self, spbm_investor_portf_def: Optional[SPBMInvestorPortfDef], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_portf_margin_ratio(self, investor_portf_margin_ratio: Optional[InvestorPortfMarginRatio], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_prod_spbm_detail(self, investor_prod_spbm_detail: Optional[InvestorProdSPBMDetail], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_commodity_spmm_margin(self, investor_commodity_spmm_margin: Optional[InvestorCommoditySPMMMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_commodity_group_spmm_margin(self, investor_commodity_group_spmm_margin: Optional[InvestorCommodityGroupSPMMMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spmm_inst_param(self, spmm_inst_param: Optional[SPMMInstParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spmm_product_param(self, spmm_product_param: Optional[SPMMProductParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_spbm_add_on_inter_parameter(self, spbm_add_on_inter_parameter: Optional[SPBMAddOnInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rcams_comb_product_info(self, rcams_comb_product_info: Optional[RCAMSCombProductInfo], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rcams_instr_parameter(self, rcams_instr_parameter: Optional[RCAMSInstrParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rcams_intra_parameter(self, rcams_intra_parameter: Optional[RCAMSIntraParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rcams_inter_parameter(self, rcams_inter_parameter: Optional[RCAMSInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rcams_short_opt_adjust_param(self, rcams_short_opt_adjust_param: Optional[RCAMSShortOptAdjustParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rcams_investor_comb_position(self, rcams_investor_comb_position: Optional[RCAMSInvestorCombPosition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_prod_rcams_margin(self, investor_prod_rcams_margin: Optional[InvestorProdRCAMSMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rule_instr_parameter(self, rule_instr_parameter: Optional[RULEInstrParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rule_intra_parameter(self, rule_intra_parameter: Optional[RULEIntraParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_rule_inter_parameter(self, rule_inter_parameter: Optional[RULEInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_prod_rule_margin(self, investor_prod_rule_margin: Optional[InvestorProdRULEMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_portf_setting(self, investor_portf_setting: Optional[InvestorPortfSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_investor_info_comm_rec(self, investor_info_comm_rec: Optional[InvestorInfoCommRec], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_comb_leg(self, comb_leg: Optional[CombLeg], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_offset_setting(self, input_offset_setting: Optional[InputOffsetSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_cancel_offset_setting(self, input_offset_setting: Optional[InputOffsetSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rtn_offset_setting(self, offset_setting: OffsetSetting) -> None:
        pass
    def on_err_rtn_offset_setting(self, input_offset_setting: Optional[InputOffsetSetting], rsp_info: RspInfo) -> None:
        pass
    def on_err_rtn_cancel_offset_setting(self, input_offset_setting: Optional[CancelOffsetSetting], rsp_info: RspInfo) -> None:
        pass
    def on_rsp_qry_offset_setting(self, offset_setting: Optional[OffsetSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_trader_offer(self, trader_offer: Optional[TraderOffer], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_risk_settle_invest_position(self, risk_settle_invest_position: Optional[RiskSettleInvestPosition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass
    def on_rsp_qry_risk_settle_product_status(self, risk_settle_product_status: Optional[RiskSettleProductStatus], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        pass








class MdApi:
    """
    CTP PC版行情 API 接口类

    方案3说明：
    - 此类仅用于类型提示
    - 实际的类由 C++ 模块提供
    - Python 层不需要继承任何基类
    """

    @staticmethod
    def create_ftdc_md_api(flow_path: str = "", is_using_udp: bool = False, is_multicast: bool = False) -> "MdApi": ...

    @staticmethod
    def get_api_version() -> str: ...

    def init(self) -> None: ...
    def join(self) -> int: ...
    def get_trading_day(self) -> str: ...
    def register_front(self, front_address: Union[str, list[str], np.ndarray]) -> None: ...
    def register_name_server(self, ns_address: str) -> None: ...
    def register_fens_user_info(self, fens_user_info: FensUserInfo) -> None: ...
    def register_spi(self, spi: PyMdSpi) -> None: ...
    def subscribe_market_data(self, instrument_ids: Union[List[str], np.ndarray]) -> int: ...
    def un_subscribe_market_data(self, instrument_ids: Union[List[str], np.ndarray]) -> int: ...
    def subscribe_for_quote_rsp(self, instrument_ids: Union[List[str], np.ndarray]) -> int: ...
    def un_subscribe_for_quote_rsp(self, instrument_ids: Union[List[str], np.ndarray]) -> int: ...
    def req_user_login(self, req_user_login: ReqUserLogin, request_id: int) -> int: ...
    def req_user_logout(self, user_logout: UserLogout, request_id: int) -> int: ...
    def req_qry_multicast_instrument(self, qry_multicast_instrument: QryMulticastInstrument, request_id: int) -> int: ...
    def release(self) -> None: ...


# =============================================================================
# PyTradeSpi 类定义（仅在类型检查时可用）
# 这些类由 C++ 模块提供，这里仅定义类型提示用于 IDE 和 mypy
# =============================================================================

class TradeApi:
    """CTP 交易 API 类

    对应 C++ 的 CThostFtdcTraderApi 类，提供交易相关的所有接口方法。
    """

    # -------------------------------------------------------------------------
    # 静态方法和创建相关
    # -------------------------------------------------------------------------

    @staticmethod
    def create_ftdc_trader_api(flow_path: str = "", is_using_udp: bool = False) -> "TradeApi":
        """创建 TraderApi 实例

        Args:
            flow_path: 存贮订阅信息文件的目录，默认为当前目录
            is_using_udp: 是否使用 UDP（已废弃，仅用于兼容）

        Returns:
            创建出的 TraderApi 实例
        """
        ...

    @staticmethod
    def get_api_version() -> str:
        """获取 API 的版本信息

        Returns:
            获取到的版本号
        """
        ...

    # -------------------------------------------------------------------------
    # 初始化和连接相关
    # -------------------------------------------------------------------------

    def init(self) -> None:
        """初始化

        初始化运行环境，只有调用后，接口才开始工作。
        """
        ...

    def join(self) -> int:
        """等待接口线程结束运行

        Returns:
            线程退出代码
        """
        ...

    def get_trading_day(self) -> str:
        """获取当前交易日

        Returns:
            获取到的交易日（只有登录成功后，才能得到正确的交易日）
        """
        ...

    def get_front_info(self) -> dict:
        """获取已连接的前置的信息

        Returns:
            前置信息字典，包含以下字段：
            - front_id: 前置编号
            - broker_id: 经纪公司代码
            - broker_name: 经纪公司名称
            - broker_type: 经纪公司类型
            - broker_version: 经纪公司版本
            - link_status: 连接状态
            - link_time: 连接时间
            - last_heartbeat_time: 最后心跳时间
            - user_ip_count: 用户IP数量
            - reserved: 保留字段
        """
        ...

    def register_front(self, addresses: Union[str , list[str] , np.ndarray]) -> None:
        """注册前置机网络地址

        Args:
            addresses: 前置机网络地址，支持以下格式：
                - str: 单个地址，如 "tcp://127.0.0.1:17001"
                - list[str]: 地址列表
                - numpy.ndarray: 字符串类型的 NumPy 数组

        网络地址的格式为："protocol://ipaddress:port"，如："tcp://127.0.0.1:17001"
        """
        ...

    def register_name_server(self, addresses: Union[str , list[str] , np.ndarray]) -> None:
        """注册名字服务器网络地址

        Args:
            addresses: 名字服务器网络地址，格式同 register_front

        RegisterNameServer 优先于 RegisterFront。
        """
        ...

    def register_fens_user_info(self, fens_user_info: FensUserInfo) -> None:
        """注册名字服务器用户信息

        Args:
            fens_user_info: 用户信息
        """
        ...

    def register_spi(self, spi: PyTradeSpi) -> None:
        """注册回调接口

        Args:
            spi: 派生自回调接口类的实例
        """
        ...

    def subscribe_private_topic(self, resume_type: ResumeType) -> None:
        """订阅私有流

        Args:
            resume_type: 私有流重传方式
                - ResumeType.RESTART: 从本交易日开始重传
                - ResumeType.RESUME: 从上次收到的续传
                - ResumeType.QUICK: 只传送登录后私有流的内容

        该方法要在 Init 方法前调用。若不调用则不会收到私有流的数据。
        """
        ...

    def subscribe_public_topic(self, resume_type: ResumeType) -> None:
        """订阅公共流

        Args:
            resume_type: 公共流重传方式
                - ResumeType.RESTART: 从本交易日开始重传
                - ResumeType.RESUME: 从上次收到的续传
                - ResumeType.QUICK: 只传送登录后公共流的内容
                - ResumeType.NONE: 取消订阅公共流

        该方法要在 Init 方法前调用。若不调用则不会收到公共流的数据。
        """
        ...

    def release(self) -> None:
        """删除接口对象本身

        不再使用本接口对象时，调用该函数删除接口对象。
        """
        ...

    # -------------------------------------------------------------------------
    # 认证登录相关
    # -------------------------------------------------------------------------

    def req_authenticate(self, req_authenticate: ReqAuthenticate, request_id: int) -> int:
        """客户端认证请求

        Args:
            req_authenticate: 认证请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def register_user_system_info(self, user_system_info: UserSystemInfo) -> int:
        """注册用户终端信息，用于中继服务器多连接模式

        需要在终端认证成功后，用户登录前调用该接口。

        Args:
            user_system_info: 用户系统信息

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def submit_user_system_info(self, user_system_info: UserSystemInfo) -> int:
        """上报用户终端信息，用于中继服务器操作员登录模式

        操作员登录后，可以多次调用该接口上报客户信息。

        Args:
            user_system_info: 用户系统信息

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def register_wechat_user_system_info(self, user_system_info: WechatUserSystemInfo) -> int:
        """注册用户终端信息，用于中继服务器多连接模式

        用于微信小程序等应用上报信息。

        Args:
            user_system_info: 微信用户系统信息

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def submit_wechat_user_system_info(self, user_system_info: WechatUserSystemInfo) -> int:
        """上报用户终端信息，用于中继服务器操作员登录模式

        用于微信小程序等应用上报信息。

        Args:
            user_system_info: 微信用户系统信息

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_user_login(self, req_user_login: ReqUserLogin, request_id: int) -> int:
        """用户登录请求

        Args:
            req_user_login: 登录请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_user_logout(self, user_logout: UserLogout, request_id: int) -> int:
        """登出请求

        Args:
            user_logout: 登出参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_user_password_update(self, user_password_update: UserPasswordUpdate, request_id: int) -> int:
        """用户口令更新请求

        Args:
            user_password_update: 口令更新参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_trading_account_password_update(self, trading_account_password_update: TradingAccountPasswordUpdate, request_id: int) -> int:
        """资金账户口令更新请求

        Args:
            trading_account_password_update: 资金账户口令更新参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_user_auth_method(self, req_user_auth_method: ReqUserAuthMethod, request_id: int) -> int:
        """查询用户当前支持的认证模式

        Args:
            req_user_auth_method: 认证方法查询请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_gen_user_captcha(self, req_gen_user_captcha: ReqGenUserCaptcha, request_id: int) -> int:
        """用户发出获取图形验证码请求

        Args:
            req_gen_user_captcha: 图形验证码请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_gen_user_text(self, req_gen_user_text: ReqGenUserText, request_id: int) -> int:
        """用户发出获取短信验证码请求

        Args:
            req_gen_user_text: 短信验证码请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_user_login_with_captcha(self, req_user_login_with_captcha: ReqUserLoginWithCaptcha, request_id: int) -> int:
        """用户发出带有图片验证码的登陆请求

        Args:
            req_user_login_with_captcha: 带验证码的登录请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_user_login_with_text(self, req_user_login_with_text: ReqUserLoginWithText, request_id: int) -> int:
        """用户发出带有短信验证码的登陆请求

        Args:
            req_user_login_with_text: 带短信验证码的登录请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_user_login_with_otp(self, req_user_login_with_otp: ReqUserLoginWithOTP, request_id: int) -> int:
        """用户发出带有动态口令的登陆请求

        Args:
            req_user_login_with_otp: 带动态口令的登录请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 报单相关
    # -------------------------------------------------------------------------

    def req_order_insert(self, input_order: InputOrder, request_id: int) -> int:
        """报单录入请求

        Args:
            input_order: 报单录入参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_parked_order_insert(self, parked_order: ParkedOrder, request_id: int) -> int:
        """预埋单录入请求

        Args:
            parked_order: 预埋单参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_parked_order_action(self, parked_order_action: ParkedOrderAction, request_id: int) -> int:
        """预埋撤单录入请求

        Args:
            parked_order_action: 预埋撤单参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_order_action(self, input_order_action: InputOrderAction, request_id: int) -> int:
        """报单操作请求

        Args:
            input_order_action: 报单操作参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_max_order_volume(self, qry_max_order_volume: QryMaxOrderVolume, request_id: int) -> int:
        """查询最大报单数量请求

        Args:
            qry_max_order_volume: 最大报单数量查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_settlement_info_confirm(self, settlement_info_confirm: SettlementInfoConfirm, request_id: int) -> int:
        """投资者结算结果确认

        Args:
            settlement_info_confirm: 结算信息确认参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_remove_parked_order(self, remove_parked_order: RemoveParkedOrder, request_id: int) -> int:
        """请求删除预埋单

        Args:
            remove_parked_order: 删除预埋单参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_remove_parked_order_action(self, remove_parked_order_action: RemoveParkedOrderAction, request_id: int) -> int:
        """请求删除预埋撤单

        Args:
            remove_parked_order_action: 删除预埋撤单参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_batch_order_action(self, input_batch_order_action: InputBatchOrderAction, request_id: int) -> int:
        """批量报单操作请求

        Args:
            input_batch_order_action: 批量报单操作参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 执行宣告相关
    # -------------------------------------------------------------------------

    def req_exec_order_insert(self, input_exec_order: InputExecOrder, request_id: int) -> int:
        """执行宣告录入请求

        Args:
            input_exec_order: 执行宣告录入参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_exec_order_action(self, input_exec_order_action: InputExecOrderAction, request_id: int) -> int:
        """执行宣告操作请求

        Args:
            input_exec_order_action: 执行宣告操作参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 询价报价相关
    # -------------------------------------------------------------------------

    def req_for_quote_insert(self, input_for_quote: InputForQuote, request_id: int) -> int:
        """询价录入请求

        Args:
            input_for_quote: 询价录入参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_quote_insert(self, input_quote: InputQuote, request_id: int) -> int:
        """报价录入请求

        Args:
            input_quote: 报价录入参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_quote_action(self, input_quote_action: InputQuoteAction, request_id: int) -> int:
        """报价操作请求

        Args:
            input_quote_action: 报价操作参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 期权自对冲相关
    # -------------------------------------------------------------------------

    def req_option_self_close_insert(self, input_option_self_close: InputOptionSelfClose, request_id: int) -> int:
        """期权自对冲录入请求

        Args:
            input_option_self_close: 期权自对冲录入参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_option_self_close_action(self, input_option_self_close_action: InputOptionSelfCloseAction, request_id: int) -> int:
        """期权自对冲操作请求

        Args:
            input_option_self_close_action: 期权自对冲操作参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 组合相关
    # -------------------------------------------------------------------------

    def req_comb_action_insert(self, input_comb_action: InputCombAction, request_id: int) -> int:
        """申请组合录入请求

        Args:
            input_comb_action: 组合录入参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 对冲设置相关
    # -------------------------------------------------------------------------

    def req_offset_setting(self, input_offset_setting: InputOffsetSetting, request_id: int) -> int:
        """对冲设置请求

        Args:
            input_offset_setting: 对冲设置参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_cancel_offset_setting(self, input_offset_setting: InputOffsetSetting, request_id: int) -> int:
        """对冲设置撤销请求

        Args:
            input_offset_setting: 对冲设置参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 查询相关 - 基础查询
    # -------------------------------------------------------------------------

    def req_qry_order(self, qry_order: QryOrder, request_id: int) -> int:
        """请求查询报单

        Args:
            qry_order: 报单查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_trade(self, qry_trade: QryTrade, request_id: int) -> int:
        """请求查询成交

        Args:
            qry_trade: 成交查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_position(self, qry_investor_position: QryInvestorPosition, request_id: int) -> int:
        """请求查询投资者持仓

        Args:
            qry_investor_position: 持仓查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_trading_account(self, qry_trading_account: QryTradingAccount, request_id: int) -> int:
        """请求查询资金账户

        Args:
            qry_trading_account: 资金账户查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor(self, qry_investor: QryInvestor, request_id: int) -> int:
        """请求查询投资者

        Args:
            qry_investor: 投资者查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_trading_code(self, qry_trading_code: QryTradingCode, request_id: int) -> int:
        """请求查询交易编码

        Args:
            qry_trading_code: 交易编码查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_instrument_margin_rate(self, qry_instrument_margin_rate: QryInstrumentMarginRate, request_id: int) -> int:
        """请求查询合约保证金率

        Args:
            qry_instrument_margin_rate: 合约保证金率查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_instrument_commission_rate(self, qry_instrument_commission_rate: QryInstrumentCommissionRate, request_id: int) -> int:
        """请求查询合约手续费率

        Args:
            qry_instrument_commission_rate: 合约手续费率查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_user_session(self, qry_user_session: QryUserSession, request_id: int) -> int:
        """请求查询用户会话

        Args:
            qry_user_session: 用户会话查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_exchange(self, qry_exchange: QryExchange, request_id: int) -> int:
        """请求查询交易所

        Args:
            qry_exchange: 交易所查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_product(self, qry_product: QryProduct, request_id: int) -> int:
        """请求查询产品

        Args:
            qry_product: 产品查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_instrument(self, qry_instrument: QryInstrument, request_id: int) -> int:
        """请求查询合约

        Args:
            qry_instrument: 合约查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_depth_market_data(self, qry_depth_market_data: QryDepthMarketData, request_id: int) -> int:
        """请求查询行情

        Args:
            qry_depth_market_data: 行情查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_settlement_info(self, qry_settlement_info: QrySettlementInfo, request_id: int) -> int:
        """请求查询投资者结算结果

        Args:
            qry_settlement_info: 结算信息查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_transfer_bank(self, qry_transfer_bank: QryTransferBank, request_id: int) -> int:
        """请求查询转帐银行

        Args:
            qry_transfer_bank: 转帐银行查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_position_detail(self, qry_investor_position_detail: QryInvestorPositionDetail, request_id: int) -> int:
        """请求查询投资者持仓明细

        Args:
            qry_investor_position_detail: 持仓明细查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_notice(self, qry_notice: QryNotice, request_id: int) -> int:
        """请求查询客户通知

        Args:
            qry_notice: 通知查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_settlement_info_confirm(self, qry_settlement_info_confirm: QrySettlementInfoConfirm, request_id: int) -> int:
        """请求查询结算信息确认

        Args:
            qry_settlement_info_confirm: 结算信息确认查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_position_combine_detail(self, qry_investor_position_combine_detail: QryInvestorPositionCombineDetail, request_id: int) -> int:
        """请求查询投资者持仓明细

        Args:
            qry_investor_position_combine_detail: 组合持仓明细查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_cfmmc_trading_account_key(self, qry_cfmmc_trading_account_key: QryCFMMCTradingAccountKey, request_id: int) -> int:
        """请求查询保证金监管系统经纪公司资金账户密钥

        Args:
            qry_cfmmc_trading_account_key: 经纪公司资金账户密钥查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_ewarrant_offset(self, qry_ewarrant_offset: QryEWarrantOffset, request_id: int) -> int:
        """请求查询仓单折抵信息

        Args:
            qry_ewarrant_offset: 仓单折抵信息查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_product_group_margin(self, qry_investor_product_group_margin: QryInvestorProductGroupMargin, request_id: int) -> int:
        """请求查询投资者品种/跨品种保证金

        Args:
            qry_investor_product_group_margin: 投资者品种/跨品种保证金查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_exchange_margin_rate(self, qry_exchange_margin_rate: QryExchangeMarginRate, request_id: int) -> int:
        """请求查询交易所保证金率

        Args:
            qry_exchange_margin_rate: 交易所保证金率查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_exchange_margin_rate_adjust(self, qry_exchange_margin_rate_adjust: QryExchangeMarginRateAdjust, request_id: int) -> int:
        """请求查询交易所调整保证金率

        Args:
            qry_exchange_margin_rate_adjust: 交易所调整保证金率查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_exchange_rate(self, qry_exchange_rate: QryExchangeRate, request_id: int) -> int:
        """请求查询汇率

        Args:
            qry_exchange_rate: 汇率查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_sec_agent_acid_map(self, qry_sec_agent_acid_map: QrySecAgentACIDMap, request_id: int) -> int:
        """请求查询二级代理操作员银期权限

        Args:
            qry_sec_agent_acid_map: 二级代理操作员银期权限查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_product_exch_rate(self, qry_product_exch_rate: QryProductExchRate, request_id: int) -> int:
        """请求查询产品报价汇率

        Args:
            qry_product_exch_rate: 产品报价汇率查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_product_group(self, qry_product_group: QryProductGroup, request_id: int) -> int:
        """请求查询产品组

        Args:
            qry_product_group: 产品组查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_mm_instrument_commission_rate(self, qry_mm_instrument_commission_rate: QryMMInstrumentCommissionRate, request_id: int) -> int:
        """请求查询做市商合约手续费率

        Args:
            qry_mm_instrument_commission_rate: 做市商合约手续费率查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_mm_option_instr_comm_rate(self, qry_mm_option_instr_comm_rate: QryMMOptionInstrCommRate, request_id: int) -> int:
        """请求查询做市商期权合约手续费

        Args:
            qry_mm_option_instr_comm_rate: 做市商期权合约手续费查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_instrument_order_comm_rate(self, qry_instrument_order_comm_rate: QryInstrumentOrderCommRate, request_id: int) -> int:
        """请求查询报单手续费

        Args:
            qry_instrument_order_comm_rate: 报单手续费查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_sec_agent_trading_account(self, qry_trading_account: QryTradingAccount, request_id: int) -> int:
        """请求查询资金账户

        Args:
            qry_trading_account: 资金账户查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_sec_agent_check_mode(self, qry_sec_agent_check_mode: QrySecAgentCheckMode, request_id: int) -> int:
        """请求查询二级代理商资金校验模式

        Args:
            qry_sec_agent_check_mode: 二级代理商资金校验模式查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_sec_agent_trade_info(self, qry_sec_agent_trade_info: QrySecAgentTradeInfo, request_id: int) -> int:
        """请求查询二级代理商信息

        Args:
            qry_sec_agent_trade_info: 二级代理商信息查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_option_instr_trade_cost(self, qry_option_instr_trade_cost: QryOptionInstrTradeCost, request_id: int) -> int:
        """请求查询期权交易成本

        Args:
            qry_option_instr_trade_cost: 期权交易成本查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_option_instr_comm_rate(self, qry_option_instr_comm_rate: QryOptionInstrCommRate, request_id: int) -> int:
        """请求查询期权合约手续费

        Args:
            qry_option_instr_comm_rate: 期权合约手续费查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_exec_order(self, qry_exec_order: QryExecOrder, request_id: int) -> int:
        """请求查询执行宣告

        Args:
            qry_exec_order: 执行宣告查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_for_quote(self, qry_for_quote: QryForQuote, request_id: int) -> int:
        """请求查询询价

        Args:
            qry_for_quote: 询价查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_quote(self, qry_quote: QryQuote, request_id: int) -> int:
        """请求查询报价

        Args:
            qry_quote: 报价查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_option_self_close(self, qry_option_self_close: QryOptionSelfClose, request_id: int) -> int:
        """请求查询期权自对冲

        Args:
            qry_option_self_close: 期权自对冲查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_invest_unit(self, qry_invest_unit: QryInvestUnit, request_id: int) -> int:
        """请求查询投资单元

        Args:
            qry_invest_unit: 投资单元查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_comb_instrument_guard(self, qry_comb_instrument_guard: QryCombInstrumentGuard, request_id: int) -> int:
        """请求查询组合合约安全系数

        Args:
            qry_comb_instrument_guard: 组合合约安全系数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_comb_action(self, qry_comb_action: QryCombAction, request_id: int) -> int:
        """请求查询申请组合

        Args:
            qry_comb_action: 申请组合查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 查询相关 - 银期转账
    # -------------------------------------------------------------------------

    def req_qry_transfer_serial(self, qry_transfer_serial: QryTransferSerial, request_id: int) -> int:
        """请求查询转帐流水

        Args:
            qry_transfer_serial: 转帐流水查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_account_register(self, qry_account_register: QryAccountRegister, request_id: int) -> int:
        """请求查询银期签约关系

        Args:
            qry_account_register: 银期签约关系查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_contract_bank(self, qry_contract_bank: QryContractBank, request_id: int) -> int:
        """请求查询签约银行

        Args:
            qry_contract_bank: 签约银行查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_from_bank_to_future_by_future(self, req_transfer: ReqTransfer, request_id: int) -> int:
        """期货发起银行资金转期货请求

        Args:
            req_transfer: 转账请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_from_future_to_bank_by_future(self, req_transfer: ReqTransfer, request_id: int) -> int:
        """期货发起期货资金转银行请求

        Args:
            req_transfer: 转账请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_query_bank_account_money_by_future(self, req_query_account: ReqQueryAccount, request_id: int) -> int:
        """期货发起查询银行余额请求

        Args:
            req_query_account: 查询银行余额请求参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 查询相关 - 预埋单
    # -------------------------------------------------------------------------

    def req_qry_parked_order(self, qry_parked_order: QryParkedOrder, request_id: int) -> int:
        """请求查询预埋单

        Args:
            qry_parked_order: 预埋单查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_parked_order_action(self, qry_parked_order_action: QryParkedOrderAction, request_id: int) -> int:
        """请求查询预埋撤单

        Args:
            qry_parked_order_action: 预埋撤单查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # 查询相关 - 其他
    # -------------------------------------------------------------------------

    def req_qry_trading_notice(self, qry_trading_notice: QryTradingNotice, request_id: int) -> int:
        """请求查询交易通知

        Args:
            qry_trading_notice: 交易通知查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_broker_trading_params(self, qry_broker_trading_params: QryBrokerTradingParams, request_id: int) -> int:
        """请求查询经纪公司交易参数

        Args:
            qry_broker_trading_params: 经纪公司交易参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_broker_trading_algos(self, qry_broker_trading_algos: QryBrokerTradingAlgos, request_id: int) -> int:
        """请求查询经纪公司交易算法

        Args:
            qry_broker_trading_algos: 经纪公司交易算法查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_query_cfmmc_trading_account_token(self, query_cfmmc_trading_account_token: QueryCFMMCTradingAccountToken, request_id: int) -> int:
        """请求查询监控中心用户令牌

        Args:
            query_cfmmc_trading_account_token: 监控中心用户令牌查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_classified_instrument(self, qry_classified_instrument: QryClassifiedInstrument, request_id: int) -> int:
        """请求查询分类合约

        Args:
            qry_classified_instrument: 分类合约查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_comb_promotion_param(self, qry_comb_promotion_param: QryCombPromotionParam, request_id: int) -> int:
        """请求组合优惠比例

        Args:
            qry_comb_promotion_param: 组合优惠比例查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_offset_setting(self, qry_offset_setting: QryOffsetSetting, request_id: int) -> int:
        """投资者对冲设置查询

        Args:
            qry_offset_setting: 对冲设置查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # SPBM 组合保证金查询
    # -------------------------------------------------------------------------

    def req_qry_spbm_future_parameter(self, qry_spbm_future_parameter: QrySPBMFutureParameter, request_id: int) -> int:
        """SPBM 期货合约参数查询

        Args:
            qry_spbm_future_parameter: SPBM 期货合约参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spbm_option_parameter(self, qry_spbm_option_parameter: QrySPBMOptionParameter, request_id: int) -> int:
        """SPBM 期权合约参数查询

        Args:
            qry_spbm_option_parameter: SPBM 期权合约参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spbm_intra_parameter(self, qry_spbm_intra_parameter: QrySPBMIntraParameter, request_id: int) -> int:
        """SPBM 品种内对锁仓折扣参数查询

        Args:
            qry_spbm_intra_parameter: SPBM 品种内对锁仓折扣参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spbm_inter_parameter(self, qry_spbm_inter_parameter: QrySPBMInterParameter, request_id: int) -> int:
        """SPBM 跨品种抵扣参数查询

        Args:
            qry_spbm_inter_parameter: SPBM 跨品种抵扣参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spbm_portf_definition(self, qry_spbm_portf_definition: QrySPBMPortfDefinition, request_id: int) -> int:
        """SPBM 组合保证金套餐查询

        Args:
            qry_spbm_portf_definition: SPBM 组合保证金套餐查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spbm_investor_portf_def(self, qry_spbm_investor_portf_def: QrySPBMInvestorPortfDef, request_id: int) -> int:
        """投资者 SPBM 套餐选择查询

        Args:
            qry_spbm_investor_portf_def: 投资者 SPBM 套餐选择查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_portf_margin_ratio(self, qry_investor_portf_margin_ratio: QryInvestorPortfMarginRatio, request_id: int) -> int:
        """投资者新型组合保证金系数查询

        Args:
            qry_investor_portf_margin_ratio: 投资者新型组合保证金系数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_prod_spbm_detail(self, qry_investor_prod_spbm_detail: QryInvestorProdSPBMDetail, request_id: int) -> int:
        """投资者产品 SPBM 明细查询

        Args:
            qry_investor_prod_spbm_detail: 投资者产品 SPBM 明细查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_commodity_spmm_margin(self, qry_investor_commodity_spmm_margin: QryInvestorCommoditySPMMMargin, request_id: int) -> int:
        """投资者商品 SPMM 记录查询

        Args:
            qry_investor_commodity_spmm_margin: 投资者商品 SPMM 记录查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_commodity_group_spmm_margin(self, qry_investor_commodity_group_spmm_margin: QryInvestorCommodityGroupSPMMMargin, request_id: int) -> int:
        """投资者商品群 SPMM 记录查询

        Args:
            qry_investor_commodity_group_spmm_margin: 投资者商品群 SPMM 记录查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spmm_inst_param(self, qry_spmm_inst_param: QrySPMMInstParam, request_id: int) -> int:
        """SPMM 合约参数查询

        Args:
            qry_spmm_inst_param: SPMM 合约参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spmm_product_param(self, qry_spmm_product_param: QrySPMMProductParam, request_id: int) -> int:
        """SPMM 产品参数查询

        Args:
            qry_spmm_product_param: SPMM 产品参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_spbm_add_on_inter_parameter(self, qry_spbm_add_on_inter_parameter: QrySPBMAddOnInterParameter, request_id: int) -> int:
        """SPBM 附加跨品种抵扣参数查询

        Args:
            qry_spbm_add_on_inter_parameter: SPBM 附加跨品种抵扣参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # RCAMS 组合保证金查询
    # -------------------------------------------------------------------------

    def req_qry_rcams_comb_product_info(self, qry_rcams_comb_product_info: QryRCAMSCombProductInfo, request_id: int) -> int:
        """RCAMS 产品组合信息查询

        Args:
            qry_rcams_comb_product_info: RCAMS 产品组合信息查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_rcams_instr_parameter(self, qry_rcams_instr_parameter: QryRCAMSInstrParameter, request_id: int) -> int:
        """RCAMS 同合约风险对冲参数查询

        Args:
            qry_rcams_instr_parameter: RCAMS 同合约风险对冲参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_rcams_intra_parameter(self, qry_rcams_intra_parameter: QryRCAMSIntraParameter, request_id: int) -> int:
        """RCAMS 品种内风险对冲参数查询

        Args:
            qry_rcams_intra_parameter: RCAMS 品种内风险对冲参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_rcams_inter_parameter(self, qry_rcams_inter_parameter: QryRCAMSInterParameter, request_id: int) -> int:
        """RCAMS 跨品种风险折抵参数查询

        Args:
            qry_rcams_inter_parameter: RCAMS 跨品种风险折抵参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_rcams_short_opt_adjust_param(self, qry_rcams_short_opt_adjust_param: QryRCAMSShortOptAdjustParam, request_id: int) -> int:
        """RCAMS 空头期权风险调整参数查询

        Args:
            qry_rcams_short_opt_adjust_param: RCAMS 空头期权风险调整参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_rcams_investor_comb_position(self, qry_rcams_investor_comb_position: QryRCAMSInvestorCombPosition, request_id: int) -> int:
        """RCAMS 策略组合持仓查询

        Args:
            qry_rcams_investor_comb_position: RCAMS 策略组合持仓查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_prod_rcams_margin(self, qry_investor_prod_rcams_margin: QryInvestorProdRCAMSMargin, request_id: int) -> int:
        """投资者品种 RCAMS 保证金查询

        Args:
            qry_investor_prod_rcams_margin: 投资者品种 RCAMS 保证金查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    # -------------------------------------------------------------------------
    # RULE 组合保证金查询
    # -------------------------------------------------------------------------

    def req_qry_rule_instr_parameter(self, qry_rule_instr_parameter: QryRULEInstrParameter, request_id: int) -> int:
        """RULE 合约保证金参数查询

        Args:
            qry_rule_instr_parameter: RULE 合约保证金参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_rule_intra_parameter(self, qry_rule_intra_parameter: QryRULEIntraParameter, request_id: int) -> int:
        """RULE 品种内对锁仓折扣参数查询

        Args:
            qry_rule_intra_parameter: RULE 品种内对锁仓折扣参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_rule_inter_parameter(self, qry_rule_inter_parameter: QryRULEInterParameter, request_id: int) -> int:
        """RULE 跨品种抵扣参数查询

        Args:
            qry_rule_inter_parameter: RULE 跨品种抵扣参数查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_prod_rule_margin(self, qry_investor_prod_rule_margin: QryInvestorProdRULEMargin, request_id: int) -> int:
        """投资者产品 RULE 保证金查询

        Args:
            qry_investor_prod_rule_margin: 投资者产品 RULE 保证金查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_portf_setting(self, qry_investor_portf_setting: QryInvestorPortfSetting, request_id: int) -> int:
        """投资者投资者新组保设置查询

        Args:
            qry_investor_portf_setting: 投资者新组保设置查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_investor_info_comm_rec(self, qry_investor_info_comm_rec: QryInvestorInfoCommRec, request_id: int) -> int:
        """投资者申报费阶梯收取记录查询

        Args:
            qry_investor_info_comm_rec: 投资者申报费阶梯收取记录查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_comb_leg(self, qry_comb_leg: QryCombLeg, request_id: int) -> int:
        """组合腿信息查询

        Args:
            qry_comb_leg: 组合腿信息查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...

    def req_qry_trader_offer(self, qry_trader_offer: QryTraderOffer, request_id: int) -> int:
        """报单查询

        Args:
            qry_trader_offer: 报单查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...
    def req_qry_risk_settle_invest_position(self, qry_risk_settle_invest_position: QryRiskSettleInvestPosition, request_id: int) -> int:
        """投资者风险对冲持仓查询

        Args:
            qry_risk_settle_invest_position: 投资者风险对冲持仓查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...
    def req_qry_risk_settle_product_status(self, qry_risk_settle_product_status: QryRiskSettleProductStatus, request_id: int) -> int:
        """风险对冲产品状态查询

        Args:
            qry_risk_settle_product_status: 风险对冲产品状态查询参数
            request_id: 请求ID

        Returns:
            请求结果，0表示成功，非0表示失败
        """
        ...



# =============================================================================
# 字符串池监控和清理工具函数
# =============================================================================
def cleanup_temporal_pools() -> None:
    """
    清理日期和时间字符串池

    建议在每个交易日收盘后调用，释放累积的日期和时间字符串。
    """
    ...

def cleanup_instruments() -> None:
    """
    清理合约代码字符串池

    建议在切换交易日或重新订阅合约前调用。
    """
    ...

def check_instrument_pool_size() -> int:
    """
    检查合约池大小并返回当前值

    当合约池大小 >= 950 时会自动发出 RuntimeWarning。
    :return: 当前合约池中的合约数量
    """
    ...

def get_pool_sizes() -> Dict[str, int]:
    """
    获取所有字符串池的大小统计

    :return: 包含各池大小的字典，键为：
        - exchanges: 交易所代码池大小
        - dates: 日期字符串池大小
        - times: 时间字符串池大小
        - instruments: 合约代码池大小
        - users: 用户代码池大小
        - brokers: 经纪公司代码池大小
    """
    ...

# =============================================================================
# PyCapsule 辅助函数（用于零拷贝数据传递）
# =============================================================================

def pycapsule_check_exact(obj: object) -> int:
    """
    检查对象是否为 Capsule 类型

    Args:
        obj: 待检查的对象

    Returns:
        1 如果是 Capsule，0 否则
    """
    ...

def pycapsule_get_pointer(capsule: object, name: str) -> int:
    """
    从 Capsule 获取 C 指针

    Args:
        capsule: Capsule 对象
        name: Capsule 名称（如 "DepthMarketData"）

    Returns:
        C 结构体指针地址（整数）
    """
    ...

def pycapsule_new(ptr: int, name: str, destructor: object = None) -> object:
    """
    创建新的 Capsule 对象

    Args:
        ptr: C 指针地址（整数）
        name: Capsule 名称
        destructor: 析构函数（可选）

    Returns:
        新创建的 Capsule 对象
    """
    ...

# =============================================================================
# FIX 穿透式监管数据采集模块
# =============================================================================

class Fix:
    """
    FIX 数据采集模块

    用于中继模式下采集终端系统信息。

    注意：
    - 仅中继模式需要使用此模块
    - 直连模式不需要调用 collect_system_info()，CTP 会自动采集

    使用示例：
        >>> from PcCTP import Fix
        >>> system_info = Fix.collect_system_info()
        >>> version = Fix.get_fix_api_version()
    """

    @staticmethod
    def collect_system_info() -> bytes:
        """
        采集系统信息

        返回：
            bytes 对象，包含采集的系统信息（至少 270 字节）
            用于中继模式下传递给 RegisterUserSystemInfo 或 SubmitUserSystemInfo

        注意：
            1. 采集库不是线程安全的，多线程调用时需要加锁
            2. 采集的信息是二进制数据，不是字符串
            3. 直连模式不需要调用此函数，CTP 会自动采集
        """
        ...

    @staticmethod
    def get_fix_api_version() -> str:
        """
        获取 FIX 采集库版本

        返回：
            版本字符串（如 "sfit_pro_1.0_20220124_1468"）
        """
        ...
