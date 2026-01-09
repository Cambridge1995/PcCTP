from abc import ABC, ABCMeta
from typing import Optional, Any, Type, get_origin, get_args, Union
import functools
import inspect
from PcCTP.types import *

# 导出
__all__ = ['PyMdSpi', 'PyTradeSpi']


# =============================================================================
# 类型工具函数
# =============================================================================

def _unwrap_optional_type(annotation: Any) -> Optional[Type]:
    """
    从类型注解中提取实际类型，处理 Optional[T] 和 T | None

    Args:
        annotation: 类型注解

    Returns:
        实际的类型，如果无法提取则返回 None
    """
    if annotation is inspect.Parameter.empty:
        return None

    # 处理 Optional[T] (Python 3.10+ 也支持 T | None)
    origin = get_origin(annotation)
    if origin is Union:
        # Optional[T] 或 Union[T, None] 或 T | None
        args = get_args(annotation)
        for arg in args:
            if arg is not type(None):
                return arg
        return None

    # 直接类型
    if isinstance(annotation, type):
        return annotation

    return None


# =============================================================================
# 自动应用装饰器的元类
# =============================================================================

class _AutoCapsuleMeta(ABCMeta):
    """
    元类：自动为所有回调方法应用 Capsule 转换

    工作原理：
    1. 在类创建时扫描所有方法
    2. 分析参数类型注解
    3. 为 CapsuleStruct 类型参数自动创建转换包装器
    4. 即使用户重写方法，转换仍然自动生效
    """

    def __new__(mcs, name, bases, namespace):
        # 先收集需要包装的方法（在类创建前）
        methods_to_wrap = {}

        # 遍历类的所有方法
        for attr_name, attr_value in namespace.items():
            # 只处理函数（方法）
            if not callable(attr_value):
                continue

            # 跳过特殊方法
            if attr_name.startswith('__'):
                continue

            # 只处理以 on_ 开头的回调方法
            if not attr_name.startswith('on_'):
                continue

            # 获取方法的签名，分析哪些参数需要转换
            sig = inspect.signature(attr_value)
            param_names_to_convert = []

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                # 检查参数的类型注解
                expected_type = _unwrap_optional_type(param.annotation)
                if expected_type is not None:
                    # 检查是否是 CapsuleStruct 子类
                    try:
                        from PcCTP.types.base import CapsuleStruct
                        if inspect.isclass(expected_type) and issubclass(expected_type, CapsuleStruct):
                            param_names_to_convert.append((param_name, expected_type))
                    except (ImportError, TypeError):
                        pass

            # 如果有需要转换的参数，记录下来
            if param_names_to_convert:
                # 直接使用 namespace 中的原始函数，不要用 getattr
                methods_to_wrap[attr_name] = (attr_value, param_names_to_convert)

        # 创建类
        cls = super().__new__(mcs, name, bases, namespace)

        # 为需要包装的方法创建包装器并设置到类上
        for attr_name, (original_method, param_names_with_types) in methods_to_wrap.items():
            decorated_method = mcs._create_wrapper(original_method, param_names_with_types)
            setattr(cls, attr_name, decorated_method)

        return cls

    @staticmethod
    def _create_wrapper(method: callable, param_names_with_types: list) -> callable:
        """
        创建参数转换包装器

        Args:
            method: 原始方法
            param_names_with_types: 需要转换的参数列表，每个元素是 (param_name, expected_type) 元组
        """
        # 创建参数名到类型的映射，便于快速查找
        param_type_map = {name: typ for name, typ in param_names_with_types}

        # 在函数定义时就获取签名
        sig = inspect.signature(method)
        # 跳过 self 参数，因为 args 中不包含它
        parameters = [p for p in sig.parameters.values() if p.name != 'self']

        @functools.wraps(method)
        def wrapper(self, *args):
            # 转换参数
            converted_args = []

            # 遍历每个参数
            for i, arg_value in enumerate(args):
                if i < len(parameters):
                    param = parameters[i]
                    param_name = param.name

                    # 检查是否需要转换此参数
                    if param_name in param_type_map:
                        expected_type = param_type_map[param_name]

                        try:
                            # 调用 from_capsule() 方法进行转换
                            result = expected_type.from_capsule(arg_value)
                            converted_args.append(result)
                        except (TypeError, ValueError, AttributeError, OSError, KeyError):
                            # 转换失败，返回原 capsule
                            converted_args.append(arg_value)
                    else:
                        # 不需要转换，直接使用原参数
                        converted_args.append(arg_value)
                else:
                    converted_args.append(arg_value)

            # 调用原始方法
            return method(self, *converted_args)

        return wrapper


# =============================================================================
# 行情回调基类
# =============================================================================

class PyMdSpi(ABC, metaclass=_AutoCapsuleMeta):
    """
    CTP PC版行情回调接口基类

    使用元类自动为所有回调方法应用 @auto_capsule 装饰器。
    子类继承后重写方法，装饰器仍然会自动生效。

    使用示例：
        class MySpi(PyMdSpi):
            def on_rsp_user_login(self, rsp_user_login, rsp_info, request_id, is_last):
                # rsp_user_login 和 rsp_info 已自动转换为对象
                print(rsp_user_login.trading_day)
    """

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



class PyTradeSpi(ABC, metaclass=_AutoCapsuleMeta):
    """
    CTP PC版交易回调接口基类

    使用元类自动为所有回调方法应用 @auto_capsule 装饰器。
    子类继承后重写方法，装饰器仍然会自动生效。

    使用示例：
        class MySpi(PyTradeSpi):
            def on_rsp_user_login(self, rsp_user_login, rsp_info, request_id, is_last):
                # rsp_user_login 和 rsp_info 已自动转换为对象
                print(rsp_user_login.trading_day)

    注意：由于回调方法数量众多（100+），以下方法已添加装饰器。
    如果需要实现其他回调方法，请参考下方示例添加装饰器。
    """

    # 连接相关回调

    def on_front_connected(self) -> None:
        """当客户端与交易后台建立起通信连接时（还未登录前），该方法被调用"""
        pass

    def on_front_disconnected(self, reason: int) -> None:
        """当客户端与交易后台通信连接断开时，该方法被调用"""
        pass

    def on_heart_beat_warning(self, time_lapse: int) -> None:
        """心跳超时警告"""
        pass

    # 认证和登录相关回调
    def on_rsp_authenticate(self, rsp_authenticate: Optional[RspAuthenticate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """客户端认证响应"""
        pass
    def on_rsp_user_login(self, rsp_user_login: Optional[RspUserLogin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """登录请求响应"""
        pass
    def on_rsp_user_logout(self, user_logout: Optional[UserLogout], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """登出请求响应"""
        pass
    def on_rsp_user_password_update(self, user_password_update: Optional[UserPasswordUpdate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """用户口令更新请求响应"""
        pass
    def on_rsp_trading_account_password_update(self, trading_account_password_update: Optional[TradingAccountPasswordUpdate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """资金账户口令更新请求响应"""
        pass
    def on_rsp_user_auth_method(self, rsp_user_auth_method: Optional[RspUserAuthMethod], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """查询用户当前支持的认证模式的回复"""
        pass
    def on_rsp_gen_user_captcha(self, rsp_gen_user_captcha: Optional[RspGenUserCaptcha], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """获取图形验证码请求的回复"""
        pass
    def on_rsp_gen_user_text(self, rsp_gen_user_text: Optional[RspGenUserText], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """获取短信验证码请求的回复"""
        pass

    # 报单相关回调（示例）
    def on_rsp_order_insert(self, input_order: Optional[InputOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """报单录入请求响应"""
        pass
    def on_rtn_order(self, order: Order) -> None:
        """报单通知"""
        pass
    def on_rtn_trade(self, trade: Trade) -> None:
        """成交通知"""
        pass
    def on_err_rtn_order_insert(self, input_order: Optional[InputOrder], rsp_info: RspInfo) -> None:
        """报单录入错误回报"""
        pass
    def on_err_rtn_order_action(self, order_action: Optional[OrderAction], rsp_info: RspInfo) -> None:
        """报单操作错误回报"""
        pass

    # 查询相关回调（示例）
    def on_rsp_qry_order(self, order: Optional[Order], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询报单响应"""
        pass
    def on_rsp_qry_trade(self, trade: Optional[Trade], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询成交响应"""
        pass
    def on_rsp_qry_investor_position(self, investor_position: Optional[InvestorPosition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询投资者持仓响应"""
        pass
    def on_rsp_qry_trading_account(self, trading_account: Optional[TradingAccount], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询资金账户响应"""
        pass
    def on_rsp_qry_instrument(self, instrument: Optional[Instrument], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询合约响应"""
        pass

    # 以下方法省略装饰器，用户如需实现可参考上方示例添加 @auto_capsule 装饰器
    # 由于 PyTradeSpi 有 100+ 个回调方法，完整实现请参考脚本自动生成
    def on_rsp_qry_depth_market_data(self, depth_market_data: Optional[DepthMarketData], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询行情响应"""
        pass
    def on_rsp_qry_settlement_info(self, settlement_info: Optional[SettlementInfo], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询投资者结算结果响应"""
        pass
    def on_rsp_qry_transfer_bank(self, transfer_bank: Optional[TransferBank], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询转帐银行响应"""
        pass
    def on_rsp_qry_investor_position_detail(self, investor_position_detail: Optional[InvestorPositionDetail], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询投资者持仓明细响应"""
        pass
    def on_rsp_qry_notice(self, notice: Optional[Notice], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询客户通知响应"""
        pass
    def on_rsp_qry_settlement_info_confirm(self, settlement_info_confirm: Optional[SettlementInfoConfirm], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询结算信息确认响应"""
        pass
    def on_rsp_qry_investor_position_combine_detail(self, investor_position_combine_detail: Optional[InvestorPositionCombineDetail], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询投资者持仓明细响应"""
        pass
    def on_rsp_qry_cfmmc_trading_account_key(self, cfmmc_trading_account_key: Optional[CFMMCTradingAccountKey], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """查询保证金监管系统经纪公司资金账户密钥响应"""
        pass
    def on_rsp_qry_ewarrant_offset(self, ewarrant_offset: Optional[EWarrantOffset], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询仓单折抵信息响应"""
        pass
    def on_rsp_qry_investor_product_group_margin(self, investor_product_group_margin: Optional[InvestorProductGroupMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询投资者品种/跨品种保证金响应"""
        pass
    def on_rsp_qry_exchange_margin_rate(self, exchange_margin_rate: Optional[ExchangeMarginRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询交易所保证金率响应"""
        pass
    def on_rsp_qry_exchange_margin_rate_adjust(self, exchange_margin_rate_adjust: Optional[ExchangeMarginRateAdjust], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询交易所调整保证金率响应"""
        pass
    def on_rsp_qry_exchange_rate(self, exchange_rate: Optional[ExchangeRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询汇率响应"""
        pass
    def on_rsp_qry_sec_agent_acid_map(self, sec_agent_acid_map: Optional[SecAgentACIDMap], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询二级代理操作员银期权限响应"""
        pass
    def on_rsp_qry_product_exch_rate(self, product_exch_rate: Optional[ProductExchRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询产品报价汇率响应"""
        pass
    def on_rsp_qry_product_group(self, product_group: Optional[ProductGroup], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询产品组响应"""
        pass
    def on_rsp_qry_mm_instrument_commission_rate(self, mm_instrument_commission_rate: Optional[MMInstrumentCommissionRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询做市商合约手续费率响应"""
        pass
    def on_rsp_qry_mm_option_instr_comm_rate(self, mm_option_instr_comm_rate: Optional[MMOptionInstrCommRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询做市商期权合约手续费响应"""
        pass
    def on_rsp_qry_instrument_order_comm_rate(self, instrument_order_comm_rate: Optional[InstrumentOrderCommRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询报单手续费响应"""
        pass
    def on_rsp_qry_sec_agent_trading_account(self, trading_account: Optional[TradingAccount], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询资金账户响应"""
        pass
    def on_rsp_qry_sec_agent_check_mode(self, sec_agent_check_mode: Optional[SecAgentCheckMode], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询二级代理商资金校验模式响应"""
        pass
    def on_rsp_qry_sec_agent_trade_info(self, sec_agent_trade_info: Optional[SecAgentTradeInfo], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询二级代理商信息响应"""
        pass
    def on_rsp_qry_option_instr_trade_cost(self, option_instr_trade_cost: Optional[OptionInstrTradeCost], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询期权交易成本响应"""
        pass
    def on_rsp_qry_option_instr_comm_rate(self, option_instr_comm_rate: Optional[OptionInstrCommRate], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询期权合约手续费响应"""
        pass
    def on_rsp_qry_exec_order(self, exec_order: Optional[ExecOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询执行宣告响应"""
        pass
    def on_rsp_qry_for_quote(self, for_quote: Optional[ForQuote], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询询价响应"""
        pass
    def on_rsp_qry_quote(self, quote: Optional[Quote], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询报价响应"""
        pass
    def on_rsp_qry_option_self_close(self, option_self_close: Optional[OptionSelfClose], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询期权自对冲响应"""
        pass
    def on_rsp_qry_invest_unit(self, invest_unit: Optional[InvestUnit], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询投资单元响应"""
        pass
    def on_rsp_qry_comb_instrument_guard(self, comb_instrument_guard: Optional[CombInstrumentGuard], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询组合合约安全系数响应"""
        pass
    def on_rsp_qry_comb_action(self, comb_action: Optional[CombAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询申请组合响应"""
        pass
    def on_rsp_qry_transfer_serial(self, transfer_serial: Optional[TransferSerial], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询转帐流水响应"""
        pass
    def on_rsp_qry_account_register(self, account_register: Optional[AccountRegister], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询银期签约关系响应"""
        pass

    # 错误应答回调
    def on_rsp_error(self, rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """错误应答"""
        pass


    # 合约交易状态通知
    def on_rtn_instrument_status(self, instrument_status: InstrumentStatus) -> None:
        """合约交易状态通知"""
        pass

    # 交易所公告通知
    def on_rtn_bulletin(self, bulletin: Bulletin) -> None:
        """交易所公告通知"""
        pass

    # 交易通知
    def on_rtn_trading_notice(self, trading_notice_info: TradingNoticeInfo) -> None:
        """交易通知"""
        pass

    # 提示条件单校验错误
    def on_rtn_error_conditional_order(self, error_conditional_order: ErrorConditionalOrder) -> None:
        """提示条件单校验错误"""
        pass

    # 执行宣告通知
    def on_rtn_exec_order(self, exec_order: ExecOrder) -> None:
        """执行宣告通知"""
        pass
    def on_err_rtn_exec_order_insert(self, input_exec_order: Optional[InputExecOrder], rsp_info: RspInfo) -> None:
        """执行宣告录入错误回报"""
        pass
    def on_err_rtn_exec_order_action(self, exec_order_action: Optional[ExecOrderAction], rsp_info: RspInfo) -> None:
        """执行宣告操作错误回报"""
        pass

    # 询价相关回报
    def on_err_rtn_for_quote_insert(self, input_for_quote: Optional[InputForQuote], rsp_info: RspInfo) -> None:
        """询价录入错误回报"""
        pass

    # 报价通知
    def on_rtn_quote(self, quote: Quote) -> None:
        """报价通知"""
        pass
    def on_err_rtn_quote_insert(self, input_quote: Optional[InputQuote], rsp_info: RspInfo) -> None:
        """报价录入错误回报"""
        pass
    def on_err_rtn_quote_action(self, quote_action: Optional[QuoteAction], rsp_info: RspInfo) -> None:
        """报价操作错误回报"""
        pass

    # 询价通知
    def on_rtn_for_quote_rsp(self, for_quote_rsp: ForQuoteRsp) -> None:
        """询价通知"""
        pass

    # 保证金监控中心用户令牌
    def on_rtn_cfmmc_trading_account_token(self, cfmmc_trading_account_token: CFMMCTradingAccountToken) -> None:
        """保证金监控中心用户令牌"""
        pass

    # 批量报单操作错误回报
    def on_err_rtn_batch_order_action(self, batch_order_action: Optional[BatchOrderAction], rsp_info: RspInfo) -> None:
        """批量报单操作错误回报"""
        pass

    # 期权自对冲通知
    def on_rtn_option_self_close(self, option_self_close: OptionSelfClose) -> None:
        """期权自对冲通知"""
        pass
    def on_err_rtn_option_self_close_insert(self, input_option_self_close: Optional[InputOptionSelfClose], rsp_info: RspInfo) -> None:
        """期权自对冲录入错误回报"""
        pass
    def on_err_rtn_option_self_close_action(self, option_self_close_action: Optional[OptionSelfCloseAction], rsp_info: RspInfo) -> None:
        """期权自对冲操作错误回报"""
        pass

    # 申请组合通知
    def on_rtn_comb_action(self, comb_action: CombAction) -> None:
        """申请组合通知"""
        pass
    def on_err_rtn_comb_action_insert(self, input_comb_action: Optional[InputCombAction], rsp_info: RspInfo) -> None:
        """申请组合录入错误回报"""
        pass

    # 查询签约银行响应
    def on_rsp_qry_contract_bank(self, contract_bank: Optional[ContractBank], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询签约银行响应"""
        pass

    # 查询预埋单响应
    def on_rsp_qry_parked_order(self, parked_order: Optional[ParkedOrder], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询预埋单响应"""
        pass
    def on_rsp_qry_parked_order_action(self, parked_order_action: Optional[ParkedOrderAction], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询预埋撤单响应"""
        pass

    # 查询交易通知响应
    def on_rsp_qry_trading_notice(self, trading_notice: Optional[TradingNotice], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询交易通知响应"""
        pass

    # 查询经纪公司交易参数响应
    def on_rsp_qry_broker_trading_params(self, broker_trading_params: Optional[BrokerTradingParams], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询经纪公司交易参数响应"""
        pass

    # 查询经纪公司交易算法响应
    def on_rsp_qry_broker_trading_algos(self, broker_trading_algos: Optional[BrokerTradingAlgos], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询经纪公司交易算法响应"""
        pass

    # 查询监控中心用户令牌
    def on_rsp_query_cfmmc_trading_account_token(self, query_cfmmc_trading_account_token: Optional[QueryCFMMCTradingAccountToken], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询监控中心用户令牌"""
        pass

    # 银期转账相关通知
    def on_rtn_from_bank_to_future_by_bank(self, rsp_transfer: RspTransfer) -> None:
        """银行发起银行资金转期货通知"""
        pass
    def on_rtn_from_future_to_bank_by_bank(self, rsp_transfer: RspTransfer) -> None:
        """银行发起期货资金转银行通知"""
        pass
    def on_rtn_repeal_from_bank_to_future_by_bank(self, rsp_repeal: RspRepeal) -> None:
        """银行发起冲正银行转期货通知"""
        pass
    def on_rtn_repeal_from_future_to_bank_by_bank(self, rsp_repeal: RspRepeal) -> None:
        """银行发起冲正期货转银行通知"""
        pass
    def on_rtn_from_bank_to_future_by_future(self, rsp_transfer: RspTransfer) -> None:
        """期货发起银行资金转期货通知"""
        pass
    def on_rtn_from_future_to_bank_by_future(self, rsp_transfer: RspTransfer) -> None:
        """期货发起期货资金转银行通知"""
        pass
    def on_rtn_repeal_from_bank_to_future_by_future_manual(self, rsp_repeal: RspRepeal) -> None:
        """系统运行时期货端手工发起冲正银行转期货请求，银行处理完毕后报盘发回的通知"""
        pass
    def on_rtn_repeal_from_future_to_bank_by_future_manual(self, rsp_repeal: RspRepeal) -> None:
        """系统运行时期货端手工发起冲正期货转银行请求，银行处理完毕后报盘发回的通知"""
        pass
    def on_rtn_query_bank_balance_by_future(self, notify_query_account: NotifyQueryAccount) -> None:
        """期货发起查询银行余额通知"""
        pass

    # 银期转账错误回报
    def on_err_rtn_bank_to_future_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo) -> None:
        """期货发起银行资金转期货错误回报"""
        pass
    def on_err_rtn_future_to_bank_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo) -> None:
        """期货发起期货资金转银行错误回报"""
        pass
    def on_err_rtn_repeal_bank_to_future_by_future_manual(self, req_repeal: Optional[ReqRepeal], rsp_info: RspInfo) -> None:
        """系统运行时期货端手工发起冲正银行转期货错误回报"""
        pass
    def on_err_rtn_repeal_future_to_bank_by_future_manual(self, req_repeal: Optional[ReqRepeal], rsp_info: RspInfo) -> None:
        """系统运行时期货端手工发起冲正期货转银行错误回报"""
        pass
    def on_err_rtn_query_bank_balance_by_future(self, req_query_account: Optional[ReqQueryAccount], rsp_info: RspInfo) -> None:
        """期货发起查询银行余额错误回报"""
        pass

    # 期货发起冲正请求通知
    def on_rtn_repeal_from_bank_to_future_by_future(self, rsp_repeal: RspRepeal) -> None:
        """期货发起冲正银行转期货请求，银行处理完毕后报盘发回的通知"""
        pass
    def on_rtn_repeal_from_future_to_bank_by_future(self, rsp_repeal: RspRepeal) -> None:
        """期货发起冲正期货转银行请求，银行处理完毕后报盘发回的通知"""
        pass
    """期货发起冲正期货转银行请求，银行处理完毕后报盘发回的通知"""

    # 期货发起银期转账应答
    def on_rsp_from_bank_to_future_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """期货发起银行资金转期货应答"""
        pass
    def on_rsp_from_future_to_bank_by_future(self, req_transfer: Optional[ReqTransfer], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """期货发起期货资金转银行应答"""
        pass
    def on_rsp_query_bank_account_money_by_future(self, req_query_account: Optional[ReqQueryAccount], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """期货发起查询银行余额应答"""
        pass

    # 银行发起银期开户/销户/变更通知
    def on_rtn_open_account_by_bank(self, open_account: OpenAccount) -> None:
        """银行发起银期开户通知"""
        pass
    def on_rtn_cancel_account_by_bank(self, cancel_account: CancelAccount) -> None:
        """银行发起银期销户通知"""
        pass
    def on_rtn_change_account_by_bank(self, change_account: ChangeAccount) -> None:
        """银行发起变更银行账号通知"""
        pass

    # 查询分类合约响应
    def on_rsp_qry_classified_instrument(self, instrument: Optional[Instrument], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求查询分类合约响应"""
        pass

    # 请求组合优惠比例响应
    def on_rsp_qry_comb_promotion_param(self, comb_promotion_param: Optional[CombPromotionParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """请求组合优惠比例响应"""
        pass

    # SPBM参数查询响应
    def on_rsp_qry_spbm_future_parameter(self, spbm_future_parameter: Optional[SPBMFutureParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPBM期货合约参数查询响应"""
        pass
    def on_rsp_qry_spbm_option_parameter(self, spbm_option_parameter: Optional[SPBMOptionParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPBM期权合约参数查询响应"""
        pass
    def on_rsp_qry_spbm_intra_parameter(self, spbm_intra_parameter: Optional[SPBMIntraParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPBM品种内对锁仓折扣参数查询响应"""
        pass
    def on_rsp_qry_spbm_inter_parameter(self, spbm_inter_parameter: Optional[SPBMInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPBM跨品种抵扣参数查询响应"""
        pass
    def on_rsp_qry_spbm_portf_definition(self, spbm_portf_definition: Optional[SPBMPortfDefinition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPBM组合保证金套餐查询响应"""
        pass
    def on_rsp_qry_spbm_investor_portf_def(self, spbm_investor_portf_def: Optional[SPBMInvestorPortfDef], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者SPBM套餐选择查询响应"""
        pass
    def on_rsp_qry_investor_portf_margin_ratio(self, investor_portf_margin_ratio: Optional[InvestorPortfMarginRatio], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者新型组合保证金系数查询响应"""
        pass
    def on_rsp_qry_investor_prod_spbm_detail(self, investor_prod_spbm_detail: Optional[InvestorProdSPBMDetail], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者产品SPBM明细查询响应"""
        pass
    def on_rsp_qry_investor_commodity_spmm_margin(self, investor_commodity_spmm_margin: Optional[InvestorCommoditySPMMMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者商品组SPMM记录查询响应"""
        pass
    def on_rsp_qry_investor_commodity_group_spmm_margin(self, investor_commodity_group_spmm_margin: Optional[InvestorCommodityGroupSPMMMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者商品群SPMM记录查询响应"""
        pass

    # SPMM参数查询响应
    def on_rsp_qry_spmm_inst_param(self, spmm_inst_param: Optional[SPMMInstParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPMM合约参数查询响应"""
        pass
    def on_rsp_qry_spmm_product_param(self, spmm_product_param: Optional[SPMMProductParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPMM产品参数查询响应"""
        pass

    # SPBM附加跨品种抵扣参数查询响应
    def on_rsp_qry_spbm_add_on_inter_parameter(self, spbm_add_on_inter_parameter: Optional[SPBMAddOnInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """SPBM附加跨品种抵扣参数查询响应"""
        pass

    # RCAMS参数查询响应
    def on_rsp_qry_rcams_comb_product_info(self, rcams_comb_product_info: Optional[RCAMSCombProductInfo], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RCAMS产品组合信息查询响应"""
        pass
    def on_rsp_qry_rcams_instr_parameter(self, rcams_instr_parameter: Optional[RCAMSInstrParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RCAMS同合约风险对冲参数查询响应"""
        pass
    def on_rsp_qry_rcams_intra_parameter(self, rcams_intra_parameter: Optional[RCAMSIntraParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RCAMS品种内风险对冲参数查询响应"""
        pass
    def on_rsp_qry_rcams_inter_parameter(self, rcams_inter_parameter: Optional[RCAMSInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RCAMS跨品种风险折抵参数查询响应"""
        pass
    def on_rsp_qry_rcams_short_opt_adjust_param(self, rcams_short_opt_adjust_param: Optional[RCAMSShortOptAdjustParam], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RCAMS空头期权风险调整参数查询响应"""
        pass
    def on_rsp_qry_rcams_investor_comb_position(self, rcams_investor_comb_position: Optional[RCAMSInvestorCombPosition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RCAMS策略组合持仓查询响应"""
        pass
    def on_rsp_qry_investor_prod_rcams_margin(self, investor_prod_rcams_margin: Optional[InvestorProdRCAMSMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者品种RCAMS保证金查询响应"""
        pass

    # RULE参数查询响应
    def on_rsp_qry_rule_instr_parameter(self, rule_instr_parameter: Optional[RULEInstrParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RULE合约保证金参数查询响应"""
        pass
    def on_rsp_qry_rule_intra_parameter(self, rule_intra_parameter: Optional[RULEIntraParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RULE品种内对锁仓折扣参数查询响应"""
        pass
    def on_rsp_qry_rule_inter_parameter(self, rule_inter_parameter: Optional[RULEInterParameter], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """RULE跨品种抵扣参数查询响应"""
        pass
    def on_rsp_qry_investor_prod_rule_margin(self, investor_prod_rule_margin: Optional[InvestorProdRULEMargin], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者产品RULE保证金查询响应"""
        pass

    # 投资者新组保设置查询响应
    def on_rsp_qry_investor_portf_setting(self, investor_portf_setting: Optional[InvestorPortfSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者新组保设置查询响应"""
        pass

    # 投资者申报费阶梯收取记录查询响应
    def on_rsp_qry_investor_info_comm_rec(self, investor_info_comm_rec: Optional[InvestorInfoCommRec], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者申报费阶梯收取记录查询响应"""
        pass

    # 组合腿信息查询响应
    def on_rsp_qry_comb_leg(self, comb_leg: Optional[CombLeg], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """组合腿信息查询响应"""
        pass

    # 对冲设置相关回调
    def on_rsp_offset_setting(self, input_offset_setting: Optional[InputOffsetSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """对冲设置请求响应"""
        pass
    def on_rsp_cancel_offset_setting(self, input_offset_setting: Optional[InputOffsetSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """对冲设置撤销请求响应"""
        pass
    def on_rtn_offset_setting(self, offset_setting: OffsetSetting) -> None:
        """对冲设置通知"""
        pass
    def on_err_rtn_offset_setting(self, input_offset_setting: Optional[InputOffsetSetting], rsp_info: RspInfo) -> None:
        """对冲设置错误回报"""
        pass
    def on_err_rtn_cancel_offset_setting(self, cancel_offset_setting: Optional[CancelOffsetSetting], rsp_info: RspInfo) -> None:
        """对冲设置撤销错误回报"""
        pass
    def on_rsp_qry_offset_setting(self, offset_setting: Optional[OffsetSetting], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者对冲设置查询响应"""
        pass
    def on_rsp_qry_trader_offer(self, trader_offer: Optional[TraderOffer], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """报单查询响应"""
        pass
    def on_rsp_qry_risk_settle_invest_position(self, risk_settle_invest_position: Optional[RiskSettleInvestPosition], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """投资者风险对冲持仓查询响应"""
        pass
    def on_rsp_qry_risk_settle_product_status(self, risk_settle_product_status: Optional[RiskSettleProductStatus], rsp_info: RspInfo, request_id: int, is_last: bool) -> None:
        """风险对冲产品状态查询响应"""
        pass
