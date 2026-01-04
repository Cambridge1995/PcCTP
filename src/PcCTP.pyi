from typing import TypedDict, NotRequired, Required,  Union, List, Dict, Optional
import numpy as np


class ReqUserLogin(TypedDict):
    """用户登录请求

    用于向 CTP 系统发起登录请求，包含用户身份认证信息。
    """
    trading_day: NotRequired[str]  # 交易日，可选
    broker_id: Required[str]  # 经纪公司代码，必需
    user_id: Required[str]  # 用户代码，必需
    password: Required[str]  # 密码，必需
    user_product_info: NotRequired[str]  # 用户产品信息，可选
    interface_product_info: NotRequired[str]  # 接口产品信息，可选
    protocol_info: NotRequired[str]  # 协议信息，可选
    mac_address: NotRequired[str]  # Mac 地址，可选
    one_time_password: NotRequired[str]  # 一次性口令，可选
    client_ip_address: NotRequired[str]  # 客户端 IP 地址，可选
    login_remark: NotRequired[str]  # 登录备注，可选

class RspUserLogin(TypedDict):
    """登录响应

    CTP 系统返回的登录响应信息，包含会话信息和各交易所时间。
    """
    trading_day: str  # 当前交易日
    login_time: str  # 登录成功时间
    broker_id: str  # 经纪公司代码
    user_id: str  # 用户代码
    system_name: str  # 交易系统名称
    front_id: int  # 前置编号
    session_id: int  # 会话编号
    max_order_ref: str  # 最大报单引用
    shfe_time: str  # 上期所时间
    dce_time: str  # 大商所时间
    czce_time: str  # 郑商所时间
    ffex_time: str  # 中金所时间
    ine_time: str  # 能源中心时间
    gfex_time: str  # 广期所时间

class UserLogout(TypedDict):
    """用户登出请求

    用于向 CTP 系统发起登出请求。
    """
    broker_id: Required[str]  # 经纪公司代码，必需
    user_id: Required[str]  # 用户代码，必需

class RspInfo(TypedDict):
    """响应信息

    CTP API 通用响应信息，包含错误代码和错误描述。
    """
    error_id: int  # 错误代码，0 表示无错误
    error_msg: str  # 错误信息描述

class DepthMarketData(TypedDict):
    """深度行情数据

    CTP 行情推送的完整深度行情数据，包含最新价格、成交量、五档买卖价等信息。
    注意：exchange_id 和 exchange_inst_id 字段在某些服务器上可能为空。
    """
    # ===== 字符串字段 =====
    trading_day: str  # 交易日
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码（可能为空）
    exchange_inst_id: str  # 合约在交易所的代码（可能为空）
    update_time: str  # 最后修改时间 (HH:MM:SS 格式)
    action_day: str  # 业务日期

    # ===== 价格字段 =====
    last_price: float  # 最新价
    pre_settlement_price: float  # 昨结算价
    pre_close_price: float  # 昨收盘价
    pre_open_interest: float  # 昨持仓量
    open_price: float  # 今开盘价
    highest_price: float  # 最高价
    lowest_price: float  # 最低价
    close_price: float  # 今收盘价
    settlement_price: float  # 今结算价
    upper_limit_price: float  # 涨停板价
    lower_limit_price: float  # 跌停板价
    pre_delta: float  # 昨虚实度
    curr_delta: float  # 今虚实度
    average_price: float  # 当日均价

    # ===== 成交量字段 =====
    volume: int  # 成交量
    turnover: float  # 成交金额
    open_interest: float  # 持仓量
    update_millisec: int  # 最后修改毫秒

    # ===== 五档买卖价 =====
    # 买档
    bid_price1: float  # 买一价
    bid_volume1: int  # 买一量
    bid_price2: float  # 买二价
    bid_volume2: int  # 买二量
    bid_price3: float  # 买三价
    bid_volume3: int  # 买三量
    bid_price4: float  # 买四价
    bid_volume4: int  # 买四量
    bid_price5: float  # 买五价
    bid_volume5: int  # 买五量
    # 卖档
    ask_price1: float  # 卖一价
    ask_volume1: int  # 卖一量
    ask_price2: float  # 卖二价
    ask_volume2: int  # 卖二量
    ask_price3: float  # 卖三价
    ask_volume3: int  # 卖三量
    ask_price4: float  # 卖四价
    ask_volume4: int  # 卖四量
    ask_price5: float  # 卖五价
    ask_volume5: int  # 卖五量

class ForQuoteRsp(TypedDict):
    """询价响应

    做市商接收的投资者询价信息。
    """
    trading_day: str  # 交易日
    for_quote_sys_id: str  # 询价编号
    for_quote_time: str  # 询价时间
    action_day: str  # 业务日期
    exchange_id: str  # 交易所代码
    instrument_id: str  # 合约代码

class FensUserInfo(TypedDict):
    """FENS 用户信息

    用于 FENS（行情增强服务）用户注册信息。
    """
    broker_id: Required[str]  # 经纪公司代码，必需
    user_id: Required[str]  # 用户代码，必需
    login_mode: NotRequired[str]  # 登录模式，可选

class QryMulticastInstrument(TypedDict):
    """查询组播合约请求

    用于查询组播合约信息。
    """
    topic_id: NotRequired[int]  # 主题号（整数），可选
    instrument_id: NotRequired[str]  # 合约代码，可选

class MulticastInstrument(TypedDict):
    """组播合约信息

    查询组播合约的响应信息。
    """
    instrument_id: str  # 合约代码
    topic_id: int  # 主题号
    instrument_no: int  # 合约编号
    code_price: float  # 基准价
    volume_multiple: int  # 合约数量乘数
    price_tick: float  # 最小变动价位


# =============================================================================
# 交易 API 数据结构类型定义 (TypedDict)
# =============================================================================

class RspAuthenticate(TypedDict):
    """客户端认证响应"""
    broker_id: str  # 经纪公司代码
    user_id: str  # 用户代码
    user_product_info: str  # 用户端产品信息
    app_id: str  # App代码
    app_type: str  # App类型

class UserPasswordUpdate(TypedDict):
    """用户口令更新请求"""
    broker_id: str
    user_id: str
    old_password: str
    new_password: str

class TradingAccountPasswordUpdate(TypedDict):
    """资金账户口令更新请求"""
    broker_id: str
    account_id: str
    old_password: str
    new_password: str
    currency_id: str

class InputOrder(TypedDict):
    """报单录入请求"""
    broker_id: str
    investor_id: str
    order_ref: str
    user_id: str
    instrument_id: str
    exchange_id: str
    order_price_type: str
    direction: str
    comb_offset_flag: str
    comb_hedge_flag: str
    limit_price: float
    volume_total_original: int
    time_condition: str
    gtd_date: str
    volume_condition: str
    min_volume: int
    contingent_condition: str
    stop_price: float
    force_close_reason: str
    is_auto_suspend: int
    business_unit: str
    user_force_close: int
    is_swap_order: int
    invest_unit_id: str
    account_id: str
    currency_id: str
    client_id: str
    ip_address: str
    mac_address: str
    order_memo: str

class Order(TypedDict):
    """报单"""
    broker_id: str
    investor_id: str
    order_ref: str
    user_id: str
    instrument_id: str
    exchange_id: str
    order_price_type: str
    direction: str
    comb_offset_flag: str
    comb_hedge_flag: str
    limit_price: float
    volume_total_original: int
    time_condition: str
    gtd_date: str
    volume_condition: str
    min_volume: int
    contingent_condition: str
    stop_price: float
    force_close_reason: str
    is_auto_suspend: int
    order_status: str
    order_submit_status: str
    volume_traded: int
    volume_total: int
    front_id: int
    session_id: int
    order_sys_id: str
    order_local_id: str
    participant_id: str
    client_id: str
    trader_id: str
    trading_day: str
    settlement_id: int
    order_source: str
    order_type: str
    insert_date: str
    insert_time: str
    active_time: str
    suspend_time: str
    update_time: str
    cancel_time: str
    active_trader_id: str
    business_unit: str
    user_force_close: int
    is_swap_order: int
    clearing_part_id: str
    sequence_no: int
    broker_order_seq: int
    status_msg: str
    branch_id: str
    invest_unit_id: str
    account_id: str
    currency_id: str
    ip_address: str
    mac_address: str
    order_memo: str

class Trade(TypedDict):
    """成交"""
    broker_id: str
    investor_id: str
    order_ref: str
    user_id: str
    instrument_id: str
    exchange_id: str
    direction: str
    offset_flag: str
    hedge_flag: str
    price: float
    volume: int
    trade_id: str
    order_sys_id: str
    order_local_id: str
    participant_id: str
    client_id: str
    trader_id: str
    trading_role: str
    trading_day: str
    settlement_id: int
    trade_type: str
    price_source: str
    trade_source: str
    trade_date: int
    trade_time: str
    business_unit: str
    clearing_part_id: str
    sequence_no: int
    broker_order_seq: int
    branch_id: str
    invest_unit_id: str
    account_id: str
    currency_id: str
    ip_address: str
    mac_address: str

class InputOrderAction(TypedDict):
    """报单操作请求"""
    broker_id: str
    investor_id: str
    order_action_ref: str
    order_ref: str
    user_id: str
    instrument_id: str
    exchange_id: str
    front_id: int
    session_id: int
    order_sys_id: str
    action_flag: str
    limit_price: float
    volume_change: int
    invest_unit_id: str
    ip_address: str
    mac_address: str
    order_memo: str

class OrderAction(TypedDict):
    """报单操作"""
    broker_id: str
    investor_id: str
    order_action_ref: str
    order_ref: str
    user_id: str
    instrument_id: str
    exchange_id: str
    front_id: int
    session_id: int
    order_sys_id: str
    action_flag: str
    limit_price: float
    volume_change: int
    action_date: str
    action_time: str
    trader_id: str
    install_id: int
    order_action_status: str
    branch_id: str
    invest_unit_id: str
    ip_address: str
    mac_address: str

class ParkedOrder(TypedDict):
    """预埋单"""
    broker_id: str
    investor_id: str
    order_ref: str
    user_id: str
    instrument_id: str
    exchange_id: str
    order_price_type: str
    direction: str
    comb_offset_flag: str
    comb_hedge_flag: str
    limit_price: float
    volume_total_original: int
    time_condition: str
    gtd_date: str
    volume_condition: str
    min_volume: int
    contingent_condition: str
    stop_price: float
    force_close_reason: str
    is_auto_suspend: int
    business_unit: str
    user_force_close: int
    parked_order_id: str
    user_type: str
    status: str
    error_id: int
    error_msg: str
    invest_unit_id: str
    account_id: str
    currency_id: str
    client_id: str
    ip_address: str
    mac_address: str

class ParkedOrderAction(TypedDict):
    """预埋撤单"""
    broker_id: str
    investor_id: str
    order_action_ref: str
    order_ref: str
    user_id: str
    instrument_id: str
    exchange_id: str
    front_id: int
    session_id: int
    order_sys_id: str
    action_flag: str
    limit_price: float
    volume_change: int
    parked_order_action_id: str
    user_type: str
    status: str
    error_id: int
    error_msg: str
    invest_unit_id: str
    branch_id: str
    ip_address: str
    mac_address: str

class InvestorPosition(TypedDict):
    """投资者持仓"""
    broker_id: str
    investor_id: str
    instrument_id: str
    posi_direction: str
    hedge_flag: str
    position_date: str
    yd_position: int
    position: int
    long_frozen: int
    short_frozen: int
    long_frozen_amount: float
    short_frozen_amount: float
    open_volume: int
    close_volume: int
    open_amount: float
    close_amount: float
    position_cost: float
    pre_margin: float
    use_margin: float
    frozen_margin: float
    frozen_cash: float
    frozen_commission: float
    cash_in: float
    commission: float
    close_profit: float
    position_profit: float
    settlement_id: int
    trading_day: str
    invest_unit_id: str
    exchange_id: str

class TradingAccount(TypedDict):
    """资金账户"""
    broker_id: str
    account_id: str
    trading_day: str
    settlement_id: int
    pre_mortgage: float
    pre_credit: float
    pre_deposit: float
    pre_balance: float
    pre_margin: float
    deposit: float
    withdraw: float
    frozen_margin: float
    frozen_cash: float
    frozen_commission: float
    curr_margin: float
    cash_in: float
    commission: float
    close_profit: float
    position_profit: float
    balance: float
    available: float
    withdraw_quota: float
    reserve: float
    credit: float
    mortgage: float
    exchange_margin: float
    delivery_margin: float
    exchange_delivery_margin: float
    interest_base: float
    interest: float
    currency_id: str

class Investor(TypedDict):
    """投资者"""
    investor_id: str
    broker_id: str
    investor_group_id: str
    investor_name: str
    identified_card_type: str
    identified_card_no: str
    is_active: int

class TradingCode(TypedDict):
    """交易编码"""
    investor_id: str
    broker_id: str
    exchange_id: str
    client_id: str
    is_active: int
    client_id_type: str
    branch_id: str
    biz_type: str
    invest_unit_id: str

class InstrumentMarginRate(TypedDict):
    """合约保证金率"""
    broker_id: str
    investor_id: str
    instrument_id: str
    exchange_id: str
    invest_unit_id: str
    investor_range: str
    hedge_flag: str
    long_margin_ratio_by_money: float
    long_margin_ratio_by_volume: float
    short_margin_ratio_by_money: float
    short_margin_ratio_by_volume: float
    is_relative: int

class InstrumentCommissionRate(TypedDict):
    """合约手续费率"""
    broker_id: str
    investor_id: str
    instrument_id: str
    exchange_id: str
    invest_unit_id: str
    investor_range: str
    open_ratio_by_money: float
    open_ratio_by_volume: float
    close_ratio_by_money: float
    close_ratio_by_volume: float
    close_today_ratio_by_money: float
    close_today_ratio_by_volume: float
    biz_type: str

class Instrument(TypedDict):
    """合约"""
    instrument_id: str
    exchange_id: str
    instrument_name: str
    product_class: str
    delivery_year: int
    delivery_month: int
    create_date: str
    open_date: str
    expire_date: str
    start_deliv_date: str
    end_deliv_date: str
    volume_multiple: int
    price_tick: float
    max_market_order_volume: int
    min_market_order_volume: int
    max_limit_order_volume: int
    min_limit_order_volume: int
    inst_life_phase: str
    is_trading: int
    position_type: str
    position_date_type: str
    long_margin_ratio: float
    short_margin_ratio: float
    max_margin_side_algorithm: str
    strike_price: float
    options_type: str
    underlying_instr_id: str
    strike_mode: str

class SettlementInfo(TypedDict):
    """投资者结算结果"""
    trading_day: str
    settlement_id: int
    broker_id: str
    investor_id: str
    sequence_no: int
    content: str
    account_id: str
    currency_id: str

class InstrumentStatus(TypedDict):
    """合约交易状态"""
    exchange_id: str
    settlement_group_id: str
    instrument_status: str
    trading_segment_sn: int
    enter_time: str
    enter_reason: str
    exchange_inst_id: str
    instrument_id: str
class RspGenUserCaptcha(TypedDict):
    """获取图形验证码请求的回复"""
    broker_id: str  # 经纪公司代码
    user_id: str  # 用户代码
    captcha_info_len: int  # 图片信息长度
    captcha_info: bytes  # 图片信息 (二进制数据)


class RspGenUserText(TypedDict):
    """获取短信验证码请求的回复"""
    user_text_seq: int  # 短信验证码序号


class SettlementInfoConfirm(TypedDict):
    """投资者结算结果确认"""
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    confirm_date: str  # 确认日期
    confirm_time: str  # 确认时间
    settlement_id: int  # 结算编号


class Exchange(TypedDict):
    """交易所"""
    exchange_id: str  # 交易所代码
    exchange_name: str  # 交易所名称
    exchange_property: str  # 交易所属性 (ExchangeProperty枚举)


class Product(TypedDict):
    """产品"""
    product_name: str  # 产品名称
    exchange_id: str  # 交易所代码
    product_class: str  # 产品类型 (ProductClass枚举)
    volume_multiple: int  # 合约数量乘数
    price_tick: float  # 最小变动价位
    max_market_order_volume: int  # 市价单最大下单量
    min_market_order_volume: int  # 市价单最小下单量
    max_limit_order_volume: int  # 限价单最大下单量
    min_limit_order_volume: int  # 限价单最小下单量


class TransferBank(TypedDict):
    """查询转帐银行响应"""
    bank_id: str  # 银行代码
    bank_brch_id: str  # 银行分中心代码
    bank_name: str  # 银行名称
    is_active: int  # 是否活跃


class InvestorPositionDetail(TypedDict):
    """查询投资者持仓明细响应"""
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    hedge_flag: str  # 投机套保标志 (HedgeFlag枚举)
    direction: str  # 买卖 (Direction枚举)
    open_date: str  # 开仓日期
    trade_id: str  # 成交编号
    volume: int  # 数量
    open_price: float  # 开仓价
    trading_day: str  # 交易日
    settlement_id: int  # 结算编号
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码
    close_profit_by_date: float  # 按日平仓盈亏
    position_profit_by_date: float  # 按日持仓盈亏
    margin: float  # 保证金
    exch_margin: float  # 交易所保证金
    close_amount: float  # 平仓金额
    margin_rate_by_money: float  # 保证金率
    last_settlement_price: float  # 上次结算价
    settlement_price: float  # 结算价
    close_volume: int  # 平仓量
    diff_price: float  # 价格差


class Notice(TypedDict):
    """查询客户通知响应"""
    broker_id: str  # 经纪公司代码
    content: str  # 消息正文
    sequence_label: str  # 经纪公司通知内容序列号


class TradingNotice(TypedDict):
    """交易通知"""
    broker_id: str  # 经纪公司代码
    investor_range: str  # 投资者范围 (InvestorRange枚举)
    investor_id: str  # 投资者代码
    sequence_series: int  # 序列系列号
    user_id: str  # 用户代码
    send_time: str  # 发送时间
    sequence_no: int  # 序列号
    field_content: str  # 域内容


class Bulletin(TypedDict):
    """交易所公告通知"""
    exchange_id: str  # 交易所代码
    trading_day: str  # 交易日
    bulletin_id: str  # 公告编号
    sequence_no: int  # 序列号
    news_type: str  # 公告类型
    news_urgency: str  # 紧急程度
    send_time: str  # 发送时间
    abstract: str  # 摘要
    content: bytes  # 内容 (二进制数据)
    source_file: bytes  # 来源文件 (二进制数据)


class ErrorConditionalOrder(TypedDict):
    """提示条件单校验错误"""
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    order_price_type: str  # 报单价格条件 (OrderPriceType枚举)
    direction: str  # 买卖方向 (Direction枚举)
    limit_price: float  # 价格
    volume_total_original: int  # 数量
    time_condition: str  # 有效期类型
    gtd_date: str  # GTD日期
    volume_condition: str  # 成交量类型
    min_volume: int  # 最小成交量
    contingent_condition: str  # 触发条件
    stop_price: float  # 止损价
    force_close_reason: str  # 强平原因
    is_auto_suspend: int  # 自动挂起标志
    business_unit: str  # 业务单元
    user_force_close: int  # 用户强平标志
    error_code: str  # 错误代码
    error_msg: str  # 错误信息
    req_rtn_order_insert_field: str  # 请求报单录入字段
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码


class ContractBank(TypedDict):
    """查询签约银行响应"""
    broker_id: str  # 经纪公司代码
    bank_id: str  # 银行代码
    bank_brch_id: str  # 银行分中心代码
    bank_name: str  # 银行名称


class QueryCFMMCTradingAccountToken(TypedDict):
    """请求查询监控中心用户令牌"""
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    invest_unit_id: str  # 投资单元代码


class CFMMCTradingAccountToken(TypedDict):
    """保证金监控中心用户令牌"""
    broker_id: str  # 经纪公司代码
    participant_id: str  # 经纪公司统一编码
    account_id: str  # 投资者账号
    key_id: int  # 密钥编号
    token: str  # 动态令牌
class InputExecOrder(TypedDict):
    """输入执行宣告"""
    broker_id: str
    investor_id: str
    exec_order_ref: str
    user_id: str
    volume: int
    request_id: int
    business_unit: str
    offset_flag: str  # 枚举: OffsetFlagType
    hedge_flag: str  # 枚举: HedgeFlagType
    action_type: str  # 枚举: ActionTypeType
    posi_direction: str  # 枚举: PosiDirectionType
    reserve_position_flag: str  # 枚举: ExecOrderPositionFlagType
    close_flag: str  # 枚举: ExecOrderCloseFlagType
    exchange_id: str
    invest_unit_id: str
    account_id: str
    currency_id: str
    instrument_id: str


class InputExecOrderAction(TypedDict):
    """输入执行宣告操作"""
    broker_id: str
    investor_id: str
    exec_order_action_ref: str
    exec_order_ref: str
    request_id: int
    front_id: int
    session_id: int
    exchange_id: str
    exec_order_sys_id: str
    action_flag: str  # 枚举: ActionFlagType
    user_id: str
    invest_unit_id: str
    instrument_id: str


class ExecOrder(TypedDict):
    """执行宣告"""
    broker_id: str
    investor_id: str
    exec_order_ref: str
    user_id: str
    volume: int
    request_id: int
    business_unit: str
    offset_flag: str  # 枚举: OffsetFlagType
    hedge_flag: str  # 枚举: HedgeFlagType
    action_type: str  # 枚举: ActionTypeType
    posi_direction: str  # 枚举: PosiDirectionType
    reserve_position_flag: str  # 枚举: ExecOrderPositionFlagType
    close_flag: str  # 枚举: ExecOrderCloseFlagType
    exec_order_local_id: str
    exchange_id: str
    trading_day: str
    settlement_id: int
    exec_order_sys_id: str
    insert_date: str
    insert_time: str
    cancel_time: str
    exec_result: str  # 枚举: ExecResultType
    sequence_no: int
    front_id: int
    session_id: int
    user_product_info: str
    status_msg: str
    active_user_id: str
    broker_exec_order_seq: int
    branch_id: str
    invest_unit_id: str
    account_id: str
    currency_id: str
    instrument_id: str


class InputForQuote(TypedDict):
    """输入的询价"""
    broker_id: str
    investor_id: str
    for_quote_ref: str
    user_id: str
    exchange_id: str
    invest_unit_id: str
    instrument_id: str


class InputQuote(TypedDict):
    """输入的报价"""
    broker_id: str
    investor_id: str
    quote_ref: str
    user_id: str
    ask_price: float
    bid_price: float
    ask_volume: int
    bid_volume: int
    request_id: int
    business_unit: str
    ask_offset_flag: str  # 枚举: OffsetFlagType
    bid_offset_flag: str  # 枚举: OffsetFlagType
    ask_hedge_flag: str  # 枚举: HedgeFlagType
    bid_hedge_flag: str  # 枚举: HedgeFlagType
    ask_order_ref: str
    bid_order_ref: str
    for_quote_sys_id: str
    exchange_id: str
    invest_unit_id: str
    client_id: str
    instrument_id: str
    replace_sys_id: str
    time_condition: str  # 枚举: TimeConditionType
    order_memo: str


class InputQuoteAction(TypedDict):
    """输入报价操作"""
    broker_id: str
    investor_id: str
    quote_action_ref: str
    quote_ref: str
    request_id: int
    front_id: int
    session_id: int
    exchange_id: str
    quote_sys_id: str
    action_flag: str  # 枚举: ActionFlagType
    user_id: str
    invest_unit_id: str
    client_id: str
    instrument_id: str
    order_memo: str





class Quote(TypedDict):
    """报价"""
    broker_id: str
    investor_id: str
    quote_ref: str
    user_id: str
    ask_price: float
    bid_price: float
    ask_volume: int
    bid_volume: int
    request_id: int
    business_unit: str
    ask_offset_flag: str  # 枚举: OffsetFlagType
    bid_offset_flag: str  # 枚举: OffsetFlagType
    ask_hedge_flag: str  # 枚举: HedgeFlagType
    bid_hedge_flag: str  # 枚举: HedgeFlagType
    ask_order_ref: str
    bid_order_ref: str
    for_quote_sys_id: str
    exchange_id: str
    quote_local_id: str
    quote_sys_id: str
    trading_day: str
    settlement_id: int
    quote_status: str  # 枚举: ForQuoteStatusType
    front_id: int
    session_id: int
    status_msg: str
    active_user_id: str
    broker_quote_seq: int
    invest_unit_id: str
    instrument_id: str
    exchange_inst_id: str


class ForQuote(TypedDict):
    """询价"""
    broker_id: str
    investor_id: str
    for_quote_ref: str
    user_id: str
    for_quote_local_id: str
    exchange_id: str
    insert_date: str
    insert_time: str
    for_quote_status: str  # 枚举: ForQuoteStatusType
    front_id: int
    session_id: int
    status_msg: str
    active_user_id: str
    instrument_id: str


class QryMaxOrderVolume(TypedDict):
    """查询最大报单数量"""
    broker_id: str
    investor_id: str
    direction: str  # 枚举: DirectionType
    offset_flag: str  # 枚举: OffsetFlagType
    hedge_flag: str  # 枚举: HedgeFlagType
    max_volume: int
    exchange_id: str
    invest_unit_id: str
    instrument_id: str


class InputBatchOrderAction(TypedDict):
    """输入批量报单操作"""
    broker_id: str
    investor_id: str
    order_action_ref: str
    request_id: int
    front_id: int
    session_id: int
    exchange_id: str
    user_id: str
    invest_unit_id: str


class BatchOrderAction(TypedDict):
    """批量报单操作"""
    broker_id: str
    investor_id: str
    order_action_ref: str
    request_id: int
    front_id: int
    session_id: int
    exchange_id: str
    action_date: str
    action_time: str
    action_local_id: str
    order_action_status: str  # 枚举: OrderActionStatusType
    user_id: str
    status_msg: str
    invest_unit_id: str

class SecAgentACIDMap(TypedDict):
    """二级代理ACID映射"""
    broker_id: str
    user_id: str
    account_id: str
    currency_id: str
    broker_sec_agent_id: str


class ProductExchRate(TypedDict):
    """产品汇率"""
    quote_currency_id: str
    exchange_rate: float
    exchange_id: str
    product_id: str


class ProductGroup(TypedDict):
    """产品组"""
    exchange_id: str
    product_id: str
    product_group_id: str


class MMInstrumentCommissionRate(TypedDict):
    """做市商合约手续费率"""
    broker_id: str
    investor_id: str
    open_ratio_by_money: float
    open_ratio_by_volume: float
    close_ratio_by_money: float
    close_ratio_by_volume: float
    close_today_ratio_by_money: float
    close_today_ratio_by_volume: float
    instrument_id: str


class MMOptionInstrCommRate(TypedDict):
    """做市商期权合约手续费率"""
    broker_id: str
    investor_id: str
    open_ratio_by_money: float
    open_ratio_by_volume: float
    close_ratio_by_money: float
    close_ratio_by_volume: float
    close_today_ratio_by_money: float
    close_today_ratio_by_volume: float


class InstrumentOrderCommRate(TypedDict):
    """合约报单手续费率"""
    broker_id: str
    investor_id: str
    hedge_flag: str  # 枚举: HedgeFlagType
    order_comm_by_volume: float
    order_action_comm_by_volume: float
    exchange_id: str
    invest_unit_id: str
    instrument_id: str


class SecAgentCheckMode(TypedDict):
    """二级代理校验模式"""
    investor_id: str
    broker_id: str
    currency_id: str
    broker_sec_agent_id: str
    check_self_account: bool


class SecAgentTradeInfo(TypedDict):
    """二级代理交易信息"""
    broker_id: str
    broker_sec_agent_id: str
    investor_id: str
    long_customer_name: str


class OptionInstrTradeCost(TypedDict):
    """期权合约交易成本"""
    broker_id: str
    investor_id: str
    hedge_flag: str  # 枚举: HedgeFlagType
    fixed_margin: float
    mini_margin: float
    royalty: float
    exch_fixed_margin: float
    exch_mini_margin: float
    exchange_id: str
    invest_unit_id: str
    instrument_id: str


class OptionInstrCommRate(TypedDict):
    """期权合约手续费率"""
    broker_id: str
    investor_id: str
    open_ratio_by_money: float
    open_ratio_by_volume: float
    close_ratio_by_money: float
    close_ratio_by_volume: float
    close_today_ratio_by_money: float
    close_today_ratio_by_volume: float
    strike_ratio_by_money: float
    strike_ratio_by_volume: float


class BrokerTradingParams(TypedDict):
    """经纪公司交易参数"""
    broker_id: str
    investor_id: str
    margin_price_type: str  # 枚举: MarginPriceTypeType
    algorithm: str  # 枚举: AlgorithmType
    avail_include_close_profit: str  # 枚举: IncludeCloseProfitType
    currency_id: str
    option_royalty_price_type: str  # 枚举: OptionRoyaltyPriceTypeType
    account_id: str


class BrokerTradingAlgos(TypedDict):
    """经纪公司交易算法"""
    broker_id: str
    exchange_id: str
    handle_position_algo_id: str
    find_margin_rate_algo_id: str
    handle_trading_account_algo_id: str
    instrument_id: str


class RemoveParkedOrder(TypedDict):
    """删除预埋报单"""
    broker_id: str
    investor_id: str
    parked_order_id: str
    invest_unit_id: str


class RemoveParkedOrderAction(TypedDict):
    """删除预埋撤单"""
    broker_id: str
    investor_id: str
    parked_order_action_id: str
    invest_unit_id: str



class InputOptionSelfClose(TypedDict):
    """输入期权自对冲"""
    broker_id: str
    investor_id: str
    option_self_close_ref: str
    user_id: str
    volume: int
    request_id: int
    business_unit: str
    hedge_flag: str  # 枚举: HedgeFlagType
    opt_self_close_flag: str  # 枚举: OptSelfCloseFlagType
    exchange_id: str
    invest_unit_id: str
    account_id: str
    currency_id: str
    client_id: str
    instrument_id: str


class InputOptionSelfCloseAction(TypedDict):
    """输入期权自对冲操作"""
    broker_id: str
    investor_id: str
    option_self_close_action_ref: str
    option_self_close_ref: str
    request_id: int
    front_id: int
    session_id: int
    exchange_id: str
    option_self_close_sys_id: str
    action_flag: str  # 枚举: ActionFlagType
    user_id: str
    invest_unit_id: str
    instrument_id: str


class OptionSelfClose(TypedDict):
    """期权自对冲"""
    broker_id: str
    investor_id: str
    option_self_close_ref: str
    user_id: str
    volume: int
    request_id: int
    business_unit: str
    hedge_flag: str  # 枚举: HedgeFlagType
    opt_self_close_flag: str  # 枚举: OptSelfCloseFlagType
    option_self_close_local_id: str
    exchange_id: str
    trading_day: str
    settlement_id: int
    option_self_close_sys_id: str
    insert_date: str
    insert_time: str
    cancel_time: str
    exec_result: str  # 枚举: ExecResultType
    sequence_no: int
    front_id: int
    session_id: int
    user_product_info: str
    status_msg: str
    active_user_id: str
    broker_option_self_close_seq: int
    branch_id: str
    invest_unit_id: str
    account_id: str
    currency_id: str
    instrument_id: str


class InputCombAction(TypedDict):
    """输入的申请组合"""
    broker_id: str
    investor_id: str
    comb_action_ref: str
    user_id: str
    direction: str  # 枚举: DirectionType
    volume: int
    comb_direction: str  # 枚举: CombDirectionType
    hedge_flag: str  # 枚举: HedgeFlagType
    exchange_id: str
    invest_unit_id: str
    instrument_id: str


class CombAction(TypedDict):
    """申请组合"""
    broker_id: str
    investor_id: str
    comb_action_ref: str
    user_id: str
    direction: str  # 枚举: DirectionType
    volume: int
    comb_direction: str  # 枚举: CombDirectionType
    hedge_flag: str  # 枚举: HedgeFlagType
    action_local_id: str
    exchange_id: str
    trading_day: str
    settlement_id: int
    sequence_no: int
    front_id: int
    session_id: int
    user_product_info: str
    status_msg: str
    com_trade_id: str
    branch_id: str
    invest_unit_id: str
    instrument_id: str


class CombInstrumentGuard(TypedDict):
    """组合合约安全系数"""
    broker_id: str
    guarant_ratio: int
    exchange_id: str
    instrument_id: str


class InvestorPositionCombineDetail(TypedDict):
    """投资者持仓组合明细"""
    trading_day: str
    open_date: str
    exchange_id: str
    settlement_id: int
    broker_id: str
    investor_id: str
    com_trade_id: str
    trade_id: str
    hedge_flag: str  # 枚举: HedgeFlagType
    direction: str  # 枚举: DirectionType
    total_amt: int
    margin: float
    exch_margin: float
    margin_rate_by_money: float
    margin_rate_by_volume: float
    leg_id: str
    leg_multiple: int
    trade_group_id: str
    invest_unit_id: str
    instrument_id: str
    comb_instrument_id: str


class EWarrantOffset(TypedDict):
    """仓单折抵信息"""
    trading_day: str
    broker_id: str
    investor_id: str
    exchange_id: str
    direction: str  # 枚举: DirectionType
    hedge_flag: str  # 枚举: HedgeFlagType
    volume: int
    invest_unit_id: str
    instrument_id: str


class InvestorProductGroupMargin(TypedDict):
    """投资者品种/跨品种保证金"""
    broker_id: str
    investor_id: str
    trading_day: str
    settlement_id: int
    frozen_margin: float
    long_frozen_margin: float
    short_frozen_margin: float
    use_margin: float
    long_use_margin: float
    short_use_margin: float


class TransferSerial(TypedDict):
    """转账流水"""
    plate_serial: int
    trade_date: str
    trading_day: str
    trade_time: str
    trade_code: str
    session_id: int
    bank_id: str
    bank_account: str
    broker_id: str
    investor_id: str
    account_id: str
    currency_id: str
    trade_amount: float
    cust_fee: float


class AccountRegister(TypedDict):
    """银期签约"""
    trade_day: str
    bank_id: str
    bank_account: str
    broker_id: str
    account_id: str
    customer_name: str
    currency_id: str
    reg_date: str
    out_date: str


class ExchangeMarginRate(TypedDict):
    """交易所保证金率"""
    broker_id: str
    hedge_flag: str  # 枚举: HedgeFlagType
    long_margin_ratio_by_money: float
    long_margin_ratio_by_volume: float
    short_margin_ratio_by_money: float
    short_margin_ratio_by_volume: float
    exchange_id: str
    instrument_id: str


class ExchangeMarginRateAdjust(TypedDict):
    """交易所保证金率调整"""
    broker_id: str
    hedge_flag: str  # 枚举: HedgeFlagType
    long_margin_ratio_by_money: float
    long_margin_ratio_by_volume: float
    short_margin_ratio_by_money: float
    short_margin_ratio_by_volume: float
    exchange_id: str
    instrument_id: str


class ExchangeRate(TypedDict):
    """汇率"""
    broker_id: str
    from_currency_id: str
    from_currency_unit: int
    to_currency_id: str
    exchange_rate: float


class InvestUnit(TypedDict):
    """投资单元"""
    broker_id: str
    investor_id: str
    invest_unit_id: str
    investor_unit_name: str
    investor_group_id: str
    comm_model_id: str
    margin_model_id: str
    account_id: str
    currency_id: str
class RspTransfer(TypedDict):
    """转账响应 (核心字段)"""
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    customer_name: str
    id_card_type: str
    identified_card_no: str
    cust_type: str
    bank_account: str
    account_id: str
    user_id: str
    currency_id: str
    trade_amount: float
    future_fetch_amount: float
    fee_pay_flag: str
    cust_fee: float
    broker_fee: float
    message: str
    request_id: int
    tid: str
    transfer_status: str
    error_id: int
    error_msg: str


class RspRepeal(TypedDict):
    """冲正响应 (核心字段)"""
    repeal_time_interval: int
    repealed_times: int
    bank_repeal_flag: str
    broker_repeal_flag: str
    plate_repeal_serial: int
    bank_repeal_serial: str
    future_repeal_serial: int
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    customer_name: str
    account_id: str
    user_id: str
    currency_id: str
    trade_amount: float
    message: str
    request_id: int
    tid: str
    error_id: int
    error_msg: str


class NotifyQueryAccount(TypedDict):
    """查询账户通知 (核心字段)"""
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    customer_name: str
    bank_account: str
    account_id: str
    user_id: str
    currency_id: str
    bank_balance: float
    future_fetch_amount: float
    error_id: int
    error_msg: str


class ReqTransfer(TypedDict):
    """转账请求 (核心字段)"""
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    customer_name: str
    bank_account: str
    account_id: str
    user_id: str
    currency_id: str
    trade_amount: float


class ReqRepeal(TypedDict):
    """冲正请求 (核心字段)"""
    repeal_time_interval: int
    repealed_times: int
    bank_repeal_flag: str
    broker_repeal_flag: str
    plate_repeal_serial: int
    bank_repeal_serial: str
    future_repeal_serial: int
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    bank_account: str
    account_id: str
    user_id: str
    currency_id: str
    trade_amount: float


class ReqQueryAccount(TypedDict):
    """查询账户请求 (核心字段)"""
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    bank_account: str
    account_id: str
    user_id: str
    currency_id: str


class OpenAccount(TypedDict):
    """开户 (核心字段)"""
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    customer_name: str
    bank_account: str
    account_id: str
    user_id: str
    error_id: int
    error_msg: str


class CancelAccount(TypedDict):
    """销户 (核心字段)"""
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    customer_name: str
    bank_account: str
    account_id: str
    user_id: str
    error_id: int
    error_msg: str


class ChangeAccount(TypedDict):
    """变更账户 (核心字段)"""
    trade_code: str
    bank_id: str
    bank_branch_id: str
    broker_id: str
    future_branch_id: str
    trade_date: str
    trade_time: str
    bank_serial: str
    trading_day: str
    plate_serial: int
    session_id: int
    customer_name: str
    bank_account: str
    account_id: str
    user_id: str
    error_id: int
    error_msg: str

# =============================================================================
# PyMdSpi 和 MdApi 类定义（仅在类型检查时可用）
# 这些类由 C++ 模块提供，这里仅定义类型提示用于 IDE 和 mypy
# =============================================================================

class PyMdSpi:
    """
    CTP PC版行情回调接口基类

    方案3说明：
    - 此类仅用于类型提示
    - 实际使用时不需要继承任何基类
    - 只需实现对应的方法即可
    """
    def __init__(self) -> None: ...

    def on_front_connected(self) -> None: ...
    def on_front_disconnected(self, reason: int) -> None: ...
    def on_heart_beat_warning(self, time_lapse: int) -> None: ...
    def on_rsp_user_login(self, rsp_user_login: RspUserLogin, rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...
    def on_rsp_user_logout(self, user_logout: UserLogout, rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...
    def on_rsp_error(self, rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...
    def on_rsp_sub_market_data(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...
    def on_rsp_un_sub_market_data(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...
    def on_rsp_sub_for_quote_rsp(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...
    def on_rsp_un_sub_for_quote_rsp(self, instrument_id: Optional[str], rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...
    def on_rtn_depth_market_data(self, depth_market_data: DepthMarketData) -> None: ...
    def on_rtn_for_quote_rsp(self, for_quote_rsp: ForQuoteRsp) -> None: ...
    def on_rsp_qry_multicast_instrument(self, multicast_instrument: Optional[MulticastInstrument], rsp_info: RspInfo, request_id: int, is_last: bool) -> None: ...

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
    def register_front(self, front_address: str) -> None: ...
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

class PyTradeSpi:
    """
    CTP PC版交易回调接口基类

    方案3说明：
    - 此类仅用于类型提示
    - 实际使用时不需要继承任何基类
    - 只需实现对应的方法即可
    """
    def __init__(self) -> None: ...

    # 连接相关 (3个)
    def on_front_connected(self) -> None: ...
    def on_front_disconnected(self, reason: int) -> None: ...
    def on_heart_beat_warning(self, time_lapse: int) -> None: ...

    # 认证登录相关 (5个)
    def on_rsp_authenticate(self, rsp_authenticate: Optional["RspAuthenticate"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_user_login(self, rsp_user_login: Optional["RspUserLogin"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_user_logout(self, user_logout: Optional["UserLogout"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_user_password_update(self, user_password_update: Optional["UserPasswordUpdate"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_trading_account_password_update(self, trading_account_password_update: Optional["TradingAccountPasswordUpdate"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...

    # 报单相关 (8个)
    def on_rsp_order_insert(self, input_order: Optional["InputOrder"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_parked_order_insert(self, parked_order: Optional["ParkedOrder"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_parked_order_action(self, parked_order_action: Optional["ParkedOrderAction"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_order_action(self, input_order_action: Optional["InputOrderAction"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rtn_order(self, order: "Order") -> None: ...
    def on_rtn_trade(self, trade: "Trade") -> None: ...
    def on_err_rtn_order_insert(self, input_order: Optional["InputOrder"], rsp_info: "RspInfo") -> None: ...
    def on_err_rtn_order_action(self, order_action: Optional["OrderAction"], rsp_info: "RspInfo") -> None: ...

    # 查询相关 (10个)
    def on_rsp_qry_order(self, order: Optional["Order"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_trade(self, trade: Optional["Trade"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_investor_position(self, investor_position: Optional["InvestorPosition"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_trading_account(self, trading_account: Optional["TradingAccount"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_investor(self, investor: Optional["Investor"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_trading_code(self, trading_code: Optional["TradingCode"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_instrument_margin_rate(self, instrument_margin_rate: Optional["InstrumentMarginRate"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_instrument_commission_rate(self, instrument_commission_rate: Optional["InstrumentCommissionRate"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_instrument(self, instrument: Optional["Instrument"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rsp_qry_settlement_info(self, settlement_info: Optional["SettlementInfo"], rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...

    # 错误和状态 (2个)
    def on_rsp_error(self, rsp_info: "RspInfo", request_id: int, is_last: bool) -> None: ...
    def on_rtn_instrument_status(self, instrument_status: "InstrumentStatus") -> None: ...


# =============================================================================
# 字符串池监控和清理工具函数
# =============================================================================


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
