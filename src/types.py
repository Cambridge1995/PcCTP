"""
PcCTP - CTP API Python 绑定
数据结构类型定义 (TypedDict) - 类型定义
该文件定义了所有 CTP API 相关的数据结构类型

方案3说明（Nanobind + Python C API 融合）：
- Python 层不需要继承任何基类
- PyMdSpi 和 MdApi 类定义在 TYPE_CHECKING 块中，仅用于类型提示
- 实际的 MdApi 类由 C++ 模块提供

优化说明：
- 所有数据结构使用 dict 以及 numpy.ndarray数组 进行零拷贝传递
- 使用 TypedDict 可以获得更好的类型提示支持
- 使用枚举常量代替魔法数字
"""
from typing import TypedDict, Required, NotRequired
from .enums import *

# =============================================================================
# 数据结构类型定义 (TypedDict - 用于类型提示和运行时导入)
# =============================================================================

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
    """客户端认证响应

    CTP 系统返回的客户端认证响应信息。
    """
    broker_id: str  # 经纪公司代码
    user_id: str  # 用户代码
    user_product_info: str  # 用户端产品信息
    app_id: str  # App代码
    app_type: str  # App类型

class UserPasswordUpdate(TypedDict):
    """用户口令更新请求

    用于更新用户登录密码。
    """
    broker_id: str  # 经纪公司代码
    user_id: str  # 用户代码
    old_password: str  # 原来的口令
    new_password: str  # 新的口令

class TradingAccountPasswordUpdate(TypedDict):
    """资金账户口令更新请求

    用于更新资金账户密码。
    """
    broker_id: str  # 经纪公司代码
    account_id: str  # 投资者账号
    old_password: str  # 原来的口令
    new_password: str  # 新的口令
    currency_id: str  # 币种代码

class InputOrder(TypedDict):
    """报单录入请求

    用于向CTP系统录入报单。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码

    # 报单属性
    order_price_type: OrderPriceType  # 报单价格条件 (OrderPriceType枚举)
    direction: Direction  # 买卖方向 (Direction枚举)
    comb_offset_flag: str  # 组合开平标志
    comb_hedge_flag: str  # 组合投机套保标志
    limit_price: float  # 价格
    volume_total_original: int  # 数量

    # 有效期条件
    time_condition: TimeCondition  # 有效期类型 (TimeCondition枚举)
    gtd_date: str  # GTD日期
    volume_condition: VolumeCondition  # 成交量类型 (VolumeCondition枚举)
    min_volume: int  # 最小成交量

    # 触发条件
    contingent_condition: ContingentCondition  # 触发条件 (ContingentCondition枚举)
    stop_price: float  # 止损价
    force_close_reason: ForceCloseReason  # 强平原因 (ForceCloseReason枚举)
    is_auto_suspend: int  # 自动挂起标志

    # 其他字段
    business_unit: str  # 业务单元
    user_force_close: int  # 用户强平标志
    is_swap_order: int  # 互换单标志
    invest_unit_id: str  # 投资单元代码
    account_id: str  # 资金账号
    currency_id: str  # 币种代码
    client_id: str  # 交易编码
    ip_address: str  # IP地址
    mac_address: str  # Mac地址
    order_memo: str  # 报单回显字段

class InvestorPosition(TypedDict):
    """投资者持仓

    CTP 系统中的投资者持仓信息。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    instrument_id: str  # 合约代码

    # 持仓属性
    posi_direction: PositionDirection  # 持仓多空方向 (PositionDirection枚举)
    hedge_flag: HedgeFlag  # 投机套保标志 (HedgeFlag枚举)
    position_date: PositionDate  # 持仓日期 (PositionDate枚举)

    # 数量字段
    yd_position: int  # 上日持仓
    position: int  # 今日持仓
    long_frozen: int  # 多头冻结
    short_frozen: int  # 空头冻结
    long_frozen_amount: float  # 开仓冻结金额
    short_frozen_amount: float  # 开仓冻结金额
    open_volume: int  # 开仓量
    close_volume: int  # 平仓量

    # 金额字段
    open_amount: float  # 开仓金额
    close_amount: float  # 平仓金额
    position_cost: float  # 持仓成本
    pre_margin: float  # 上次占用的保证金
    use_margin: float  # 占用的保证金
    frozen_margin: float  # 冻结的保证金
    frozen_cash: float  # 冻结的资金
    frozen_commission: float  # 冻结的手续费
    cash_in: float  # 资金差额
    commission: float  # 手续费

    # 盈亏字段
    close_profit: float  # 平仓盈亏
    position_profit: float  # 持仓盈亏

    # 其他字段
    settlement_id: int  # 结算编号
    trading_day: str  # 交易日
    invest_unit_id: str  # 投资单元代码
    exchange_id: str  # 交易所代码

class Order(TypedDict):
    """报单

    CTP 系统中的报单信息，包含报单的所有详细字段。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码

    # 报单属性
    order_price_type: OrderPriceType  # 报单价格条件 (OrderPriceType枚举)
    direction: Direction  # 买卖方向 (Direction枚举)
    comb_offset_flag: str  # 组合开平标志
    comb_hedge_flag: str  # 组合投机套保标志
    limit_price: float  # 价格
    volume_total_original: int  # 数量

    # 有效期条件
    time_condition: TimeCondition  # 有效期类型 (TimeCondition枚举)
    gtd_date: str  # GTD日期
    volume_condition: VolumeCondition  # 成交量类型 (VolumeCondition枚举)
    min_volume: int  # 最小成交量

    # 触发条件
    contingent_condition: ContingentCondition  # 触发条件 (ContingentCondition枚举)
    stop_price: float  # 止损价
    force_close_reason: ForceCloseReason  # 强平原因 (ForceCloseReason枚举)
    is_auto_suspend: int  # 自动挂起标志

    # 状态字段
    order_status: OrderStatus  # 报单状态 (OrderStatus枚举)
    order_submit_status: OrderSubmitStatus  # 报单提交状态 (OrderSubmitStatus枚举)
    volume_traded: int  # 今成交数量
    volume_total: int  # 剩余数量

    # 标识字段
    front_id: int  # 前置编号
    session_id: int  # 会话编号
    order_sys_id: str  # 报单编号
    order_local_id: str  # 本地报单编号
    participant_id: str  # 会员代码
    client_id: str  # 客户代码
    trader_id: str  # 交易所交易员代码
    trading_day: str  # 交易日
    settlement_id: int  # 结算编号
    order_source: OrderSource  # 报单来源 (OrderSource枚举)
    order_type: OrderType  # 报单类型 (OrderType枚举)

    # 时间字段
    insert_date: str  # 报单日期
    insert_time: str  # 委托时间
    active_time: str  # 激活时间
    suspend_time: str  # 挂起时间
    update_time: str  # 最后修改时间
    cancel_time: str  # 撤销时间
    active_trader_id: str  # 最后修改交易所交易员代码

    # 其他字段
    business_unit: str  # 业务单元
    user_force_close: int  # 用户强平标志
    is_swap_order: int  # 互换单标志
    clearing_part_id: str  # 结算会员编号
    sequence_no: int  # 序号
    broker_order_seq: int  # 经纪公司报单编号
    status_msg: str  # 状态信息
    branch_id: str  # 营业部编号
    invest_unit_id: str  # 投资单元代码
    account_id: str  # 资金账号
    currency_id: str  # 币种代码
    ip_address: str  # IP地址
    mac_address: str  # Mac地址
    order_memo: str  # 报单回显字段


class Trade(TypedDict):
    """成交

    CTP 系统中的成交信息。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码

    # 成交属性
    direction: Direction  # 买卖方向 (Direction枚举)
    offset_flag: OffsetFlag  # 开平标志 (OffsetFlag枚举)
    hedge_flag: HedgeFlag  # 投机套保标志 (HedgeFlag枚举)
    price: float  # 价格
    volume: int  # 数量

    # 标识字段
    trade_id: str  # 成交编号
    order_sys_id: str  # 报单编号
    order_local_id: str  # 本地报单编号
    participant_id: str  # 会员代码
    client_id: str  # 客户代码
    trader_id: str  # 交易所交易员代码
    trading_role: TradingRole  # 交易角色 (TradingRole枚举)
    trading_day: str  # 交易日
    settlement_id: int  # 结算编号
    trade_type: TradeType  # 成交类型 (TradeType枚举)
    price_source: PriceSource  # 成交价来源 (PriceSource枚举)
    trade_source: TradeSource  # 成交来源 (TradeSource枚举)

    # 时间字段
    trade_date: int  # 成交时期
    trade_time: str  # 成交时间

    # 其他字段
    business_unit: str  # 业务单元
    clearing_part_id: str  # 结算会员编号
    sequence_no: int  # 序号
    broker_order_seq: int  # 经纪公司报单编号
    branch_id: str  # 营业部编号
    invest_unit_id: str  # 投资单元代码
    account_id: str  # 资金账号
    currency_id: str  # 币种代码
    ip_address: str  # IP地址
    mac_address: str  # Mac地址

class InputOrderAction(TypedDict):
    """报单操作请求

    用于撤单或其他报单操作。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_action_ref: str  # 报单操作引用
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码

    # 操作参数
    front_id: int  # 前置编号
    session_id: int  # 会话编号
    order_sys_id: str  # 报单编号
    action_flag: ActionFlag  # 操作标志 (ActionFlag枚举)
    limit_price: float  # 价格
    volume_change: int  # 数量变化

    # 其他字段
    invest_unit_id: str  # 投资单元代码
    ip_address: str  # IP地址
    mac_address: str  # Mac地址
    order_memo: str  # 报单回显字段

class OrderAction(TypedDict):
    """报单操作

    CTP 系统中的报单操作信息。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_action_ref: str  # 报单操作引用
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码

    # 操作参数
    front_id: int  # 前置编号
    session_id: int  # 会话编号
    order_sys_id: str  # 报单编号
    action_flag: ActionFlag  # 操作标志 (ActionFlag枚举)
    limit_price: float  # 价格
    volume_change: int  # 数量变化

    # 时间字段
    action_date: str  # 操作日期
    action_time: str  # 操作时间

    # 其他字段
    trader_id: str  # 交易所交易员代码
    install_id: int  # 安装编号
    order_action_status: str  # 报单操作状态 (OrderActionStatus枚举)
    user_id: str  # 用户代码
    branch_id: str  # 营业部编号
    invest_unit_id: str  # 投资单元代码
    ip_address: str  # IP地址
    mac_address: str  # Mac地址

class ParkedOrder(TypedDict):
    """预埋单

    CTP 系统中的预埋单信息。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码

    # 报单属性
    order_price_type: OrderPriceType  # 报单价格条件 (OrderPriceType枚举)
    direction: Direction  # 买卖方向 (Direction枚举)
    comb_offset_flag: str  # 组合开平标志
    comb_hedge_flag: str  # 组合投机套保标志
    limit_price: float  # 价格
    volume_total_original: int  # 数量

    # 有效期条件
    time_condition: TimeCondition  # 有效期类型 (TimeCondition枚举)
    gtd_date: str  # GTD日期
    volume_condition: VolumeCondition  # 成交量类型 (VolumeCondition枚举)
    min_volume: int  # 最小成交量

    # 触发条件
    contingent_condition: ContingentCondition  # 触发条件 (ContingentCondition枚举)
    stop_price: float  # 止损价
    force_close_reason: ForceCloseReason  # 强平原因 (ForceCloseReason枚举)
    is_auto_suspend: int  # 自动挂起标志

    # 其他字段
    business_unit: str  # 业务单元
    user_force_close: int  # 用户强平标志
    parked_order_id: str  # 预埋报单编号
    user_type: str  # 用户类型
    status: ParkedOrderStatus  # 预埋撤单状态 (ParkedOrderStatus枚举)
    error_id: int  # 错误代码
    error_msg: str  # 错误信息
    invest_unit_id: str  # 投资单元代码
    account_id: str  # 资金账号
    currency_id: str  # 币种代码
    client_id: str  # 交易编码
    ip_address: str  # IP地址
    mac_address: str  # Mac地址

class ParkedOrderAction(TypedDict):
    """预埋撤单

    CTP 系统中的预埋撤单信息。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    order_action_ref: str  # 报单操作引用
    order_ref: str  # 报单引用
    user_id: str  # 用户代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码

    # 操作参数
    front_id: int  # 前置编号
    session_id: int  # 会话编号
    order_sys_id: str  # 报单编号
    action_flag: ActionFlag  # 操作标志 (ActionFlag枚举)
    limit_price: float  # 价格
    volume_change: int  # 数量变化

    # 预埋撤单字段
    parked_order_action_id: str  # 预埋撤单单编号
    user_type: str  # 用户类型
    status: ParkedOrderStatus  # 预埋撤单状态 (ParkedOrderStatus枚举)
    error_id: int  # 错误代码
    error_msg: str  # 错误信息

    # 其他字段
    invest_unit_id: str  # 投资单元代码
    branch_id: str  # 营业部编号
    ip_address: str  # IP地址
    mac_address: str  # Mac地址


class TradingAccount(TypedDict):
    """资金账户

    CTP 系统中的资金账户信息。
    """
    # 基础字段
    broker_id: str  # 经纪公司代码
    account_id: str  # 投资者账号
    trading_day: str  # 交易日
    settlement_id: int  # 结算编号

    # 上日资金
    pre_mortgage: float  # 上次质押金额
    pre_credit: float  # 上次信用额度
    pre_deposit: float  # 上次存款额
    pre_balance: float  # 上次结算准备金
    pre_margin: float  # 上次占用的保证金

    # 当日资金
    deposit: float  # 入金金额
    withdraw: float  # 出金金额
    frozen_margin: float  # 冻结的保证金
    frozen_cash: float  # 冻结的资金
    frozen_commission: float  # 冻结的手续费
    curr_margin: float  # 当前保证金总额
    cash_in: float  # 资金差额
    commission: float  # 手续费

    # 盈亏
    close_profit: float  # 平仓盈亏
    position_profit: float  # 持仓盈亏

    # 资金余额
    balance: float  # 期货结算准备金
    available: float  # 可用资金
    withdraw_quota: float  # 可取资金
    reserve: float  # 基本准备金

    # 其他资金
    credit: float  # 信用额度
    mortgage: float  # 质押金额
    exchange_margin: float  # 交易所保证金
    delivery_margin: float  # 投资者交割保证金
    exchange_delivery_margin: float  # 交易所交割保证金

    # 利息
    interest_base: float  # 利息基数
    interest: float  # 利息收入

    # 其他字段
    currency_id: str  # 币种代码

class Investor(TypedDict):
    """投资者

    CTP 系统中的投资者信息。
    """
    investor_id: str  # 投资者代码
    broker_id: str  # 经纪公司代码
    investor_group_id: str  # 投资者分组代码
    investor_name: str  # 投资者名称
    identified_card_type: IdCardType  # 证件类型 (IdCardType枚举)
    identified_card_no: str  # 证件号码
    is_active: int  # 是否活跃

class TradingCode(TypedDict):
    """交易编码

    CTP 系统中的交易编码信息。
    """
    investor_id: str  # 投资者代码
    broker_id: str  # 经纪公司代码
    exchange_id: str  # 交易所代码
    client_id: str  # 客户代码
    is_active: int  # 是否活跃
    client_id_type: ClientIDType  # 交易编码类型 (ClientIDType枚举)
    branch_id: str  # 营业部编号
    biz_type: str  # 业务类型 (BusinessType枚举)
    invest_unit_id: str  # 投资单元代码

class InstrumentMarginRate(TypedDict):
    """合约保证金率

    CTP 系统中的合约保证金率信息。
    """
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码
    invest_unit_id: str  # 投资单元代码

    # 保证金率
    investor_range: InvestorRange  # 投资者范围 (InvestorRange枚举)
    hedge_flag: HedgeFlag  # 投机套保标志 (HedgeFlag枚举)
    long_margin_ratio_by_money: float  # 多头保证金率
    long_margin_ratio_by_volume: float  # 多头保证金费
    short_margin_ratio_by_money: float  # 空头保证金率
    short_margin_ratio_by_volume: float  # 空头保证金费
    is_relative: int  # 是否相对交易所收取

class InstrumentCommissionRate(TypedDict):
    """合约手续费率

    CTP 系统中的合约手续费率信息。
    """
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码
    invest_unit_id: str  # 投资单元代码

    # 手续费率
    investor_range: InvestorRange  # 投资者范围 (InvestorRange枚举)
    open_ratio_by_money: float  # 开仓手续费率
    open_ratio_by_volume: float  # 开仓手续费
    close_ratio_by_money: float  # 平仓手续费率
    close_ratio_by_volume: float  # 平仓手续费
    close_today_ratio_by_money: float  # 平今手续费率
    close_today_ratio_by_volume: float  # 平今手续费
    biz_type: str  # 业务类型 (BusinessType枚举)

class Instrument(TypedDict):
    """合约

    CTP 系统中的合约信息。
    """
    # 基础字段
    instrument_id: str  # 合约代码
    exchange_id: str  # 交易所代码
    instrument_name: str  # 合约名称
    product_class: ProductClass  # 产品类型 (ProductClass枚举)

    # 交割信息
    delivery_year: int  # 交割年份
    delivery_month: int  # 交割月
    create_date: str  # 创建日
    open_date: str  # 上市日
    expire_date: str  # 到期日
    start_deliv_date: str  # 开始交割日
    end_deliv_date: str  # 结束交割日

    # 合约属性
    volume_multiple: int  # 合约数量乘数
    price_tick: float  # 最小变动价位

    # 下单量限制
    max_market_order_volume: int  # 市价单最大下单量
    min_market_order_volume: int  # 市价单最小下单量
    max_limit_order_volume: int  # 限价单最大下单量
    min_limit_order_volume: int  # 限价单最小下单量

    # 生命周期状态
    inst_life_phase: InstLifePhase  # 合约生命周期状态 (InstLifePhase枚举)
    is_trading: int  # 当前是否交易
    position_type: PositionType  # 持仓类型 (PositionType枚举)
    position_date_type: PositionDateType  # 持仓日期类型 (PositionDateType枚举)

    # 保证金率
    long_margin_ratio: float  # 多头保证金率
    short_margin_ratio: float  # 空头保证金率
    max_margin_side_algorithm: str  # 是否使用大额单边保证金算法 (MaxMarginSideAlgorithmType枚举)

    # 期权相关
    strike_price: float  # 执行价
    options_type: OptionsType  # 期权类型 (OptionsType枚举)
    underlying_instr_id: str  # 基础合约代码
    strike_mode: StrikeMode  # 执行方式 (StrikeMode枚举)

class SettlementInfo(TypedDict):
    """投资者结算结果

    CTP 系统中的投资者结算结果信息。
    """
    trading_day: str  # 交易日
    settlement_id: int  # 结算编号
    broker_id: str  # 经纪公司代码
    investor_id: str  # 投资者代码
    sequence_no: int  # 序号
    content: str  # 消息正文
    account_id: str  # 投资者账号
    currency_id: str  # 币种代码

class InstrumentStatusField(TypedDict):
    """合约交易状态

    CTP 系统中的合约交易状态信息。
    """
    exchange_id: str  # 交易所代码
    settlement_group_id: str  # 结算组代码
    instrument_status: InstrumentStatus  # 合约交易状态 (InstrumentStatus枚举)
    trading_segment_sn: int  # 交易阶段编号
    enter_time: str  # 进入本状态时间
    enter_reason: InstStatusEnterReason  # 进入本状态原因 (InstStatusEnterReason枚举)
    exchange_inst_id: str  # 合约在交易所的代码
    instrument_id: str  # 合约代码


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
    product_class: ProductClass  # 产品类型 (ProductClass枚举)
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
    hedge_flag: HedgeFlag  # 投机套保标志 (HedgeFlag枚举)
    direction: Direction  # 买卖 (Direction枚举)
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
    investor_range: InvestorRange  # 投资者范围 (InvestorRange枚举)
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
    order_price_type: OrderPriceType  # 报单价格条件 (OrderPriceType枚举)
    direction: Direction  # 买卖方向 (Direction枚举)
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
    offset_flag: OffsetFlag  # 枚举: OffsetFlag
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    action_type: ActionType  # 枚举: ActionType
    posi_direction: PositionDirection  # 枚举: PosiDirection
    reserve_position_flag: ExecOrderPositionFlag  # 枚举: ExecOrderPositionFlag
    close_flag: ExecOrderCloseFlag  # 枚举: ExecOrderCloseFlag
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
    action_flag: ActionFlag  # 枚举: ActionFlag
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
    offset_flag: OffsetFlag  # 枚举: OffsetFlag
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    action_type: ActionType  # 枚举: ActionType
    posi_direction: PositionDirection  # 枚举: PosiDirection
    reserve_position_flag: ExecOrderPositionFlag  # 枚举: ExecOrderPositionFlag
    close_flag: ExecOrderCloseFlag  # 枚举: ExecOrderCloseFlag
    exec_order_local_id: str
    exchange_id: str
    trading_day: str
    settlement_id: int
    exec_order_sys_id: str
    insert_date: str
    insert_time: str
    cancel_time: str
    exec_result: ExecResult  # 枚举: ExecResult
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
    ask_offset_flag: OffsetFlag  # 枚举: OffsetFlag
    bid_offset_flag: OffsetFlag  # 枚举: OffsetFlag
    ask_hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    bid_hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    ask_order_ref: str
    bid_order_ref: str
    for_quote_sys_id: str
    exchange_id: str
    invest_unit_id: str
    client_id: str
    instrument_id: str
    replace_sys_id: str
    time_condition: TimeCondition  # 枚举: TimeCondition
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
    action_flag: ActionFlag  # 枚举: ActionFlag
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
    ask_offset_flag: OffsetFlag  # 枚举: OffsetFlag
    bid_offset_flag: OffsetFlag  # 枚举: OffsetFlag
    ask_hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    bid_hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    ask_order_ref: str
    bid_order_ref: str
    for_quote_sys_id: str
    exchange_id: str
    quote_local_id: str
    quote_sys_id: str
    trading_day: str
    settlement_id: int
    quote_status: ForQuoteStatus  # 枚举: ForQuoteStatusType
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
    for_quote_status: ForQuoteStatus  # 枚举: ForQuoteStatus
    front_id: int
    session_id: int
    status_msg: str
    active_user_id: str
    instrument_id: str


class QryMaxOrderVolume(TypedDict):
    """查询最大报单数量"""
    broker_id: str
    investor_id: str
    direction: Direction  # 枚举: Direction
    offset_flag: OffsetFlag  # 枚举: OffsetFlag
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
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
    order_action_status: OrderActionStatus  # 枚举: OrderActionStatus
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
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
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
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
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
    margin_price_type: MarginPriceType  # 枚举: MarginPriceType
    algorithm: Algorithm  # 枚举: Algorithm
    avail_include_close_profit: IncludeCloseProfit  # 枚举: IncludeCloseProfit
    currency_id: str
    option_royalty_price_type: OptionRoyaltyPriceType  # 枚举: OptionRoyaltyPriceType
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
    hedge_flag: HedgeFlag   # 枚举: HedgeFlag
    opt_self_close_flag: OptSelfCloseFlag  # 枚举: OptSelfCloseFlag
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
    action_flag: ActionFlag  # 枚举: ActionFlag
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
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    opt_self_close_flag: OptSelfCloseFlag  # 枚举: OptSelfCloseFlag
    option_self_close_local_id: str
    exchange_id: str
    trading_day: str
    settlement_id: int
    option_self_close_sys_id: str
    insert_date: str
    insert_time: str
    cancel_time: str
    exec_result: ExecResult  # 枚举: ExecResult
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
    direction: Direction  # 枚举: Direction
    volume: int
    comb_direction: CombDirection  # 枚举: CombDirection
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    exchange_id: str
    invest_unit_id: str
    instrument_id: str


class CombAction(TypedDict):
    """申请组合"""
    broker_id: str
    investor_id: str
    comb_action_ref: str
    user_id: str
    direction: Direction  # 枚举: Direction
    volume: int
    comb_direction: CombDirection  # 枚举: CombDirection
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
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
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    direction: Direction  # 枚举: Direction
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
    direction: Direction  # 枚举: Direction
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
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
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
    long_margin_ratio_by_money: float
    long_margin_ratio_by_volume: float
    short_margin_ratio_by_money: float
    short_margin_ratio_by_volume: float
    exchange_id: str
    instrument_id: str


class ExchangeMarginRateAdjust(TypedDict):
    """交易所保证金率调整"""
    broker_id: str
    hedge_flag: HedgeFlag  # 枚举: HedgeFlag
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
