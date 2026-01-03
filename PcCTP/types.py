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




