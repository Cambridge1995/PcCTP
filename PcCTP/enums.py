"""
CTP API 枚举常量定义

本模块定义了CTP API中使用的所有枚举常量，提供类型安全的常量访问。
使用枚举可以避免使用魔法数字（如'0', '1'）导致的交易事故。

共包含 319 个枚举类型，自动生成自 docs/CTP常量定义.md
"""

from enum import StrEnum, IntEnum

# 错误码映射
reason_map = {
    0x1001: "网络读失败",
    0x1002: "网络写失败",
    0x2001: "接收心跳超时",
    0x2002: "发送心跳失败",
    0x2003: "收到错误报文",
}
# =============================================================================
# 交易相关核心枚举 - 必须正确使用，否则会导致交易事故！
# =============================================================================

class Direction(StrEnum):
    """买卖方向

    ⚠️ 警告：错误使用会导致严重的交易事故！

    Attributes:
        Buy: 买入
        Sell: 卖出

    Example:
        # 买入开仓
        api.req_order_insert({
             "direction": Direction.Buy,
             "offset_flag": OffsetFlag.Sell,
             ...
        })
    """
    Buy = "0"  # 买
    Sell = "1"  # 卖


class OffsetFlag(StrEnum):
    """开平标志

    ⚠️ 警告：这是期货交易中最关键的参数，错误使用会导致严重事故！

    * OPEN: 新开仓位
    * CLOSE: 平掉历史仓位
    * CLOSE_TODAY: 平掉今日仓位（部分交易所支持）

    事故场景：
    - 本想平仓，错选为OPEN → 反向开仓，双倍持仓
    - 本想开仓，错选为CLOSE → 平掉已有仓位，可能亏损
    - 本想平仓，错选CLOSE_TODAY → 可能只平部分仓位

    Attributes:
        Open: 开仓
        Close: 平仓
        ForceClose: 强平
        CloseToday: 平今
        CloseYesterday: 平昨
        ForceOff: 强减
        LocalForceClose: 本地强平

    Example:
        # 开多仓
        api.req_order_insert({
             "direction": Direction.Buy,
             "offset_flag": OffsetFlag.OPEN,  # 开仓
             ...
        })

        # 平仓
        api.req_order_insert({
            "direction": Direction.Sell,
            "offset_flag": OffsetFlag.CLOSE,  # 平仓
            ...
        })
    """
    Open = "0"  # 开仓
    Close = "1"  # 平仓
    ForceClose = "2"  # 强平
    CloseToday = "3"  # 平今
    CloseYesterday = "4"  # 平昨
    ForceOff = "5"  # 强减
    LocalForceClose = "6"  # 本地强平


class HedgeFlag(StrEnum):
    """投机套保标志类型

    Attributes:
        Speculation: 投机
        Arbitrage: 套利
        Hedge: 套保
        MarketMaker: 做市商
        SpecHedge: 第一腿投机第二腿套保
        HedgeSpec: 第一腿套保第二腿投机
    """

    Speculation = "1"  # 投机
    Arbitrage = "2"  # 套利
    Hedge = "3"  # 套保
    MarketMaker = "5"  # 做市商
    SpecHedge = "6"  # 第一腿投机第二腿套保
    HedgeSpec = "7"  # 第一腿套保第二腿投机


class OrderPriceType(StrEnum):
    """报单价格条件类型

    Attributes:
        AnyPrice: 任意价
        LimitPrice: 限价
        BestPrice: 最优价
        LastPrice: 最新价
        LastPricePlusOneTicks: 最新价浮动上浮1个ticks
        LastPricePlusTwoTicks: 最新价浮动上浮2个ticks
        LastPricePlusThreeTicks: 最新价浮动上浮3个ticks
        AskPrice1: 卖一价
        AskPrice1PlusOneTicks: 卖一价浮动上浮1个ticks
        AskPrice1PlusTwoTicks: 卖一价浮动上浮2个ticks
        AskPrice1PlusThreeTicks: 卖一价浮动上浮3个ticks
        BidPrice1: 买一价
        BidPrice1PlusOneTicks: 买一价浮动上浮1个ticks
        BidPrice1PlusTwoTicks: 买一价浮动上浮2个ticks
        BidPrice1PlusThreeTicks: 买一价浮动上浮3个ticks
        FiveLevelPrice: 五档价
    """

    AnyPrice = "1"  # 任意价
    LimitPrice = "2"  # 限价
    BestPrice = "3"  # 最优价
    LastPrice = "4"  # 最新价
    LastPricePlusOneTicks = "5"  # 最新价浮动上浮1个ticks
    LastPricePlusTwoTicks = "6"  # 最新价浮动上浮2个ticks
    LastPricePlusThreeTicks = "7"  # 最新价浮动上浮3个ticks
    AskPrice1 = "8"  # 卖一价
    AskPrice1PlusOneTicks = "9"  # 卖一价浮动上浮1个ticks
    AskPrice1PlusTwoTicks = "A"  # 卖一价浮动上浮2个ticks
    AskPrice1PlusThreeTicks = "B"  # 卖一价浮动上浮3个ticks
    BidPrice1 = "C"  # 买一价
    BidPrice1PlusOneTicks = "D"  # 买一价浮动上浮1个ticks
    BidPrice1PlusTwoTicks = "E"  # 买一价浮动上浮2个ticks
    BidPrice1PlusThreeTicks = "F"  # 买一价浮动上浮3个ticks
    FiveLevelPrice = "G"  # 五档价


class TimeCondition(StrEnum):
    """有效期类型类型

    Attributes:
        IOC: 立即完成，否则撤销
        GFS: 本节有效
        GFD: 当日有效
        GTD: 指定日期前有效
        GTC: 撤销前有效
        GFA: 集合竞价有效
    """

    IOC = "1"  # 立即完成，否则撤销
    GFS = "2"  # 本节有效
    GFD = "3"  # 当日有效
    GTD = "4"  # 指定日期前有效
    GTC = "5"  # 撤销前有效
    GFA = "6"  # 集合竞价有效


class VolumeCondition(StrEnum):
    """成交量类型类型

    Attributes:
        AV: 任何数量
        MV: 最小数量
        CV: 全部数量
    """

    AV = "1"  # 任何数量
    MV = "2"  # 最小数量
    CV = "3"  # 全部数量


class ContingentCondition(StrEnum):
    """触发条件类型

    Attributes:
        Immediately: 立即
        Touch: 止损
        TouchProfit: 止赢
        ParkedOrder: 预埋单
        LastPriceGreaterThanStopPrice: 最新价大于条件价
        LastPriceGreaterEqualStopPrice: 最新价大于等于条件价
        LastPriceLesserThanStopPrice: 最新价小于条件价
        LastPriceLesserEqualStopPrice: 最新价小于等于条件价
        AskPriceGreaterThanStopPrice: 卖一价大于条件价
        AskPriceGreaterEqualStopPrice: 卖一价大于等于条件价
        AskPriceLesserThanStopPrice: 卖一价小于条件价
        AskPriceLesserEqualStopPrice: 卖一价小于等于条件价
        BidPriceGreaterThanStopPrice: 买一价大于条件价
        BidPriceGreaterEqualStopPrice: 买一价大于等于条件价
        BidPriceLesserThanStopPrice: 买一价小于条件价
        BidPriceLesserEqualStopPrice: 买一价小于等于条件价
    """

    Immediately = "1"  # 立即
    Touch = "2"  # 止损
    TouchProfit = "3"  # 止赢
    ParkedOrder = "4"  # 预埋单
    LastPriceGreaterThanStopPrice = "5"  # 最新价大于条件价
    LastPriceGreaterEqualStopPrice = "6"  # 最新价大于等于条件价
    LastPriceLesserThanStopPrice = "7"  # 最新价小于条件价
    LastPriceLesserEqualStopPrice = "8"  # 最新价小于等于条件价
    AskPriceGreaterThanStopPrice = "9"  # 卖一价大于条件价
    AskPriceGreaterEqualStopPrice = "A"  # 卖一价大于等于条件价
    AskPriceLesserThanStopPrice = "B"  # 卖一价小于条件价
    AskPriceLesserEqualStopPrice = "C"  # 卖一价小于等于条件价
    BidPriceGreaterThanStopPrice = "D"  # 买一价大于条件价
    BidPriceGreaterEqualStopPrice = "E"  # 买一价大于等于条件价
    BidPriceLesserThanStopPrice = "F"  # 买一价小于条件价
    BidPriceLesserEqualStopPrice = "H"  # 买一价小于等于条件价


class OrderStatus(StrEnum):
    """报单状态类型

    Attributes:
        AllTraded: 全部成交
        PartTradedQueueing: 部分成交还在队列中
        PartTradedNotQueueing: 部分成交不在队列中
        NoTradeQueueing: 未成交还在队列中
        NoTradeNotQueueing: 未成交不在队列中
        Canceled: 撤单
        Unknown: 未知
        NotTouched: 尚未触发
        Touched: 已触发
    """

    AllTraded = "0"  # 全部成交
    PartTradedQueueing = "1"  # 部分成交还在队列中
    PartTradedNotQueueing = "2"  # 部分成交不在队列中
    NoTradeQueueing = "3"  # 未成交还在队列中
    NoTradeNotQueueing = "4"  # 未成交不在队列中
    Canceled = "5"  # 撤单
    Unknown = "a"  # 未知
    NotTouched = "b"  # 尚未触发
    Touched = "c"  # 已触发


class OrderSubmitStatus(StrEnum):
    """报单提交状态类型

    Attributes:
        InsertSubmitted: 已经提交
        CancelSubmitted: 撤单已经提交
        ModifySubmitted: 修改已经提交
        Accepted: 已经接受
        InsertRejected: 报单已经被拒绝
        CancelRejected: 撤单已经被拒绝
        ModifyRejected: 改单已经被拒绝
    """

    InsertSubmitted = "0"  # 已经提交
    CancelSubmitted = "1"  # 撤单已经提交
    ModifySubmitted = "2"  # 修改已经提交
    Accepted = "3"  # 已经接受
    InsertRejected = "4"  # 报单已经被拒绝
    CancelRejected = "5"  # 撤单已经被拒绝
    ModifyRejected = "6"  # 改单已经被拒绝


class OrderActionStatus(StrEnum):
    """报单操作状态类型

    Attributes:
        Submitted: 已经提交
        Accepted: 已经接受
        Rejected: 已经被拒绝
    """

    Submitted = "a"  # 已经提交
    Accepted = "b"  # 已经接受
    Rejected = "c"  # 已经被拒绝


class ForceCloseReason(StrEnum):
    """强平原因类型

    Attributes:
        NotForceClose: 非强平
        LackDeposit: 资金不足
        ClientOverPositionLimit: 客户超仓
        MemberOverPositionLimit: 会员超仓
        NotMultiple: 持仓非整数倍
        Violation: 违规
        Other: 其它
        PersonDeliv: 自然人临近交割
        Notverifycapital: 本地强平资金不足忽略敞口
        LocalLackDeposit: 本地强平资金不足
        LocalViolationNocheck: 本地强平违规持仓忽略敞口
        LocalViolation: 本地强平违规持仓
    """

    NotForceClose = "0"  # 非强平
    LackDeposit = "1"  # 资金不足
    ClientOverPositionLimit = "2"  # 客户超仓
    MemberOverPositionLimit = "3"  # 会员超仓
    NotMultiple = "4"  # 持仓非整数倍
    Violation = "5"  # 违规
    Other = "6"  # 其它
    PersonDeliv = "7"  # 自然人临近交割
    Notverifycapital = "8"  # 本地强平资金不足忽略敞口
    LocalLackDeposit = "9"  # 本地强平资金不足
    LocalViolationNocheck = "a"  # 本地强平违规持仓忽略敞口
    LocalViolation = "b"  # 本地强平违规持仓


# =============================================================================
# 期权相关枚举
# =============================================================================


class OptionsType(StrEnum):
    """期权类型类型

    Attributes:
        CallOptions: 看涨
        PutOptions: 看跌
    """

    CallOptions = "1"  # 看涨
    PutOptions = "2"  # 看跌


class StrikeMode(StrEnum):
    """执行方式类型

    Attributes:
        Continental: 欧式
        American: 美式
        Bermuda: 百慕大
    """

    Continental = "0"  # 欧式
    American = "1"  # 美式
    Bermuda = "2"  # 百慕大


# =============================================================================
# 市场相关枚举
# =============================================================================
class ProductClass(StrEnum):
    """产品类型类型

    Attributes:
        Futures: 期货
        Options: 期货期权
        Combination: 组合
        Spot: 即期
        EFP: 期转现
        SpotOption: 现货期权
        TAS: TAS合约
        MI: 金属指数
    """

    Futures = "1"  # 期货
    Options = "2"  # 期货期权
    Combination = "3"  # 组合
    Spot = "4"  # 即期
    EFP = "5"  # 期转现
    SpotOption = "6"  # 现货期权
    TAS = "7"  # TAS合约
    MI = "I"  # 金属指数


class PositionDate(StrEnum):
    """持仓日期类型

    Attributes:
        Today: 今日持仓
        History: 历史持仓
    """

    Today = "1"  # 今日持仓
    History = "2"  # 历史持仓


class PositionDateType(StrEnum):
    """持仓日期类型类型

    Attributes:
        UseHistory: 使用历史持仓
        NoUseHistory: 不使用历史持仓
    """

    UseHistory = "1"  # 使用历史持仓
    NoUseHistory = "2"  # 不使用历史持仓


class TradingRole(StrEnum):
    """交易角色类型

    Attributes:
        Broker: 代理
        Host: 自营
        Maker: 做市商
    """

    Broker = "1"  # 代理
    Host = "2"  # 自营
    Maker = "3"  # 做市商


class InstrumentStatus(StrEnum):
    """合约交易状态类型

    Attributes:
        BeforeTrading: 开盘前
        NoTrading: 非交易
        Continuous: 连续交易
        AuctionOrdering: 集合竞价报单
        AuctionBalance: 集合竞价价格平衡
        AuctionMatch: 集合竞价撮合
        Closed: 收盘
        TransactionProcessing: 交易业务处理
    """

    BeforeTrading = "0"  # 开盘前
    NoTrading = "1"  # 非交易
    Continuous = "2"  # 连续交易
    AuctionOrdering = "3"  # 集合竞价报单
    AuctionBalance = "4"  # 集合竞价价格平衡
    AuctionMatch = "5"  # 集合竞价撮合
    Closed = "6"  # 收盘
    TransactionProcessing = "7"  # 交易业务处理


class InstLifePhase(StrEnum):
    """合约生命周期状态类型

    Attributes:
        NotStart: 未上市
        Started: 上市
        Pause: 停牌
        Expired: 到期
    """

    NotStart = "0"  # 未上市
    Started = "1"  # 上市
    Pause = "2"  # 停牌
    Expired = "3"  # 到期

class PositionDirection(StrEnum):
    """持仓多空方向类型

    Attributes:
        Net: 净
        Long: 多头
        Short: 空头
    """

    Net = "1"  # 净
    Long = "2"  # 多头
    Short = "3"  # 空头


# =============================================================================
# 系统相关枚举
# =============================================================================
class SettlementStatus(StrEnum):
    """结算状态类型

    Attributes:
        Initialize: 初始
        Settling: 结算中
        Settled: 已结算
        Finished: 结算完成
    """

    Initialize = "0"  # 初始
    Settling = "1"  # 结算中
    Settled = "2"  # 已结算
    Finished = "3"  # 结算完成

class SysSettlementStatus(StrEnum):
    """系统结算状态类型

    Attributes:
        NonActive: 不活跃
        Startup: 启动
        Operating: 操作
        Settlement: 结算
        SettlementFinished: 结算完成
    """

    NonActive = "1"  # 不活跃
    Startup = "2"  # 启动
    Operating = "3"  # 操作
    Settlement = "4"  # 结算
    SettlementFinished = "5"  # 结算完成


# =============================================================================
# 其他枚举
# =============================================================================
class PositionType(StrEnum):
    """持仓类型类型

    Attributes:
        Net: 净持仓
        Gross: 综合持仓
    """

    Net = "1"  # 净持仓
    Gross = "2"  # 综合持仓


class RatioAttr(StrEnum):
    """费率属性类型

    Attributes:
        Trade: 交易费率
        Settlement: 结算费率
    """

    Trade = "0"  # 交易费率
    Settlement = "1"  # 结算费率


class BillHedgeFlag(StrEnum):
    """投机套保标志类型

    Attributes:
        Speculation: 投机
        Arbitrage: 套利
        Hedge: 套保
    """

    Speculation = "1"  # 投机
    Arbitrage = "2"  # 套利
    Hedge = "3"  # 套保


class ClientIDType(StrEnum):
    """交易编码类型类型

    Attributes:
        Speculation: 投机
        Arbitrage: 套利
        Hedge: 套保
        MarketMaker: 做市商
    """

    Speculation = "1"  # 投机
    Arbitrage = "2"  # 套利
    Hedge = "3"  # 套保
    MarketMaker = "5"  # 做市商


class OrderType(StrEnum):
    """报单类型类型

    Attributes:
        Normal: 正常
        DeriveFromQuote: 报价衍生
        DeriveFromCombination: 组合衍生
        Combination: 组合报单
        ConditionalOrder: 条件单
        Swap: 互换单
        DeriveFromBlockTrade: 大宗交易成交衍生
        DeriveFromEFPTrade: 期转现成交衍生
    """

    Normal = "0"  # 正常
    DeriveFromQuote = "1"  # 报价衍生
    DeriveFromCombination = "2"  # 组合衍生
    Combination = "3"  # 组合报单
    ConditionalOrder = "4"  # 条件单
    Swap = "5"  # 互换单
    DeriveFromBlockTrade = "6"  # 大宗交易成交衍生
    DeriveFromEFPTrade = "7"  # 期转现成交衍生


class ActionFlag(StrEnum):
    """操作标志类型

    Attributes:
        Delete: 删除
        Modify: 修改
    """

    Delete = "0"  # 删除
    Modify = "3"  # 修改


class TradingRight(StrEnum):
    """交易权限类型

    Attributes:
        Allow: 可以交易
        CloseOnly: 只能平仓
        Forbidden: 不能交易
    """

    Allow = "0"  # 可以交易
    CloseOnly = "1"  # 只能平仓
    Forbidden = "2"  # 不能交易


class APIProductClass(StrEnum):
    """产品类型类型

    Attributes:
        FutureSingle: 期货单一合约
        OptionSingle: 期权单一合约
        Futures: 可交易期货(含期货组合和期货单一合约)
        Options: 可交易期权(含期权组合和期权单一合约)
        TradingComb: 可下单套利组合
        UnTradingComb: 可申请的组合（可以申请的组合合约 包含可以交易的合约）
        AllTrading: 所有可以交易合约
        All: 所有合约（包含不能交易合约 慎用）
    """

    FutureSingle = "1"  # 期货单一合约
    OptionSingle = "2"  # 期权单一合约
    Futures = "3"  # 可交易期货(含期货组合和期货单一合约)
    Options = "4"  # 可交易期权(含期权组合和期权单一合约)
    TradingComb = "5"  # 可下单套利组合
    UnTradingComb = "6"  # 可申请的组合（可以申请的组合合约 包含可以交易的合约）
    AllTrading = "7"  # 所有可以交易合约
    All = "8"  # 所有合约（包含不能交易合约 慎用）


class IdCardType(StrEnum):
    """证件类型

    Attributes:
        EID: 个人身份证
        IDCard: 身份证
        OfficerIDCard :  军官证
        PoliceIDCard :   警官证
        SoldierIDCard :  士兵证
        HouseholdRegister :  户口本
        Passport :  护照
        TaiwanCompatriotIDCard : 台胞证
        HomeComingCard :  港澳同胞回乡证
        LicenseNo :  营业执照号
        TaxNo :  税务登记号
        HMMainlandTravelPermit : 港澳居民来往内地通行证
        TwMainlandTravelPermit :  台湾居民来往大陆通行证
        DrivingLicense :  机动车驾驶证
        SocialID :  当地社保ID(社会保障号)
        LocalID :  当地身份证
        BusinessRegistration :  营业注册号
        HKMCIDCard :  港澳永久性居民身份证
        AccountsPermits :  人行开户许可证
        FrgPrmtRdCard :  外国人永久居留证
        CptMngPrdLetter :  船员证
        HKMCTwResidencePermit : 港澳台居民居留签证
        UniformSocialCreditCode : 统一社会信用代码
        CorporationCertNo :  机构成立证明文件
        OtherCard :  其他证件
    """
    EID = "0"  # 组织机构代码
    IDCard = "1"  # 中国公民身份证
    OfficerIDCard = "2"  # 军官证
    PoliceIDCard = "3"  # 警官证
    SoldierIDCard = "4"  # 士兵证
    HouseholdRegister = "5"  # 户口本
    Passport = "6"  # 护照
    TaiwanCompatriotIDCard = "7"  # 台胞证
    HomeComingCard = "8"  # 港澳同胞回乡证
    LicenseNo = "9"  # 营业执照号
    TaxNo = "A"  # 税务登记号
    HMMainlandTravelPermit = "B"  # 港澳居民来往内地通行证
    TwMainlandTravelPermit = "C"  # 台湾居民来往大陆通行证
    DrivingLicense = "D"  # 机动车驾驶证
    SocialID = "F"  # 当地社保ID(社会保障号)
    LocalID = "G"  # 当地身份证
    BusinessRegistration = "H"  # 营业注册号
    HKMCIDCard = "I"  # 港澳永久性居民身份证
    AccountsPermits = "J"  # 人行开户许可证
    FrgPrmtRdCard = "K"  # 外国人永久居留证
    CptMngPrdLetter = "L"  # 资管产品备案函(船员证)
    HKMCTwResidencePermit = "M"  # 港澳台居民居住证
    UniformSocialCreditCode = "N"  # 统一社会信用代码
    CorporationCertNo = "O"  # 机构成立证明文件
    OtherCard = "x"  # 其他证件


class InvestorRange(StrEnum):
    """投资者范围类型

    Attributes:
        All: 所有
        Group: 投资者组
        Single: 单一投资者
    """

    All = "1"  # 所有
    Group = "2"  # 投资者组
    Single = "3"  # 单一投资者

# =============================================================================
# 组合相关枚举
# =============================================================================
class CombinationType(StrEnum):
    """组合类型类型

    Attributes:
        Future: 期货组合
        BUL: 垂直价差BUL
        BER: 垂直价差BER
        STD: 跨式组合
        STG: 宽跨式组合
        PRT: 备兑组合
        CAS: 时间价差组合
        OPL: 期权对锁组合
        BFO: 买备兑组合
        BLS: 买入期权垂直价差组合
        BES: 卖出期权垂直价差组合
    """

    Future = "0"  # 期货组合
    BUL = "1"  # 垂直价差BUL
    BER = "2"  # 垂直价差BER
    STD = "3"  # 跨式组合
    STG = "4"  # 宽跨式组合
    PRT = "5"  # 备兑组合
    CAS = "6"  # 时间价差组合
    OPL = "7"  # 期权对锁组合
    BFO = "8"  # 买备兑组合
    BLS = "9"  # 买入期权垂直价差组合
    BES = "a"  # 卖出期权垂直价差组合


class DceCombinationType(StrEnum):
    """组合类型类型

    Attributes:
        SPL: 期货对锁组合
        OPL: 期权对锁组合
        SP: 期货跨期组合
        SPC: 期货跨品种组合
        BLS: 买入期权垂直价差组合
        BES: 卖出期权垂直价差组合
        CAS: 期权日历价差组合
        STD: 期权跨式组合
        STG: 期权宽跨式组合
        BFO: 买入期货期权组合
        SFO: 卖出期货期权组合
    """

    SPL = "0"  # 期货对锁组合
    OPL = "1"  # 期权对锁组合
    SP = "2"  # 期货跨期组合
    SPC = "3"  # 期货跨品种组合
    BLS = "4"  # 买入期权垂直价差组合
    BES = "5"  # 卖出期权垂直价差组合
    CAS = "6"  # 期权日历价差组合
    STD = "7"  # 期权跨式组合
    STG = "8"  # 期权宽跨式组合
    BFO = "9"  # 买入期货期权组合
    SFO = "a"  # 卖出期货期权组合

# =============================================================================
# 交易所相关枚举
# =============================================================================
class ExchangeIDType(StrEnum):
    """交易所编号类型

    Attributes:
        SHFE: 上海期货交易所(上期所)
        CZCE: 郑州商品交易所(郑商所)
        DCE: 大连商品交易所(大商所)
        CFFEX: 中国金融期货交易所(中金所)
        INE: 上海国际能源交易中心股份有限公司(能源中心)
    """

    SHFE = "S"  # 上海期货交易所(上期所)
    CZCE = "Z"  # 郑州商品交易所(郑商所)
    DCE = "D"  # 大连商品交易所(大商所)
    CFFEX = "J"  # 中国金融期货交易所(中金所)
    INE = "N"  # 上海国际能源交易中心股份有限公司(能源中心)


# =============================================================================
# 其他系统枚举
# =============================================================================


class TradingType(StrEnum):
    """合约交易状态分类方式类型

    Attributes:
        ALL: 所有状态
        TRADE: 交易
        UNTRADE: 非交易
    """

    ALL = "0"  # 所有状态
    TRADE = "1"  # 交易
    UNTRADE = "2"  # 非交易

class OffsetType(StrEnum):
    """对冲类型

    Attributes:
        OPT_OFFSET: 期权对冲
        FUT_OFFSET: 期货对冲
        EXEC_OFFSET: 行权后期货对冲
        PERFORM_OFFSET: 履约后期货对冲
    """
    OPT_OFFSET = "0"  # 期权对冲
    FUT_OFFSET = "1"  # 期货对冲
    EXEC_OFFSET = "2"  # 行权后期货对冲
    PERFORM_OFFSET = "3"  # 履约后期货对冲



class TradeSource(StrEnum):
    """成交来源类型

    Attributes:
        NORMAL: 来自交易所普通回报
        QUERY: 来自查询
    """

    NORMAL = "0"  # 来自交易所普通回报
    QUERY = "1"  # 来自查询


class ExportFileType(StrEnum):
    """导出文件类型

    Attributes:
        CSV: CSV文件
        EXCEL: Excel文件
        DBF: DBF文件
    """
    CSV = "0"  # CSV文件
    EXCEL = "1"  # Excel文件
    DBF = "2"  # DBF文件



class MarketMakeState(StrEnum):
    """做市状态

    Attributes:
        NO: 否
        YES: 是
    """
    NO = "0"  # 否
    YES = "1"  # 是

class ExecResult(StrEnum):
    """执行结果类型

    Attributes:
        NoExec: 没有执行
        Canceled: 已经取消
        OK: 执行成功
        NoPosition: 期权持仓不够
        NoDeposit: 资金不够
        NoParticipant: 会员不存在
        NoClient: 客户不存在
        NoInstrument: 合约不存在
        NoRight: 没有执行权限
        InvalidVolume: 不合理的数量
        NoEnoughHistoryTrade: 没有足够的历史成交
        Unknown: 未知
    """

    NoExec = "n"  # 没有执行
    Canceled = "c"  # 已经取消
    OK = "0"  # 执行成功
    NoPosition = "1"  # 期权持仓不够
    NoDeposit = "2"  # 资金不够
    NoParticipant = "3"  # 会员不存在
    NoClient = "4"  # 客户不存在
    NoInstrument = "6"  # 合约不存在
    NoRight = "7"  # 没有执行权限
    InvalidVolume = "8"  # 不合理的数量
    NoEnoughHistoryTrade = "9"  # 没有足够的历史成交
    Unknown = "a"  # 未知



class StrikeType(StrEnum):
    """执行类型类型

    Attributes:
        Hedge: 自身对冲
        Match: 匹配执行
    """

    Hedge = "0"  # 自身对冲
    Match = "1"  # 匹配执行


# =============================================================================
# 辅助函数
# =============================================================================

def validate_direction(value: str) -> bool:
    """验证买卖方向"""
    return value in Direction.__members__.values()


def validate_offset_flag(value: str) -> bool:
    """验证开平标志"""
    return value in OffsetFlag.__members__.values()


def validate_order_price_type(value: str) -> bool:
    """验证报单价格类型"""
    return value in OrderPriceType.__members__.values()


def get_direction_name(value: Direction) -> str:
    """获取买卖方向中文名"""
    names = {
        Direction.Buy: "买入",
        Direction.Sell: "卖出"
    }
    return names.get(value, "未知")


def get_offset_flag_name(value: OffsetFlag) -> str:
    """获取开平标志中文名"""
    names = {
        OffsetFlag.Open: "开仓",
        OffsetFlag.Close: "平仓",
        OffsetFlag.ForceClose: "强平",
        OffsetFlag.CloseToday: "平今",
        OffsetFlag.CloseYesterday: "平昨"
    }
    return names.get(value, "未知")


class ExchangeProperty(StrEnum):
    """交易所属性类型

    Attributes:
        Normal: 正常
        GenOrderByTrade: 根据成交生成报单
    """

    Normal = "0"  # 正常
    GenOrderByTrade = "1"  # 根据成交生成报单


class DepartmentRange(StrEnum):
    """投资者范围类型

    Attributes:
        All: 所有
        Group: 组织架构
        Single: 单一投资者
    """

    All = "1"  # 所有
    Group = "2"  # 组织架构
    Single = "3"  # 单一投资者


class DataSyncStatus(StrEnum):
    """数据同步状态类型

    Attributes:
        Asynchronous: 未同步
        Synchronizing: 同步中
        Synchronized: 已同步
    """

    Asynchronous = "1"  # 未同步
    Synchronizing = "2"  # 同步中
    Synchronized = "3"  # 已同步


class BrokerDataSyncStatus(StrEnum):
    """经纪公司数据同步状态类型

    Attributes:
        Synchronized: 已同步
        Synchronizing: 同步中
    """

    Synchronized = "1"  # 已同步
    Synchronizing = "2"  # 同步中


class ExchangeConnectStatus(StrEnum):
    """交易所连接状态类型

    Attributes:
        NoConnection: 没有任何连接
        QryInstrumentSent: 已经发出合约查询请求
        GotInformation: 已经获取信息
    """

    NoConnection = "1"  # 没有任何连接
    QryInstrumentSent = "2"  # 已经发出合约查询请求
    GotInformation = "9"  # 已经获取信息


class TraderConnectStatus(StrEnum):
    """交易所交易员连接状态类型

    Attributes:
        NotConnected: 没有任何连接
        Connected: 已经连接
        QryInstrumentSent: 已经发出合约查询请求
        SubPrivateFlow: 订阅私有流
    """

    NotConnected = "1"  # 没有任何连接
    Connected = "2"  # 已经连接
    QryInstrumentSent = "3"  # 已经发出合约查询请求
    SubPrivateFlow = "4"  # 订阅私有流


class FunctionCode(StrEnum):
    """功能代码类型

    Attributes:
        DataAsync: 数据异步化
        ForceUserLogout: 强制用户登出
        UserPasswordUpdate: 变更管理用户口令
        BrokerPasswordUpdate: 变更经纪公司口令
        InvestorPasswordUpdate: 变更投资者口令
        OrderInsert: 报单插入
        OrderAction: 报单操作
        SyncSystemData: 同步系统数据
        SyncBrokerData: 同步经纪公司数据
        BachSyncBrokerData: 批量同步经纪公司数据
        SuperQuery: 超级查询
        ParkedOrderInsert: 预埋报单插入
        ParkedOrderAction: 预埋报单操作
        SyncOTP: 同步动态令牌
        DeleteOrder: 删除未知单
        ExitEmergency: 退出紧急状态
    """

    DataAsync = "1"  # 数据异步化
    ForceUserLogout = "2"  # 强制用户登出
    UserPasswordUpdate = "3"  # 变更管理用户口令
    BrokerPasswordUpdate = "4"  # 变更经纪公司口令
    InvestorPasswordUpdate = "5"  # 变更投资者口令
    OrderInsert = "6"  # 报单插入
    OrderAction = "7"  # 报单操作
    SyncSystemData = "8"  # 同步系统数据
    SyncBrokerData = "9"  # 同步经纪公司数据
    BachSyncBrokerData = "A"  # 批量同步经纪公司数据
    SuperQuery = "B"  # 超级查询
    ParkedOrderInsert = "C"  # 预埋报单插入
    ParkedOrderAction = "D"  # 预埋报单操作
    SyncOTP = "E"  # 同步动态令牌
    DeleteOrder = "F"  # 删除未知单
    ExitEmergency = "G"  # 退出紧急状态


class BrokerFunctionCode(StrEnum):
    """经纪公司功能代码类型

    Attributes:
        ForceUserLogout: 强制用户登出
        UserPasswordUpdate: 变更用户口令
        SyncBrokerData: 同步经纪公司数据
        BachSyncBrokerData: 批量同步经纪公司数据
        OrderInsert: 报单插入
        OrderAction: 报单操作
        AllQuery: 全部查询
        log: 系统功能：登入
        BaseQry: 基本查询：查询基础数据，如合约，交易所等常量
        TradeQry: 交易查询：如查成交，委托
        Trade: 交易功能：报单，撤单
        Virement: 银期转账
        Risk: 风险监控
        Session: 查询
        RiskNoticeCtl: 风控通知控制
        RiskNotice: 风控通知发送
        BrokerDeposit: 察看经纪公司资金权限
        QueryFund: 资金查询
        QueryOrder: 报单查询
        QueryTrade: 成交查询
        QueryPosition: 持仓查询
        QueryMarketData: 行情查询
        QueryUserEvent: 用户事件查询
        QueryRiskNotify: 风险通知查询
        QueryFundChange: 出入金查询
        QueryInvestor: 投资者信息查询
        QueryTradingCode: 交易编码查询
        ForceClose: 强平
        PressTest: 压力测试
        RemainCalc: 权益反算
        NetPositionInd: 净持仓保证金指标
        RiskPredict: 风险预算
        DataExport: 数据导出
        RiskTargetSetup: 风控指标设置
        MarketDataWarn: 行情预警
        QryBizNotice: 业务通知查询
        CfgBizNotice: 业务通知模板设置
        SyncOTP: 同步动态令牌
        SendBizNotice: 发送业务通知
        CfgRiskLevelStd: 风险级别标准设置
        TbCommand: 交易终端应急功能
        DeleteOrder: 删除未知单
        ParkedOrderInsert: 预埋报单插入
        ParkedOrderAction: 预埋报单操作
        ExecOrderNoCheck: 资金不够仍允许行权
        Designate: 指定
        StockDisposal: 证券处置
        BrokerDepositWarn: 席位资金预警
        CoverWarn: 备兑不足预警
        PreExecOrder: 行权试算
        ExecOrderRisk: 行权交收风险
        PosiLimitWarn: 持仓限额预警
        QryPosiLimit: 持仓限额查询
        FBSign: 银期签到签退
        FBAccount: 银期签约解约
    """

    ForceUserLogout = "1"  # 强制用户登出
    UserPasswordUpdate = "2"  # 变更用户口令
    SyncBrokerData = "3"  # 同步经纪公司数据
    BachSyncBrokerData = "4"  # 批量同步经纪公司数据
    OrderInsert = "5"  # 报单插入
    OrderAction = "6"  # 报单操作
    AllQuery = "7"  # 全部查询
    log = "a"  # 系统功能：登入
    BaseQry = "b"  # 基本查询：查询基础数据，如合约，交易所等常量
    TradeQry = "c"  # 交易查询：如查成交，委托
    Trade = "d"  # 交易功能：报单，撤单
    Virement = "e"  # 银期转账
    Risk = "f"  # 风险监控
    Session = "g"  # 查询
    RiskNoticeCtl = "h"  # 风控通知控制
    RiskNotice = "i"  # 风控通知发送
    BrokerDeposit = "j"  # 察看经纪公司资金权限
    QueryFund = "k"  # 资金查询
    QueryOrder = "l"  # 报单查询
    QueryTrade = "m"  # 成交查询
    QueryPosition = "n"  # 持仓查询
    QueryMarketData = "o"  # 行情查询
    QueryUserEvent = "p"  # 用户事件查询
    QueryRiskNotify = "q"  # 风险通知查询
    QueryFundChange = "r"  # 出入金查询
    QueryInvestor = "s"  # 投资者信息查询
    QueryTradingCode = "t"  # 交易编码查询
    ForceClose = "u"  # 强平
    PressTest = "v"  # 压力测试
    RemainCalc = "w"  # 权益反算
    NetPositionInd = "x"  # 净持仓保证金指标
    RiskPredict = "y"  # 风险预算
    DataExport = "z"  # 数据导出
    RiskTargetSetup = "A"  # 风控指标设置
    MarketDataWarn = "B"  # 行情预警
    QryBizNotice = "C"  # 业务通知查询
    CfgBizNotice = "D"  # 业务通知模板设置
    SyncOTP = "E"  # 同步动态令牌
    SendBizNotice = "F"  # 发送业务通知
    CfgRiskLevelStd = "G"  # 风险级别标准设置
    TbCommand = "H"  # 交易终端应急功能
    DeleteOrder = "J"  # 删除未知单
    ParkedOrderInsert = "K"  # 预埋报单插入
    ParkedOrderAction = "L"  # 预埋报单操作
    ExecOrderNoCheck = "M"  # 资金不够仍允许行权
    Designate = "N"  # 指定
    StockDisposal = "O"  # 证券处置
    BrokerDepositWarn = "Q"  # 席位资金预警
    CoverWarn = "S"  # 备兑不足预警
    PreExecOrder = "T"  # 行权试算
    ExecOrderRisk = "P"  # 行权交收风险
    PosiLimitWarn = "U"  # 持仓限额预警
    QryPosiLimit = "V"  # 持仓限额查询
    FBSign = "W"  # 银期签到签退
    FBAccount = "X"  # 银期签约解约








class OrderSource(StrEnum):
    """报单来源类型

    Attributes:
        Participant: 来自参与者
        Administrator: 来自管理员
    """

    Participant = "0"  # 来自参与者
    Administrator = "1"  # 来自管理员


class TradeType(StrEnum):
    """成交类型类型

    Attributes:
        SplitCombination: 组合持仓拆分为单一持仓,初始化不应包含该类型的持仓
        Common: 普通成交
        OptionsExecution: 期权执行
        OTC: OTC成交
        EFPDerived: 期转现衍生成交
        CombinationDerived: 组合衍生成交
        BlockTrade: 大宗交易成交
    """

    SplitCombination = "#"  # 组合持仓拆分为单一持仓,初始化不应包含该类型的持仓
    Common = "0"  # 普通成交
    OptionsExecution = "1"  # 期权执行
    OTC = "2"  # OTC成交
    EFPDerived = "3"  # 期转现衍生成交
    CombinationDerived = "4"  # 组合衍生成交
    BlockTrade = "5"  # 大宗交易成交


class SpecPosiType(StrEnum):
    """特殊持仓明细标识类型

    Attributes:
        Common: 普通持仓明细
        Tas: TAS合约成交产生的标的合约持仓明细
    """

    Common = "#"  # 普通持仓明细
    Tas = "0"  # TAS合约成交产生的标的合约持仓明细


class PriceSource(StrEnum):
    """成交价来源类型

    Attributes:
        LastPrice: 前成交价
        Buy: 买委托价
        Sell: 卖委托价
        OTC: 场外成交价
    """

    LastPrice = "0"  # 前成交价
    Buy = "1"  # 买委托价
    Sell = "2"  # 卖委托价
    OTC = "3"  # 场外成交价


class InstStatusEnterReason(StrEnum):
    """品种进入交易状态原因类型

    Attributes:
        Automatic: 自动切换
        Manual: 手动切换
        Fuse: 熔断
    """

    Automatic = "1"  # 自动切换
    Manual = "2"  # 手动切换
    Fuse = "3"  # 熔断


class BatchStatus(StrEnum):
    """处理状态类型

    Attributes:
        NoUpload: 未上传
        Uploaded: 已上传
        Failed: 审核失败
    """

    NoUpload = "1"  # 未上传
    Uploaded = "2"  # 已上传
    Failed = "3"  # 审核失败


class ReturnStyle(StrEnum):
    """按品种返还方式类型

    Attributes:
        All: 按所有品种
        ByProduct: 按品种
    """

    All = "1"  # 按所有品种
    ByProduct = "2"  # 按品种


class ReturnPattern(StrEnum):
    """返还模式类型

    Attributes:
        ByVolume: 按成交手数
        ByFeeOnHand: 按留存手续费
    """

    ByVolume = "1"  # 按成交手数
    ByFeeOnHand = "2"  # 按留存手续费


class ReturnLevel(StrEnum):
    """返还级别类型

    Attributes:
        Level1: 级别1
        Level2: 级别2
        Level3: 级别3
        Level4: 级别4
        Level5: 级别5
        Level6: 级别6
        Level7: 级别7
        Level8: 级别8
        Level9: 级别9
    """

    Level1 = "1"  # 级别1
    Level2 = "2"  # 级别2
    Level3 = "3"  # 级别3
    Level4 = "4"  # 级别4
    Level5 = "5"  # 级别5
    Level6 = "6"  # 级别6
    Level7 = "7"  # 级别7
    Level8 = "8"  # 级别8
    Level9 = "9"  # 级别9


class ReturnStandard(StrEnum):
    """返还标准类型

    Attributes:
        ByPeriod: 分阶段返还
        ByStandard: 按某一标准
    """

    ByPeriod = "1"  # 分阶段返还
    ByStandard = "2"  # 按某一标准


class MortgageType(StrEnum):
    """质押类型类型

    Attributes:
        Out: 质出
        In: 质入
    """

    Out = "0"  # 质出
    In = "1"  # 质入


class InvestorSettlementParamID(StrEnum):
    """投资者结算参数代码类型

    Attributes:
        MortgageRatio: 质押比例
        MarginWay: 保证金算法
        BillDeposit: 结算单结存是否包含质押
    """

    MortgageRatio = "4"  # 质押比例
    MarginWay = "5"  # 保证金算法
    BillDeposit = "9"  # 结算单结存是否包含质押


class ExchangeSettlementParamID(StrEnum):
    """交易所结算参数代码类型

    Attributes:
        MortgageRatio: 质押比例
        OtherFundItem: 分项资金导入项
        OtherFundImport: 分项资金入交易所出入金
        CFFEXMinPrepare: 中金所开户最低可用金额
        CZCESettlementType: 郑商所结算方式
        ExchDelivFeeMode: 交易所交割手续费收取方式
        DelivFeeMode: 投资者交割手续费收取方式
        CZCEComMarginType: 郑商所组合持仓保证金收取方式
        DceComMarginType: 大商所套利保证金是否优惠
        OptOutDisCountRate: 虚值期权保证金优惠比率
        OptMiniGuarantee: 最低保障系数
    """

    MortgageRatio = "1"  # 质押比例
    OtherFundItem = "2"  # 分项资金导入项
    OtherFundImport = "3"  # 分项资金入交易所出入金
    CFFEXMinPrepare = "6"  # 中金所开户最低可用金额
    CZCESettlementType = "7"  # 郑商所结算方式
    ExchDelivFeeMode = "9"  # 交易所交割手续费收取方式
    DelivFeeMode = "0"  # 投资者交割手续费收取方式
    CZCEComMarginType = "A"  # 郑商所组合持仓保证金收取方式
    DceComMarginType = "B"  # 大商所套利保证金是否优惠
    OptOutDisCountRate = "a"  # 虚值期权保证金优惠比率
    OptMiniGuarantee = "b"  # 最低保障系数


class SystemParamID(StrEnum):
    """系统参数代码类型

    Attributes:
        InvestorIDMinLength: 投资者代码最小长度
        AccountIDMinLength: 投资者帐号代码最小长度
        UserRightLogon: 投资者开户默认登录权限
        SettlementBillTrade: 投资者交易结算单成交汇总方式
        TradingCode: 统一开户更新交易编码方式
        CheckFund: 结算是否判断存在未复核的出入金和分项资金
        CommModelRight: 是否启用手续费模板数据权限
        MarginModelRight: 是否启用保证金率模板数据权限
        IsStandardActive: 是否规范用户才能激活
        UploadSettlementFile: 上传的交易所结算文件路径
        DownloadCSRCFile: 上报保证金监控中心文件路径
        SettlementBillFile: 生成的结算单文件路径
        CSRCOthersFile: 证监会文件标识
        InvestorPhoto: 投资者照片路径
        CSRCData: 全结经纪公司上传文件路径
        InvestorPwdModel: 开户密码录入方式
        CFFEXInvestorSettleFile: 投资者中金所结算文件下载路径
        InvestorIDType: 投资者代码编码方式
        FreezeMaxReMain: 休眠户最高权益
        IsSync: 手续费相关操作实时上场开关
        RelieveOpenLimit: 解除开仓权限限制
        IsStandardFreeze: 是否规范用户才能休眠
        CZCENormalProductHedge: 郑商所是否开放所有品种套保交易
    """

    InvestorIDMinLength = "1"  # 投资者代码最小长度
    AccountIDMinLength = "2"  # 投资者账号代码最小长度
    UserRightLogon = "3"  # 投资者开户默认登录权限
    SettlementBillTrade = "4"  # 投资者交易结算单成交汇总方式
    TradingCode = "5"  # 统一开户更新交易编码方式
    CheckFund = "6"  # 结算是否判断存在未复核的出入金和分项资金
    CommModelRight = "7"  # 是否启用手续费模板数据权限
    MarginModelRight = "9"  # 是否启用保证金率模板数据权限
    IsStandardActive = "8"  # 是否规范用户才能激活
    UploadSettlementFile = "U"  # 上传的交易所结算文件路径
    DownloadCSRCFile = "D"  # 上报保证金监控中心文件路径
    SettlementBillFile = "S"  # 生成的结算单文件路径
    CSRCOthersFile = "C"  # 证监会文件标识
    InvestorPhoto = "P"  # 投资者照片路径
    CSRCData = "R"  # 全结经纪公司上传文件路径
    InvestorPwdModel = "I"  # 开户密码录入方式
    CFFEXInvestorSettleFile = "F"  # 投资者中金所结算文件下载路径
    InvestorIDType = "a"  # 投资者代码编码方式
    FreezeMaxReMain = "r"  # 休眠户最高权益
    IsSync = "A"  # 手续费相关操作实时上场开关
    RelieveOpenLimit = "O"  # 解除开仓权限限制
    IsStandardFreeze = "X"  # 是否规范用户才能休眠
    CZCENormalProductHedge = "B"  # 郑商所是否开放所有品种套保交易


class TradeParamID(StrEnum):
    """交易系统参数代码类型

    Attributes:
        EncryptionStandard: 系统加密算法
        RiskMode: 系统风险算法
        RiskModeGlobal: 系统风险算法是否全局 0-否 1-是
        modeEncode: 密码加密算法
        tickMode: 价格小数位数参数
        SingleUserSessionMaxNum: 用户最大会话数
        LoginFailMaxNum: 最大连续登录失败数
        IsAuthForce: 是否强制认证
        IsPosiFreeze: 是否冻结证券持仓
        IsPosiLimit: 是否限仓
        ForQuoteTimeInterval: 郑商所询价时间间隔
        IsFuturePosiLimit: 是否期货限仓
        IsFutureOrderFreq: 是否期货下单频率限制
        IsExecOrderProfit: 行权冻结是否计算盈利
        IsCheckBankAcc: 银期开户是否验证开户银行卡号是否是预留银行账户
        PasswordDeadLine: 弱密码最后修改日期
        IsStrongPassword: 强密码校验
        BalanceMortgage: 自有资金质押比
        MinPwdLen: 最小密码长度
        LoginFailMaxNumForIP: IP当日最大登陆失败次数
        PasswordPeriod: 密码有效期
        PwdHistoryCmp: 历史密码重复限制次数
    """

    EncryptionStandard = "E"  # 系统加密算法
    RiskMode = "R"  # 系统风险算法
    RiskModeGlobal = "G"  # 系统风险算法是否全局 0-否 1-是
    modeEncode = "P"  # 密码加密算法
    tickMode = "T"  # 价格小数位数参数
    SingleUserSessionMaxNum = "S"  # 用户最大会话数
    LoginFailMaxNum = "L"  # 最大连续登录失败数
    IsAuthForce = "A"  # 是否强制认证
    IsPosiFreeze = "F"  # 是否冻结证券持仓
    IsPosiLimit = "M"  # 是否限仓
    ForQuoteTimeInterval = "Q"  # 郑商所询价时间间隔
    IsFuturePosiLimit = "B"  # 是否期货限仓
    IsFutureOrderFreq = "C"  # 是否期货下单频率限制
    IsExecOrderProfit = "H"  # 行权冻结是否计算盈利
    IsCheckBankAcc = "I"  # 银期开户是否验证开户银行卡号是否是预留银行账户
    PasswordDeadLine = "J"  # 弱密码最后修改日期
    IsStrongPassword = "K"  # 强密码校验
    BalanceMortgage = "a"  # 自有资金质押比
    MinPwdLen = "O"  # 最小密码长度
    LoginFailMaxNumForIP = "U"  # IP当日最大登陆失败次数
    PasswordPeriod = "V"  # 密码有效期
    PwdHistoryCmp = "X"  # 历史密码重复限制次数


class FileID(StrEnum):
    """文件标识类型

    Attributes:
        SettlementFund: 资金数据
        Trade: 成交数据
        InvestorPosition: 投资者持仓数据
        SubEntryFund: 投资者分项资金数据
        CZCECombinationPos: 组合持仓数据
        CSRCData: 上报保证金监控中心数据
        CZCEClose: 郑商所平仓了结数据
        CZCENoClose: 郑商所非平仓了结数据
        PositionDtl: 持仓明细数据
        OptionStrike: 期权执行文件
        SettlementPriceComparison: 结算价比对文件
        NonTradePosChange: 上期所非持仓变动明细
    """

    SettlementFund = "F"  # 资金数据
    Trade = "T"  # 成交数据
    InvestorPosition = "P"  # 投资者持仓数据
    SubEntryFund = "O"  # 投资者分项资金数据
    CZCECombinationPos = "C"  # 组合持仓数据
    CSRCData = "R"  # 上报保证金监控中心数据
    CZCEClose = "L"  # 郑商所平仓了结数据
    CZCENoClose = "N"  # 郑商所非平仓了结数据
    PositionDtl = "D"  # 持仓明细数据
    OptionStrike = "S"  # 期权执行文件
    SettlementPriceComparison = "M"  # 结算价比对文件
    NonTradePosChange = "B"  # 上期所非持仓变动明细


class FileType(StrEnum):
    """文件上传类型类型

    Attributes:
        Settlement: 结算
        Check: 核对
    """

    Settlement = "0"  # 结算
    Check = "1"  # 核对


class FileFormat(StrEnum):
    """文件格式类型

    Attributes:
        Txt: 文本文件(.txt)
        Zip: 压缩文件(.zip)
        DBF: DBF文件(.dbf)
    """

    Txt = "0"  # 文本文件(.txt)
    Zip = "1"  # 压缩文件(.zip)
    DBF = "2"  # DBF文件(.dbf)


class FileUploadStatus(StrEnum):
    """文件状态类型

    Attributes:
        SucceedUpload: 上传成功
        FailedUpload: 上传失败
        SucceedLoad: 导入成功
        PartSucceedLoad: 导入部分成功
        FailedLoad: 导入失败
    """

    SucceedUpload = "1"  # 上传成功
    FailedUpload = "2"  # 上传失败
    SucceedLoad = "3"  # 导入成功
    PartSucceedLoad = "4"  # 导入部分成功
    FailedLoad = "5"  # 导入失败


class TransferDirection(StrEnum):
    """移仓方向类型

    Attributes:
        Out: 移出
        In: 移入
    """

    Out = "0"  # 移出
    In = "1"  # 移入


class SpecialCreateRule(StrEnum):
    """特殊的创建规则类型

    Attributes:
        NoSpecialRule: 没有特殊创建规则
        NoSpringFestival: 不包含春节
    """

    NoSpecialRule = "0"  # 没有特殊创建规则
    NoSpringFestival = "1"  # 不包含春节


class BasisPriceType(StrEnum):
    """挂牌基准价类型类型

    Attributes:
        LastSettlement: 上一合约结算价
        LaseClose: 上一合约收盘价
    """

    LastSettlement = "1"  # 上一合约结算价
    LaseClose = "2"  # 上一合约收盘价


class ProductLifePhase(StrEnum):
    """产品生命周期状态类型

    Attributes:
        Active: 活跃
        NonActive: 不活跃
        Canceled: 注销
    """

    Active = "1"  # 活跃
    NonActive = "2"  # 不活跃
    Canceled = "3"  # 注销


class DeliveryMode(StrEnum):
    """交割方式类型

    Attributes:
        CashDeliv: 现金交割
        CommodityDeliv: 实物交割
    """

    CashDeliv = "1"  # 现金交割
    CommodityDeliv = "2"  # 实物交割


class FundIOType(StrEnum):
    """出入金类型类型

    Attributes:
        FundIO: 出入金
        Transfer: 银期转帐
        SwapCurrency: 银期换汇
    """

    FundIO = "1"  # 出入金
    Transfer = "2"  # 银期转帐
    SwapCurrency = "3"  # 银期换汇


class FundType(StrEnum):
    """资金类型类型

    Attributes:
        Deposit: 银行存款
        ItemFund: 分项资金
        Company: 公司调整
        InnerTransfer: 资金内转
    """

    Deposit = "1"  # 银行存款
    ItemFund = "2"  # 分项资金
    Company = "3"  # 公司调整
    InnerTransfer = "4"  # 资金内转


class FundDirection(StrEnum):
    """出入金方向类型

    Attributes:
        In: 入金
        Out: 出金
    """

    In = "1"  # 入金
    Out = "2"  # 出金


class FundStatus(StrEnum):
    """资金状态类型

    Attributes:
        Record: 已录入
        Check: 已复核
        Charge: 已冲销
    """

    Record = "1"  # 已录入
    Check = "2"  # 已复核
    Charge = "3"  # 已冲销


class PublishStatus(StrEnum):
    """发布状态类型

    Attributes:
        NONE: 未发布
        Publishing: 正在发布
        Published: 已发布
    """

    NONE = "1"  # 未发布
    Publishing = "2"  # 正在发布
    Published = "3"  # 已发布


class SystemStatus(StrEnum):
    """系统状态类型

    Attributes:
        NonActive: 不活跃
        Startup: 启动
        Initialize: 交易开始初始化
        Initialized: 交易完成初始化
        Close: 收市开始
        Closed: 收市完成
        Settlement: 结算
    """

    NonActive = "1"  # 不活跃
    Startup = "2"  # 启动
    Initialize = "3"  # 交易开始初始化
    Initialized = "4"  # 交易完成初始化
    Close = "5"  # 收市开始
    Closed = "6"  # 收市完成
    Settlement = "7"  # 结算


class InvestorType(StrEnum):
    """投资者类型类型

    Attributes:
        Person: 自然人
        Company: 法人
        Fund: 投资基金
        SpecialOrgan: 特殊法人
        Asset: 资管户
    """

    Person = "0"  # 自然人
    Company = "1"  # 法人
    Fund = "2"  # 投资基金
    SpecialOrgan = "3"  # 特殊法人
    Asset = "4"  # 资管户


class BrokerType(StrEnum):
    """经纪公司类型类型

    Attributes:
        Trade: 交易会员
        TradeSettle: 交易结算会员
    """

    Trade = "0"  # 交易会员
    TradeSettle = "1"  # 交易结算会员


class RiskLevel(StrEnum):
    """风险等级类型

    Attributes:
        Low: 低风险客户
        Normal: 普通客户
        Focus: 关注客户
        Risk: 风险客户
    """

    Low = "1"  # 低风险客户
    Normal = "2"  # 普通客户
    Focus = "3"  # 关注客户
    Risk = "4"  # 风险客户


class FeeAcceptStyle(StrEnum):
    """手续费收取方式类型

    Attributes:
        ByTrade: 按交易收取
        ByDeliv: 按交割收取
        NONE: 不收
        FixFee: 按指定手续费收取
    """

    ByTrade = "1"  # 按交易收取
    ByDeliv = "2"  # 按交割收取
    NONE = "3"  # 不收
    FixFee = "4"  # 按指定手续费收取


class PasswordType(StrEnum):
    """密码类型类型

    Attributes:
        Trade: 交易密码
        Account: 资金密码
    """

    Trade = "1"  # 交易密码
    Account = "2"  # 资金密码


class Algorithm(StrEnum):
    """盈亏算法类型

    Attributes:
        All: 浮盈浮亏都计算
        OnlyLost: 浮盈不计，浮亏计
        OnlyGain: 浮盈计，浮亏不计
        NONE: 浮盈浮亏都不计算
    """

    All = "1"  # 浮盈浮亏都计算
    OnlyLost = "2"  # 浮盈不计，浮亏计
    OnlyGain = "3"  # 浮盈计，浮亏不计
    NONE = "4"  # 浮盈浮亏都不计算


class IncludeCloseProfit(StrEnum):
    """是否包含平仓盈利类型

    Attributes:
        Include: 包含平仓盈利
        NotInclude: 不包含平仓盈利
    """

    Include = "0"  # 包含平仓盈利
    NotInclude = "2"  # 不包含平仓盈利


class AllWithoutTrade(StrEnum):
    """是否受可提比例限制类型

    Attributes:
        Enable: 无仓无成交不受可提比例限制
        Disable: 受可提比例限制
        NoHoldEnable: 无仓不受可提比例限制
    """

    Enable = "0"  # 无仓无成交不受可提比例限制
    Disable = "2"  # 受可提比例限制
    NoHoldEnable = "3"  # 无仓不受可提比例限制


class FuturePwdFlag(StrEnum):
    """资金密码核对标志类型

    Attributes:
        UnCheck: 不核对
        Check: 核对
    """

    UnCheck = "0"  # 不核对
    Check = "1"  # 核对


class TransferType(StrEnum):
    """银期转账类型类型

    Attributes:
        BankToFuture: 银行转期货
        FutureToBank: 期货转银行
    """

    BankToFuture = "0"  # 银行转期货
    FutureToBank = "1"  # 期货转银行


class TransferValidFlag(StrEnum):
    """转账有效标志类型

    Attributes:
        Invalid: 无效或失败
        Valid: 有效
        Reverse: 冲正
    """

    Invalid = "0"  # 无效或失败
    Valid = "1"  # 有效
    Reverse = "2"  # 冲正


class Reason(StrEnum):
    """事由类型

    Attributes:
        CD: 错单
        ZT: 资金在途
        QT: 其它
    """

    CD = "0"  # 错单
    ZT = "1"  # 资金在途
    QT = "2"  # 其它


class Sex(StrEnum):
    """性别类型

    Attributes:
        NONE: 未知
        Man: 男
        Woman: 女
    """

    NONE = "0"  # 未知
    Man = "1"  # 男
    Woman = "2"  # 女


class UserType(StrEnum):
    """用户类型类型

    Attributes:
        Investor: 投资者
        Operator: 操作员
        SuperUser: 管理员
    """

    Investor = "0"  # 投资者
    Operator = "1"  # 操作员
    SuperUser = "2"  # 管理员


class RateType(StrEnum):
    """费率类型类型

    Attributes:
        MarginRate: 保证金率
    """

    MarginRate = "2"  # 保证金率


class NoteType(StrEnum):
    """通知类型类型

    Attributes:
        TradeSettleBill: 交易结算单
        TradeSettleMonth: 交易结算月报
        CallMarginNotes: 追加保证金通知书
        ForceCloseNotes: 强行平仓通知书
        TradeNotes: 成交通知书
        DelivNotes: 交割通知书
    """

    TradeSettleBill = "1"  # 交易结算单
    TradeSettleMonth = "2"  # 交易结算月报
    CallMarginNotes = "3"  # 追加保证金通知书
    ForceCloseNotes = "4"  # 强行平仓通知书
    TradeNotes = "5"  # 成交通知书
    DelivNotes = "6"  # 交割通知书


class SettlementStyle(StrEnum):
    """结算单方式类型

    Attributes:
        Day: 逐日盯市
        Volume: 逐笔对冲
    """

    Day = "1"  # 逐日盯市
    Volume = "2"  # 逐笔对冲


class SettlementBillType(StrEnum):
    """结算单类型类型

    Attributes:
        Day: 日报
        Month: 月报
    """

    Day = "0"  # 日报
    Month = "1"  # 月报


class UserRightType(StrEnum):
    """客户权限类型类型

    Attributes:
        Logon: 登录
        Transfer: 银期转帐
        EMail: 邮寄结算单
        Fax: 传真结算单
        ConditionOrder: 条件单
    """

    Logon = "1"  # 登录
    Transfer = "2"  # 银期转帐
    EMail = "3"  # 邮寄结算单
    Fax = "4"  # 传真结算单
    ConditionOrder = "5"  # 条件单


class MarginPriceType(StrEnum):
    """保证金价格类型类型

    Attributes:
        PreSettlementPrice: 昨结算价
        SettlementPrice: 最新价
        AveragePrice: 成交均价
        OpenPrice: 开仓价
    """

    PreSettlementPrice = "1"  # 昨结算价
    SettlementPrice = "2"  # 最新价
    AveragePrice = "3"  # 成交均价
    OpenPrice = "4"  # 开仓价


class BillGenStatus(StrEnum):
    """结算单生成状态类型

    Attributes:
        NONE: 未生成
        NoGenerated: 生成中
        Generated: 已生成
    """

    NONE = "0"  # 未生成
    NoGenerated = "1"  # 生成中
    Generated = "2"  # 已生成


class AlgoType(StrEnum):
    """算法类型类型

    Attributes:
        HandlePositionAlgo: 持仓处理算法
        FindMarginRateAlgo: 寻找保证金率算法
    """

    HandlePositionAlgo = "1"  # 持仓处理算法
    FindMarginRateAlgo = "2"  # 寻找保证金率算法


class HandlePositionAlgoID(StrEnum):
    """持仓处理算法编号类型

    Attributes:
        Base: 基本
        DCE: 大连商品交易所
        CZCE: 郑州商品交易所
    """

    Base = "1"  # 基本
    DCE = "2"  # 大连商品交易所
    CZCE = "3"  # 郑州商品交易所


class FindMarginRateAlgoID(StrEnum):
    """寻找保证金率算法编号类型

    Attributes:
        Base: 基本
        DCE: 大连商品交易所
        CZCE: 郑州商品交易所
    """

    Base = "1"  # 基本
    DCE = "2"  # 大连商品交易所
    CZCE = "3"  # 郑州商品交易所


class HandleTradingAccountAlgoID(StrEnum):
    """资金处理算法编号类型

    Attributes:
        Base: 基本
        DCE: 大连商品交易所
        CZCE: 郑州商品交易所
    """

    Base = "1"  # 基本
    DCE = "2"  # 大连商品交易所
    CZCE = "3"  # 郑州商品交易所


class PersonType(StrEnum):
    """联系人类型类型

    Attributes:
        Order: 指定下单人
        Open: 开户授权人
        Fund: 资金调拨人
        Settlement: 结算单确认人
        Company: 法人
        Corporation: 法人代表
        LinkMan: 投资者联系人
        Ledger: 分户管理资产负责人
        Trustee: 托（保）管人
        TrusteeCorporation: 托（保）管机构法人代表
        TrusteeOpen: 托（保）管机构开户授权人
        TrusteeContact: 托（保）管机构联系人
        ForeignerRefer: 境外自然人参考证件
        CorporationRefer: 法人代表参考证件
    """

    Order = "1"  # 指定下单人
    Open = "2"  # 开户授权人
    Fund = "3"  # 资金调拨人
    Settlement = "4"  # 结算单确认人
    Company = "5"  # 法人
    Corporation = "6"  # 法人代表
    LinkMan = "7"  # 投资者联系人
    Ledger = "8"  # 分户管理资产负责人
    Trustee = "9"  # 托（保）管人
    TrusteeCorporation = "A"  # 托（保）管机构法人代表
    TrusteeOpen = "B"  # 托（保）管机构开户授权人
    TrusteeContact = "C"  # 托（保）管机构联系人
    ForeignerRefer = "D"  # 境外自然人参考证件
    CorporationRefer = "E"  # 法人代表参考证件


class QueryInvestorRange(StrEnum):
    """查询范围类型

    Attributes:
        All: 所有
        Group: 查询分类
        Single: 单一投资者
    """

    All = "1"  # 所有
    Group = "2"  # 查询分类
    Single = "3"  # 单一投资者


class InvestorRiskStatus(StrEnum):
    """投资者风险状态类型

    Attributes:
        Normal: 正常
        Warn: 警告
        Call: 追保
        Force: 强平
        Exception: 异常
    """

    Normal = "1"  # 正常
    Warn = "2"  # 警告
    Call = "3"  # 追保
    Force = "4"  # 强平
    Exception = "5"  # 异常


class UserEventType(StrEnum):
    """用户事件类型类型

    Attributes:
        Login: 登录
        Logout: 登出
        Trading: CTP校验通过
        TradingError: CTP校验失败
        UpdatePassword: 修改密码
        Authenticate: 客户端认证
        SubmitSysInfo: 终端信息上报
        Transfer: 转账
        Other: 其他
        UpdateTradingAccountPassword: 修改资金密码
    """

    Login = "1"  # 登录
    Logout = "2"  # 登出
    Trading = "3"  # CTP校验通过
    TradingError = "4"  # CTP校验失败
    UpdatePassword = "5"  # 修改密码
    Authenticate = "6"  # 客户端认证
    SubmitSysInfo = "7"  # 终端信息上报
    Transfer = "8"  # 转账
    Other = "9"  # 其他
    UpdateTradingAccountPassword = "a"  # 修改资金密码


class CloseStyle(StrEnum):
    """平仓方式类型

    Attributes:
        Close: 先开先平
        CloseToday: 先平今再平昨
    """

    Close = "0"  # 先开先平
    CloseToday = "1"  # 先平今再平昨


class StatMode(StrEnum):
    """统计方式类型

    Attributes:
        Non: ----
        Instrument: 按合约统计
        Product: 按产品统计
        Investor: 按投资者统计
    """

    Non = "0"  # ----
    Instrument = "1"  # 按合约统计
    Product = "2"  # 按产品统计
    Investor = "3"  # 按投资者统计


class ParkedOrderStatus(StrEnum):
    """预埋单状态类型

    Attributes:
        NotSend: 未发送
        Send: 已发送
        Deleted: 已删除
    """

    NotSend = "1"  # 未发送
    Send = "2"  # 已发送
    Deleted = "3"  # 已删除


class VirDealStatus(StrEnum):
    """处理状态类型

    Attributes:
        Dealing: 正在处理
        DealSucceed: 处理成功
    """

    Dealing = "1"  # 正在处理
    DealSucceed = "2"  # 处理成功


class OrgSystemID(StrEnum):
    """原有系统代码类型

    Attributes:
        Standard: 综合交易平台
        ESunny: 易盛系统
        KingStarV6: 金仕达V6系统
    """

    Standard = "0"  # 综合交易平台
    ESunny = "1"  # 易盛系统
    KingStarV6 = "2"  # 金仕达V6系统


class VirTradeStatus(StrEnum):
    """交易状态类型

    Attributes:
        NaturalDeal: 正常处理中
        SucceedEnd: 成功结束
        FailedEND: 失败结束
        Exception: 异常中
        ManualDeal: 已人工异常处理
        MesException: 通讯异常 ，请人工处理
        SysException: 系统出错，请人工处理
    """

    NaturalDeal = "0"  # 正常处理中
    SucceedEnd = "1"  # 成功结束
    FailedEND = "2"  # 失败结束
    Exception = "3"  # 异常中
    ManualDeal = "4"  # 已人工异常处理
    MesException = "5"  # 通讯异常 ，请人工处理
    SysException = "6"  # 系统出错，请人工处理


class VirBankAccType(StrEnum):
    """银行帐户类型类型

    Attributes:
        BankBook: 存折
        BankCard: 储蓄卡
        CreditCard: 信用卡
    """

    BankBook = "1"  # 存折
    BankCard = "2"  # 储蓄卡
    CreditCard = "3"  # 信用卡


class VirementStatus(StrEnum):
    """银行帐户类型类型

    Attributes:
        Natural: 正常
        Canceled: 销户
    """

    Natural = "0"  # 正常
    Canceled = "9"  # 销户


class VirementAvailAbility(StrEnum):
    """有效标志类型

    Attributes:
        NoAvailAbility: 未确认
        AvailAbility: 有效
        Repeal: 冲正
    """

    NoAvailAbility = "0"  # 未确认
    AvailAbility = "1"  # 有效
    Repeal = "2"  # 冲正


class VirementTradeCode(StrEnum):
    """交易代码类型

    Attributes:
        BankBankToFuture: 银行发起银行资金转期货
        BankFutureToBank: 银行发起期货资金转银行
        FutureBankToFuture: 期货发起银行资金转期货
        FutureFutureToBank: 期货发起期货资金转银行
    """

    BankBankToFuture = "102001"  # 银行发起银行资金转期货
    BankFutureToBank = "102002"  # 银行发起期货资金转银行
    FutureBankToFuture = "202001"  # 期货发起银行资金转期货
    FutureFutureToBank = "202002"  # 期货发起期货资金转银行


class AMLGenStatus(StrEnum):
    """Aml生成方式类型

    Attributes:
        Program: 程序生成
        HandWork: 人工生成
    """

    Program = "0"  # 程序生成
    HandWork = "1"  # 人工生成


class CFMMCKeyKind(StrEnum):
    """动态密钥类别(保证金监管)类型

    Attributes:
        REQUEST: 主动请求更新
        AUTO: CFMMC自动更新
        MANUAL: CFMMC手动更新
    """

    REQUEST = "R"  # 主动请求更新
    AUTO = "A"  # CFMMC自动更新
    MANUAL = "M"  # CFMMC手动更新


class CertificationType(StrEnum):
    """证件类型类型

    Attributes:
        IDCard: 身份证
        Passport: 护照
        OfficerIDCard: 军官证
        SoldierIDCard: 士兵证
        HomeComingCard: 回乡证
        LicenseNo: 营业执照号
        InstitutionCodeCard: 组织机构代码证
        TempLicenseNo: 临时营业执照号
        NoEnterpriseLicenseNo: 民办非企业登记证书
        OtherCard: 其他证件
        SuperDepAgree: 主管部门批文
    """

    IDCard = "0"  # 身份证
    Passport = "1"  # 护照
    OfficerIDCard = "2"  # 军官证
    SoldierIDCard = "3"  # 士兵证
    HomeComingCard = "4"  # 回乡证
    LicenseNo = "6"  # 营业执照号
    InstitutionCodeCard = "7"  # 组织机构代码证
    TempLicenseNo = "8"  # 临时营业执照号
    NoEnterpriseLicenseNo = "9"  # 民办非企业登记证书
    OtherCard = "x"  # 其他证件
    SuperDepAgree = "a"  # 主管部门批文


class FileBusinessCode(StrEnum):
    """文件业务功能类型

    Attributes:
        Others: 其他
        TransferDetails: 转账交易明细对账
        CustAccStatus: 客户账户状态对账
        AccountTradeDetails: 账户类交易明细对账
        FutureAccountChangeInfoDetails: 期货账户信息变更明细对账
        CustMoneyDetail: 客户资金台账余额明细对账
        CustCancelAccountInfo: 客户销户结息明细对账
        CustMoneyResult: 客户资金余额对账结果
        OthersExceptionResult: 其它对账异常结果文件
        CustInterestNetMoneyDetails: 客户结息净额明细
        CustMoneySendAndReceiveDetails: 客户资金交收明细
        CorporationMoneyTotal: 法人存管银行资金交收汇总
        MainbodyMoneyTotal: 主体间资金交收汇总
        MainPartMonitorData: 总分平衡监管数据
        PreparationMoney: 存管银行备付金余额
        BankMoneyMonitorData: 协办存管银行资金监管数据
    """

    Others = "0"  # 其他
    TransferDetails = "1"  # 转账交易明细对账
    CustAccStatus = "2"  # 客户账户状态对账
    AccountTradeDetails = "3"  # 账户类交易明细对账
    FutureAccountChangeInfoDetails = "4"  # 期货账户信息变更明细对账
    CustMoneyDetail = "5"  # 客户资金台账余额明细对账
    CustCancelAccountInfo = "6"  # 客户销户结息明细对账
    CustMoneyResult = "7"  # 客户资金余额对账结果
    OthersExceptionResult = "8"  # 其它对账异常结果文件
    CustInterestNetMoneyDetails = "9"  # 客户结息净额明细
    CustMoneySendAndReceiveDetails = "a"  # 客户资金交收明细
    CorporationMoneyTotal = "b"  # 法人存管银行资金交收汇总
    MainbodyMoneyTotal = "c"  # 主体间资金交收汇总
    MainPartMonitorData = "d"  # 总分平衡监管数据
    PreparationMoney = "e"  # 存管银行备付金余额
    BankMoneyMonitorData = "f"  # 协办存管银行资金监管数据


class CashExchangeCode(StrEnum):
    """汇钞标志类型

    Attributes:
        Exchange: 汇
        Cash: 钞
    """

    Exchange = "1"  # 汇
    Cash = "2"  # 钞


class YesNoIndicator(StrEnum):
    """是或否标识类型

    Attributes:
        Yes: 是
        No: 否
    """

    Yes = "0"  # 是
    No = "1"  # 否


class BalanceType(StrEnum):
    """余额类型类型

    Attributes:
        CurrentMoney: 当前余额
        UsableMoney: 可用余额
        FetchableMoney: 可取余额
        FreezeMoney: 冻结余额
    """

    CurrentMoney = "0"  # 当前余额
    UsableMoney = "1"  # 可用余额
    FetchableMoney = "2"  # 可取余额
    FreezeMoney = "3"  # 冻结余额


class Gender(StrEnum):
    """性别类型

    Attributes:
        Unknown: 未知状态
        Male: 男
        Female: 女
    """

    Unknown = "0"  # 未知状态
    Male = "1"  # 男
    Female = "2"  # 女


class FeePayFlag(StrEnum):
    """费用支付标志类型

    Attributes:
        BEN: 由受益方支付费用
        OUR: 由发送方支付费用
        SHA: 由发送方支付发起的费用，受益方支付接受的费用
    """

    BEN = "0"  # 由受益方支付费用
    OUR = "1"  # 由发送方支付费用
    SHA = "2"  # 由发送方支付发起的费用，受益方支付接受的费用


class PassWordKeyType(StrEnum):
    """密钥类型类型

    Attributes:
        ExchangeKey: 交换密钥
        PassWordKey: 密码密钥
        MACKey: MAC密钥
        MessageKey: 报文密钥
    """

    ExchangeKey = "0"  # 交换密钥
    PassWordKey = "1"  # 密码密钥
    MACKey = "2"  # MAC密钥
    MessageKey = "3"  # 报文密钥


class FBTPassWordType(StrEnum):
    """密码类型类型

    Attributes:
        Query: 查询
        Fetch: 取款
        Transfer: 转帐
        Trade: 交易
    """

    Query = "0"  # 查询
    Fetch = "1"  # 取款
    Transfer = "2"  # 转帐
    Trade = "3"  # 交易


class FBTEncryMode(StrEnum):
    """加密方式类型

    Attributes:
        NoEncry: 不加密
        DES: DES
        DES3: 3DES
    """

    NoEncry = "0"  # 不加密
    DES = "1"  # DES
    DES3 = "2"  # 3DES


class BankRepealFlag(StrEnum):
    """银行冲正标志类型

    Attributes:
        BankNotNeedRepeal: 银行无需自动冲正
        BankWaitingRepeal: 银行待自动冲正
        BankBeenRepealed: 银行已自动冲正
    """

    BankNotNeedRepeal = "0"  # 银行无需自动冲正
    BankWaitingRepeal = "1"  # 银行待自动冲正
    BankBeenRepealed = "2"  # 银行已自动冲正


class BrokerRepealFlag(StrEnum):
    """期商冲正标志类型

    Attributes:
        BrokerNotNeedRepeal: 期商无需自动冲正
        BrokerWaitingRepeal: 期商待自动冲正
        BrokerBeenRepealed: 期商已自动冲正
    """

    BrokerNotNeedRepeal = "0"  # 期商无需自动冲正
    BrokerWaitingRepeal = "1"  # 期商待自动冲正
    BrokerBeenRepealed = "2"  # 期商已自动冲正


class InstitutionType(StrEnum):
    """机构类别类型

    Attributes:
        Bank: 银行
        Future: 期商
        Store: 券商
    """

    Bank = "0"  # 银行
    Future = "1"  # 期商
    Store = "2"  # 券商


class LastFragment(StrEnum):
    """最后分片标志类型

    Attributes:
        Yes: 是最后分片
        No: 不是最后分片
    """

    Yes = "0"  # 是最后分片
    No = "1"  # 不是最后分片


class BankAccStatus(StrEnum):
    """银行账户状态类型

    Attributes:
        Normal: 正常
        Freeze: 冻结
        ReportLoss: 挂失
    """

    Normal = "0"  # 正常
    Freeze = "1"  # 冻结
    ReportLoss = "2"  # 挂失


class MoneyAccountStatus(StrEnum):
    """资金账户状态类型

    Attributes:
        Normal: 正常
        Cancel: 销户
    """

    Normal = "0"  # 正常
    Cancel = "1"  # 销户


class ManageStatus(StrEnum):
    """存管状态类型

    Attributes:
        Point: 指定存管
        PrePoint: 预指定
        CancelPoint: 撤销指定
    """

    Point = "0"  # 指定存管
    PrePoint = "1"  # 预指定
    CancelPoint = "2"  # 撤销指定


class SystemType(StrEnum):
    """应用系统类型类型

    Attributes:
        FutureBankTransfer: 银期转帐
        StockBankTransfer: 银证转帐
        TheThirdPartStore: 第三方存管
    """

    FutureBankTransfer = "0"  # 银期转帐
    StockBankTransfer = "1"  # 银证转帐
    TheThirdPartStore = "2"  # 第三方存管


class TxnEndFlag(StrEnum):
    """银期转帐划转结果标志类型

    Attributes:
        NormalProcessing: 正常处理中
        Success: 成功结束
        Failed: 失败结束
        Abnormal: 异常中
        ManualProcessedForException: 已人工异常处理
        CommFailedNeedManualProcess: 通讯异常 ，请人工处理
        SysErrorNeedManualProcess: 系统出错，请人工处理
    """

    NormalProcessing = "0"  # 正常处理中
    Success = "1"  # 成功结束
    Failed = "2"  # 失败结束
    Abnormal = "3"  # 异常中
    ManualProcessedForException = "4"  # 已人工异常处理
    CommFailedNeedManualProcess = "5"  # 通讯异常 ，请人工处理
    SysErrorNeedManualProcess = "6"  # 系统出错，请人工处理


class ProcessStatus(StrEnum):
    """银期转帐服务处理状态类型

    Attributes:
        NotProcess: 未处理
        StartProcess: 开始处理
        Finished: 处理完成
    """

    NotProcess = "0"  # 未处理
    StartProcess = "1"  # 开始处理
    Finished = "2"  # 处理完成


class CustType(StrEnum):
    """客户类型类型

    Attributes:
        Person: 自然人
        Institution: 机构户
    """

    Person = "0"  # 自然人
    Institution = "1"  # 机构户


class FBTTransferDirection(StrEnum):
    """银期转帐方向类型

    Attributes:
        FromBankToFuture: 入金，银行转期货
        FromFutureToBank: 出金，期货转银行
    """

    FromBankToFuture = "1"  # 入金，银行转期货
    FromFutureToBank = "2"  # 出金，期货转银行


class OpenOrDestroy(StrEnum):
    """开销户类别类型

    Attributes:
        Open: 开户
        Destroy: 销户
    """

    Open = "1"  # 开户
    Destroy = "0"  # 销户


class AvailabilityFlag(StrEnum):
    """有效标志类型

    Attributes:
        Invalid: 未确认
        Valid: 有效
        Repeal: 冲正
    """

    Invalid = "0"  # 未确认
    Valid = "1"  # 有效
    Repeal = "2"  # 冲正


class OrganType(StrEnum):
    """机构类型类型

    Attributes:
        Bank: 银行代理
        Future: 交易前置
        PlateForm: 银期转帐平台管理
    """

    Bank = "1"  # 银行代理
    Future = "2"  # 交易前置
    PlateForm = "9"  # 银期转帐平台管理


class OrganLevel(StrEnum):
    """机构级别类型

    Attributes:
        HeadQuarters: 银行总行或期商总部
        Branch: 银行分中心或期货公司营业部
    """

    HeadQuarters = "1"  # 银行总行或期商总部
    Branch = "2"  # 银行分中心或期货公司营业部


class ProtocolID(StrEnum):
    """协议类型类型

    Attributes:
        FutureProtocol: 期商协议
        ICBCProtocol: 工行协议
        ABCProtocol: 农行协议
        CBCProtocol: 中国银行协议
        CCBProtocol: 建行协议
        BOCOMProtocol: 交行协议
        FBTPlateFormProtocol: 银期转帐平台协议
    """

    FutureProtocol = "0"  # 期商协议
    ICBCProtocol = "1"  # 工行协议
    ABCProtocol = "2"  # 农行协议
    CBCProtocol = "3"  # 中国银行协议
    CCBProtocol = "4"  # 建行协议
    BOCOMProtocol = "5"  # 交行协议
    FBTPlateFormProtocol = "X"  # 银期转帐平台协议


class ConnectMode(StrEnum):
    """套接字连接方式类型

    Attributes:
        ShortConnect: 短连接
        LongConnect: 长连接
    """

    ShortConnect = "0"  # 短连接
    LongConnect = "1"  # 长连接


class SyncMode(StrEnum):
    """套接字通信方式类型

    Attributes:
        ASync: 异步
        Sync: 同步
    """

    ASync = "0"  # 异步
    Sync = "1"  # 同步


class BankAccType(StrEnum):
    """银行帐号类型类型

    Attributes:
        BankBook: 银行存折
        SavingCard: 储蓄卡
        CreditCard: 信用卡
    """

    BankBook = "1"  # 银行存折
    SavingCard = "2"  # 储蓄卡
    CreditCard = "3"  # 信用卡


class FutureAccType(StrEnum):
    """期货公司帐号类型类型

    Attributes:
        BankBook: 银行存折
        SavingCard: 储蓄卡
        CreditCard: 信用卡
    """

    BankBook = "1"  # 银行存折
    SavingCard = "2"  # 储蓄卡
    CreditCard = "3"  # 信用卡


class OrganStatus(StrEnum):
    """接入机构状态类型

    Attributes:
        Ready: 启用
        CheckIn: 签到
        CheckOut: 签退
        CheckFileArrived: 对帐文件到达
        CheckDetail: 对帐
        DayEndClean: 日终清理
        Invalid: 注销
    """

    Ready = "0"  # 启用
    CheckIn = "1"  # 签到
    CheckOut = "2"  # 签退
    CheckFileArrived = "3"  # 对帐文件到达
    CheckDetail = "4"  # 对帐
    DayEndClean = "5"  # 日终清理
    Invalid = "9"  # 注销


class CCBFeeMode(StrEnum):
    """建行收费模式类型

    Attributes:
        ByAmount: 按金额扣收
        ByMonth: 按月扣收
    """

    ByAmount = "1"  # 按金额扣收
    ByMonth = "2"  # 按月扣收


class CommApiType(StrEnum):
    """通讯API类型类型

    Attributes:
        Client: 客户端
        Server: 服务端
        UserApi: 交易系统的UserApi
    """

    Client = "1"  # 客户端
    Server = "2"  # 服务端
    UserApi = "3"  # 交易系统的UserApi


class LinkStatus(StrEnum):
    """连接状态类型

    Attributes:
        Connected: 已经连接
        Disconnected: 没有连接
    """

    Connected = "1"  # 已经连接
    Disconnected = "2"  # 没有连接


class PwdFlag(StrEnum):
    """密码核对标志类型

    Attributes:
        NoCheck: 不核对
        BlankCheck: 明文核对
        EncryptCheck: 密文核对
    """

    NoCheck = "0"  # 不核对
    BlankCheck = "1"  # 明文核对
    EncryptCheck = "2"  # 密文核对


class SecuAccType(StrEnum):
    """期货帐号类型类型

    Attributes:
        AccountID: 资金帐号
        CardID: 资金卡号
        SHStockholderID: 上海股东帐号
        SZStockholderID: 深圳股东帐号
    """

    AccountID = "1"  # 资金账号
    CardID = "2"  # 资金卡号
    SHStockholderID = "3"  # 上海股东账号
    SZStockholderID = "4"  # 深圳股东账号


class TransferStatus(StrEnum):
    """转账交易状态类型

    Attributes:
        Normal: 正常
        Repealed: 被冲正
    """

    Normal = "0"  # 正常
    Repealed = "1"  # 被冲正


class SponsorType(StrEnum):
    """发起方类型

    Attributes:
        Broker: 期商
        Bank: 银行
    """

    Broker = "0"  # 期商
    Bank = "1"  # 银行


class ReqRspType(StrEnum):
    """请求响应类别类型

    Attributes:
        Request: 请求
        Response: 响应
    """

    Request = "0"  # 请求
    Response = "1"  # 响应


class FBTUserEventType(StrEnum):
    """银期转帐用户事件类型类型

    Attributes:
        SignIn: 签到
        FromBankToFuture: 银行转期货
        FromFutureToBank: 期货转银行
        OpenAccount: 开户
        CancelAccount: 销户
        ChangeAccount: 变更银行账户
        RepealFromBankToFuture: 冲正银行转期货
        RepealFromFutureToBank: 冲正期货转银行
        QueryBankAccount: 查询银行账户
        QueryFutureAccount: 查询期货账户
        SignOut: 签退
        SyncKey: 密钥同步
        ReserveOpenAccount: 预约开户
        CancelReserveOpenAccount: 撤销预约开户
        ReserveOpenAccountConfirm: 预约开户确认
        Other: 其他
    """

    SignIn = "0"  # 签到
    FromBankToFuture = "1"  # 银行转期货
    FromFutureToBank = "2"  # 期货转银行
    OpenAccount = "3"  # 开户
    CancelAccount = "4"  # 销户
    ChangeAccount = "5"  # 变更银行账户
    RepealFromBankToFuture = "6"  # 冲正银行转期货
    RepealFromFutureToBank = "7"  # 冲正期货转银行
    QueryBankAccount = "8"  # 查询银行账户
    QueryFutureAccount = "9"  # 查询期货账户
    SignOut = "A"  # 签退
    SyncKey = "B"  # 密钥同步
    ReserveOpenAccount = "C"  # 预约开户
    CancelReserveOpenAccount = "D"  # 撤销预约开户
    ReserveOpenAccountConfirm = "E"  # 预约开户确认
    Other = "Z"  # 其他


class DBOperation(StrEnum):
    """记录操作类型类型

    Attributes:
        Insert: 插入
        Update: 更新
        Delete: 删除
    """

    Insert = "0"  # 插入
    Update = "1"  # 更新
    Delete = "2"  # 删除


class SyncFlag(StrEnum):
    """同步标记类型

    Attributes:
        Yes: 已同步
        No: 未同步
    """

    Yes = "0"  # 已同步
    No = "1"  # 未同步


class SyncType(StrEnum):
    """同步类型类型

    Attributes:
        OneOffSync: 一次同步
        TimerSync: 定时同步
        TimerFullSync: 定时完全同步
    """

    OneOffSync = "0"  # 一次同步
    TimerSync = "1"  # 定时同步
    TimerFullSync = "2"  # 定时完全同步


class ExDirection(StrEnum):
    """换汇方向类型

    Attributes:
        Settlement: 结汇
        Sale: 售汇
    """

    Settlement = "0"  # 结汇
    Sale = "1"  # 售汇


class FBEResultFlag(StrEnum):
    """换汇成功标志类型

    Attributes:
        Success: 成功
        InsufficientBalance: 账户余额不足
        UnknownTrading: 交易结果未知
        Fail: 失败
    """

    Success = "0"  # 成功
    InsufficientBalance = "1"  # 账户余额不足
    UnknownTrading = "8"  # 交易结果未知
    Fail = "x"  # 失败


class FBEExchStatus(StrEnum):
    """换汇交易状态类型

    Attributes:
        Normal: 正常
        ReExchange: 交易重发
    """

    Normal = "0"  # 正常
    ReExchange = "1"  # 交易重发


class FBEFileFlag(StrEnum):
    """换汇文件标志类型

    Attributes:
        DataPackage: 数据包
        File: 文件
    """

    DataPackage = "0"  # 数据包
    File = "1"  # 文件


class FBEAlreadyTrade(StrEnum):
    """换汇已交易标志类型

    Attributes:
        NotTrade: 未交易
        Trade: 已交易
    """

    NotTrade = "0"  # 未交易
    Trade = "1"  # 已交易


class FBEUserEventType(StrEnum):
    """银期换汇用户事件类型类型

    Attributes:
        SignIn: 签到
        Exchange: 换汇
        ReExchange: 换汇重发
        QueryBankAccount: 银行账户查询
        QueryExchDetail: 换汇明细查询
        QueryExchSummary: 换汇汇总查询
        QueryExchRate: 换汇汇率查询
        CheckBankAccount: 对账文件通知
        SignOut: 签退
        Other: 其他
    """

    SignIn = "0"  # 签到
    Exchange = "1"  # 换汇
    ReExchange = "2"  # 换汇重发
    QueryBankAccount = "3"  # 银行账户查询
    QueryExchDetail = "4"  # 换汇明细查询
    QueryExchSummary = "5"  # 换汇汇总查询
    QueryExchRate = "6"  # 换汇汇率查询
    CheckBankAccount = "7"  # 对账文件通知
    SignOut = "8"  # 签退
    Other = "Z"  # 其他


class FBEReqFlag(StrEnum):
    """换汇发送标志类型

    Attributes:
        UnProcessed: 未处理
        WaitSend: 等待发送
        SendSuccess: 发送成功
        SendFailed: 发送失败
        WaitReSend: 等待重发
    """

    UnProcessed = "0"  # 未处理
    WaitSend = "1"  # 等待发送
    SendSuccess = "2"  # 发送成功
    SendFailed = "3"  # 发送失败
    WaitReSend = "4"  # 等待重发


class NotifyClass(StrEnum):
    """风险通知类型类型

    Attributes:
        NOERROR: 正常
        Warn: 警示
        Call: 追保
        Force: 强平
        CHUANCANG: 穿仓
        Exception: 异常
    """

    NOERROR = "0"  # 正常
    Warn = "1"  # 警示
    Call = "2"  # 追保
    Force = "3"  # 强平
    CHUANCANG = "4"  # 穿仓
    Exception = "5"  # 异常


class ForceCloseType(StrEnum):
    """强平单类型类型

    Attributes:
        Manual: 手工强平
        Single: 单一投资者辅助强平
        Group: 批量投资者辅助强平
    """

    Manual = "0"  # 手工强平
    Single = "1"  # 单一投资者辅助强平
    Group = "2"  # 批量投资者辅助强平


class RiskNotifyMethod(StrEnum):
    """风险通知途径类型

    Attributes:
        System: 系统通知
        SMS: 短信通知
        EMail: 邮件通知
        Manual: 人工通知
    """

    System = "0"  # 系统通知
    SMS = "1"  # 短信通知
    EMail = "2"  # 邮件通知
    Manual = "3"  # 人工通知


class RiskNotifyStatus(StrEnum):
    """风险通知状态类型

    Attributes:
        NotGen: 未生成
        Generated: 已生成未发送
        SendError: 发送失败
        SendOk: 已发送未接收
        Received: 已接收未确认
        Confirmed: 已确认
    """

    NotGen = "0"  # 未生成
    Generated = "1"  # 已生成未发送
    SendError = "2"  # 发送失败
    SendOk = "3"  # 已发送未接收
    Received = "4"  # 已接收未确认
    Confirmed = "5"  # 已确认


class RiskUserEvent(StrEnum):
    """风控用户操作事件类型

    Attributes:
        ExportData: 导出数据
    """

    ExportData = "0"  # 导出数据


class ConditionalOrderSortType(StrEnum):
    """条件单索引条件类型

    Attributes:
        LastPriceAsc: 使用最新价升序
        LastPriceDesc: 使用最新价降序
        AskPriceAsc: 使用卖价升序
        AskPriceDesc: 使用卖价降序
        BidPriceAsc: 使用买价升序
        BidPriceDesc: 使用买价降序
    """

    LastPriceAsc = "0"  # 使用最新价升序
    LastPriceDesc = "1"  # 使用最新价降序
    AskPriceAsc = "2"  # 使用卖价升序
    AskPriceDesc = "3"  # 使用卖价降序
    BidPriceAsc = "4"  # 使用买价升序
    BidPriceDesc = "5"  # 使用买价降序


class SendType(StrEnum):
    """报送状态类型

    Attributes:
        NoSend: 未发送
        Sent: 已发送
        Generated: 已生成
        SendFail: 报送失败
        Success: 接收成功
        Fail: 接收失败
        Cancel: 取消报送
    """

    NoSend = "0"  # 未发送
    Sent = "1"  # 已发送
    Generated = "2"  # 已生成
    SendFail = "3"  # 报送失败
    Success = "4"  # 接收成功
    Fail = "5"  # 接收失败
    Cancel = "6"  # 取消报送


class ClientIDStatus(StrEnum):
    """交易编码状态类型

    Attributes:
        NoApply: 未申请
        Submitted: 已提交申请
        Sent: 已发送申请
        Success: 完成
        Refuse: 拒绝
        Cancel: 已撤销编码
    """

    NoApply = "1"  # 未申请
    Submitted = "2"  # 已提交申请
    Sent = "3"  # 已发送申请
    Success = "4"  # 完成
    Refuse = "5"  # 拒绝
    Cancel = "6"  # 已撤销编码


class QuestionType(StrEnum):
    """特有信息类型类型

    Attributes:
        Radio: 单选
        Option: 多选
        Blank: 填空
    """

    Radio = "1"  # 单选
    Option = "2"  # 多选
    Blank = "3"  # 填空


class BusinessType(StrEnum):
    """业务类型类型

    Attributes:
        Request: 请求
        Response: 应答
        Notice: 通知
    """

    Request = "1"  # 请求
    Response = "2"  # 应答
    Notice = "3"  # 通知


class CfmmcReturnCode(StrEnum):
    """监控中心返回码类型

    Attributes:
        Success: 成功
        Working: 该客户已经有流程在处理中
        InfoFail: 监控中客户资料检查失败
        IDCardFail: 监控中实名制检查失败
        OtherFail: 其他错误
    """

    Success = "0"  # 成功
    Working = "1"  # 该客户已经有流程在处理中
    InfoFail = "2"  # 监控中客户资料检查失败
    IDCardFail = "3"  # 监控中实名制检查失败
    OtherFail = "4"  # 其他错误


class ClientType(StrEnum):
    """客户类型类型

    Attributes:
        All: 所有
        Person: 个人
        Company: 单位
        Other: 其他
        SpecialOrgan: 特殊法人
        Asset: 资管户
    """

    All = "0"  # 所有
    Person = "1"  # 个人
    Company = "2"  # 单位
    Other = "3"  # 其他
    SpecialOrgan = "4"  # 特殊法人
    Asset = "5"  # 资管户


class ExClientIDType(StrEnum):
    """交易编码类型类型

    Attributes:
        Hedge: 套保
        Arbitrage: 套利
        Speculation: 投机
    """

    Hedge = "1"  # 套保
    Arbitrage = "2"  # 套利
    Speculation = "3"  # 投机


class UpdateFlag(StrEnum):
    """更新状态类型

    Attributes:
        NoUpdate: 未更新
        Success: 更新全部信息成功
        Fail: 更新全部信息失败
        TCSuccess: 更新交易编码成功
        TCFail: 更新交易编码失败
        Cancel: 已丢弃
    """

    NoUpdate = "0"  # 未更新
    Success = "1"  # 更新全部信息成功
    Fail = "2"  # 更新全部信息失败
    TCSuccess = "3"  # 更新交易编码成功
    TCFail = "4"  # 更新交易编码失败
    Cancel = "5"  # 已丢弃


class ApplyOperateID(StrEnum):
    """申请动作类型

    Attributes:
        OpenInvestor: 开户
        ModifyIDCard: 修改身份信息
        ModifyNoIDCard: 修改一般信息
        ApplyTradingCode: 申请交易编码
        CancelTradingCode: 撤销交易编码
        CancelInvestor: 销户
        FreezeAccount: 账户休眠
        ActiveFreezeAccount: 激活休眠账户
    """

    OpenInvestor = "1"  # 开户
    ModifyIDCard = "2"  # 修改身份信息
    ModifyNoIDCard = "3"  # 修改一般信息
    ApplyTradingCode = "4"  # 申请交易编码
    CancelTradingCode = "5"  # 撤销交易编码
    CancelInvestor = "6"  # 销户
    FreezeAccount = "8"  # 账户休眠
    ActiveFreezeAccount = "9"  # 激活休眠账户


class ApplyStatusID(StrEnum):
    """申请状态类型

    Attributes:
        NoComplete: 未补全
        Submitted: 已提交
        Checked: 已审核
        Refused: 已拒绝
        Deleted: 已删除
    """

    NoComplete = "1"  # 未补全
    Submitted = "2"  # 已提交
    Checked = "3"  # 已审核
    Refused = "4"  # 已拒绝
    Deleted = "5"  # 已删除


class SendMethod(StrEnum):
    """发送方式类型

    Attributes:
        ByAPI: 文件发送
        ByFile: 电子发送
    """

    ByAPI = "1"  # 文件发送
    ByFile = "2"  # 电子发送


class EventMode(StrEnum):
    """操作方法类型

    Attributes:
        ADD: 增加
        UPDATE: 修改
        DELETE: 删除
        CHECK: 复核
        COPY: 复制
        CANCEL: 注销
        Reverse: 冲销
    """

    ADD = "1"  # 增加
    UPDATE = "2"  # 修改
    DELETE = "3"  # 删除
    CHECK = "4"  # 复核
    COPY = "5"  # 复制
    CANCEL = "6"  # 注销
    Reverse = "7"  # 冲销


class UOAAutoSend(StrEnum):
    """统一开户申请自动发送类型

    Attributes:
        ASR: 自动发送并接收
        ASNR: 自动发送，不自动接收
        NSAR: 不自动发送，自动接收
        NSR: 不自动发送，也不自动接收
    """

    ASR = "1"  # 自动发送并接收
    ASNR = "2"  # 自动发送，不自动接收
    NSAR = "3"  # 不自动发送，自动接收
    NSR = "4"  # 不自动发送，也不自动接收


class FlowID(StrEnum):
    """流程ID类型

    Attributes:
        InvestorGroupFlow: 投资者对应投资者组设置
        InvestorRate: 投资者手续费率设置
        InvestorCommRateModel: 投资者手续费率模板关系设置
    """

    InvestorGroupFlow = "1"  # 投资者对应投资者组设置
    InvestorRate = "2"  # 投资者手续费率设置
    InvestorCommRateModel = "3"  # 投资者手续费率模板关系设置


class CheckLevel(StrEnum):
    """复核级别类型

    Attributes:
        Zero: 零级复核
        One: 一级复核
        Two: 二级复核
    """

    Zero = "0"  # 零级复核
    One = "1"  # 一级复核
    Two = "2"  # 二级复核


class CheckStatus(StrEnum):
    """复核级别类型

    Attributes:
        Init: 未复核
        Checking: 复核中
        Checked: 已复核
        Refuse: 拒绝
        Cancel: 作废
    """

    Init = "0"  # 未复核
    Checking = "1"  # 复核中
    Checked = "2"  # 已复核
    Refuse = "3"  # 拒绝
    Cancel = "4"  # 作废


class UsedStatus(StrEnum):
    """生效状态类型

    Attributes:
        Unused: 未生效
        Used: 已生效
        Fail: 生效失败
    """

    Unused = "0"  # 未生效
    Used = "1"  # 已生效
    Fail = "2"  # 生效失败


class BankAccountOrigin(StrEnum):
    """账户来源类型

    Attributes:
        ByAccProperty: 手工录入
        ByFBTransfer: 银期转账
    """

    ByAccProperty = "0"  # 手工录入
    ByFBTransfer = "1"  # 银期转账


class MonthBillTradeSum(StrEnum):
    """结算单月报成交汇总方式类型

    Attributes:
        ByInstrument: 同日同合约
        ByDayInsPrc: 同日同合约同价格
        ByDayIns: 同合约
    """

    ByInstrument = "0"  # 同日同合约
    ByDayInsPrc = "1"  # 同日同合约同价格
    ByDayIns = "2"  # 同合约


class FBTTradeCodeEnum(StrEnum):
    """银期交易代码枚举类型

    Attributes:
        BankLaunchBankToBroker: 银行发起银行转期货
        BrokerLaunchBankToBroker: 期货发起银行转期货
        BankLaunchBrokerToBank: 银行发起期货转银行
        BrokerLaunchBrokerToBank: 期货发起期货转银行
    """

    BankLaunchBankToBroker = "102001"  # 银行发起银行转期货
    BrokerLaunchBankToBroker = "202001"  # 期货发起银行转期货
    BankLaunchBrokerToBank = "102002"  # 银行发起期货转银行
    BrokerLaunchBrokerToBank = "202002"  # 期货发起期货转银行


class OTPType(StrEnum):
    """动态令牌类型类型

    Attributes:
        NONE: 无动态令牌
        TOTP: 时间令牌
    """

    NONE = "0"  # 无动态令牌
    TOTP = "1"  # 时间令牌


class OTPStatus(StrEnum):
    """动态令牌状态类型

    Attributes:
        Unused: 未使用
        Used: 已使用
        Disuse: 注销
    """

    Unused = "0"  # 未使用
    Used = "1"  # 已使用
    Disuse = "2"  # 注销


class BrokerUserType(StrEnum):
    """经济公司用户类型类型

    Attributes:
        Investor: 投资者
        BrokerUser: 操作员
    """

    Investor = "1"  # 投资者
    BrokerUser = "2"  # 操作员


class FutureType(StrEnum):
    """期货类型类型

    Attributes:
        Commodity: 商品期货
        Financial: 金融期货
    """

    Commodity = "1"  # 商品期货
    Financial = "2"  # 金融期货


class FundEventType(StrEnum):
    """资金管理操作类型类型

    Attributes:
        Restriction: 转账限额
        TodayRestriction: 当日转账限额
        Transfer: 期商流水
        Credit: 资金冻结
        InvestorWithdrawAlm: 投资者可提资金比例
        BankRestriction: 单个银行帐户转账限额
        AccountRegister: 银期签约账户
        ExchangeFundIO: 交易所出入金
        InvestorFundIO: 投资者出入金
    """

    Restriction = "0"  # 转账限额
    TodayRestriction = "1"  # 当日转账限额
    Transfer = "2"  # 期商流水
    Credit = "3"  # 资金冻结
    InvestorWithdrawAlm = "4"  # 投资者可提资金比例
    BankRestriction = "5"  # 单个银行账户转账限额
    AccountRegister = "6"  # 银期签约账户
    ExchangeFundIO = "7"  # 交易所出入金
    InvestorFundIO = "8"  # 投资者出入金


class AccountSourceType(StrEnum):
    """资金账户来源类型

    Attributes:
        FBTransfer: 银期同步
        ManualEntry: 手工录入
    """

    FBTransfer = "0"  # 银期同步
    ManualEntry = "1"  # 手工录入


class CodeSourceType(StrEnum):
    """交易编码来源类型

    Attributes:
        UnifyAccount: 统一开户(已规范)
        ManualEntry: 手工录入(未规范)
    """

    UnifyAccount = "0"  # 统一开户(已规范)
    ManualEntry = "1"  # 手工录入(未规范)


class UserRange(StrEnum):
    """操作员范围类型

    Attributes:
        All: 所有
        Single: 单一操作员
    """

    All = "0"  # 所有
    Single = "1"  # 单一操作员


class ByGroup(StrEnum):
    """交易统计表按客户统计方式类型

    Attributes:
        Investor: 按投资者统计
        Group: 按类统计
    """

    Investor = "2"  # 按投资者统计
    Group = "1"  # 按类统计


class TradeSumStatMode(StrEnum):
    """交易统计表按范围统计方式类型

    Attributes:
        Instrument: 按合约统计
        Product: 按产品统计
        Exchange: 按交易所统计
    """

    Instrument = "1"  # 按合约统计
    Product = "2"  # 按产品统计
    Exchange = "3"  # 按交易所统计


class ExprSetMode(StrEnum):
    """日期表达式设置类型类型

    Attributes:
        Relative: 相对已有规则设置
        Typical: 典型设置
    """

    Relative = "1"  # 相对已有规则设置
    Typical = "2"  # 典型设置


class RateInvestorRange(StrEnum):
    """投资者范围类型

    Attributes:
        All: 公司标准
        Model: 模板
        Single: 单一投资者
    """

    All = "1"  # 公司标准
    Model = "2"  # 模板
    Single = "3"  # 单一投资者


class SyncDataStatus(StrEnum):
    """主次用系统数据同步状态类型

    Attributes:
        Initialize: 未同步
        Syncing: 同步中
        Synchronized: 已同步
    """

    Initialize = "0"  # 未同步
    Syncing = "1"  # 同步中
    Synchronized = "2"  # 已同步


class FlexStatMode(StrEnum):
    """产品合约统计方式类型

    Attributes:
        Product: 产品统计
        Exchange: 交易所统计
        All: 统计所有
    """

    Product = "1"  # 产品统计
    Exchange = "2"  # 交易所统计
    All = "3"  # 统计所有


class ByInvestorRange(StrEnum):
    """投资者范围统计方式类型

    Attributes:
        Property: 属性统计
        All: 统计所有
    """

    Property = "1"  # 属性统计
    All = "2"  # 统计所有


class PropertyInvestorRange(StrEnum):
    """投资者范围类型

    Attributes:
        All: 所有
        Property: 投资者属性
        Single: 单一投资者
    """

    All = "1"  # 所有
    Property = "2"  # 投资者属性
    Single = "3"  # 单一投资者


class FileStatus(StrEnum):
    """文件状态类型

    Attributes:
        NoCreate: 未生成
        Created: 已生成
        Failed: 生成失败
    """

    NoCreate = "0"  # 未生成
    Created = "1"  # 已生成
    Failed = "2"  # 生成失败


class FileGenStyle(StrEnum):
    """文件生成方式类型

    Attributes:
        FileTransmit: 下发
        FileGen: 生成
    """

    FileTransmit = "0"  # 下发
    FileGen = "1"  # 生成


class SysOperMode(StrEnum):
    """系统日志操作方法类型

    Attributes:
        Add: 增加
        Update: 修改
        Delete: 删除
        Copy: 复制
        Active: 激活
        CanCel: 注销
        ReSet: 重置
    """

    Add = "1"  # 增加
    Update = "2"  # 修改
    Delete = "3"  # 删除
    Copy = "4"  # 复制
    Active = "5"  # 激活
    CanCel = "6"  # 注销
    ReSet = "7"  # 重置


class SysOperType(StrEnum):
    """系统日志操作类型类型

    Attributes:
        UpdatePassword: 修改操作员密码
        UserDepartment: 操作员组织架构关系
        RoleManager: 角色管理
        RoleFunction: 角色功能设置
        BaseParam: 基础参数设置
        SetUserID: 设置操作员
        SetUserRole: 用户角色设置
        UserIpRestriction: 用户IP限制
        DepartmentManager: 组织架构管理
        DepartmentCopy: 组织架构向查询分类复制
        TradingCode: 交易编码管理
        InvestorStatus: 投资者状态维护
        InvestorAuthority: 投资者权限管理
        PropertySet: 属性设置
        ReSetInvestorPasswd: 重置投资者密码
        InvestorPersonalityInfo: 投资者个性信息维护
    """

    UpdatePassword = "0"  # 修改操作员密码
    UserDepartment = "1"  # 操作员组织架构关系
    RoleManager = "2"  # 角色管理
    RoleFunction = "3"  # 角色功能设置
    BaseParam = "4"  # 基础参数设置
    SetUserID = "5"  # 设置操作员
    SetUserRole = "6"  # 用户角色设置
    UserIpRestriction = "7"  # 用户IP限制
    DepartmentManager = "8"  # 组织架构管理
    DepartmentCopy = "9"  # 组织架构向查询分类复制
    TradingCode = "A"  # 交易编码管理
    InvestorStatus = "B"  # 投资者状态维护
    InvestorAuthority = "C"  # 投资者权限管理
    PropertySet = "D"  # 属性设置
    ReSetInvestorPasswd = "E"  # 重置投资者密码
    InvestorPersonalityInfo = "F"  # 投资者个性信息维护


class CSRCDataQueryType(StrEnum):
    """上报数据查询类型类型

    Attributes:
        Current: 查询当前交易日报送的数据
        History: 查询历史报送的代理经纪公司的数据
    """

    Current = "0"  # 查询当前交易日报送的数据
    History = "1"  # 查询历史报送的代理经纪公司的数据


class FreezeStatus(StrEnum):
    """休眠状态类型

    Attributes:
        Normal: 活跃
        Freeze: 休眠
    """

    Normal = "1"  # 活跃
    Freeze = "0"  # 休眠


class StandardStatus(StrEnum):
    """规范状态类型

    Attributes:
        Standard: 已规范
        NonStandard: 未规范
    """

    Standard = "0"  # 已规范
    NonStandard = "1"  # 未规范


class RightParamType(StrEnum):
    """配置类型类型

    Attributes:
        Freeze: 休眠户
        FreezeActive: 激活休眠户
        OpenLimit: 开仓权限限制
        RelieveOpenLimit: 解除开仓权限限制
    """

    Freeze = "1"  # 休眠户
    FreezeActive = "2"  # 激活休眠户
    OpenLimit = "3"  # 开仓权限限制
    RelieveOpenLimit = "4"  # 解除开仓权限限制


class DataStatus(StrEnum):
    """反洗钱审核表数据状态类型

    Attributes:
        Normal: 正常
        Deleted: 已删除
    """

    Normal = "0"  # 正常
    Deleted = "1"  # 已删除


class AMLCheckStatus(StrEnum):
    """审核状态类型

    Attributes:
        Init: 未复核
        Checking: 复核中
        Checked: 已复核
        RefuseReport: 拒绝上报
    """

    Init = "0"  # 未复核
    Checking = "1"  # 复核中
    Checked = "2"  # 已复核
    RefuseReport = "3"  # 拒绝上报


class AmlDateType(StrEnum):
    """日期类型类型

    Attributes:
        DrawDay: 检查日期
        TouchDay: 发生日期
    """

    DrawDay = "0"  # 检查日期
    TouchDay = "1"  # 发生日期


class AmlCheckLevel(StrEnum):
    """审核级别类型

    Attributes:
        CheckLevel0: 零级审核
        CheckLevel1: 一级审核
        CheckLevel2: 二级审核
        CheckLevel3: 三级审核
    """

    CheckLevel0 = "0"  # 零级审核
    CheckLevel1 = "1"  # 一级审核
    CheckLevel2 = "2"  # 二级审核
    CheckLevel3 = "3"  # 三级审核


class SettleManagerType(StrEnum):
    """结算配置类型类型

    Attributes:
        Before: 结算前准备
        Settlement: 结算
        After: 结算后核对
        Settlemented: 结算后处理
    """

    Before = "1"  # 结算前准备
    Settlement = "2"  # 结算
    After = "3"  # 结算后核对
    Settlemented = "4"  # 结算后处理


class SettleManagerLevel(StrEnum):
    """结算配置等级类型

    Attributes:
        Must: 必要
        Alarm: 警告
        Prompt: 提示
        Ignore: 不检查
    """

    Must = "1"  # 必要
    Alarm = "2"  # 警告
    Prompt = "3"  # 提示
    Ignore = "4"  # 不检查


class SettleManagerGroup(StrEnum):
    """模块分组类型

    Attributes:
        Exchange: 交易所核对
        ASP: 内部核对
        CSRC: 上报数据核对
    """

    Exchange = "1"  # 交易所核对
    ASP = "2"  # 内部核对
    CSRC = "3"  # 上报数据核对


class LimitUseType(StrEnum):
    """保值额度使用类型类型

    Attributes:
        Repeatable: 可重复使用
        Unrepeatable: 不可重复使用
    """

    Repeatable = "1"  # 可重复使用
    Unrepeatable = "2"  # 不可重复使用


class DataResource(StrEnum):
    """数据来源类型

    Attributes:
        Settle: 本系统
        Exchange: 交易所
        CSRC: 报送数据
    """

    Settle = "1"  # 本系统
    Exchange = "2"  # 交易所
    CSRC = "3"  # 报送数据


class MarginType(StrEnum):
    """保证金类型类型

    Attributes:
        ExchMarginRate: 交易所保证金率
        InstrMarginRate: 投资者保证金率
        InstrMarginRateTrade: 投资者交易保证金率
    """

    ExchMarginRate = "0"  # 交易所保证金率
    InstrMarginRate = "1"  # 投资者保证金率
    InstrMarginRateTrade = "2"  # 投资者交易保证金率


class ActiveType(StrEnum):
    """生效类型类型

    Attributes:
        Intraday: 仅当日生效
        Long: 长期生效
    """

    Intraday = "1"  # 仅当日生效
    Long = "2"  # 长期生效


class MarginRateType(StrEnum):
    """冲突保证金率类型类型

    Attributes:
        Exchange: 交易所保证金率
        Investor: 投资者保证金率
        InvestorTrade: 投资者交易保证金率
    """

    Exchange = "1"  # 交易所保证金率
    Investor = "2"  # 投资者保证金率
    InvestorTrade = "3"  # 投资者交易保证金率


class BackUpStatus(StrEnum):
    """备份数据状态类型

    Attributes:
        UnBack: 未生成备份数据
        Backup: 备份数据生成中
        Backuped: 已生成备份数据
        BackFail: 备份数据失败
    """

    UnBack = "0"  # 未生成备份数据
    Backup = "1"  # 备份数据生成中
    Backuped = "2"  # 已生成备份数据
    BackFail = "3"  # 备份数据失败


class InitSettlement(StrEnum):
    """结算初始化状态类型

    Attributes:
        UnInitialize: 结算初始化未开始
        Initialize: 结算初始化中
        Initialized: 结算初始化完成
    """

    UnInitialize = "0"  # 结算初始化未开始
    Initialize = "1"  # 结算初始化中
    Initialized = "2"  # 结算初始化完成


class ReportStatus(StrEnum):
    """报表数据生成状态类型

    Attributes:
        NoCreate: 未生成报表数据
        Create: 报表数据生成中
        Created: 已生成报表数据
        CreateFail: 生成报表数据失败
    """

    NoCreate = "0"  # 未生成报表数据
    Create = "1"  # 报表数据生成中
    Created = "2"  # 已生成报表数据
    CreateFail = "3"  # 生成报表数据失败


class SaveStatus(StrEnum):
    """数据归档状态类型

    Attributes:
        UnSaveData: 归档未完成
        SaveData: 归档完成
    """

    UnSaveData = "0"  # 归档未完成
    SaveData = "1"  # 归档完成


class SettArchiveStatus(StrEnum):
    """结算确认数据归档状态类型

    Attributes:
        UnArchived: 未归档数据
        Archiving: 数据归档中
        Archived: 已归档数据
        ArchiveFail: 归档数据失败
    """

    UnArchived = "0"  # 未归档数据
    Archiving = "1"  # 数据归档中
    Archived = "2"  # 已归档数据
    ArchiveFail = "3"  # 归档数据失败


class CTPType(StrEnum):
    """CTP交易系统类型类型

    Attributes:
        Unknown: 未知类型
        MainCenter: 主中心
        BackUp: 备中心
    """

    Unknown = "0"  # 未知类型
    MainCenter = "1"  # 主中心
    BackUp = "2"  # 备中心


class CloseDealType(StrEnum):
    """平仓处理类型类型

    Attributes:
        Normal: 正常
        SpecFirst: 投机平仓优先
    """

    Normal = "0"  # 正常
    SpecFirst = "1"  # 投机平仓优先


class MortgageFundUseRange(StrEnum):
    """货币质押资金可用范围类型

    Attributes:
        NONE: 不能使用
        Margin: 用于保证金
        All: 用于手续费、盈亏、保证金
        CNY3: 人民币方案3
    """

    NONE = "0"  # 不能使用
    Margin = "1"  # 用于保证金
    All = "2"  # 用于手续费、盈亏、保证金
    CNY3 = "3"  # 人民币方案3


class SpecProductType(StrEnum):
    """特殊产品类型类型

    Attributes:
        CzceHedge: 郑商所套保产品
        IneForeignCurrency: 货币质押产品
        DceOpenClose: 大连短线开平仓产品
    """

    CzceHedge = "1"  # 郑商所套保产品
    IneForeignCurrency = "2"  # 货币质押产品
    DceOpenClose = "3"  # 大连短线开平仓产品


class FundMortgageType(StrEnum):
    """货币质押类型类型

    Attributes:
        Mortgage: 质押
        Redemption: 解质
    """

    Mortgage = "1"  # 质押
    Redemption = "2"  # 解质


class AccountSettlementParamID(StrEnum):
    """投资者账户结算参数代码类型

    Attributes:
        BaseMargin: 基础保证金
        LowestInterest: 最低权益标准
    """

    BaseMargin = "1"  # 基础保证金
    LowestInterest = "2"  # 最低权益标准


class FundMortDirection(StrEnum):
    """货币质押方向类型

    Attributes:
        In: 货币质入
        Out: 货币质出
    """

    In = "1"  # 货币质入
    Out = "2"  # 货币质出


class BusinessClass(StrEnum):
    """换汇类别类型

    Attributes:
        Profit: 盈利
        Loss: 亏损
        Other: 其他
    """

    Profit = "0"  # 盈利
    Loss = "1"  # 亏损
    Other = "Z"  # 其他


class SwapSourceType(StrEnum):
    """换汇数据来源类型

    Attributes:
        Manual: 手工
        Automatic: 自动生成
    """

    Manual = "0"  # 手工
    Automatic = "1"  # 自动生成


class CurrExDirection(StrEnum):
    """换汇类型类型

    Attributes:
        Settlement: 结汇
        Sale: 售汇
    """

    Settlement = "0"  # 结汇
    Sale = "1"  # 售汇


class CurrencySwapStatus(StrEnum):
    """申请状态类型

    Attributes:
        Entry: 已录入
        Approve: 已审核
        Refuse: 已拒绝
        Revoke: 已撤销
        Send: 已发送
        Success: 换汇成功
        Failure: 换汇失败
    """

    Entry = "1"  # 已录入
    Approve = "2"  # 已审核
    Refuse = "3"  # 已拒绝
    Revoke = "4"  # 已撤销
    Send = "5"  # 已发送
    Success = "6"  # 换汇成功
    Failure = "7"  # 换汇失败


class ReqFlag(StrEnum):
    """换汇发送标志类型

    Attributes:
        NoSend: 未发送
        SendSuccess: 发送成功
        SendFailed: 发送失败
        WaitReSend: 等待重发
    """

    NoSend = "0"  # 未发送
    SendSuccess = "1"  # 发送成功
    SendFailed = "2"  # 发送失败
    WaitReSend = "3"  # 等待重发


class ResFlag(StrEnum):
    """换汇返回成功标志类型

    Attributes:
        Success: 成功
        Insufficient: 账户余额不足
        UnKnown: 交易结果未知
    """

    Success = "0"  # 成功
    Insufficient = "1"  # 账户余额不足
    UnKnown = "8"  # 交易结果未知


class ExStatus(StrEnum):
    """修改状态类型

    Attributes:
        Before: 修改前
        After: 修改后
    """

    Before = "0"  # 修改前
    After = "1"  # 修改后


class ClientRegion(StrEnum):
    """开户客户地域类型

    Attributes:
        Domestic: 国内客户
        GMT: 港澳台客户
        Foreign: 国外客户
    """

    Domestic = "1"  # 国内客户
    GMT = "2"  # 港澳台客户
    Foreign = "3"  # 国外客户


class HasBoard(StrEnum):
    """是否有董事会类型

    Attributes:
        No: 没有
        Yes: 有
    """

    No = "0"  # 没有
    Yes = "1"  # 有


class StartMode(StrEnum):
    """启动模式类型

    Attributes:
        Normal: 正常
        Emerge: 应急
        Restore: 恢复
    """

    Normal = "1"  # 正常
    Emerge = "2"  # 应急
    Restore = "3"  # 恢复


class TemplateType(StrEnum):
    """模型类型类型

    Attributes:
        Full: 全量
        Increment: 增量
        BackUp: 备份
    """

    Full = "1"  # 全量
    Increment = "2"  # 增量
    BackUp = "3"  # 备份


class LoginMode(StrEnum):
    """登录模式类型

    Attributes:
        Trade: 交易
        Transfer: 转账
    """

    Trade = "0"  # 交易
    Transfer = "1"  # 转账


class PromptType(StrEnum):
    """日历提示类型类型

    Attributes:
        Instrument: 合约上下市
        Margin: 保证金分段生效
    """

    Instrument = "1"  # 合约上下市
    Margin = "2"  # 保证金分段生效


class HasTrustee(StrEnum):
    """是否有托管人类型

    Attributes:
        Yes: 有
        No: 没有
    """

    Yes = "1"  # 有
    No = "0"  # 没有


class AmType(StrEnum):
    """机构类型类型

    Attributes:
        Bank: 银行
        Securities: 证券公司
        Fund: 基金公司
        Insurance: 保险公司
        Trust: 信托公司
        Other: 其他
    """

    Bank = "1"  # 银行
    Securities = "2"  # 证券公司
    Fund = "3"  # 基金公司
    Insurance = "4"  # 保险公司
    Trust = "5"  # 信托公司
    Other = "9"  # 其他


class CSRCFundIOType(StrEnum):
    """出入金类型类型

    Attributes:
        FundIO: 出入金
        SwapCurrency: 银期换汇
    """

    FundIO = "0"  # 出入金
    SwapCurrency = "1"  # 银期换汇


class CusAccountType(StrEnum):
    """结算账户类型类型

    Attributes:
        Futures: 期货结算账户
        AssetmgrFuture: 纯期货资管业务下的资管结算账户
        AssetmgrTrustee: 综合类资管业务下的期货资管托管账户
        AssetmgrTransfer: 综合类资管业务下的资金中转账户
    """

    Futures = "1"  # 期货结算账户
    AssetmgrFuture = "2"  # 纯期货资管业务下的资管结算账户
    AssetmgrTrustee = "3"  # 综合类资管业务下的期货资管托管账户
    AssetmgrTransfer = "4"  # 综合类资管业务下的资金中转账户


class LanguageType(StrEnum):
    """通知语言类型类型

    Attributes:
        Chinese: 中文
        English: 英文
    """

    Chinese = "1"  # 中文
    English = "2"  # 英文


class AssetmgrClientType(StrEnum):
    """资产管理客户类型类型

    Attributes:
        Person: 个人资管客户
        Organ: 单位资管客户
        SpecialOrgan: 特殊单位资管客户
    """

    Person = "1"  # 个人资管客户
    Organ = "2"  # 单位资管客户
    SpecialOrgan = "4"  # 特殊单位资管客户


class AssetmgrType(StrEnum):
    """投资类型类型

    Attributes:
        Futures: 期货类
        SpecialOrgan: 综合类
    """

    Futures = "3"  # 期货类
    SpecialOrgan = "4"  # 综合类


class CheckInstrType(StrEnum):
    """合约比较类型类型

    Attributes:
        HasExch: 合约交易所不存在
        HasATP: 合约本系统不存在
        HasDiff: 合约比较不一致
    """

    HasExch = "0"  # 合约交易所不存在
    HasATP = "1"  # 合约本系统不存在
    HasDiff = "2"  # 合约比较不一致


class DeliveryType(StrEnum):
    """交割类型类型

    Attributes:
        HandDeliv: 手工交割
        PersonDeliv: 到期交割
    """

    HandDeliv = "1"  # 手工交割
    PersonDeliv = "2"  # 到期交割


class MaxMarginSideAlgorithm(StrEnum):
    """大额单边保证金算法类型

    Attributes:
        NO: 不使用大额单边保证金算法
        YES: 使用大额单边保证金算法
    """

    NO = "0"  # 不使用大额单边保证金算法
    YES = "1"  # 使用大额单边保证金算法


class DAClientType(StrEnum):
    """资产管理客户类型类型

    Attributes:
        Person: 自然人
        Company: 法人
        Other: 其他
    """

    Person = "0"  # 自然人
    Company = "1"  # 法人
    Other = "2"  # 其他


class UOAAssetmgrType(StrEnum):
    """投资类型类型

    Attributes:
        Futures: 期货类
        SpecialOrgan: 综合类
    """

    Futures = "1"  # 期货类
    SpecialOrgan = "2"  # 综合类


class DirectionEn(StrEnum):
    """买卖方向类型

    Attributes:
        Buy: Buy
        Sell: Sell
    """

    Buy = "0"  # Buy
    Sell = "1"  # Sell

class OffsetFlagEn(StrEnum):
    """开平标志类型

    Attributes:
        Open: Position Opening
        Close: Position Closing
        ForceClose: Forced Liquidation
        CloseToday: Close Today
        CloseYesterday: Close Prev
        ForceOff: Forced Reduction
        LocalForceClose: Local Forced Liquidation
    """
    Open = "0" # Position Opening
    Close = "1" # Position Closing
    ForceClose = "2" # Forced Liquidation
    CloseToday = "3" # Close Today
    CloseYesterday = "4" # Close Prev
    ForceOff = "5"  # Forced Reduction
    LocalForceClose = "6" # Local Forced Liquidation




class HedgeFlagEn(StrEnum):
    """投机套保标志类型

    Attributes:
        Speculation: Speculation
        Arbitrage: Arbitrage
        Hedge: Hedge
    """

    Speculation = "1"  # Speculation
    Arbitrage = "2"  # Arbitrage
    Hedge = "3"  # Hedge


class FundIOTypeEn(StrEnum):
    """出入金类型类型

    Attributes:
        FundIO: Deposit
        Transfer: Bank-Futures Transfer
        SwapCurrency: Bank-Futures FX Exchange
    """

    FundIO = "1"  # Deposit
    Transfer = "2"  # Bank-Futures Transfer
    SwapCurrency = "3"  # Bank-Futures FX Exchange


class FundTypeEn(StrEnum):
    """资金类型类型

    Attributes:
        Deposit: 银行存款
        ItemFund: Payment
        Company: Brokerage Adj
        InnerTransfer: Internal Transfer
    """

    Deposit = "1"  # 银行存款
    ItemFund = "2"  # Payment
    Company = "3"  # Brokerage Adj
    InnerTransfer = "4"  # Internal Transfer


class FundDirectionEn(StrEnum):
    """出入金方向类型

    Attributes:
        In: Deposit
        Out: Withdrawal
    """

    In = "1"  # Deposit
    Out = "2"  # Withdrawal


class FundMortDirectionEn(StrEnum):
    """货币质押方向类型

    Attributes:
        In: Pledge
        Out: Redemption
    """

    In = "1"  # Pledge
    Out = "2"  # Redemption


class ApplyType(StrEnum):
    """中金所期权放弃执行申请类型类型

    Attributes:
        NotStrikeNum: 不执行数量
    """

    NotStrikeNum = "4"  # 不执行数量


class GiveUpDataSource(StrEnum):
    """放弃执行申请数据来源类型

    Attributes:
        Gen: 系统生成
        Hand: 手工添加
    """

    Gen = "0"  # 系统生成
    Hand = "1"  # 手工添加











class OptionRoyaltyPriceType(StrEnum):
    """期权权利金价格类型类型

    Attributes:
        PreSettlementPrice: 昨结算价
        OpenPrice: 开仓价
        MaxPreSettlementPrice: 最新价与昨结算价较大值
    """

    PreSettlementPrice = "1"  # 昨结算价
    OpenPrice = "4"  # 开仓价
    MaxPreSettlementPrice = "5"  # 最新价与昨结算价较大值


class BalanceAlgorithm(StrEnum):
    """权益算法类型

    Attributes:
        Default: 不计算期权市值盈亏
        IncludeOptValLost: 计算期权市值亏损
    """

    Default = "1"  # 不计算期权市值盈亏
    IncludeOptValLost = "2"  # 计算期权市值亏损


class ActionType(StrEnum):
    """执行类型类型

    Attributes:
        Exec: 执行
        Abandon: 放弃
    """

    Exec = "1"  # 执行
    Abandon = "2"  # 放弃


class ForQuoteStatus(StrEnum):
    """询价状态类型

    Attributes:
        Submitted: 已经提交
        Accepted: 已经接受
        Rejected: 已经被拒绝
    """

    Submitted = "a"  # 已经提交
    Accepted = "b"  # 已经接受
    Rejected = "c"  # 已经被拒绝


class ValueMethod(StrEnum):
    """取值方式类型

    Attributes:
        Absolute: 按绝对值
        Ratio: 按比率
    """

    Absolute = "0"  # 按绝对值
    Ratio = "1"  # 按比率


class ExecOrderPositionFlag(StrEnum):
    """期权行权后是否保留期货头寸的标记类型

    Attributes:
        Reserve: 保留
        UnReserve: 不保留
    """

    Reserve = "0"  # 保留
    UnReserve = "1"  # 不保留


class ExecOrderCloseFlag(StrEnum):
    """期权行权后生成的头寸是否自动平仓类型

    Attributes:
        AutoClose: 自动平仓
        NotToClose: 免于自动平仓
    """

    AutoClose = "0"  # 自动平仓
    NotToClose = "1"  # 免于自动平仓


class ProductType(StrEnum):
    """产品类型类型

    Attributes:
        Futures: 期货
        Options: 期权
    """

    Futures = "1"  # 期货
    Options = "2"  # 期权


class CZCEUploadFileName(StrEnum):
    """郑商所结算文件名类型
    """

    O = "O"  # ^\d{8}*zz*\d{4}
    T = "T"  # ^\d{8}成交表
    P = "P"  # ^\d{8}单腿持仓表new
    N = "N"  # ^\d{8}非平仓了结表
    L = "L"  # ^\d{8}平仓表
    F = "F"  # ^\d{8}资金表
    C = "C"  # ^\d{8}组合持仓表
    M = "M"  # ^\d{8}保证金参数表


class DCEUploadFileName(StrEnum):
    """大商所结算文件名类型
    """

    O = "O"  # ^\d{8}*dl*\d{3}
    T = "T"  # ^\d{8}_成交表
    P = "P"  # ^\d{8}_持仓表
    F = "F"  # ^\d{8}_资金结算表
    C = "C"  # ^\d{8}_优惠组合持仓明细表
    D = "D"  # ^\d{8}_持仓明细表
    M = "M"  # ^\d{8}_保证金参数表
    S = "S"  # ^\d{8}_期权执行表


class SHFEUploadFileName(StrEnum):
    """上期所结算文件名类型
    """

    O = "O"  # ^\d{4}*\d{8}*\d{8}_DailyFundChg
    T = "T"  # ^\d{4}*\d{8}*\d{8}_Trade
    P = "P"  # ^\d{4}*\d{8}*\d{8}_SettlementDetail
    F = "F"  # ^\d{4}*\d{8}*\d{8}_Capital


class CFFEXUploadFileName(StrEnum):
    """中金所结算文件名类型
    """

    T = "T"  # ^\d{4}*SG\d{1}*\d{8}_\d{1}_Trade
    P = "P"  # ^\d{4}*SG\d{1}*\d{8}_\d{1}_SettlementDetail
    F = "F"  # ^\d{4}*SG\d{1}*\d{8}_\d{1}_Capital
    S = "S"  # ^\d{4}*SG\d{1}*\d{8}_\d{1}_OptionExec


class CombDirection(StrEnum):
    """组合指令方向类型

    Attributes:
        Comb: 申请组合
        UnComb: 申请拆分
        DelComb: 操作员删组合单
    """

    Comb = "0"  # 申请组合
    UnComb = "1"  # 申请拆分
    DelComb = "2"  # 操作员删组合单


class StrikeOffsetType(StrEnum):
    """行权偏移类型类型

    Attributes:
        RealValue: 实值额
        ProfitValue: 盈利额
        RealRatio: 实值比例
        ProfitRatio: 盈利比例
    """

    RealValue = "1"  # 实值额
    ProfitValue = "2"  # 盈利额
    RealRatio = "3"  # 实值比例
    ProfitRatio = "4"  # 盈利比例


class ReserveOpenAccStas(StrEnum):
    """预约开户状态类型

    Attributes:
        Processing: 等待处理中
        Cancelled: 已撤销
        Opened: 已开户
        Invalid: 无效请求
    """

    Processing = "0"  # 等待处理中
    Cancelled = "1"  # 已撤销
    Opened = "2"  # 已开户
    Invalid = "3"  # 无效请求


class WeakPasswordSource(StrEnum):
    """弱密码来源类型

    Attributes:
        Lib: 弱密码库
        Manual: 手工录入
    """

    Lib = "1"  # 弱密码库
    Manual = "2"  # 手工录入


class OptSelfCloseFlag(StrEnum):
    """期权行权的头寸是否自对冲类型

    Attributes:
        CloseSelfOptionPosition: 自对冲期权仓位
        ReserveOptionPosition: 保留期权仓位
        SellCloseSelfFuturePosition: 自对冲卖方履约后的期货仓位
        ReserveFuturePosition: 保留卖方履约后的期货仓位
    """

    CloseSelfOptionPosition = "1"  # 自对冲期权仓位
    ReserveOptionPosition = "2"  # 保留期权仓位
    SellCloseSelfFuturePosition = "3"  # 自对冲卖方履约后的期货仓位
    ReserveFuturePosition = "4"  # 保留卖方履约后的期货仓位


class BizType(StrEnum):
    """业务类型类型

    Attributes:
        Future: 期货
        Stock: 证券
    """

    Future = "1"  # 期货
    Stock = "2"  # 证券


class AppType(StrEnum):
    """用户App类型类型

    Attributes:
        Investor: 直连的投资者
        InvestorRelay: 为每个投资者都创建连接的中继
        OperatorRelay: 所有投资者共享一个操作员连接的中继
        UnKnown: 未知
    """

    Investor = "1"  # 直连的投资者
    InvestorRelay = "2"  # 为每个投资者都创建连接的中继
    OperatorRelay = "3"  # 所有投资者共享一个操作员连接的中继
    UnKnown = "4"  # 未知


class ResponseValue(StrEnum):
    """应答类型类型

    Attributes:
        Right: 检查成功
        Refuse: 检查失败
    """

    Right = "0"  # 检查成功
    Refuse = "1"  # 检查失败


class OTCTradeType(StrEnum):
    """OTC成交类型类型

    Attributes:
        Block: 大宗交易
        EFP: 期转现
    """

    Block = "0"  # 大宗交易
    EFP = "1"  # 期转现


class MatchType(StrEnum):
    """期现风险匹配方式类型

    Attributes:
        DV01: 基点价值
        ParValue: 面值
    """

    DV01 = "1"  # 基点价值
    ParValue = "2"  # 面值


class AuthType(StrEnum):
    """用户终端认证方式类型

    Attributes:
        WHITE: 白名单校验
        BLACK: 黑名单校验
    """

    WHITE = "0"  # 白名单校验
    BLACK = "1"  # 黑名单校验


class ClassType(StrEnum):
    """合约分类方式类型

    Attributes:
        ALL: 所有合约
        FUTURE: 期货、即期、期转现、Tas、金属指数合约
        OPTION: 期货、现货期权合约
        COMB: 组合合约
    """

    ALL = "0"  # 所有合约
    FUTURE = "1"  # 期货、即期、期转现、Tas、金属指数合约
    OPTION = "2"  # 期货、现货期权合约
    COMB = "3"  # 组合合约





class ProductStatus(StrEnum):
    """产品状态类型

    Attributes:
        tradeable: 可交易
        untradeable: 不可交易
    """

    tradeable = "1"  # 可交易
    untradeable = "2"  # 不可交易


class SyncDeltaStatus(StrEnum):
    """追平状态类型

    Attributes:
        Readable: 交易可读
        Reading: 交易在读
        ReadEnd: 交易读取完成
        OptErr: 追平失败 交易本地状态结算不存在
    """

    Readable = "1"  # 交易可读
    Reading = "2"  # 交易在读
    ReadEnd = "3"  # 交易读取完成
    OptErr = "e"  # 追平失败 交易本地状态结算不存在


class ActionDirection(StrEnum):
    """操作标志类型

    Attributes:
        Add: 增加
        Del: 删除
        Upd: 更新
    """

    Add = "1"  # 增加
    Del = "2"  # 删除
    Upd = "3"  # 更新


class OrderCancelAlg(StrEnum):
    """撤单时选择席位算法类型

    Attributes:
        Balance: 轮询席位撤单
        OrigFirst: 优先原报单席位撤单
    """

    Balance = "1"  # 轮询席位撤单
    OrigFirst = "2"  # 优先原报单席位撤单


class OpenLimitControlLevel(StrEnum):
    """开仓量限制粒度类型

    Attributes:
        NONE: 不控制
        Product: 产品级别
        Inst: 合约级别
    """

    NONE = "0"  # 不控制
    Product = "1"  # 产品级别
    Inst = "2"  # 合约级别


class OrderFreqControlLevel(StrEnum):
    """报单频率控制粒度类型

    Attributes:
        NONE: 不控制
        Product: 产品级别
        Inst: 合约级别
    """

    NONE = "0"  # 不控制
    Product = "1"  # 产品级别
    Inst = "2"  # 合约级别


class EnumBool(StrEnum):
    """枚举bool类型类型

    Attributes:
        _False: false
        _True: true
    """

    _False = "0"  # false
    _True = "1"  # true


class TimeRange(StrEnum):
    """期货合约阶段标识类型

    Attributes:
        USUAL: 一般月份
        FNSP: 交割月前一个月上半月
        BNSP: 交割月前一个月下半月
        SPOT: 交割月份
    """

    USUAL = "1"  # 一般月份
    FNSP = "2"  # 交割月前一个月上半月
    BNSP = "3"  # 交割月前一个月下半月
    SPOT = "4"  # 交割月份


class Portfolio(StrEnum):
    """新型组保算法类型

    Attributes:
        NONE: 不使用新型组保算法
        SPBM: SPBM算法
        RULE: RULE算法
        SPMM: SPMM算法
        RCAMS: RCAMS算法
    """

    NONE = "0"  # 不使用新型组保算法
    SPBM = "1"  # SPBM算法
    RULE = "2"  # RULE算法
    SPMM = "3"  # SPMM算法
    RCAMS = "4"  # RCAMS算法


class WithDrawParamID(StrEnum):
    """可提参数代码类型

    Attributes:
        CashIn: 权利金收支是否可提 1 代表可提 0 不可提
    """

    CashIn = "C"  # 权利金收支是否可提 1 代表可提 0 不可提


class InvestTradingRight(StrEnum):
    """投资者交易权限类型

    Attributes:
        CloseOnly: 只能平仓
        Forbidden: 不能交易
    """

    CloseOnly = "1"  # 只能平仓
    Forbidden = "2"  # 不能交易


class InstMarginCalID(StrEnum):
    """SPMM合约保证金算法类型

    Attributes:
        BothSide: 标准算法收取双边
        MMSA: 单向大边
        SPMM: 新组保SPMM
    """

    BothSide = "1"  # 标准算法收取双边
    MMSA = "2"  # 单向大边
    SPMM = "3"  # 新组保SPMM


class RCAMSCombinationType(StrEnum):
    """RCAMS组合类型类型

    Attributes:
        BUC: 牛市看涨价差组合
        BEC: 熊市看涨价差组合
        BEP: 熊市看跌价差组合
        BUP: 牛市看跌价差组合
        CAS: 日历价差组合
    """

    BUC = "0"  # 牛市看涨价差组合
    BEC = "1"  # 熊市看涨价差组合
    BEP = "2"  # 熊市看跌价差组合
    BUP = "3"  # 牛市看跌价差组合
    CAS = "4"  # 日历价差组合


class PortfType(StrEnum):
    """新组保算法启用类型类型

    Attributes:
        NONE: 使用初版交易所算法
        SPBM_AddOnHedge: SPBM算法V1.1.0_附加保证金调整
    """

    NONE = "0"  # 使用初版交易所算法
    SPBM_AddOnHedge = "1"  # SPBM算法V1.1.0_附加保证金调整


class InstrumentClass(StrEnum):
    """合约类型类型

    Attributes:
        Usual: 一般月份合约
        Delivery: 临近交割合约
        NonComb: 非组合合约
    """

    Usual = "1"  # 一般月份合约
    Delivery = "2"  # 临近交割合约
    NonComb = "3"  # 非组合合约


class ProdChangeFlag(StrEnum):
    """品种记录改变状态类型

    Attributes:
        NONE: 持仓量和冻结量均无变化
        OnlyFrozen: 持仓量无变化，冻结量有变化
        PositionChange: 持仓量有变化
    """

    NONE = "0"  # 持仓量和冻结量均无变化
    OnlyFrozen = "1"  # 持仓量无变化，冻结量有变化
    PositionChange = "2"  # 持仓量有变化


class PwdRcdSrc(StrEnum):
    """历史密码来源类型

    Attributes:
        Init: 来源于Sync初始化数据
        Sync: 来源于实时上场数据
        UserUpd: 来源于用户修改
        SuperUserUpd: 来源于超户修改，很可能来自主席同步数据
    """

    Init = "0"  # 来源于Sync初始化数据
    Sync = "1"  # 来源于实时上场数据
    UserUpd = "2"  # 来源于用户修改
    SuperUserUpd = "3"  # 来源于超户修改，很可能来自主席同步数据


class ResumeType(IntEnum):
    """私有/公共流重传方式

    Attributes:
        RESTART: 从本交易日开始重传
        RESUME: 从上次收到的续传
        QUICK: 只传送登录后公共流的内容
        NONE: 取消订阅公共流
    """

    RESTART = 0  # 从本交易日开始重传
    RESUME = 1  # 从上次收到的续传
    QUICK = 2  # 只传送登录后公共流的内容
    NONE = 3  # 取消订阅公共流
