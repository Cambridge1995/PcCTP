"""
Trade
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class TradingAccount(CapsuleStruct):
    """资金账户"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("PreMortgage", ctypes.c_double),            # 上次质押金额
            ("PreCredit", ctypes.c_double),              # 上次信用额度
            ("PreDeposit", ctypes.c_double),             # 上次存款额
            ("PreBalance", ctypes.c_double),             # 上次结算准备金
            ("PreMargin", ctypes.c_double),              # 上次占用的保证金
            ("InterestBase", ctypes.c_double),           # 利息基数
            ("Interest", ctypes.c_double),               # 利息收入
            ("Deposit", ctypes.c_double),                # 入金金额
            ("Withdraw", ctypes.c_double),               # 出金金额
            ("FrozenMargin", ctypes.c_double),           # 冻结的保证金
            ("FrozenCash", ctypes.c_double),             # 冻结的资金
            ("FrozenCommission", ctypes.c_double),       # 冻结的手续费
            ("CurrMargin", ctypes.c_double),             # 当前保证金总额
            ("CashIn", ctypes.c_double),                 # 资金差额
            ("Commission", ctypes.c_double),             # 手续费
            ("CloseProfit", ctypes.c_double),            # 平仓盈亏
            ("PositionProfit", ctypes.c_double),         # 持仓盈亏
            ("Balance", ctypes.c_double),                # 期货结算准备金
            ("Available", ctypes.c_double),              # 可用资金
            ("WithdrawQuota", ctypes.c_double),          # 可取资金
            ("Reserve", ctypes.c_double),                # 基本准备金
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("Credit", ctypes.c_double),                 # 信用额度
            ("Mortgage", ctypes.c_double),               # 质押金额
            ("ExchangeMargin", ctypes.c_double),         # 交易所保证金
            ("DeliveryMargin", ctypes.c_double),         # 投资者交割保证金
            ("ExchangeDeliveryMargin", ctypes.c_double), # 交易所交割保证金
            ("ReserveBalance", ctypes.c_double),         # 保底期货结算准备金
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
            ("PreFundMortgageIn", ctypes.c_double),      # 上次货币质入金额
            ("PreFundMortgageOut", ctypes.c_double),     # 上次货币质出金额
            ("FundMortgageIn", ctypes.c_double),         # 货币质入金额
            ("FundMortgageOut", ctypes.c_double),        # 货币质出金额
            ("FundMortgageAvailable", ctypes.c_double), # 货币质押余额
            ("MortgageableFund", ctypes.c_double),       # 可质押货币金额
            ("SpecProductMargin", ctypes.c_double),      # 特殊产品占用保证金
            ("SpecProductFrozenMargin", ctypes.c_double), # 特殊产品冻结保证金
            ("SpecProductCommission", ctypes.c_double),  # 特殊产品手续费
            ("SpecProductFrozenCommission", ctypes.c_double), # 特殊产品冻结手续费
            ("SpecProductPositionProfit", ctypes.c_double), # 特殊产品持仓盈亏
            ("SpecProductCloseProfit", ctypes.c_double), # 特殊产品平仓盈亏
            ("SpecProductPositionProfitByAlg", ctypes.c_double), # 根据持仓盈亏算法计算的特殊产品持仓盈亏
            ("SpecProductExchangeMargin", ctypes.c_double), # 特殊产品交易所保证金
            ("BizType", ctypes.c_char),                  # 业务类型
            ("FrozenSwap", ctypes.c_double),             # 延时换汇冻结金额
            ("RemainSwap", ctypes.c_double),             # 剩余换汇额度
            ("OptionValue", ctypes.c_double),            # 期权市值
        ]

    _capsule_name = "TradingAccount"

    _field_mappings = {
        "broker_id": "BrokerID",
        "account_id": "AccountID",
        "pre_mortgage": "PreMortgage",
        "pre_credit": "PreCredit",
        "pre_deposit": "PreDeposit",
        "pre_balance": "PreBalance",
        "pre_margin": "PreMargin",
        "interest_base": "InterestBase",
        "interest": "Interest",
        "deposit": "Deposit",
        "withdraw": "Withdraw",
        "frozen_margin": "FrozenMargin",
        "frozen_cash": "FrozenCash",
        "frozen_commission": "FrozenCommission",
        "curr_margin": "CurrMargin",
        "cash_in": "CashIn",
        "commission": "Commission",
        "close_profit": "CloseProfit",
        "position_profit": "PositionProfit",
        "balance": "Balance",
        "available": "Available",
        "withdraw_quota": "WithdrawQuota",
        "reserve": "Reserve",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "credit": "Credit",
        "mortgage": "Mortgage",
        "exchange_margin": "ExchangeMargin",
        "delivery_margin": "DeliveryMargin",
        "exchange_delivery_margin": "ExchangeDeliveryMargin",
        "reserve_balance": "ReserveBalance",
        "currency_id": "CurrencyID",
        "pre_fund_mortgage_in": "PreFundMortgageIn",
        "pre_fund_mortgage_out": "PreFundMortgageOut",
        "fund_mortgage_in": "FundMortgageIn",
        "fund_mortgage_out": "FundMortgageOut",
        "fund_mortgage_available": "FundMortgageAvailable",
        "mortgageable_fund": "MortgageableFund",
        "spec_product_margin": "SpecProductMargin",
        "spec_product_frozen_margin": "SpecProductFrozenMargin",
        "spec_product_commission": "SpecProductCommission",
        "spec_product_frozen_commission": "SpecProductFrozenCommission",
        "spec_product_position_profit": "SpecProductPositionProfit",
        "spec_product_close_profit": "SpecProductCloseProfit",
        "spec_product_position_profit_by_alg": "SpecProductPositionProfitByAlg",
        "spec_product_exchange_margin": "SpecProductExchangeMargin",
        "biz_type": "BizType",
        "frozen_swap": "FrozenSwap",
        "remain_swap": "RemainSwap",
        "option_value": "OptionValue",
    }

    @property
    def broker_id(self) -> str:
        """经纪公司代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置经纪公司代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def account_id(self) -> str:
        """资金账号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置资金账号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AccountID = encoded
        self._cache['account_id'] = value

    # 数值属性（零拷贝）
    @property
    def pre_mortgage(self) -> float:
        """上次质押"""
        return self._struct.PreMortgage

    @pre_mortgage.setter
    def pre_mortgage(self, value: float):
        """设置上次质押"""
        self._struct.PreMortgage = value

    @property
    def pre_credit(self) -> float:
        """上次信用额度"""
        return self._struct.PreCredit

    @pre_credit.setter
    def pre_credit(self, value: float):
        """设置上次信用额度"""
        self._struct.PreCredit = value

    @property
    def pre_deposit(self) -> float:
        """上次存款"""
        return self._struct.PreDeposit

    @pre_deposit.setter
    def pre_deposit(self, value: float):
        """设置上次存款"""
        self._struct.PreDeposit = value

    @property
    def pre_balance(self) -> float:
        """上次结算准备金"""
        return self._struct.PreBalance

    @pre_balance.setter
    def pre_balance(self, value: float):
        """设置上次结算准备金"""
        self._struct.PreBalance = value

    @property
    def pre_margin(self) -> float:
        """上次占用的保证金"""
        return self._struct.PreMargin

    @pre_margin.setter
    def pre_margin(self, value: float):
        """设置上次占用的保证金"""
        self._struct.PreMargin = value

    @property
    def interest_base(self) -> float:
        """利息基数"""
        return self._struct.InterestBase

    @interest_base.setter
    def interest_base(self, value: float):
        """设置利息基数"""
        self._struct.InterestBase = value

    @property
    def interest(self) -> float:
        """利息收入"""
        return self._struct.Interest

    @interest.setter
    def interest(self, value: float):
        """设置利息收入"""
        self._struct.Interest = value

    @property
    def deposit(self) -> float:
        """入金"""
        return self._struct.Deposit

    @deposit.setter
    def deposit(self, value: float):
        """设置入金"""
        self._struct.Deposit = value

    @property
    def withdraw(self) -> float:
        """出金"""
        return self._struct.Withdraw

    @withdraw.setter
    def withdraw(self, value: float):
        """设置出金"""
        self._struct.Withdraw = value

    @property
    def frozen_margin(self) -> float:
        """冻结的保证金"""
        return self._struct.FrozenMargin

    @frozen_margin.setter
    def frozen_margin(self, value: float):
        """设置冻结的保证金"""
        self._struct.FrozenMargin = value

    @property
    def frozen_cash(self) -> float:
        """冻结的资金"""
        return self._struct.FrozenCash

    @frozen_cash.setter
    def frozen_cash(self, value: float):
        """设置冻结的资金"""
        self._struct.FrozenCash = value

    @property
    def frozen_commission(self) -> float:
        """冻结的手续费"""
        return self._struct.FrozenCommission

    @frozen_commission.setter
    def frozen_commission(self, value: float):
        """设置冻结的手续费"""
        self._struct.FrozenCommission = value

    @property
    def curr_margin(self) -> float:
        """当前保证金总额"""
        return self._struct.CurrMargin

    @curr_margin.setter
    def curr_margin(self, value: float):
        """设置当前保证金总额"""
        self._struct.CurrMargin = value

    @property
    def cash_in(self) -> float:
        """资金差额"""
        return self._struct.CashIn

    @cash_in.setter
    def cash_in(self, value: float):
        """设置资金差额"""
        self._struct.CashIn = value

    @property
    def commission(self) -> float:
        """手续费"""
        return self._struct.Commission

    @commission.setter
    def commission(self, value: float):
        """设置手续费"""
        self._struct.Commission = value

    @property
    def close_profit(self) -> float:
        """平仓盈亏"""
        return self._struct.CloseProfit

    @close_profit.setter
    def close_profit(self, value: float):
        """设置平仓盈亏"""
        self._struct.CloseProfit = value

    @property
    def position_profit(self) -> float:
        """持仓盈亏"""
        return self._struct.PositionProfit

    @position_profit.setter
    def position_profit(self, value: float):
        """设置持仓盈亏"""
        self._struct.PositionProfit = value

    @property
    def balance(self) -> float:
        """结算准备金"""
        return self._struct.Balance

    @balance.setter
    def balance(self, value: float):
        """设置结算准备金"""
        self._struct.Balance = value

    @property
    def available(self) -> float:
        """可取资金"""
        return self._struct.Available

    @available.setter
    def available(self, value: float):
        """设置可取资金"""
        self._struct.Available = value

    @property
    def withdraw_quota(self) -> float:
        """可取货币资金"""
        return self._struct.WithdrawQuota

    @withdraw_quota.setter
    def withdraw_quota(self, value: float):
        """设置可取货币资金"""
        self._struct.WithdrawQuota = value

    @property
    def reserve(self) -> float:
        """保留"""
        return self._struct.Reserve

    @reserve.setter
    def reserve(self, value: float):
        """设置保留"""
        self._struct.Reserve = value

    @property
    def trading_day(self) -> str:
        """交易日"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易日"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    @property
    def settlement_id(self) -> int:
        """结算编号"""
        return self._struct.SettlementID

    @settlement_id.setter
    def settlement_id(self, value: int):
        """设置结算编号"""
        self._struct.SettlementID = value

    @property
    def credit(self) -> float:
        """信用额度"""
        return self._struct.Credit

    @credit.setter
    def credit(self, value: float):
        """设置信用额度"""
        self._struct.Credit = value

    @property
    def mortgage(self) -> float:
        """质押金额"""
        return self._struct.Mortgage

    @mortgage.setter
    def mortgage(self, value: float):
        """设置质押金额"""
        self._struct.Mortgage = value

    @property
    def exchange_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchangeMargin

    @exchange_margin.setter
    def exchange_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchangeMargin = value

    @property
    def delivery_margin(self) -> float:
        """交割保证金"""
        return self._struct.DeliveryMargin

    @delivery_margin.setter
    def delivery_margin(self, value: float):
        """设置交割保证金"""
        self._struct.DeliveryMargin = value

    @property
    def exchange_delivery_margin(self) -> float:
        """交易所交割保证金"""
        return self._struct.ExchangeDeliveryMargin

    @exchange_delivery_margin.setter
    def exchange_delivery_margin(self, value: float):
        """设置交易所交割保证金"""
        self._struct.ExchangeDeliveryMargin = value

    @property
    def reserve_balance(self) -> float:
        """保底期货结算准备金"""
        return self._struct.ReserveBalance

    @reserve_balance.setter
    def reserve_balance(self, value: float):
        """设置保底期货结算准备金"""
        self._struct.ReserveBalance = value

    @property
    def currency_id(self) -> str:
        """币种代码"""
        if 'currency_id' not in self._cache:
            value = self._struct.CurrencyID.rstrip(b'\x00').decode('ascii')
            self._cache['currency_id'] = value
        return self._cache['currency_id']

    @currency_id.setter
    def currency_id(self, value: str):
        """设置币种代码"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.CurrencyID = encoded
        self._cache['currency_id'] = value

    @property
    def pre_fund_mortgage_in(self) -> float:
        """上次货币质入金额"""
        return self._struct.PreFundMortgageIn

    @pre_fund_mortgage_in.setter
    def pre_fund_mortgage_in(self, value: float):
        """设置上次货币质入金额"""
        self._struct.PreFundMortgageIn = value

    @property
    def pre_fund_mortgage_out(self) -> float:
        """上次货币质出金额"""
        return self._struct.PreFundMortgageOut

    @pre_fund_mortgage_out.setter
    def pre_fund_mortgage_out(self, value: float):
        """设置上次货币质出金额"""
        self._struct.PreFundMortgageOut = value

    @property
    def fund_mortgage_in(self) -> float:
        """货币质入金额"""
        return self._struct.FundMortgageIn

    @fund_mortgage_in.setter
    def fund_mortgage_in(self, value: float):
        """设置货币质入金额"""
        self._struct.FundMortgageIn = value

    @property
    def fund_mortgage_out(self) -> float:
        """货币质出金额"""
        return self._struct.FundMortgageOut

    @fund_mortgage_out.setter
    def fund_mortgage_out(self, value: float):
        """设置货币质出金额"""
        self._struct.FundMortgageOut = value

    @property
    def fund_mortgage_available(self) -> float:
        """货币质押余额"""
        return self._struct.FundMortgageAvailable

    @fund_mortgage_available.setter
    def fund_mortgage_available(self, value: float):
        """设置货币质押余额"""
        self._struct.FundMortgageAvailable = value

    @property
    def mortgageable_fund(self) -> float:
        """可质押货币金额"""
        return self._struct.MortgageableFund

    @mortgageable_fund.setter
    def mortgageable_fund(self, value: float):
        """设置可质押货币金额"""
        self._struct.MortgageableFund = value

    @property
    def spec_product_margin(self) -> float:
        """特殊产品占用保证金"""
        return self._struct.SpecProductMargin

    @spec_product_margin.setter
    def spec_product_margin(self, value: float):
        """设置特殊产品占用保证金"""
        self._struct.SpecProductMargin = value

    @property
    def spec_product_frozen_margin(self) -> float:
        """特殊产品冻结保证金"""
        return self._struct.SpecProductFrozenMargin

    @spec_product_frozen_margin.setter
    def spec_product_frozen_margin(self, value: float):
        """设置特殊产品冻结保证金"""
        self._struct.SpecProductFrozenMargin = value

    @property
    def spec_product_commission(self) -> float:
        """特殊产品手续费"""
        return self._struct.SpecProductCommission

    @spec_product_commission.setter
    def spec_product_commission(self, value: float):
        """设置特殊产品手续费"""
        self._struct.SpecProductCommission = value

    @property
    def spec_product_frozen_commission(self) -> float:
        """特殊产品冻结手续费"""
        return self._struct.SpecProductFrozenCommission

    @spec_product_frozen_commission.setter
    def spec_product_frozen_commission(self, value: float):
        """设置特殊产品冻结手续费"""
        self._struct.SpecProductFrozenCommission = value

    @property
    def spec_product_position_profit(self) -> float:
        """特殊产品持仓盈亏"""
        return self._struct.SpecProductPositionProfit

    @spec_product_position_profit.setter
    def spec_product_position_profit(self, value: float):
        """设置特殊产品持仓盈亏"""
        self._struct.SpecProductPositionProfit = value

    @property
    def spec_product_close_profit(self) -> float:
        """特殊产品平仓盈亏"""
        return self._struct.SpecProductCloseProfit

    @spec_product_close_profit.setter
    def spec_product_close_profit(self, value: float):
        """设置特殊产品平仓盈亏"""
        self._struct.SpecProductCloseProfit = value

    @property
    def spec_product_position_profit_by_alg(self) -> float:
        """根据持仓盈亏算法计算的特殊产品持仓盈亏"""
        return self._struct.SpecProductPositionProfitByAlg

    @spec_product_position_profit_by_alg.setter
    def spec_product_position_profit_by_alg(self, value: float):
        """设置根据持仓盈亏算法计算的特殊产品持仓盈亏"""
        self._struct.SpecProductPositionProfitByAlg = value

    @property
    def spec_product_exchange_margin(self) -> float:
        """特殊产品交易所保证金"""
        return self._struct.SpecProductExchangeMargin

    @spec_product_exchange_margin.setter
    def spec_product_exchange_margin(self, value: float):
        """设置特殊产品交易所保证金"""
        self._struct.SpecProductExchangeMargin = value

    @property
    def biz_type(self) -> str:
        """业务类型"""
        if 'biz_type' not in self._cache:
            value = self._struct.BizType.decode('ascii')
            self._cache['biz_type'] = value
        return self._cache['biz_type']

    @biz_type.setter
    def biz_type(self, value: str):
        """设置业务类型"""
        self._struct.BizType = value.encode('ascii')[0]
        self._cache['biz_type'] = value

    @property
    def frozen_swap(self) -> float:
        """延时换汇冻结金额"""
        return self._struct.FrozenSwap

    @frozen_swap.setter
    def frozen_swap(self, value: float):
        """设置延时换汇冻结金额"""
        self._struct.FrozenSwap = value

    @property
    def remain_swap(self) -> float:
        """剩余换汇额度"""
        return self._struct.RemainSwap

    @remain_swap.setter
    def remain_swap(self, value: float):
        """设置剩余换汇额度"""
        self._struct.RemainSwap = value

    @property
    def option_value(self) -> float:
        """期权市值"""
        return self._struct.OptionValue

    @option_value.setter
    def option_value(self, value: float):
        """设置期权市值"""
        self._struct.OptionValue = value


# =============================================================================
# Investor - 投资者
# =============================================================================


class Investor(CapsuleStruct):
    """投资者"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorGroupID", ctypes.c_char * 13),     # 投资者分组代码
            ("InvestorName", ctypes.c_char * 81),        # 投资者名称
            ("IdentifiedCardType", ctypes.c_char),       # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),    # 证件号码
            ("IsActive", ctypes.c_int),                  # 是否活跃
            ("Telephone", ctypes.c_char * 41),           # 联系电话
            ("Address", ctypes.c_char * 101),            # 通讯地址
            ("OpenDate", ctypes.c_char * 9),             # 开户日期
            ("Mobile", ctypes.c_char * 41),              # 手机
            ("CommModelID", ctypes.c_char * 11),         # 手续费率模板代码
            ("MarginModelID", ctypes.c_char * 11),       # 保证金率模板代码
            ("IsOrderFreq", ctypes.c_int),               # 是否频率控制
            ("IsOpenVolLimit", ctypes.c_int),            # 是否开仓限制
        ]

    _capsule_name = "Investor"

    _field_mappings = {
        "investor_id": "InvestorID",
        "broker_id": "BrokerID",
        "investor_group_id": "InvestorGroupID",
        "investor_name": "InvestorName",
        "identified_card_type": "IdentifiedCardType",
        "identified_card_no": "IdentifiedCardNo",
        "is_active": "IsActive",
        "telephone": "Telephone",
        "address": "Address",
        "open_date": "OpenDate",
        "mobile": "Mobile",
        "comm_model_id": "CommModelID",
        "margin_model_id": "MarginModelID",
        "is_order_freq": "IsOrderFreq",
        "is_open_vol_limit": "IsOpenVolLimit",
    }

    @property
    def investor_id(self) -> str:
        """投资者代码"""
        if 'investor_id' not in self._cache:
            value = self._struct.InvestorID.rstrip(b'\x00').decode('ascii')
            self._cache['investor_id'] = value
        return self._cache['investor_id']

    @investor_id.setter
    def investor_id(self, value: str):
        """设置投资者代码"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.InvestorID = encoded
        self._cache['investor_id'] = value

    @property
    def broker_id(self) -> str:
        """经纪公司代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置经纪公司代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def investor_group_id(self) -> str:
        """投资者分组代码"""
        if 'investor_group_id' not in self._cache:
            value = self._struct.InvestorGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['investor_group_id'] = value
        return self._cache['investor_group_id']

    @investor_group_id.setter
    def investor_group_id(self, value: str):
        """设置投资者分组代码"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.InvestorGroupID = encoded
        self._cache['investor_group_id'] = value

    @property
    def investor_name(self) -> str:
        """投资者名称（GBK 编码）"""
        if 'investor_name' not in self._cache:
            value = self._struct.InvestorName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['investor_name'] = value
        return self._cache['investor_name']

    @investor_name.setter
    def investor_name(self, value: str):
        """设置投资者名称（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.InvestorName = encoded
        self._cache['investor_name'] = value

    @property
    def identified_card_type(self) -> str:
        """证件类型"""
        if 'identified_card_type' not in self._cache:
            value = self._struct.IdentifiedCardType.decode('ascii')
            self._cache['identified_card_type'] = value
        return self._cache['identified_card_type']

    @identified_card_type.setter
    def identified_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdentifiedCardType = value.encode('ascii')[0]
        self._cache['identified_card_type'] = value

    @property
    def identified_card_no(self) -> str:
        """证件号码"""
        if 'identified_card_no' not in self._cache:
            value = self._struct.IdentifiedCardNo.rstrip(b'\x00').decode('ascii')
            self._cache['identified_card_no'] = value
        return self._cache['identified_card_no']

    @identified_card_no.setter
    def identified_card_no(self, value: str):
        """设置证件号码"""
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.IdentifiedCardNo = encoded
        self._cache['identified_card_no'] = value

    @property
    def is_active(self) -> int:
        """是否活跃"""
        return self._struct.IsActive

    @is_active.setter
    def is_active(self, value: int):
        """设置是否活跃"""
        self._struct.IsActive = value

    @property
    def telephone(self) -> str:
        """联系电话"""
        if 'telephone' not in self._cache:
            value = self._struct.Telephone.rstrip(b'\x00').decode('ascii')
            self._cache['telephone'] = value
        return self._cache['telephone']

    @telephone.setter
    def telephone(self, value: str):
        """设置联系电话"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Telephone = encoded
        self._cache['telephone'] = value

    @property
    def address(self) -> str:
        """联系地址（GBK 编码）"""
        if 'address' not in self._cache:
            value = self._struct.Address.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['address'] = value
        return self._cache['address']

    @address.setter
    def address(self, value: str):
        """设置联系地址（GBK 编码）"""
        encoded = value.encode('gbk')[:100].ljust(101, b'\x00')
        self._struct.Address = encoded
        self._cache['address'] = value

    @property
    def open_date(self) -> str:
        """开户日期"""
        if 'open_date' not in self._cache:
            value = self._struct.OpenDate.rstrip(b'\x00').decode('ascii')
            self._cache['open_date'] = value
        return self._cache['open_date']

    @open_date.setter
    def open_date(self, value: str):
        """设置开户日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.OpenDate = encoded
        self._cache['open_date'] = value

    @property
    def mobile(self) -> str:
        """手机"""
        if 'mobile' not in self._cache:
            value = self._struct.Mobile.rstrip(b'\x00').decode('ascii')
            self._cache['mobile'] = value
        return self._cache['mobile']

    @mobile.setter
    def mobile(self, value: str):
        """设置手机"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Mobile = encoded
        self._cache['mobile'] = value

    @property
    def comm_model_id(self) -> str:
        """手续费模板代码"""
        if 'comm_model_id' not in self._cache:
            value = self._struct.CommModelID.rstrip(b'\x00').decode('ascii')
            self._cache['comm_model_id'] = value
        return self._cache['comm_model_id']

    @comm_model_id.setter
    def comm_model_id(self, value: str):
        """设置手续费率模板代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.CommModelID = encoded
        self._cache['comm_model_id'] = value

    @property
    def margin_model_id(self) -> str:
        """保证金率模板代码"""
        if 'margin_model_id' not in self._cache:
            value = self._struct.MarginModelID.rstrip(b'\x00').decode('ascii')
            self._cache['margin_model_id'] = value
        return self._cache['margin_model_id']

    @margin_model_id.setter
    def margin_model_id(self, value: str):
        """设置保证金率模板代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.MarginModelID = encoded
        self._cache['margin_model_id'] = value

    @property
    def is_order_freq(self) -> int:
        """是否频率控制"""
        return self._struct.IsOrderFreq

    @is_order_freq.setter
    def is_order_freq(self, value: int):
        """设置是否频率控制"""
        self._struct.IsOrderFreq = value

    @property
    def is_open_vol_limit(self) -> int:
        """是否开仓限制"""
        return self._struct.IsOpenVolLimit

    @is_open_vol_limit.setter
    def is_open_vol_limit(self, value: int):
        """设置是否开仓限制"""
        self._struct.IsOpenVolLimit = value


# =============================================================================
# TradingCode - 交易编码
# =============================================================================


class TradingCode(CapsuleStruct):
    """交易编码"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("IsActive", ctypes.c_int),                  # 是否活跃
            ("ClientIDType", ctypes.c_char),             # 交易编码类型
            ("BranchID", ctypes.c_char * 9),             # 营业部编号
            ("BizType", ctypes.c_char),                  # 业务类型
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
        ]

    _capsule_name = "TradingCode"

    _field_mappings = {
        "investor_id": "InvestorID",
        "broker_id": "BrokerID",
        "exchange_id": "ExchangeID",
        "client_id": "ClientID",
        "is_active": "IsActive",
        "client_id_type": "ClientIDType",
        "branch_id": "BranchID",
        "biz_type": "BizType",
        "invest_unit_id": "InvestUnitID",
    }

    @property
    def investor_id(self) -> str:
        """投资者代码"""
        if 'investor_id' not in self._cache:
            value = self._struct.InvestorID.rstrip(b'\x00').decode('ascii')
            self._cache['investor_id'] = value
        return self._cache['investor_id']

    @investor_id.setter
    def investor_id(self, value: str):
        """设置投资者代码"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.InvestorID = encoded
        self._cache['investor_id'] = value

    @property
    def broker_id(self) -> str:
        """经纪公司代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置经纪公司代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def exchange_id(self) -> str:
        """交易所代码"""
        if 'exchange_id' not in self._cache:
            value = self._struct.ExchangeID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_id'] = value
        return self._cache['exchange_id']

    @exchange_id.setter
    def exchange_id(self, value: str):
        """设置交易所代码"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ExchangeID = encoded
        self._cache['exchange_id'] = value

    @property
    def client_id(self) -> str:
        """客户代码"""
        if 'client_id' not in self._cache:
            value = self._struct.ClientID.rstrip(b'\x00').decode('ascii')
            self._cache['client_id'] = value
        return self._cache['client_id']

    @client_id.setter
    def client_id(self, value: str):
        """设置客户代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClientID = encoded
        self._cache['client_id'] = value

    @property
    def is_active(self) -> int:
        """是否活跃"""
        return self._struct.IsActive

    @is_active.setter
    def is_active(self, value: int):
        """设置是否活跃"""
        self._struct.IsActive = value

    @property
    def client_id_type(self) -> str:
        """交易编码类型"""
        if 'client_id_type' not in self._cache:
            value = self._struct.ClientIDType.decode('ascii')
            self._cache['client_id_type'] = value
        return self._cache['client_id_type']

    @client_id_type.setter
    def client_id_type(self, value: str):
        """设置交易编码类型"""
        self._struct.ClientIDType = value.encode('ascii')[0]
        self._cache['client_id_type'] = value

    @property
    def branch_id(self) -> str:
        """营业部编号"""
        if 'branch_id' not in self._cache:
            value = self._struct.BranchID.rstrip(b'\x00').decode('ascii')
            self._cache['branch_id'] = value
        return self._cache['branch_id']

    @branch_id.setter
    def branch_id(self, value: str):
        """设置营业部编号"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.BranchID = encoded
        self._cache['branch_id'] = value

    @property
    def biz_type(self) -> str:
        """业务类型"""
        if 'biz_type' not in self._cache:
            value = self._struct.BizType.decode('ascii')
            self._cache['biz_type'] = value
        return self._cache['biz_type']

    @biz_type.setter
    def biz_type(self, value: str):
        """设置业务类型"""
        self._struct.BizType = value.encode('ascii')[0]
        self._cache['biz_type'] = value

    @property
    def invest_unit_id(self) -> str:
        """投资单元代码"""
        if 'invest_unit_id' not in self._cache:
            value = self._struct.InvestUnitID.rstrip(b'\x00').decode('ascii')
            self._cache['invest_unit_id'] = value
        return self._cache['invest_unit_id']

    @invest_unit_id.setter
    def invest_unit_id(self, value: str):
        """设置投资单元代码"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.InvestUnitID = encoded
        self._cache['invest_unit_id'] = value


# =============================================================================
# InstrumentMarginRate - 合约保证金率
# =============================================================================


class InvestUnit(CapsuleStruct):
    """投资单元"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InvestorUnitName", ctypes.c_char * 81),    # 投资者单元名称
            ("InvestorGroupID", ctypes.c_char * 13),     # 投资者分组代码
            ("CommModelID", ctypes.c_char * 13),         # 手续费率模板代码
            ("MarginModelID", ctypes.c_char * 13),       # 保证金率模板代码
            ("AccountID", ctypes.c_char * 13),           # 资金账号
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
        ]

    _capsule_name = "InvestUnit"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "invest_unit_id": "InvestUnitID",
        "investor_unit_name": "InvestorUnitName",
        "investor_group_id": "InvestorGroupID",
        "comm_model_id": "CommModelID",
        "margin_model_id": "MarginModelID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
    }

    @property
    def broker_id(self) -> str:
        """经纪公司代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置经纪公司代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def investor_id(self) -> str:
        """投资者代码"""
        if 'investor_id' not in self._cache:
            value = self._struct.InvestorID.rstrip(b'\x00').decode('ascii')
            self._cache['investor_id'] = value
        return self._cache['investor_id']

    @investor_id.setter
    def investor_id(self, value: str):
        """设置投资者代码"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.InvestorID = encoded
        self._cache['investor_id'] = value

    @property
    def invest_unit_id(self) -> str:
        """投资单元代码"""
        if 'invest_unit_id' not in self._cache:
            value = self._struct.InvestUnitID.rstrip(b'\x00').decode('ascii')
            self._cache['invest_unit_id'] = value
        return self._cache['invest_unit_id']

    @invest_unit_id.setter
    def invest_unit_id(self, value: str):
        """设置投资单元代码"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.InvestUnitID = encoded
        self._cache['invest_unit_id'] = value

    @property
    def investor_unit_name(self) -> str:
        """投资者单元名称（GBK 编码）"""
        if 'investor_unit_name' not in self._cache:
            value = self._struct.InvestorUnitName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['investor_unit_name'] = value
        return self._cache['investor_unit_name']

    @investor_unit_name.setter
    def investor_unit_name(self, value: str):
        """设置投资者单元名称（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.InvestorUnitName = encoded
        self._cache['investor_unit_name'] = value

    @property
    def investor_group_id(self) -> str:
        """投资者分组代码"""
        if 'investor_group_id' not in self._cache:
            value = self._struct.InvestorGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['investor_group_id'] = value
        return self._cache['investor_group_id']

    @investor_group_id.setter
    def investor_group_id(self, value: str):
        """设置投资者分组代码"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.InvestorGroupID = encoded
        self._cache['investor_group_id'] = value

    @property
    def comm_model_id(self) -> str:
        """手续费率模板代码"""
        if 'comm_model_id' not in self._cache:
            value = self._struct.CommModelID.rstrip(b'\x00').decode('ascii')
            self._cache['comm_model_id'] = value
        return self._cache['comm_model_id']

    @comm_model_id.setter
    def comm_model_id(self, value: str):
        """设置手续费率模板代码"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.CommModelID = encoded
        self._cache['comm_model_id'] = value

    @property
    def margin_model_id(self) -> str:
        """保证金率模板代码"""
        if 'margin_model_id' not in self._cache:
            value = self._struct.MarginModelID.rstrip(b'\x00').decode('ascii')
            self._cache['margin_model_id'] = value
        return self._cache['margin_model_id']

    @margin_model_id.setter
    def margin_model_id(self, value: str):
        """设置保证金率模板代码"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.MarginModelID = encoded
        self._cache['margin_model_id'] = value

    @property
    def account_id(self) -> str:
        """资金账号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置资金账号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AccountID = encoded
        self._cache['account_id'] = value

    @property
    def currency_id(self) -> str:
        """币种代码"""
        if 'currency_id' not in self._cache:
            value = self._struct.CurrencyID.rstrip(b'\x00').decode('ascii')
            self._cache['currency_id'] = value
        return self._cache['currency_id']

    @currency_id.setter
    def currency_id(self, value: str):
        """设置币种代码"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.CurrencyID = encoded
        self._cache['currency_id'] = value



