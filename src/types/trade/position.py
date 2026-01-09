"""
Trade
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class InvestorPosition(CapsuleStruct):
    """投资者持仓"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 9),             # 保留的无效字段
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("PosiDirection", ctypes.c_char),            # 持仓多空方向
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("PositionDate", ctypes.c_char),             # 持仓日期
            ("YdPosition", ctypes.c_int),                # 上日持仓
            ("Position", ctypes.c_int),                  # 今日持仓
            ("LongFrozen", ctypes.c_int),                # 多头冻结
            ("ShortFrozen", ctypes.c_int),               # 空头冻结
            ("LongFrozenAmount", ctypes.c_double),       # 开仓冻结金额
            ("ShortFrozenAmount", ctypes.c_double),      # 开仓冻结金额
            ("OpenVolume", ctypes.c_int),                # 开仓量
            ("CloseVolume", ctypes.c_int),               # 平仓量
            ("OpenAmount", ctypes.c_double),             # 开仓金额
            ("CloseAmount", ctypes.c_double),            # 平仓金额
            ("PositionCost", ctypes.c_double),           # 持仓成本
            ("PreMargin", ctypes.c_double),              # 上次占用的保证金
            ("UseMargin", ctypes.c_double),              # 占用的保证金
            ("FrozenMargin", ctypes.c_double),           # 冻结的保证金
            ("FrozenCash", ctypes.c_double),             # 冻结的资金
            ("FrozenCommission", ctypes.c_double),       # 冻结的手续费
            ("CashIn", ctypes.c_double),                 # 资金差额
            ("Commission", ctypes.c_double),             # 手续费
            ("CloseProfit", ctypes.c_double),            # 平仓盈亏
            ("PositionProfit", ctypes.c_double),         # 持仓盈亏
            ("PreSettlementPrice", ctypes.c_double),     # 上次结算价
            ("SettlementPrice", ctypes.c_double),        # 本次结算价
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("OpenCost", ctypes.c_double),               # 开仓成本
            ("ExchangeMargin", ctypes.c_double),         # 交易所保证金
            ("CombPosition", ctypes.c_int),             # 组合成交形成的持仓
            ("CombLongFrozen", ctypes.c_int),            # 组合多头冻结
            ("CombShortFrozen", ctypes.c_int),           # 组合空头冻结
            ("CloseProfitByDate", ctypes.c_double),     # 逐日盯市平仓盈亏
            ("CloseProfitByTrade", ctypes.c_double),    # 逐笔对冲平仓盈亏
            ("TodayPosition", ctypes.c_int),             # 今日持仓
            ("MarginRateByMoney", ctypes.c_double),      # 保证金率
            ("MarginRateByVolume", ctypes.c_double),     # 保证金率(按手数)
            ("StrikeFrozen", ctypes.c_int),              # 执行冻结
            ("StrikeFrozenAmount", ctypes.c_double),     # 执行冻结金额
            ("AbandonFrozen", ctypes.c_int),             # 放弃执行冻结
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("YdStrikeFrozen", ctypes.c_int),            # 执行冻结的昨仓
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("PositionCostOffset", ctypes.c_double),     # 持仓成本差值
            ("TasPosition", ctypes.c_int),              # tas持仓手数
            ("TasPositionCost", ctypes.c_double),        # tas持仓成本
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("OptionValue", ctypes.c_double),            # 期权市值
        ]

    _capsule_name = "InvestorPosition"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "posi_direction": "PosiDirection",
        "hedge_flag": "HedgeFlag",
        "position_date": "PositionDate",
        "yd_position": "YdPosition",
        "position": "Position",
        "long_frozen": "LongFrozen",
        "short_frozen": "ShortFrozen",
        "long_frozen_amount": "LongFrozenAmount",
        "short_frozen_amount": "ShortFrozenAmount",
        "open_volume": "OpenVolume",
        "close_volume": "CloseVolume",
        "open_amount": "OpenAmount",
        "close_amount": "CloseAmount",
        "position_cost": "PositionCost",
        "pre_margin": "PreMargin",
        "use_margin": "UseMargin",
        "frozen_margin": "FrozenMargin",
        "frozen_cash": "FrozenCash",
        "frozen_commission": "FrozenCommission",
        "cash_in": "CashIn",
        "commission": "Commission",
        "close_profit": "CloseProfit",
        "position_profit": "PositionProfit",
        "pre_settlement_price": "PreSettlementPrice",
        "settlement_price": "SettlementPrice",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "open_cost": "OpenCost",
        "exchange_margin": "ExchangeMargin",
        "comb_position": "CombPosition",
        "comb_long_frozen": "CombLongFrozen",
        "comb_short_frozen": "CombShortFrozen",
        "close_profit_by_date": "CloseProfitByDate",
        "close_profit_by_trade": "CloseProfitByTrade",
        "today_position": "TodayPosition",
        "margin_rate_by_money": "MarginRateByMoney",
        "margin_rate_by_volume": "MarginRateByVolume",
        "strike_frozen": "StrikeFrozen",
        "strike_frozen_amount": "StrikeFrozenAmount",
        "abandon_frozen": "AbandonFrozen",
        "exchange_id": "ExchangeID",
        "yd_strike_frozen": "YdStrikeFrozen",
        "invest_unit_id": "InvestUnitID",
        "position_cost_offset": "PositionCostOffset",
        "tas_position": "TasPosition",
        "tas_position_cost": "TasPositionCost",
        "instrument_id": "InstrumentID",
        "option_value": "OptionValue",
    }

    # 字符串属性（懒加载 + 缓存）
    @property
    def instrument_id(self) -> str:
        """合约代码"""
        if 'instrument_id' not in self._cache:
            value = self._struct.InstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['instrument_id'] = value
        return self._cache['instrument_id']

    @instrument_id.setter
    def instrument_id(self, value: str):
        """设置合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.InstrumentID = encoded
        self._cache['instrument_id'] = value

    @property
    def option_value(self) -> float:
        """期权市值"""
        return self._struct.OptionValue

    @option_value.setter
    def option_value(self, value: float):
        """设置期权市值"""
        self._struct.OptionValue = value

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
    def posi_direction(self) -> str:
        """持仓方向"""
        if 'posi_direction' not in self._cache:
            value = self._struct.PosiDirection.decode('ascii')
            self._cache['posi_direction'] = value
        return self._cache['posi_direction']

    @posi_direction.setter
    def posi_direction(self, value: str):
        """设置持仓方向"""
        self._struct.PosiDirection = value.encode('ascii')[0]
        self._cache['posi_direction'] = value

    @property
    def hedge_flag(self) -> str:
        """投机套保标志"""
        if 'hedge_flag' not in self._cache:
            value = self._struct.HedgeFlag.decode('ascii')
            self._cache['hedge_flag'] = value
        return self._cache['hedge_flag']

    @hedge_flag.setter
    def hedge_flag(self, value: str):
        """设置投机套保标志"""
        self._struct.HedgeFlag = value.encode('ascii')[0]
        self._cache['hedge_flag'] = value

    @property
    def position_date(self) -> str:
        """持仓日期"""
        if 'position_date' not in self._cache:
            value = self._struct.PositionDate.decode('ascii')
            self._cache['position_date'] = value
        return self._cache['position_date']

    @position_date.setter
    def position_date(self, value: str):
        """设置持仓日期"""
        self._struct.PositionDate = value.encode('ascii')[0]
        self._cache['position_date'] = value

    @property
    def yd_position(self) -> int:
        """昨持仓"""
        return self._struct.YdPosition

    @yd_position.setter
    def yd_position(self, value: int):
        """设置昨持仓"""
        self._struct.YdPosition = value

    @property
    def position(self) -> int:
        """今日持仓"""
        return self._struct.Position

    @position.setter
    def position(self, value: int):
        """设置今日持仓"""
        self._struct.Position = value

    @property
    def long_frozen(self) -> int:
        """冻结数量"""
        return self._struct.LongFrozen

    @long_frozen.setter
    def long_frozen(self, value: int):
        """设置冻结数量"""
        self._struct.LongFrozen = value

    @property
    def short_frozen(self) -> int:
        """冻结数量"""
        return self._struct.ShortFrozen

    @short_frozen.setter
    def short_frozen(self, value: int):
        """设置冻结数量"""
        self._struct.ShortFrozen = value

    @property
    def long_frozen_amount(self) -> float:
        """冻结金额"""
        return self._struct.LongFrozenAmount

    @long_frozen_amount.setter
    def long_frozen_amount(self, value: float):
        """设置冻结金额"""
        self._struct.LongFrozenAmount = value

    @property
    def short_frozen_amount(self) -> float:
        """冻结金额"""
        return self._struct.ShortFrozenAmount

    @short_frozen_amount.setter
    def short_frozen_amount(self, value: float):
        """设置冻结金额"""
        self._struct.ShortFrozenAmount = value

    @property
    def open_volume(self) -> int:
        """开仓量"""
        return self._struct.OpenVolume

    @open_volume.setter
    def open_volume(self, value: int):
        """设置开仓量"""
        self._struct.OpenVolume = value

    @property
    def close_volume(self) -> int:
        """平仓量"""
        return self._struct.CloseVolume

    @close_volume.setter
    def close_volume(self, value: int):
        """设置平仓量"""
        self._struct.CloseVolume = value

    @property
    def open_amount(self) -> float:
        """开仓金额"""
        return self._struct.OpenAmount

    @open_amount.setter
    def open_amount(self, value: float):
        """设置开仓金额"""
        self._struct.OpenAmount = value

    @property
    def close_amount(self) -> float:
        """平仓金额"""
        return self._struct.CloseAmount

    @close_amount.setter
    def close_amount(self, value: float):
        """设置平仓金额"""
        self._struct.CloseAmount = value

    @property
    def position_cost(self) -> float:
        """持仓成本"""
        return self._struct.PositionCost

    @position_cost.setter
    def position_cost(self, value: float):
        """设置持仓成本"""
        self._struct.PositionCost = value

    @property
    def pre_margin(self) -> float:
        """上次占用的保证金"""
        return self._struct.PreMargin

    @pre_margin.setter
    def pre_margin(self, value: float):
        """设置上次占用的保证金"""
        self._struct.PreMargin = value

    @property
    def use_margin(self) -> float:
        """占用的保证金"""
        return self._struct.UseMargin

    @use_margin.setter
    def use_margin(self, value: float):
        """设置占用的保证金"""
        self._struct.UseMargin = value

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
    def commission(self) -> float:
        """手续费"""
        return self._struct.Commission

    @commission.setter
    def commission(self, value: float):
        """设置手续费"""
        self._struct.Commission = value

    @property
    def cash_in(self) -> float:
        """资金存入"""
        return self._struct.CashIn

    @cash_in.setter
    def cash_in(self, value: float):
        """设置资金差额"""
        self._struct.CashIn = value

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
    def pre_settlement_price(self) -> float:
        """上次结算价"""
        return self._struct.PreSettlementPrice

    @pre_settlement_price.setter
    def pre_settlement_price(self, value: float):
        """设置上次结算价"""
        self._struct.PreSettlementPrice = value

    @property
    def settlement_price(self) -> float:
        """本次结算价"""
        return self._struct.SettlementPrice

    @settlement_price.setter
    def settlement_price(self, value: float):
        """设置本次结算价"""
        self._struct.SettlementPrice = value

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
    def open_cost(self) -> float:
        """开仓成本"""
        return self._struct.OpenCost

    @open_cost.setter
    def open_cost(self, value: float):
        """设置开仓成本"""
        self._struct.OpenCost = value

    @property
    def exchange_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchangeMargin

    @exchange_margin.setter
    def exchange_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchangeMargin = value

    @property
    def comb_position(self) -> int:
        """组合成交形成的持仓"""
        return self._struct.CombPosition

    @comb_position.setter
    def comb_position(self, value: int):
        """设置组合成交形成的持仓"""
        self._struct.CombPosition = value

    @property
    def comb_long_frozen(self) -> int:
        """组合多头冻结"""
        return self._struct.CombLongFrozen

    @comb_long_frozen.setter
    def comb_long_frozen(self, value: int):
        """设置组合多头冻结"""
        self._struct.CombLongFrozen = value

    @property
    def comb_short_frozen(self) -> int:
        """组合空头冻结"""
        return self._struct.CombShortFrozen

    @comb_short_frozen.setter
    def comb_short_frozen(self, value: int):
        """设置组合空头冻结"""
        self._struct.CombShortFrozen = value

    @property
    def close_profit_by_date(self) -> float:
        """按日平仓盈亏"""
        return self._struct.CloseProfitByDate

    @close_profit_by_date.setter
    def close_profit_by_date(self, value: float):
        """设置按日平仓盈亏"""
        self._struct.CloseProfitByDate = value

    @property
    def close_profit_by_trade(self) -> float:
        """按仓平仓盈亏"""
        return self._struct.CloseProfitByTrade

    @close_profit_by_trade.setter
    def close_profit_by_trade(self, value: float):
        """设置按仓平仓盈亏"""
        self._struct.CloseProfitByTrade = value

    @property
    def today_position(self) -> int:
        """今日持仓"""
        return self._struct.TodayPosition

    @today_position.setter
    def today_position(self, value: int):
        """设置今日持仓"""
        self._struct.TodayPosition = value

    @property
    def margin_rate_by_money(self) -> float:
        """资金保证金率"""
        return self._struct.MarginRateByMoney

    @margin_rate_by_money.setter
    def margin_rate_by_money(self, value: float):
        """设置资金保证金率"""
        self._struct.MarginRateByMoney = value

    @property
    def margin_rate_by_volume(self) -> float:
        """手续费保证金率"""
        return self._struct.MarginRateByVolume

    @margin_rate_by_volume.setter
    def margin_rate_by_volume(self, value: float):
        """设置手续费保证金率"""
        self._struct.MarginRateByVolume = value

    @property
    def strike_frozen(self) -> int:
        """执行冻结"""
        return self._struct.StrikeFrozen

    @strike_frozen.setter
    def strike_frozen(self, value: int):
        """设置执行冻结"""
        self._struct.StrikeFrozen = value

    @property
    def strike_frozen_amount(self) -> float:
        """执行冻结金额"""
        return self._struct.StrikeFrozenAmount

    @strike_frozen_amount.setter
    def strike_frozen_amount(self, value: float):
        """设置执行冻结金额"""
        self._struct.StrikeFrozenAmount = value

    @property
    def abandon_frozen(self) -> int:
        """放弃执行冻结"""
        return self._struct.AbandonFrozen

    @abandon_frozen.setter
    def abandon_frozen(self, value: int):
        """设置放弃执行冻结"""
        self._struct.AbandonFrozen = value

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
    def yd_strike_frozen(self) -> int:
        """执行冻结的昨仓"""
        return self._struct.YdStrikeFrozen

    @yd_strike_frozen.setter
    def yd_strike_frozen(self, value: int):
        """设置执行冻结的昨仓"""
        self._struct.YdStrikeFrozen = value

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
    def position_cost_offset(self) -> float:
        """持仓成本差值"""
        return self._struct.PositionCostOffset

    @position_cost_offset.setter
    def position_cost_offset(self, value: float):
        """设置持仓成本差值"""
        self._struct.PositionCostOffset = value

    @property
    def tas_position(self) -> int:
        """tas持仓手数"""
        return self._struct.TasPosition

    @tas_position.setter
    def tas_position(self, value: int):
        """设置tas持仓手数"""
        self._struct.TasPosition = value

    @property
    def tas_position_cost(self) -> float:
        """tas持仓成本"""
        return self._struct.TasPositionCost

    @tas_position_cost.setter
    def tas_position_cost(self, value: float):
        """设置tas持仓成本"""
        self._struct.TasPositionCost = value


# =============================================================================
# TradingAccount - 资金账户
# =============================================================================


class InvestorPositionDetail(CapsuleStruct):
    """投资者持仓明细"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("Direction", ctypes.c_char),                # 买卖
            ("OpenDate", ctypes.c_char * 9),             # 开仓日期
            ("TradeID", ctypes.c_char * 21),             # 成交编号
            ("Volume", ctypes.c_int),                    # 数量
            ("OpenPrice", ctypes.c_double),              # 开仓价
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("TradeType", ctypes.c_char),                # 成交类型
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("CloseProfitByDate", ctypes.c_double),      # 逐日盯市平仓盈亏
            ("CloseProfitByTrade", ctypes.c_double),     # 逐笔对冲平仓盈亏
            ("PositionProfitByDate", ctypes.c_double),   # 逐日盯市持仓盈亏
            ("PositionProfitByTrade", ctypes.c_double),  # 逐笔对冲持仓盈亏
            ("Margin", ctypes.c_double),                 # 投资者保证金
            ("ExchMargin", ctypes.c_double),             # 交易所保证金
            ("MarginRateByMoney", ctypes.c_double),      # 保证金率
            ("MarginRateByVolume", ctypes.c_double),     # 保证金率(按手数)
            ("LastSettlementPrice", ctypes.c_double),    # 昨结算价
            ("SettlementPrice", ctypes.c_double),        # 结算价
            ("CloseVolume", ctypes.c_int),               # 平仓量
            ("CloseAmount", ctypes.c_double),            # 平仓金额
            ("TimeFirstVolume", ctypes.c_int),           # 先开先平剩余数量
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("SpecPosiType", ctypes.c_char),             # 特殊持仓标志
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "InvestorPositionDetail"

    _field_mappings = {
        "reserve1": "reserve1",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "hedge_flag": "HedgeFlag",
        "direction": "Direction",
        "open_date": "OpenDate",
        "trade_id": "TradeID",
        "volume": "Volume",
        "open_price": "OpenPrice",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "trade_type": "TradeType",
        "reserve2": "reserve2",
        "exchange_id": "ExchangeID",
        "close_profit_by_date": "CloseProfitByDate",
        "close_profit_by_trade": "CloseProfitByTrade",
        "position_profit_by_date": "PositionProfitByDate",
        "position_profit_by_trade": "PositionProfitByTrade",
        "margin": "Margin",
        "exch_margin": "ExchMargin",
        "margin_rate_by_money": "MarginRateByMoney",
        "margin_rate_by_volume": "MarginRateByVolume",
        "last_settlement_price": "LastSettlementPrice",
        "settlement_price": "SettlementPrice",
        "close_volume": "CloseVolume",
        "close_amount": "CloseAmount",
        "time_first_volume": "TimeFirstVolume",
        "invest_unit_id": "InvestUnitID",
        "spec_posi_type": "SpecPosiType",
        "instrument_id": "InstrumentID",
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
    def hedge_flag(self) -> str:
        """投机套保标志"""
        if 'hedge_flag' not in self._cache:
            value = self._struct.HedgeFlag.decode('ascii')
            self._cache['hedge_flag'] = value
        return self._cache['hedge_flag']

    @hedge_flag.setter
    def hedge_flag(self, value: str):
        """设置投机套保标志"""
        self._struct.HedgeFlag = value.encode('ascii')[0]
        self._cache['hedge_flag'] = value

    @property
    def direction(self) -> str:
        """买卖方向"""
        if 'direction' not in self._cache:
            value = self._struct.Direction.decode('ascii')
            self._cache['direction'] = value
        return self._cache['direction']

    @direction.setter
    def direction(self, value: str):
        """设置买卖方向"""
        self._struct.Direction = value.encode('ascii')[0]
        self._cache['direction'] = value

    @property
    def open_date(self) -> str:
        """开仓日期"""
        if 'open_date' not in self._cache:
            value = self._struct.OpenDate.rstrip(b'\x00').decode('ascii')
            self._cache['open_date'] = value
        return self._cache['open_date']

    @open_date.setter
    def open_date(self, value: str):
        """设置开仓日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.OpenDate = encoded
        self._cache['open_date'] = value

    @property
    def trade_id(self) -> str:
        """成交编号"""
        if 'trade_id' not in self._cache:
            value = self._struct.TradeID.rstrip(b'\x00').decode('ascii')
            self._cache['trade_id'] = value
        return self._cache['trade_id']

    @trade_id.setter
    def trade_id(self, value: str):
        """设置成交编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.TradeID = encoded
        self._cache['trade_id'] = value

    @property
    def volume(self) -> int:
        """数量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置数量"""
        self._struct.Volume = value

    @property
    def open_price(self) -> float:
        """开仓价"""
        return self._struct.OpenPrice

    @open_price.setter
    def open_price(self, value: float):
        """设置开仓价"""
        self._struct.OpenPrice = value

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
    def trade_type(self) -> str:
        """成交类型"""
        if 'trade_type' not in self._cache:
            value = self._struct.TradeType.decode('ascii')
            self._cache['trade_type'] = value
        return self._cache['trade_type']

    @trade_type.setter
    def trade_type(self, value: str):
        """设置成交类型"""
        self._struct.TradeType = value.encode('ascii')[0]
        self._cache['trade_type'] = value

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
    def close_profit_by_date(self) -> float:
        """逐日盯市平仓盈亏"""
        return self._struct.CloseProfitByDate

    @close_profit_by_date.setter
    def close_profit_by_date(self, value: float):
        """设置逐日盯市平仓盈亏"""
        self._struct.CloseProfitByDate = value

    @property
    def close_profit_by_trade(self) -> float:
        """逐笔对冲平仓盈亏"""
        return self._struct.CloseProfitByTrade

    @close_profit_by_trade.setter
    def close_profit_by_trade(self, value: float):
        """设置逐笔对冲平仓盈亏"""
        self._struct.CloseProfitByTrade = value

    @property
    def position_profit_by_date(self) -> float:
        """逐日盯市持仓盈亏"""
        return self._struct.PositionProfitByDate

    @position_profit_by_date.setter
    def position_profit_by_date(self, value: float):
        """设置逐日盯市持仓盈亏"""
        self._struct.PositionProfitByDate = value

    @property
    def position_profit_by_trade(self) -> float:
        """逐笔对冲持仓盈亏"""
        return self._struct.PositionProfitByTrade

    @position_profit_by_trade.setter
    def position_profit_by_trade(self, value: float):
        """设置逐笔对冲持仓盈亏"""
        self._struct.PositionProfitByTrade = value

    @property
    def margin(self) -> float:
        """投资者保证金"""
        return self._struct.Margin

    @margin.setter
    def margin(self, value: float):
        """设置投资者保证金"""
        self._struct.Margin = value

    @property
    def exch_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchMargin = value

    @property
    def margin_rate_by_money(self) -> float:
        """保证金率"""
        return self._struct.MarginRateByMoney

    @margin_rate_by_money.setter
    def margin_rate_by_money(self, value: float):
        """设置保证金率"""
        self._struct.MarginRateByMoney = value

    @property
    def margin_rate_by_volume(self) -> float:
        """保证金率(按手数)"""
        return self._struct.MarginRateByVolume

    @margin_rate_by_volume.setter
    def margin_rate_by_volume(self, value: float):
        """设置保证金率(按手数)"""
        self._struct.MarginRateByVolume = value

    @property
    def last_settlement_price(self) -> float:
        """昨结算价"""
        return self._struct.LastSettlementPrice

    @last_settlement_price.setter
    def last_settlement_price(self, value: float):
        """设置昨结算价"""
        self._struct.LastSettlementPrice = value

    @property
    def settlement_price(self) -> float:
        """结算价"""
        return self._struct.SettlementPrice

    @settlement_price.setter
    def settlement_price(self, value: float):
        """设置结算价"""
        self._struct.SettlementPrice = value

    @property
    def close_volume(self) -> int:
        """平仓量"""
        return self._struct.CloseVolume

    @close_volume.setter
    def close_volume(self, value: int):
        """设置平仓量"""
        self._struct.CloseVolume = value

    @property
    def close_amount(self) -> float:
        """平仓金额"""
        return self._struct.CloseAmount

    @close_amount.setter
    def close_amount(self, value: float):
        """设置平仓金额"""
        self._struct.CloseAmount = value

    @property
    def time_first_volume(self) -> int:
        """先开先平剩余数量"""
        return self._struct.TimeFirstVolume

    @time_first_volume.setter
    def time_first_volume(self, value: int):
        """设置先开先平剩余数量"""
        self._struct.TimeFirstVolume = value

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
    def spec_posi_type(self) -> str:
        """特殊持仓标志"""
        if 'spec_posi_type' not in self._cache:
            value = self._struct.SpecPosiType.decode('ascii')
            self._cache['spec_posi_type'] = value
        return self._cache['spec_posi_type']

    @spec_posi_type.setter
    def spec_posi_type(self, value: str):
        """设置特殊持仓标志"""
        self._struct.SpecPosiType = value.encode('ascii')[0]
        self._cache['spec_posi_type'] = value

    @property
    def instrument_id(self) -> str:
        """合约代码"""
        if 'instrument_id' not in self._cache:
            value = self._struct.InstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['instrument_id'] = value
        return self._cache['instrument_id']

    @instrument_id.setter
    def instrument_id(self, value: str):
        """设置合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.InstrumentID = encoded
        self._cache['instrument_id'] = value



class InvestorPositionCombineDetail(CapsuleStruct):
    """投资者持仓组合明细"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("OpenDate", ctypes.c_char * 9),             # 开仓日期
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("ComTradeID", ctypes.c_char * 21),          # 组合编号
            ("TradeID", ctypes.c_char * 21),             # 撮合编号
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("Direction", ctypes.c_char),                # 买卖
            ("TotalAmt", ctypes.c_int),                  # 持仓量
            ("Margin", ctypes.c_double),                 # 投资者保证金
            ("ExchMargin", ctypes.c_double),             # 交易所保证金
            ("MarginRateByMoney", ctypes.c_double),      # 保证金率
            ("MarginRateByVolume", ctypes.c_double),     # 保证金率(按手数)
            ("Leg1InstrumentID", ctypes.c_char * 81),    # 第一条合约代码
            ("Leg1Direction", ctypes.c_char),            # 第一条买卖方向
            ("Leg1Volume", ctypes.c_int),                # 第一条数量
            ("Leg2InstrumentID", ctypes.c_char * 81),    # 第二条合约代码
            ("Leg2Direction", ctypes.c_char),            # 第二条买卖方向
            ("Leg2Volume", ctypes.c_int),                # 第二条数量
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
        ]

    _capsule_name = "InvestorPositionCombineDetail"

    _field_mappings = {
        "trading_day": "TradingDay",
        "open_date": "OpenDate",
        "exchange_id": "ExchangeID",
        "settlement_id": "SettlementID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "com_trade_id": "ComTradeID",
        "trade_id": "TradeID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
        "direction": "Direction",
        "total_amt": "TotalAmt",
        "margin": "Margin",
        "exch_margin": "ExchMargin",
        "margin_rate_by_money": "MarginRateByMoney",
        "margin_rate_by_volume": "MarginRateByVolume",
        "leg1_instrument_id": "Leg1InstrumentID",
        "leg1_direction": "Leg1Direction",
        "leg1_volume": "Leg1Volume",
        "leg2_instrument_id": "Leg2InstrumentID",
        "leg2_direction": "Leg2Direction",
        "leg2_volume": "Leg2Volume",
        "invest_unit_id": "InvestUnitID",
    }

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
    def open_date(self) -> str:
        """开仓日期"""
        if 'open_date' not in self._cache:
            value = self._struct.OpenDate.rstrip(b'\x00').decode('ascii')
            self._cache['open_date'] = value
        return self._cache['open_date']

    @open_date.setter
    def open_date(self, value: str):
        """设置开仓日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.OpenDate = encoded
        self._cache['open_date'] = value

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
    def settlement_id(self) -> int:
        """结算编号"""
        return self._struct.SettlementID

    @settlement_id.setter
    def settlement_id(self, value: int):
        """设置结算编号"""
        self._struct.SettlementID = value

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
    def com_trade_id(self) -> str:
        """组合编号"""
        if 'com_trade_id' not in self._cache:
            value = self._struct.ComTradeID.rstrip(b'\x00').decode('ascii')
            self._cache['com_trade_id'] = value
        return self._cache['com_trade_id']

    @com_trade_id.setter
    def com_trade_id(self, value: str):
        """设置组合编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ComTradeID = encoded
        self._cache['com_trade_id'] = value

    @property
    def trade_id(self) -> str:
        """撮合编号"""
        if 'trade_id' not in self._cache:
            value = self._struct.TradeID.rstrip(b'\x00').decode('ascii')
            self._cache['trade_id'] = value
        return self._cache['trade_id']

    @trade_id.setter
    def trade_id(self, value: str):
        """设置撮合编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.TradeID = encoded
        self._cache['trade_id'] = value

    @property
    def hedge_flag(self) -> str:
        """投机套保标志"""
        if 'hedge_flag' not in self._cache:
            value = self._struct.HedgeFlag.decode('ascii')
            self._cache['hedge_flag'] = value
        return self._cache['hedge_flag']

    @hedge_flag.setter
    def hedge_flag(self, value: str):
        """设置投机套保标志"""
        self._struct.HedgeFlag = value.encode('ascii')[0]
        self._cache['hedge_flag'] = value

    @property
    def direction(self) -> str:
        """买卖方向"""
        if 'direction' not in self._cache:
            value = self._struct.Direction.decode('ascii')
            self._cache['direction'] = value
        return self._cache['direction']

    @direction.setter
    def direction(self, value: str):
        """设置买卖方向"""
        self._struct.Direction = value.encode('ascii')[0]
        self._cache['direction'] = value

    @property
    def total_amt(self) -> int:
        """持仓量"""
        return self._struct.TotalAmt

    @total_amt.setter
    def total_amt(self, value: int):
        """设置持仓量"""
        self._struct.TotalAmt = value

    @property
    def margin(self) -> float:
        """投资者保证金"""
        return self._struct.Margin

    @margin.setter
    def margin(self, value: float):
        """设置投资者保证金"""
        self._struct.Margin = value

    @property
    def exch_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchMargin = value

    @property
    def margin_rate_by_money(self) -> float:
        """保证金率"""
        return self._struct.MarginRateByMoney

    @margin_rate_by_money.setter
    def margin_rate_by_money(self, value: float):
        """设置保证金率"""
        self._struct.MarginRateByMoney = value

    @property
    def margin_rate_by_volume(self) -> float:
        """保证金率(按手数)"""
        return self._struct.MarginRateByVolume

    @margin_rate_by_volume.setter
    def margin_rate_by_volume(self, value: float):
        """设置保证金率(按手数)"""
        self._struct.MarginRateByVolume = value

    @property
    def leg1_instrument_id(self) -> str:
        """第一条合约代码"""
        if 'leg1_instrument_id' not in self._cache:
            value = self._struct.Leg1InstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['leg1_instrument_id'] = value
        return self._cache['leg1_instrument_id']

    @leg1_instrument_id.setter
    def leg1_instrument_id(self, value: str):
        """设置第一条合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg1InstrumentID = encoded
        self._cache['leg1_instrument_id'] = value

    @property
    def leg1_direction(self) -> str:
        """第一条买卖方向"""
        if 'leg1_direction' not in self._cache:
            value = self._struct.Leg1Direction.decode('ascii')
            self._cache['leg1_direction'] = value
        return self._cache['leg1_direction']

    @leg1_direction.setter
    def leg1_direction(self, value: str):
        """设置第一条买卖方向"""
        self._struct.Leg1Direction = value.encode('ascii')[0]
        self._cache['leg1_direction'] = value

    @property
    def leg1_volume(self) -> int:
        """第一条数量"""
        return self._struct.Leg1Volume

    @leg1_volume.setter
    def leg1_volume(self, value: int):
        """设置第一条数量"""
        self._struct.Leg1Volume = value

    @property
    def leg2_instrument_id(self) -> str:
        """第二条合约代码"""
        if 'leg2_instrument_id' not in self._cache:
            value = self._struct.Leg2InstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['leg2_instrument_id'] = value
        return self._cache['leg2_instrument_id']

    @leg2_instrument_id.setter
    def leg2_instrument_id(self, value: str):
        """设置第二条合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg2InstrumentID = encoded
        self._cache['leg2_instrument_id'] = value

    @property
    def leg2_direction(self) -> str:
        """第二条买卖方向"""
        if 'leg2_direction' not in self._cache:
            value = self._struct.Leg2Direction.decode('ascii')
            self._cache['leg2_direction'] = value
        return self._cache['leg2_direction']

    @leg2_direction.setter
    def leg2_direction(self, value: str):
        """设置第二条买卖方向"""
        self._struct.Leg2Direction = value.encode('ascii')[0]
        self._cache['leg2_direction'] = value

    @property
    def leg2_volume(self) -> int:
        """第二条数量"""
        return self._struct.Leg2Volume

    @leg2_volume.setter
    def leg2_volume(self, value: int):
        """设置第二条数量"""
        self._struct.Leg2Volume = value

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



