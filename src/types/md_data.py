"""
Md Data
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class DepthMarketData(CapsuleStruct):
    """
    深度行情数据（懒加载 + 缓存 + 零拷贝）

    性能特性：
    - C++ 端零拷贝：~130 ns
    - Python 端懒加载：首次访问字符串 ~150 ns，后续 ~10 ns
    - Python 端数值访问：直接内存访问 ~10 ns

    使用示例：
        class MySpi(MdSpi):
            def on_rtn_depth_market_data(self, data: DepthMarketData):
                # 首次访问，懒加载 + 缓存
                print(data.trading_day)  # ~150 ns

                # 后续访问，直接从缓存读取
                print(data.trading_day)  # ~10 ns

                # 数值直接访问
                print(data.last_price)  # ~10 ns
    """

    # C 结构体定义
    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("LastPrice", ctypes.c_double),              # 最新价
            ("PreSettlementPrice", ctypes.c_double),     # 上次结算价
            ("PreClosePrice", ctypes.c_double),          # 昨收盘
            ("PreOpenInterest", ctypes.c_double),        # 昨持仓量
            ("OpenPrice", ctypes.c_double),              # 今开盘
            ("HighestPrice", ctypes.c_double),           # 最高价
            ("LowestPrice", ctypes.c_double),            # 最低价
            ("Volume", ctypes.c_int),                    # 数量
            ("Turnover", ctypes.c_double),               # 成交金额
            ("OpenInterest", ctypes.c_double),           # 持仓量
            ("ClosePrice", ctypes.c_double),             # 今收盘
            ("SettlementPrice", ctypes.c_double),        # 本次结算价
            ("UpperLimitPrice", ctypes.c_double),        # 涨停板价
            ("LowerLimitPrice", ctypes.c_double),        # 跌停板价
            ("PreDelta", ctypes.c_double),               # 昨虚实度
            ("CurrDelta", ctypes.c_double),              # 今虚实度
            ("UpdateTime", ctypes.c_char * 9),           # 最后修改时间
            ("UpdateMillisec", ctypes.c_int),            # 最后修改毫秒
            ("BidPrice1", ctypes.c_double),              # 申买价一
            ("BidVolume1", ctypes.c_int),                # 申买量一
            ("AskPrice1", ctypes.c_double),              # 申卖价一
            ("AskVolume1", ctypes.c_int),                # 申卖量一
            ("BidPrice2", ctypes.c_double),              # 申买价二
            ("BidVolume2", ctypes.c_int),                # 申买量二
            ("AskPrice2", ctypes.c_double),              # 申卖价二
            ("AskVolume2", ctypes.c_int),                # 申卖量二
            ("BidPrice3", ctypes.c_double),              # 申买价三
            ("BidVolume3", ctypes.c_int),                # 申买量三
            ("AskPrice3", ctypes.c_double),              # 申卖价三
            ("AskVolume3", ctypes.c_int),                # 申卖量三
            ("BidPrice4", ctypes.c_double),              # 申买价四
            ("BidVolume4", ctypes.c_int),                # 申买量四
            ("AskPrice4", ctypes.c_double),              # 申卖价四
            ("AskVolume4", ctypes.c_int),                # 申卖量四
            ("BidPrice5", ctypes.c_double),              # 申买价五
            ("BidVolume5", ctypes.c_int),                # 申买量五
            ("AskPrice5", ctypes.c_double),              # 申卖价五
            ("AskVolume5", ctypes.c_int),                # 申卖量五
            ("AveragePrice", ctypes.c_double),           # 当日均价
            ("ActionDay", ctypes.c_char * 9),            # 业务日期
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 81),      # 合约在交易所的代码
            ("BandingUpperPrice", ctypes.c_double),      # 上带价
            ("BandingLowerPrice", ctypes.c_double),      # 下带价
        ]

    _capsule_name = "DepthMarketData"

    # 字段映射（snake_case -> PascalCase）
    _field_mappings = {
        "trading_day": "TradingDay",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "reserve2": "reserve2",
        "last_price": "LastPrice",
        "pre_settlement_price": "PreSettlementPrice",
        "pre_close_price": "PreClosePrice",
        "pre_open_interest": "PreOpenInterest",
        "open_price": "OpenPrice",
        "highest_price": "HighestPrice",
        "lowest_price": "LowestPrice",
        "volume": "Volume",
        "turnover": "Turnover",
        "open_interest": "OpenInterest",
        "close_price": "ClosePrice",
        "settlement_price": "SettlementPrice",
        "upper_limit_price": "UpperLimitPrice",
        "lower_limit_price": "LowerLimitPrice",
        "pre_delta": "PreDelta",
        "curr_delta": "CurrDelta",
        "update_time": "UpdateTime",
        "update_millisec": "UpdateMillisec",
        "bid_price1": "BidPrice1",
        "bid_volume1": "BidVolume1",
        "ask_price1": "AskPrice1",
        "ask_volume1": "AskVolume1",
        "bid_price2": "BidPrice2",
        "bid_volume2": "BidVolume2",
        "ask_price2": "AskPrice2",
        "ask_volume2": "AskVolume2",
        "bid_price3": "BidPrice3",
        "bid_volume3": "BidVolume3",
        "ask_price3": "AskPrice3",
        "ask_volume3": "AskVolume3",
        "bid_price4": "BidPrice4",
        "bid_volume4": "BidVolume4",
        "ask_price4": "AskPrice4",
        "ask_volume4": "AskVolume4",
        "bid_price5": "BidPrice5",
        "bid_volume5": "BidVolume5",
        "ask_price5": "AskPrice5",
        "ask_volume5": "AskVolume5",
        "average_price": "AveragePrice",
        "action_day": "ActionDay",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
        "banding_upper_price": "BandingUpperPrice",
        "banding_lower_price": "BandingLowerPrice",
    }

    # ========== 字符串属性（懒加载 + 缓存） ==========

    @property
    def trading_day(self) -> str:
        """交易日（懒加载 + 缓存）"""
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
    def reserve1(self) -> str:
        """保留的无效字段（懒加载 + 缓存）"""
        if 'reserve1' not in self._cache:
            value = self._struct.reserve1.rstrip(b'\x00').decode('ascii')
            self._cache['reserve1'] = value
        return self._cache['reserve1']

    @reserve1.setter
    def reserve1(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.reserve1 = encoded
        self._cache['reserve1'] = value

    @property
    def exchange_id(self) -> str:
        """交易所代码（懒加载 + 缓存）"""
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
    def reserve2(self) -> str:
        """保留的无效字段（懒加载 + 缓存）"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def update_time(self) -> str:
        """最后修改时间（懒加载 + 缓存）"""
        if 'update_time' not in self._cache:
            value = self._struct.UpdateTime.rstrip(b'\x00').decode('ascii')
            self._cache['update_time'] = value
        return self._cache['update_time']

    @update_time.setter
    def update_time(self, value: str):
        """设置最后修改时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.UpdateTime = encoded
        self._cache['update_time'] = value

    @property
    def action_day(self) -> str:
        """业务日期（懒加载 + 缓存）"""
        if 'action_day' not in self._cache:
            value = self._struct.ActionDay.rstrip(b'\x00').decode('ascii')
            self._cache['action_day'] = value
        return self._cache['action_day']

    @action_day.setter
    def action_day(self, value: str):
        """设置业务日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionDay = encoded
        self._cache['action_day'] = value

    @property
    def instrument_id(self) -> str:
        """合约代码（懒加载 + 缓存）"""
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
    def exchange_inst_id(self) -> str:
        """合约在交易所的代码（懒加载 + 缓存）"""
        if 'exchange_inst_id' not in self._cache:
            value = self._struct.ExchangeInstID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_inst_id'] = value
        return self._cache['exchange_inst_id']

    @exchange_inst_id.setter
    def exchange_inst_id(self, value: str):
        """设置合约在交易所的代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ExchangeInstID = encoded
        self._cache['exchange_inst_id'] = value

    # ========== 数值属性（零拷贝，不缓存） ==========

    @property
    def last_price(self) -> float:
        """最新价（零拷贝）"""
        return self._struct.LastPrice

    @last_price.setter
    def last_price(self, value: float):
        """设置最新价"""
        self._struct.LastPrice = value

    @property
    def pre_settlement_price(self) -> float:
        """上次结算价"""
        return self._struct.PreSettlementPrice

    @pre_settlement_price.setter
    def pre_settlement_price(self, value: float):
        """设置上次结算价"""
        self._struct.PreSettlementPrice = value

    @property
    def pre_close_price(self) -> float:
        """昨收盘"""
        return self._struct.PreClosePrice

    @pre_close_price.setter
    def pre_close_price(self, value: float):
        """设置昨收盘"""
        self._struct.PreClosePrice = value

    @property
    def pre_open_interest(self) -> float:
        """昨持仓量"""
        return self._struct.PreOpenInterest

    @pre_open_interest.setter
    def pre_open_interest(self, value: float):
        """设置昨持仓量"""
        self._struct.PreOpenInterest = value

    @property
    def open_price(self) -> float:
        """今开盘"""
        return self._struct.OpenPrice

    @open_price.setter
    def open_price(self, value: float):
        """设置今开盘"""
        self._struct.OpenPrice = value

    @property
    def highest_price(self) -> float:
        """最高价"""
        return self._struct.HighestPrice

    @highest_price.setter
    def highest_price(self, value: float):
        """设置最高价"""
        self._struct.HighestPrice = value

    @property
    def lowest_price(self) -> float:
        """最低价"""
        return self._struct.LowestPrice

    @lowest_price.setter
    def lowest_price(self, value: float):
        """设置最低价"""
        self._struct.LowestPrice = value

    @property
    def volume(self) -> int:
        """数量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置数量"""
        self._struct.Volume = value

    @property
    def turnover(self) -> float:
        """成交金额"""
        return self._struct.Turnover

    @turnover.setter
    def turnover(self, value: float):
        """设置成交金额"""
        self._struct.Turnover = value

    @property
    def open_interest(self) -> float:
        """持仓量"""
        return self._struct.OpenInterest

    @open_interest.setter
    def open_interest(self, value: float):
        """设置持仓量"""
        self._struct.OpenInterest = value

    @property
    def close_price(self) -> float:
        """今收盘"""
        return self._struct.ClosePrice

    @close_price.setter
    def close_price(self, value: float):
        """设置今收盘"""
        self._struct.ClosePrice = value

    @property
    def settlement_price(self) -> float:
        """本次结算价"""
        return self._struct.SettlementPrice

    @settlement_price.setter
    def settlement_price(self, value: float):
        """设置本次结算价"""
        self._struct.SettlementPrice = value

    @property
    def upper_limit_price(self) -> float:
        """涨停板价"""
        return self._struct.UpperLimitPrice

    @upper_limit_price.setter
    def upper_limit_price(self, value: float):
        """设置涨停板价"""
        self._struct.UpperLimitPrice = value

    @property
    def lower_limit_price(self) -> float:
        """跌停板价"""
        return self._struct.LowerLimitPrice

    @lower_limit_price.setter
    def lower_limit_price(self, value: float):
        """设置跌停板价"""
        self._struct.LowerLimitPrice = value

    @property
    def pre_delta(self) -> float:
        """昨虚实度"""
        return self._struct.PreDelta

    @pre_delta.setter
    def pre_delta(self, value: float):
        """设置昨虚实度"""
        self._struct.PreDelta = value

    @property
    def curr_delta(self) -> float:
        """今虚实度"""
        return self._struct.CurrDelta

    @curr_delta.setter
    def curr_delta(self, value: float):
        """设置今虚实度"""
        self._struct.CurrDelta = value

    @property
    def update_millisec(self) -> int:
        """最后修改毫秒"""
        return self._struct.UpdateMillisec

    @update_millisec.setter
    def update_millisec(self, value: int):
        """设置最后修改毫秒"""
        self._struct.UpdateMillisec = value

    @property
    def bid_price1(self) -> float:
        """申买价一"""
        return self._struct.BidPrice1

    @bid_price1.setter
    def bid_price1(self, value: float):
        """设置申买价一"""
        self._struct.BidPrice1 = value

    @property
    def bid_volume1(self) -> int:
        """申买量一"""
        return self._struct.BidVolume1

    @bid_volume1.setter
    def bid_volume1(self, value: int):
        """设置申买量一"""
        self._struct.BidVolume1 = value

    @property
    def ask_price1(self) -> float:
        """申卖价一"""
        return self._struct.AskPrice1

    @ask_price1.setter
    def ask_price1(self, value: float):
        """设置申卖价一"""
        self._struct.AskPrice1 = value

    @property
    def ask_volume1(self) -> int:
        """申卖量一"""
        return self._struct.AskVolume1

    @ask_volume1.setter
    def ask_volume1(self, value: int):
        """设置申卖量一"""
        self._struct.AskVolume1 = value

    @property
    def bid_price2(self) -> float:
        """申买价二"""
        return self._struct.BidPrice2

    @bid_price2.setter
    def bid_price2(self, value: float):
        """设置申买价二"""
        self._struct.BidPrice2 = value

    @property
    def bid_volume2(self) -> int:
        """申买量二"""
        return self._struct.BidVolume2

    @bid_volume2.setter
    def bid_volume2(self, value: int):
        """设置申买量二"""
        self._struct.BidVolume2 = value

    @property
    def ask_price2(self) -> float:
        """申卖价二"""
        return self._struct.AskPrice2

    @ask_price2.setter
    def ask_price2(self, value: float):
        """设置申卖价二"""
        self._struct.AskPrice2 = value

    @property
    def ask_volume2(self) -> int:
        """申卖量二"""
        return self._struct.AskVolume2

    @ask_volume2.setter
    def ask_volume2(self, value: int):
        """设置申卖量二"""
        self._struct.AskVolume2 = value

    @property
    def bid_price3(self) -> float:
        """申买价三"""
        return self._struct.BidPrice3

    @bid_price3.setter
    def bid_price3(self, value: float):
        """设置申买价三"""
        self._struct.BidPrice3 = value

    @property
    def bid_volume3(self) -> int:
        """申买量三"""
        return self._struct.BidVolume3

    @bid_volume3.setter
    def bid_volume3(self, value: int):
        """设置申买量三"""
        self._struct.BidVolume3 = value

    @property
    def ask_price3(self) -> float:
        """申卖价三"""
        return self._struct.AskPrice3

    @ask_price3.setter
    def ask_price3(self, value: float):
        """设置申卖价三"""
        self._struct.AskPrice3 = value

    @property
    def ask_volume3(self) -> int:
        """申卖量三"""
        return self._struct.AskVolume3

    @ask_volume3.setter
    def ask_volume3(self, value: int):
        """设置申卖量三"""
        self._struct.AskVolume3 = value

    @property
    def bid_price4(self) -> float:
        """申买价四"""
        return self._struct.BidPrice4

    @bid_price4.setter
    def bid_price4(self, value: float):
        """设置申买价四"""
        self._struct.BidPrice4 = value

    @property
    def bid_volume4(self) -> int:
        """申买量四"""
        return self._struct.BidVolume4

    @bid_volume4.setter
    def bid_volume4(self, value: int):
        """设置申买量四"""
        self._struct.BidVolume4 = value

    @property
    def ask_price4(self) -> float:
        """申卖价四"""
        return self._struct.AskPrice4

    @ask_price4.setter
    def ask_price4(self, value: float):
        """设置申卖价四"""
        self._struct.AskPrice4 = value

    @property
    def ask_volume4(self) -> int:
        """申卖量四"""
        return self._struct.AskVolume4

    @ask_volume4.setter
    def ask_volume4(self, value: int):
        """设置申卖量四"""
        self._struct.AskVolume4 = value

    @property
    def bid_price5(self) -> float:
        """申买价五"""
        return self._struct.BidPrice5

    @bid_price5.setter
    def bid_price5(self, value: float):
        """设置申买价五"""
        self._struct.BidPrice5 = value

    @property
    def bid_volume5(self) -> int:
        """申买量五"""
        return self._struct.BidVolume5

    @bid_volume5.setter
    def bid_volume5(self, value: int):
        """设置申买量五"""
        self._struct.BidVolume5 = value

    @property
    def ask_price5(self) -> float:
        """申卖价五"""
        return self._struct.AskPrice5

    @ask_price5.setter
    def ask_price5(self, value: float):
        """设置申卖价五"""
        self._struct.AskPrice5 = value

    @property
    def ask_volume5(self) -> int:
        """申卖量五"""
        return self._struct.AskVolume5

    @ask_volume5.setter
    def ask_volume5(self, value: int):
        """设置申卖量五"""
        self._struct.AskVolume5 = value

    @property
    def average_price(self) -> float:
        """当日均价"""
        return self._struct.AveragePrice

    @average_price.setter
    def average_price(self, value: float):
        """设置当日均价"""
        self._struct.AveragePrice = value

    @property
    def banding_upper_price(self) -> float:
        """上带价"""
        return self._struct.BandingUpperPrice

    @banding_upper_price.setter
    def banding_upper_price(self, value: float):
        """设置上带价"""
        self._struct.BandingUpperPrice = value

    @property
    def banding_lower_price(self) -> float:
        """下带价"""
        return self._struct.BandingLowerPrice

    @banding_lower_price.setter
    def banding_lower_price(self, value: float):
        """设置下带价"""
        self._struct.BandingLowerPrice = value


# =============================================================================
# RspInfo - 响应信息
# =============================================================================


