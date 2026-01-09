"""
Margin
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class InstrumentMarginRate(CapsuleStruct):
    """合约保证金率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 9),             # 保留的无效字段
            ("InvestorRange", ctypes.c_char),            # 投资者范围
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("LongMarginRatioByMoney", ctypes.c_double), # 多头保证金率
            ("LongMarginRatioByVolume", ctypes.c_double), # 多头保证金费
            ("ShortMarginRatioByMoney", ctypes.c_double), # 空头保证金率
            ("ShortMarginRatioByVolume", ctypes.c_double), # 空头保证金费
            ("IsRelative", ctypes.c_int),                # 是否相对交易所收取
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "InstrumentMarginRate"

    _field_mappings = {
        "investor_range": "InvestorRange",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "hedge_flag": "HedgeFlag",
        "long_margin_ratio_by_money": "LongMarginRatioByMoney",
        "long_margin_ratio_by_volume": "LongMarginRatioByVolume",
        "short_margin_ratio_by_money": "ShortMarginRatioByMoney",
        "short_margin_ratio_by_volume": "ShortMarginRatioByVolume",
        "is_relative": "IsRelative",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
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
    def long_margin_ratio_by_money(self) -> float:
        """多头保证金率(按金额)"""
        return self._struct.LongMarginRatioByMoney

    @long_margin_ratio_by_money.setter
    def long_margin_ratio_by_money(self, value: float):
        """设置多头保证金率(按金额)"""
        self._struct.LongMarginRatioByMoney = value

    @property
    def long_margin_ratio_by_volume(self) -> float:
        """多头保证金率(按手数)"""
        return self._struct.LongMarginRatioByVolume

    @long_margin_ratio_by_volume.setter
    def long_margin_ratio_by_volume(self, value: float):
        """设置多头保证金率(按手数)"""
        self._struct.LongMarginRatioByVolume = value

    @property
    def short_margin_ratio_by_money(self) -> float:
        """空头保证金率(按金额)"""
        return self._struct.ShortMarginRatioByMoney

    @short_margin_ratio_by_money.setter
    def short_margin_ratio_by_money(self, value: float):
        """设置空头保证金率(按金额)"""
        self._struct.ShortMarginRatioByMoney = value

    @property
    def short_margin_ratio_by_volume(self) -> float:
        """空头保证金率(按手数)"""
        return self._struct.ShortMarginRatioByVolume

    @short_margin_ratio_by_volume.setter
    def short_margin_ratio_by_volume(self, value: float):
        """设置空头保证金率(按手数)"""
        self._struct.ShortMarginRatioByVolume = value

    @property
    def is_relative(self) -> int:
        """是否相对交易所收取"""
        return self._struct.IsRelative

    @is_relative.setter
    def is_relative(self, value: int):
        """设置是否相对交易所收取"""
        self._struct.IsRelative = value

    @property
    def investor_range(self) -> str:
        """投资者范围"""
        if 'investor_range' not in self._cache:
            value = self._struct.InvestorRange.decode('ascii')
            self._cache['investor_range'] = value
        return self._cache['investor_range']

    @investor_range.setter
    def investor_range(self, value: str):
        """设置投资者范围"""
        self._struct.InvestorRange = value.encode('ascii')[0]
        self._cache['investor_range'] = value

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


# =============================================================================
# InstrumentCommissionRate - 合约手续费率
# =============================================================================


class InstrumentCommissionRate(CapsuleStruct):
    """合约手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 9),             # 保留的无效字段
            ("InvestorRange", ctypes.c_char),            # 投资者范围
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("OpenRatioByMoney", ctypes.c_double),       # 开仓手续费率
            ("OpenRatioByVolume", ctypes.c_double),      # 开仓手续费
            ("CloseRatioByMoney", ctypes.c_double),      # 平仓手续费率
            ("CloseRatioByVolume", ctypes.c_double),     # 平仓手续费
            ("CloseTodayRatioByMoney", ctypes.c_double), # 平今手续费率
            ("CloseTodayRatioByVolume", ctypes.c_double), # 平今手续费
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("BizType", ctypes.c_char),                  # 业务类型
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "InstrumentCommissionRate"

    _field_mappings = {
        "investor_range": "InvestorRange",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "open_ratio_by_money": "OpenRatioByMoney",
        "open_ratio_by_volume": "OpenRatioByVolume",
        "close_ratio_by_money": "CloseRatioByMoney",
        "close_ratio_by_volume": "CloseRatioByVolume",
        "close_today_ratio_by_money": "CloseTodayRatioByMoney",
        "close_today_ratio_by_volume": "CloseTodayRatioByVolume",
        "exchange_id": "ExchangeID",
        "biz_type": "BizType",
        "invest_unit_id": "InvestUnitID",
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
    def open_ratio_by_money(self) -> float:
        """开仓手续费率(按金额)"""
        return self._struct.OpenRatioByMoney

    @open_ratio_by_money.setter
    def open_ratio_by_money(self, value: float):
        """设置开仓手续费率(按金额)"""
        self._struct.OpenRatioByMoney = value

    @property
    def open_ratio_by_volume(self) -> float:
        """开仓手续费率(按手数)"""
        return self._struct.OpenRatioByVolume

    @open_ratio_by_volume.setter
    def open_ratio_by_volume(self, value: float):
        """设置开仓手续费率(按手数)"""
        self._struct.OpenRatioByVolume = value

    @property
    def close_ratio_by_money(self) -> float:
        """平仓手续费率(按金额)"""
        return self._struct.CloseRatioByMoney

    @close_ratio_by_money.setter
    def close_ratio_by_money(self, value: float):
        """设置平仓手续费率(按金额)"""
        self._struct.CloseRatioByMoney = value

    @property
    def close_ratio_by_volume(self) -> float:
        """平仓手续费率(按手数)"""
        return self._struct.CloseRatioByVolume

    @close_ratio_by_volume.setter
    def close_ratio_by_volume(self, value: float):
        """设置平仓手续费率(按手数)"""
        self._struct.CloseRatioByVolume = value

    @property
    def close_today_ratio_by_money(self) -> float:
        """平今手续费率(按金额)"""
        return self._struct.CloseTodayRatioByMoney

    @close_today_ratio_by_money.setter
    def close_today_ratio_by_money(self, value: float):
        """设置平今手续费率(按金额)"""
        self._struct.CloseTodayRatioByMoney = value

    @property
    def close_today_ratio_by_volume(self) -> float:
        """平今手续费率(按手数)"""
        return self._struct.CloseTodayRatioByVolume

    @close_today_ratio_by_volume.setter
    def close_today_ratio_by_volume(self, value: float):
        """设置平今手续费"""
        self._struct.CloseTodayRatioByVolume = value

    @property
    def investor_range(self) -> str:
        """投资者范围"""
        if 'investor_range' not in self._cache:
            value = self._struct.InvestorRange.decode('ascii')
            self._cache['investor_range'] = value
        return self._cache['investor_range']

    @investor_range.setter
    def investor_range(self, value: str):
        """设置投资者范围"""
        self._struct.InvestorRange = value.encode('ascii')[0]
        self._cache['investor_range'] = value

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


# =============================================================================
# UserSession - 用户会话
# =============================================================================


class ExchangeMarginRate(CapsuleStruct):
    """交易所保证金率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("LongMarginRatioByMoney", ctypes.c_double), # 多头保证金率
            ("LongMarginRatioByVolume", ctypes.c_double),# 多头保证金费
            ("ShortMarginRatioByMoney", ctypes.c_double),# 空头保证金率
            ("ShortMarginRatioByVolume", ctypes.c_double),# 空头保证金费
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "ExchangeMarginRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
        "long_margin_ratio_by_money": "LongMarginRatioByMoney",
        "long_margin_ratio_by_volume": "LongMarginRatioByVolume",
        "short_margin_ratio_by_money": "ShortMarginRatioByMoney",
        "short_margin_ratio_by_volume": "ShortMarginRatioByVolume",
        "exchange_id": "ExchangeID",
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
    def long_margin_ratio_by_money(self) -> float:
        """多头保证金率"""
        return self._struct.LongMarginRatioByMoney

    @long_margin_ratio_by_money.setter
    def long_margin_ratio_by_money(self, value: float):
        """设置多头保证金率"""
        self._struct.LongMarginRatioByMoney = value

    @property
    def long_margin_ratio_by_volume(self) -> float:
        """多头保证金费"""
        return self._struct.LongMarginRatioByVolume

    @long_margin_ratio_by_volume.setter
    def long_margin_ratio_by_volume(self, value: float):
        """设置多头保证金费"""
        self._struct.LongMarginRatioByVolume = value

    @property
    def short_margin_ratio_by_money(self) -> float:
        """空头保证金率"""
        return self._struct.ShortMarginRatioByMoney

    @short_margin_ratio_by_money.setter
    def short_margin_ratio_by_money(self, value: float):
        """设置空头保证金率"""
        self._struct.ShortMarginRatioByMoney = value

    @property
    def short_margin_ratio_by_volume(self) -> float:
        """空头保证金费"""
        return self._struct.ShortMarginRatioByVolume

    @short_margin_ratio_by_volume.setter
    def short_margin_ratio_by_volume(self, value: float):
        """设置空头保证金费"""
        self._struct.ShortMarginRatioByVolume = value

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



class ExchangeMarginRateAdjust(CapsuleStruct):
    """交易所保证金率调整"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("LongMarginRatioByMoney", ctypes.c_double), # 跟随交易所投资者多头保证金率
            ("LongMarginRatioByVolume", ctypes.c_double),# 跟随交易所投资者多头保证金费
            ("ShortMarginRatioByMoney", ctypes.c_double),# 跟随交易所投资者空头保证金率
            ("ShortMarginRatioByVolume", ctypes.c_double),# 跟随交易所投资者空头保证金费
            ("ExchLongMarginRatioByMoney", ctypes.c_double),# 交易所多头保证金率
            ("ExchLongMarginRatioByVolume", ctypes.c_double),# 交易所多头保证金费
            ("ExchShortMarginRatioByMoney", ctypes.c_double),# 交易所空头保证金率
            ("ExchShortMarginRatioByVolume", ctypes.c_double),# 交易所空头保证金费
            ("NoLongMarginRatioByMoney", ctypes.c_double),# 不跟随交易所投资者多头保证金率
            ("NoLongMarginRatioByVolume", ctypes.c_double),# 不跟随交易所投资者多头保证金费
            ("NoShortMarginRatioByMoney", ctypes.c_double),# 不跟随交易所投资者空头保证金率
            ("NoShortMarginRatioByVolume", ctypes.c_double),# 不跟随交易所投资者空头保证金费
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "ExchangeMarginRateAdjust"

    _field_mappings = {
        "broker_id": "BrokerID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
        "long_margin_ratio_by_money": "LongMarginRatioByMoney",
        "long_margin_ratio_by_volume": "LongMarginRatioByVolume",
        "short_margin_ratio_by_money": "ShortMarginRatioByMoney",
        "short_margin_ratio_by_volume": "ShortMarginRatioByVolume",
        "exch_long_margin_ratio_by_money": "ExchLongMarginRatioByMoney",
        "exch_long_margin_ratio_by_volume": "ExchLongMarginRatioByVolume",
        "exch_short_margin_ratio_by_money": "ExchShortMarginRatioByMoney",
        "exch_short_margin_ratio_by_volume": "ExchShortMarginRatioByVolume",
        "no_long_margin_ratio_by_money": "NoLongMarginRatioByMoney",
        "no_long_margin_ratio_by_volume": "NoLongMarginRatioByVolume",
        "no_short_margin_ratio_by_money": "NoShortMarginRatioByMoney",
        "no_short_margin_ratio_by_volume": "NoShortMarginRatioByVolume",
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
    def long_margin_ratio_by_money(self) -> float:
        """跟随交易所投资者多头保证金率"""
        return self._struct.LongMarginRatioByMoney

    @long_margin_ratio_by_money.setter
    def long_margin_ratio_by_money(self, value: float):
        """设置跟随交易所投资者多头保证金率"""
        self._struct.LongMarginRatioByMoney = value

    @property
    def long_margin_ratio_by_volume(self) -> float:
        """跟随交易所投资者多头保证金费"""
        return self._struct.LongMarginRatioByVolume

    @long_margin_ratio_by_volume.setter
    def long_margin_ratio_by_volume(self, value: float):
        """设置跟随交易所投资者多头保证金费"""
        self._struct.LongMarginRatioByVolume = value

    @property
    def short_margin_ratio_by_money(self) -> float:
        """跟随交易所投资者空头保证金率"""
        return self._struct.ShortMarginRatioByMoney

    @short_margin_ratio_by_money.setter
    def short_margin_ratio_by_money(self, value: float):
        """设置跟随交易所投资者空头保证金率"""
        self._struct.ShortMarginRatioByMoney = value

    @property
    def short_margin_ratio_by_volume(self) -> float:
        """跟随交易所投资者空头保证金费"""
        return self._struct.ShortMarginRatioByVolume

    @short_margin_ratio_by_volume.setter
    def short_margin_ratio_by_volume(self, value: float):
        """设置跟随交易所投资者空头保证金费"""
        self._struct.ShortMarginRatioByVolume = value

    @property
    def exch_long_margin_ratio_by_money(self) -> float:
        """交易所多头保证金率"""
        return self._struct.ExchLongMarginRatioByMoney

    @exch_long_margin_ratio_by_money.setter
    def exch_long_margin_ratio_by_money(self, value: float):
        """设置交易所多头保证金率"""
        self._struct.ExchLongMarginRatioByMoney = value

    @property
    def exch_long_margin_ratio_by_volume(self) -> float:
        """交易所多头保证金费"""
        return self._struct.ExchLongMarginRatioByVolume

    @exch_long_margin_ratio_by_volume.setter
    def exch_long_margin_ratio_by_volume(self, value: float):
        """设置交易所多头保证金费"""
        self._struct.ExchLongMarginRatioByVolume = value

    @property
    def exch_short_margin_ratio_by_money(self) -> float:
        """交易所空头保证金率"""
        return self._struct.ExchShortMarginRatioByMoney

    @exch_short_margin_ratio_by_money.setter
    def exch_short_margin_ratio_by_money(self, value: float):
        """设置交易所空头保证金率"""
        self._struct.ExchShortMarginRatioByMoney = value

    @property
    def exch_short_margin_ratio_by_volume(self) -> float:
        """交易所空头保证金费"""
        return self._struct.ExchShortMarginRatioByVolume

    @exch_short_margin_ratio_by_volume.setter
    def exch_short_margin_ratio_by_volume(self, value: float):
        """设置交易所空头保证金费"""
        self._struct.ExchShortMarginRatioByVolume = value

    @property
    def no_long_margin_ratio_by_money(self) -> float:
        """不跟随交易所投资者多头保证金率"""
        return self._struct.NoLongMarginRatioByMoney

    @no_long_margin_ratio_by_money.setter
    def no_long_margin_ratio_by_money(self, value: float):
        """设置不跟随交易所投资者多头保证金率"""
        self._struct.NoLongMarginRatioByMoney = value

    @property
    def no_long_margin_ratio_by_volume(self) -> float:
        """不跟随交易所投资者多头保证金费"""
        return self._struct.NoLongMarginRatioByVolume

    @no_long_margin_ratio_by_volume.setter
    def no_long_margin_ratio_by_volume(self, value: float):
        """设置不跟随交易所投资者多头保证金费"""
        self._struct.NoLongMarginRatioByVolume = value

    @property
    def no_short_margin_ratio_by_money(self) -> float:
        """不跟随交易所投资者空头保证金率"""
        return self._struct.NoShortMarginRatioByMoney

    @no_short_margin_ratio_by_money.setter
    def no_short_margin_ratio_by_money(self, value: float):
        """设置不跟随交易所投资者空头保证金率"""
        self._struct.NoShortMarginRatioByMoney = value

    @property
    def no_short_margin_ratio_by_volume(self) -> float:
        """不跟随交易所投资者空头保证金费"""
        return self._struct.NoShortMarginRatioByVolume

    @no_short_margin_ratio_by_volume.setter
    def no_short_margin_ratio_by_volume(self, value: float):
        """设置不跟随交易所投资者空头保证金费"""
        self._struct.NoShortMarginRatioByVolume = value

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



class MMInstrumentCommissionRate(CapsuleStruct):
    """做市商合约手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestorRange", ctypes.c_char),            # 投资者范围
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("OpenRatioByMoney", ctypes.c_double),       # 开仓手续费率
            ("OpenRatioByVolume", ctypes.c_double),      # 开仓手续费
            ("CloseRatioByMoney", ctypes.c_double),      # 平仓手续费率
            ("CloseRatioByVolume", ctypes.c_double),     # 平仓手续费
            ("CloseTodayRatioByMoney", ctypes.c_double),# 平今手续费率
            ("CloseTodayRatioByVolume", ctypes.c_double),# 平今手续费
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "MMInstrumentCommissionRate"

    _field_mappings = {
        "reserve1": "reserve1",
        "investor_range": "InvestorRange",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "open_ratio_by_money": "OpenRatioByMoney",
        "open_ratio_by_volume": "OpenRatioByVolume",
        "close_ratio_by_money": "CloseRatioByMoney",
        "close_ratio_by_volume": "CloseRatioByVolume",
        "close_today_ratio_by_money": "CloseTodayRatioByMoney",
        "close_today_ratio_by_volume": "CloseTodayRatioByVolume",
        "instrument_id": "InstrumentID",
    }

    @property
    def investor_range(self) -> str:
        """投资者范围"""
        if 'investor_range' not in self._cache:
            value = self._struct.InvestorRange.decode('ascii')
            self._cache['investor_range'] = value
        return self._cache['investor_range']

    @investor_range.setter
    def investor_range(self, value: str):
        """设置投资者范围"""
        self._struct.InvestorRange = value.encode('ascii')[0]
        self._cache['investor_range'] = value

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
    def open_ratio_by_money(self) -> float:
        """开仓手续费率"""
        return self._struct.OpenRatioByMoney

    @open_ratio_by_money.setter
    def open_ratio_by_money(self, value: float):
        """设置开仓手续费率"""
        self._struct.OpenRatioByMoney = value

    @property
    def open_ratio_by_volume(self) -> float:
        """开仓手续费"""
        return self._struct.OpenRatioByVolume

    @open_ratio_by_volume.setter
    def open_ratio_by_volume(self, value: float):
        """设置开仓手续费"""
        self._struct.OpenRatioByVolume = value

    @property
    def close_ratio_by_money(self) -> float:
        """平仓手续费率"""
        return self._struct.CloseRatioByMoney

    @close_ratio_by_money.setter
    def close_ratio_by_money(self, value: float):
        """设置平仓手续费率"""
        self._struct.CloseRatioByMoney = value

    @property
    def close_ratio_by_volume(self) -> float:
        """平仓手续费"""
        return self._struct.CloseRatioByVolume

    @close_ratio_by_volume.setter
    def close_ratio_by_volume(self, value: float):
        """设置平仓手续费"""
        self._struct.CloseRatioByVolume = value

    @property
    def close_today_ratio_by_money(self) -> float:
        """平今手续费率"""
        return self._struct.CloseTodayRatioByMoney

    @close_today_ratio_by_money.setter
    def close_today_ratio_by_money(self, value: float):
        """设置平今手续费率"""
        self._struct.CloseTodayRatioByMoney = value

    @property
    def close_today_ratio_by_volume(self) -> float:
        """平今手续费"""
        return self._struct.CloseTodayRatioByVolume

    @close_today_ratio_by_volume.setter
    def close_today_ratio_by_volume(self, value: float):
        """设置平今手续费"""
        self._struct.CloseTodayRatioByVolume = value

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



class MMOptionInstrCommRate(CapsuleStruct):
    """做市商期权手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestorRange", ctypes.c_char),            # 投资者范围
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("OpenRatioByMoney", ctypes.c_double),       # 开仓手续费率
            ("OpenRatioByVolume", ctypes.c_double),      # 开仓手续费
            ("CloseRatioByMoney", ctypes.c_double),      # 平仓手续费率
            ("CloseRatioByVolume", ctypes.c_double),     # 平仓手续费
            ("CloseTodayRatioByMoney", ctypes.c_double),# 平今手续费率
            ("CloseTodayRatioByVolume", ctypes.c_double),# 平今手续费
            ("StrikeRatioByMoney", ctypes.c_double),     # 执行手续费率
            ("StrikeRatioByVolume", ctypes.c_double),    # 执行手续费
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "MMOptionInstrCommRate"

    _field_mappings = {
        "reserve1": "reserve1",
        "investor_range": "InvestorRange",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "open_ratio_by_money": "OpenRatioByMoney",
        "open_ratio_by_volume": "OpenRatioByVolume",
        "close_ratio_by_money": "CloseRatioByMoney",
        "close_ratio_by_volume": "CloseRatioByVolume",
        "close_today_ratio_by_money": "CloseTodayRatioByMoney",
        "close_today_ratio_by_volume": "CloseTodayRatioByVolume",
        "strike_ratio_by_money": "StrikeRatioByMoney",
        "strike_ratio_by_volume": "StrikeRatioByVolume",
        "instrument_id": "InstrumentID",
    }

    @property
    def investor_range(self) -> str:
        """投资者范围"""
        if 'investor_range' not in self._cache:
            value = self._struct.InvestorRange.decode('ascii')
            self._cache['investor_range'] = value
        return self._cache['investor_range']

    @investor_range.setter
    def investor_range(self, value: str):
        """设置投资者范围"""
        self._struct.InvestorRange = value.encode('ascii')[0]
        self._cache['investor_range'] = value

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
    def open_ratio_by_money(self) -> float:
        """开仓手续费率"""
        return self._struct.OpenRatioByMoney

    @open_ratio_by_money.setter
    def open_ratio_by_money(self, value: float):
        """设置开仓手续费率"""
        self._struct.OpenRatioByMoney = value

    @property
    def open_ratio_by_volume(self) -> float:
        """开仓手续费"""
        return self._struct.OpenRatioByVolume

    @open_ratio_by_volume.setter
    def open_ratio_by_volume(self, value: float):
        """设置开仓手续费"""
        self._struct.OpenRatioByVolume = value

    @property
    def close_ratio_by_money(self) -> float:
        """平仓手续费率"""
        return self._struct.CloseRatioByMoney

    @close_ratio_by_money.setter
    def close_ratio_by_money(self, value: float):
        """设置平仓手续费率"""
        self._struct.CloseRatioByMoney = value

    @property
    def close_ratio_by_volume(self) -> float:
        """平仓手续费"""
        return self._struct.CloseRatioByVolume

    @close_ratio_by_volume.setter
    def close_ratio_by_volume(self, value: float):
        """设置平仓手续费"""
        self._struct.CloseRatioByVolume = value

    @property
    def close_today_ratio_by_money(self) -> float:
        """平今手续费率"""
        return self._struct.CloseTodayRatioByMoney

    @close_today_ratio_by_money.setter
    def close_today_ratio_by_money(self, value: float):
        """设置平今手续费率"""
        self._struct.CloseTodayRatioByMoney = value

    @property
    def close_today_ratio_by_volume(self) -> float:
        """平今手续费"""
        return self._struct.CloseTodayRatioByVolume

    @close_today_ratio_by_volume.setter
    def close_today_ratio_by_volume(self, value: float):
        """设置平今手续费"""
        self._struct.CloseTodayRatioByVolume = value

    @property
    def strike_ratio_by_money(self) -> float:
        """执行手续费率"""
        return self._struct.StrikeRatioByMoney

    @strike_ratio_by_money.setter
    def strike_ratio_by_money(self, value: float):
        """设置执行手续费率"""
        self._struct.StrikeRatioByMoney = value

    @property
    def strike_ratio_by_volume(self) -> float:
        """执行手续费"""
        return self._struct.StrikeRatioByVolume

    @strike_ratio_by_volume.setter
    def strike_ratio_by_volume(self, value: float):
        """设置执行手续费"""
        self._struct.StrikeRatioByVolume = value

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



class InstrumentOrderCommRate(CapsuleStruct):
    """报单手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestorRange", ctypes.c_char),            # 投资者范围
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("OrderCommByVolume", ctypes.c_double),      # 报单手续费
            ("OrderActionCommByVolume", ctypes.c_double),# 撤单手续费
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("OrderCommByTrade", ctypes.c_double),       # 报单手续费
            ("OrderActionCommByTrade", ctypes.c_double),# 撤单手续费
        ]

    _capsule_name = "InstrumentOrderCommRate"

    _field_mappings = {
        "reserve1": "reserve1",
        "investor_range": "InvestorRange",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "hedge_flag": "HedgeFlag",
        "order_comm_by_volume": "OrderCommByVolume",
        "order_action_comm_by_volume": "OrderActionCommByVolume",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "instrument_id": "InstrumentID",
        "order_comm_by_trade": "OrderCommByTrade",
        "order_action_comm_by_trade": "OrderActionCommByTrade",
    }

    @property
    def investor_range(self) -> str:
        """投资者范围"""
        if 'investor_range' not in self._cache:
            value = self._struct.InvestorRange.decode('ascii')
            self._cache['investor_range'] = value
        return self._cache['investor_range']

    @investor_range.setter
    def investor_range(self, value: str):
        """设置投资者范围"""
        self._struct.InvestorRange = value.encode('ascii')[0]
        self._cache['investor_range'] = value

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
    def order_comm_by_volume(self) -> float:
        """报单手续费"""
        return self._struct.OrderCommByVolume

    @order_comm_by_volume.setter
    def order_comm_by_volume(self, value: float):
        """设置报单手续费"""
        self._struct.OrderCommByVolume = value

    @property
    def order_action_comm_by_volume(self) -> float:
        """撤单手续费"""
        return self._struct.OrderActionCommByVolume

    @order_action_comm_by_volume.setter
    def order_action_comm_by_volume(self, value: float):
        """设置撤单手续费"""
        self._struct.OrderActionCommByVolume = value

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
    def order_comm_by_trade(self) -> float:
        """报单手续费"""
        return self._struct.OrderCommByTrade

    @order_comm_by_trade.setter
    def order_comm_by_trade(self, value: float):
        """设置报单手续费"""
        self._struct.OrderCommByTrade = value

    @property
    def order_action_comm_by_trade(self) -> float:
        """撤单手续费"""
        return self._struct.OrderActionCommByTrade

    @order_action_comm_by_trade.setter
    def order_action_comm_by_trade(self, value: float):
        """设置撤单手续费"""
        self._struct.OrderActionCommByTrade = value




class OptionInstrTradeCost(CapsuleStruct):
    """期权交易成本"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("FixedMargin", ctypes.c_double),            # 期权合约保证金不变部分
            ("MiniMargin", ctypes.c_double),             # 期权合约最小保证金
            ("Royalty", ctypes.c_double),                # 期权合约权利金
            ("ExchFixedMargin", ctypes.c_double),        # 交易所期权合约保证金不变部分
            ("ExchMiniMargin", ctypes.c_double),         # 交易所期权合约最小保证金
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "OptionInstrTradeCost"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
        "fixed_margin": "FixedMargin",
        "mini_margin": "MiniMargin",
        "royalty": "Royalty",
        "exch_fixed_margin": "ExchFixedMargin",
        "exch_mini_margin": "ExchMiniMargin",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
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
    def fixed_margin(self) -> float:
        """期权合约保证金不变部分"""
        return self._struct.FixedMargin

    @fixed_margin.setter
    def fixed_margin(self, value: float):
        """设置期权合约保证金不变部分"""
        self._struct.FixedMargin = value

    @property
    def mini_margin(self) -> float:
        """期权合约最小保证金"""
        return self._struct.MiniMargin

    @mini_margin.setter
    def mini_margin(self, value: float):
        """设置期权合约最小保证金"""
        self._struct.MiniMargin = value

    @property
    def royalty(self) -> float:
        """期权合约权利金"""
        return self._struct.Royalty

    @royalty.setter
    def royalty(self, value: float):
        """设置期权合约权利金"""
        self._struct.Royalty = value

    @property
    def exch_fixed_margin(self) -> float:
        """交易所期权合约保证金不变部分"""
        return self._struct.ExchFixedMargin

    @exch_fixed_margin.setter
    def exch_fixed_margin(self, value: float):
        """设置交易所期权合约保证金不变部分"""
        self._struct.ExchFixedMargin = value

    @property
    def exch_mini_margin(self) -> float:
        """交易所期权合约最小保证金"""
        return self._struct.ExchMiniMargin

    @exch_mini_margin.setter
    def exch_mini_margin(self, value: float):
        """设置交易所期权合约最小保证金"""
        self._struct.ExchMiniMargin = value

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



class OptionInstrCommRate(CapsuleStruct):
    """期权手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestorRange", ctypes.c_char),            # 投资者范围
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("OpenRatioByMoney", ctypes.c_double),       # 开仓手续费率
            ("OpenRatioByVolume", ctypes.c_double),      # 开仓手续费
            ("CloseRatioByMoney", ctypes.c_double),      # 平仓手续费率
            ("CloseRatioByVolume", ctypes.c_double),     # 平仓手续费
            ("CloseTodayRatioByMoney", ctypes.c_double),# 平今手续费率
            ("CloseTodayRatioByVolume", ctypes.c_double),# 平今手续费
            ("StrikeRatioByMoney", ctypes.c_double),     # 执行手续费率
            ("StrikeRatioByVolume", ctypes.c_double),    # 执行手续费
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "OptionInstrCommRate"

    _field_mappings = {
        "reserve1": "reserve1",
        "investor_range": "InvestorRange",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "open_ratio_by_money": "OpenRatioByMoney",
        "open_ratio_by_volume": "OpenRatioByVolume",
        "close_ratio_by_money": "CloseRatioByMoney",
        "close_ratio_by_volume": "CloseRatioByVolume",
        "close_today_ratio_by_money": "CloseTodayRatioByMoney",
        "close_today_ratio_by_volume": "CloseTodayRatioByVolume",
        "strike_ratio_by_money": "StrikeRatioByMoney",
        "strike_ratio_by_volume": "StrikeRatioByVolume",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "instrument_id": "InstrumentID",
    }

    @property
    def investor_range(self) -> str:
        """投资者范围"""
        if 'investor_range' not in self._cache:
            value = self._struct.InvestorRange.decode('ascii')
            self._cache['investor_range'] = value
        return self._cache['investor_range']

    @investor_range.setter
    def investor_range(self, value: str):
        """设置投资者范围"""
        self._struct.InvestorRange = value.encode('ascii')[0]
        self._cache['investor_range'] = value

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
    def open_ratio_by_money(self) -> float:
        """开仓手续费率"""
        return self._struct.OpenRatioByMoney

    @open_ratio_by_money.setter
    def open_ratio_by_money(self, value: float):
        """设置开仓手续费率"""
        self._struct.OpenRatioByMoney = value

    @property
    def open_ratio_by_volume(self) -> float:
        """开仓手续费"""
        return self._struct.OpenRatioByVolume

    @open_ratio_by_volume.setter
    def open_ratio_by_volume(self, value: float):
        """设置开仓手续费"""
        self._struct.OpenRatioByVolume = value

    @property
    def close_ratio_by_money(self) -> float:
        """平仓手续费率"""
        return self._struct.CloseRatioByMoney

    @close_ratio_by_money.setter
    def close_ratio_by_money(self, value: float):
        """设置平仓手续费率"""
        self._struct.CloseRatioByMoney = value

    @property
    def close_ratio_by_volume(self) -> float:
        """平仓手续费"""
        return self._struct.CloseRatioByVolume

    @close_ratio_by_volume.setter
    def close_ratio_by_volume(self, value: float):
        """设置平仓手续费"""
        self._struct.CloseRatioByVolume = value

    @property
    def close_today_ratio_by_money(self) -> float:
        """平今手续费率"""
        return self._struct.CloseTodayRatioByMoney

    @close_today_ratio_by_money.setter
    def close_today_ratio_by_money(self, value: float):
        """设置平今手续费率"""
        self._struct.CloseTodayRatioByMoney = value

    @property
    def close_today_ratio_by_volume(self) -> float:
        """平今手续费"""
        return self._struct.CloseTodayRatioByVolume

    @close_today_ratio_by_volume.setter
    def close_today_ratio_by_volume(self, value: float):
        """设置平今手续费"""
        self._struct.CloseTodayRatioByVolume = value

    @property
    def strike_ratio_by_money(self) -> float:
        """执行手续费率"""
        return self._struct.StrikeRatioByMoney

    @strike_ratio_by_money.setter
    def strike_ratio_by_money(self, value: float):
        """设置执行手续费率"""
        self._struct.StrikeRatioByMoney = value

    @property
    def strike_ratio_by_volume(self) -> float:
        """执行手续费"""
        return self._struct.StrikeRatioByVolume

    @strike_ratio_by_volume.setter
    def strike_ratio_by_volume(self, value: float):
        """设置执行手续费"""
        self._struct.StrikeRatioByVolume = value

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



