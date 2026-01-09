"""
Req
"""

import ctypes
from PcCTP.types.base import CapsuleStruct
from typing import Dict


class QryMulticastInstrument(CapsuleStruct):
    """查询组播行情合约"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TopicID", ctypes.c_int),                 # 主题号
            ("reserve1", ctypes.c_char * 31),          # 保留的无效字段
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
        ]

    _capsule_name = "QryMulticastInstrument"

    _field_mappings = {
        "topic_id": "TopicID",
        "reserve1": "reserve1",
        "instrument_id": "InstrumentID",
    }

    @property
    def topic_id(self) -> int:
        """主题号"""
        return self._struct.TopicID

    @topic_id.setter
    def topic_id(self, value: int):
        """设置主题号"""
        self._struct.TopicID = value

    @property
    def reserve1(self) -> str:
        """保留的无效字段"""
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
# RspAuthenticate - 客户端认证响应
# =============================================================================


class QryMaxOrderVolume(CapsuleStruct):
    """查询最大报单数量"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("Direction", ctypes.c_char),                # 买卖方向
            ("OffsetFlag", ctypes.c_char),               # 开平标志
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("MaxVolume", ctypes.c_int),                 # 最大允许报单数量
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryMaxOrderVolume"

    _field_mappings: Dict[str, str] = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "direction": "Direction",
        "offset_flag": "OffsetFlag",
        "hedge_flag": "HedgeFlag",
        "max_volume": "MaxVolume",
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
    def offset_flag(self) -> str:
        """开平标志"""
        if 'offset_flag' not in self._cache:
            value = self._struct.OffsetFlag.decode('ascii')
            self._cache['offset_flag'] = value
        return self._cache['offset_flag']

    @offset_flag.setter
    def offset_flag(self, value: str):
        """设置开平标志"""
        self._struct.OffsetFlag = value.encode('ascii')[0]
        self._cache['offset_flag'] = value

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
    def max_volume(self) -> int:
        """最大允许报单数量"""
        return self._struct.MaxVolume

    @max_volume.setter
    def max_volume(self, value: int):
        """设置最大允许报单数量"""
        self._struct.MaxVolume = value

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




class QryOrder(CapsuleStruct):
    """查询报单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("OrderSysID", ctypes.c_char * 21),           # 报单编号
            ("InsertTimeStart", ctypes.c_char * 9),       # 开始时间
            ("InsertTimeEnd", ctypes.c_char * 9),         # 结束时间
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
        ]

    _capsule_name = "QryOrder"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "order_sys_id": "OrderSysID",
        "insert_time_start": "InsertTimeStart",
        "insert_time_end": "InsertTimeEnd",
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
    def order_sys_id(self) -> str:
        """报单编号"""
        if 'order_sys_id' not in self._cache:
            value = self._struct.OrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['order_sys_id'] = value
        return self._cache['order_sys_id']

    @order_sys_id.setter
    def order_sys_id(self, value: str):
        """设置报单编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.OrderSysID = encoded
        self._cache['order_sys_id'] = value

    @property
    def insert_time_start(self) -> str:
        """开始时间"""
        if 'insert_time_start' not in self._cache:
            value = self._struct.InsertTimeStart.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_start'] = value
        return self._cache['insert_time_start']

    @insert_time_start.setter
    def insert_time_start(self, value: str):
        """设置开始时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeStart = encoded
        self._cache['insert_time_start'] = value

    @property
    def insert_time_end(self) -> str:
        """结束时间"""
        if 'insert_time_end' not in self._cache:
            value = self._struct.InsertTimeEnd.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_end'] = value
        return self._cache['insert_time_end']

    @insert_time_end.setter
    def insert_time_end(self, value: str):
        """设置结束时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeEnd = encoded
        self._cache['insert_time_end'] = value

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



class QryTrade(CapsuleStruct):
    """查询成交"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("TradeID", ctypes.c_char * 21),              # 成交编号
            ("TradeTimeStart", ctypes.c_char * 9),        # 开始时间
            ("TradeTimeEnd", ctypes.c_char * 9),          # 结束时间
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
        ]

    _capsule_name = "QryTrade"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "trade_id": "TradeID",
        "trade_time_start": "TradeTimeStart",
        "trade_time_end": "TradeTimeEnd",
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
    def trade_time_start(self) -> str:
        """开始时间"""
        if 'trade_time_start' not in self._cache:
            value = self._struct.TradeTimeStart.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time_start'] = value
        return self._cache['trade_time_start']

    @trade_time_start.setter
    def trade_time_start(self, value: str):
        """设置开始时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTimeStart = encoded
        self._cache['trade_time_start'] = value

    @property
    def trade_time_end(self) -> str:
        """结束时间"""
        if 'trade_time_end' not in self._cache:
            value = self._struct.TradeTimeEnd.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time_end'] = value
        return self._cache['trade_time_end']

    @trade_time_end.setter
    def trade_time_end(self, value: str):
        """设置结束时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTimeEnd = encoded
        self._cache['trade_time_end'] = value

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



class QryInvestorPosition(CapsuleStruct):
    """查询投资者持仓"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
        ]

    _capsule_name = "QryInvestorPosition"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryTradingAccount(CapsuleStruct):
    """查询资金账户"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
            ("BizType", ctypes.c_char),                   # 业务类型
            ("AccountID", ctypes.c_char * 13),            # 投资者帐号
        ]

    _capsule_name = "QryTradingAccount"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "currency_id": "CurrencyID",
        "biz_type": "BizType",
        "account_id": "AccountID",
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
    def account_id(self) -> str:
        """投资者帐号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置投资者帐号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AccountID = encoded
        self._cache['account_id'] = value



class QryInvestor(CapsuleStruct):
    """查询投资者"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
        ]

    _capsule_name = "QryInvestor"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
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



class QryTradingCode(CapsuleStruct):
    """查询交易编码"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("ClientID", ctypes.c_char * 11),             # 客户代码
            ("ClientIDType", ctypes.c_char),              # 交易编码类型
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
        ]

    _capsule_name = "QryTradingCode"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "exchange_id": "ExchangeID",
        "client_id": "ClientID",
        "client_id_type": "ClientIDType",
        "invest_unit_id": "InvestUnitID",
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



class QryInstrumentMarginRate(CapsuleStruct):
    """查询合约保证金率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                 # 投机套保标志
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
        ]

    _capsule_name = "QryInstrumentMarginRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
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



class QryInstrumentCommissionRate(CapsuleStruct):
    """查询手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
        ]

    _capsule_name = "QryInstrumentCommissionRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryUserSession(CapsuleStruct):
    """查询用户会话"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("FrontID", ctypes.c_int),                    # 前置编号
            ("SessionID", ctypes.c_int),                  # 会话编号
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("UserID", ctypes.c_char * 16),               # 用户代码
        ]

    _capsule_name = "QryUserSession"

    _field_mappings = {
        "front_id": "FrontID",
        "session_id": "SessionID",
        "broker_id": "BrokerID",
        "user_id": "UserID",
    }

    @property
    def front_id(self) -> int:
        """前置编号"""
        return self._struct.FrontID

    @front_id.setter
    def front_id(self, value: int):
        """设置前置编号"""
        self._struct.FrontID = value

    @property
    def session_id(self) -> int:
        """会话编号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话编号"""
        self._struct.SessionID = value

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
    def user_id(self) -> str:
        """用户代码"""
        if 'user_id' not in self._cache:
            value = self._struct.UserID.rstrip(b'\x00').decode('ascii')
            self._cache['user_id'] = value
        return self._cache['user_id']

    @user_id.setter
    def user_id(self, value: str):
        """设置用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.UserID = encoded
        self._cache['user_id'] = value



class QryExchange(CapsuleStruct):
    """查询交易所"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
        ]

    _capsule_name = "QryExchange"

    _field_mappings = {
        "exchange_id": "ExchangeID",
    }

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



class QryProduct(CapsuleStruct):
    """查询产品"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ProductClass", ctypes.c_char),              # 产品类型
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("ProductID", ctypes.c_char * 81),            # 产品代码
        ]

    _capsule_name = "QryProduct"

    _field_mappings = {
        "reserve1": "reserve1",
        "product_class": "ProductClass",
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
    }

    @property
    def product_class(self) -> str:
        """产品类型"""
        if 'product_class' not in self._cache:
            value = self._struct.ProductClass.decode('ascii')
            self._cache['product_class'] = value
        return self._cache['product_class']

    @product_class.setter
    def product_class(self, value: str):
        """设置产品类型"""
        self._struct.ProductClass = value.encode('ascii')[0]
        self._cache['product_class'] = value

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
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value



class QryInstrument(CapsuleStruct):
    """查询合约"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("reserve2", ctypes.c_char * 31),             # 保留的无效字段
            ("reserve3", ctypes.c_char * 31),             # 保留的无效字段
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
            ("ExchangeInstID", ctypes.c_char * 81),       # 合约在交易所的代码
            ("ProductID", ctypes.c_char * 81),            # 产品代码
        ]

    _capsule_name = "QryInstrument"

    _field_mappings = {
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "reserve2": "reserve2",
        "reserve3": "reserve3",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
        "product_id": "ProductID",
    }

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

    @property
    def exchange_inst_id(self) -> str:
        """合约在交易所的代码"""
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

    @property
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value



class QryDepthMarketData(CapsuleStruct):
    """查询行情"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
            ("ProductClass", ctypes.c_char),              # 产品类型
        ]

    _capsule_name = "QryDepthMarketData"

    _field_mappings = {
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "product_class": "ProductClass",
    }

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

    @property
    def product_class(self) -> str:
        """产品类型"""
        if 'product_class' not in self._cache:
            value = self._struct.ProductClass.decode('ascii')
            self._cache['product_class'] = value
        return self._cache['product_class']

    @product_class.setter
    def product_class(self, value: str):
        """设置产品类型"""
        self._struct.ProductClass = value.encode('ascii')[0]
        self._cache['product_class'] = value



class QrySettlementInfo(CapsuleStruct):
    """查询投资者结算结果"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("TradingDay", ctypes.c_char * 9),            # 交易日
            ("AccountID", ctypes.c_char * 13),            # 投资者帐号
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
        ]

    _capsule_name = "QrySettlementInfo"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "trading_day": "TradingDay",
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
    def account_id(self) -> str:
        """投资者帐号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置投资者帐号"""
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



class QryTransferBank(CapsuleStruct):
    """查询转帐银行"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BankID", ctypes.c_char * 4),                # 银行代码
            ("BankBrchID", ctypes.c_char * 5),            # 银行分中心代码
        ]

    _capsule_name = "QryTransferBank"

    _field_mappings = {
        "bank_id": "BankID",
        "bank_brch_id": "BankBrchID",
    }

    @property
    def bank_id(self) -> str:
        """银行代码"""
        if 'bank_id' not in self._cache:
            value = self._struct.BankID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_id'] = value
        return self._cache['bank_id']

    @bank_id.setter
    def bank_id(self, value: str):
        """设置银行代码"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.BankID = encoded
        self._cache['bank_id'] = value

    @property
    def bank_brch_id(self) -> str:
        """银行分中心代码"""
        if 'bank_brch_id' not in self._cache:
            value = self._struct.BankBrchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_brch_id'] = value
        return self._cache['bank_brch_id']

    @bank_brch_id.setter
    def bank_brch_id(self, value: str):
        """设置银行分中心代码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBrchID = encoded
        self._cache['bank_brch_id'] = value



class QryInvestorPositionDetail(CapsuleStruct):
    """查询投资者持仓明细"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),         # 合约代码
        ]

    _capsule_name = "QryInvestorPositionDetail"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryNotice(CapsuleStruct):
    """查询客户通知"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
        ]

    _capsule_name = "QryNotice"

    _field_mappings = {
        "broker_id": "BrokerID",
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



class QrySettlementInfoConfirm(CapsuleStruct):
    """查询结算信息确认"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("AccountID", ctypes.c_char * 13),            # 投资者帐号
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
        ]

    _capsule_name = "QrySettlementInfoConfirm"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
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
    def account_id(self) -> str:
        """投资者帐号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置投资者帐号"""
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



class QryInvestorPositionCombineDetail(CapsuleStruct):
    """查询投资者持仓明细（组合）"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("reserve1", ctypes.c_char * 31),             # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),            # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),         # 投资单元代码
            ("CombInstrumentID", ctypes.c_char * 81),     # 组合持仓合约编码
        ]

    _capsule_name = "QryInvestorPositionCombineDetail"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "comb_instrument_id": "CombInstrumentID",
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
    def comb_instrument_id(self) -> str:
        """组合持仓合约编码"""
        if 'comb_instrument_id' not in self._cache:
            value = self._struct.CombInstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['comb_instrument_id'] = value
        return self._cache['comb_instrument_id']

    @comb_instrument_id.setter
    def comb_instrument_id(self, value: str):
        """设置组合持仓合约编码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.CombInstrumentID = encoded
        self._cache['comb_instrument_id'] = value




class QryCFMMCTradingAccountKey(CapsuleStruct):
    """查询保证金账户资金账户Key"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
        ]

    _capsule_name = "QryCFMMCTradingAccountKey"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
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



class QryEWarrantOffset(CapsuleStruct):
    """查询质押抵质押品"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("InvestUnitID", ctypes.c_char * 17),          # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
        ]

    _capsule_name = "QryEWarrantOffset"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "exchange_id": "ExchangeID",
        "reserve1": "reserve1",
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



class QryInvestorProductGroupMargin(CapsuleStruct):
    """查询投资者品种/跨品种保证金"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                  # 投机套保标志
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),          # 投资单元代码
            ("ProductGroupID", ctypes.c_char * 81),        # 品种/跨品种标示
        ]

    _capsule_name = "QryInvestorProductGroupMargin"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "product_group_id": "ProductGroupID",
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
    def product_group_id(self) -> str:
        """品种/跨品种标示"""
        if 'product_group_id' not in self._cache:
            value = self._struct.ProductGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['product_group_id'] = value
        return self._cache['product_group_id']

    @product_group_id.setter
    def product_group_id(self, value: str):
        """设置品种/跨品种标示"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductGroupID = encoded
        self._cache['product_group_id'] = value



class QryExchangeMarginRate(CapsuleStruct):
    """查询交易所保证金率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                  # 投机套保标志
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
        ]

    _capsule_name = "QryExchangeMarginRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
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



class QryExchangeMarginRateAdjust(CapsuleStruct):
    """查询交易所调整保证金率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                  # 投机套保标志
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
        ]

    _capsule_name = "QryExchangeMarginRateAdjust"

    _field_mappings = {
        "broker_id": "BrokerID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
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



class QryExchangeRate(CapsuleStruct):
    """查询汇率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("FromCurrencyID", ctypes.c_char * 4),         # 源币种
            ("ToCurrencyID", ctypes.c_char * 4),           # 目标币种
        ]

    _capsule_name = "QryExchangeRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "from_currency_id": "FromCurrencyID",
        "to_currency_id": "ToCurrencyID",
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
    def from_currency_id(self) -> str:
        """源币种"""
        if 'from_currency_id' not in self._cache:
            value = self._struct.FromCurrencyID.rstrip(b'\x00').decode('ascii')
            self._cache['from_currency_id'] = value
        return self._cache['from_currency_id']

    @from_currency_id.setter
    def from_currency_id(self, value: str):
        """设置源币种"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.FromCurrencyID = encoded
        self._cache['from_currency_id'] = value

    @property
    def to_currency_id(self) -> str:
        """目标币种"""
        if 'to_currency_id' not in self._cache:
            value = self._struct.ToCurrencyID.rstrip(b'\x00').decode('ascii')
            self._cache['to_currency_id'] = value
        return self._cache['to_currency_id']

    @to_currency_id.setter
    def to_currency_id(self, value: str):
        """设置目标币种"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.ToCurrencyID = encoded
        self._cache['to_currency_id'] = value



class QrySecAgentACIDMap(CapsuleStruct):
    """查询二级代理资金账户映射"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("UserID", ctypes.c_char * 16),                # 用户代码
            ("AccountID", ctypes.c_char * 13),             # 资金账户
            ("CurrencyID", ctypes.c_char * 4),             # 币种
        ]

    _capsule_name = "QrySecAgentACIDMap"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
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
    def user_id(self) -> str:
        """用户代码"""
        if 'user_id' not in self._cache:
            value = self._struct.UserID.rstrip(b'\x00').decode('ascii')
            self._cache['user_id'] = value
        return self._cache['user_id']

    @user_id.setter
    def user_id(self, value: str):
        """设置用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.UserID = encoded
        self._cache['user_id'] = value

    @property
    def account_id(self) -> str:
        """资金账户"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置资金账户"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AccountID = encoded
        self._cache['account_id'] = value

    @property
    def currency_id(self) -> str:
        """币种"""
        if 'currency_id' not in self._cache:
            value = self._struct.CurrencyID.rstrip(b'\x00').decode('ascii')
            self._cache['currency_id'] = value
        return self._cache['currency_id']

    @currency_id.setter
    def currency_id(self, value: str):
        """设置币种"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.CurrencyID = encoded
        self._cache['currency_id'] = value



class QryProductExchRate(CapsuleStruct):
    """查询产品汇率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("ProductID", ctypes.c_char * 81),             # 产品代码
        ]

    _capsule_name = "QryProductExchRate"

    _field_mappings = {
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
    }

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
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value



class QryProductGroup(CapsuleStruct):
    """查询产品组"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("ProductID", ctypes.c_char * 81),             # 产品代码
        ]

    _capsule_name = "QryProductGroup"

    _field_mappings = {
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
    }

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
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value



class QryMMInstrumentCommissionRate(CapsuleStruct):
    """查询做市商合约手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
        ]

    _capsule_name = "QryMMInstrumentCommissionRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryMMOptionInstrCommRate(CapsuleStruct):
    """查询做市商期权合约手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
        ]

    _capsule_name = "QryMMOptionInstrCommRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryInstrumentOrderCommRate(CapsuleStruct):
    """查询报单手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
        ]

    _capsule_name = "QryInstrumentOrderCommRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QrySecAgentCheckMode(CapsuleStruct):
    """查询二级代理商校验模式"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
        ]

    _capsule_name = "QrySecAgentCheckMode"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
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



class QrySecAgentTradeInfo(CapsuleStruct):
    """查询二级代理商交易信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("BrokerSecAgentID", ctypes.c_char * 13),      # 境外中介机构资金帐号
        ]

    _capsule_name = "QrySecAgentTradeInfo"

    _field_mappings = {
        "broker_id": "BrokerID",
        "broker_sec_agent_id": "BrokerSecAgentID",
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
    def broker_sec_agent_id(self) -> str:
        """境外中介机构资金帐号"""
        if 'broker_sec_agent_id' not in self._cache:
            value = self._struct.BrokerSecAgentID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_sec_agent_id'] = value
        return self._cache['broker_sec_agent_id']

    @broker_sec_agent_id.setter
    def broker_sec_agent_id(self, value: str):
        """设置境外中介机构资金帐号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BrokerSecAgentID = encoded
        self._cache['broker_sec_agent_id'] = value



class QryOptionInstrTradeCost(CapsuleStruct):
    """查询期权交易成本"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("reserve1", ctypes.c_char * 31),              # 保留的无效字段
            ("HedgeFlag", ctypes.c_char),                  # 投机套保标志
            ("InputPrice", ctypes.c_double),               # 期权合约报价
            ("UnderlyingPrice", ctypes.c_double),          # 标的价格,填0则用昨结算价
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),          # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
        ]

    _capsule_name = "QryOptionInstrTradeCost"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "hedge_flag": "HedgeFlag",
        "input_price": "InputPrice",
        "underlying_price": "UnderlyingPrice",
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
    def input_price(self) -> float:
        """期权合约报价"""
        return self._struct.InputPrice

    @input_price.setter
    def input_price(self, value: float):
        """设置期权合约报价"""
        self._struct.InputPrice = value

    @property
    def underlying_price(self) -> float:
        """标价格,填0则用昨结算价"""
        return self._struct.UnderlyingPrice

    @underlying_price.setter
    def underlying_price(self, value: float):
        """设置标价格,填0则用昨结算价"""
        self._struct.UnderlyingPrice = value

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



class QryOptionInstrCommRate(CapsuleStruct):
    """查询期权合约手续费率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryOptionInstrCommRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryExecOrder(CapsuleStruct):
    """查询执行宣告"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ExecOrderSysID", ctypes.c_char * 21),      # 执行宣告编号
            ("InsertTimeStart", ctypes.c_char * 9),      # 开始时间
            ("InsertTimeEnd", ctypes.c_char * 9),        # 结束时间
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryExecOrder"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "exec_order_sys_id": "ExecOrderSysID",
        "insert_time_start": "InsertTimeStart",
        "insert_time_end": "InsertTimeEnd",
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
    def exec_order_sys_id(self) -> str:
        """执行宣告编号"""
        if 'exec_order_sys_id' not in self._cache:
            value = self._struct.ExecOrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_sys_id'] = value
        return self._cache['exec_order_sys_id']

    @exec_order_sys_id.setter
    def exec_order_sys_id(self, value: str):
        """设置执行宣告编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ExecOrderSysID = encoded
        self._cache['exec_order_sys_id'] = value

    @property
    def insert_time_start(self) -> str:
        """开始时间"""
        if 'insert_time_start' not in self._cache:
            value = self._struct.InsertTimeStart.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_start'] = value
        return self._cache['insert_time_start']

    @insert_time_start.setter
    def insert_time_start(self, value: str):
        """设置开始时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeStart = encoded
        self._cache['insert_time_start'] = value

    @property
    def insert_time_end(self) -> str:
        """结束时间"""
        if 'insert_time_end' not in self._cache:
            value = self._struct.InsertTimeEnd.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_end'] = value
        return self._cache['insert_time_end']

    @insert_time_end.setter
    def insert_time_end(self, value: str):
        """设置结束时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeEnd = encoded
        self._cache['insert_time_end'] = value

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



class QryForQuote(CapsuleStruct):
    """查询询价"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InsertTimeStart", ctypes.c_char * 9),      # 开始时间
            ("InsertTimeEnd", ctypes.c_char * 9),        # 结束时间
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryForQuote"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "insert_time_start": "InsertTimeStart",
        "insert_time_end": "InsertTimeEnd",
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
    def insert_time_start(self) -> str:
        """开始时间"""
        if 'insert_time_start' not in self._cache:
            value = self._struct.InsertTimeStart.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_start'] = value
        return self._cache['insert_time_start']

    @insert_time_start.setter
    def insert_time_start(self, value: str):
        """设置开始时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeStart = encoded
        self._cache['insert_time_start'] = value

    @property
    def insert_time_end(self) -> str:
        """结束时间"""
        if 'insert_time_end' not in self._cache:
            value = self._struct.InsertTimeEnd.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_end'] = value
        return self._cache['insert_time_end']

    @insert_time_end.setter
    def insert_time_end(self, value: str):
        """设置结束时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeEnd = encoded
        self._cache['insert_time_end'] = value

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



class QryQuote(CapsuleStruct):
    """查询报价"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("QuoteSysID", ctypes.c_char * 21),          # 报价编号
            ("InsertTimeStart", ctypes.c_char * 9),      # 开始时间
            ("InsertTimeEnd", ctypes.c_char * 9),        # 结束时间
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryQuote"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "quote_sys_id": "QuoteSysID",
        "insert_time_start": "InsertTimeStart",
        "insert_time_end": "InsertTimeEnd",
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
    def quote_sys_id(self) -> str:
        """报价编号"""
        if 'quote_sys_id' not in self._cache:
            value = self._struct.QuoteSysID.rstrip(b'\x00').decode('ascii')
            self._cache['quote_sys_id'] = value
        return self._cache['quote_sys_id']

    @quote_sys_id.setter
    def quote_sys_id(self, value: str):
        """设置报价编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.QuoteSysID = encoded
        self._cache['quote_sys_id'] = value

    @property
    def insert_time_start(self) -> str:
        """开始时间"""
        if 'insert_time_start' not in self._cache:
            value = self._struct.InsertTimeStart.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_start'] = value
        return self._cache['insert_time_start']

    @insert_time_start.setter
    def insert_time_start(self, value: str):
        """设置开始时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeStart = encoded
        self._cache['insert_time_start'] = value

    @property
    def insert_time_end(self) -> str:
        """结束时间"""
        if 'insert_time_end' not in self._cache:
            value = self._struct.InsertTimeEnd.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_end'] = value
        return self._cache['insert_time_end']

    @insert_time_end.setter
    def insert_time_end(self, value: str):
        """设置结束时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeEnd = encoded
        self._cache['insert_time_end'] = value

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



class QryOptionSelfClose(CapsuleStruct):
    """查询期权自对冲"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("OptionSelfCloseSysID", ctypes.c_char * 21), # 期权自对冲编号
            ("InsertTimeStart", ctypes.c_char * 9),      # 开始时间
            ("InsertTimeEnd", ctypes.c_char * 9),        # 结束时间
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryOptionSelfClose"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "option_self_close_sys_id": "OptionSelfCloseSysID",
        "insert_time_start": "InsertTimeStart",
        "insert_time_end": "InsertTimeEnd",
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
    def option_self_close_sys_id(self) -> str:
        """期权自对冲编号"""
        if 'option_self_close_sys_id' not in self._cache:
            value = self._struct.OptionSelfCloseSysID.rstrip(b'\x00').decode('ascii')
            self._cache['option_self_close_sys_id'] = value
        return self._cache['option_self_close_sys_id']

    @option_self_close_sys_id.setter
    def option_self_close_sys_id(self, value: str):
        """设置期权自对冲编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.OptionSelfCloseSysID = encoded
        self._cache['option_self_close_sys_id'] = value

    @property
    def insert_time_start(self) -> str:
        """开始时间"""
        if 'insert_time_start' not in self._cache:
            value = self._struct.InsertTimeStart.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_start'] = value
        return self._cache['insert_time_start']

    @insert_time_start.setter
    def insert_time_start(self, value: str):
        """设置开始时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeStart = encoded
        self._cache['insert_time_start'] = value

    @property
    def insert_time_end(self) -> str:
        """结束时间"""
        if 'insert_time_end' not in self._cache:
            value = self._struct.InsertTimeEnd.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time_end'] = value
        return self._cache['insert_time_end']

    @insert_time_end.setter
    def insert_time_end(self, value: str):
        """设置结束时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTimeEnd = encoded
        self._cache['insert_time_end'] = value

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



class QryInvestUnit(CapsuleStruct):
    """查询投资单元"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
        ]

    _capsule_name = "QryInvestUnit"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "invest_unit_id": "InvestUnitID",
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



class QryCombInstrumentGuard(CapsuleStruct):
    """查询组合合约保证金"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryCombInstrumentGuard"

    _field_mappings = {
        "broker_id": "BrokerID",
        "reserve1": "reserve1",
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



class QryCombAction(CapsuleStruct):
    """查询组合申报"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryCombAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryTransferSerial(CapsuleStruct):
    """查询转账流水"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("BankID", ctypes.c_char * 4),               # 银行编码
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
        ]

    _capsule_name = "QryTransferSerial"

    _field_mappings = {
        "broker_id": "BrokerID",
        "account_id": "AccountID",
        "bank_id": "BankID",
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
    def account_id(self) -> str:
        """投资者帐号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置投资者帐号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AccountID = encoded
        self._cache['account_id'] = value

    @property
    def bank_id(self) -> str:
        """银行编码"""
        if 'bank_id' not in self._cache:
            value = self._struct.BankID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_id'] = value
        return self._cache['bank_id']

    @bank_id.setter
    def bank_id(self, value: str):
        """设置银行编码"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.BankID = encoded
        self._cache['bank_id'] = value

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



class QryAccountRegister(CapsuleStruct):
    """查询银期签约"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("BankID", ctypes.c_char * 4),               # 银行编码
            ("BankBranchID", ctypes.c_char * 5),         # 银行分支机构编码
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
        ]

    _capsule_name = "QryAccountRegister"

    _field_mappings = {
        "broker_id": "BrokerID",
        "account_id": "AccountID",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
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
    def account_id(self) -> str:
        """投资者帐号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置投资者帐号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AccountID = encoded
        self._cache['account_id'] = value

    @property
    def bank_id(self) -> str:
        """银行编码"""
        if 'bank_id' not in self._cache:
            value = self._struct.BankID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_id'] = value
        return self._cache['bank_id']

    @bank_id.setter
    def bank_id(self, value: str):
        """设置银行编码"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.BankID = encoded
        self._cache['bank_id'] = value

    @property
    def bank_branch_id(self) -> str:
        """银行分支机构编码"""
        if 'bank_branch_id' not in self._cache:
            value = self._struct.BankBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_branch_id'] = value
        return self._cache['bank_branch_id']

    @bank_branch_id.setter
    def bank_branch_id(self, value: str):
        """设置银行分支机构编码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBranchID = encoded
        self._cache['bank_branch_id'] = value

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



class QryContractBank(CapsuleStruct):
    """查询签约银行"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("BankID", ctypes.c_char * 4),               # 银行代码
            ("BankBrchID", ctypes.c_char * 5),           # 银行分中心代码
        ]

    _capsule_name = "QryContractBank"

    _field_mappings = {
        "broker_id": "BrokerID",
        "bank_id": "BankID",
        "bank_brch_id": "BankBrchID",
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
    def bank_id(self) -> str:
        """银行代码"""
        if 'bank_id' not in self._cache:
            value = self._struct.BankID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_id'] = value
        return self._cache['bank_id']

    @bank_id.setter
    def bank_id(self, value: str):
        """设置银行代码"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.BankID = encoded
        self._cache['bank_id'] = value

    @property
    def bank_brch_id(self) -> str:
        """银行分中心代码"""
        if 'bank_brch_id' not in self._cache:
            value = self._struct.BankBrchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_brch_id'] = value
        return self._cache['bank_brch_id']

    @bank_brch_id.setter
    def bank_brch_id(self, value: str):
        """设置银行分中心代码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBrchID = encoded
        self._cache['bank_brch_id'] = value



class QryParkedOrder(CapsuleStruct):
    """查询预埋单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryParkedOrder"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryParkedOrderAction(CapsuleStruct):
    """查询预埋撤单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryParkedOrderAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
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



class QryTradingNotice(CapsuleStruct):
    """查询交易提示"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
        ]

    _capsule_name = "QryTradingNotice"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "invest_unit_id": "InvestUnitID",
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



class QryBrokerTradingParams(CapsuleStruct):
    """查询经纪公司交易参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
        ]

    _capsule_name = "QryBrokerTradingParams"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "currency_id": "CurrencyID",
        "account_id": "AccountID",
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
    def account_id(self) -> str:
        """投资者帐号"""
        if 'account_id' not in self._cache:
            value = self._struct.AccountID.rstrip(b'\x00').decode('ascii')
            self._cache['account_id'] = value
        return self._cache['account_id']

    @account_id.setter
    def account_id(self, value: str):
        """设置投资者帐号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AccountID = encoded
        self._cache['account_id'] = value



class QryBrokerTradingAlgos(CapsuleStruct):
    """查询经纪公司交易算法"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "QryBrokerTradingAlgos"

    _field_mappings = {
        "broker_id": "BrokerID",
        "exchange_id": "ExchangeID",
        "reserve1": "reserve1",
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




class QryClassifiedInstrument(CapsuleStruct):
    """分级分类结算参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ExchangeInstID", ctypes.c_char * 81),     # 合约在交易所的代码
            ("ProductID", ctypes.c_char * 81),          # 产品代码
            ("TradingType", ctypes.c_char),             # 合约交易状态
            ("ClassType", ctypes.c_char),               # 合约分类类型
        ]

    _capsule_name = "QryClassifiedInstrument"

    _field_mappings = {
        "instrument_id": "InstrumentID",
        "exchange_id": "ExchangeID",
        "exchange_inst_id": "ExchangeInstID",
        "product_id": "ProductID",
        "trading_type": "TradingType",
        "class_type": "ClassType",
    }

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
    def exchange_inst_id(self) -> str:
        """合约在交易所的代码"""
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

    @property
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value

    @property
    def trading_type(self) -> str:
        """合约交易状态"""
        if 'trading_type' not in self._cache:
            value = self._struct.TradingType.decode('ascii')
            self._cache['trading_type'] = value
        return self._cache['trading_type']

    @trading_type.setter
    def trading_type(self, value: str):
        """设置合约交易状态"""
        self._struct.TradingType = value.encode('ascii')[0]
        self._cache['trading_type'] = value

    @property
    def class_type(self) -> str:
        """合约分类类型"""
        if 'class_type' not in self._cache:
            value = self._struct.ClassType.decode('ascii')
            self._cache['class_type'] = value
        return self._cache['class_type']

    @class_type.setter
    def class_type(self, value: str):
        """设置合约分类类型"""
        self._struct.ClassType = value.encode('ascii')[0]
        self._cache['class_type'] = value



class QryCombPromotionParam(CapsuleStruct):
    """查询组合优惠比例"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
        ]

    _capsule_name = "QryCombPromotionParam"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
    }

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



class QryOffsetSetting(CapsuleStruct):
    """对冲设置查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("ProductID", ctypes.c_char * 41),          # 产品代码
            ("OffsetType", ctypes.c_char),              # 对冲类型
        ]

    _capsule_name = "QryOffsetSetting"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "product_id": "ProductID",
        "offset_type": "OffsetType",
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
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value

    @property
    def offset_type(self) -> str:
        """对冲类型"""
        if 'offset_type' not in self._cache:
            value = self._struct.OffsetType.decode('ascii')
            self._cache['offset_type'] = value
        return self._cache['offset_type']

    @offset_type.setter
    def offset_type(self, value: str):
        """设置对冲类型"""
        self._struct.OffsetType = value.encode('ascii')[0]
        self._cache['offset_type'] = value



class QrySPBMFutureParameter(CapsuleStruct):
    """SPBM期货合约保证金参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("ProdFamilyCode", ctypes.c_char * 81),     # 品种代码
        ]

    _capsule_name = "QrySPBMFutureParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "prod_family_code": "ProdFamilyCode",
    }

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

    @property
    def prod_family_code(self) -> str:
        """品种代码"""
        if 'prod_family_code' not in self._cache:
            value = self._struct.ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['prod_family_code'] = value
        return self._cache['prod_family_code']

    @prod_family_code.setter
    def prod_family_code(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProdFamilyCode = encoded
        self._cache['prod_family_code'] = value



class QrySPBMOptionParameter(CapsuleStruct):
    """SPBM期权合约保证金参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("ProdFamilyCode", ctypes.c_char * 81),     # 品种代码
        ]

    _capsule_name = "QrySPBMOptionParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "prod_family_code": "ProdFamilyCode",
    }

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

    @property
    def prod_family_code(self) -> str:
        """品种代码"""
        if 'prod_family_code' not in self._cache:
            value = self._struct.ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['prod_family_code'] = value
        return self._cache['prod_family_code']

    @prod_family_code.setter
    def prod_family_code(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProdFamilyCode = encoded
        self._cache['prod_family_code'] = value



class QrySPBMIntraParameter(CapsuleStruct):
    """SPBM品种内对锁仓折扣参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ProdFamilyCode", ctypes.c_char * 81),     # 品种代码
        ]

    _capsule_name = "QrySPBMIntraParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "prod_family_code": "ProdFamilyCode",
    }

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
    def prod_family_code(self) -> str:
        """品种代码"""
        if 'prod_family_code' not in self._cache:
            value = self._struct.ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['prod_family_code'] = value
        return self._cache['prod_family_code']

    @prod_family_code.setter
    def prod_family_code(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProdFamilyCode = encoded
        self._cache['prod_family_code'] = value



class QrySPBMInterParameter(CapsuleStruct):
    """SPBM跨品种抵扣参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("Leg1ProdFamilyCode", ctypes.c_char * 81), # 第一腿构成品种
            ("Leg2ProdFamilyCode", ctypes.c_char * 81), # 第二腿构成品种
        ]

    _capsule_name = "QrySPBMInterParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "leg1_prod_family_code": "Leg1ProdFamilyCode",
        "leg2_prod_family_code": "Leg2ProdFamilyCode",
    }

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
    def leg1_prod_family_code(self) -> str:
        """第一腿构成品种"""
        if 'leg1_prod_family_code' not in self._cache:
            value = self._struct.Leg1ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['leg1_prod_family_code'] = value
        return self._cache['leg1_prod_family_code']

    @leg1_prod_family_code.setter
    def leg1_prod_family_code(self, value: str):
        """设置第一腿构成品种"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg1ProdFamilyCode = encoded
        self._cache['leg1_prod_family_code'] = value

    @property
    def leg2_prod_family_code(self) -> str:
        """第二腿构成品种"""
        if 'leg2_prod_family_code' not in self._cache:
            value = self._struct.Leg2ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['leg2_prod_family_code'] = value
        return self._cache['leg2_prod_family_code']

    @leg2_prod_family_code.setter
    def leg2_prod_family_code(self, value: str):
        """设置第二腿构成品种"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg2ProdFamilyCode = encoded
        self._cache['leg2_prod_family_code'] = value



class QrySPBMPortfDefinition(CapsuleStruct):
    """组合保证金套餐定义查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("PortfolioDefID", ctypes.c_char * 41),     # 组合保证金套餐代码
            ("ProdFamilyCode", ctypes.c_char * 81),     # 品种代码
        ]

    _capsule_name = "QrySPBMPortfDefinition"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "portfolio_def_id": "PortfolioDefID",
        "prod_family_code": "ProdFamilyCode",
    }

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
    def portfolio_def_id(self) -> str:
        """组合保证金套餐代码"""
        if 'portfolio_def_id' not in self._cache:
            value = self._struct.PortfolioDefID.rstrip(b'\x00').decode('ascii')
            self._cache['portfolio_def_id'] = value
        return self._cache['portfolio_def_id']

    @portfolio_def_id.setter
    def portfolio_def_id(self, value: str):
        """设置组合保证金套餐代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.PortfolioDefID = encoded
        self._cache['portfolio_def_id'] = value

    @property
    def prod_family_code(self) -> str:
        """品种代码"""
        if 'prod_family_code' not in self._cache:
            value = self._struct.ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['prod_family_code'] = value
        return self._cache['prod_family_code']

    @prod_family_code.setter
    def prod_family_code(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProdFamilyCode = encoded
        self._cache['prod_family_code'] = value



class QrySPBMInvestorPortfDef(CapsuleStruct):
    """投资者套餐选择查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
        ]

    _capsule_name = "QrySPBMInvestorPortfDef"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
    }

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



class QryInvestorPortfMarginRatio(CapsuleStruct):
    """投资者新型组合保证金系数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ProductGroupID", ctypes.c_char * 41),     # 产品群代码
        ]

    _capsule_name = "QryInvestorPortfMarginRatio"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "exchange_id": "ExchangeID",
        "product_group_id": "ProductGroupID",
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
    def product_group_id(self) -> str:
        """产品群代码"""
        if 'product_group_id' not in self._cache:
            value = self._struct.ProductGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['product_group_id'] = value
        return self._cache['product_group_id']

    @product_group_id.setter
    def product_group_id(self, value: str):
        """设置产品群代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductGroupID = encoded
        self._cache['product_group_id'] = value



class QryInvestorProdSPBMDetail(CapsuleStruct):
    """投资者产品SPBM明细查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("ProdFamilyCode", ctypes.c_char * 81),     # 品种代码
        ]

    _capsule_name = "QryInvestorProdSPBMDetail"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "prod_family_code": "ProdFamilyCode",
    }

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
    def prod_family_code(self) -> str:
        """品种代码"""
        if 'prod_family_code' not in self._cache:
            value = self._struct.ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['prod_family_code'] = value
        return self._cache['prod_family_code']

    @prod_family_code.setter
    def prod_family_code(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProdFamilyCode = encoded
        self._cache['prod_family_code'] = value



class QryInvestorCommoditySPMMMargin(CapsuleStruct):
    """投资者商品组SPMM记录查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("CommodityID", ctypes.c_char * 41),        # 商品组代码
        ]

    _capsule_name = "QryInvestorCommoditySPMMMargin"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "commodity_id": "CommodityID",
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
    def commodity_id(self) -> str:
        """商品组代码"""
        if 'commodity_id' not in self._cache:
            value = self._struct.CommodityID.rstrip(b'\x00').decode('ascii')
            self._cache['commodity_id'] = value
        return self._cache['commodity_id']

    @commodity_id.setter
    def commodity_id(self, value: str):
        """设置商品组代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CommodityID = encoded
        self._cache['commodity_id'] = value



class QryInvestorCommodityGroupSPMMMargin(CapsuleStruct):
    """投资者商品群SPMM记录查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("CommodityGroupID", ctypes.c_char * 41),   # 商品群代码
        ]

    _capsule_name = "QryInvestorCommodityGroupSPMMMargin"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "commodity_group_id": "CommodityGroupID",
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
    def commodity_group_id(self) -> str:
        """商品群代码"""
        if 'commodity_group_id' not in self._cache:
            value = self._struct.CommodityGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['commodity_group_id'] = value
        return self._cache['commodity_group_id']

    @commodity_group_id.setter
    def commodity_group_id(self, value: str):
        """设置商品群代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CommodityGroupID = encoded
        self._cache['commodity_group_id'] = value



class QrySPMMInstParam(CapsuleStruct):
    """SPMM合约参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
        ]

    _capsule_name = "QrySPMMInstParam"

    _field_mappings = {
        "instrument_id": "InstrumentID",
    }

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



class QrySPMMProductParam(CapsuleStruct):
    """SPMM产品参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ProductID", ctypes.c_char * 41),          # 产品代码
        ]

    _capsule_name = "QrySPMMProductParam"

    _field_mappings = {
        "product_id": "ProductID",
    }

    @property
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value



class QrySPBMAddOnInterParameter(CapsuleStruct):
    """SPBM跨品种抵扣附加参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("Leg1ProdFamilyCode", ctypes.c_char * 81), # 第一腿构成品种
            ("Leg2ProdFamilyCode", ctypes.c_char * 81), # 第二腿构成品种
        ]

    _capsule_name = "QrySPBMAddOnInterParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "leg1_prod_family_code": "Leg1ProdFamilyCode",
        "leg2_prod_family_code": "Leg2ProdFamilyCode",
    }

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
    def leg1_prod_family_code(self) -> str:
        """第一腿构成品种"""
        if 'leg1_prod_family_code' not in self._cache:
            value = self._struct.Leg1ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['leg1_prod_family_code'] = value
        return self._cache['leg1_prod_family_code']

    @leg1_prod_family_code.setter
    def leg1_prod_family_code(self, value: str):
        """设置第一腿构成品种"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg1ProdFamilyCode = encoded
        self._cache['leg1_prod_family_code'] = value

    @property
    def leg2_prod_family_code(self) -> str:
        """第二腿构成品种"""
        if 'leg2_prod_family_code' not in self._cache:
            value = self._struct.Leg2ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['leg2_prod_family_code'] = value
        return self._cache['leg2_prod_family_code']

    @leg2_prod_family_code.setter
    def leg2_prod_family_code(self, value: str):
        """设置第二腿构成品种"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg2ProdFamilyCode = encoded
        self._cache['leg2_prod_family_code'] = value



class QryRCAMSCombProductInfo(CapsuleStruct):
    """RCAMS组合产品信息查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ProductID", ctypes.c_char * 41),          # 产品代码
            ("CombProductID", ctypes.c_char * 41),      # 商品组代码
            ("ProductGroupID", ctypes.c_char * 41),     # 商品群代码
        ]

    _capsule_name = "QryRCAMSCombProductInfo"

    _field_mappings = {
        "product_id": "ProductID",
        "comb_product_id": "CombProductID",
        "product_group_id": "ProductGroupID",
    }

    @property
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value

    @property
    def comb_product_id(self) -> str:
        """商品组代码"""
        if 'comb_product_id' not in self._cache:
            value = self._struct.CombProductID.rstrip(b'\x00').decode('ascii')
            self._cache['comb_product_id'] = value
        return self._cache['comb_product_id']

    @comb_product_id.setter
    def comb_product_id(self, value: str):
        """设置商品组代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CombProductID = encoded
        self._cache['comb_product_id'] = value

    @property
    def product_group_id(self) -> str:
        """商品群代码"""
        if 'product_group_id' not in self._cache:
            value = self._struct.ProductGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['product_group_id'] = value
        return self._cache['product_group_id']

    @product_group_id.setter
    def product_group_id(self, value: str):
        """设置商品群代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductGroupID = encoded
        self._cache['product_group_id'] = value



class QryRCAMSInstrParameter(CapsuleStruct):
    """RCAMS合约保证金参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ProductID", ctypes.c_char * 41),          # 产品代码
        ]

    _capsule_name = "QryRCAMSInstrParameter"

    _field_mappings = {
        "product_id": "ProductID",
    }

    @property
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value



class QryRCAMSIntraParameter(CapsuleStruct):
    """RCAMS品种内风险对冲参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("CombProductID", ctypes.c_char * 41),      # 产品组合代码
        ]

    _capsule_name = "QryRCAMSIntraParameter"

    _field_mappings = {
        "comb_product_id": "CombProductID",
    }

    @property
    def comb_product_id(self) -> str:
        """产品组合代码"""
        if 'comb_product_id' not in self._cache:
            value = self._struct.CombProductID.rstrip(b'\x00').decode('ascii')
            self._cache['comb_product_id'] = value
        return self._cache['comb_product_id']

    @comb_product_id.setter
    def comb_product_id(self, value: str):
        """设置产品组合代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CombProductID = encoded
        self._cache['comb_product_id'] = value



class QryRCAMSInterParameter(CapsuleStruct):
    """RCAMS跨品种风险折抵参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ProductGroupID", ctypes.c_char * 41),     # 商品群代码
            ("CombProduct1", ctypes.c_char * 41),        # 产品组合代码1
            ("CombProduct2", ctypes.c_char * 41),        # 产品组合代码2
        ]

    _capsule_name = "QryRCAMSInterParameter"

    _field_mappings = {
        "product_group_id": "ProductGroupID",
        "comb_product1": "CombProduct1",
        "comb_product2": "CombProduct2",
    }

    @property
    def product_group_id(self) -> str:
        """商品群代码"""
        if 'product_group_id' not in self._cache:
            value = self._struct.ProductGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['product_group_id'] = value
        return self._cache['product_group_id']

    @product_group_id.setter
    def product_group_id(self, value: str):
        """设置商品群代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductGroupID = encoded
        self._cache['product_group_id'] = value

    @property
    def comb_product1(self) -> str:
        """产品组合代码1"""
        if 'comb_product1' not in self._cache:
            value = self._struct.CombProduct1.rstrip(b'\x00').decode('ascii')
            self._cache['comb_product1'] = value
        return self._cache['comb_product1']

    @comb_product1.setter
    def comb_product1(self, value: str):
        """设置产品组合代码1"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CombProduct1 = encoded
        self._cache['comb_product1'] = value

    @property
    def comb_product2(self) -> str:
        """产品组合代码2"""
        if 'comb_product2' not in self._cache:
            value = self._struct.CombProduct2.rstrip(b'\x00').decode('ascii')
            self._cache['comb_product2'] = value
        return self._cache['comb_product2']

    @comb_product2.setter
    def comb_product2(self, value: str):
        """设置产品组合代码2"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CombProduct2 = encoded
        self._cache['comb_product2'] = value



class QryRCAMSShortOptAdjustParam(CapsuleStruct):
    """RCAMS空头期权风险调整参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("CombProductID", ctypes.c_char * 41),      # 产品组合代码
        ]

    _capsule_name = "QryRCAMSShortOptAdjustParam"

    _field_mappings = {
        "comb_product_id": "CombProductID",
    }

    @property
    def comb_product_id(self) -> str:
        """产品组合代码"""
        if 'comb_product_id' not in self._cache:
            value = self._struct.CombProductID.rstrip(b'\x00').decode('ascii')
            self._cache['comb_product_id'] = value
        return self._cache['comb_product_id']

    @comb_product_id.setter
    def comb_product_id(self, value: str):
        """设置产品组合代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CombProductID = encoded
        self._cache['comb_product_id'] = value



class QryRCAMSInvestorCombPosition(CapsuleStruct):
    """RCAMS策略组合持仓查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("CombInstrumentID", ctypes.c_char * 81),   # 组合合约代码
        ]

    _capsule_name = "QryRCAMSInvestorCombPosition"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "instrument_id": "InstrumentID",
        "comb_instrument_id": "CombInstrumentID",
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
    def comb_instrument_id(self) -> str:
        """组合合约代码"""
        if 'comb_instrument_id' not in self._cache:
            value = self._struct.CombInstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['comb_instrument_id'] = value
        return self._cache['comb_instrument_id']

    @comb_instrument_id.setter
    def comb_instrument_id(self, value: str):
        """设置组合合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.CombInstrumentID = encoded
        self._cache['comb_instrument_id'] = value



class QryInvestorProdRCAMSMargin(CapsuleStruct):
    """投资者品种RCAMS保证金查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("CombProductID", ctypes.c_char * 41),      # 产品组合代码
            ("ProductGroupID", ctypes.c_char * 41),     # 商品群代码
        ]

    _capsule_name = "QryInvestorProdRCAMSMargin"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "comb_product_id": "CombProductID",
        "product_group_id": "ProductGroupID",
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
    def comb_product_id(self) -> str:
        """产品组合代码"""
        if 'comb_product_id' not in self._cache:
            value = self._struct.CombProductID.rstrip(b'\x00').decode('ascii')
            self._cache['comb_product_id'] = value
        return self._cache['comb_product_id']

    @comb_product_id.setter
    def comb_product_id(self, value: str):
        """设置产品组合代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.CombProductID = encoded
        self._cache['comb_product_id'] = value

    @property
    def product_group_id(self) -> str:
        """商品群代码"""
        if 'product_group_id' not in self._cache:
            value = self._struct.ProductGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['product_group_id'] = value
        return self._cache['product_group_id']

    @product_group_id.setter
    def product_group_id(self, value: str):
        """设置商品群代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductGroupID = encoded
        self._cache['product_group_id'] = value



class QryRULEInstrParameter(CapsuleStruct):
    """RULE合约保证金参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
        ]

    _capsule_name = "QryRULEInstrParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
    }

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



class QryRULEIntraParameter(CapsuleStruct):
    """RULE品种内对锁仓折扣参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ProdFamilyCode", ctypes.c_char * 81),     # 品种代码
        ]

    _capsule_name = "QryRULEIntraParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "prod_family_code": "ProdFamilyCode",
    }

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
    def prod_family_code(self) -> str:
        """品种代码"""
        if 'prod_family_code' not in self._cache:
            value = self._struct.ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['prod_family_code'] = value
        return self._cache['prod_family_code']

    @prod_family_code.setter
    def prod_family_code(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProdFamilyCode = encoded
        self._cache['prod_family_code'] = value



class QryRULEInterParameter(CapsuleStruct):
    """RULE跨品种抵扣参数查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("Leg1ProdFamilyCode", ctypes.c_char * 81), # 第一腿构成品种
            ("Leg2ProdFamilyCode", ctypes.c_char * 81), # 第二腿构成品种
            ("CommodityGroupID", ctypes.c_int),         # 商品群号
        ]

    _capsule_name = "QryRULEInterParameter"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "leg1_prod_family_code": "Leg1ProdFamilyCode",
        "leg2_prod_family_code": "Leg2ProdFamilyCode",
        "commodity_group_id": "CommodityGroupID",
    }

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
    def leg1_prod_family_code(self) -> str:
        """第一腿构成品种"""
        if 'leg1_prod_family_code' not in self._cache:
            value = self._struct.Leg1ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['leg1_prod_family_code'] = value
        return self._cache['leg1_prod_family_code']

    @leg1_prod_family_code.setter
    def leg1_prod_family_code(self, value: str):
        """设置第一腿构成品种"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg1ProdFamilyCode = encoded
        self._cache['leg1_prod_family_code'] = value

    @property
    def leg2_prod_family_code(self) -> str:
        """第二腿构成品种"""
        if 'leg2_prod_family_code' not in self._cache:
            value = self._struct.Leg2ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['leg2_prod_family_code'] = value
        return self._cache['leg2_prod_family_code']

    @leg2_prod_family_code.setter
    def leg2_prod_family_code(self, value: str):
        """设置第二腿构成品种"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.Leg2ProdFamilyCode = encoded
        self._cache['leg2_prod_family_code'] = value

    @property
    def commodity_group_id(self) -> int:
        """商品群号"""
        return self._struct.CommodityGroupID

    @commodity_group_id.setter
    def commodity_group_id(self, value: int):
        """设置商品群号"""
        self._struct.CommodityGroupID = value



class QryInvestorProdRULEMargin(CapsuleStruct):
    """投资者产品RULE保证金查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("ProdFamilyCode", ctypes.c_char * 81),     # 品种代码
            ("CommodityGroupID", ctypes.c_int),         # 商品群号
        ]

    _capsule_name = "QryInvestorProdRULEMargin"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "prod_family_code": "ProdFamilyCode",
        "commodity_group_id": "CommodityGroupID",
    }

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
    def prod_family_code(self) -> str:
        """品种代码"""
        if 'prod_family_code' not in self._cache:
            value = self._struct.ProdFamilyCode.rstrip(b'\x00').decode('ascii')
            self._cache['prod_family_code'] = value
        return self._cache['prod_family_code']

    @prod_family_code.setter
    def prod_family_code(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProdFamilyCode = encoded
        self._cache['prod_family_code'] = value

    @property
    def commodity_group_id(self) -> int:
        """商品群号"""
        return self._struct.CommodityGroupID

    @commodity_group_id.setter
    def commodity_group_id(self, value: int):
        """设置商品群号"""
        self._struct.CommodityGroupID = value



class QryInvestorPortfSetting(CapsuleStruct):
    """投资者组合设置查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者编号
        ]

    _capsule_name = "QryInvestorPortfSetting"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
    }

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
        """投资者编号"""
        if 'investor_id' not in self._cache:
            value = self._struct.InvestorID.rstrip(b'\x00').decode('ascii')
            self._cache['investor_id'] = value
        return self._cache['investor_id']

    @investor_id.setter
    def investor_id(self, value: str):
        """设置投资者编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.InvestorID = encoded
        self._cache['investor_id'] = value



class QryInvestorInfoCommRec(CapsuleStruct):
    """投资者信息通讯记录查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("InstrumentID", ctypes.c_char * 81),       # 商品代码
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
        ]

    _capsule_name = "QryInvestorInfoCommRec"

    _field_mappings = {
        "investor_id": "InvestorID",
        "instrument_id": "InstrumentID",
        "broker_id": "BrokerID",
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
    def instrument_id(self) -> str:
        """商品代码"""
        if 'instrument_id' not in self._cache:
            value = self._struct.InstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['instrument_id'] = value
        return self._cache['instrument_id']

    @instrument_id.setter
    def instrument_id(self, value: str):
        """设置商品代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.InstrumentID = encoded
        self._cache['instrument_id'] = value

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



class QryCombLeg(CapsuleStruct):
    """组合腿查询"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("LegInstrumentID", ctypes.c_char * 81),    # 单腿合约代码
        ]

    _capsule_name = "QryCombLeg"

    _field_mappings = {
        "leg_instrument_id": "LegInstrumentID",
    }

    @property
    def leg_instrument_id(self) -> str:
        """单腿合约代码"""
        if 'leg_instrument_id' not in self._cache:
            value = self._struct.LegInstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['leg_instrument_id'] = value
        return self._cache['leg_instrument_id']

    @leg_instrument_id.setter
    def leg_instrument_id(self, value: str):
        """设置单腿合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.LegInstrumentID = encoded
        self._cache['leg_instrument_id'] = value






class QryTraderOffer(CapsuleStruct):
    """查询交易员报盘"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),      # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),  # 会员代码
            ("TraderID", ctypes.c_char * 21),       # 交易所交易员代码
        ]

    _capsule_name = "QryTraderOffer"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "trader_id": "TraderID",
    }

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
    def participant_id(self) -> str:
        """会员代码"""
        if 'participant_id' not in self._cache:
            value = self._struct.ParticipantID.rstrip(b'\x00').decode('ascii')
            self._cache['participant_id'] = value
        return self._cache['participant_id']

    @participant_id.setter
    def participant_id(self, value: str):
        """设置会员代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ParticipantID = encoded
        self._cache['participant_id'] = value

    @property
    def trader_id(self) -> str:
        """交易所交易员代码"""
        if 'trader_id' not in self._cache:
            value = self._struct.TraderID.rstrip(b'\x00').decode('ascii')
            self._cache['trader_id'] = value
        return self._cache['trader_id']

    @trader_id.setter
    def trader_id(self, value: str):
        """设置交易所交易员代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.TraderID = encoded
        self._cache['trader_id'] = value



class QryRiskSettleInvestPosition(CapsuleStruct):
    """查询投资者风险结算持仓"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),      # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),    # 投资者代码
            ("InstrumentID", ctypes.c_char * 81),  # 合约代码
        ]

    _capsule_name = "QryRiskSettleInvstPosition"  # 注意: C++ 使用 Invst 而非 Invest

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
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



class QryRiskSettleProductStatus(CapsuleStruct):
    """查询风险结算产品状态"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ProductID", ctypes.c_char * 81),  # 产品代码
        ]

    _capsule_name = "QryRiskSettleProductStatus"

    _field_mappings = {
        "product_id": "ProductID",
    }

    @property
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value


