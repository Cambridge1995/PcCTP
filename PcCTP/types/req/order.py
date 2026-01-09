"""
Req
"""

import ctypes
from PcCTP.types.base import CapsuleStruct
from typing import Dict


class InputOrder(CapsuleStruct):
    """报单录入"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("OrderRef", ctypes.c_char * 13),            # 报单引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("OrderPriceType", ctypes.c_char),           # 报单价格条件
            ("Direction", ctypes.c_char),                # 买卖方向
            ("CombOffsetFlag", ctypes.c_char * 5),       # 组合开平标志
            ("CombHedgeFlag", ctypes.c_char * 5),        # 组合投机套保标志
            ("LimitPrice", ctypes.c_double),             # 价格
            ("VolumeTotalOriginal", ctypes.c_int),       # 数量
            ("TimeCondition", ctypes.c_char),            # 有效期类型
            ("GTDDate", ctypes.c_char * 9),              # GTD日期
            ("VolumeCondition", ctypes.c_char),          # 成交量类型
            ("MinVolume", ctypes.c_int),                 # 最小成交量
            ("ContingentCondition", ctypes.c_char),      # 触发条件
            ("StopPrice", ctypes.c_double),              # 止损价
            ("ForceCloseReason", ctypes.c_char),         # 强平原因
            ("IsAutoSuspend", ctypes.c_int),             # 自动挂起标志
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("UserForceClose", ctypes.c_int),            # 用户强平标志
            ("IsSwapOrder", ctypes.c_int),               # 互换单标志
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("AccountID", ctypes.c_char * 13),           # 资金账号
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
            ("ClientID", ctypes.c_char * 11),            # 交易编码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
            ("OrderMemo", ctypes.c_char * 13),           # 报单回显字段
            ("SessionReqSeq", ctypes.c_int),             # session上请求计数 api自动维护
        ]

    _capsule_name = "InputOrder"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "order_ref": "OrderRef",
        "user_id": "UserID",
        "order_price_type": "OrderPriceType",
        "direction": "Direction",
        "comb_offset_flag": "CombOffsetFlag",
        "comb_hedge_flag": "CombHedgeFlag",
        "limit_price": "LimitPrice",
        "volume_total_original": "VolumeTotalOriginal",
        "time_condition": "TimeCondition",
        "gtd_date": "GTDDate",
        "volume_condition": "VolumeCondition",
        "min_volume": "MinVolume",
        "contingent_condition": "ContingentCondition",
        "stop_price": "StopPrice",
        "force_close_reason": "ForceCloseReason",
        "is_auto_suspend": "IsAutoSuspend",
        "business_unit": "BusinessUnit",
        "request_id": "RequestID",
        "user_force_close": "UserForceClose",
        "is_swap_order": "IsSwapOrder",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
        "order_memo": "OrderMemo",
        "session_req_seq": "SessionReqSeq",
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
    def order_ref(self) -> str:
        """报单引用"""
        if 'order_ref' not in self._cache:
            value = self._struct.OrderRef.rstrip(b'\x00').decode('ascii')
            self._cache['order_ref'] = value
        return self._cache['order_ref']

    @order_ref.setter
    def order_ref(self, value: str):
        """设置报单引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OrderRef = encoded
        self._cache['order_ref'] = value

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
    def order_price_type(self) -> str:
        """报单价格条件"""
        if 'order_price_type' not in self._cache:
            value = self._struct.OrderPriceType.decode('ascii')
            self._cache['order_price_type'] = value
        return self._cache['order_price_type']

    @order_price_type.setter
    def order_price_type(self, value: str):
        """设置报单价格条件"""
        self._struct.OrderPriceType = value.encode('ascii')[0]
        self._cache['order_price_type'] = value

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
    def comb_offset_flag(self) -> str:
        """组合开平标志"""
        if 'comb_offset_flag' not in self._cache:
            value = self._struct.CombOffsetFlag.rstrip(b'\x00').decode('ascii')
            self._cache['comb_offset_flag'] = value
        return self._cache['comb_offset_flag']

    @comb_offset_flag.setter
    def comb_offset_flag(self, value: str):
        """设置组合开平标志"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.CombOffsetFlag = encoded
        self._cache['comb_offset_flag'] = value

    @property
    def comb_hedge_flag(self) -> str:
        """组合投机套保标志"""
        if 'comb_hedge_flag' not in self._cache:
            value = self._struct.CombHedgeFlag.rstrip(b'\x00').decode('ascii')
            self._cache['comb_hedge_flag'] = value
        return self._cache['comb_hedge_flag']

    @comb_hedge_flag.setter
    def comb_hedge_flag(self, value: str):
        """设置组合投机套保标志"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.CombHedgeFlag = encoded
        self._cache['comb_hedge_flag'] = value

    @property
    def limit_price(self) -> float:
        """价格"""
        return self._struct.LimitPrice

    @limit_price.setter
    def limit_price(self, value: float):
        """设置价格"""
        self._struct.LimitPrice = value

    @property
    def volume_total_original(self) -> int:
        """数量"""
        return self._struct.VolumeTotalOriginal

    @volume_total_original.setter
    def volume_total_original(self, value: int):
        """设置数量"""
        self._struct.VolumeTotalOriginal = value

    @property
    def time_condition(self) -> str:
        """有效期类型"""
        if 'time_condition' not in self._cache:
            value = self._struct.TimeCondition.decode('ascii')
            self._cache['time_condition'] = value
        return self._cache['time_condition']

    @time_condition.setter
    def time_condition(self, value: str):
        """设置有效期类型"""
        self._struct.TimeCondition = value.encode('ascii')[0]
        self._cache['time_condition'] = value

    @property
    def gtd_date(self) -> str:
        """GTD日期"""
        if 'gtd_date' not in self._cache:
            value = self._struct.GTDDate.rstrip(b'\x00').decode('ascii')
            self._cache['gtd_date'] = value
        return self._cache['gtd_date']

    @gtd_date.setter
    def gtd_date(self, value: str):
        """设置GTD日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.GTDDate = encoded
        self._cache['gtd_date'] = value

    @property
    def volume_condition(self) -> str:
        """成交量类型"""
        if 'volume_condition' not in self._cache:
            value = self._struct.VolumeCondition.decode('ascii')
            self._cache['volume_condition'] = value
        return self._cache['volume_condition']

    @volume_condition.setter
    def volume_condition(self, value: str):
        """设置成交量类型"""
        self._struct.VolumeCondition = value.encode('ascii')[0]
        self._cache['volume_condition'] = value

    @property
    def min_volume(self) -> int:
        """最小成交量"""
        return self._struct.MinVolume

    @min_volume.setter
    def min_volume(self, value: int):
        """设置最小成交量"""
        self._struct.MinVolume = value

    @property
    def contingent_condition(self) -> str:
        """触发条件"""
        if 'contingent_condition' not in self._cache:
            value = self._struct.ContingentCondition.decode('ascii')
            self._cache['contingent_condition'] = value
        return self._cache['contingent_condition']

    @contingent_condition.setter
    def contingent_condition(self, value: str):
        """设置触发条件"""
        self._struct.ContingentCondition = value.encode('ascii')[0]
        self._cache['contingent_condition'] = value

    @property
    def stop_price(self) -> float:
        """止损价"""
        return self._struct.StopPrice

    @stop_price.setter
    def stop_price(self, value: float):
        """设置止损价"""
        self._struct.StopPrice = value

    @property
    def force_close_reason(self) -> str:
        """强平原因"""
        if 'force_close_reason' not in self._cache:
            value = self._struct.ForceCloseReason.decode('ascii')
            self._cache['force_close_reason'] = value
        return self._cache['force_close_reason']

    @force_close_reason.setter
    def force_close_reason(self, value: str):
        """设置强平原因"""
        self._struct.ForceCloseReason = value.encode('ascii')[0]
        self._cache['force_close_reason'] = value

    @property
    def is_auto_suspend(self) -> int:
        """自动挂起标志"""
        return self._struct.IsAutoSuspend

    @is_auto_suspend.setter
    def is_auto_suspend(self, value: int):
        """设置自动挂起标志"""
        self._struct.IsAutoSuspend = value

    @property
    def business_unit(self) -> str:
        """业务单元"""
        if 'business_unit' not in self._cache:
            value = self._struct.BusinessUnit.rstrip(b'\x00').decode('ascii')
            self._cache['business_unit'] = value
        return self._cache['business_unit']

    @business_unit.setter
    def business_unit(self, value: str):
        """设置业务单元"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.BusinessUnit = encoded
        self._cache['business_unit'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def user_force_close(self) -> int:
        """用户强平标志"""
        return self._struct.UserForceClose

    @user_force_close.setter
    def user_force_close(self, value: int):
        """设置用户强平标志"""
        self._struct.UserForceClose = value

    @property
    def is_swap_order(self) -> int:
        """互换单标志"""
        return self._struct.IsSwapOrder

    @is_swap_order.setter
    def is_swap_order(self, value: int):
        """设置互换单标志"""
        self._struct.IsSwapOrder = value

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

    @property
    def client_id(self) -> str:
        """交易编码"""
        if 'client_id' not in self._cache:
            value = self._struct.ClientID.rstrip(b'\x00').decode('ascii')
            self._cache['client_id'] = value
        return self._cache['client_id']

    @client_id.setter
    def client_id(self, value: str):
        """设置交易编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClientID = encoded
        self._cache['client_id'] = value

    @property
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value

    @property
    def order_memo(self) -> str:
        """报单回显字段"""
        if 'order_memo' not in self._cache:
            value = self._struct.OrderMemo.rstrip(b'\x00').decode('ascii')
            self._cache['order_memo'] = value
        return self._cache['order_memo']

    @order_memo.setter
    def order_memo(self, value: str):
        """设置报单回显字段"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OrderMemo = encoded
        self._cache['order_memo'] = value

    @property
    def session_req_seq(self) -> int:
        """session上请求计数 api自动维护"""
        return self._struct.SessionReqSeq

    @session_req_seq.setter
    def session_req_seq(self, value: int):
        """设置session上请求计数"""
        self._struct.SessionReqSeq = value


# =============================================================================
# InputExecOrderAction - 执行宣告操作请求
# =============================================================================


class InputOrderAction(CapsuleStruct):
    """输入报单操作"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("OrderActionRef", ctypes.c_int),            # 报单操作引用
            ("OrderRef", ctypes.c_char * 13),            # 报单引用
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("OrderSysID", ctypes.c_char * 21),          # 报单编号
            ("ActionFlag", ctypes.c_char),               # 操作标志
            ("LimitPrice", ctypes.c_double),             # 价格
            ("VolumeChange", ctypes.c_int),              # 数量变化
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
            ("OrderMemo", ctypes.c_char * 13),           # 报单回显字段
            ("SessionReqSeq", ctypes.c_int),             # session上请求计数 api自动维护
        ]

    _capsule_name = "InputOrderAction"

    _field_mappings: Dict[str, str] = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "order_action_ref": "OrderActionRef",
        "order_ref": "OrderRef",
        "request_id": "RequestID",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "exchange_id": "ExchangeID",
        "order_sys_id": "OrderSysID",
        "action_flag": "ActionFlag",
        "limit_price": "LimitPrice",
        "volume_change": "VolumeChange",
        "user_id": "UserID",
        "reserve1": "reserve1",
        "invest_unit_id": "InvestUnitID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
        "order_memo": "OrderMemo",
        "session_req_seq": "SessionReqSeq",
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
    def order_action_ref(self) -> int:
        """报单操作引用"""
        return self._struct.OrderActionRef

    @order_action_ref.setter
    def order_action_ref(self, value: int):
        """设置报单操作引用"""
        self._struct.OrderActionRef = value

    @property
    def action_flag(self) -> str:
        """操作标志"""
        if 'action_flag' not in self._cache:
            value = self._struct.ActionFlag.decode('ascii')
            self._cache['action_flag'] = value
        return self._cache['action_flag']

    @action_flag.setter
    def action_flag(self, value: str):
        """设置操作标志"""
        self._struct.ActionFlag = value.encode('ascii')[0]
        self._cache['action_flag'] = value

    @property
    def limit_price(self) -> float:
        """价格"""
        return self._struct.LimitPrice

    @limit_price.setter
    def limit_price(self, value: float):
        """设置价格"""
        self._struct.LimitPrice = value

    @property
    def volume_change(self) -> int:
        """数量变化"""
        return self._struct.VolumeChange

    @volume_change.setter
    def volume_change(self, value: int):
        """设置数量变化"""
        self._struct.VolumeChange = value



class InputExecOrder(CapsuleStruct):
    """输入执行宣告"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("reserve1", ctypes.c_char * 31),           # 保留的无效字段
            ("ExecOrderRef", ctypes.c_char * 13),       # 执行宣告引用
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("Volume", ctypes.c_int),                   # 数量
            ("RequestID", ctypes.c_int),                # 请求编号
            ("BusinessUnit", ctypes.c_char * 21),       # 业务单元
            ("OffsetFlag", ctypes.c_char),              # 开平标志
            ("HedgeFlag", ctypes.c_char),               # 投机套保标志
            ("ActionType", ctypes.c_char),              # 执行类型
            ("PosiDirection", ctypes.c_char),           # 保留头寸申请的持仓方向
            ("ReservePositionFlag", ctypes.c_char),     # 期权行权后是否保留期货头寸的标记
            ("CloseFlag", ctypes.c_char),               # 期权行权后生成的头寸是否自动平仓
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("AccountID", ctypes.c_char * 13),          # 资金账号
            ("CurrencyID", ctypes.c_char * 4),          # 币种代码
            ("ClientID", ctypes.c_char * 11),           # 交易编码
            ("reserve2", ctypes.c_char * 16),           # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "InputExecOrder"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "exec_order_ref": "ExecOrderRef",
        "user_id": "UserID",
        "volume": "Volume",
        "request_id": "RequestID",
        "business_unit": "BusinessUnit",
        "offset_flag": "OffsetFlag",
        "hedge_flag": "HedgeFlag",
        "action_type": "ActionType",
        "posi_direction": "PosiDirection",
        "reserve_position_flag": "ReservePositionFlag",
        "close_flag": "CloseFlag",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
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
    def exec_order_ref(self) -> str:
        """执行宣告引用"""
        if 'exec_order_ref' not in self._cache:
            value = self._struct.ExecOrderRef.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_ref'] = value
        return self._cache['exec_order_ref']

    @exec_order_ref.setter
    def exec_order_ref(self, value: str):
        """设置执行宣告引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ExecOrderRef = encoded
        self._cache['exec_order_ref'] = value

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
    def volume(self) -> int:
        """数量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置数量"""
        self._struct.Volume = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def business_unit(self) -> str:
        """业务单元"""
        if 'business_unit' not in self._cache:
            value = self._struct.BusinessUnit.rstrip(b'\x00').decode('ascii')
            self._cache['business_unit'] = value
        return self._cache['business_unit']

    @business_unit.setter
    def business_unit(self, value: str):
        """设置业务单元"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.BusinessUnit = encoded
        self._cache['business_unit'] = value

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
    def action_type(self) -> str:
        """执行类型"""
        if 'action_type' not in self._cache:
            value = self._struct.ActionType.decode('ascii')
            self._cache['action_type'] = value
        return self._cache['action_type']

    @action_type.setter
    def action_type(self, value: str):
        """设置执行类型"""
        self._struct.ActionType = value.encode('ascii')[0]
        self._cache['action_type'] = value

    @property
    def posi_direction(self) -> str:
        """保留头寸申请的持仓方向"""
        if 'posi_direction' not in self._cache:
            value = self._struct.PosiDirection.decode('ascii')
            self._cache['posi_direction'] = value
        return self._cache['posi_direction']

    @posi_direction.setter
    def posi_direction(self, value: str):
        """设置保留头寸申请的持仓方向"""
        self._struct.PosiDirection = value.encode('ascii')[0]
        self._cache['posi_direction'] = value

    @property
    def reserve_position_flag(self) -> str:
        """期权行权后是否保留期货头寸的标记"""
        if 'reserve_position_flag' not in self._cache:
            value = self._struct.ReservePositionFlag.decode('ascii')
            self._cache['reserve_position_flag'] = value
        return self._cache['reserve_position_flag']

    @reserve_position_flag.setter
    def reserve_position_flag(self, value: str):
        """设置期权行权后是否保留期货头寸的标记"""
        self._struct.ReservePositionFlag = value.encode('ascii')[0]
        self._cache['reserve_position_flag'] = value

    @property
    def close_flag(self) -> str:
        """期权行权后生成的头寸是否自动平仓"""
        if 'close_flag' not in self._cache:
            value = self._struct.CloseFlag.decode('ascii')
            self._cache['close_flag'] = value
        return self._cache['close_flag']

    @close_flag.setter
    def close_flag(self, value: str):
        """设置期权行权后生成的头寸是否自动平仓"""
        self._struct.CloseFlag = value.encode('ascii')[0]
        self._cache['close_flag'] = value

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

    @property
    def client_id(self) -> str:
        """交易编码"""
        if 'client_id' not in self._cache:
            value = self._struct.ClientID.rstrip(b'\x00').decode('ascii')
            self._cache['client_id'] = value
        return self._cache['client_id']

    @client_id.setter
    def client_id(self, value: str):
        """设置交易编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClientID = encoded
        self._cache['client_id'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value



class InputExecOrderAction(CapsuleStruct):
    """执行宣告操作请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("ExecOrderActionRef", ctypes.c_char * 13),   # 执行宣告操作引用
            ("ExecOrderRef", ctypes.c_char * 13),         # 执行宣告引用
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ExecOrderSysID", ctypes.c_char * 21),      # 执行宣告操作编号
            ("ActionFlag", ctypes.c_char),               # 操作标志
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "InputExecOrderAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "exec_order_action_ref": "ExecOrderActionRef",
        "exec_order_ref": "ExecOrderRef",
        "request_id": "RequestID",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "exchange_id": "ExchangeID",
        "exec_order_sys_id": "ExecOrderSysID",
        "action_flag": "ActionFlag",
        "user_id": "UserID",
        "reserve1": "reserve1",
        "invest_unit_id": "InvestUnitID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
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
    def exec_order_action_ref(self) -> str:
        """执行宣告操作引用"""
        if 'exec_order_action_ref' not in self._cache:
            value = self._struct.ExecOrderActionRef.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_action_ref'] = value
        return self._cache['exec_order_action_ref']

    @exec_order_action_ref.setter
    def exec_order_action_ref(self, value: str):
        """设置执行宣告操作引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ExecOrderActionRef = encoded
        self._cache['exec_order_action_ref'] = value

    @property
    def exec_order_ref(self) -> str:
        """执行宣告引用"""
        if 'exec_order_ref' not in self._cache:
            value = self._struct.ExecOrderRef.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_ref'] = value
        return self._cache['exec_order_ref']

    @exec_order_ref.setter
    def exec_order_ref(self, value: str):
        """设置执行宣告引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ExecOrderRef = encoded
        self._cache['exec_order_ref'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

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
        """执行宣告操作编号"""
        if 'exec_order_sys_id' not in self._cache:
            value = self._struct.ExecOrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_sys_id'] = value
        return self._cache['exec_order_sys_id']

    @exec_order_sys_id.setter
    def exec_order_sys_id(self, value: str):
        """设置执行宣告操作编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ExecOrderSysID = encoded
        self._cache['exec_order_sys_id'] = value

    @property
    def action_flag(self) -> str:
        """操作标志"""
        if 'action_flag' not in self._cache:
            value = self._struct.ActionFlag.decode('ascii')
            self._cache['action_flag'] = value
        return self._cache['action_flag']

    @action_flag.setter
    def action_flag(self, value: str):
        """设置操作标志"""
        self._struct.ActionFlag = value.encode('ascii')[0]
        self._cache['action_flag'] = value

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
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value


# =============================================================================
# InputForQuote - 询价请求
# =============================================================================


class InputForQuote(CapsuleStruct):
    """询价请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ForQuoteRef", ctypes.c_char * 13),         # 询价引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "InputForQuote"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "for_quote_ref": "ForQuoteRef",
        "user_id": "UserID",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
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
    def for_quote_ref(self) -> str:
        """询价引用"""
        if 'for_quote_ref' not in self._cache:
            value = self._struct.ForQuoteRef.rstrip(b'\x00').decode('ascii')
            self._cache['for_quote_ref'] = value
        return self._cache['for_quote_ref']

    @for_quote_ref.setter
    def for_quote_ref(self, value: str):
        """设置询价引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ForQuoteRef = encoded
        self._cache['for_quote_ref'] = value

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
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value


# =============================================================================
# InputQuote - 报价录入
# =============================================================================


class InputQuote(CapsuleStruct):
    """报价录入"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("QuoteRef", ctypes.c_char * 13),            # 报价引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("AskPrice", ctypes.c_double),               # 卖价
            ("BidPrice", ctypes.c_double),               # 买价
            ("AskVolume", ctypes.c_int),                 # 卖量
            ("BidVolume", ctypes.c_int),                 # 买量
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("AskOffsetFlag", ctypes.c_char),            # 卖开平标志
            ("BidOffsetFlag", ctypes.c_char),            # 买开平标志
            ("AskHedgeFlag", ctypes.c_char),             # 卖投机套保标志
            ("BidHedgeFlag", ctypes.c_char),             # 买投机套保标志
            ("AskOrderRef", ctypes.c_char * 13),         # 卖报单引用
            ("BidOrderRef", ctypes.c_char * 13),         # 买报单引用
            ("ForQuoteSysID", ctypes.c_char * 21),       # 询价编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("AccountID", ctypes.c_char * 13),           # 资金账号
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
            ("ClientID", ctypes.c_char * 11),            # 交易编码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
            ("ReplaceSysID", ctypes.c_char * 21),        # 被替换的报价编号
            ("TimeCondition", ctypes.c_char),            # 时间条件
            ("OrderMemo", ctypes.c_char * 61),           # 报单备注
            ("SessionReqSeq", ctypes.c_short),           # 会话请求序号
        ]

    _capsule_name = "InputQuote"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "quote_ref": "QuoteRef",
        "user_id": "UserID",
        "ask_price": "AskPrice",
        "bid_price": "BidPrice",
        "ask_volume": "AskVolume",
        "bid_volume": "BidVolume",
        "request_id": "RequestID",
        "business_unit": "BusinessUnit",
        "ask_offset_flag": "AskOffsetFlag",
        "bid_offset_flag": "BidOffsetFlag",
        "ask_hedge_flag": "AskHedgeFlag",
        "bid_hedge_flag": "BidHedgeFlag",
        "ask_order_ref": "AskOrderRef",
        "bid_order_ref": "BidOrderRef",
        "for_quote_sys_id": "ForQuoteSysID",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
        "replace_sys_id": "ReplaceSysID",
        "time_condition": "TimeCondition",
        "order_memo": "OrderMemo",
        "session_req_seq": "SessionReqSeq",
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
    def quote_ref(self) -> str:
        """报价引用"""
        if 'quote_ref' not in self._cache:
            value = self._struct.QuoteRef.rstrip(b'\x00').decode('ascii')
            self._cache['quote_ref'] = value
        return self._cache['quote_ref']

    @quote_ref.setter
    def quote_ref(self, value: str):
        """设置报价引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.QuoteRef = encoded
        self._cache['quote_ref'] = value

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
    def ask_price(self) -> float:
        """卖价"""
        return self._struct.AskPrice

    @ask_price.setter
    def ask_price(self, value: float):
        """设置卖价"""
        self._struct.AskPrice = value

    @property
    def bid_price(self) -> float:
        """买价"""
        return self._struct.BidPrice

    @bid_price.setter
    def bid_price(self, value: float):
        """设置买价"""
        self._struct.BidPrice = value

    @property
    def ask_volume(self) -> int:
        """卖量"""
        return self._struct.AskVolume

    @ask_volume.setter
    def ask_volume(self, value: int):
        """设置卖量"""
        self._struct.AskVolume = value

    @property
    def bid_volume(self) -> int:
        """买量"""
        return self._struct.BidVolume

    @bid_volume.setter
    def bid_volume(self, value: int):
        """设置买量"""
        self._struct.BidVolume = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def business_unit(self) -> str:
        """业务单元"""
        if 'business_unit' not in self._cache:
            value = self._struct.BusinessUnit.rstrip(b'\x00').decode('ascii')
            self._cache['business_unit'] = value
        return self._cache['business_unit']

    @business_unit.setter
    def business_unit(self, value: str):
        """设置业务单元"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.BusinessUnit = encoded
        self._cache['business_unit'] = value

    @property
    def ask_offset_flag(self) -> str:
        """卖开平标志"""
        if 'ask_offset_flag' not in self._cache:
            value = self._struct.AskOffsetFlag.decode('ascii')
            self._cache['ask_offset_flag'] = value
        return self._cache['ask_offset_flag']

    @ask_offset_flag.setter
    def ask_offset_flag(self, value: str):
        """设置卖开平标志"""
        self._struct.AskOffsetFlag = value.encode('ascii')[0]
        self._cache['ask_offset_flag'] = value

    @property
    def bid_offset_flag(self) -> str:
        """买开平标志"""
        if 'bid_offset_flag' not in self._cache:
            value = self._struct.BidOffsetFlag.decode('ascii')
            self._cache['bid_offset_flag'] = value
        return self._cache['bid_offset_flag']

    @bid_offset_flag.setter
    def bid_offset_flag(self, value: str):
        """设置买开平标志"""
        self._struct.BidOffsetFlag = value.encode('ascii')[0]
        self._cache['bid_offset_flag'] = value

    @property
    def ask_hedge_flag(self) -> str:
        """卖投机套保标志"""
        if 'ask_hedge_flag' not in self._cache:
            value = self._struct.AskHedgeFlag.decode('ascii')
            self._cache['ask_hedge_flag'] = value
        return self._cache['ask_hedge_flag']

    @ask_hedge_flag.setter
    def ask_hedge_flag(self, value: str):
        """设置卖投机套保标志"""
        self._struct.AskHedgeFlag = value.encode('ascii')[0]
        self._cache['ask_hedge_flag'] = value

    @property
    def bid_hedge_flag(self) -> str:
        """买投机套保标志"""
        if 'bid_hedge_flag' not in self._cache:
            value = self._struct.BidHedgeFlag.decode('ascii')
            self._cache['bid_hedge_flag'] = value
        return self._cache['bid_hedge_flag']

    @bid_hedge_flag.setter
    def bid_hedge_flag(self, value: str):
        """设置买投机套保标志"""
        self._struct.BidHedgeFlag = value.encode('ascii')[0]
        self._cache['bid_hedge_flag'] = value

    @property
    def ask_order_ref(self) -> str:
        """卖报单引用"""
        if 'ask_order_ref' not in self._cache:
            value = self._struct.AskOrderRef.rstrip(b'\x00').decode('ascii')
            self._cache['ask_order_ref'] = value
        return self._cache['ask_order_ref']

    @ask_order_ref.setter
    def ask_order_ref(self, value: str):
        """设置卖报单引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AskOrderRef = encoded
        self._cache['ask_order_ref'] = value

    @property
    def bid_order_ref(self) -> str:
        """买报单引用"""
        if 'bid_order_ref' not in self._cache:
            value = self._struct.BidOrderRef.rstrip(b'\x00').decode('ascii')
            self._cache['bid_order_ref'] = value
        return self._cache['bid_order_ref']

    @bid_order_ref.setter
    def bid_order_ref(self, value: str):
        """设置买报单引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BidOrderRef = encoded
        self._cache['bid_order_ref'] = value

    @property
    def for_quote_sys_id(self) -> str:
        """询价编号"""
        if 'for_quote_sys_id' not in self._cache:
            value = self._struct.ForQuoteSysID.rstrip(b'\x00').decode('ascii')
            self._cache['for_quote_sys_id'] = value
        return self._cache['for_quote_sys_id']

    @for_quote_sys_id.setter
    def for_quote_sys_id(self, value: str):
        """设置询价编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ForQuoteSysID = encoded
        self._cache['for_quote_sys_id'] = value

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

    @property
    def client_id(self) -> str:
        """交易编码"""
        if 'client_id' not in self._cache:
            value = self._struct.ClientID.rstrip(b'\x00').decode('ascii')
            self._cache['client_id'] = value
        return self._cache['client_id']

    @client_id.setter
    def client_id(self, value: str):
        """设置交易编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClientID = encoded
        self._cache['client_id'] = value

    @property
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value

    @property
    def replace_sys_id(self) -> str:
        """被替换的报价编号"""
        if 'replace_sys_id' not in self._cache:
            value = self._struct.ReplaceSysID.rstrip(b'\x00').decode('ascii')
            self._cache['replace_sys_id'] = value
        return self._cache['replace_sys_id']

    @replace_sys_id.setter
    def replace_sys_id(self, value: str):
        """设置被替换的报价编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ReplaceSysID = encoded
        self._cache['replace_sys_id'] = value

    @property
    def time_condition(self) -> str:
        """时间条件"""
        if 'time_condition' not in self._cache:
            value = self._struct.TimeCondition.decode('ascii')
            self._cache['time_condition'] = value
        return self._cache['time_condition']

    @time_condition.setter
    def time_condition(self, value: str):
        """设置时间条件"""
        self._struct.TimeCondition = value.encode('ascii')[0]
        self._cache['time_condition'] = value

    @property
    def order_memo(self) -> str:
        """报单备注"""
        if 'order_memo' not in self._cache:
            value = self._struct.OrderMemo.rstrip(b'\x00').decode('ascii')
            self._cache['order_memo'] = value
        return self._cache['order_memo']

    @order_memo.setter
    def order_memo(self, value: str):
        """设置报单备注"""
        encoded = value.encode('ascii')[:60].ljust(61, b'\x00')
        self._struct.OrderMemo = encoded
        self._cache['order_memo'] = value

    @property
    def session_req_seq(self) -> int:
        """会话请求序号"""
        return self._struct.SessionReqSeq

    @session_req_seq.setter
    def session_req_seq(self, value: int):
        """设置会话请求序号"""
        self._struct.SessionReqSeq = value


# =============================================================================
# InputQuoteAction - 报价操作请求
# =============================================================================


class InputQuoteAction(CapsuleStruct):
    """报价操作请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("QuoteActionRef", ctypes.c_char * 13),      # 报价操作引用
            ("QuoteRef", ctypes.c_char * 13),            # 报价引用
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("QuoteSysID", ctypes.c_char * 21),          # 报价编号
            ("ActionFlag", ctypes.c_char),               # 操作标志
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("ClientID", ctypes.c_char * 11),            # 交易编码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
            ("OrderMemo", ctypes.c_char * 61),           # 报单备注
            ("SessionReqSeq", ctypes.c_short),           # 会话请求序号
        ]

    _capsule_name = "InputQuoteAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "quote_action_ref": "QuoteActionRef",
        "quote_ref": "QuoteRef",
        "request_id": "RequestID",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "exchange_id": "ExchangeID",
        "quote_sys_id": "QuoteSysID",
        "action_flag": "ActionFlag",
        "user_id": "UserID",
        "reserve1": "reserve1",
        "invest_unit_id": "InvestUnitID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
        "order_memo": "OrderMemo",
        "session_req_seq": "SessionReqSeq",
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
    def quote_action_ref(self) -> str:
        """报价操作引用"""
        if 'quote_action_ref' not in self._cache:
            value = self._struct.QuoteActionRef.rstrip(b'\x00').decode('ascii')
            self._cache['quote_action_ref'] = value
        return self._cache['quote_action_ref']

    @quote_action_ref.setter
    def quote_action_ref(self, value: str):
        """设置报价操作引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.QuoteActionRef = encoded
        self._cache['quote_action_ref'] = value

    @property
    def quote_ref(self) -> str:
        """报价引用"""
        if 'quote_ref' not in self._cache:
            value = self._struct.QuoteRef.rstrip(b'\x00').decode('ascii')
            self._cache['quote_ref'] = value
        return self._cache['quote_ref']

    @quote_ref.setter
    def quote_ref(self, value: str):
        """设置报价引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.QuoteRef = encoded
        self._cache['quote_ref'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

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
    def action_flag(self) -> str:
        """操作标志"""
        if 'action_flag' not in self._cache:
            value = self._struct.ActionFlag.decode('ascii')
            self._cache['action_flag'] = value
        return self._cache['action_flag']

    @action_flag.setter
    def action_flag(self, value: str):
        """设置操作标志"""
        self._struct.ActionFlag = value.encode('ascii')[0]
        self._cache['action_flag'] = value

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
    def client_id(self) -> str:
        """交易编码"""
        if 'client_id' not in self._cache:
            value = self._struct.ClientID.rstrip(b'\x00').decode('ascii')
            self._cache['client_id'] = value
        return self._cache['client_id']

    @client_id.setter
    def client_id(self, value: str):
        """设置交易编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClientID = encoded
        self._cache['client_id'] = value

    @property
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value

    @property
    def order_memo(self) -> str:
        """报单备注"""
        if 'order_memo' not in self._cache:
            value = self._struct.OrderMemo.rstrip(b'\x00').decode('ascii')
            self._cache['order_memo'] = value
        return self._cache['order_memo']

    @order_memo.setter
    def order_memo(self, value: str):
        """设置报单备注"""
        encoded = value.encode('ascii')[:60].ljust(61, b'\x00')
        self._struct.OrderMemo = encoded
        self._cache['order_memo'] = value

    @property
    def session_req_seq(self) -> int:
        """会话请求序号"""
        return self._struct.SessionReqSeq

    @session_req_seq.setter
    def session_req_seq(self, value: int):
        """设置会话请求序号"""
        self._struct.SessionReqSeq = value


# =============================================================================
# InputBatchOrderAction - 批量报单操作请求
# =============================================================================


class InputBatchOrderAction(CapsuleStruct):
    """批量报单操作请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("OrderActionRef", ctypes.c_char * 13),      # 报单操作引用
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("reserve1", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "InputBatchOrderAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "order_action_ref": "OrderActionRef",
        "request_id": "RequestID",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "exchange_id": "ExchangeID",
        "user_id": "UserID",
        "invest_unit_id": "InvestUnitID",
        "reserve1": "reserve1",
        "mac_address": "MacAddress",
        "ip_address": "IPAddress",
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
    def order_action_ref(self) -> str:
        """报单操作引用"""
        if 'order_action_ref' not in self._cache:
            value = self._struct.OrderActionRef.rstrip(b'\x00').decode('ascii')
            self._cache['order_action_ref'] = value
        return self._cache['order_action_ref']

    @order_action_ref.setter
    def order_action_ref(self, value: str):
        """设置报单操作引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OrderActionRef = encoded
        self._cache['order_action_ref'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

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
    def reserve1(self) -> str:
        """保留的无效字段"""
        if 'reserve1' not in self._cache:
            value = self._struct.reserve1.rstrip(b'\x00').decode('ascii')
            self._cache['reserve1'] = value
        return self._cache['reserve1']

    @reserve1.setter
    def reserve1(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve1 = encoded
        self._cache['reserve1'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

    @property
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value


# =============================================================================
# InputOptionSelfClose - 期权自对冲录入
# =============================================================================


class InputOptionSelfClose(CapsuleStruct):
    """期权自对冲录入"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("OptionSelfCloseRef", ctypes.c_char * 13),  # 期权自对冲引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Volume", ctypes.c_int),                    # 数量
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("OptSelfCloseFlag", ctypes.c_char),         # 期权行权头寸是否自对冲
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("AccountID", ctypes.c_char * 13),           # 资金账号
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
            ("ClientID", ctypes.c_char * 11),            # 交易编码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "InputOptionSelfClose"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "option_self_close_ref": "OptionSelfCloseRef",
        "user_id": "UserID",
        "volume": "Volume",
        "request_id": "RequestID",
        "business_unit": "BusinessUnit",
        "hedge_flag": "HedgeFlag",
        "opt_self_close_flag": "OptSelfCloseFlag",
        "exchange_id": "ExchangeID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
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
    def option_self_close_ref(self) -> str:
        """期权自对冲引用"""
        if 'option_self_close_ref' not in self._cache:
            value = self._struct.OptionSelfCloseRef.rstrip(b'\x00').decode('ascii')
            self._cache['option_self_close_ref'] = value
        return self._cache['option_self_close_ref']

    @option_self_close_ref.setter
    def option_self_close_ref(self, value: str):
        """设置期权自对冲引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OptionSelfCloseRef = encoded
        self._cache['option_self_close_ref'] = value

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
    def volume(self) -> int:
        """数量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置数量"""
        self._struct.Volume = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def business_unit(self) -> str:
        """业务单元"""
        if 'business_unit' not in self._cache:
            value = self._struct.BusinessUnit.rstrip(b'\x00').decode('ascii')
            self._cache['business_unit'] = value
        return self._cache['business_unit']

    @business_unit.setter
    def business_unit(self, value: str):
        """设置业务单元"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.BusinessUnit = encoded
        self._cache['business_unit'] = value

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
    def opt_self_close_flag(self) -> str:
        """期权行权头寸是否自对冲"""
        if 'opt_self_close_flag' not in self._cache:
            value = self._struct.OptSelfCloseFlag.decode('ascii')
            self._cache['opt_self_close_flag'] = value
        return self._cache['opt_self_close_flag']

    @opt_self_close_flag.setter
    def opt_self_close_flag(self, value: str):
        """设置期权行权头寸是否自对冲"""
        self._struct.OptSelfCloseFlag = value.encode('ascii')[0]
        self._cache['opt_self_close_flag'] = value

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

    @property
    def client_id(self) -> str:
        """交易编码"""
        if 'client_id' not in self._cache:
            value = self._struct.ClientID.rstrip(b'\x00').decode('ascii')
            self._cache['client_id'] = value
        return self._cache['client_id']

    @client_id.setter
    def client_id(self, value: str):
        """设置交易编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClientID = encoded
        self._cache['client_id'] = value

    @property
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value


# =============================================================================
# InputOptionSelfCloseAction - 期权自对冲操作请求
# =============================================================================


class InputOptionSelfCloseAction(CapsuleStruct):
    """期权自对冲操作请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("OptionSelfCloseActionRef", ctypes.c_char * 13),  # 期权自对冲操作引用
            ("OptionSelfCloseRef", ctypes.c_char * 13),  # 期权自对冲引用
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("OptionSelfCloseSysID", ctypes.c_char * 21),  # 期权自对冲操作编号
            ("ActionFlag", ctypes.c_char),               # 操作标志
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "InputOptionSelfCloseAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "option_self_close_action_ref": "OptionSelfCloseActionRef",
        "option_self_close_ref": "OptionSelfCloseRef",
        "request_id": "RequestID",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "exchange_id": "ExchangeID",
        "option_self_close_sys_id": "OptionSelfCloseSysID",
        "action_flag": "ActionFlag",
        "user_id": "UserID",
        "reserve1": "reserve1",
        "invest_unit_id": "InvestUnitID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
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
    def option_self_close_action_ref(self) -> str:
        """期权自对冲操作引用"""
        if 'option_self_close_action_ref' not in self._cache:
            value = self._struct.OptionSelfCloseActionRef.rstrip(b'\x00').decode('ascii')
            self._cache['option_self_close_action_ref'] = value
        return self._cache['option_self_close_action_ref']

    @option_self_close_action_ref.setter
    def option_self_close_action_ref(self, value: str):
        """设置期权自对冲操作引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OptionSelfCloseActionRef = encoded
        self._cache['option_self_close_action_ref'] = value

    @property
    def option_self_close_ref(self) -> str:
        """期权自对冲引用"""
        if 'option_self_close_ref' not in self._cache:
            value = self._struct.OptionSelfCloseRef.rstrip(b'\x00').decode('ascii')
            self._cache['option_self_close_ref'] = value
        return self._cache['option_self_close_ref']

    @option_self_close_ref.setter
    def option_self_close_ref(self, value: str):
        """设置期权自对冲引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OptionSelfCloseRef = encoded
        self._cache['option_self_close_ref'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

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
    def option_self_close_sys_id(self) -> str:
        """期权自对冲操作编号"""
        if 'option_self_close_sys_id' not in self._cache:
            value = self._struct.OptionSelfCloseSysID.rstrip(b'\x00').decode('ascii')
            self._cache['option_self_close_sys_id'] = value
        return self._cache['option_self_close_sys_id']

    @option_self_close_sys_id.setter
    def option_self_close_sys_id(self, value: str):
        """设置期权自对冲操作编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.OptionSelfCloseSysID = encoded
        self._cache['option_self_close_sys_id'] = value

    @property
    def action_flag(self) -> str:
        """操作标志"""
        if 'action_flag' not in self._cache:
            value = self._struct.ActionFlag.decode('ascii')
            self._cache['action_flag'] = value
        return self._cache['action_flag']

    @action_flag.setter
    def action_flag(self, value: str):
        """设置操作标志"""
        self._struct.ActionFlag = value.encode('ascii')[0]
        self._cache['action_flag'] = value

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
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

    @property
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value


# =============================================================================
# InputCombAction - 组合报单操作请求
# =============================================================================


class InputCombAction(CapsuleStruct):
    """组合报单操作请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("CombActionRef", ctypes.c_char * 13),       # 组合引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Direction", ctypes.c_char),                # 买卖方向
            ("Volume", ctypes.c_int),                    # 数量
            ("CombDirection", ctypes.c_char),            # 组合指令方向
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "InputCombAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "comb_action_ref": "CombActionRef",
        "user_id": "UserID",
        "direction": "Direction",
        "volume": "Volume",
        "comb_direction": "CombDirection",
        "hedge_flag": "HedgeFlag",
        "exchange_id": "ExchangeID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "invest_unit_id": "InvestUnitID",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
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
    def comb_action_ref(self) -> str:
        """组合引用"""
        if 'comb_action_ref' not in self._cache:
            value = self._struct.CombActionRef.rstrip(b'\x00').decode('ascii')
            self._cache['comb_action_ref'] = value
        return self._cache['comb_action_ref']

    @comb_action_ref.setter
    def comb_action_ref(self, value: str):
        """设置组合引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.CombActionRef = encoded
        self._cache['comb_action_ref'] = value

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
    def volume(self) -> int:
        """数量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置数量"""
        self._struct.Volume = value

    @property
    def comb_direction(self) -> str:
        """组合指令方向"""
        if 'comb_direction' not in self._cache:
            value = self._struct.CombDirection.decode('ascii')
            self._cache['comb_direction'] = value
        return self._cache['comb_direction']

    @comb_direction.setter
    def comb_direction(self, value: str):
        """设置组合指令方向"""
        self._struct.CombDirection = value.encode('ascii')[0]
        self._cache['comb_direction'] = value

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
    def reserve2(self) -> str:
        """保留的无效字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value


# =============================================================================
# Order - 报单
# =============================================================================


class InputOffsetSetting(CapsuleStruct):
    """输入对冲设置"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("UnderlyingInstrID", ctypes.c_char * 81),  # 标的期货合约代码
            ("ProductID", ctypes.c_char * 41),          # 产品代码
            ("OffsetType", ctypes.c_char),              # 对冲类型
            ("Volume", ctypes.c_int),                   # 申请对冲的合约数量
            ("IsOffset", ctypes.c_int),                 # 是否对冲
            ("RequestID", ctypes.c_int),                # 请求编号
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("IPAddress", ctypes.c_char * 33),          # IP地址
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
        ]

    _capsule_name = "InputOffsetSetting"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "instrument_id": "InstrumentID",
        "underlying_instr_id": "UnderlyingInstrID",
        "product_id": "ProductID",
        "offset_type": "OffsetType",
        "volume": "Volume",
        "is_offset": "IsOffset",
        "request_id": "RequestID",
        "user_id": "UserID",
        "exchange_id": "ExchangeID",
        "ip_address": "IPAddress",
        "mac_address": "MacAddress",
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
    def underlying_instr_id(self) -> str:
        """标的期货合约代码"""
        if 'underlying_instr_id' not in self._cache:
            value = self._struct.UnderlyingInstrID.rstrip(b'\x00').decode('ascii')
            self._cache['underlying_instr_id'] = value
        return self._cache['underlying_instr_id']

    @underlying_instr_id.setter
    def underlying_instr_id(self, value: str):
        """设置标的期货合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.UnderlyingInstrID = encoded
        self._cache['underlying_instr_id'] = value

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

    @property
    def volume(self) -> int:
        """申请对冲的合约数量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置申请对冲的合约数量"""
        self._struct.Volume = value

    @property
    def is_offset(self) -> int:
        """是否对冲"""
        return self._struct.IsOffset

    @is_offset.setter
    def is_offset(self, value: int):
        """设置是否对冲"""
        self._struct.IsOffset = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

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
    def ip_address(self) -> str:
        """IP地址"""
        if 'ip_address' not in self._cache:
            value = self._struct.IPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['ip_address'] = value
        return self._cache['ip_address']

    @ip_address.setter
    def ip_address(self, value: str):
        """设置IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.IPAddress = encoded
        self._cache['ip_address'] = value

    @property
    def mac_address(self) -> str:
        """Mac地址"""
        if 'mac_address' not in self._cache:
            value = self._struct.MacAddress.rstrip(b'\x00').decode('ascii')
            self._cache['mac_address'] = value
        return self._cache['mac_address']

    @mac_address.setter
    def mac_address(self, value: str):
        """设置Mac地址"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.MacAddress = encoded
        self._cache['mac_address'] = value




