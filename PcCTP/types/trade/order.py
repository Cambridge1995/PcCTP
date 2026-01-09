"""
Trade
"""

import ctypes
from PcCTP.types.base import CapsuleStruct
from typing import Dict

class Order(CapsuleStruct):
    """报单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 9),             # 保留的无效字段
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
            ("OrderLocalID", ctypes.c_char * 13),        # 本地报单编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),       # 参与者代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("reserve2", ctypes.c_char * 21),            # 保留的无效字段
            ("TraderID", ctypes.c_char * 21),            # 交易员代码
            ("InstallID", ctypes.c_int),                 # 安装编号
            ("OrderSubmitStatus", ctypes.c_char),        # 报单提交状态
            ("NotifySequence", ctypes.c_int),            # 报单提示序号
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("OrderSysID", ctypes.c_char * 21),          # 报单编号
            ("OrderSource", ctypes.c_char),              # 报单来源
            ("OrderStatus", ctypes.c_char),              # 报单状态
            ("OrderType", ctypes.c_char),                # 报单类型
            ("VolumeTraded", ctypes.c_int),              # 今成交数量
            ("VolumeTotal", ctypes.c_int),               # 剩余数量
            ("InsertDate", ctypes.c_char * 9),           # 报单日期
            ("InsertTime", ctypes.c_char * 9),           # 委托时间
            ("ActiveTime", ctypes.c_char * 9),           # 激活时间
            ("SuspendTime", ctypes.c_char * 9),          # 挂起时间
            ("UpdateTime", ctypes.c_char * 9),           # 最后修改时间
            ("CancelTime", ctypes.c_char * 9),           # 撤销时间
            ("ActiveTraderID", ctypes.c_char * 21),      # 最后修改交易所交易员代码
            ("ClearingPartID", ctypes.c_char * 11),      # 结算会员编号
            ("SequenceNo", ctypes.c_int),                # 序号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("UserProductInfo", ctypes.c_char * 11),     # 用户端产品信息
            ("StatusMsg", ctypes.c_char * 81),           # 状态信息
            ("UserForceClose", ctypes.c_int),            # 用户强平标志
            ("ActiveUserID", ctypes.c_char * 16),        # 操作用户代码
            ("BrokerOrderSeq", ctypes.c_int),            # 经纪公司报单序号
            ("RelativeOrderSysID", ctypes.c_char * 21),  # 相关报单
            ("ZCETotalTradedVolume", ctypes.c_int),      # 郑商所成交数量
            ("IsSwapOrder", ctypes.c_int),               # 互换单标志
            ("BranchID", ctypes.c_char * 9),             # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("AccountID", ctypes.c_char * 13),           # 资金账号
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
            ("reserve3", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 81),      # 合约在交易所的代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
            ("OrderMemo", ctypes.c_char * 13),           # 报单回显字段
            ("SessionReqSeq", ctypes.c_int),             # session上请求计数
        ]

    _capsule_name = "Order"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
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
        "order_local_id": "OrderLocalID",
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "order_submit_status": "OrderSubmitStatus",
        "notify_sequence": "NotifySequence",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "order_sys_id": "OrderSysID",
        "order_source": "OrderSource",
        "order_status": "OrderStatus",
        "order_type": "OrderType",
        "volume_traded": "VolumeTraded",
        "volume_total": "VolumeTotal",
        "insert_date": "InsertDate",
        "insert_time": "InsertTime",
        "active_time": "ActiveTime",
        "suspend_time": "SuspendTime",
        "update_time": "UpdateTime",
        "cancel_time": "CancelTime",
        "active_trader_id": "ActiveTraderID",
        "clearing_part_id": "ClearingPartID",
        "sequence_no": "SequenceNo",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "user_product_info": "UserProductInfo",
        "status_msg": "StatusMsg",
        "user_force_close": "UserForceClose",
        "active_user_id": "ActiveUserID",
        "broker_order_seq": "BrokerOrderSeq",
        "relative_order_sys_id": "RelativeOrderSysID",
        "zce_total_traded_volume": "ZCETotalTradedVolume",
        "is_swap_order": "IsSwapOrder",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
        "ip_address": "IPAddress",
        "order_memo": "OrderMemo",
        "session_req_seq": "SessionReqSeq",
    }

    # 字符串属性（懒加载 + 缓存）
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
    def order_local_id(self) -> str:
        """本地报单编号"""
        if 'order_local_id' not in self._cache:
            value = self._struct.OrderLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['order_local_id'] = value
        return self._cache['order_local_id']

    @order_local_id.setter
    def order_local_id(self, value: str):
        """设置本地报单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OrderLocalID = encoded
        self._cache['order_local_id'] = value

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
        """参与者代码"""
        if 'participant_id' not in self._cache:
            value = self._struct.ParticipantID.rstrip(b'\x00').decode('ascii')
            self._cache['participant_id'] = value
        return self._cache['participant_id']

    @participant_id.setter
    def participant_id(self, value: str):
        """设置参与者代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ParticipantID = encoded
        self._cache['participant_id'] = value

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
    def trader_id(self) -> str:
        """交易员代码"""
        if 'trader_id' not in self._cache:
            value = self._struct.TraderID.rstrip(b'\x00').decode('ascii')
            self._cache['trader_id'] = value
        return self._cache['trader_id']

    @trader_id.setter
    def trader_id(self, value: str):
        """设置交易员代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.TraderID = encoded
        self._cache['trader_id'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def order_submit_status(self) -> str:
        """报单提交状态"""
        if 'order_submit_status' not in self._cache:
            value = self._struct.OrderSubmitStatus.decode('ascii')
            self._cache['order_submit_status'] = value
        return self._cache['order_submit_status']

    @order_submit_status.setter
    def order_submit_status(self, value: str):
        """设置报单提交状态"""
        self._struct.OrderSubmitStatus = value.encode('ascii')[0]
        self._cache['order_submit_status'] = value

    @property
    def notify_sequence(self) -> int:
        """报单提示序号"""
        return self._struct.NotifySequence

    @notify_sequence.setter
    def notify_sequence(self, value: int):
        """设置报单提示序号"""
        self._struct.NotifySequence = value

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
    def order_source(self) -> str:
        """报单来源"""
        if 'order_source' not in self._cache:
            value = self._struct.OrderSource.decode('ascii')
            self._cache['order_source'] = value
        return self._cache['order_source']

    @order_source.setter
    def order_source(self, value: str):
        """设置报单来源"""
        self._struct.OrderSource = value.encode('ascii')[0]
        self._cache['order_source'] = value

    @property
    def order_status(self) -> str:
        """报单状态"""
        if 'order_status' not in self._cache:
            value = self._struct.OrderStatus.decode('ascii')
            self._cache['order_status'] = value
        return self._cache['order_status']

    @order_status.setter
    def order_status(self, value: str):
        """设置报单状态"""
        self._struct.OrderStatus = value.encode('ascii')[0]
        self._cache['order_status'] = value

    @property
    def order_type(self) -> str:
        """报单类型"""
        if 'order_type' not in self._cache:
            value = self._struct.OrderType.decode('ascii')
            self._cache['order_type'] = value
        return self._cache['order_type']

    @order_type.setter
    def order_type(self, value: str):
        """设置报单类型"""
        self._struct.OrderType = value.encode('ascii')[0]
        self._cache['order_type'] = value

    @property
    def volume_traded(self) -> int:
        """成交数量"""
        return self._struct.VolumeTraded

    @volume_traded.setter
    def volume_traded(self, value: int):
        """设置成交数量"""
        self._struct.VolumeTraded = value

    @property
    def volume_total(self) -> int:
        """剩余数量"""
        return self._struct.VolumeTotal

    @volume_total.setter
    def volume_total(self, value: int):
        """设置剩余数量"""
        self._struct.VolumeTotal = value

    @property
    def insert_date(self) -> str:
        """报单日期"""
        if 'insert_date' not in self._cache:
            value = self._struct.InsertDate.rstrip(b'\x00').decode('ascii')
            self._cache['insert_date'] = value
        return self._cache['insert_date']

    @insert_date.setter
    def insert_date(self, value: str):
        """设置报单日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertDate = encoded
        self._cache['insert_date'] = value

    @property
    def insert_time(self) -> str:
        """委托时间"""
        if 'insert_time' not in self._cache:
            value = self._struct.InsertTime.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time'] = value
        return self._cache['insert_time']

    @insert_time.setter
    def insert_time(self, value: str):
        """设置委托时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTime = encoded
        self._cache['insert_time'] = value

    @property
    def active_time(self) -> str:
        """激活时间"""
        if 'active_time' not in self._cache:
            value = self._struct.ActiveTime.rstrip(b'\x00').decode('ascii')
            self._cache['active_time'] = value
        return self._cache['active_time']

    @active_time.setter
    def active_time(self, value: str):
        """设置激活时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActiveTime = encoded
        self._cache['active_time'] = value

    @property
    def suspend_time(self) -> str:
        """挂起时间"""
        if 'suspend_time' not in self._cache:
            value = self._struct.SuspendTime.rstrip(b'\x00').decode('ascii')
            self._cache['suspend_time'] = value
        return self._cache['suspend_time']

    @suspend_time.setter
    def suspend_time(self, value: str):
        """设置挂起时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.SuspendTime = encoded
        self._cache['suspend_time'] = value

    @property
    def update_time(self) -> str:
        """最后修改时间"""
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
    def cancel_time(self) -> str:
        """撤销时间"""
        if 'cancel_time' not in self._cache:
            value = self._struct.CancelTime.rstrip(b'\x00').decode('ascii')
            self._cache['cancel_time'] = value
        return self._cache['cancel_time']

    @cancel_time.setter
    def cancel_time(self, value: str):
        """设置撤销时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.CancelTime = encoded
        self._cache['cancel_time'] = value

    @property
    def active_trader_id(self) -> str:
        """最后修改交易所交易员代码"""
        if 'active_trader_id' not in self._cache:
            value = self._struct.ActiveTraderID.rstrip(b'\x00').decode('ascii')
            self._cache['active_trader_id'] = value
        return self._cache['active_trader_id']

    @active_trader_id.setter
    def active_trader_id(self, value: str):
        """设置最后修改交易所交易员代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ActiveTraderID = encoded
        self._cache['active_trader_id'] = value

    @property
    def clearing_part_id(self) -> str:
        """结算会员编号"""
        if 'clearing_part_id' not in self._cache:
            value = self._struct.ClearingPartID.rstrip(b'\x00').decode('ascii')
            self._cache['clearing_part_id'] = value
        return self._cache['clearing_part_id']

    @clearing_part_id.setter
    def clearing_part_id(self, value: str):
        """设置结算会员编号"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClearingPartID = encoded
        self._cache['clearing_part_id'] = value

    @property
    def sequence_no(self) -> int:
        """序号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序号"""
        self._struct.SequenceNo = value

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
    def user_product_info(self) -> str:
        """用户端产品信息"""
        if 'user_product_info' not in self._cache:
            value = self._struct.UserProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['user_product_info'] = value
        return self._cache['user_product_info']

    @user_product_info.setter
    def user_product_info(self, value: str):
        """设置用户端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.UserProductInfo = encoded
        self._cache['user_product_info'] = value

    @property
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

    @property
    def user_force_close(self) -> int:
        """用户强平标志"""
        return self._struct.UserForceClose

    @user_force_close.setter
    def user_force_close(self, value: int):
        """设置用户强平标志"""
        self._struct.UserForceClose = value

    @property
    def active_user_id(self) -> str:
        """操作用户代码"""
        if 'active_user_id' not in self._cache:
            value = self._struct.ActiveUserID.rstrip(b'\x00').decode('ascii')
            self._cache['active_user_id'] = value
        return self._cache['active_user_id']

    @active_user_id.setter
    def active_user_id(self, value: str):
        """设置操作用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.ActiveUserID = encoded
        self._cache['active_user_id'] = value

    @property
    def broker_order_seq(self) -> int:
        """经纪公司报单序号"""
        return self._struct.BrokerOrderSeq

    @broker_order_seq.setter
    def broker_order_seq(self, value: int):
        """设置经纪公司报单序号"""
        self._struct.BrokerOrderSeq = value

    @property
    def relative_order_sys_id(self) -> str:
        """相关报单编号"""
        if 'relative_order_sys_id' not in self._cache:
            value = self._struct.RelativeOrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['relative_order_sys_id'] = value
        return self._cache['relative_order_sys_id']

    @relative_order_sys_id.setter
    def relative_order_sys_id(self, value: str):
        """设置相关报单编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.RelativeOrderSysID = encoded
        self._cache['relative_order_sys_id'] = value

    @property
    def zce_total_traded_volume(self) -> int:
        """郑商所总成交数量"""
        return self._struct.ZCETotalTradedVolume

    @zce_total_traded_volume.setter
    def zce_total_traded_volume(self, value: int):
        """设置郑商所总成交数量"""
        self._struct.ZCETotalTradedVolume = value

    @property
    def is_swap_order(self) -> int:
        """互换单标志"""
        return self._struct.IsSwapOrder

    @is_swap_order.setter
    def is_swap_order(self, value: int):
        """设置互换单标志"""
        self._struct.IsSwapOrder = value

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
        """session上请求计数"""
        return self._struct.SessionReqSeq

    @session_req_seq.setter
    def session_req_seq(self, value: int):
        """设置session上请求计数"""
        self._struct.SessionReqSeq = value



# =============================================================================
# Trade - 成交
# =============================================================================


class Trade(CapsuleStruct):
    """成交"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 9),             # 保留的无效字段
            ("OrderRef", ctypes.c_char * 13),            # 报单引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("TradeID", ctypes.c_char * 21),             # 成交编号
            ("Direction", ctypes.c_char),                # 买卖方向
            ("OrderSysID", ctypes.c_char * 21),          # 报单编号
            ("ParticipantID", ctypes.c_char * 11),       # 参与者代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("TradingRole", ctypes.c_char),              # 交易角色
            ("reserve2", ctypes.c_char * 21),            # 保留的无效字段
            ("OffsetFlag", ctypes.c_char),               # 开平标志
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("Price", ctypes.c_double),                  # 成交价
            ("Volume", ctypes.c_int),                    # 成交量
            ("TradeDate", ctypes.c_char * 9),            # 成交日期
            ("TradeTime", ctypes.c_char * 9),            # 成交时间
            ("TradeType", ctypes.c_char),                # 成交类型
            ("PriceSource", ctypes.c_char),              # 成交价来源
            ("TraderID", ctypes.c_char * 21),            # 交易所交易员代码
            ("OrderLocalID", ctypes.c_char * 13),        # 本地报单编号
            ("ClearingPartID", ctypes.c_char * 11),      # 结算会员编号
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("SequenceNo", ctypes.c_int),                # 序号
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("BrokerOrderSeq", ctypes.c_int),            # 经纪公司报单编号
            ("TradeSource", ctypes.c_char),              # 成交来源
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 81),      # 合约在交易所的代码
        ]

    _capsule_name = "Trade"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "order_ref": "OrderRef",
        "user_id": "UserID",
        "exchange_id": "ExchangeID",
        "trade_id": "TradeID",
        "direction": "Direction",
        "order_sys_id": "OrderSysID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "trading_role": "TradingRole",
        "offset_flag": "OffsetFlag",
        "hedge_flag": "HedgeFlag",
        "price": "Price",
        "volume": "Volume",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "trade_type": "TradeType",
        "price_source": "PriceSource",
        "trader_id": "TraderID",
        "order_local_id": "OrderLocalID",
        "clearing_part_id": "ClearingPartID",
        "business_unit": "BusinessUnit",
        "sequence_no": "SequenceNo",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "broker_order_seq": "BrokerOrderSeq",
        "trade_source": "TradeSource",
        "invest_unit_id": "InvestUnitID",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
    }

    # 字符串属性（懒加载 + 缓存）
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
    def participant_id(self) -> str:
        """参与者代码"""
        if 'participant_id' not in self._cache:
            value = self._struct.ParticipantID.rstrip(b'\x00').decode('ascii')
            self._cache['participant_id'] = value
        return self._cache['participant_id']

    @participant_id.setter
    def participant_id(self, value: str):
        """设置参与者代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ParticipantID = encoded
        self._cache['participant_id'] = value

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
    def trading_role(self) -> str:
        """交易角色"""
        if 'trading_role' not in self._cache:
            value = self._struct.TradingRole.decode('ascii')
            self._cache['trading_role'] = value
        return self._cache['trading_role']

    @trading_role.setter
    def trading_role(self, value: str):
        """设置交易角色"""
        self._struct.TradingRole = value.encode('ascii')[0]
        self._cache['trading_role'] = value

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
    def price(self) -> float:
        """成交价"""
        return self._struct.Price

    @price.setter
    def price(self, value: float):
        """设置成交价"""
        self._struct.Price = value

    @property
    def volume(self) -> int:
        """成交量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置成交量"""
        self._struct.Volume = value

    @property
    def trade_date(self) -> str:
        """成交日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置成交日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trade_time(self) -> str:
        """成交时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置成交时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

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
    def price_source(self) -> str:
        """成交价来源"""
        if 'price_source' not in self._cache:
            value = self._struct.PriceSource.decode('ascii')
            self._cache['price_source'] = value
        return self._cache['price_source']

    @price_source.setter
    def price_source(self, value: str):
        """设置成交价来源"""
        self._struct.PriceSource = value.encode('ascii')[0]
        self._cache['price_source'] = value

    @property
    def trader_id(self) -> str:
        """交易员代码"""
        if 'trader_id' not in self._cache:
            value = self._struct.TraderID.rstrip(b'\x00').decode('ascii')
            self._cache['trader_id'] = value
        return self._cache['trader_id']

    @trader_id.setter
    def trader_id(self, value: str):
        """设置交易员代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.TraderID = encoded
        self._cache['trader_id'] = value

    @property
    def order_local_id(self) -> str:
        """本地报单编号"""
        if 'order_local_id' not in self._cache:
            value = self._struct.OrderLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['order_local_id'] = value
        return self._cache['order_local_id']

    @order_local_id.setter
    def order_local_id(self, value: str):
        """设置本地报单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OrderLocalID = encoded
        self._cache['order_local_id'] = value

    @property
    def clearing_part_id(self) -> str:
        """结算参与编号"""
        if 'clearing_part_id' not in self._cache:
            value = self._struct.ClearingPartID.rstrip(b'\x00').decode('ascii')
            self._cache['clearing_part_id'] = value
        return self._cache['clearing_part_id']

    @clearing_part_id.setter
    def clearing_part_id(self, value: str):
        """设置结算参与编号"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClearingPartID = encoded
        self._cache['clearing_part_id'] = value

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
    def sequence_no(self) -> int:
        """序号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序号"""
        self._struct.SequenceNo = value

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
    def broker_order_seq(self) -> int:
        """经纪公司报单序号"""
        return self._struct.BrokerOrderSeq

    @broker_order_seq.setter
    def broker_order_seq(self, value: int):
        """设置经纪公司报单序号"""
        self._struct.BrokerOrderSeq = value

    @property
    def trade_source(self) -> str:
        """成交来源"""
        if 'trade_source' not in self._cache:
            value = self._struct.TradeSource.decode('ascii')
            self._cache['trade_source'] = value
        return self._cache['trade_source']

    @trade_source.setter
    def trade_source(self, value: str):
        """设置成交来源"""
        self._struct.TradeSource = value.encode('ascii')[0]
        self._cache['trade_source'] = value

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
# InvestorPosition - 投资者持仓
# =============================================================================


class OrderAction(CapsuleStruct):
    """报单操作"""

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
            ("ActionDate", ctypes.c_char * 9),           # 操作日期
            ("ActionTime", ctypes.c_char * 9),           # 操作时间
            ("TraderID", ctypes.c_char * 21),            # 交易所交易员代码
            ("InstallID", ctypes.c_int),                 # 安装编号
            ("OrderLocalID", ctypes.c_char * 13),        # 本地报单编号
            ("ActionLocalID", ctypes.c_char * 13),       # 操作本地编号
            ("ParticipantID", ctypes.c_char * 11),       # 会员代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("OrderActionStatus", ctypes.c_char),        # 报单操作状态
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("StatusMsg", ctypes.c_char * 81),           # 状态信息
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("BranchID", ctypes.c_char * 9),             # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("reserve2", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
            ("OrderMemo", ctypes.c_char * 51),           # 报单回显字段
            ("SessionReqSeq", ctypes.c_int),             # session上请求计数 api自动维护
        ]

    _capsule_name = "OrderAction"

    _field_mappings = {
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
        "action_date": "ActionDate",
        "action_time": "ActionTime",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "order_local_id": "OrderLocalID",
        "action_local_id": "ActionLocalID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "business_unit": "BusinessUnit",
        "order_action_status": "OrderActionStatus",
        "user_id": "UserID",
        "status_msg": "StatusMsg",
        "reserve1": "reserve1",
        "branch_id": "BranchID",
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

    @property
    def action_date(self) -> str:
        """操作日期"""
        if 'action_date' not in self._cache:
            value = self._struct.ActionDate.rstrip(b'\x00').decode('ascii')
            self._cache['action_date'] = value
        return self._cache['action_date']

    @action_date.setter
    def action_date(self, value: str):
        """设置操作日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionDate = encoded
        self._cache['action_date'] = value

    @property
    def action_time(self) -> str:
        """操作时间"""
        if 'action_time' not in self._cache:
            value = self._struct.ActionTime.rstrip(b'\x00').decode('ascii')
            self._cache['action_time'] = value
        return self._cache['action_time']

    @action_time.setter
    def action_time(self, value: str):
        """设置操作时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionTime = encoded
        self._cache['action_time'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def order_local_id(self) -> str:
        """本地报单编号"""
        if 'order_local_id' not in self._cache:
            value = self._struct.OrderLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['order_local_id'] = value
        return self._cache['order_local_id']

    @order_local_id.setter
    def order_local_id(self, value: str):
        """设置本地报单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OrderLocalID = encoded
        self._cache['order_local_id'] = value

    @property
    def action_local_id(self) -> str:
        """操作本地编号"""
        if 'action_local_id' not in self._cache:
            value = self._struct.ActionLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['action_local_id'] = value
        return self._cache['action_local_id']

    @action_local_id.setter
    def action_local_id(self, value: str):
        """设置操作本地编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ActionLocalID = encoded
        self._cache['action_local_id'] = value

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
    def order_action_status(self) -> str:
        """报单操作状态"""
        if 'order_action_status' not in self._cache:
            value = self._struct.OrderActionStatus.decode('ascii')
            self._cache['order_action_status'] = value
        return self._cache['order_action_status']

    @order_action_status.setter
    def order_action_status(self, value: str):
        """设置报单操作状态"""
        self._struct.OrderActionStatus = value.encode('ascii')[0]
        self._cache['order_action_status'] = value

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
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

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
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.OrderMemo = encoded
        self._cache['order_memo'] = value

    @property
    def session_req_seq(self) -> int:
        """session上请求计数 api自动维护"""
        return self._struct.SessionReqSeq

    @session_req_seq.setter
    def session_req_seq(self, value: int):
        """设置session上请求计数 api自动维护"""
        self._struct.SessionReqSeq = value



class ExecOrder(CapsuleStruct):
    """执行宣告"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExecOrderRef", ctypes.c_int),              # 执行宣告引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Volume", ctypes.c_int),                    # 数量
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("OffsetFlag", ctypes.c_char),               # 开平标志
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("ActionType", ctypes.c_char),               # 执行类型
            ("PosiDirection", ctypes.c_char),            # 保留头寸申请的持仓方向
            ("ReservePositionFlag", ctypes.c_char),      # 期权行权后是否保留期货头寸的标记
            ("CloseFlag", ctypes.c_char),                # 期权行权后生成的头寸是否自动平仓
            ("ExecOrderLocalID", ctypes.c_char * 13),    # 本地执行宣告编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),       # 会员代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("TraderID", ctypes.c_char * 21),            # 交易所交易员代码
            ("InstallID", ctypes.c_int),                 # 安装编号
            ("OrderSubmitStatus", ctypes.c_char),        # 执行宣告提交状态
            ("NotifySequence", ctypes.c_int),            # 报单提示序号
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("ExecOrderSysID", ctypes.c_char * 13),      # 执行宣告编号
            ("InsertDate", ctypes.c_char * 9),           # 报单日期
            ("InsertTime", ctypes.c_char * 9),           # 插入时间
            ("CancelTime", ctypes.c_char * 9),           # 撤销时间
            ("ExecResult", ctypes.c_int),                # 执行结果
            ("ClearingPartID", ctypes.c_char * 11),      # 结算分支代码
            ("SequenceNo", ctypes.c_int),                # 序号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("UserProductInfo", ctypes.c_char * 11),     # 用户端产品信息
            ("StatusMsg", ctypes.c_char * 81),           # 状态信息
            ("ActiveUserID", ctypes.c_char * 16),        # 操作用户代码
            ("BrokerExecOrderSeq", ctypes.c_int),        # 经纪公司执行宣告流水号
            ("BranchID", ctypes.c_char * 9),             # 分支机构代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
            ("reserve3", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 51),      # 合约在交易所的代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "ExecOrder"

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
        "exec_order_local_id": "ExecOrderLocalID",
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "order_submit_status": "OrderSubmitStatus",
        "notify_sequence": "NotifySequence",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "exec_order_sys_id": "ExecOrderSysID",
        "insert_date": "InsertDate",
        "insert_time": "InsertTime",
        "cancel_time": "CancelTime",
        "exec_result": "ExecResult",
        "clearing_part_id": "ClearingPartID",
        "sequence_no": "SequenceNo",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "user_product_info": "UserProductInfo",
        "status_msg": "StatusMsg",
        "active_user_id": "ActiveUserID",
        "broker_exec_order_seq": "BrokerExecOrderSeq",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "reserve3": "reserve3",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
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
    def exec_order_ref(self) -> int:
        """执行宣告引用"""
        return self._struct.ExecOrderRef

    @exec_order_ref.setter
    def exec_order_ref(self, value: int):
        """设置执行宣告引用"""
        self._struct.ExecOrderRef = value

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
    def exec_order_local_id(self) -> str:
        """本地执行宣告编号"""
        if 'exec_order_local_id' not in self._cache:
            value = self._struct.ExecOrderLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_local_id'] = value
        return self._cache['exec_order_local_id']

    @exec_order_local_id.setter
    def exec_order_local_id(self, value: str):
        """设置本地执行宣告编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ExecOrderLocalID = encoded
        self._cache['exec_order_local_id'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def order_submit_status(self) -> str:
        """执行宣告提交状态"""
        if 'order_submit_status' not in self._cache:
            value = self._struct.OrderSubmitStatus.decode('ascii')
            self._cache['order_submit_status'] = value
        return self._cache['order_submit_status']

    @order_submit_status.setter
    def order_submit_status(self, value: str):
        """设置执行宣告提交状态"""
        self._struct.OrderSubmitStatus = value.encode('ascii')[0]
        self._cache['order_submit_status'] = value

    @property
    def notify_sequence(self) -> int:
        """报单提示序号"""
        return self._struct.NotifySequence

    @notify_sequence.setter
    def notify_sequence(self, value: int):
        """设置报单提示序号"""
        self._struct.NotifySequence = value

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
    def exec_order_sys_id(self) -> str:
        """执行宣告编号"""
        if 'exec_order_sys_id' not in self._cache:
            value = self._struct.ExecOrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_sys_id'] = value
        return self._cache['exec_order_sys_id']

    @exec_order_sys_id.setter
    def exec_order_sys_id(self, value: str):
        """设置执行宣告编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ExecOrderSysID = encoded
        self._cache['exec_order_sys_id'] = value

    @property
    def insert_date(self) -> str:
        """报单日期"""
        if 'insert_date' not in self._cache:
            value = self._struct.InsertDate.rstrip(b'\x00').decode('ascii')
            self._cache['insert_date'] = value
        return self._cache['insert_date']

    @insert_date.setter
    def insert_date(self, value: str):
        """设置报单日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertDate = encoded
        self._cache['insert_date'] = value

    @property
    def insert_time(self) -> str:
        """插入时间"""
        if 'insert_time' not in self._cache:
            value = self._struct.InsertTime.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time'] = value
        return self._cache['insert_time']

    @insert_time.setter
    def insert_time(self, value: str):
        """设置插入时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTime = encoded
        self._cache['insert_time'] = value

    @property
    def cancel_time(self) -> str:
        """撤销时间"""
        if 'cancel_time' not in self._cache:
            value = self._struct.CancelTime.rstrip(b'\x00').decode('ascii')
            self._cache['cancel_time'] = value
        return self._cache['cancel_time']

    @cancel_time.setter
    def cancel_time(self, value: str):
        """设置撤销时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.CancelTime = encoded
        self._cache['cancel_time'] = value

    @property
    def exec_result(self) -> int:
        """执行结果"""
        return self._struct.ExecResult

    @exec_result.setter
    def exec_result(self, value: int):
        """设置执行结果"""
        self._struct.ExecResult = value

    @property
    def clearing_part_id(self) -> str:
        """结算分支代码"""
        if 'clearing_part_id' not in self._cache:
            value = self._struct.ClearingPartID.rstrip(b'\x00').decode('ascii')
            self._cache['clearing_part_id'] = value
        return self._cache['clearing_part_id']

    @clearing_part_id.setter
    def clearing_part_id(self, value: str):
        """设置结算分支代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClearingPartID = encoded
        self._cache['clearing_part_id'] = value

    @property
    def sequence_no(self) -> int:
        """序号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序号"""
        self._struct.SequenceNo = value

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
    def user_product_info(self) -> str:
        """用户端产品信息"""
        if 'user_product_info' not in self._cache:
            value = self._struct.UserProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['user_product_info'] = value
        return self._cache['user_product_info']

    @user_product_info.setter
    def user_product_info(self, value: str):
        """设置用户端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.UserProductInfo = encoded
        self._cache['user_product_info'] = value

    @property
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

    @property
    def active_user_id(self) -> str:
        """操作用户代码"""
        if 'active_user_id' not in self._cache:
            value = self._struct.ActiveUserID.rstrip(b'\x00').decode('ascii')
            self._cache['active_user_id'] = value
        return self._cache['active_user_id']

    @active_user_id.setter
    def active_user_id(self, value: str):
        """设置操作用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.ActiveUserID = encoded
        self._cache['active_user_id'] = value

    @property
    def broker_exec_order_seq(self) -> int:
        """经纪公司执行宣告流水号"""
        return self._struct.BrokerExecOrderSeq

    @broker_exec_order_seq.setter
    def broker_exec_order_seq(self, value: int):
        """设置经纪公司执行宣告流水号"""
        self._struct.BrokerExecOrderSeq = value

    @property
    def branch_id(self) -> str:
        """分支机构代码"""
        if 'branch_id' not in self._cache:
            value = self._struct.BranchID.rstrip(b'\x00').decode('ascii')
            self._cache['branch_id'] = value
        return self._cache['branch_id']

    @branch_id.setter
    def branch_id(self, value: str):
        """设置分支机构代码"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.BranchID = encoded
        self._cache['branch_id'] = value

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
    def exchange_inst_id(self) -> str:
        """合约在交易所的代码"""
        if 'exchange_inst_id' not in self._cache:
            value = self._struct.ExchangeInstID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_inst_id'] = value
        return self._cache['exchange_inst_id']

    @exchange_inst_id.setter
    def exchange_inst_id(self, value: str):
        """设置合约在交易所的代码"""
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.ExchangeInstID = encoded
        self._cache['exchange_inst_id'] = value

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



class ExecOrderAction(CapsuleStruct):
    """执行宣告操作"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("ExecOrderActionRef", ctypes.c_int),       # 执行宣告操作引用
            ("ExecOrderRef", ctypes.c_char * 13),       # 执行宣告引用
            ("RequestID", ctypes.c_int),                # 请求编号
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ExecOrderSysID", ctypes.c_char * 21),     # 执行宣告操作编号
            ("ActionFlag", ctypes.c_char),              # 操作标志
            ("ActionDate", ctypes.c_char * 9),          # 操作日期
            ("ActionTime", ctypes.c_char * 9),          # 操作时间
            ("TraderID", ctypes.c_char * 21),           # 交易所交易员代码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("ExecOrderLocalID", ctypes.c_char * 13),   # 本地执行宣告编号
            ("ActionLocalID", ctypes.c_char * 13),      # 操作本地编号
            ("ParticipantID", ctypes.c_char * 11),      # 会员代码
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("BusinessUnit", ctypes.c_char * 21),       # 业务单元
            ("OrderActionStatus", ctypes.c_char),       # 报单操作状态
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("ActionType", ctypes.c_char),              # 执行类型
            ("StatusMsg", ctypes.c_char * 81),          # 状态信息
            ("reserve1", ctypes.c_char * 31),           # 保留的无效字段
            ("BranchID", ctypes.c_char * 9),            # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("reserve2", ctypes.c_char * 16),           # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
        ]

    _capsule_name = "ExecOrderAction"

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
        "action_date": "ActionDate",
        "action_time": "ActionTime",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "exec_order_local_id": "ExecOrderLocalID",
        "action_local_id": "ActionLocalID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "business_unit": "BusinessUnit",
        "order_action_status": "OrderActionStatus",
        "user_id": "UserID",
        "action_type": "ActionType",
        "status_msg": "StatusMsg",
        "reserve1": "reserve1",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
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
    def exec_order_action_ref(self) -> int:
        """执行宣告操作引用"""
        return self._struct.ExecOrderActionRef

    @exec_order_action_ref.setter
    def exec_order_action_ref(self, value: int):
        """设置执行宣告操作引用"""
        self._struct.ExecOrderActionRef = value

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
    def action_date(self) -> str:
        """操作日期"""
        if 'action_date' not in self._cache:
            value = self._struct.ActionDate.rstrip(b'\x00').decode('ascii')
            self._cache['action_date'] = value
        return self._cache['action_date']

    @action_date.setter
    def action_date(self, value: str):
        """设置操作日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionDate = encoded
        self._cache['action_date'] = value

    @property
    def action_time(self) -> str:
        """操作时间"""
        if 'action_time' not in self._cache:
            value = self._struct.ActionTime.rstrip(b'\x00').decode('ascii')
            self._cache['action_time'] = value
        return self._cache['action_time']

    @action_time.setter
    def action_time(self, value: str):
        """设置操作时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionTime = encoded
        self._cache['action_time'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def exec_order_local_id(self) -> str:
        """本地执行宣告编号"""
        if 'exec_order_local_id' not in self._cache:
            value = self._struct.ExecOrderLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['exec_order_local_id'] = value
        return self._cache['exec_order_local_id']

    @exec_order_local_id.setter
    def exec_order_local_id(self, value: str):
        """设置本地执行宣告编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ExecOrderLocalID = encoded
        self._cache['exec_order_local_id'] = value

    @property
    def action_local_id(self) -> str:
        """操作本地编号"""
        if 'action_local_id' not in self._cache:
            value = self._struct.ActionLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['action_local_id'] = value
        return self._cache['action_local_id']

    @action_local_id.setter
    def action_local_id(self, value: str):
        """设置操作本地编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ActionLocalID = encoded
        self._cache['action_local_id'] = value

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
    def order_action_status(self) -> str:
        """报单操作状态"""
        if 'order_action_status' not in self._cache:
            value = self._struct.OrderActionStatus.decode('ascii')
            self._cache['order_action_status'] = value
        return self._cache['order_action_status']

    @order_action_status.setter
    def order_action_status(self, value: str):
        """设置报单操作状态"""
        self._struct.OrderActionStatus = value.encode('ascii')[0]
        self._cache['order_action_status'] = value

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
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

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



class ForQuote(CapsuleStruct):
    """询价"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ForQuoteRef", ctypes.c_int),              # 询价引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("ForQuoteLocalID", ctypes.c_char * 13),    # 本地询价编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),       # 会员代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("TraderID", ctypes.c_char * 21),            # 交易所交易员代码
            ("InstallID", ctypes.c_int),                 # 安装编号
            ("InsertDate", ctypes.c_char * 9),           # 报单日期
            ("InsertTime", ctypes.c_char * 9),           # 插入时间
            ("ForQuoteStatus", ctypes.c_char),           # 询价状态
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("StatusMsg", ctypes.c_char * 81),           # 状态信息
            ("ActiveUserID", ctypes.c_char * 16),        # 操作用户代码
            ("BrokerForQutoSeq", ctypes.c_int),         # 经纪公司询价编号
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("reserve3", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),           # Mac地址
            ("IPAddress", ctypes.c_char * 33),            # IP地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 51),      # 合约在交易所的代码
        ]

    _capsule_name = "ForQuote"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "reserve1": "reserve1",
        "for_quote_ref": "ForQuoteRef",
        "user_id": "UserID",
        "for_quote_local_id": "ForQuoteLocalID",
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "insert_date": "InsertDate",
        "insert_time": "InsertTime",
        "for_quote_status": "ForQuoteStatus",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "status_msg": "StatusMsg",
        "active_user_id": "ActiveUserID",
        "broker_for_quote_seq": "BrokerForQutoSeq",
        "invest_unit_id": "InvestUnitID",
        "reserve3": "reserve3",
        "mac_address": "MacAddress",
        "ip_address": "IPAddress",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
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
    def for_quote_ref(self) -> int:
        """询价引用"""
        return self._struct.ForQuoteRef

    @for_quote_ref.setter
    def for_quote_ref(self, value: int):
        """设置询价引用"""
        self._struct.ForQuoteRef = value

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
    def for_quote_local_id(self) -> str:
        """本地询价编号"""
        if 'for_quote_local_id' not in self._cache:
            value = self._struct.ForQuoteLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['for_quote_local_id'] = value
        return self._cache['for_quote_local_id']

    @for_quote_local_id.setter
    def for_quote_local_id(self, value: str):
        """设置本地询价编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ForQuoteLocalID = encoded
        self._cache['for_quote_local_id'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def insert_date(self) -> str:
        """报单日期"""
        if 'insert_date' not in self._cache:
            value = self._struct.InsertDate.rstrip(b'\x00').decode('ascii')
            self._cache['insert_date'] = value
        return self._cache['insert_date']

    @insert_date.setter
    def insert_date(self, value: str):
        """设置报单日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertDate = encoded
        self._cache['insert_date'] = value

    @property
    def insert_time(self) -> str:
        """插入时间"""
        if 'insert_time' not in self._cache:
            value = self._struct.InsertTime.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time'] = value
        return self._cache['insert_time']

    @insert_time.setter
    def insert_time(self, value: str):
        """设置插入时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTime = encoded
        self._cache['insert_time'] = value

    @property
    def for_quote_status(self) -> str:
        """询价状态"""
        if 'for_quote_status' not in self._cache:
            value = self._struct.ForQuoteStatus.decode('ascii')
            self._cache['for_quote_status'] = value
        return self._cache['for_quote_status']

    @for_quote_status.setter
    def for_quote_status(self, value: str):
        """设置询价状态"""
        self._struct.ForQuoteStatus = value.encode('ascii')[0]
        self._cache['for_quote_status'] = value

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
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

    @property
    def active_user_id(self) -> str:
        """操作用户代码"""
        if 'active_user_id' not in self._cache:
            value = self._struct.ActiveUserID.rstrip(b'\x00').decode('ascii')
            self._cache['active_user_id'] = value
        return self._cache['active_user_id']

    @active_user_id.setter
    def active_user_id(self, value: str):
        """设置操作用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.ActiveUserID = encoded
        self._cache['active_user_id'] = value

    @property
    def broker_for_quote_seq(self) -> int:
        """经纪公司询价编号"""
        return self._struct.BrokerForQutoSeq

    @broker_for_quote_seq.setter
    def broker_for_quote_seq(self, value: int):
        """设置经纪公司询价编号"""
        self._struct.BrokerForQutoSeq = value

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
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.ExchangeInstID = encoded
        self._cache['exchange_inst_id'] = value



class Quote(CapsuleStruct):
    """报价"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("QuoteRef", ctypes.c_int),                  # 报价引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("AskPrice", ctypes.c_double),                # 卖价格
            ("BidPrice", ctypes.c_double),                # 买价格
            ("AskVolume", ctypes.c_int),                 # 卖数量
            ("BidVolume", ctypes.c_int),                 # 买数量
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("AskOffsetFlag", ctypes.c_char),            # 卖开平标志
            ("BidOffsetFlag", ctypes.c_char),            # 买开平标志
            ("AskHedgeFlag", ctypes.c_char),             # 卖投机套保标志
            ("BidHedgeFlag", ctypes.c_char),             # 买投机套保标志
            ("QuoteLocalID", ctypes.c_char * 13),         # 本地报价编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),       # 会员代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("TraderID", ctypes.c_char * 21),            # 交易所交易员代码
            ("InstallID", ctypes.c_int),                 # 安装编号
            ("NotifySequence", ctypes.c_int),           # 报价提示序号
            ("OrderSubmitStatus", ctypes.c_char),        # 报价提交状态
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("QuoteSysID", ctypes.c_char * 13),          # 报价编号
            ("InsertDate", ctypes.c_char * 9),           # 报单日期
            ("InsertTime", ctypes.c_char * 9),           # 插入时间
            ("CancelTime", ctypes.c_char * 9),           # 撤销时间
            ("QuoteStatus", ctypes.c_char),              # 报价状态
            ("ClearingPartID", ctypes.c_char * 11),      # 结算会员编号
            ("SequenceNo", ctypes.c_int),                # 序号
            ("AskOrderSysID", ctypes.c_char * 13),       # 卖方报单编号
            ("BidOrderSysID", ctypes.c_char * 13),       # 买方报单编号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("UserProductInfo", ctypes.c_char * 11),     # 用户端产品信息
            ("StatusMsg", ctypes.c_char * 81),           # 状态信息
            ("ActiveUserID", ctypes.c_char * 16),        # 操作用户代码
            ("BrokerQuoteSeq", ctypes.c_int),            # 经纪公司报价编号
            ("AskOrderRef", ctypes.c_int),               # 衍生卖报单引用
            ("BidOrderRef", ctypes.c_int),               # 衍生买报单引用
            ("ForQuoteSysID", ctypes.c_char * 13),       # 应价编号
            ("BranchID", ctypes.c_char * 9),             # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("AccountID", ctypes.c_char * 13),           # 资金账号
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
            ("reserve3", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 51),      # 合约在交易所的代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
            ("ReplaceSysID", ctypes.c_char * 13),        # 被顶单编号
            ("TimeCondition", ctypes.c_char),            # 有效期类型
            ("OrderMemo", ctypes.c_char * 81),           # 报单备注
            ("SessionReqSeq", ctypes.c_int),             # session上请求计数
        ]

    _capsule_name = "Quote"

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
        "quote_local_id": "QuoteLocalID",
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "notify_sequence": "NotifySequence",
        "order_submit_status": "OrderSubmitStatus",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "quote_sys_id": "QuoteSysID",
        "insert_date": "InsertDate",
        "insert_time": "InsertTime",
        "cancel_time": "CancelTime",
        "quote_status": "QuoteStatus",
        "reserve3": "reserve3",
        "ask_order_sys_id": "AskOrderSysID",
        "bid_order_sys_id": "BidOrderSysID",
        "clearing_part_id": "ClearingPartID",
        "sequence_no": "SequenceNo",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "user_product_info": "UserProductInfo",
        "status_msg": "StatusMsg",
        "active_user_id": "ActiveUserID",
        "broker_quote_seq": "BrokerQuoteSeq",
        "ask_order_ref": "AskOrderRef",
        "bid_order_ref": "BidOrderRef",
        "for_quote_sys_id": "ForQuoteSysID",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "reserve3_2": "reserve3",  # 注意：有两个reserve3字段
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
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
    def quote_ref(self) -> int:
        """报价引用"""
        return self._struct.QuoteRef

    @quote_ref.setter
    def quote_ref(self, value: int):
        """设置报价引用"""
        self._struct.QuoteRef = value

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
        """卖价格"""
        return self._struct.AskPrice

    @ask_price.setter
    def ask_price(self, value: float):
        """设置卖价格"""
        self._struct.AskPrice = value

    @property
    def bid_price(self) -> float:
        """买价格"""
        return self._struct.BidPrice

    @bid_price.setter
    def bid_price(self, value: float):
        """设置买价格"""
        self._struct.BidPrice = value

    @property
    def ask_volume(self) -> int:
        """卖数量"""
        return self._struct.AskVolume

    @ask_volume.setter
    def ask_volume(self, value: int):
        """设置卖数量"""
        self._struct.AskVolume = value

    @property
    def bid_volume(self) -> int:
        """买数量"""
        return self._struct.BidVolume

    @bid_volume.setter
    def bid_volume(self, value: int):
        """设置买数量"""
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
    def quote_local_id(self) -> str:
        """本地报价编号"""
        if 'quote_local_id' not in self._cache:
            value = self._struct.QuoteLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['quote_local_id'] = value
        return self._cache['quote_local_id']

    @quote_local_id.setter
    def quote_local_id(self, value: str):
        """设置本地报价编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.QuoteLocalID = encoded
        self._cache['quote_local_id'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def notify_sequence(self) -> int:
        """报价提示序号"""
        return self._struct.NotifySequence

    @notify_sequence.setter
    def notify_sequence(self, value: int):
        """设置报价提示序号"""
        self._struct.NotifySequence = value

    @property
    def order_submit_status(self) -> str:
        """报价提交状态"""
        if 'order_submit_status' not in self._cache:
            value = self._struct.OrderSubmitStatus.decode('ascii')
            self._cache['order_submit_status'] = value
        return self._cache['order_submit_status']

    @order_submit_status.setter
    def order_submit_status(self, value: str):
        """设置报价提交状态"""
        self._struct.OrderSubmitStatus = value.encode('ascii')[0]
        self._cache['order_submit_status'] = value

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
    def quote_sys_id(self) -> str:
        """报价编号"""
        if 'quote_sys_id' not in self._cache:
            value = self._struct.QuoteSysID.rstrip(b'\x00').decode('ascii')
            self._cache['quote_sys_id'] = value
        return self._cache['quote_sys_id']

    @quote_sys_id.setter
    def quote_sys_id(self, value: str):
        """设置报价编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.QuoteSysID = encoded
        self._cache['quote_sys_id'] = value

    @property
    def insert_date(self) -> str:
        """报单日期"""
        if 'insert_date' not in self._cache:
            value = self._struct.InsertDate.rstrip(b'\x00').decode('ascii')
            self._cache['insert_date'] = value
        return self._cache['insert_date']

    @insert_date.setter
    def insert_date(self, value: str):
        """设置报单日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertDate = encoded
        self._cache['insert_date'] = value

    @property
    def insert_time(self) -> str:
        """插入时间"""
        if 'insert_time' not in self._cache:
            value = self._struct.InsertTime.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time'] = value
        return self._cache['insert_time']

    @insert_time.setter
    def insert_time(self, value: str):
        """设置插入时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTime = encoded
        self._cache['insert_time'] = value

    @property
    def cancel_time(self) -> str:
        """撤销时间"""
        if 'cancel_time' not in self._cache:
            value = self._struct.CancelTime.rstrip(b'\x00').decode('ascii')
            self._cache['cancel_time'] = value
        return self._cache['cancel_time']

    @cancel_time.setter
    def cancel_time(self, value: str):
        """设置撤销时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.CancelTime = encoded
        self._cache['cancel_time'] = value

    @property
    def quote_status(self) -> str:
        """报价状态"""
        if 'quote_status' not in self._cache:
            value = self._struct.QuoteStatus.decode('ascii')
            self._cache['quote_status'] = value
        return self._cache['quote_status']

    @quote_status.setter
    def quote_status(self, value: str):
        """设置报价状态"""
        self._struct.QuoteStatus = value.encode('ascii')[0]
        self._cache['quote_status'] = value

    @property
    def ask_order_sys_id(self) -> str:
        """卖方报单编号"""
        if 'ask_order_sys_id' not in self._cache:
            value = self._struct.AskOrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['ask_order_sys_id'] = value
        return self._cache['ask_order_sys_id']

    @ask_order_sys_id.setter
    def ask_order_sys_id(self, value: str):
        """设置卖方报单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.AskOrderSysID = encoded
        self._cache['ask_order_sys_id'] = value

    @property
    def bid_order_sys_id(self) -> str:
        """买方报单编号"""
        if 'bid_order_sys_id' not in self._cache:
            value = self._struct.BidOrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['bid_order_sys_id'] = value
        return self._cache['bid_order_sys_id']

    @bid_order_sys_id.setter
    def bid_order_sys_id(self, value: str):
        """设置买方报单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BidOrderSysID = encoded
        self._cache['bid_order_sys_id'] = value

    @property
    def clearing_part_id(self) -> str:
        """结算分支代码"""
        if 'clearing_part_id' not in self._cache:
            value = self._struct.ClearingPartID.rstrip(b'\x00').decode('ascii')
            self._cache['clearing_part_id'] = value
        return self._cache['clearing_part_id']

    @clearing_part_id.setter
    def clearing_part_id(self, value: str):
        """设置结算分支代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClearingPartID = encoded
        self._cache['clearing_part_id'] = value

    @property
    def sequence_no(self) -> int:
        """序号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序号"""
        self._struct.SequenceNo = value

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
    def user_product_info(self) -> str:
        """用户端产品信息"""
        if 'user_product_info' not in self._cache:
            value = self._struct.UserProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['user_product_info'] = value
        return self._cache['user_product_info']

    @user_product_info.setter
    def user_product_info(self, value: str):
        """设置用户端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.UserProductInfo = encoded
        self._cache['user_product_info'] = value

    @property
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

    @property
    def active_user_id(self) -> str:
        """操作用户代码"""
        if 'active_user_id' not in self._cache:
            value = self._struct.ActiveUserID.rstrip(b'\x00').decode('ascii')
            self._cache['active_user_id'] = value
        return self._cache['active_user_id']

    @active_user_id.setter
    def active_user_id(self, value: str):
        """设置操作用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.ActiveUserID = encoded
        self._cache['active_user_id'] = value

    @property
    def broker_quote_seq(self) -> int:
        """经纪公司报价编号"""
        return self._struct.BrokerQuoteSeq

    @broker_quote_seq.setter
    def broker_quote_seq(self, value: int):
        """设置经纪公司报价编号"""
        self._struct.BrokerQuoteSeq = value

    @property
    def ask_order_ref(self) -> int:
        """衍生卖报单引用"""
        return self._struct.AskOrderRef

    @ask_order_ref.setter
    def ask_order_ref(self, value: int):
        """设置衍生卖报单引用"""
        self._struct.AskOrderRef = value

    @property
    def bid_order_ref(self) -> int:
        """衍生买报单引用"""
        return self._struct.BidOrderRef

    @bid_order_ref.setter
    def bid_order_ref(self, value: int):
        """设置衍生买报单引用"""
        self._struct.BidOrderRef = value

    @property
    def for_quote_sys_id(self) -> str:
        """应价编号"""
        if 'for_quote_sys_id' not in self._cache:
            value = self._struct.ForQuoteSysID.rstrip(b'\x00').decode('ascii')
            self._cache['for_quote_sys_id'] = value
        return self._cache['for_quote_sys_id']

    @for_quote_sys_id.setter
    def for_quote_sys_id(self, value: str):
        """设置应价编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ForQuoteSysID = encoded
        self._cache['for_quote_sys_id'] = value

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
    def exchange_inst_id(self) -> str:
        """合约在交易所的代码"""
        if 'exchange_inst_id' not in self._cache:
            value = self._struct.ExchangeInstID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_inst_id'] = value
        return self._cache['exchange_inst_id']

    @exchange_inst_id.setter
    def exchange_inst_id(self, value: str):
        """设置合约在交易所的代码"""
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.ExchangeInstID = encoded
        self._cache['exchange_inst_id'] = value

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
        """被顶单编号"""
        if 'replace_sys_id' not in self._cache:
            value = self._struct.ReplaceSysID.rstrip(b'\x00').decode('ascii')
            self._cache['replace_sys_id'] = value
        return self._cache['replace_sys_id']

    @replace_sys_id.setter
    def replace_sys_id(self, value: str):
        """设置被顶单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ReplaceSysID = encoded
        self._cache['replace_sys_id'] = value

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
    def order_memo(self) -> str:
        """报单备注（GBK 编码）"""
        if 'order_memo' not in self._cache:
            value = self._struct.OrderMemo.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['order_memo'] = value
        return self._cache['order_memo']

    @order_memo.setter
    def order_memo(self, value: str):
        """设置报单备注（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.OrderMemo = encoded
        self._cache['order_memo'] = value

    @property
    def session_req_seq(self) -> int:
        """session上请求计数"""
        return self._struct.SessionReqSeq

    @session_req_seq.setter
    def session_req_seq(self, value: int):
        """设置session上请求计数"""
        self._struct.SessionReqSeq = value



class QuoteAction(CapsuleStruct):
    """报价操作"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("QuoteActionRef", ctypes.c_int),           # 报价操作引用
            ("QuoteRef", ctypes.c_char * 13),           # 报价引用
            ("RequestID", ctypes.c_int),                # 请求编号
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("QuoteSysID", ctypes.c_char * 21),         # 报价操作编号
            ("ActionFlag", ctypes.c_char),              # 操作标志
            ("ActionDate", ctypes.c_char * 9),          # 操作日期
            ("ActionTime", ctypes.c_char * 9),          # 操作时间
            ("TraderID", ctypes.c_char * 21),           # 交易所交易员代码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("QuoteLocalID", ctypes.c_char * 13),       # 本地报价编号
            ("ActionLocalID", ctypes.c_char * 13),      # 操作本地编号
            ("ParticipantID", ctypes.c_char * 11),      # 会员代码
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("BusinessUnit", ctypes.c_char * 21),       # 业务单元
            ("OrderActionStatus", ctypes.c_char),       # 报单操作状态
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("StatusMsg", ctypes.c_char * 81),          # 状态信息
            ("reserve1", ctypes.c_char * 31),           # 保留的无效字段
            ("BranchID", ctypes.c_char * 9),            # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("reserve2", ctypes.c_char * 16),           # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "QuoteAction"

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
        "action_date": "ActionDate",
        "action_time": "ActionTime",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "quote_local_id": "QuoteLocalID",
        "action_local_id": "ActionLocalID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "business_unit": "BusinessUnit",
        "order_action_status": "OrderActionStatus",
        "user_id": "UserID",
        "status_msg": "StatusMsg",
        "reserve1": "reserve1",
        "branch_id": "BranchID",
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
    def quote_action_ref(self) -> int:
        """报价操作引用"""
        return self._struct.QuoteActionRef

    @quote_action_ref.setter
    def quote_action_ref(self, value: int):
        """设置报价操作引用"""
        self._struct.QuoteActionRef = value

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
        """报价操作编号"""
        if 'quote_sys_id' not in self._cache:
            value = self._struct.QuoteSysID.rstrip(b'\x00').decode('ascii')
            self._cache['quote_sys_id'] = value
        return self._cache['quote_sys_id']

    @quote_sys_id.setter
    def quote_sys_id(self, value: str):
        """设置报价操作编号"""
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
    def action_date(self) -> str:
        """操作日期"""
        if 'action_date' not in self._cache:
            value = self._struct.ActionDate.rstrip(b'\x00').decode('ascii')
            self._cache['action_date'] = value
        return self._cache['action_date']

    @action_date.setter
    def action_date(self, value: str):
        """设置操作日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionDate = encoded
        self._cache['action_date'] = value

    @property
    def action_time(self) -> str:
        """操作时间"""
        if 'action_time' not in self._cache:
            value = self._struct.ActionTime.rstrip(b'\x00').decode('ascii')
            self._cache['action_time'] = value
        return self._cache['action_time']

    @action_time.setter
    def action_time(self, value: str):
        """设置操作时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionTime = encoded
        self._cache['action_time'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def quote_local_id(self) -> str:
        """本地报价编号"""
        if 'quote_local_id' not in self._cache:
            value = self._struct.QuoteLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['quote_local_id'] = value
        return self._cache['quote_local_id']

    @quote_local_id.setter
    def quote_local_id(self, value: str):
        """设置本地报价编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.QuoteLocalID = encoded
        self._cache['quote_local_id'] = value

    @property
    def action_local_id(self) -> str:
        """操作本地编号"""
        if 'action_local_id' not in self._cache:
            value = self._struct.ActionLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['action_local_id'] = value
        return self._cache['action_local_id']

    @action_local_id.setter
    def action_local_id(self, value: str):
        """设置操作本地编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ActionLocalID = encoded
        self._cache['action_local_id'] = value

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
    def order_action_status(self) -> str:
        """报单操作状态"""
        if 'order_action_status' not in self._cache:
            value = self._struct.OrderActionStatus.decode('ascii')
            self._cache['order_action_status'] = value
        return self._cache['order_action_status']

    @order_action_status.setter
    def order_action_status(self, value: str):
        """设置报单操作状态"""
        self._struct.OrderActionStatus = value.encode('ascii')[0]
        self._cache['order_action_status'] = value

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
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

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



class BatchOrderAction(CapsuleStruct):
    """批量报单操作"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("OrderActionRef", ctypes.c_int),           # 报单操作引用
            ("RequestID", ctypes.c_int),                # 请求编号
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ActionDate", ctypes.c_char * 9),          # 操作日期
            ("ActionTime", ctypes.c_char * 9),          # 操作时间
            ("TraderID", ctypes.c_char * 21),           # 交易所交易员代码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("ActionLocalID", ctypes.c_char * 13),      # 操作本地编号
            ("ParticipantID", ctypes.c_char * 11),      # 会员代码
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("BusinessUnit", ctypes.c_char * 21),       # 业务单元
            ("OrderActionStatus", ctypes.c_char),       # 报单操作状态
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("StatusMsg", ctypes.c_char * 81),          # 状态信息
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("reserve1", ctypes.c_char * 16),           # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "BatchOrderAction"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "order_action_ref": "OrderActionRef",
        "request_id": "RequestID",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "exchange_id": "ExchangeID",
        "action_date": "ActionDate",
        "action_time": "ActionTime",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "action_local_id": "ActionLocalID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "business_unit": "BusinessUnit",
        "order_action_status": "OrderActionStatus",
        "user_id": "UserID",
        "status_msg": "StatusMsg",
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
    def order_action_ref(self) -> int:
        """报单操作引用"""
        return self._struct.OrderActionRef

    @order_action_ref.setter
    def order_action_ref(self, value: int):
        """设置报单操作引用"""
        self._struct.OrderActionRef = value

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
    def action_date(self) -> str:
        """操作日期"""
        if 'action_date' not in self._cache:
            value = self._struct.ActionDate.rstrip(b'\x00').decode('ascii')
            self._cache['action_date'] = value
        return self._cache['action_date']

    @action_date.setter
    def action_date(self, value: str):
        """设置操作日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionDate = encoded
        self._cache['action_date'] = value

    @property
    def action_time(self) -> str:
        """操作时间"""
        if 'action_time' not in self._cache:
            value = self._struct.ActionTime.rstrip(b'\x00').decode('ascii')
            self._cache['action_time'] = value
        return self._cache['action_time']

    @action_time.setter
    def action_time(self, value: str):
        """设置操作时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionTime = encoded
        self._cache['action_time'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def action_local_id(self) -> str:
        """操作本地编号"""
        if 'action_local_id' not in self._cache:
            value = self._struct.ActionLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['action_local_id'] = value
        return self._cache['action_local_id']

    @action_local_id.setter
    def action_local_id(self, value: str):
        """设置操作本地编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ActionLocalID = encoded
        self._cache['action_local_id'] = value

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
    def order_action_status(self) -> str:
        """报单操作状态"""
        if 'order_action_status' not in self._cache:
            value = self._struct.OrderActionStatus.decode('ascii')
            self._cache['order_action_status'] = value
        return self._cache['order_action_status']

    @order_action_status.setter
    def order_action_status(self, value: str):
        """设置报单操作状态"""
        self._struct.OrderActionStatus = value.encode('ascii')[0]
        self._cache['order_action_status'] = value

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
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

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



class OptionSelfClose(CapsuleStruct):
    """期权自对冲"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("OptionSelfCloseRef", ctypes.c_int),        # 期权自对冲引用
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Volume", ctypes.c_int),                    # 数量
            ("RequestID", ctypes.c_int),                 # 请求编号
            ("BusinessUnit", ctypes.c_char * 21),        # 业务单元
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("OptSelfCloseFlag", ctypes.c_char),         # 期权行权的头寸是否自对冲
            ("OptionSelfCloseLocalID", ctypes.c_char * 13), # 本地期权自对冲编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),       # 会员代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("TraderID", ctypes.c_char * 21),            # 交易所交易员代码
            ("InstallID", ctypes.c_int),                 # 安装编号
            ("OrderSubmitStatus", ctypes.c_char),        # 期权自对冲提交状态
            ("NotifySequence", ctypes.c_int),            # 报单提示序号
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("OptionSelfCloseSysID", ctypes.c_char * 21), # 期权自对冲编号
            ("InsertDate", ctypes.c_char * 9),           # 报单日期
            ("InsertTime", ctypes.c_char * 9),           # 插入时间
            ("CancelTime", ctypes.c_char * 9),           # 撤销时间
            ("ExecResult", ctypes.c_int),                # 自对冲结果
            ("ClearingPartID", ctypes.c_char * 11),      # 结算会员编号
            ("SequenceNo", ctypes.c_int),                # 序号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("UserProductInfo", ctypes.c_char * 11),     # 用户端产品信息
            ("StatusMsg", ctypes.c_char * 81),           # 状态信息
            ("ActiveUserID", ctypes.c_char * 16),        # 操作用户代码
            ("BrokerOptionSelfCloseSeq", ctypes.c_int),  # 经纪公司期权自对流水号
            ("BranchID", ctypes.c_char * 9),             # 分支机构代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
            ("reserve3", ctypes.c_char * 16),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 51),      # 合约在交易所的代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "OptionSelfClose"

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
        "option_self_close_local_id": "OptionSelfCloseLocalID",
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "order_submit_status": "OrderSubmitStatus",
        "notify_sequence": "NotifySequence",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "option_self_close_sys_id": "OptionSelfCloseSysID",
        "insert_date": "InsertDate",
        "insert_time": "InsertTime",
        "cancel_time": "CancelTime",
        "exec_result": "ExecResult",
        "clearing_part_id": "ClearingPartID",
        "sequence_no": "SequenceNo",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "user_product_info": "UserProductInfo",
        "status_msg": "StatusMsg",
        "active_user_id": "ActiveUserID",
        "broker_option_self_close_seq": "BrokerOptionSelfCloseSeq",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "reserve3": "reserve3",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
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
    def option_self_close_ref(self) -> int:
        """期权自对冲引用"""
        return self._struct.OptionSelfCloseRef

    @option_self_close_ref.setter
    def option_self_close_ref(self, value: int):
        """设置期权自对冲引用"""
        self._struct.OptionSelfCloseRef = value

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
        """期权行权的头寸是否自对冲"""
        if 'opt_self_close_flag' not in self._cache:
            value = self._struct.OptSelfCloseFlag.decode('ascii')
            self._cache['opt_self_close_flag'] = value
        return self._cache['opt_self_close_flag']

    @opt_self_close_flag.setter
    def opt_self_close_flag(self, value: str):
        """设置期权行权的头寸是否自对冲"""
        self._struct.OptSelfCloseFlag = value.encode('ascii')[0]
        self._cache['opt_self_close_flag'] = value

    @property
    def option_self_close_local_id(self) -> str:
        """本地期权自对冲编号"""
        if 'option_self_close_local_id' not in self._cache:
            value = self._struct.OptionSelfCloseLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['option_self_close_local_id'] = value
        return self._cache['option_self_close_local_id']

    @option_self_close_local_id.setter
    def option_self_close_local_id(self, value: str):
        """设置本地期权自对冲编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OptionSelfCloseLocalID = encoded
        self._cache['option_self_close_local_id'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def order_submit_status(self) -> str:
        """期权自对冲提交状态"""
        if 'order_submit_status' not in self._cache:
            value = self._struct.OrderSubmitStatus.decode('ascii')
            self._cache['order_submit_status'] = value
        return self._cache['order_submit_status']

    @order_submit_status.setter
    def order_submit_status(self, value: str):
        """设置期权自对冲提交状态"""
        self._struct.OrderSubmitStatus = value.encode('ascii')[0]
        self._cache['order_submit_status'] = value

    @property
    def notify_sequence(self) -> int:
        """报单提示序号"""
        return self._struct.NotifySequence

    @notify_sequence.setter
    def notify_sequence(self, value: int):
        """设置报单提示序号"""
        self._struct.NotifySequence = value

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
    def insert_date(self) -> str:
        """报单日期"""
        if 'insert_date' not in self._cache:
            value = self._struct.InsertDate.rstrip(b'\x00').decode('ascii')
            self._cache['insert_date'] = value
        return self._cache['insert_date']

    @insert_date.setter
    def insert_date(self, value: str):
        """设置报单日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertDate = encoded
        self._cache['insert_date'] = value

    @property
    def insert_time(self) -> str:
        """插入时间"""
        if 'insert_time' not in self._cache:
            value = self._struct.InsertTime.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time'] = value
        return self._cache['insert_time']

    @insert_time.setter
    def insert_time(self, value: str):
        """设置插入时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTime = encoded
        self._cache['insert_time'] = value

    @property
    def cancel_time(self) -> str:
        """撤销时间"""
        if 'cancel_time' not in self._cache:
            value = self._struct.CancelTime.rstrip(b'\x00').decode('ascii')
            self._cache['cancel_time'] = value
        return self._cache['cancel_time']

    @cancel_time.setter
    def cancel_time(self, value: str):
        """设置撤销时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.CancelTime = encoded
        self._cache['cancel_time'] = value

    @property
    def exec_result(self) -> int:
        """自对冲结果"""
        return self._struct.ExecResult

    @exec_result.setter
    def exec_result(self, value: int):
        """设置自对冲结果"""
        self._struct.ExecResult = value

    @property
    def clearing_part_id(self) -> str:
        """结算会员编号"""
        if 'clearing_part_id' not in self._cache:
            value = self._struct.ClearingPartID.rstrip(b'\x00').decode('ascii')
            self._cache['clearing_part_id'] = value
        return self._cache['clearing_part_id']

    @clearing_part_id.setter
    def clearing_part_id(self, value: str):
        """设置结算会员编号"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClearingPartID = encoded
        self._cache['clearing_part_id'] = value

    @property
    def sequence_no(self) -> int:
        """序号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序号"""
        self._struct.SequenceNo = value

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
    def user_product_info(self) -> str:
        """用户端产品信息"""
        if 'user_product_info' not in self._cache:
            value = self._struct.UserProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['user_product_info'] = value
        return self._cache['user_product_info']

    @user_product_info.setter
    def user_product_info(self, value: str):
        """设置用户端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.UserProductInfo = encoded
        self._cache['user_product_info'] = value

    @property
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

    @property
    def active_user_id(self) -> str:
        """操作用户代码"""
        if 'active_user_id' not in self._cache:
            value = self._struct.ActiveUserID.rstrip(b'\x00').decode('ascii')
            self._cache['active_user_id'] = value
        return self._cache['active_user_id']

    @active_user_id.setter
    def active_user_id(self, value: str):
        """设置操作用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.ActiveUserID = encoded
        self._cache['active_user_id'] = value

    @property
    def broker_option_self_close_seq(self) -> int:
        """经纪公司期权自对流水号"""
        return self._struct.BrokerOptionSelfCloseSeq

    @broker_option_self_close_seq.setter
    def broker_option_self_close_seq(self, value: int):
        """设置经纪公司期权自对流水号"""
        self._struct.BrokerOptionSelfCloseSeq = value

    @property
    def branch_id(self) -> str:
        """分支机构代码"""
        if 'branch_id' not in self._cache:
            value = self._struct.BranchID.rstrip(b'\x00').decode('ascii')
            self._cache['branch_id'] = value
        return self._cache['branch_id']

    @branch_id.setter
    def branch_id(self, value: str):
        """设置分支机构代码"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.BranchID = encoded
        self._cache['branch_id'] = value

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
    def exchange_inst_id(self) -> str:
        """合约在交易所的代码"""
        if 'exchange_inst_id' not in self._cache:
            value = self._struct.ExchangeInstID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_inst_id'] = value
        return self._cache['exchange_inst_id']

    @exchange_inst_id.setter
    def exchange_inst_id(self, value: str):
        """设置合约在交易所的代码"""
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.ExchangeInstID = encoded
        self._cache['exchange_inst_id'] = value

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



class OptionSelfCloseAction(CapsuleStruct):
    """期权自对冲操作"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("OptionSelfCloseActionRef", ctypes.c_int), # 期权自对冲操作引用
            ("OptionSelfCloseRef", ctypes.c_char * 13), # 期权自对冲引用
            ("RequestID", ctypes.c_int),                # 请求编号
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("OptionSelfCloseSysID", ctypes.c_char * 21), # 期权自对冲操作编号
            ("ActionFlag", ctypes.c_char),              # 操作标志
            ("ActionDate", ctypes.c_char * 9),          # 操作日期
            ("ActionTime", ctypes.c_char * 9),          # 操作时间
            ("TraderID", ctypes.c_char * 21),           # 交易所交易员代码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("OptionSelfCloseLocalID", ctypes.c_char * 13), # 本地期权自对冲编号
            ("ActionLocalID", ctypes.c_char * 13),      # 操作本地编号
            ("ParticipantID", ctypes.c_char * 11),      # 会员代码
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("BusinessUnit", ctypes.c_char * 21),       # 业务单元
            ("OrderActionStatus", ctypes.c_char),       # 报单操作状态
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("StatusMsg", ctypes.c_char * 81),          # 状态信息
            ("reserve1", ctypes.c_char * 31),           # 保留的无效字段
            ("BranchID", ctypes.c_char * 9),            # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("reserve2", ctypes.c_char * 16),           # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "OptionSelfCloseAction"

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
        "action_date": "ActionDate",
        "action_time": "ActionTime",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "option_self_close_local_id": "OptionSelfCloseLocalID",
        "action_local_id": "ActionLocalID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "business_unit": "BusinessUnit",
        "order_action_status": "OrderActionStatus",
        "user_id": "UserID",
        "status_msg": "StatusMsg",
        "reserve1": "reserve1",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
        "reserve2": "reserve2",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "ip_address": "IPAddress",
    }

    # Properties and setters for OptionSelfCloseAction
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
    def option_self_close_action_ref(self) -> int:
        """期权自对冲操作引用"""
        return self._struct.OptionSelfCloseActionRef

    @option_self_close_action_ref.setter
    def option_self_close_action_ref(self, value: int):
        """设置期权自对冲操作引用"""
        self._struct.OptionSelfCloseActionRef = value

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
    def action_date(self) -> str:
        """操作日期"""
        if 'action_date' not in self._cache:
            value = self._struct.ActionDate.rstrip(b'\x00').decode('ascii')
            self._cache['action_date'] = value
        return self._cache['action_date']

    @action_date.setter
    def action_date(self, value: str):
        """设置操作日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionDate = encoded
        self._cache['action_date'] = value

    @property
    def action_time(self) -> str:
        """操作时间"""
        if 'action_time' not in self._cache:
            value = self._struct.ActionTime.rstrip(b'\x00').decode('ascii')
            self._cache['action_time'] = value
        return self._cache['action_time']

    @action_time.setter
    def action_time(self, value: str):
        """设置操作时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionTime = encoded
        self._cache['action_time'] = value

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

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def option_self_close_local_id(self) -> str:
        """本地期权自对冲编号"""
        if 'option_self_close_local_id' not in self._cache:
            value = self._struct.OptionSelfCloseLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['option_self_close_local_id'] = value
        return self._cache['option_self_close_local_id']

    @option_self_close_local_id.setter
    def option_self_close_local_id(self, value: str):
        """设置本地期权自对冲编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OptionSelfCloseLocalID = encoded
        self._cache['option_self_close_local_id'] = value

    @property
    def action_local_id(self) -> str:
        """操作本地编号"""
        if 'action_local_id' not in self._cache:
            value = self._struct.ActionLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['action_local_id'] = value
        return self._cache['action_local_id']

    @action_local_id.setter
    def action_local_id(self, value: str):
        """设置操作本地编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ActionLocalID = encoded
        self._cache['action_local_id'] = value

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
    def order_action_status(self) -> str:
        """报单操作状态"""
        if 'order_action_status' not in self._cache:
            value = self._struct.OrderActionStatus.decode('ascii')
            self._cache['order_action_status'] = value
        return self._cache['order_action_status']

    @order_action_status.setter
    def order_action_status(self, value: str):
        """设置报单操作状态"""
        self._struct.OrderActionStatus = value.encode('ascii')[0]
        self._cache['order_action_status'] = value

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
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

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



class ParkedOrder(CapsuleStruct):
    """预埋单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("reserve1", ctypes.c_char * 31),           # 保留字段
            ("OrderRef", ctypes.c_char * 13),           # 报单引用
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("OrderPriceType", ctypes.c_char),          # 报单价格条件
            ("Direction", ctypes.c_char),               # 买卖方向
            ("CombOffsetFlag", ctypes.c_char * 5),      # 组合开平标志
            ("CombHedgeFlag", ctypes.c_char * 5),       # 组合投机套保标志
            ("LimitPrice", ctypes.c_double),            # 限价
            ("VolumeTotalOriginal", ctypes.c_int),      # 数量
            ("TimeCondition", ctypes.c_char),           # 有效期类型
            ("GTDDate", ctypes.c_char * 9),             # GTD日期
            ("VolumeCondition", ctypes.c_char),         # 成交量类型
            ("MinVolume", ctypes.c_int),                # 最小成交量
            ("ContingentCondition", ctypes.c_char),     # 触发条件
            ("StopPrice", ctypes.c_double),             # 止损价
            ("ForceCloseReason", ctypes.c_char),        # 强平原因
            ("IsAutoSuspend", ctypes.c_int),            # 自动挂起标志
            ("BusinessUnit", ctypes.c_char * 21),       # 业务单元
            ("RequestID", ctypes.c_int),                # 请求编号
            ("UserForceClose", ctypes.c_int),           # 用户强平标志
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ParkedOrderID", ctypes.c_char * 13),      # 预埋单编号
            ("UserType", ctypes.c_char),                # 用户类型
            ("Status", ctypes.c_char),                  # 状态
            ("ErrorID", ctypes.c_int),                  # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),           # 错误信息
            ("IsSwapOrder", ctypes.c_int),              # 互换标志
            ("AccountID", ctypes.c_char * 13),          # 资金账号
            ("CurrencyID", ctypes.c_char * 4),          # 币种
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("reserve2", ctypes.c_char * 16),           # 保留字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "ParkedOrder"

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
        "exchange_id": "ExchangeID",
        "parked_order_id": "ParkedOrderID",
        "user_type": "UserType",
        "status": "Status",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
        "is_swap_order": "IsSwapOrder",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "client_id": "ClientID",
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
        """保留字段"""
        if 'reserve1' not in self._cache:
            value = self._struct.reserve1.rstrip(b'\x00').decode('ascii')
            self._cache['reserve1'] = value
        return self._cache['reserve1']

    @reserve1.setter
    def reserve1(self, value: str):
        """设置保留字段"""
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
        """限价"""
        return self._struct.LimitPrice

    @limit_price.setter
    def limit_price(self, value: float):
        """设置限价"""
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
    def parked_order_id(self) -> str:
        """预埋单编号"""
        if 'parked_order_id' not in self._cache:
            value = self._struct.ParkedOrderID.rstrip(b'\x00').decode('ascii')
            self._cache['parked_order_id'] = value
        return self._cache['parked_order_id']

    @parked_order_id.setter
    def parked_order_id(self, value: str):
        """设置预埋单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ParkedOrderID = encoded
        self._cache['parked_order_id'] = value

    @property
    def user_type(self) -> str:
        """用户类型"""
        if 'user_type' not in self._cache:
            value = self._struct.UserType.decode('ascii')
            self._cache['user_type'] = value
        return self._cache['user_type']

    @user_type.setter
    def user_type(self, value: str):
        """设置用户类型"""
        self._struct.UserType = value.encode('ascii')[0]
        self._cache['user_type'] = value

    @property
    def status(self) -> str:
        """状态"""
        if 'status' not in self._cache:
            value = self._struct.Status.decode('ascii')
            self._cache['status'] = value
        return self._cache['status']

    @status.setter
    def status(self, value: str):
        """设置状态"""
        self._struct.Status = value.encode('ascii')[0]
        self._cache['status'] = value

    @property
    def error_id(self) -> int:
        """错误代码"""
        return self._struct.ErrorID

    @error_id.setter
    def error_id(self, value: int):
        """设置错误代码"""
        self._struct.ErrorID = value

    @property
    def error_msg(self) -> str:
        """错误信息（GBK 编码）"""
        if 'error_msg' not in self._cache:
            value = self._struct.ErrorMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['error_msg'] = value
        return self._cache['error_msg']

    @error_msg.setter
    def error_msg(self, value: str):
        """设置错误信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.ErrorMsg = encoded
        self._cache['error_msg'] = value

    @property
    def is_swap_order(self) -> int:
        """互换标志"""
        return self._struct.IsSwapOrder

    @is_swap_order.setter
    def is_swap_order(self, value: int):
        """设置互换标志"""
        self._struct.IsSwapOrder = value

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
        """保留字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留字段"""
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



class ParkedOrderAction(CapsuleStruct):
    """预埋撤单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("OrderActionRef", ctypes.c_int),           # 报单操作引用
            ("OrderRef", ctypes.c_char * 13),           # 报单引用
            ("RequestID", ctypes.c_int),                # 请求编号
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("OrderSysID", ctypes.c_char * 21),         # 报单编号
            ("ActionFlag", ctypes.c_char),              # 操作标志
            ("LimitPrice", ctypes.c_double),            # 限价
            ("VolumeChange", ctypes.c_int),             # 数量变化
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("reserve1", ctypes.c_char * 31),           # 保留字段
            ("ParkedOrderActionID", ctypes.c_char * 13),# 预埋撤单编号
            ("UserType", ctypes.c_char),                # 用户类型
            ("Status", ctypes.c_char),                  # 状态
            ("ErrorID", ctypes.c_int),                  # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),           # 错误信息
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("reserve2", ctypes.c_char * 16),           # 保留字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "ParkedOrderAction"

    _field_mappings = {
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
        "parked_order_action_id": "ParkedOrderActionID",
        "user_type": "UserType",
        "status": "Status",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
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
    def order_action_ref(self) -> int:
        """报单操作引用"""
        return self._struct.OrderActionRef

    @order_action_ref.setter
    def order_action_ref(self, value: int):
        """设置报单操作引用"""
        self._struct.OrderActionRef = value

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
        """限价"""
        return self._struct.LimitPrice

    @limit_price.setter
    def limit_price(self, value: float):
        """设置限价"""
        self._struct.LimitPrice = value

    @property
    def volume_change(self) -> int:
        """数量变化"""
        return self._struct.VolumeChange

    @volume_change.setter
    def volume_change(self, value: int):
        """设置数量变化"""
        self._struct.VolumeChange = value

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
        """保留字段"""
        if 'reserve1' not in self._cache:
            value = self._struct.reserve1.rstrip(b'\x00').decode('ascii')
            self._cache['reserve1'] = value
        return self._cache['reserve1']

    @reserve1.setter
    def reserve1(self, value: str):
        """设置保留字段"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.reserve1 = encoded
        self._cache['reserve1'] = value

    @property
    def parked_order_action_id(self) -> str:
        """预埋撤单编号"""
        if 'parked_order_action_id' not in self._cache:
            value = self._struct.ParkedOrderActionID.rstrip(b'\x00').decode('ascii')
            self._cache['parked_order_action_id'] = value
        return self._cache['parked_order_action_id']

    @parked_order_action_id.setter
    def parked_order_action_id(self, value: str):
        """设置预埋撤单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ParkedOrderActionID = encoded
        self._cache['parked_order_action_id'] = value

    @property
    def user_type(self) -> str:
        """用户类型"""
        if 'user_type' not in self._cache:
            value = self._struct.UserType.decode('ascii')
            self._cache['user_type'] = value
        return self._cache['user_type']

    @user_type.setter
    def user_type(self, value: str):
        """设置用户类型"""
        self._struct.UserType = value.encode('ascii')[0]
        self._cache['user_type'] = value

    @property
    def status(self) -> str:
        """状态"""
        if 'status' not in self._cache:
            value = self._struct.Status.decode('ascii')
            self._cache['status'] = value
        return self._cache['status']

    @status.setter
    def status(self, value: str):
        """设置状态"""
        self._struct.Status = value.encode('ascii')[0]
        self._cache['status'] = value

    @property
    def error_id(self) -> int:
        """错误代码"""
        return self._struct.ErrorID

    @error_id.setter
    def error_id(self, value: int):
        """设置错误代码"""
        self._struct.ErrorID = value

    @property
    def error_msg(self) -> str:
        """错误信息（GBK 编码）"""
        if 'error_msg' not in self._cache:
            value = self._struct.ErrorMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['error_msg'] = value
        return self._cache['error_msg']

    @error_msg.setter
    def error_msg(self, value: str):
        """设置错误信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.ErrorMsg = encoded
        self._cache['error_msg'] = value

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
        """保留字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留字段"""
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



class RemoveParkedOrder(CapsuleStruct):
    """删除预埋单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("ParkedOrderID", ctypes.c_char * 13),       # 预埋报单编号
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
        ]

    _capsule_name = "RemoveParkedOrder"

    _field_mappings: Dict[str, str] = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "parked_order_id": "ParkedOrderID",
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
    def parked_order_id(self) -> str:
        """预埋报单编号"""
        if 'parked_order_id' not in self._cache:
            value = self._struct.ParkedOrderID.rstrip(b'\x00').decode('ascii')
            self._cache['parked_order_id'] = value
        return self._cache['parked_order_id']

    @parked_order_id.setter
    def parked_order_id(self, value: str):
        """设置预埋报单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ParkedOrderID = encoded
        self._cache['parked_order_id'] = value

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



class RemoveParkedOrderAction(CapsuleStruct):
    """删除预埋撤单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("ParkedOrderActionID", ctypes.c_char * 13), # 预埋撤单编号
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
        ]

    _capsule_name = "RemoveParkedOrderAction"

    _field_mappings: Dict[str, str] = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "parked_order_action_id": "ParkedOrderActionID",
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
    def parked_order_action_id(self) -> str:
        """预埋撤单编号"""
        if 'parked_order_action_id' not in self._cache:
            value = self._struct.ParkedOrderActionID.rstrip(b'\x00').decode('ascii')
            self._cache['parked_order_action_id'] = value
        return self._cache['parked_order_action_id']

    @parked_order_action_id.setter
    def parked_order_action_id(self, value: str):
        """设置预埋撤单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ParkedOrderActionID = encoded
        self._cache['parked_order_action_id'] = value

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



class ErrorConditionalOrder(CapsuleStruct):
    """错误条件单"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("reserve1", ctypes.c_char * 31),           # 保留字段
            ("OrderRef", ctypes.c_char * 13),           # 报单引用
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("OrderPriceType", ctypes.c_char),          # 报单价格条件
            ("Direction", ctypes.c_char),               # 买卖方向
            ("CombOffsetFlag", ctypes.c_char * 5),      # 组合开平标志
            ("CombHedgeFlag", ctypes.c_char * 5),       # 组合投机套保标志
            ("LimitPrice", ctypes.c_double),            # 限价
            ("VolumeTotalOriginal", ctypes.c_int),      # 数量
            ("TimeCondition", ctypes.c_char),           # 有效期类型
            ("GTDDate", ctypes.c_char * 9),             # GTD日期
            ("VolumeCondition", ctypes.c_char),         # 成交量类型
            ("MinVolume", ctypes.c_int),                # 最小成交量
            ("ContingentCondition", ctypes.c_char),     # 触发条件
            ("StopPrice", ctypes.c_double),             # 止损价
            ("ForceCloseReason", ctypes.c_char),        # 强平原因
            ("IsAutoSuspend", ctypes.c_int),            # 自动挂起标志
            ("BusinessUnit", ctypes.c_char * 21),       # 业务单元
            ("RequestID", ctypes.c_int),                # 请求编号
            ("OrderLocalID", ctypes.c_char * 13),       # 本地报单编号
            ("ExchangeID", ctypes.c_char * 9),          # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),      # 会员代码
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("reserve2", ctypes.c_char * 31),           # 保留字段
            ("TraderID", ctypes.c_char * 21),           # 交易员代码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("OrderSubmitStatus", ctypes.c_char),       # 提交状态
            ("NotifySequence", ctypes.c_int),           # 通知序号
            ("TradingDay", ctypes.c_char * 9),          # 交易日
            ("SettlementID", ctypes.c_int),             # 结算编号
            ("OrderSysID", ctypes.c_char * 21),         # 报单编号
            ("OrderSource", ctypes.c_char),             # 报单来源
            ("OrderStatus", ctypes.c_char),             # 报单状态
            ("OrderType", ctypes.c_char),               # 报单类型
            ("VolumeTraded", ctypes.c_int),             # 成交数量
            ("VolumeTotal", ctypes.c_int),              # 剩余数量
            ("InsertDate", ctypes.c_char * 9),          # 插入日期
            ("InsertTime", ctypes.c_char * 9),          # 插入时间
            ("ActiveTime", ctypes.c_char * 9),          # 激活时间
            ("SuspendTime", ctypes.c_char * 9),         # 暂停时间
            ("UpdateTime", ctypes.c_char * 9),          # 修改时间
            ("CancelTime", ctypes.c_char * 9),          # 撤销时间
            ("ActiveTraderID", ctypes.c_char * 21),     # 最后修改交易员
            ("ClearingPartID", ctypes.c_char * 11),     # 结算会员代码
            ("SequenceNo", ctypes.c_int),               # 序号
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("UserProductInfo", ctypes.c_char * 11),    # 用户产品信息
            ("StatusMsg", ctypes.c_char * 81),          # 状态信息
            ("UserForceClose", ctypes.c_int),           # 用户强平标志
            ("ActiveUserID", ctypes.c_char * 16),       # 操作用户代码
            ("BrokerOrderSeq", ctypes.c_int),           # 经纪公司报单序号
            ("RelativeOrderSysID", ctypes.c_char * 21), # 相关报单编号
            ("ZCETotalTradedVolume", ctypes.c_int),     # 郑商所成交数量
            ("ErrorID", ctypes.c_int),                  # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),           # 错误信息
            ("IsSwapOrder", ctypes.c_int),              # 互换标志
            ("BranchID", ctypes.c_char * 9),            # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
            ("AccountID", ctypes.c_char * 13),          # 资金账号
            ("CurrencyID", ctypes.c_char * 4),          # 币种
            ("reserve3", ctypes.c_char * 16),           # 保留字段
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("InstrumentID", ctypes.c_char * 81),       # 合约代码
            ("ExchangeInstID", ctypes.c_char * 81),     # 合约在交易所的代码
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "ErrorConditionalOrder"

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
        "order_local_id": "OrderLocalID",
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "order_submit_status": "OrderSubmitStatus",
        "notify_sequence": "NotifySequence",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "order_sys_id": "OrderSysID",
        "order_source": "OrderSource",
        "order_status": "OrderStatus",
        "order_type": "OrderType",
        "volume_traded": "VolumeTraded",
        "volume_total": "VolumeTotal",
        "insert_date": "InsertDate",
        "insert_time": "InsertTime",
        "active_time": "ActiveTime",
        "suspend_time": "SuspendTime",
        "update_time": "UpdateTime",
        "cancel_time": "CancelTime",
        "active_trader_id": "ActiveTraderID",
        "clearing_part_id": "ClearingPartID",
        "sequence_no": "SequenceNo",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "user_product_info": "UserProductInfo",
        "status_msg": "StatusMsg",
        "user_force_close": "UserForceClose",
        "active_user_id": "ActiveUserID",
        "broker_order_seq": "BrokerOrderSeq",
        "relative_order_sys_id": "RelativeOrderSysID",
        "zce_total_traded_volume": "ZCETotalTradedVolume",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
        "is_swap_order": "IsSwapOrder",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
        "reserve3": "reserve3",
        "mac_address": "MacAddress",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
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
        """保留字段"""
        if 'reserve1' not in self._cache:
            value = self._struct.reserve1.rstrip(b'\x00').decode('ascii')
            self._cache['reserve1'] = value
        return self._cache['reserve1']

    @reserve1.setter
    def reserve1(self, value: str):
        """设置保留字段"""
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
        """限价"""
        return self._struct.LimitPrice

    @limit_price.setter
    def limit_price(self, value: float):
        """设置限价"""
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
    def order_local_id(self) -> str:
        """本地报单编号"""
        if 'order_local_id' not in self._cache:
            value = self._struct.OrderLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['order_local_id'] = value
        return self._cache['order_local_id']

    @order_local_id.setter
    def order_local_id(self, value: str):
        """设置本地报单编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.OrderLocalID = encoded
        self._cache['order_local_id'] = value

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
    def reserve2(self) -> str:
        """保留字段"""
        if 'reserve2' not in self._cache:
            value = self._struct.reserve2.rstrip(b'\x00').decode('ascii')
            self._cache['reserve2'] = value
        return self._cache['reserve2']

    @reserve2.setter
    def reserve2(self, value: str):
        """设置保留字段"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.reserve2 = encoded
        self._cache['reserve2'] = value

    @property
    def trader_id(self) -> str:
        """交易员代码"""
        if 'trader_id' not in self._cache:
            value = self._struct.TraderID.rstrip(b'\x00').decode('ascii')
            self._cache['trader_id'] = value
        return self._cache['trader_id']

    @trader_id.setter
    def trader_id(self, value: str):
        """设置交易员代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.TraderID = encoded
        self._cache['trader_id'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def order_submit_status(self) -> str:
        """提交状态"""
        if 'order_submit_status' not in self._cache:
            value = self._struct.OrderSubmitStatus.decode('ascii')
            self._cache['order_submit_status'] = value
        return self._cache['order_submit_status']

    @order_submit_status.setter
    def order_submit_status(self, value: str):
        """设置提交状态"""
        self._struct.OrderSubmitStatus = value.encode('ascii')[0]
        self._cache['order_submit_status'] = value

    @property
    def notify_sequence(self) -> int:
        """通知序号"""
        return self._struct.NotifySequence

    @notify_sequence.setter
    def notify_sequence(self, value: int):
        """设置通知序号"""
        self._struct.NotifySequence = value

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
    def order_source(self) -> str:
        """报单来源"""
        if 'order_source' not in self._cache:
            value = self._struct.OrderSource.decode('ascii')
            self._cache['order_source'] = value
        return self._cache['order_source']

    @order_source.setter
    def order_source(self, value: str):
        """设置报单来源"""
        self._struct.OrderSource = value.encode('ascii')[0]
        self._cache['order_source'] = value

    @property
    def order_status(self) -> str:
        """报单状态"""
        if 'order_status' not in self._cache:
            value = self._struct.OrderStatus.decode('ascii')
            self._cache['order_status'] = value
        return self._cache['order_status']

    @order_status.setter
    def order_status(self, value: str):
        """设置报单状态"""
        self._struct.OrderStatus = value.encode('ascii')[0]
        self._cache['order_status'] = value

    @property
    def order_type(self) -> str:
        """报单类型"""
        if 'order_type' not in self._cache:
            value = self._struct.OrderType.decode('ascii')
            self._cache['order_type'] = value
        return self._cache['order_type']

    @order_type.setter
    def order_type(self, value: str):
        """设置报单类型"""
        self._struct.OrderType = value.encode('ascii')[0]
        self._cache['order_type'] = value

    @property
    def volume_traded(self) -> int:
        """成交数量"""
        return self._struct.VolumeTraded

    @volume_traded.setter
    def volume_traded(self, value: int):
        """设置成交数量"""
        self._struct.VolumeTraded = value

    @property
    def volume_total(self) -> int:
        """剩余数量"""
        return self._struct.VolumeTotal

    @volume_total.setter
    def volume_total(self, value: int):
        """设置剩余数量"""
        self._struct.VolumeTotal = value

    @property
    def insert_date(self) -> str:
        """插入日期"""
        if 'insert_date' not in self._cache:
            value = self._struct.InsertDate.rstrip(b'\x00').decode('ascii')
            self._cache['insert_date'] = value
        return self._cache['insert_date']

    @insert_date.setter
    def insert_date(self, value: str):
        """设置插入日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertDate = encoded
        self._cache['insert_date'] = value

    @property
    def insert_time(self) -> str:
        """插入时间"""
        if 'insert_time' not in self._cache:
            value = self._struct.InsertTime.rstrip(b'\x00').decode('ascii')
            self._cache['insert_time'] = value
        return self._cache['insert_time']

    @insert_time.setter
    def insert_time(self, value: str):
        """设置插入时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.InsertTime = encoded
        self._cache['insert_time'] = value

    @property
    def active_time(self) -> str:
        """激活时间"""
        if 'active_time' not in self._cache:
            value = self._struct.ActiveTime.rstrip(b'\x00').decode('ascii')
            self._cache['active_time'] = value
        return self._cache['active_time']

    @active_time.setter
    def active_time(self, value: str):
        """设置激活时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActiveTime = encoded
        self._cache['active_time'] = value

    @property
    def suspend_time(self) -> str:
        """暂停时间"""
        if 'suspend_time' not in self._cache:
            value = self._struct.SuspendTime.rstrip(b'\x00').decode('ascii')
            self._cache['suspend_time'] = value
        return self._cache['suspend_time']

    @suspend_time.setter
    def suspend_time(self, value: str):
        """设置暂停时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.SuspendTime = encoded
        self._cache['suspend_time'] = value

    @property
    def update_time(self) -> str:
        """修改时间"""
        if 'update_time' not in self._cache:
            value = self._struct.UpdateTime.rstrip(b'\x00').decode('ascii')
            self._cache['update_time'] = value
        return self._cache['update_time']

    @update_time.setter
    def update_time(self, value: str):
        """设置修改时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.UpdateTime = encoded
        self._cache['update_time'] = value

    @property
    def cancel_time(self) -> str:
        """撤销时间"""
        if 'cancel_time' not in self._cache:
            value = self._struct.CancelTime.rstrip(b'\x00').decode('ascii')
            self._cache['cancel_time'] = value
        return self._cache['cancel_time']

    @cancel_time.setter
    def cancel_time(self, value: str):
        """设置撤销时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.CancelTime = encoded
        self._cache['cancel_time'] = value

    @property
    def active_trader_id(self) -> str:
        """最后修改交易员"""
        if 'active_trader_id' not in self._cache:
            value = self._struct.ActiveTraderID.rstrip(b'\x00').decode('ascii')
            self._cache['active_trader_id'] = value
        return self._cache['active_trader_id']

    @active_trader_id.setter
    def active_trader_id(self, value: str):
        """设置最后修改交易员"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ActiveTraderID = encoded
        self._cache['active_trader_id'] = value

    @property
    def clearing_part_id(self) -> str:
        """结算会员代码"""
        if 'clearing_part_id' not in self._cache:
            value = self._struct.ClearingPartID.rstrip(b'\x00').decode('ascii')
            self._cache['clearing_part_id'] = value
        return self._cache['clearing_part_id']

    @clearing_part_id.setter
    def clearing_part_id(self, value: str):
        """设置结算会员代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ClearingPartID = encoded
        self._cache['clearing_part_id'] = value

    @property
    def sequence_no(self) -> int:
        """序号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序号"""
        self._struct.SequenceNo = value

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
    def user_product_info(self) -> str:
        """用户产品信息"""
        if 'user_product_info' not in self._cache:
            value = self._struct.UserProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['user_product_info'] = value
        return self._cache['user_product_info']

    @user_product_info.setter
    def user_product_info(self, value: str):
        """设置用户产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.UserProductInfo = encoded
        self._cache['user_product_info'] = value

    @property
    def status_msg(self) -> str:
        """状态信息（GBK 编码）"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value

    @property
    def user_force_close(self) -> int:
        """用户强平标志"""
        return self._struct.UserForceClose

    @user_force_close.setter
    def user_force_close(self, value: int):
        """设置用户强平标志"""
        self._struct.UserForceClose = value

    @property
    def active_user_id(self) -> str:
        """操作用户代码"""
        if 'active_user_id' not in self._cache:
            value = self._struct.ActiveUserID.rstrip(b'\x00').decode('ascii')
            self._cache['active_user_id'] = value
        return self._cache['active_user_id']

    @active_user_id.setter
    def active_user_id(self, value: str):
        """设置操作用户代码"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.ActiveUserID = encoded
        self._cache['active_user_id'] = value

    @property
    def broker_order_seq(self) -> int:
        """经纪公司报单序号"""
        return self._struct.BrokerOrderSeq

    @broker_order_seq.setter
    def broker_order_seq(self, value: int):
        """设置经纪公司报单序号"""
        self._struct.BrokerOrderSeq = value

    @property
    def relative_order_sys_id(self) -> str:
        """相关报单编号"""
        if 'relative_order_sys_id' not in self._cache:
            value = self._struct.RelativeOrderSysID.rstrip(b'\x00').decode('ascii')
            self._cache['relative_order_sys_id'] = value
        return self._cache['relative_order_sys_id']

    @relative_order_sys_id.setter
    def relative_order_sys_id(self, value: str):
        """设置相关报单编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.RelativeOrderSysID = encoded
        self._cache['relative_order_sys_id'] = value

    @property
    def zce_total_traded_volume(self) -> int:
        """郑商所成交数量"""
        return self._struct.ZCETotalTradedVolume

    @zce_total_traded_volume.setter
    def zce_total_traded_volume(self, value: int):
        """设置郑商所成交数量"""
        self._struct.ZCETotalTradedVolume = value

    @property
    def error_id(self) -> int:
        """错误代码"""
        return self._struct.ErrorID

    @error_id.setter
    def error_id(self, value: int):
        """设置错误代码"""
        self._struct.ErrorID = value

    @property
    def error_msg(self) -> str:
        """错误信息（GBK 编码）"""
        if 'error_msg' not in self._cache:
            value = self._struct.ErrorMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['error_msg'] = value
        return self._cache['error_msg']

    @error_msg.setter
    def error_msg(self, value: str):
        """设置错误信息（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.ErrorMsg = encoded
        self._cache['error_msg'] = value

    @property
    def is_swap_order(self) -> int:
        """互换标志"""
        return self._struct.IsSwapOrder

    @is_swap_order.setter
    def is_swap_order(self, value: int):
        """设置互换标志"""
        self._struct.IsSwapOrder = value

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

    @property
    def reserve3(self) -> str:
        """保留字段"""
        if 'reserve3' not in self._cache:
            value = self._struct.reserve3.rstrip(b'\x00').decode('ascii')
            self._cache['reserve3'] = value
        return self._cache['reserve3']

    @reserve3.setter
    def reserve3(self, value: str):
        """设置保留字段"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.reserve3 = encoded
        self._cache['reserve3'] = value

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



