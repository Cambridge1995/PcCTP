"""
Rsp
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class RspInfo(CapsuleStruct):
    """响应信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ErrorID", ctypes.c_int),                # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),         # 错误信息
        ]

    _capsule_name = "RspInfo"

    _field_mappings = {
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
    }

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


# =============================================================================
# RspUserLogin - 用户登录响应
# =============================================================================


class RspUserLogin(CapsuleStruct):
    """用户登录响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("LoginTime", ctypes.c_char * 9),            # 登录成功时间
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("SystemName", ctypes.c_char * 21),          # 系统名称
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("MaxOrderRef", ctypes.c_char * 13),          # 最大报单引用
            ("SHFETime", ctypes.c_char * 9),             # 上期所时间
            ("DCETime", ctypes.c_char * 9),              # 郑商所时间
            ("CZCETime", ctypes.c_char * 9),             # 大商所时间
            ("FFEXTime", ctypes.c_char * 9),             # 能源所时间
            ("INETTime", ctypes.c_char * 9),             # 英特所时间
            ("Field1", ctypes.c_char * 201),             # 保留字段1
            ("Field2", ctypes.c_char * 41),              # 保留字段2
            ("Field3", ctypes.c_char * 81),              # 保留字段3
        ]

    _capsule_name = "RspUserLogin"

    _field_mappings = {
        "trading_day": "TradingDay",
        "login_time": "LoginTime",
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "system_name": "SystemName",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "max_order_ref": "MaxOrderRef",
        "shfe_time": "SHFETime",
        "dce_time": "DCETime",
        "czce_time": "CZCETime",
        "ffex_time": "FFEXTime",
        "inet_time": "INETTime",
        "field1": "Field1",
        "field2": "Field2",
        "field3": "Field3",
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
    def login_time(self) -> str:
        """登录成功时间"""
        if 'login_time' not in self._cache:
            value = self._struct.LoginTime.rstrip(b'\x00').decode('ascii')
            self._cache['login_time'] = value
        return self._cache['login_time']

    @login_time.setter
    def login_time(self, value: str):
        """设置登录成功时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.LoginTime = encoded
        self._cache['login_time'] = value

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
    def system_name(self) -> str:
        """系统名称"""
        if 'system_name' not in self._cache:
            value = self._struct.SystemName.rstrip(b'\x00').decode('ascii')
            self._cache['system_name'] = value
        return self._cache['system_name']

    @system_name.setter
    def system_name(self, value: str):
        """设置系统名称"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.SystemName = encoded
        self._cache['system_name'] = value

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
    def max_order_ref(self) -> str:
        """最大报单引用"""
        if 'max_order_ref' not in self._cache:
            value = self._struct.MaxOrderRef.rstrip(b'\x00').decode('ascii')
            self._cache['max_order_ref'] = value
        return self._cache['max_order_ref']

    @max_order_ref.setter
    def max_order_ref(self, value: str):
        """设置最大报单引用"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.MaxOrderRef = encoded
        self._cache['max_order_ref'] = value


# =============================================================================
# UserLogout - 用户登出
# =============================================================================


class ForQuoteRsp(CapsuleStruct):
    """询价响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("ForQuoteRef", ctypes.c_char * 13),         # 询价引用
            ("ActionDay", ctypes.c_char * 9),            # 业务日期
            ("ActionTime", ctypes.c_char * 9),            # 行动时间
        ]

    _capsule_name = "ForQuoteRsp"

    _field_mappings = {
        "trading_day": "TradingDay",
        "instrument_id": "InstrumentID",
        "broker_id": "BrokerID",
        "for_quote_ref": "ForQuoteRef",
        "action_day": "ActionDay",
        "action_time": "ActionTime",
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
    def action_day(self) -> str:
        """业务日期"""
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
    def action_time(self) -> str:
        """行动时间"""
        if 'action_time' not in self._cache:
            value = self._struct.ActionTime.rstrip(b'\x00').decode('ascii')
            self._cache['action_time'] = value
        return self._cache['action_time']

    @action_time.setter
    def action_time(self, value: str):
        """设置行动时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ActionTime = encoded
        self._cache['action_time'] = value


# =============================================================================
# FensUserInfo - 银期转账用户信息
# =============================================================================


class FensUserInfo(CapsuleStruct):
    """银期转账用户信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("LoginMode", ctypes.c_char),                # 登录模式
        ]

    _capsule_name = "FensUserInfo"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "login_mode": "LoginMode",
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
    def login_mode(self) -> str:
        """登录模式"""
        if 'login_mode' not in self._cache:
            value = self._struct.LoginMode.decode('ascii')
            self._cache['login_mode'] = value
        return self._cache['login_mode']

    @login_mode.setter
    def login_mode(self, value: str):
        """设置登录模式"""
        self._struct.LoginMode = value.encode('ascii')[0]
        self._cache['login_mode'] = value


# =============================================================================
# MulticastInstrument - 组播行情合约
# =============================================================================


