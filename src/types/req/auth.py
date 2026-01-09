"""
Req
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class ReqUserLogin(CapsuleStruct):
    """用户登录请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Password", ctypes.c_char * 41),            # 密码
            ("UserProductInfo", ctypes.c_char * 11),     # 用户端产品信息
            ("InterfaceProductInfo", ctypes.c_char * 11), # 接口端产品信息
            ("ProtocolInfo", ctypes.c_char * 11),        # 协议信息
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("OneTimePassword", ctypes.c_char * 41),     # 动态密码
            ("ClientIPAddress", ctypes.c_char * 33),     # 终端IP地址
            ("Remark", ctypes.c_char * 36),              # 备注
        ]

    _capsule_name = "ReqUserLogin"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "password": "Password",
        "user_product_info": "UserProductInfo",
        "interface_product_info": "InterfaceProductInfo",
        "protocol_info": "ProtocolInfo",
        "mac_address": "MacAddress",
        "one_time_password": "OneTimePassword",
        "client_ip_address": "ClientIPAddress",
        "remark": "Remark",
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
    def password(self) -> str:
        """密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

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
    def interface_product_info(self) -> str:
        """接口端产品信息"""
        if 'interface_product_info' not in self._cache:
            value = self._struct.InterfaceProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['interface_product_info'] = value
        return self._cache['interface_product_info']

    @interface_product_info.setter
    def interface_product_info(self, value: str):
        """设置接口端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.InterfaceProductInfo = encoded
        self._cache['interface_product_info'] = value

    @property
    def protocol_info(self) -> str:
        """协议信息"""
        if 'protocol_info' not in self._cache:
            value = self._struct.ProtocolInfo.rstrip(b'\x00').decode('ascii')
            self._cache['protocol_info'] = value
        return self._cache['protocol_info']

    @protocol_info.setter
    def protocol_info(self, value: str):
        """设置协议信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ProtocolInfo = encoded
        self._cache['protocol_info'] = value

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
    def one_time_password(self) -> str:
        """动态密码"""
        if 'one_time_password' not in self._cache:
            value = self._struct.OneTimePassword.rstrip(b'\x00').decode('ascii')
            self._cache['one_time_password'] = value
        return self._cache['one_time_password']

    @one_time_password.setter
    def one_time_password(self, value: str):
        """设置动态密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.OneTimePassword = encoded
        self._cache['one_time_password'] = value

    @property
    def client_ip_address(self) -> str:
        """终端IP地址"""
        if 'client_ip_address' not in self._cache:
            value = self._struct.ClientIPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['client_ip_address'] = value
        return self._cache['client_ip_address']

    @client_ip_address.setter
    def client_ip_address(self, value: str):
        """设置终端IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientIPAddress = encoded
        self._cache['client_ip_address'] = value

    @property
    def remark(self) -> str:
        """备注"""
        if 'remark' not in self._cache:
            value = self._struct.Remark.rstrip(b'\x00').decode('ascii')
            self._cache['remark'] = value
        return self._cache['remark']

    @remark.setter
    def remark(self, value: str):
        """设置备注"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.Remark = encoded
        self._cache['remark'] = value


# =============================================================================
# ForQuoteRsp - 询价响应
# =============================================================================


class ReqAuthenticate(CapsuleStruct):
    """客户端认证请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),         # 经纪公司代码
            ("UserID", ctypes.c_char * 16),           # 用户代码
            ("UserProductInfo", ctypes.c_char * 11),   # 用户端产品信息
            ("AuthCode", ctypes.c_char * 17),          # 认证码
            ("AppID", ctypes.c_char * 33),             # App代码
        ]

    _capsule_name = "ReqAuthenticate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "user_product_info": "UserProductInfo",
        "auth_code": "AuthCode",
        "app_id": "AppID",
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
    def auth_code(self) -> str:
        """认证码"""
        if 'auth_code' not in self._cache:
            value = self._struct.AuthCode.rstrip(b'\x00').decode('ascii')
            self._cache['auth_code'] = value
        return self._cache['auth_code']

    @auth_code.setter
    def auth_code(self, value: str):
        """设置认证码"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.AuthCode = encoded
        self._cache['auth_code'] = value

    @property
    def app_id(self) -> str:
        """App代码"""
        if 'app_id' not in self._cache:
            value = self._struct.AppID.rstrip(b'\x00').decode('ascii')
            self._cache['app_id'] = value
        return self._cache['app_id']

    @app_id.setter
    def app_id(self, value: str):
        """设置App代码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.AppID = encoded
        self._cache['app_id'] = value



class UserPasswordUpdate(CapsuleStruct):
    """用户密码更新"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("OldPassword", ctypes.c_char * 41),         # 原来的口令
            ("NewPassword", ctypes.c_char * 41),         # 新的口令
        ]

    _capsule_name = "UserPasswordUpdate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "old_password": "OldPassword",
        "new_password": "NewPassword",
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
    def old_password(self) -> str:
        """原来的口令"""
        if 'old_password' not in self._cache:
            value = self._struct.OldPassword.rstrip(b'\x00').decode('ascii')
            self._cache['old_password'] = value
        return self._cache['old_password']

    @old_password.setter
    def old_password(self, value: str):
        """设置原来的口令"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.OldPassword = encoded
        self._cache['old_password'] = value

    @property
    def new_password(self) -> str:
        """新的口令"""
        if 'new_password' not in self._cache:
            value = self._struct.NewPassword.rstrip(b'\x00').decode('ascii')
            self._cache['new_password'] = value
        return self._cache['new_password']

    @new_password.setter
    def new_password(self, value: str):
        """设置新的口令"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.NewPassword = encoded
        self._cache['new_password'] = value


# =============================================================================
# TradingAccountPasswordUpdate - 资金账户密码更新
# =============================================================================


class TradingAccountPasswordUpdate(CapsuleStruct):
    """资金账户密码更新"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("OldPassword", ctypes.c_char * 41),         # 原来的口令
            ("NewPassword", ctypes.c_char * 41),         # 新的口令
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
        ]

    _capsule_name = "TradingAccountPasswordUpdate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "account_id": "AccountID",
        "old_password": "OldPassword",
        "new_password": "NewPassword",
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
    def old_password(self) -> str:
        """原来的口令"""
        if 'old_password' not in self._cache:
            value = self._struct.OldPassword.rstrip(b'\x00').decode('ascii')
            self._cache['old_password'] = value
        return self._cache['old_password']

    @old_password.setter
    def old_password(self, value: str):
        """设置原来的口令"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.OldPassword = encoded
        self._cache['old_password'] = value

    @property
    def new_password(self) -> str:
        """新的口令"""
        if 'new_password' not in self._cache:
            value = self._struct.NewPassword.rstrip(b'\x00').decode('ascii')
            self._cache['new_password'] = value
        return self._cache['new_password']

    @new_password.setter
    def new_password(self, value: str):
        """设置新的口令"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.NewPassword = encoded
        self._cache['new_password'] = value

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


# =============================================================================
# RspGenUserCaptcha - 生成验证码响应
# =============================================================================


class ReqGenUserCaptcha(CapsuleStruct):
    """用户发出获取安全安全登陆方法请求（生成验证码）"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),      # 交易日
            ("BrokerID", ctypes.c_char * 11),       # 经纪公司代码
            ("UserID", ctypes.c_char * 16),         # 用户代码
        ]

    _capsule_name = "ReqGenUserCaptcha"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "user_id": "UserID",
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



class ReqGenUserText(CapsuleStruct):
    """用户发出获取安全安全登陆方法请求（生成短信验证码）"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),      # 交易日
            ("BrokerID", ctypes.c_char * 11),       # 经纪公司代码
            ("UserID", ctypes.c_char * 16),         # 用户代码
        ]

    _capsule_name = "ReqGenUserText"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "user_id": "UserID",
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



class ReqUserAuthMethod(CapsuleStruct):
    """用户发出获取安全安全登陆方法请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),      # 交易日
            ("BrokerID", ctypes.c_char * 11),       # 经纪公司代码
            ("UserID", ctypes.c_char * 16),         # 用户代码
        ]

    _capsule_name = "ReqUserAuthMethod"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "user_id": "UserID",
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





class ReqUserLoginWithCaptcha(CapsuleStruct):
    """用户发出带图形验证码的登录请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Password", ctypes.c_char * 41),            # 密码
            ("UserProductInfo", ctypes.c_char * 11),      # 用户端产品信息
            ("InterfaceProductInfo", ctypes.c_char * 11), # 接口端产品信息
            ("ProtocolInfo", ctypes.c_char * 11),        # 协议信息
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("reserve1", ctypes.c_char * 16),            # 保留的无效字段
            ("LoginRemark", ctypes.c_char * 36),         # 登录备注
            ("Captcha", ctypes.c_char * 41),             # 图形验证码的文字内容
            ("ClientIPPort", ctypes.c_int),              # 终端IP端口
            ("ClientIPAddress", ctypes.c_char * 33),     # 终端IP地址
        ]

    _capsule_name = "ReqUserLoginWithCaptcha"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "password": "Password",
        "user_product_info": "UserProductInfo",
        "interface_product_info": "InterfaceProductInfo",
        "protocol_info": "ProtocolInfo",
        "mac_address": "MacAddress",
        "reserve1": "reserve1",
        "login_remark": "LoginRemark",
        "captcha": "Captcha",
        "client_ip_port": "ClientIPPort",
        "client_ip_address": "ClientIPAddress",
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
    def password(self) -> str:
        """密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

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
    def interface_product_info(self) -> str:
        """接口端产品信息"""
        if 'interface_product_info' not in self._cache:
            value = self._struct.InterfaceProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['interface_product_info'] = value
        return self._cache['interface_product_info']

    @interface_product_info.setter
    def interface_product_info(self, value: str):
        """设置接口端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.InterfaceProductInfo = encoded
        self._cache['interface_product_info'] = value

    @property
    def protocol_info(self) -> str:
        """协议信息"""
        if 'protocol_info' not in self._cache:
            value = self._struct.ProtocolInfo.rstrip(b'\x00').decode('ascii')
            self._cache['protocol_info'] = value
        return self._cache['protocol_info']

    @protocol_info.setter
    def protocol_info(self, value: str):
        """设置协议信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ProtocolInfo = encoded
        self._cache['protocol_info'] = value

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
    def login_remark(self) -> str:
        """登录备注"""
        if 'login_remark' not in self._cache:
            value = self._struct.LoginRemark.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['login_remark'] = value
        return self._cache['login_remark']

    @login_remark.setter
    def login_remark(self, value: str):
        """设置登录备注（GBK 编码）"""
        encoded = value.encode('gbk')[:35].ljust(36, b'\x00')
        self._struct.LoginRemark = encoded
        self._cache['login_remark'] = value

    @property
    def captcha(self) -> str:
        """图形验证码的文字内容"""
        if 'captcha' not in self._cache:
            value = self._struct.Captcha.rstrip(b'\x00').decode('ascii')
            self._cache['captcha'] = value
        return self._cache['captcha']

    @captcha.setter
    def captcha(self, value: str):
        """设置图形验证码的文字内容"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Captcha = encoded
        self._cache['captcha'] = value

    @property
    def client_ip_port(self) -> int:
        """终端IP端口"""
        return self._struct.ClientIPPort

    @client_ip_port.setter
    def client_ip_port(self, value: int):
        """设置终端IP端口"""
        self._struct.ClientIPPort = value

    @property
    def client_ip_address(self) -> str:
        """终端IP地址"""
        if 'client_ip_address' not in self._cache:
            value = self._struct.ClientIPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['client_ip_address'] = value
        return self._cache['client_ip_address']

    @client_ip_address.setter
    def client_ip_address(self, value: str):
        """设置终端IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientIPAddress = encoded
        self._cache['client_ip_address'] = value



class ReqUserLoginWithText(CapsuleStruct):
    """用户发出带短信验证码的登录请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Password", ctypes.c_char * 41),            # 密码
            ("UserProductInfo", ctypes.c_char * 11),      # 用户端产品信息
            ("InterfaceProductInfo", ctypes.c_char * 11), # 接口端产品信息
            ("ProtocolInfo", ctypes.c_char * 11),        # 协议信息
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("reserve1", ctypes.c_char * 16),            # 保留的无效字段
            ("LoginRemark", ctypes.c_char * 36),         # 登录备注
            ("Text", ctypes.c_char * 41),                # 短信验证码文字内容
            ("ClientIPPort", ctypes.c_int),              # 终端IP端口
            ("ClientIPAddress", ctypes.c_char * 33),     # 终端IP地址
        ]

    _capsule_name = "ReqUserLoginWithText"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "password": "Password",
        "user_product_info": "UserProductInfo",
        "interface_product_info": "InterfaceProductInfo",
        "protocol_info": "ProtocolInfo",
        "mac_address": "MacAddress",
        "reserve1": "reserve1",
        "login_remark": "LoginRemark",
        "text": "Text",
        "client_ip_port": "ClientIPPort",
        "client_ip_address": "ClientIPAddress",
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
    def password(self) -> str:
        """密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

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
    def interface_product_info(self) -> str:
        """接口端产品信息"""
        if 'interface_product_info' not in self._cache:
            value = self._struct.InterfaceProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['interface_product_info'] = value
        return self._cache['interface_product_info']

    @interface_product_info.setter
    def interface_product_info(self, value: str):
        """设置接口端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.InterfaceProductInfo = encoded
        self._cache['interface_product_info'] = value

    @property
    def protocol_info(self) -> str:
        """协议信息"""
        if 'protocol_info' not in self._cache:
            value = self._struct.ProtocolInfo.rstrip(b'\x00').decode('ascii')
            self._cache['protocol_info'] = value
        return self._cache['protocol_info']

    @protocol_info.setter
    def protocol_info(self, value: str):
        """设置协议信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ProtocolInfo = encoded
        self._cache['protocol_info'] = value

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
    def login_remark(self) -> str:
        """登录备注"""
        if 'login_remark' not in self._cache:
            value = self._struct.LoginRemark.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['login_remark'] = value
        return self._cache['login_remark']

    @login_remark.setter
    def login_remark(self, value: str):
        """设置登录备注（GBK 编码）"""
        encoded = value.encode('gbk')[:35].ljust(36, b'\x00')
        self._struct.LoginRemark = encoded
        self._cache['login_remark'] = value

    @property
    def text(self) -> str:
        """短信验证码文字内容"""
        if 'text' not in self._cache:
            value = self._struct.Text.rstrip(b'\x00').decode('ascii')
            self._cache['text'] = value
        return self._cache['text']

    @text.setter
    def text(self, value: str):
        """设置短信验证码文字内容"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Text = encoded
        self._cache['text'] = value

    @property
    def client_ip_port(self) -> int:
        """终端IP端口"""
        return self._struct.ClientIPPort

    @client_ip_port.setter
    def client_ip_port(self, value: int):
        """设置终端IP端口"""
        self._struct.ClientIPPort = value

    @property
    def client_ip_address(self) -> str:
        """终端IP地址"""
        if 'client_ip_address' not in self._cache:
            value = self._struct.ClientIPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['client_ip_address'] = value
        return self._cache['client_ip_address']

    @client_ip_address.setter
    def client_ip_address(self, value: str):
        """设置终端IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientIPAddress = encoded
        self._cache['client_ip_address'] = value



class ReqUserLoginWithOTP(CapsuleStruct):
    """用户发出带动态验证码的登录请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("Password", ctypes.c_char * 41),            # 密码
            ("UserProductInfo", ctypes.c_char * 11),      # 用户端产品信息
            ("InterfaceProductInfo", ctypes.c_char * 11), # 接口端产品信息
            ("ProtocolInfo", ctypes.c_char * 11),        # 协议信息
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("reserve1", ctypes.c_char * 16),            # 保留的无效字段
            ("LoginRemark", ctypes.c_char * 36),         # 登录备注
            ("OTPPassword", ctypes.c_char * 41),         # OTP密码
            ("ClientIPPort", ctypes.c_int),              # 终端IP端口
            ("ClientIPAddress", ctypes.c_char * 33),     # 终端IP地址
        ]

    _capsule_name = "ReqUserLoginWithOTP"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "password": "Password",
        "user_product_info": "UserProductInfo",
        "interface_product_info": "InterfaceProductInfo",
        "protocol_info": "ProtocolInfo",
        "mac_address": "MacAddress",
        "reserve1": "reserve1",
        "login_remark": "LoginRemark",
        "otp_password": "OTPPassword",
        "client_ip_port": "ClientIPPort",
        "client_ip_address": "ClientIPAddress",
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
    def password(self) -> str:
        """密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

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
    def interface_product_info(self) -> str:
        """接口端产品信息"""
        if 'interface_product_info' not in self._cache:
            value = self._struct.InterfaceProductInfo.rstrip(b'\x00').decode('ascii')
            self._cache['interface_product_info'] = value
        return self._cache['interface_product_info']

    @interface_product_info.setter
    def interface_product_info(self, value: str):
        """设置接口端产品信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.InterfaceProductInfo = encoded
        self._cache['interface_product_info'] = value

    @property
    def protocol_info(self) -> str:
        """协议信息"""
        if 'protocol_info' not in self._cache:
            value = self._struct.ProtocolInfo.rstrip(b'\x00').decode('ascii')
            self._cache['protocol_info'] = value
        return self._cache['protocol_info']

    @protocol_info.setter
    def protocol_info(self, value: str):
        """设置协议信息"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ProtocolInfo = encoded
        self._cache['protocol_info'] = value

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
    def login_remark(self) -> str:
        """登录备注"""
        if 'login_remark' not in self._cache:
            value = self._struct.LoginRemark.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['login_remark'] = value
        return self._cache['login_remark']

    @login_remark.setter
    def login_remark(self, value: str):
        """设置登录备注（GBK 编码）"""
        encoded = value.encode('gbk')[:35].ljust(36, b'\x00')
        self._struct.LoginRemark = encoded
        self._cache['login_remark'] = value

    @property
    def otp_password(self) -> str:
        """OTP密码"""
        if 'otp_password' not in self._cache:
            value = self._struct.OTPPassword.rstrip(b'\x00').decode('ascii')
            self._cache['otp_password'] = value
        return self._cache['otp_password']

    @otp_password.setter
    def otp_password(self, value: str):
        """设置OTP密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.OTPPassword = encoded
        self._cache['otp_password'] = value

    @property
    def client_ip_port(self) -> int:
        """终端IP端口"""
        return self._struct.ClientIPPort

    @client_ip_port.setter
    def client_ip_port(self, value: int):
        """设置终端IP端口"""
        self._struct.ClientIPPort = value

    @property
    def client_ip_address(self) -> str:
        """终端IP地址"""
        if 'client_ip_address' not in self._cache:
            value = self._struct.ClientIPAddress.rstrip(b'\x00').decode('ascii')
            self._cache['client_ip_address'] = value
        return self._cache['client_ip_address']

    @client_ip_address.setter
    def client_ip_address(self, value: str):
        """设置终端IP地址"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientIPAddress = encoded
        self._cache['client_ip_address'] = value




class UserLogout(CapsuleStruct):
    """用户登出"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
        ]

    _capsule_name = "UserLogout"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
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


# =============================================================================
# ReqUserLogin - 用户登录请求
# =============================================================================


