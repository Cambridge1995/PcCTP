"""
Rsp
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class RspAuthenticate(CapsuleStruct):
    """客户端认证响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("UserProductInfo", ctypes.c_char * 11),     # 用户端产品信息
            ("AppID", ctypes.c_char * 33),               # App代码
            ("AppType", ctypes.c_char),                  # App类型
        ]

    _capsule_name = "RspAuthenticate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "user_product_info": "UserProductInfo",
        "app_id": "AppID",
        "app_type": "AppType",
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

    @property
    def app_type(self) -> str:
        """App类型"""
        if 'app_type' not in self._cache:
            value = self._struct.AppType.decode('ascii')
            self._cache['app_type'] = value
        return self._cache['app_type']

    @app_type.setter
    def app_type(self, value: str):
        """设置App类型"""
        self._struct.AppType = value.encode('ascii')[0]
        self._cache['app_type'] = value


# =============================================================================
# UserPasswordUpdate - 用户密码更新
# =============================================================================


class RspGenUserCaptcha(CapsuleStruct):
    """生成验证码响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("CaptchaInfoLen", ctypes.c_int),            # 图片信息长度
            ("CaptchaInfo", ctypes.c_char * 2561),        # 图片信息
        ]

    _capsule_name = "RspGenUserCaptcha"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "captcha_info_len": "CaptchaInfoLen",
        "captcha_info": "CaptchaInfo",
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
    def captcha_info_len(self) -> int:
        """图片信息长度"""
        return self._struct.CaptchaInfoLen

    @captcha_info_len.setter
    def captcha_info_len(self, value: int):
        """设置图片信息长度"""
        self._struct.CaptchaInfoLen = value

    @property
    def captcha_info(self) -> str:
        """图片信息"""
        if 'captcha_info' not in self._cache:
            value = self._struct.CaptchaInfo[:self.captcha_info_len].decode('ascii', errors='ignore')
            self._cache['captcha_info'] = value
        return self._cache['captcha_info']

    @captcha_info.setter
    def captcha_info(self, value: str):
        """设置图片信息"""
        encoded = value.encode('ascii')[:2560].ljust(2561, b'\x00')
        self._struct.CaptchaInfo = encoded
        self._struct.CaptchaInfoLen = len(value)
        self._cache['captcha_info'] = value


# =============================================================================
# InputOrder - 报单录入
# =============================================================================


class RspGenUserText(CapsuleStruct):
    """用户短信验证码响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("UserTextSeq", ctypes.c_int),  # 短信验证码序号
        ]

    _capsule_name = "RspGenUserText"

    _field_mappings = {
        "user_text_seq": "UserTextSeq",
    }

    @property
    def user_text_seq(self) -> int:
        """短信验证码序号"""
        return self._struct.UserTextSeq

    @user_text_seq.setter
    def user_text_seq(self, value: int):
        """设置短信验证码序号"""
        self._struct.UserTextSeq = value



class RspUserAuthMethod(CapsuleStruct):
    """用户认证模式响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("UsableAuthMethod", ctypes.c_int),  # 当前可以用的认证模式
        ]

    _capsule_name = "RspUserAuthMethod"

    _field_mappings = {
        "usable_auth_method": "UsableAuthMethod",
    }

    @property
    def usable_auth_method(self) -> int:
        """当前可以用的认证模式"""
        return self._struct.UsableAuthMethod

    @usable_auth_method.setter
    def usable_auth_method(self, value: int):
        """设置当前可以用的认证模式"""
        self._struct.UsableAuthMethod = value





