"""
Req
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class UserSession(CapsuleStruct):
    """用户会话"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("UserID", ctypes.c_char * 16),             # 用户代码
            ("LoginDate", ctypes.c_char * 9),           # 登录日期
            ("LoginTime", ctypes.c_char * 9),           # 登录时间
            ("reserve1", ctypes.c_char * 16),           # 保留的无效字段
            ("UserProductInfo", ctypes.c_char * 11),    # 用户端产品信息
            ("InterfaceProductInfo", ctypes.c_char * 11), # 接口端产品信息
            ("ProtocolInfo", ctypes.c_char * 11),       # 协议信息
            ("MacAddress", ctypes.c_char * 21),         # Mac地址
            ("LoginRemark", ctypes.c_char * 36),        # 登录备注
            ("IPAddress", ctypes.c_char * 33),          # IP地址
        ]

    _capsule_name = "UserSession"

    _field_mappings = {
        "front_id": "FrontID",
        "session_id": "SessionID",
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "login_date": "LoginDate",
        "login_time": "LoginTime",
        "user_product_info": "UserProductInfo",
        "interface_product_info": "InterfaceProductInfo",
        "protocol_info": "ProtocolInfo",
        "mac_address": "MacAddress",
        "login_remark": "LoginRemark",
        "ip_address": "IPAddress",
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

    @property
    def login_date(self) -> str:
        """登录日期"""
        if 'login_date' not in self._cache:
            value = self._struct.LoginDate.rstrip(b'\x00').decode('ascii')
            self._cache['login_date'] = value
        return self._cache['login_date']

    @login_date.setter
    def login_date(self, value: str):
        """设置登录日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.LoginDate = encoded
        self._cache['login_date'] = value

    @property
    def login_time(self) -> str:
        """登录时间"""
        if 'login_time' not in self._cache:
            value = self._struct.LoginTime.rstrip(b'\x00').decode('ascii')
            self._cache['login_time'] = value
        return self._cache['login_time']

    @login_time.setter
    def login_time(self, value: str):
        """设置登录时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.LoginTime = encoded
        self._cache['login_time'] = value

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
            value = self._struct.LoginRemark.rstrip(b'\x00').decode('ascii')
            self._cache['login_remark'] = value
        return self._cache['login_remark']

    @login_remark.setter
    def login_remark(self, value: str):
        """设置登录备注"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.LoginRemark = encoded
        self._cache['login_remark'] = value

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
# Exchange - 交易所
# =============================================================================


class SettlementInfo(CapsuleStruct):
    """投资者结算信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),         # 交易日
            ("SettlementID", ctypes.c_int),             # 结算编号
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("SequenceNo", ctypes.c_int),               # 序号
            ("Content", ctypes.c_char * 501),           # 消息正文
            ("AccountID", ctypes.c_char * 13),          # 投资者帐号
            ("CurrencyID", ctypes.c_char * 4),          # 币种代码
        ]

    _capsule_name = "SettlementInfo"

    _field_mappings = {
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "sequence_no": "SequenceNo",
        "content": "Content",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
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
    def sequence_no(self) -> int:
        """序号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序号"""
        self._struct.SequenceNo = value

    @property
    def content(self) -> str:
        """消息正文（GBK 编码）"""
        if 'content' not in self._cache:
            value = self._struct.Content.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['content'] = value
        return self._cache['content']

    @content.setter
    def content(self, value: str):
        """设置消息正文（GBK 编码）"""
        encoded = value.encode('gbk')[:500].ljust(501, b'\x00')
        self._struct.Content = encoded
        self._cache['content'] = value

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



class TransferBank(CapsuleStruct):
    """转帐银行"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BankID", ctypes.c_char * 4),               # 银行代码
            ("BankBrchID", ctypes.c_char * 5),           # 银行分中心代码
            ("BankName", ctypes.c_char * 101),           # 银行名称
            ("IsActive", ctypes.c_int),                  # 是否活跃
        ]

    _capsule_name = "TransferBank"

    _field_mappings = {
        "bank_id": "BankID",
        "bank_brch_id": "BankBrchID",
        "bank_name": "BankName",
        "is_active": "IsActive",
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

    @property
    def bank_name(self) -> str:
        """银行名称（GBK 编码）"""
        if 'bank_name' not in self._cache:
            value = self._struct.BankName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['bank_name'] = value
        return self._cache['bank_name']

    @bank_name.setter
    def bank_name(self, value: str):
        """设置银行名称（GBK 编码）"""
        encoded = value.encode('gbk')[:100].ljust(101, b'\x00')
        self._struct.BankName = encoded
        self._cache['bank_name'] = value

    @property
    def is_active(self) -> int:
        """是否活跃"""
        return self._struct.IsActive

    @is_active.setter
    def is_active(self, value: int):
        """设置是否活跃"""
        self._struct.IsActive = value



class Notice(CapsuleStruct):
    """经纪公司通知"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("Content", ctypes.c_char * 501),            # 消息正文
            ("SequenceLabel", ctypes.c_int),             # 经纪公司通知内容序列号
        ]

    _capsule_name = "Notice"

    _field_mappings = {
        "broker_id": "BrokerID",
        "content": "Content",
        "sequence_label": "SequenceLabel",
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
    def content(self) -> str:
        """消息正文（GBK 编码）"""
        if 'content' not in self._cache:
            value = self._struct.Content.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['content'] = value
        return self._cache['content']

    @content.setter
    def content(self, value: str):
        """设置消息正文（GBK 编码）"""
        encoded = value.encode('gbk')[:500].ljust(501, b'\x00')
        self._struct.Content = encoded
        self._cache['content'] = value

    @property
    def sequence_label(self) -> int:
        """经纪公司通知内容序列号"""
        return self._struct.SequenceLabel

    @sequence_label.setter
    def sequence_label(self, value: int):
        """设置经纪公司通知内容序列号"""
        self._struct.SequenceLabel = value



class SettlementInfoConfirm(CapsuleStruct):
    """投资者结算信息确认"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("ConfirmDate", ctypes.c_char * 9),          # 确认日期
            ("ConfirmTime", ctypes.c_char * 9),          # 确认时间
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("CurrencyID", ctypes.c_char * 4),           # 币种代码
        ]

    _capsule_name = "SettlementInfoConfirm"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "confirm_date": "ConfirmDate",
        "confirm_time": "ConfirmTime",
        "settlement_id": "SettlementID",
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
    def confirm_date(self) -> str:
        """确认日期"""
        if 'confirm_date' not in self._cache:
            value = self._struct.ConfirmDate.rstrip(b'\x00').decode('ascii')
            self._cache['confirm_date'] = value
        return self._cache['confirm_date']

    @confirm_date.setter
    def confirm_date(self, value: str):
        """设置确认日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ConfirmDate = encoded
        self._cache['confirm_date'] = value

    @property
    def confirm_time(self) -> str:
        """确认时间"""
        if 'confirm_time' not in self._cache:
            value = self._struct.ConfirmTime.rstrip(b'\x00').decode('ascii')
            self._cache['confirm_time'] = value
        return self._cache['confirm_time']

    @confirm_time.setter
    def confirm_time(self, value: str):
        """设置确认时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ConfirmTime = encoded
        self._cache['confirm_time'] = value

    @property
    def settlement_id(self) -> int:
        """结算编号"""
        return self._struct.SettlementID

    @settlement_id.setter
    def settlement_id(self, value: int):
        """设置结算编号"""
        self._struct.SettlementID = value

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



class CFMMCTradingAccountKey(CapsuleStruct):
    """保证金监管系统经纪公司资金账户密钥"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("ParticipantID", ctypes.c_char * 11),       # 经纪公司统一编码
            ("AccountID", ctypes.c_char * 13),           # 投资者帐号
            ("KeyID", ctypes.c_int),                     # 密钥编号
            ("CurrentKey", ctypes.c_char * 17),          # 动态密钥
        ]

    _capsule_name = "CFMMCTradingAccountKey"

    _field_mappings = {
        "broker_id": "BrokerID",
        "participant_id": "ParticipantID",
        "account_id": "AccountID",
        "key_id": "KeyID",
        "current_key": "CurrentKey",
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
    def participant_id(self) -> str:
        """经纪公司统一编码"""
        if 'participant_id' not in self._cache:
            value = self._struct.ParticipantID.rstrip(b'\x00').decode('ascii')
            self._cache['participant_id'] = value
        return self._cache['participant_id']

    @participant_id.setter
    def participant_id(self, value: str):
        """设置经纪公司统一编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ParticipantID = encoded
        self._cache['participant_id'] = value

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
    def key_id(self) -> int:
        """密钥编号"""
        return self._struct.KeyID

    @key_id.setter
    def key_id(self, value: int):
        """设置密钥编号"""
        self._struct.KeyID = value

    @property
    def current_key(self) -> str:
        """动态密钥"""
        if 'current_key' not in self._cache:
            value = self._struct.CurrentKey.rstrip(b'\x00').decode('ascii')
            self._cache['current_key'] = value
        return self._cache['current_key']

    @current_key.setter
    def current_key(self, value: str):
        """设置动态密钥"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.CurrentKey = encoded
        self._cache['current_key'] = value



class EWarrantOffset(CapsuleStruct):
    """仓单折抵信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),           # 交易日期
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("Direction", ctypes.c_char),                # 买卖方向
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("Volume", ctypes.c_int),                    # 数量
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "EWarrantOffset"

    _field_mappings = {
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "exchange_id": "ExchangeID",
        "reserve1": "reserve1",
        "direction": "Direction",
        "hedge_flag": "HedgeFlag",
        "volume": "Volume",
        "invest_unit_id": "InvestUnitID",
        "instrument_id": "InstrumentID",
    }

    @property
    def trading_day(self) -> str:
        """交易日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易日期"""
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
    def volume(self) -> int:
        """数量"""
        return self._struct.Volume

    @volume.setter
    def volume(self, value: int):
        """设置数量"""
        self._struct.Volume = value

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



class InvestorProductGroupMargin(CapsuleStruct):
    """投资者产品组保证金"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("FrozenMargin", ctypes.c_double),           # 冻结的保证金
            ("LongFrozenMargin", ctypes.c_double),       # 多头冻结的保证金
            ("ShortFrozenMargin", ctypes.c_double),      # 空头冻结的保证金
            ("UseMargin", ctypes.c_double),              # 占用的保证金
            ("LongUseMargin", ctypes.c_double),          # 多头保证金
            ("ShortUseMargin", ctypes.c_double),         # 空头保证金
            ("ExchMargin", ctypes.c_double),             # 交易所保证金
            ("LongExchMargin", ctypes.c_double),         # 交易所多头保证金
            ("ShortExchMargin", ctypes.c_double),        # 交易所空头保证金
            ("CloseProfit", ctypes.c_double),            # 平仓盈亏
            ("FrozenCommission", ctypes.c_double),       # 冻结的手续费
            ("Commission", ctypes.c_double),             # 手续费
            ("FrozenCash", ctypes.c_double),             # 冻结的资金
            ("CashIn", ctypes.c_double),                 # 资金差额
            ("PositionProfit", ctypes.c_double),         # 持仓盈亏
            ("OffsetAmount", ctypes.c_double),           # 折抵总金额
            ("LongOffsetAmount", ctypes.c_double),       # 多头折抵总金额
            ("ShortOffsetAmount", ctypes.c_double),      # 空头折抵总金额
            ("ExchOffsetAmount", ctypes.c_double),       # 交易所折抵总金额
            ("LongExchOffsetAmount", ctypes.c_double),   # 交易所多头折抵总金额
            ("ShortExchOffsetAmount", ctypes.c_double),  # 交易所空头折抵总金额
            ("HedgeFlag", ctypes.c_char),                # 投机套保标志
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("ProductGroupID", ctypes.c_char * 81),      # 品种/跨品种标示
        ]

    _capsule_name = "InvestorProductGroupMargin"

    _field_mappings = {
        "reserve1": "reserve1",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "frozen_margin": "FrozenMargin",
        "long_frozen_margin": "LongFrozenMargin",
        "short_frozen_margin": "ShortFrozenMargin",
        "use_margin": "UseMargin",
        "long_use_margin": "LongUseMargin",
        "short_use_margin": "ShortUseMargin",
        "exch_margin": "ExchMargin",
        "long_exch_margin": "LongExchMargin",
        "short_exch_margin": "ShortExchMargin",
        "close_profit": "CloseProfit",
        "frozen_commission": "FrozenCommission",
        "commission": "Commission",
        "frozen_cash": "FrozenCash",
        "cash_in": "CashIn",
        "position_profit": "PositionProfit",
        "offset_amount": "OffsetAmount",
        "long_offset_amount": "LongOffsetAmount",
        "short_offset_amount": "ShortOffsetAmount",
        "exch_offset_amount": "ExchOffsetAmount",
        "long_exch_offset_amount": "LongExchOffsetAmount",
        "short_exch_offset_amount": "ShortExchOffsetAmount",
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
    def frozen_margin(self) -> float:
        """冻结的保证金"""
        return self._struct.FrozenMargin

    @frozen_margin.setter
    def frozen_margin(self, value: float):
        """设置冻结的保证金"""
        self._struct.FrozenMargin = value

    @property
    def long_frozen_margin(self) -> float:
        """多头冻结的保证金"""
        return self._struct.LongFrozenMargin

    @long_frozen_margin.setter
    def long_frozen_margin(self, value: float):
        """设置多头冻结的保证金"""
        self._struct.LongFrozenMargin = value

    @property
    def short_frozen_margin(self) -> float:
        """空头冻结的保证金"""
        return self._struct.ShortFrozenMargin

    @short_frozen_margin.setter
    def short_frozen_margin(self, value: float):
        """设置空头冻结的保证金"""
        self._struct.ShortFrozenMargin = value

    @property
    def use_margin(self) -> float:
        """占用的保证金"""
        return self._struct.UseMargin

    @use_margin.setter
    def use_margin(self, value: float):
        """设置占用的保证金"""
        self._struct.UseMargin = value

    @property
    def long_use_margin(self) -> float:
        """多头保证金"""
        return self._struct.LongUseMargin

    @long_use_margin.setter
    def long_use_margin(self, value: float):
        """设置多头保证金"""
        self._struct.LongUseMargin = value

    @property
    def short_use_margin(self) -> float:
        """空头保证金"""
        return self._struct.ShortUseMargin

    @short_use_margin.setter
    def short_use_margin(self, value: float):
        """设置空头保证金"""
        self._struct.ShortUseMargin = value

    @property
    def exch_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchMargin = value

    @property
    def long_exch_margin(self) -> float:
        """交易所多头保证金"""
        return self._struct.LongExchMargin

    @long_exch_margin.setter
    def long_exch_margin(self, value: float):
        """设置交易所多头保证金"""
        self._struct.LongExchMargin = value

    @property
    def short_exch_margin(self) -> float:
        """交易所空头保证金"""
        return self._struct.ShortExchMargin

    @short_exch_margin.setter
    def short_exch_margin(self, value: float):
        """设置交易所空头保证金"""
        self._struct.ShortExchMargin = value

    @property
    def close_profit(self) -> float:
        """平仓盈亏"""
        return self._struct.CloseProfit

    @close_profit.setter
    def close_profit(self, value: float):
        """设置平仓盈亏"""
        self._struct.CloseProfit = value

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
    def frozen_cash(self) -> float:
        """冻结的资金"""
        return self._struct.FrozenCash

    @frozen_cash.setter
    def frozen_cash(self, value: float):
        """设置冻结的资金"""
        self._struct.FrozenCash = value

    @property
    def cash_in(self) -> float:
        """资金差额"""
        return self._struct.CashIn

    @cash_in.setter
    def cash_in(self, value: float):
        """设置资金差额"""
        self._struct.CashIn = value

    @property
    def position_profit(self) -> float:
        """持仓盈亏"""
        return self._struct.PositionProfit

    @position_profit.setter
    def position_profit(self, value: float):
        """设置持仓盈亏"""
        self._struct.PositionProfit = value

    @property
    def offset_amount(self) -> float:
        """折抵总金额"""
        return self._struct.OffsetAmount

    @offset_amount.setter
    def offset_amount(self, value: float):
        """设置折抵总金额"""
        self._struct.OffsetAmount = value

    @property
    def long_offset_amount(self) -> float:
        """多头折抵总金额"""
        return self._struct.LongOffsetAmount

    @long_offset_amount.setter
    def long_offset_amount(self, value: float):
        """设置多头折抵总金额"""
        self._struct.LongOffsetAmount = value

    @property
    def short_offset_amount(self) -> float:
        """空头折抵总金额"""
        return self._struct.ShortOffsetAmount

    @short_offset_amount.setter
    def short_offset_amount(self, value: float):
        """设置空头折抵总金额"""
        self._struct.ShortOffsetAmount = value

    @property
    def exch_offset_amount(self) -> float:
        """交易所折抵总金额"""
        return self._struct.ExchOffsetAmount

    @exch_offset_amount.setter
    def exch_offset_amount(self, value: float):
        """设置交易所折抵总金额"""
        self._struct.ExchOffsetAmount = value

    @property
    def long_exch_offset_amount(self) -> float:
        """交易所多头折抵总金额"""
        return self._struct.LongExchOffsetAmount

    @long_exch_offset_amount.setter
    def long_exch_offset_amount(self, value: float):
        """设置交易所多头折抵总金额"""
        self._struct.LongExchOffsetAmount = value

    @property
    def short_exch_offset_amount(self) -> float:
        """交易所空头折抵总金额"""
        return self._struct.ShortExchOffsetAmount

    @short_exch_offset_amount.setter
    def short_exch_offset_amount(self, value: float):
        """设置交易所空头折抵总金额"""
        self._struct.ShortExchOffsetAmount = value

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



class ExchangeRate(CapsuleStruct):
    """汇率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("FromCurrencyID", ctypes.c_char * 4),       # 源币种
            ("FromCurrencyUnit", ctypes.c_double),       # 源币种单位数量
            ("ToCurrencyID", ctypes.c_char * 4),         # 目标币种
            ("ExchangeRate", ctypes.c_double),           # 汇率
        ]

    _capsule_name = "ExchangeRate"

    _field_mappings = {
        "broker_id": "BrokerID",
        "from_currency_id": "FromCurrencyID",
        "from_currency_unit": "FromCurrencyUnit",
        "to_currency_id": "ToCurrencyID",
        "exchange_rate": "ExchangeRate",
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
    def from_currency_unit(self) -> float:
        """源币种单位数量"""
        return self._struct.FromCurrencyUnit

    @from_currency_unit.setter
    def from_currency_unit(self, value: float):
        """设置源币种单位数量"""
        self._struct.FromCurrencyUnit = value

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

    @property
    def exchange_rate(self) -> float:
        """汇率"""
        return self._struct.ExchangeRate

    @exchange_rate.setter
    def exchange_rate(self, value: float):
        """设置汇率"""
        self._struct.ExchangeRate = value



class SecAgentACIDMap(CapsuleStruct):
    """二级代理资金帐号映射"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("UserID", ctypes.c_char * 16),              # 用户代码
            ("AccountID", ctypes.c_char * 13),           # 资金账户
            ("CurrencyID", ctypes.c_char * 4),           # 币种
            ("BrokerSecAgentID", ctypes.c_char * 13),    # 境外中介机构资金帐号
        ]

    _capsule_name = "SecAgentACIDMap"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "account_id": "AccountID",
        "currency_id": "CurrencyID",
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



class ProductExchRate(CapsuleStruct):
    """产品报价汇率"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("QuoteCurrencyID", ctypes.c_char * 4),      # 报价币种类型
            ("ExchangeRate", ctypes.c_double),           # 汇率
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ProductID", ctypes.c_char * 81),           # 产品代码
        ]

    _capsule_name = "ProductExchRate"

    _field_mappings = {
        "reserve1": "reserve1",
        "quote_currency_id": "QuoteCurrencyID",
        "exchange_rate": "ExchangeRate",
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
    }

    @property
    def quote_currency_id(self) -> str:
        """报价币种类型"""
        if 'quote_currency_id' not in self._cache:
            value = self._struct.QuoteCurrencyID.rstrip(b'\x00').decode('ascii')
            self._cache['quote_currency_id'] = value
        return self._cache['quote_currency_id']

    @quote_currency_id.setter
    def quote_currency_id(self, value: str):
        """设置报价币种类型"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.QuoteCurrencyID = encoded
        self._cache['quote_currency_id'] = value

    @property
    def exchange_rate(self) -> float:
        """汇率"""
        return self._struct.ExchangeRate

    @exchange_rate.setter
    def exchange_rate(self, value: float):
        """设置汇率"""
        self._struct.ExchangeRate = value

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



class ProductGroup(CapsuleStruct):
    """产品组"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("ProductID", ctypes.c_char * 81),           # 产品代码
            ("ProductGroupID", ctypes.c_char * 81),      # 产品组代码
        ]

    _capsule_name = "ProductGroup"

    _field_mappings = {
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "reserve2": "reserve2",
        "product_id": "ProductID",
        "product_group_id": "ProductGroupID",
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

    @property
    def product_group_id(self) -> str:
        """产品组代码"""
        if 'product_group_id' not in self._cache:
            value = self._struct.ProductGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['product_group_id'] = value
        return self._cache['product_group_id']

    @product_group_id.setter
    def product_group_id(self, value: str):
        """设置产品组代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductGroupID = encoded
        self._cache['product_group_id'] = value



class TradingNoticeInfo(CapsuleStruct):
    """交易通知信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),         # 投资者代码
            ("SendTime", ctypes.c_char * 9),            # 发送时间
            ("FieldContent", ctypes.c_char * 501),      # 消息正文
            ("SequenceSeries", ctypes.c_short),         # 序列系列号
            ("SequenceNo", ctypes.c_int),               # 序列号
            ("InvestUnitID", ctypes.c_char * 17),       # 投资单元代码
        ]

    _capsule_name = "TradingNoticeInfo"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "send_time": "SendTime",
        "field_content": "FieldContent",
        "sequence_series": "SequenceSeries",
        "sequence_no": "SequenceNo",
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
    def send_time(self) -> str:
        """发送时间"""
        if 'send_time' not in self._cache:
            value = self._struct.SendTime.rstrip(b'\x00').decode('ascii')
            self._cache['send_time'] = value
        return self._cache['send_time']

    @send_time.setter
    def send_time(self, value: str):
        """设置发送时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.SendTime = encoded
        self._cache['send_time'] = value

    @property
    def field_content(self) -> str:
        """消息正文（GBK 编码）"""
        if 'field_content' not in self._cache:
            value = self._struct.FieldContent.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['field_content'] = value
        return self._cache['field_content']

    @field_content.setter
    def field_content(self, value: str):
        """设置消息正文（GBK 编码）"""
        encoded = value.encode('gbk')[:500].ljust(501, b'\x00')
        self._struct.FieldContent = encoded
        self._cache['field_content'] = value

    @property
    def sequence_series(self) -> int:
        """序列系列号"""
        return self._struct.SequenceSeries

    @sequence_series.setter
    def sequence_series(self, value: int):
        """设置序列系列号"""
        self._struct.SequenceSeries = value

    @property
    def sequence_no(self) -> int:
        """序列号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序列号"""
        self._struct.SequenceNo = value

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



class CFMMCTradingAccountToken(CapsuleStruct):
    """保证金监控中心交易账户令牌"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("ParticipantID", ctypes.c_char * 11),      # 经纪公司统一编码
            ("AccountID", ctypes.c_char * 13),          # 投资者帐号
            ("KeyID", ctypes.c_int),                    # 密钥编号
            ("Token", ctypes.c_char * 21),              # 动态令牌
        ]

    _capsule_name = "CFMMCTradingAccountToken"

    _field_mappings = {
        "broker_id": "BrokerID",
        "participant_id": "ParticipantID",
        "account_id": "AccountID",
        "key_id": "KeyID",
        "token": "Token",
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
    def participant_id(self) -> str:
        """经纪公司统一编码"""
        if 'participant_id' not in self._cache:
            value = self._struct.ParticipantID.rstrip(b'\x00').decode('ascii')
            self._cache['participant_id'] = value
        return self._cache['participant_id']

    @participant_id.setter
    def participant_id(self, value: str):
        """设置经纪公司统一编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.ParticipantID = encoded
        self._cache['participant_id'] = value

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
    def key_id(self) -> int:
        """密钥编号"""
        return self._struct.KeyID

    @key_id.setter
    def key_id(self, value: int):
        """设置密钥编号"""
        self._struct.KeyID = value

    @property
    def token(self) -> str:
        """动态令牌"""
        if 'token' not in self._cache:
            value = self._struct.Token.rstrip(b'\x00').decode('ascii')
            self._cache['token'] = value
        return self._cache['token']

    @token.setter
    def token(self, value: str):
        """设置动态令牌"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.Token = encoded
        self._cache['token'] = value



class ContractBank(CapsuleStruct):
    """合约银行"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),           # 经纪公司代码
            ("BankID", ctypes.c_char * 4),              # 银行代码
            ("BankBrchID", ctypes.c_char * 5),          # 银行分中心代码
            ("BankName", ctypes.c_char * 101),          # 银行名称
        ]

    _capsule_name = "ContractBank"

    _field_mappings = {
        "broker_id": "BrokerID",
        "bank_id": "BankID",
        "bank_brch_id": "BankBrchID",
        "bank_name": "BankName",
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

    @property
    def bank_name(self) -> str:
        """银行名称（GBK 编码）"""
        if 'bank_name' not in self._cache:
            value = self._struct.BankName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['bank_name'] = value
        return self._cache['bank_name']

    @bank_name.setter
    def bank_name(self, value: str):
        """设置银行名称（GBK 编码）"""
        encoded = value.encode('gbk')[:100].ljust(101, b'\x00')
        self._struct.BankName = encoded
        self._cache['bank_name'] = value



class SecAgentCheckMode(CapsuleStruct):
    """二级代理商资金校验模式"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("CurrencyID", ctypes.c_char * 4),           # 币种
            ("BrokerSecAgentID", ctypes.c_char * 13),    # 境外中介机构资金帐号
            ("CheckSelfAccount", ctypes.c_int),          # 是否需要校验自己的资金账户
        ]

    _capsule_name = "SecAgentCheckMode"

    _field_mappings = {
        "investor_id": "InvestorID",
        "broker_id": "BrokerID",
        "currency_id": "CurrencyID",
        "broker_sec_agent_id": "BrokerSecAgentID",
        "check_self_account": "CheckSelfAccount",
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

    @property
    def check_self_account(self) -> int:
        """是否需要校验自己的资金账户"""
        return self._struct.CheckSelfAccount

    @check_self_account.setter
    def check_self_account(self, value: int):
        """设置是否需要校验自己的资金账户"""
        self._struct.CheckSelfAccount = value



class SecAgentTradeInfo(CapsuleStruct):
    """二级代理商信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("BrokerSecAgentID", ctypes.c_char * 13),    # 境外中介机构资金帐号
            ("InvestorID", ctypes.c_char * 13),          # 投资者代码
            ("LongCustomerName", ctypes.c_char * 161),   # 二级代理商姓名
        ]

    _capsule_name = "SecAgentTradeInfo"

    _field_mappings = {
        "broker_id": "BrokerID",
        "broker_sec_agent_id": "BrokerSecAgentID",
        "investor_id": "InvestorID",
        "long_customer_name": "LongCustomerName",
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
    def long_customer_name(self) -> str:
        """二级代理商姓名（GBK 编码）"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置二级代理商姓名（GBK 编码）"""
        encoded = value.encode('gbk')[:160].ljust(161, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class TransferSerial(CapsuleStruct):
    """转账流水"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("PlateSerial", ctypes.c_int),                # 平台流水号
            ("TradeDate", ctypes.c_char * 9),             # 交易发起方日期
            ("TradingDay", ctypes.c_char * 9),            # 交易日期
            ("TradeTime", ctypes.c_char * 9),             # 交易时间
            ("TradeCode", ctypes.c_char * 7),             # 交易代码
            ("SessionID", ctypes.c_int),                  # 会话编号
            ("BankID", ctypes.c_char * 4),                # 银行编码
            ("BankBranchID", ctypes.c_char * 5),          # 银行分支机构编码
            ("BankAccType", ctypes.c_char),               # 银行帐号类型
            ("BankAccount", ctypes.c_char * 41),          # 银行帐号
            ("BankSerial", ctypes.c_char * 13),           # 银行流水号
            ("BrokerID", ctypes.c_char * 11),             # 期货公司编码
            ("BrokerBranchID", ctypes.c_char * 31),       # 期商分支机构代码
            ("FutureAccType", ctypes.c_char),             # 期货公司帐号类型
            ("AccountID", ctypes.c_char * 13),            # 投资者帐号
            ("InvestorID", ctypes.c_char * 13),           # 投资者代码
            ("FutureSerial", ctypes.c_int),               # 期货公司流水号
            ("IdCardType", ctypes.c_char),                # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),     # 证件号码
            ("CurrencyID", ctypes.c_char * 4),            # 币种代码
            ("TradeAmount", ctypes.c_double),             # 交易金额
            ("CustFee", ctypes.c_double),                 # 应收客户费用
            ("BrokerFee", ctypes.c_double),               # 应收期货公司费用
            ("AvailabilityFlag", ctypes.c_char),          # 有效标志
            ("OperatorCode", ctypes.c_char * 17),         # 操作员
            ("BankNewAccount", ctypes.c_char * 41),       # 新银行帐号
            ("ErrorID", ctypes.c_int),                    # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),             # 错误信息
        ]

    _capsule_name = "TransferSerial"

    _field_mappings = {
        "plate_serial": "PlateSerial",
        "trade_date": "TradeDate",
        "trading_day": "TradingDay",
        "trade_time": "TradeTime",
        "trade_code": "TradeCode",
        "session_id": "SessionID",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "bank_acc_type": "BankAccType",
        "bank_account": "BankAccount",
        "bank_serial": "BankSerial",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "future_acc_type": "FutureAccType",
        "account_id": "AccountID",
        "investor_id": "InvestorID",
        "future_serial": "FutureSerial",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "currency_id": "CurrencyID",
        "trade_amount": "TradeAmount",
        "cust_fee": "CustFee",
        "broker_fee": "BrokerFee",
        "availability_flag": "AvailabilityFlag",
        "operator_code": "OperatorCode",
        "bank_new_account": "BankNewAccount",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
    }

    @property
    def plate_serial(self) -> int:
        """平台流水号"""
        return self._struct.PlateSerial

    @plate_serial.setter
    def plate_serial(self, value: int):
        """设置平台流水号"""
        self._struct.PlateSerial = value

    @property
    def trade_date(self) -> str:
        """交易发起方日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置交易发起方日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trading_day(self) -> str:
        """交易日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    @property
    def trade_time(self) -> str:
        """交易时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置交易时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

    @property
    def trade_code(self) -> str:
        """交易代码"""
        if 'trade_code' not in self._cache:
            value = self._struct.TradeCode.rstrip(b'\x00').decode('ascii')
            self._cache['trade_code'] = value
        return self._cache['trade_code']

    @trade_code.setter
    def trade_code(self, value: str):
        """设置交易代码"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.TradeCode = encoded
        self._cache['trade_code'] = value

    @property
    def session_id(self) -> int:
        """会话编号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话编号"""
        self._struct.SessionID = value

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
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def bank_serial(self) -> str:
        """银行流水号"""
        if 'bank_serial' not in self._cache:
            value = self._struct.BankSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_serial'] = value
        return self._cache['bank_serial']

    @bank_serial.setter
    def bank_serial(self, value: str):
        """设置银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankSerial = encoded
        self._cache['bank_serial'] = value

    @property
    def broker_id(self) -> str:
        """期货公司编码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期货公司编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期商分支机构代码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期商分支机构代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

    @property
    def future_acc_type(self) -> str:
        """期货公司帐号类型"""
        if 'future_acc_type' not in self._cache:
            value = self._struct.FutureAccType.decode('ascii')
            self._cache['future_acc_type'] = value
        return self._cache['future_acc_type']

    @future_acc_type.setter
    def future_acc_type(self, value: str):
        """设置期货公司帐号类型"""
        self._struct.FutureAccType = value.encode('ascii')[0]
        self._cache['future_acc_type'] = value

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
    def future_serial(self) -> int:
        """期货公司流水号"""
        return self._struct.FutureSerial

    @future_serial.setter
    def future_serial(self, value: int):
        """设置期货公司流水号"""
        self._struct.FutureSerial = value

    @property
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def trade_amount(self) -> float:
        """交易金额"""
        return self._struct.TradeAmount

    @trade_amount.setter
    def trade_amount(self, value: float):
        """设置交易金额"""
        self._struct.TradeAmount = value

    @property
    def cust_fee(self) -> float:
        """应收客户费用"""
        return self._struct.CustFee

    @cust_fee.setter
    def cust_fee(self, value: float):
        """设置应收客户费用"""
        self._struct.CustFee = value

    @property
    def broker_fee(self) -> float:
        """应收期货公司费用"""
        return self._struct.BrokerFee

    @broker_fee.setter
    def broker_fee(self, value: float):
        """设置应收期货公司费用"""
        self._struct.BrokerFee = value

    @property
    def availability_flag(self) -> str:
        """有效标志"""
        if 'availability_flag' not in self._cache:
            value = self._struct.AvailabilityFlag.decode('ascii')
            self._cache['availability_flag'] = value
        return self._cache['availability_flag']

    @availability_flag.setter
    def availability_flag(self, value: str):
        """设置有效标志"""
        self._struct.AvailabilityFlag = value.encode('ascii')[0]
        self._cache['availability_flag'] = value

    @property
    def operator_code(self) -> str:
        """操作员"""
        if 'operator_code' not in self._cache:
            value = self._struct.OperatorCode.rstrip(b'\x00').decode('ascii')
            self._cache['operator_code'] = value
        return self._cache['operator_code']

    @operator_code.setter
    def operator_code(self, value: str):
        """设置操作员"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.OperatorCode = encoded
        self._cache['operator_code'] = value

    @property
    def bank_new_account(self) -> str:
        """新银行帐号"""
        if 'bank_new_account' not in self._cache:
            value = self._struct.BankNewAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_new_account'] = value
        return self._cache['bank_new_account']

    @bank_new_account.setter
    def bank_new_account(self, value: str):
        """设置新银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankNewAccount = encoded
        self._cache['bank_new_account'] = value

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



class Bulletin(CapsuleStruct):
    """交易所公告"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("BulletinID", ctypes.c_int),                # 公告编号
            ("SequenceNo", ctypes.c_int),                # 序列号
            ("NewsType", ctypes.c_char * 3),             # 公告类型
            ("NewsUrgency", ctypes.c_char),              # 紧急程度
            ("SendTime", ctypes.c_char * 9),             # 发送时间
            ("Abstract", ctypes.c_char * 81),            # 消息摘要
            ("ComeFrom", ctypes.c_char * 21),            # 消息来源
            ("Content", ctypes.c_char * 501),            # 消息正文（GBK 编码）
            ("URLLink", ctypes.c_char * 201),            # WEB地址
            ("MarketID", ctypes.c_char * 31),            # 市场代码
        ]

    _capsule_name = "Bulletin"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "trading_day": "TradingDay",
        "bulletin_id": "BulletinID",
        "sequence_no": "SequenceNo",
        "news_type": "NewsType",
        "news_urgency": "NewsUrgency",
        "send_time": "SendTime",
        "abstract": "Abstract",
        "come_from": "ComeFrom",
        "content": "Content",
        "url_link": "URLLink",
        "market_id": "MarketID",
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
    def bulletin_id(self) -> int:
        """公告编号"""
        return self._struct.BulletinID

    @bulletin_id.setter
    def bulletin_id(self, value: int):
        """设置公告编号"""
        self._struct.BulletinID = value

    @property
    def sequence_no(self) -> int:
        """序列号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序列号"""
        self._struct.SequenceNo = value

    @property
    def news_type(self) -> str:
        """公告类型"""
        if 'news_type' not in self._cache:
            value = self._struct.NewsType.rstrip(b'\x00').decode('ascii')
            self._cache['news_type'] = value
        return self._cache['news_type']

    @news_type.setter
    def news_type(self, value: str):
        """设置公告类型"""
        encoded = value.encode('ascii')[:2].ljust(3, b'\x00')
        self._struct.NewsType = encoded
        self._cache['news_type'] = value

    @property
    def news_urgency(self) -> str:
        """紧急程度"""
        if 'news_urgency' not in self._cache:
            value = self._struct.NewsUrgency.decode('ascii')
            self._cache['news_urgency'] = value
        return self._cache['news_urgency']

    @news_urgency.setter
    def news_urgency(self, value: str):
        """设置紧急程度"""
        self._struct.NewsUrgency = value.encode('ascii')[0]
        self._cache['news_urgency'] = value

    @property
    def send_time(self) -> str:
        """发送时间"""
        if 'send_time' not in self._cache:
            value = self._struct.SendTime.rstrip(b'\x00').decode('ascii')
            self._cache['send_time'] = value
        return self._cache['send_time']

    @send_time.setter
    def send_time(self, value: str):
        """设置发送时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.SendTime = encoded
        self._cache['send_time'] = value

    @property
    def abstract(self) -> str:
        """消息摘要（GBK 编码）"""
        if 'abstract' not in self._cache:
            value = self._struct.Abstract.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['abstract'] = value
        return self._cache['abstract']

    @abstract.setter
    def abstract(self, value: str):
        """设置消息摘要（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.Abstract = encoded
        self._cache['abstract'] = value

    @property
    def come_from(self) -> str:
        """消息来源"""
        if 'come_from' not in self._cache:
            value = self._struct.ComeFrom.rstrip(b'\x00').decode('ascii')
            self._cache['come_from'] = value
        return self._cache['come_from']

    @come_from.setter
    def come_from(self, value: str):
        """设置消息来源"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ComeFrom = encoded
        self._cache['come_from'] = value

    @property
    def content(self) -> str:
        """消息正文（GBK 编码）"""
        if 'content' not in self._cache:
            value = self._struct.Content.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['content'] = value
        return self._cache['content']

    @content.setter
    def content(self, value: str):
        """设置消息正文（GBK 编码）"""
        encoded = value.encode('gbk')[:500].ljust(501, b'\x00')
        self._struct.Content = encoded
        self._cache['content'] = value

    @property
    def url_link(self) -> str:
        """WEB地址"""
        if 'url_link' not in self._cache:
            value = self._struct.URLLink.rstrip(b'\x00').decode('ascii')
            self._cache['url_link'] = value
        return self._cache['url_link']

    @url_link.setter
    def url_link(self, value: str):
        """设置WEB地址"""
        encoded = value.encode('ascii')[:200].ljust(201, b'\x00')
        self._struct.URLLink = encoded
        self._cache['url_link'] = value

    @property
    def market_id(self) -> str:
        """市场代码"""
        if 'market_id' not in self._cache:
            value = self._struct.MarketID.rstrip(b'\x00').decode('ascii')
            self._cache['market_id'] = value
        return self._cache['market_id']

    @market_id.setter
    def market_id(self, value: str):
        """设置市场代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.MarketID = encoded
        self._cache['market_id'] = value


class TradingNotice(CapsuleStruct):
    """交易通知"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),         # 经纪公司代码
            ("InvestorRange", ctypes.c_char),          # 投资者范围
            ("InvestorID", ctypes.c_char * 13),        # 投资者代码
            ("SequenceSeries", ctypes.c_short),        # 序列系列号
            ("UserID", ctypes.c_char * 16),            # 用户代码
            ("SendTime", ctypes.c_char * 9),           # 发送时间
            ("SequenceNo", ctypes.c_int),              # 序列号
            ("FieldContent", ctypes.c_char * 501),     # 消息正文
            ("InvestUnitID", ctypes.c_char * 17),      # 投资单元代码
        ]

    _capsule_name = "TradingNotice"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_range": "InvestorRange",
        "investor_id": "InvestorID",
        "sequence_series": "SequenceSeries",
        "user_id": "UserID",
        "send_time": "SendTime",
        "sequence_no": "SequenceNo",
        "field_content": "FieldContent",
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
    def sequence_series(self) -> int:
        """序列系列号"""
        return self._struct.SequenceSeries

    @sequence_series.setter
    def sequence_series(self, value: int):
        """设置序列系列号"""
        self._struct.SequenceSeries = value

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
    def send_time(self) -> str:
        """发送时间"""
        if 'send_time' not in self._cache:
            value = self._struct.SendTime.rstrip(b'\x00').decode('ascii')
            self._cache['send_time'] = value
        return self._cache['send_time']

    @send_time.setter
    def send_time(self, value: str):
        """设置发送时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.SendTime = encoded
        self._cache['send_time'] = value

    @property
    def sequence_no(self) -> int:
        """序列号"""
        return self._struct.SequenceNo

    @sequence_no.setter
    def sequence_no(self, value: int):
        """设置序列号"""
        self._struct.SequenceNo = value

    @property
    def field_content(self) -> str:
        """消息正文"""
        if 'field_content' not in self._cache:
            value = self._struct.FieldContent.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['field_content'] = value
        return self._cache['field_content']

    @field_content.setter
    def field_content(self, value: str):
        """设置消息正文"""
        encoded = value.encode('gbk')[:500].ljust(501, b'\x00')
        self._struct.FieldContent = encoded
        self._cache['field_content'] = value

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



class BrokerTradingParams(CapsuleStruct):
    """经纪公司交易参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),         # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),        # 投资者代码
            ("MarginPriceType", ctypes.c_char),        # 保证金价格类型
            ("Algorithm", ctypes.c_char),              # 盈亏算法
            ("AvailIncludeCloseProfit", ctypes.c_char), # 可用是否包含平仓盈利
            ("CurrencyID", ctypes.c_char * 4),         # 币种代码
            ("OptionRoyaltyPriceType", ctypes.c_char), # 期权权利金价格类型
            ("AccountID", ctypes.c_char * 13),         # 投资者帐号
        ]

    _capsule_name = "BrokerTradingParams"

    _field_mappings = {
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "margin_price_type": "MarginPriceType",
        "algorithm": "Algorithm",
        "avail_include_close_profit": "AvailIncludeCloseProfit",
        "currency_id": "CurrencyID",
        "option_royalty_price_type": "OptionRoyaltyPriceType",
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
    def margin_price_type(self) -> str:
        """保证金价格类型"""
        if 'margin_price_type' not in self._cache:
            value = self._struct.MarginPriceType.decode('ascii')
            self._cache['margin_price_type'] = value
        return self._cache['margin_price_type']

    @margin_price_type.setter
    def margin_price_type(self, value: str):
        """设置保证金价格类型"""
        self._struct.MarginPriceType = value.encode('ascii')[0]
        self._cache['margin_price_type'] = value

    @property
    def algorithm(self) -> str:
        """盈亏算法"""
        if 'algorithm' not in self._cache:
            value = self._struct.Algorithm.decode('ascii')
            self._cache['algorithm'] = value
        return self._cache['algorithm']

    @algorithm.setter
    def algorithm(self, value: str):
        """设置盈亏算法"""
        self._struct.Algorithm = value.encode('ascii')[0]
        self._cache['algorithm'] = value

    @property
    def avail_include_close_profit(self) -> str:
        """可用是否包含平仓盈利"""
        if 'avail_include_close_profit' not in self._cache:
            value = self._struct.AvailIncludeCloseProfit.decode('ascii')
            self._cache['avail_include_close_profit'] = value
        return self._cache['avail_include_close_profit']

    @avail_include_close_profit.setter
    def avail_include_close_profit(self, value: str):
        """设置可用是否包含平仓盈利"""
        self._struct.AvailIncludeCloseProfit = value.encode('ascii')[0]
        self._cache['avail_include_close_profit'] = value

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
    def option_royalty_price_type(self) -> str:
        """期权权利金价格类型"""
        if 'option_royalty_price_type' not in self._cache:
            value = self._struct.OptionRoyaltyPriceType.decode('ascii')
            self._cache['option_royalty_price_type'] = value
        return self._cache['option_royalty_price_type']

    @option_royalty_price_type.setter
    def option_royalty_price_type(self, value: str):
        """设置期权权利金价格类型"""
        self._struct.OptionRoyaltyPriceType = value.encode('ascii')[0]
        self._cache['option_royalty_price_type'] = value

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



class BrokerTradingAlgos(CapsuleStruct):
    """经纪公司交易算法"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),         # 经纪公司代码
            ("ExchangeID", ctypes.c_char * 9),        # 交易所代码
            ("reserve1", ctypes.c_char * 31),         # 保留的无效字段
            ("HandlePositionAlgoID", ctypes.c_char),  # 持仓处理算法编号
            ("FindMarginRateAlgoID", ctypes.c_char),  # 寻找保证金率算法编号
            ("HandleTradingAccountAlgoID", ctypes.c_char), # 资金处理算法编号
            ("InstrumentID", ctypes.c_char * 81),     # 合约代码
        ]

    _capsule_name = "BrokerTradingAlgos"

    _field_mappings = {
        "broker_id": "BrokerID",
        "exchange_id": "ExchangeID",
        "reserve1": "reserve1",
        "handle_position_algo_id": "HandlePositionAlgoID",
        "find_margin_rate_algo_id": "FindMarginRateAlgoID",
        "handle_trading_account_algo_id": "HandleTradingAccountAlgoID",
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
    def handle_position_algo_id(self) -> str:
        """持仓处理算法编号"""
        if 'handle_position_algo_id' not in self._cache:
            value = self._struct.HandlePositionAlgoID.decode('ascii')
            self._cache['handle_position_algo_id'] = value
        return self._cache['handle_position_algo_id']

    @handle_position_algo_id.setter
    def handle_position_algo_id(self, value: str):
        """设置持仓处理算法编号"""
        self._struct.HandlePositionAlgoID = value.encode('ascii')[0]
        self._cache['handle_position_algo_id'] = value

    @property
    def find_margin_rate_algo_id(self) -> str:
        """寻找保证金率算法编号"""
        if 'find_margin_rate_algo_id' not in self._cache:
            value = self._struct.FindMarginRateAlgoID.decode('ascii')
            self._cache['find_margin_rate_algo_id'] = value
        return self._cache['find_margin_rate_algo_id']

    @find_margin_rate_algo_id.setter
    def find_margin_rate_algo_id(self, value: str):
        """设置寻找保证金率算法编号"""
        self._struct.FindMarginRateAlgoID = value.encode('ascii')[0]
        self._cache['find_margin_rate_algo_id'] = value

    @property
    def handle_trading_account_algo_id(self) -> str:
        """资金处理算法编号"""
        if 'handle_trading_account_algo_id' not in self._cache:
            value = self._struct.HandleTradingAccountAlgoID.decode('ascii')
            self._cache['handle_trading_account_algo_id'] = value
        return self._cache['handle_trading_account_algo_id']

    @handle_trading_account_algo_id.setter
    def handle_trading_account_algo_id(self, value: str):
        """设置资金处理算法编号"""
        self._struct.HandleTradingAccountAlgoID = value.encode('ascii')[0]
        self._cache['handle_trading_account_algo_id'] = value

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



class QueryCFMMCTradingAccountToken(CapsuleStruct):
    """查询监控中心交易账户令牌"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),         # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),        # 投资者代码
            ("InvestUnitID", ctypes.c_char * 17),      # 投资单元代码
        ]

    _capsule_name = "QueryCFMMCTradingAccountToken"

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



class ReqTransfer(CapsuleStruct):
    """转账请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeCode", ctypes.c_char * 7),          # 业务功能码
            ("BankID", ctypes.c_char * 4),             # 银行代码
            ("BankBranchID", ctypes.c_char * 5),       # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),          # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),    # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),          # 交易日期
            ("TradeTime", ctypes.c_char * 9),          # 交易时间
            ("BankSerial", ctypes.c_char * 13),        # 银行流水号
            ("TradingDay", ctypes.c_char * 9),         # 交易系统日期
            ("PlateSerial", ctypes.c_int),             # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),           # 最后分片标志
            ("SessionID", ctypes.c_int),               # 会话号
            ("CustomerName", ctypes.c_char * 51),      # 客户姓名
            ("IdCardType", ctypes.c_char),             # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),  # 证件号码
            ("CustType", ctypes.c_char),               # 客户类型
            ("BankAccount", ctypes.c_char * 41),       # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),      # 银行密码
            ("AccountID", ctypes.c_char * 13),         # 投资者帐号
            ("Password", ctypes.c_char * 41),          # 期货密码
            ("InstallID", ctypes.c_int),               # 安装编号
            ("FutureSerial", ctypes.c_int),            # 期货公司流水号
            ("UserID", ctypes.c_char * 16),            # 用户标识
            ("VerifyCertNoFlag", ctypes.c_char),       # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),         # 币种代码
            ("TradeAmount", ctypes.c_double),          # 转帐金额
            ("FutureFetchAmount", ctypes.c_double),    # 期货可取金额
            ("FeePayFlag", ctypes.c_char),             # 费用支付标志
            ("CustFee", ctypes.c_double),              # 应收客户费用
            ("BrokerFee", ctypes.c_double),            # 应收期货公司费用
            ("Message", ctypes.c_char * 129),          # 发送方给接收方的消息
            ("Digest", ctypes.c_char * 36),            # 摘要
            ("BankAccType", ctypes.c_char),            # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),           # 渠道标志
            ("BankSecuAccType", ctypes.c_char),        # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),    # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),       # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),            # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),            # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),            # 交易柜员
            ("RequestID", ctypes.c_int),               # 请求编号
            ("TID", ctypes.c_int),                     # 交易ID
            ("TransferStatus", ctypes.c_char),         # 转账交易状态
            ("LongCustomerName", ctypes.c_char * 161), # 长客户姓名
        ]

    _capsule_name = "ReqTransfer"

    _field_mappings = {
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "cust_type": "CustType",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "install_id": "InstallID",
        "future_serial": "FutureSerial",
        "user_id": "UserID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "trade_amount": "TradeAmount",
        "future_fetch_amount": "FutureFetchAmount",
        "fee_pay_flag": "FeePayFlag",
        "cust_fee": "CustFee",
        "broker_fee": "BrokerFee",
        "message": "Message",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "request_id": "RequestID",
        "tid": "TID",
        "transfer_status": "TransferStatus",
        "long_customer_name": "LongCustomerName",
    }

    @property
    def trade_code(self) -> str:
        """业务功能码"""
        if 'trade_code' not in self._cache:
            value = self._struct.TradeCode.rstrip(b'\x00').decode('ascii')
            self._cache['trade_code'] = value
        return self._cache['trade_code']

    @trade_code.setter
    def trade_code(self, value: str):
        """设置业务功能码"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.TradeCode = encoded
        self._cache['trade_code'] = value

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
    def bank_branch_id(self) -> str:
        """银行分支机构代码"""
        if 'bank_branch_id' not in self._cache:
            value = self._struct.BankBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_branch_id'] = value
        return self._cache['bank_branch_id']

    @bank_branch_id.setter
    def bank_branch_id(self, value: str):
        """设置银行分支机构代码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBranchID = encoded
        self._cache['bank_branch_id'] = value

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期商分支机构代码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期商分支机构代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

    @property
    def trade_date(self) -> str:
        """交易日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trade_time(self) -> str:
        """交易时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置交易时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

    @property
    def bank_serial(self) -> str:
        """银行流水号"""
        if 'bank_serial' not in self._cache:
            value = self._struct.BankSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_serial'] = value
        return self._cache['bank_serial']

    @bank_serial.setter
    def bank_serial(self, value: str):
        """设置银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankSerial = encoded
        self._cache['bank_serial'] = value

    @property
    def trading_day(self) -> str:
        """交易系统日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易系统日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    @property
    def plate_serial(self) -> int:
        """银期平台消息流水号"""
        return self._struct.PlateSerial

    @plate_serial.setter
    def plate_serial(self, value: int):
        """设置银期平台消息流水号"""
        self._struct.PlateSerial = value

    @property
    def last_fragment(self) -> str:
        """最后分片标志"""
        if 'last_fragment' not in self._cache:
            value = self._struct.LastFragment.decode('ascii')
            self._cache['last_fragment'] = value
        return self._cache['last_fragment']

    @last_fragment.setter
    def last_fragment(self, value: str):
        """设置最后分片标志"""
        self._struct.LastFragment = value.encode('ascii')[0]
        self._cache['last_fragment'] = value

    @property
    def session_id(self) -> int:
        """会话号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话号"""
        self._struct.SessionID = value

    @property
    def customer_name(self) -> str:
        """客户姓名"""
        if 'customer_name' not in self._cache:
            value = self._struct.CustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['customer_name'] = value
        return self._cache['customer_name']

    @customer_name.setter
    def customer_name(self, value: str):
        """设置客户姓名"""
        encoded = value.encode('gbk')[:50].ljust(51, b'\x00')
        self._struct.CustomerName = encoded
        self._cache['customer_name'] = value

    @property
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def cust_type(self) -> str:
        """客户类型"""
        if 'cust_type' not in self._cache:
            value = self._struct.CustType.decode('ascii')
            self._cache['cust_type'] = value
        return self._cache['cust_type']

    @cust_type.setter
    def cust_type(self, value: str):
        """设置客户类型"""
        self._struct.CustType = value.encode('ascii')[0]
        self._cache['cust_type'] = value

    @property
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def bank_pass_word(self) -> str:
        """银行密码"""
        if 'bank_pass_word' not in self._cache:
            value = self._struct.BankPassWord.rstrip(b'\x00').decode('ascii')
            self._cache['bank_pass_word'] = value
        return self._cache['bank_pass_word']

    @bank_pass_word.setter
    def bank_pass_word(self, value: str):
        """设置银行密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankPassWord = encoded
        self._cache['bank_pass_word'] = value

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
    def password(self) -> str:
        """期货密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置期货密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def future_serial(self) -> int:
        """期货公司流水号"""
        return self._struct.FutureSerial

    @future_serial.setter
    def future_serial(self, value: int):
        """设置期货公司流水号"""
        self._struct.FutureSerial = value

    @property
    def user_id(self) -> str:
        """用户标识"""
        if 'user_id' not in self._cache:
            value = self._struct.UserID.rstrip(b'\x00').decode('ascii')
            self._cache['user_id'] = value
        return self._cache['user_id']

    @user_id.setter
    def user_id(self, value: str):
        """设置用户标识"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.UserID = encoded
        self._cache['user_id'] = value

    @property
    def verify_cert_no_flag(self) -> str:
        """验证客户证件号码标志"""
        if 'verify_cert_no_flag' not in self._cache:
            value = self._struct.VerifyCertNoFlag.decode('ascii')
            self._cache['verify_cert_no_flag'] = value
        return self._cache['verify_cert_no_flag']

    @verify_cert_no_flag.setter
    def verify_cert_no_flag(self, value: str):
        """设置验证客户证件号码标志"""
        self._struct.VerifyCertNoFlag = value.encode('ascii')[0]
        self._cache['verify_cert_no_flag'] = value

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
    def trade_amount(self) -> float:
        """转帐金额"""
        return self._struct.TradeAmount

    @trade_amount.setter
    def trade_amount(self, value: float):
        """设置转帐金额"""
        self._struct.TradeAmount = value

    @property
    def future_fetch_amount(self) -> float:
        """期货可取金额"""
        return self._struct.FutureFetchAmount

    @future_fetch_amount.setter
    def future_fetch_amount(self, value: float):
        """设置期货可取金额"""
        self._struct.FutureFetchAmount = value

    @property
    def fee_pay_flag(self) -> str:
        """费用支付标志"""
        if 'fee_pay_flag' not in self._cache:
            value = self._struct.FeePayFlag.decode('ascii')
            self._cache['fee_pay_flag'] = value
        return self._cache['fee_pay_flag']

    @fee_pay_flag.setter
    def fee_pay_flag(self, value: str):
        """设置费用支付标志"""
        self._struct.FeePayFlag = value.encode('ascii')[0]
        self._cache['fee_pay_flag'] = value

    @property
    def cust_fee(self) -> float:
        """应收客户费用"""
        return self._struct.CustFee

    @cust_fee.setter
    def cust_fee(self, value: float):
        """设置应收客户费用"""
        self._struct.CustFee = value

    @property
    def broker_fee(self) -> float:
        """应收期货公司费用"""
        return self._struct.BrokerFee

    @broker_fee.setter
    def broker_fee(self, value: float):
        """设置应收期货公司费用"""
        self._struct.BrokerFee = value

    @property
    def message(self) -> str:
        """发送方给接收方的消息"""
        if 'message' not in self._cache:
            value = self._struct.Message.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['message'] = value
        return self._cache['message']

    @message.setter
    def message(self, value: str):
        """设置发送方给接收方的消息"""
        encoded = value.encode('gbk')[:128].ljust(129, b'\x00')
        self._struct.Message = encoded
        self._cache['message'] = value

    @property
    def digest(self) -> str:
        """摘要"""
        if 'digest' not in self._cache:
            value = self._struct.Digest.rstrip(b'\x00').decode('ascii')
            self._cache['digest'] = value
        return self._cache['digest']

    @digest.setter
    def digest(self, value: str):
        """设置摘要"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.Digest = encoded
        self._cache['digest'] = value

    @property
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def device_id(self) -> str:
        """渠道标志"""
        if 'device_id' not in self._cache:
            value = self._struct.DeviceID.rstrip(b'\x00').decode('ascii')
            self._cache['device_id'] = value
        return self._cache['device_id']

    @device_id.setter
    def device_id(self, value: str):
        """设置渠道标志"""
        encoded = value.encode('ascii')[:2].ljust(3, b'\x00')
        self._struct.DeviceID = encoded
        self._cache['device_id'] = value

    @property
    def bank_secu_acc_type(self) -> str:
        """期货单位帐号类型"""
        if 'bank_secu_acc_type' not in self._cache:
            value = self._struct.BankSecuAccType.decode('ascii')
            self._cache['bank_secu_acc_type'] = value
        return self._cache['bank_secu_acc_type']

    @bank_secu_acc_type.setter
    def bank_secu_acc_type(self, value: str):
        """设置期货单位帐号类型"""
        self._struct.BankSecuAccType = value.encode('ascii')[0]
        self._cache['bank_secu_acc_type'] = value

    @property
    def broker_id_by_bank(self) -> str:
        """期货公司银行编码"""
        if 'broker_id_by_bank' not in self._cache:
            value = self._struct.BrokerIDByBank.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id_by_bank'] = value
        return self._cache['broker_id_by_bank']

    @broker_id_by_bank.setter
    def broker_id_by_bank(self, value: str):
        """设置期货公司银行编码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.BrokerIDByBank = encoded
        self._cache['broker_id_by_bank'] = value

    @property
    def bank_secu_acc(self) -> str:
        """期货单位帐号"""
        if 'bank_secu_acc' not in self._cache:
            value = self._struct.BankSecuAcc.rstrip(b'\x00').decode('ascii')
            self._cache['bank_secu_acc'] = value
        return self._cache['bank_secu_acc']

    @bank_secu_acc.setter
    def bank_secu_acc(self, value: str):
        """设置期货单位帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankSecuAcc = encoded
        self._cache['bank_secu_acc'] = value

    @property
    def bank_pwd_flag(self) -> str:
        """银行密码标志"""
        if 'bank_pwd_flag' not in self._cache:
            value = self._struct.BankPwdFlag.decode('ascii')
            self._cache['bank_pwd_flag'] = value
        return self._cache['bank_pwd_flag']

    @bank_pwd_flag.setter
    def bank_pwd_flag(self, value: str):
        """设置银行密码标志"""
        self._struct.BankPwdFlag = value.encode('ascii')[0]
        self._cache['bank_pwd_flag'] = value

    @property
    def secu_pwd_flag(self) -> str:
        """期货资金密码核对标志"""
        if 'secu_pwd_flag' not in self._cache:
            value = self._struct.SecuPwdFlag.decode('ascii')
            self._cache['secu_pwd_flag'] = value
        return self._cache['secu_pwd_flag']

    @secu_pwd_flag.setter
    def secu_pwd_flag(self, value: str):
        """设置期货资金密码核对标志"""
        self._struct.SecuPwdFlag = value.encode('ascii')[0]
        self._cache['secu_pwd_flag'] = value

    @property
    def oper_no(self) -> str:
        """交易柜员"""
        if 'oper_no' not in self._cache:
            value = self._struct.OperNo.rstrip(b'\x00').decode('ascii')
            self._cache['oper_no'] = value
        return self._cache['oper_no']

    @oper_no.setter
    def oper_no(self, value: str):
        """设置交易柜员"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.OperNo = encoded
        self._cache['oper_no'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def transfer_status(self) -> str:
        """转账交易状态"""
        if 'transfer_status' not in self._cache:
            value = self._struct.TransferStatus.decode('ascii')
            self._cache['transfer_status'] = value
        return self._cache['transfer_status']

    @transfer_status.setter
    def transfer_status(self, value: str):
        """设置转账交易状态"""
        self._struct.TransferStatus = value.encode('ascii')[0]
        self._cache['transfer_status'] = value

    @property
    def long_customer_name(self) -> str:
        """长客户姓名"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名"""
        encoded = value.encode('gbk')[:160].ljust(161, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class RspTransfer(CapsuleStruct):
    """转账响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeCode", ctypes.c_char * 7),          # 业务功能码
            ("BankID", ctypes.c_char * 4),             # 银行代码
            ("BankBranchID", ctypes.c_char * 5),       # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),          # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),    # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),          # 交易日期
            ("TradeTime", ctypes.c_char * 9),          # 交易时间
            ("BankSerial", ctypes.c_char * 13),        # 银行流水号
            ("TradingDay", ctypes.c_char * 9),         # 交易系统日期
            ("PlateSerial", ctypes.c_int),             # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),           # 最后分片标志
            ("SessionID", ctypes.c_int),               # 会话号
            ("CustomerName", ctypes.c_char * 51),      # 客户姓名
            ("IdCardType", ctypes.c_char),             # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),  # 证件号码
            ("CustType", ctypes.c_char),               # 客户类型
            ("BankAccount", ctypes.c_char * 41),       # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),      # 银行密码
            ("AccountID", ctypes.c_char * 13),         # 投资者帐号
            ("Password", ctypes.c_char * 41),          # 期货密码
            ("InstallID", ctypes.c_int),               # 安装编号
            ("FutureSerial", ctypes.c_int),            # 期货公司流水号
            ("UserID", ctypes.c_char * 16),            # 用户标识
            ("VerifyCertNoFlag", ctypes.c_char),       # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),         # 币种代码
            ("TradeAmount", ctypes.c_double),          # 转帐金额
            ("FutureFetchAmount", ctypes.c_double),    # 期货可取金额
            ("FeePayFlag", ctypes.c_char),             # 费用支付标志
            ("CustFee", ctypes.c_double),              # 应收客户费用
            ("BrokerFee", ctypes.c_double),            # 应收期货公司费用
            ("Message", ctypes.c_char * 129),          # 发送方给接收方的消息
            ("Digest", ctypes.c_char * 36),            # 摘要
            ("BankAccType", ctypes.c_char),            # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),           # 渠道标志
            ("BankSecuAccType", ctypes.c_char),        # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),    # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),       # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),            # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),            # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),            # 交易柜员
            ("RequestID", ctypes.c_int),               # 请求编号
            ("TID", ctypes.c_int),                     # 交易ID
            ("TransferStatus", ctypes.c_char),         # 转账交易状态
            ("ErrorID", ctypes.c_int),                 # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),          # 错误信息
            ("LongCustomerName", ctypes.c_char * 161), # 长客户姓名
        ]

    _capsule_name = "RspTransfer"

    _field_mappings = {
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "cust_type": "CustType",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "install_id": "InstallID",
        "future_serial": "FutureSerial",
        "user_id": "UserID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "trade_amount": "TradeAmount",
        "future_fetch_amount": "FutureFetchAmount",
        "fee_pay_flag": "FeePayFlag",
        "cust_fee": "CustFee",
        "broker_fee": "BrokerFee",
        "message": "Message",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "request_id": "RequestID",
        "tid": "TID",
        "transfer_status": "TransferStatus",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
        "long_customer_name": "LongCustomerName",
    }

    @property
    def trade_code(self) -> str:
        """业务功能码"""
        if 'trade_code' not in self._cache:
            value = self._struct.TradeCode.rstrip(b'\x00').decode('ascii')
            self._cache['trade_code'] = value
        return self._cache['trade_code']

    @trade_code.setter
    def trade_code(self, value: str):
        """设置业务功能码"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.TradeCode = encoded
        self._cache['trade_code'] = value

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
    def bank_branch_id(self) -> str:
        """银行分支机构代码"""
        if 'bank_branch_id' not in self._cache:
            value = self._struct.BankBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_branch_id'] = value
        return self._cache['bank_branch_id']

    @bank_branch_id.setter
    def bank_branch_id(self, value: str):
        """设置银行分支机构代码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBranchID = encoded
        self._cache['bank_branch_id'] = value

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期商分支机构代码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期商分支机构代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

    @property
    def trade_date(self) -> str:
        """交易日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trade_time(self) -> str:
        """交易时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置交易时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

    @property
    def bank_serial(self) -> str:
        """银行流水号"""
        if 'bank_serial' not in self._cache:
            value = self._struct.BankSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_serial'] = value
        return self._cache['bank_serial']

    @bank_serial.setter
    def bank_serial(self, value: str):
        """设置银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankSerial = encoded
        self._cache['bank_serial'] = value

    @property
    def trading_day(self) -> str:
        """交易系统日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易系统日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    @property
    def plate_serial(self) -> int:
        """银期平台消息流水号"""
        return self._struct.PlateSerial

    @plate_serial.setter
    def plate_serial(self, value: int):
        """设置银期平台消息流水号"""
        self._struct.PlateSerial = value

    @property
    def last_fragment(self) -> str:
        """最后分片标志"""
        if 'last_fragment' not in self._cache:
            value = self._struct.LastFragment.decode('ascii')
            self._cache['last_fragment'] = value
        return self._cache['last_fragment']

    @last_fragment.setter
    def last_fragment(self, value: str):
        """设置最后分片标志"""
        self._struct.LastFragment = value.encode('ascii')[0]
        self._cache['last_fragment'] = value

    @property
    def session_id(self) -> int:
        """会话号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话号"""
        self._struct.SessionID = value

    @property
    def customer_name(self) -> str:
        """客户姓名"""
        if 'customer_name' not in self._cache:
            value = self._struct.CustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['customer_name'] = value
        return self._cache['customer_name']

    @customer_name.setter
    def customer_name(self, value: str):
        """设置客户姓名"""
        encoded = value.encode('gbk')[:50].ljust(51, b'\x00')
        self._struct.CustomerName = encoded
        self._cache['customer_name'] = value

    @property
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def cust_type(self) -> str:
        """客户类型"""
        if 'cust_type' not in self._cache:
            value = self._struct.CustType.decode('ascii')
            self._cache['cust_type'] = value
        return self._cache['cust_type']

    @cust_type.setter
    def cust_type(self, value: str):
        """设置客户类型"""
        self._struct.CustType = value.encode('ascii')[0]
        self._cache['cust_type'] = value

    @property
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def bank_pass_word(self) -> str:
        """银行密码"""
        if 'bank_pass_word' not in self._cache:
            value = self._struct.BankPassWord.rstrip(b'\x00').decode('ascii')
            self._cache['bank_pass_word'] = value
        return self._cache['bank_pass_word']

    @bank_pass_word.setter
    def bank_pass_word(self, value: str):
        """设置银行密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankPassWord = encoded
        self._cache['bank_pass_word'] = value

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
    def password(self) -> str:
        """期货密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置期货密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def future_serial(self) -> int:
        """期货公司流水号"""
        return self._struct.FutureSerial

    @future_serial.setter
    def future_serial(self, value: int):
        """设置期货公司流水号"""
        self._struct.FutureSerial = value

    @property
    def user_id(self) -> str:
        """用户标识"""
        if 'user_id' not in self._cache:
            value = self._struct.UserID.rstrip(b'\x00').decode('ascii')
            self._cache['user_id'] = value
        return self._cache['user_id']

    @user_id.setter
    def user_id(self, value: str):
        """设置用户标识"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.UserID = encoded
        self._cache['user_id'] = value

    @property
    def verify_cert_no_flag(self) -> str:
        """验证客户证件号码标志"""
        if 'verify_cert_no_flag' not in self._cache:
            value = self._struct.VerifyCertNoFlag.decode('ascii')
            self._cache['verify_cert_no_flag'] = value
        return self._cache['verify_cert_no_flag']

    @verify_cert_no_flag.setter
    def verify_cert_no_flag(self, value: str):
        """设置验证客户证件号码标志"""
        self._struct.VerifyCertNoFlag = value.encode('ascii')[0]
        self._cache['verify_cert_no_flag'] = value

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
    def trade_amount(self) -> float:
        """转帐金额"""
        return self._struct.TradeAmount

    @trade_amount.setter
    def trade_amount(self, value: float):
        """设置转帐金额"""
        self._struct.TradeAmount = value

    @property
    def future_fetch_amount(self) -> float:
        """期货可取金额"""
        return self._struct.FutureFetchAmount

    @future_fetch_amount.setter
    def future_fetch_amount(self, value: float):
        """设置期货可取金额"""
        self._struct.FutureFetchAmount = value

    @property
    def fee_pay_flag(self) -> str:
        """费用支付标志"""
        if 'fee_pay_flag' not in self._cache:
            value = self._struct.FeePayFlag.decode('ascii')
            self._cache['fee_pay_flag'] = value
        return self._cache['fee_pay_flag']

    @fee_pay_flag.setter
    def fee_pay_flag(self, value: str):
        """设置费用支付标志"""
        self._struct.FeePayFlag = value.encode('ascii')[0]
        self._cache['fee_pay_flag'] = value

    @property
    def cust_fee(self) -> float:
        """应收客户费用"""
        return self._struct.CustFee

    @cust_fee.setter
    def cust_fee(self, value: float):
        """设置应收客户费用"""
        self._struct.CustFee = value

    @property
    def broker_fee(self) -> float:
        """应收期货公司费用"""
        return self._struct.BrokerFee

    @broker_fee.setter
    def broker_fee(self, value: float):
        """设置应收期货公司费用"""
        self._struct.BrokerFee = value

    @property
    def message(self) -> str:
        """发送方给接收方的消息"""
        if 'message' not in self._cache:
            value = self._struct.Message.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['message'] = value
        return self._cache['message']

    @message.setter
    def message(self, value: str):
        """设置发送方给接收方的消息"""
        encoded = value.encode('gbk')[:128].ljust(129, b'\x00')
        self._struct.Message = encoded
        self._cache['message'] = value

    @property
    def digest(self) -> str:
        """摘要"""
        if 'digest' not in self._cache:
            value = self._struct.Digest.rstrip(b'\x00').decode('ascii')
            self._cache['digest'] = value
        return self._cache['digest']

    @digest.setter
    def digest(self, value: str):
        """设置摘要"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.Digest = encoded
        self._cache['digest'] = value

    @property
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def device_id(self) -> str:
        """渠道标志"""
        if 'device_id' not in self._cache:
            value = self._struct.DeviceID.rstrip(b'\x00').decode('ascii')
            self._cache['device_id'] = value
        return self._cache['device_id']

    @device_id.setter
    def device_id(self, value: str):
        """设置渠道标志"""
        encoded = value.encode('ascii')[:2].ljust(3, b'\x00')
        self._struct.DeviceID = encoded
        self._cache['device_id'] = value

    @property
    def bank_secu_acc_type(self) -> str:
        """期货单位帐号类型"""
        if 'bank_secu_acc_type' not in self._cache:
            value = self._struct.BankSecuAccType.decode('ascii')
            self._cache['bank_secu_acc_type'] = value
        return self._cache['bank_secu_acc_type']

    @bank_secu_acc_type.setter
    def bank_secu_acc_type(self, value: str):
        """设置期货单位帐号类型"""
        self._struct.BankSecuAccType = value.encode('ascii')[0]
        self._cache['bank_secu_acc_type'] = value

    @property
    def broker_id_by_bank(self) -> str:
        """期货公司银行编码"""
        if 'broker_id_by_bank' not in self._cache:
            value = self._struct.BrokerIDByBank.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id_by_bank'] = value
        return self._cache['broker_id_by_bank']

    @broker_id_by_bank.setter
    def broker_id_by_bank(self, value: str):
        """设置期货公司银行编码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.BrokerIDByBank = encoded
        self._cache['broker_id_by_bank'] = value

    @property
    def bank_secu_acc(self) -> str:
        """期货单位帐号"""
        if 'bank_secu_acc' not in self._cache:
            value = self._struct.BankSecuAcc.rstrip(b'\x00').decode('ascii')
            self._cache['bank_secu_acc'] = value
        return self._cache['bank_secu_acc']

    @bank_secu_acc.setter
    def bank_secu_acc(self, value: str):
        """设置期货单位帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankSecuAcc = encoded
        self._cache['bank_secu_acc'] = value

    @property
    def bank_pwd_flag(self) -> str:
        """银行密码标志"""
        if 'bank_pwd_flag' not in self._cache:
            value = self._struct.BankPwdFlag.decode('ascii')
            self._cache['bank_pwd_flag'] = value
        return self._cache['bank_pwd_flag']

    @bank_pwd_flag.setter
    def bank_pwd_flag(self, value: str):
        """设置银行密码标志"""
        self._struct.BankPwdFlag = value.encode('ascii')[0]
        self._cache['bank_pwd_flag'] = value

    @property
    def secu_pwd_flag(self) -> str:
        """期货资金密码核对标志"""
        if 'secu_pwd_flag' not in self._cache:
            value = self._struct.SecuPwdFlag.decode('ascii')
            self._cache['secu_pwd_flag'] = value
        return self._cache['secu_pwd_flag']

    @secu_pwd_flag.setter
    def secu_pwd_flag(self, value: str):
        """设置期货资金密码核对标志"""
        self._struct.SecuPwdFlag = value.encode('ascii')[0]
        self._cache['secu_pwd_flag'] = value

    @property
    def oper_no(self) -> str:
        """交易柜员"""
        if 'oper_no' not in self._cache:
            value = self._struct.OperNo.rstrip(b'\x00').decode('ascii')
            self._cache['oper_no'] = value
        return self._cache['oper_no']

    @oper_no.setter
    def oper_no(self, value: str):
        """设置交易柜员"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.OperNo = encoded
        self._cache['oper_no'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def transfer_status(self) -> str:
        """转账交易状态"""
        if 'transfer_status' not in self._cache:
            value = self._struct.TransferStatus.decode('ascii')
            self._cache['transfer_status'] = value
        return self._cache['transfer_status']

    @transfer_status.setter
    def transfer_status(self, value: str):
        """设置转账交易状态"""
        self._struct.TransferStatus = value.encode('ascii')[0]
        self._cache['transfer_status'] = value

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
        """错误信息"""
        if 'error_msg' not in self._cache:
            value = self._struct.ErrorMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['error_msg'] = value
        return self._cache['error_msg']

    @error_msg.setter
    def error_msg(self, value: str):
        """设置错误信息"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.ErrorMsg = encoded
        self._cache['error_msg'] = value

    @property
    def long_customer_name(self) -> str:
        """长客户姓名"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名"""
        encoded = value.encode('gbk')[:160].ljust(161, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class RspRepeal(CapsuleStruct):
    """冲正响应"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("RepealTimeInterval", ctypes.c_int),       # 冲正时间间隔
            ("RepealedTimes", ctypes.c_int),            # 已经冲正次数
            ("BankRepealFlag", ctypes.c_char),          # 银行冲正标志
            ("BrokerRepealFlag", ctypes.c_char),        # 期商冲正标志
            ("PlateRepealSerial", ctypes.c_int),        # 被冲正平台流水号
            ("BankRepealSerial", ctypes.c_char * 13),   # 被冲正银行流水号
            ("FutureRepealSerial", ctypes.c_int),       # 被冲正期货流水号
            ("TradeCode", ctypes.c_char * 7),           # 业务功能码
            ("BankID", ctypes.c_char * 4),              # 银行代码
            ("BankBranchID", ctypes.c_char * 5),        # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),           # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),     # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),           # 交易日期
            ("TradeTime", ctypes.c_char * 9),           # 交易时间
            ("BankSerial", ctypes.c_char * 13),         # 银行流水号
            ("TradingDay", ctypes.c_char * 9),          # 交易系统日期
            ("PlateSerial", ctypes.c_int),              # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),            # 最后分片标志
            ("SessionID", ctypes.c_int),                # 会话号
            ("CustomerName", ctypes.c_char * 51),       # 客户姓名
            ("IdCardType", ctypes.c_char),              # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),   # 证件号码
            ("CustType", ctypes.c_char),                # 客户类型
            ("BankAccount", ctypes.c_char * 41),        # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),       # 银行密码
            ("AccountID", ctypes.c_char * 13),          # 投资者帐号
            ("Password", ctypes.c_char * 41),           # 期货密码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("FutureSerial", ctypes.c_int),             # 期货公司流水号
            ("UserID", ctypes.c_char * 16),             # 用户标识
            ("VerifyCertNoFlag", ctypes.c_char),        # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),          # 币种代码
            ("TradeAmount", ctypes.c_double),           # 转帐金额
            ("FutureFetchAmount", ctypes.c_double),     # 期货可取金额
            ("FeePayFlag", ctypes.c_char),              # 费用支付标志
            ("CustFee", ctypes.c_double),               # 应收客户费用
            ("BrokerFee", ctypes.c_double),             # 应收期货公司费用
            ("Message", ctypes.c_char * 129),           # 发送方给接收方的消息
            ("Digest", ctypes.c_char * 36),             # 摘要
            ("BankAccType", ctypes.c_char),             # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),            # 渠道标志
            ("BankSecuAccType", ctypes.c_char),         # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),     # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),        # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),             # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),             # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),             # 交易柜员
            ("RequestID", ctypes.c_int),                # 请求编号
            ("TID", ctypes.c_int),                      # 交易ID
            ("TransferStatus", ctypes.c_char),          # 转账交易状态
            ("ErrorID", ctypes.c_int),                  # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),           # 错误信息
        ]

    _capsule_name = "RspRepeal"

    _field_mappings = {
        "repeal_time_interval": "RepealTimeInterval",
        "repealed_times": "RepealedTimes",
        "bank_repeal_flag": "BankRepealFlag",
        "broker_repeal_flag": "BrokerRepealFlag",
        "plate_repeal_serial": "PlateRepealSerial",
        "bank_repeal_serial": "BankRepealSerial",
        "future_repeal_serial": "FutureRepealSerial",
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "cust_type": "CustType",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "install_id": "InstallID",
        "future_serial": "FutureSerial",
        "user_id": "UserID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "trade_amount": "TradeAmount",
        "future_fetch_amount": "FutureFetchAmount",
        "fee_pay_flag": "FeePayFlag",
        "cust_fee": "CustFee",
        "broker_fee": "BrokerFee",
        "message": "Message",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "request_id": "RequestID",
        "tid": "TID",
        "transfer_status": "TransferStatus",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
    }

    @property
    def repeal_time_interval(self) -> int:
        """冲正时间间隔"""
        return self._struct.RepealTimeInterval

    @repeal_time_interval.setter
    def repeal_time_interval(self, value: int):
        """设置冲正时间间隔"""
        self._struct.RepealTimeInterval = value

    @property
    def repealed_times(self) -> int:
        """已经冲正次数"""
        return self._struct.RepealedTimes

    @repealed_times.setter
    def repealed_times(self, value: int):
        """设置已经冲正次数"""
        self._struct.RepealedTimes = value

    @property
    def bank_repeal_flag(self) -> str:
        """银行冲正标志"""
        if 'bank_repeal_flag' not in self._cache:
            value = self._struct.BankRepealFlag.decode('ascii')
            self._cache['bank_repeal_flag'] = value
        return self._cache['bank_repeal_flag']

    @bank_repeal_flag.setter
    def bank_repeal_flag(self, value: str):
        """设置银行冲正标志"""
        self._struct.BankRepealFlag = value.encode('ascii')[0]
        self._cache['bank_repeal_flag'] = value

    @property
    def broker_repeal_flag(self) -> str:
        """期商冲正标志"""
        if 'broker_repeal_flag' not in self._cache:
            value = self._struct.BrokerRepealFlag.decode('ascii')
            self._cache['broker_repeal_flag'] = value
        return self._cache['broker_repeal_flag']

    @broker_repeal_flag.setter
    def broker_repeal_flag(self, value: str):
        """设置期商冲正标志"""
        self._struct.BrokerRepealFlag = value.encode('ascii')[0]
        self._cache['broker_repeal_flag'] = value

    @property
    def plate_repeal_serial(self) -> int:
        """被冲正平台流水号"""
        return self._struct.PlateRepealSerial

    @plate_repeal_serial.setter
    def plate_repeal_serial(self, value: int):
        """设置被冲正平台流水号"""
        self._struct.PlateRepealSerial = value

    @property
    def bank_repeal_serial(self) -> str:
        """被冲正银行流水号"""
        if 'bank_repeal_serial' not in self._cache:
            value = self._struct.BankRepealSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_repeal_serial'] = value
        return self._cache['bank_repeal_serial']

    @bank_repeal_serial.setter
    def bank_repeal_serial(self, value: str):
        """设置被冲正银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankRepealSerial = encoded
        self._cache['bank_repeal_serial'] = value

    @property
    def future_repeal_serial(self) -> int:
        """被冲正期货流水号"""
        return self._struct.FutureRepealSerial

    @future_repeal_serial.setter
    def future_repeal_serial(self, value: int):
        """设置被冲正期货流水号"""
        self._struct.FutureRepealSerial = value

    @property
    def trade_code(self) -> str:
        """业务功能码"""
        if 'trade_code' not in self._cache:
            value = self._struct.TradeCode.rstrip(b'\x00').decode('ascii')
            self._cache['trade_code'] = value
        return self._cache['trade_code']

    @trade_code.setter
    def trade_code(self, value: str):
        """设置业务功能码"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.TradeCode = encoded
        self._cache['trade_code'] = value

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
    def bank_branch_id(self) -> str:
        """银行分支机构代码"""
        if 'bank_branch_id' not in self._cache:
            value = self._struct.BankBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_branch_id'] = value
        return self._cache['bank_branch_id']

    @bank_branch_id.setter
    def bank_branch_id(self, value: str):
        """设置银行分支机构代码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBranchID = encoded
        self._cache['bank_branch_id'] = value

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期商分支机构代码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期商分支机构代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

    @property
    def trade_date(self) -> str:
        """交易日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trade_time(self) -> str:
        """交易时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置交易时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

    @property
    def bank_serial(self) -> str:
        """银行流水号"""
        if 'bank_serial' not in self._cache:
            value = self._struct.BankSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_serial'] = value
        return self._cache['bank_serial']

    @bank_serial.setter
    def bank_serial(self, value: str):
        """设置银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankSerial = encoded
        self._cache['bank_serial'] = value

    @property
    def trading_day(self) -> str:
        """交易系统日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易系统日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    @property
    def plate_serial(self) -> int:
        """银期平台消息流水号"""
        return self._struct.PlateSerial

    @plate_serial.setter
    def plate_serial(self, value: int):
        """设置银期平台消息流水号"""
        self._struct.PlateSerial = value

    @property
    def last_fragment(self) -> str:
        """最后分片标志"""
        if 'last_fragment' not in self._cache:
            value = self._struct.LastFragment.decode('ascii')
            self._cache['last_fragment'] = value
        return self._cache['last_fragment']

    @last_fragment.setter
    def last_fragment(self, value: str):
        """设置最后分片标志"""
        self._struct.LastFragment = value.encode('ascii')[0]
        self._cache['last_fragment'] = value

    @property
    def session_id(self) -> int:
        """会话号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话号"""
        self._struct.SessionID = value

    @property
    def customer_name(self) -> str:
        """客户姓名"""
        if 'customer_name' not in self._cache:
            value = self._struct.CustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['customer_name'] = value
        return self._cache['customer_name']

    @customer_name.setter
    def customer_name(self, value: str):
        """设置客户姓名"""
        encoded = value.encode('gbk')[:50].ljust(51, b'\x00')
        self._struct.CustomerName = encoded
        self._cache['customer_name'] = value

    @property
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def cust_type(self) -> str:
        """客户类型"""
        if 'cust_type' not in self._cache:
            value = self._struct.CustType.decode('ascii')
            self._cache['cust_type'] = value
        return self._cache['cust_type']

    @cust_type.setter
    def cust_type(self, value: str):
        """设置客户类型"""
        self._struct.CustType = value.encode('ascii')[0]
        self._cache['cust_type'] = value

    @property
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def bank_pass_word(self) -> str:
        """银行密码"""
        if 'bank_pass_word' not in self._cache:
            value = self._struct.BankPassWord.rstrip(b'\x00').decode('ascii')
            self._cache['bank_pass_word'] = value
        return self._cache['bank_pass_word']

    @bank_pass_word.setter
    def bank_pass_word(self, value: str):
        """设置银行密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankPassWord = encoded
        self._cache['bank_pass_word'] = value

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
    def password(self) -> str:
        """期货密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置期货密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def future_serial(self) -> int:
        """期货公司流水号"""
        return self._struct.FutureSerial

    @future_serial.setter
    def future_serial(self, value: int):
        """设置期货公司流水号"""
        self._struct.FutureSerial = value

    @property
    def user_id(self) -> str:
        """用户标识"""
        if 'user_id' not in self._cache:
            value = self._struct.UserID.rstrip(b'\x00').decode('ascii')
            self._cache['user_id'] = value
        return self._cache['user_id']

    @user_id.setter
    def user_id(self, value: str):
        """设置用户标识"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.UserID = encoded
        self._cache['user_id'] = value

    @property
    def verify_cert_no_flag(self) -> str:
        """验证客户证件号码标志"""
        if 'verify_cert_no_flag' not in self._cache:
            value = self._struct.VerifyCertNoFlag.decode('ascii')
            self._cache['verify_cert_no_flag'] = value
        return self._cache['verify_cert_no_flag']

    @verify_cert_no_flag.setter
    def verify_cert_no_flag(self, value: str):
        """设置验证客户证件号码标志"""
        self._struct.VerifyCertNoFlag = value.encode('ascii')[0]
        self._cache['verify_cert_no_flag'] = value

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
    def trade_amount(self) -> float:
        """转帐金额"""
        return self._struct.TradeAmount

    @trade_amount.setter
    def trade_amount(self, value: float):
        """设置转帐金额"""
        self._struct.TradeAmount = value

    @property
    def future_fetch_amount(self) -> float:
        """期货可取金额"""
        return self._struct.FutureFetchAmount

    @future_fetch_amount.setter
    def future_fetch_amount(self, value: float):
        """设置期货可取金额"""
        self._struct.FutureFetchAmount = value

    @property
    def fee_pay_flag(self) -> str:
        """费用支付标志"""
        if 'fee_pay_flag' not in self._cache:
            value = self._struct.FeePayFlag.decode('ascii')
            self._cache['fee_pay_flag'] = value
        return self._cache['fee_pay_flag']

    @fee_pay_flag.setter
    def fee_pay_flag(self, value: str):
        """设置费用支付标志"""
        self._struct.FeePayFlag = value.encode('ascii')[0]
        self._cache['fee_pay_flag'] = value

    @property
    def cust_fee(self) -> float:
        """应收客户费用"""
        return self._struct.CustFee

    @cust_fee.setter
    def cust_fee(self, value: float):
        """设置应收客户费用"""
        self._struct.CustFee = value

    @property
    def broker_fee(self) -> float:
        """应收期货公司费用"""
        return self._struct.BrokerFee

    @broker_fee.setter
    def broker_fee(self, value: float):
        """设置应收期货公司费用"""
        self._struct.BrokerFee = value

    @property
    def message(self) -> str:
        """发送方给接收方的消息"""
        if 'message' not in self._cache:
            value = self._struct.Message.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['message'] = value
        return self._cache['message']

    @message.setter
    def message(self, value: str):
        """设置发送方给接收方的消息"""
        encoded = value.encode('gbk')[:128].ljust(129, b'\x00')
        self._struct.Message = encoded
        self._cache['message'] = value

    @property
    def digest(self) -> str:
        """摘要"""
        if 'digest' not in self._cache:
            value = self._struct.Digest.rstrip(b'\x00').decode('ascii')
            self._cache['digest'] = value
        return self._cache['digest']

    @digest.setter
    def digest(self, value: str):
        """设置摘要"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.Digest = encoded
        self._cache['digest'] = value

    @property
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def device_id(self) -> str:
        """渠道标志"""
        if 'device_id' not in self._cache:
            value = self._struct.DeviceID.rstrip(b'\x00').decode('ascii')
            self._cache['device_id'] = value
        return self._cache['device_id']

    @device_id.setter
    def device_id(self, value: str):
        """设置渠道标志"""
        encoded = value.encode('ascii')[:2].ljust(3, b'\x00')
        self._struct.DeviceID = encoded
        self._cache['device_id'] = value

    @property
    def bank_secu_acc_type(self) -> str:
        """期货单位帐号类型"""
        if 'bank_secu_acc_type' not in self._cache:
            value = self._struct.BankSecuAccType.decode('ascii')
            self._cache['bank_secu_acc_type'] = value
        return self._cache['bank_secu_acc_type']

    @bank_secu_acc_type.setter
    def bank_secu_acc_type(self, value: str):
        """设置期货单位帐号类型"""
        self._struct.BankSecuAccType = value.encode('ascii')[0]
        self._cache['bank_secu_acc_type'] = value

    @property
    def broker_id_by_bank(self) -> str:
        """期货公司银行编码"""
        if 'broker_id_by_bank' not in self._cache:
            value = self._struct.BrokerIDByBank.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id_by_bank'] = value
        return self._cache['broker_id_by_bank']

    @broker_id_by_bank.setter
    def broker_id_by_bank(self, value: str):
        """设置期货公司银行编码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.BrokerIDByBank = encoded
        self._cache['broker_id_by_bank'] = value

    @property
    def bank_secu_acc(self) -> str:
        """期货单位帐号"""
        if 'bank_secu_acc' not in self._cache:
            value = self._struct.BankSecuAcc.rstrip(b'\x00').decode('ascii')
            self._cache['bank_secu_acc'] = value
        return self._cache['bank_secu_acc']

    @bank_secu_acc.setter
    def bank_secu_acc(self, value: str):
        """设置期货单位帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankSecuAcc = encoded
        self._cache['bank_secu_acc'] = value

    @property
    def bank_pwd_flag(self) -> str:
        """银行密码标志"""
        if 'bank_pwd_flag' not in self._cache:
            value = self._struct.BankPwdFlag.decode('ascii')
            self._cache['bank_pwd_flag'] = value
        return self._cache['bank_pwd_flag']

    @bank_pwd_flag.setter
    def bank_pwd_flag(self, value: str):
        """设置银行密码标志"""
        self._struct.BankPwdFlag = value.encode('ascii')[0]
        self._cache['bank_pwd_flag'] = value

    @property
    def secu_pwd_flag(self) -> str:
        """期货资金密码核对标志"""
        if 'secu_pwd_flag' not in self._cache:
            value = self._struct.SecuPwdFlag.decode('ascii')
            self._cache['secu_pwd_flag'] = value
        return self._cache['secu_pwd_flag']

    @secu_pwd_flag.setter
    def secu_pwd_flag(self, value: str):
        """设置期货资金密码核对标志"""
        self._struct.SecuPwdFlag = value.encode('ascii')[0]
        self._cache['secu_pwd_flag'] = value

    @property
    def oper_no(self) -> str:
        """交易柜员"""
        if 'oper_no' not in self._cache:
            value = self._struct.OperNo.rstrip(b'\x00').decode('ascii')
            self._cache['oper_no'] = value
        return self._cache['oper_no']

    @oper_no.setter
    def oper_no(self, value: str):
        """设置交易柜员"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.OperNo = encoded
        self._cache['oper_no'] = value

    @property
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def transfer_status(self) -> str:
        """转账交易状态"""
        if 'transfer_status' not in self._cache:
            value = self._struct.TransferStatus.decode('ascii')
            self._cache['transfer_status'] = value
        return self._cache['transfer_status']

    @transfer_status.setter
    def transfer_status(self, value: str):
        """设置转账交易状态"""
        self._struct.TransferStatus = value.encode('ascii')[0]
        self._cache['transfer_status'] = value

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
        """错误信息"""
        if 'error_msg' not in self._cache:
            value = self._struct.ErrorMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['error_msg'] = value
        return self._cache['error_msg']

    @error_msg.setter
    def error_msg(self, value: str):
        """设置错误信息"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.ErrorMsg = encoded
        self._cache['error_msg'] = value



class ReqRepeal(CapsuleStruct):
    """冲正请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("RepealTimeInterval", ctypes.c_int),       # 冲正时间间隔
            ("RepealedTimes", ctypes.c_int),            # 已经冲正次数
            ("BankRepealFlag", ctypes.c_char),          # 银行冲正标志
            ("BrokerRepealFlag", ctypes.c_char),        # 期商冲正标志
            ("PlateRepealSerial", ctypes.c_int),        # 被冲正平台流水号
            ("BankRepealSerial", ctypes.c_char * 13),   # 被冲正银行流水号
            ("FutureRepealSerial", ctypes.c_int),       # 被冲正期货流水号
            ("TradeCode", ctypes.c_char * 7),           # 业务功能码
            ("BankID", ctypes.c_char * 4),              # 银行代码
            ("BankBranchID", ctypes.c_char * 5),        # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),           # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),     # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),           # 交易日期
            ("TradeTime", ctypes.c_char * 9),           # 交易时间
            ("BankSerial", ctypes.c_char * 13),         # 银行流水号
            ("TradingDay", ctypes.c_char * 9),          # 交易系统日期
            ("PlateSerial", ctypes.c_int),              # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),            # 最后分片标志
            ("SessionID", ctypes.c_int),                # 会话号
            ("CustomerName", ctypes.c_char * 51),       # 客户姓名
            ("IdCardType", ctypes.c_char),              # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),   # 证件号码
            ("CustType", ctypes.c_char),                # 客户类型
            ("BankAccount", ctypes.c_char * 41),        # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),       # 银行密码
            ("AccountID", ctypes.c_char * 13),          # 投资者帐号
            ("Password", ctypes.c_char * 41),           # 期货密码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("FutureSerial", ctypes.c_int),             # 期货公司流水号
            ("UserID", ctypes.c_char * 16),             # 用户标识
            ("VerifyCertNoFlag", ctypes.c_char),        # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),          # 币种代码
            ("TradeAmount", ctypes.c_double),           # 转帐金额
            ("FutureFetchAmount", ctypes.c_double),     # 期货可取金额
            ("FeePayFlag", ctypes.c_char),              # 费用支付标志
            ("CustFee", ctypes.c_double),               # 应收客户费用
            ("BrokerFee", ctypes.c_double),             # 应收期货公司费用
            ("Message", ctypes.c_char * 129),           # 发送方给接收方的消息
            ("Digest", ctypes.c_char * 36),             # 摘要
            ("BankAccType", ctypes.c_char),             # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),            # 渠道标志
            ("BankSecuAccType", ctypes.c_char),         # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),     # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),        # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),             # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),             # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),             # 交易柜员
            ("RequestID", ctypes.c_int),                # 请求编号
            ("TID", ctypes.c_int),                      # 交易ID
            ("TransferStatus", ctypes.c_char),          # 转账交易状态
        ]

    _capsule_name = "ReqRepeal"

    _field_mappings = {
        "repeal_time_interval": "RepealTimeInterval",
        "repealed_times": "RepealedTimes",
        "bank_repeal_flag": "BankRepealFlag",
        "broker_repeal_flag": "BrokerRepealFlag",
        "plate_repeal_serial": "PlateRepealSerial",
        "bank_repeal_serial": "BankRepealSerial",
        "future_repeal_serial": "FutureRepealSerial",
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "cust_type": "CustType",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "install_id": "InstallID",
        "future_serial": "FutureSerial",
        "user_id": "UserID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "trade_amount": "TradeAmount",
        "future_fetch_amount": "FutureFetchAmount",
        "fee_pay_flag": "FeePayFlag",
        "cust_fee": "CustFee",
        "broker_fee": "BrokerFee",
        "message": "Message",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "request_id": "RequestID",
        "tid": "TID",
        "transfer_status": "TransferStatus",
    }

    # ReqRepeal 的属性方法与 RspRepeal 相同（除了缺少 ErrorID 和 ErrorMsg）
    # 为了简化代码，这里只实现部分常用属性

    @property
    def repeal_time_interval(self) -> int:
        """冲正时间间隔"""
        return self._struct.RepealTimeInterval

    @repeal_time_interval.setter
    def repeal_time_interval(self, value: int):
        """设置冲正时间间隔"""
        self._struct.RepealTimeInterval = value

    @property
    def plate_repeal_serial(self) -> int:
        """被冲正平台流水号"""
        return self._struct.PlateRepealSerial

    @plate_repeal_serial.setter
    def plate_repeal_serial(self, value: int):
        """设置被冲正平台流水号"""
        self._struct.PlateRepealSerial = value

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
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



class ReqQueryAccount(CapsuleStruct):
    """查询账户请求"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeCode", ctypes.c_char * 7),           # 业务功能码
            ("BankID", ctypes.c_char * 4),              # 银行代码
            ("BankBranchID", ctypes.c_char * 5),        # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),           # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),     # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),           # 交易日期
            ("TradeTime", ctypes.c_char * 9),           # 交易时间
            ("BankSerial", ctypes.c_char * 13),         # 银行流水号
            ("TradingDay", ctypes.c_char * 9),          # 交易系统日期
            ("PlateSerial", ctypes.c_int),              # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),            # 最后分片标志
            ("SessionID", ctypes.c_int),                # 会话号
            ("CustomerName", ctypes.c_char * 51),       # 客户姓名
            ("IdCardType", ctypes.c_char),              # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),   # 证件号码
            ("CustType", ctypes.c_char),                # 客户类型
            ("BankAccount", ctypes.c_char * 41),        # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),       # 银行密码
            ("AccountID", ctypes.c_char * 13),          # 投资者帐号
            ("Password", ctypes.c_char * 41),           # 期货密码
            ("FutureSerial", ctypes.c_int),             # 期货公司流水号
            ("InstallID", ctypes.c_int),                # 安装编号
            ("UserID", ctypes.c_char * 16),             # 用户标识
            ("VerifyCertNoFlag", ctypes.c_char),        # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),          # 币种代码
            ("Digest", ctypes.c_char * 36),             # 摘要
            ("BankAccType", ctypes.c_char),             # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),            # 渠道标志
            ("BankSecuAccType", ctypes.c_char),         # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),     # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),        # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),             # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),             # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),             # 交易柜员
            ("RequestID", ctypes.c_int),                # 请求编号
            ("TID", ctypes.c_int),                      # 交易ID
            ("LongCustomerName", ctypes.c_char * 161),  # 长客户姓名
        ]

    _capsule_name = "ReqQueryAccount"

    _field_mappings = {
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "cust_type": "CustType",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "future_serial": "FutureSerial",
        "install_id": "InstallID",
        "user_id": "UserID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "request_id": "RequestID",
        "tid": "TID",
        "long_customer_name": "LongCustomerName",
    }

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
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
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

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
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def long_customer_name(self) -> str:
        """长客户姓名"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名"""
        encoded = value.encode('gbk')[:160].ljust(161, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class NotifyQueryAccount(CapsuleStruct):
    """查询账户通知"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeCode", ctypes.c_char * 7),           # 业务功能码
            ("BankID", ctypes.c_char * 4),              # 银行代码
            ("BankBranchID", ctypes.c_char * 5),        # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),           # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),     # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),           # 交易日期
            ("TradeTime", ctypes.c_char * 9),           # 交易时间
            ("BankSerial", ctypes.c_char * 13),         # 银行流水号
            ("TradingDay", ctypes.c_char * 9),          # 交易系统日期
            ("PlateSerial", ctypes.c_int),              # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),            # 最后分片标志
            ("SessionID", ctypes.c_int),                # 会话号
            ("CustomerName", ctypes.c_char * 51),       # 客户姓名
            ("IdCardType", ctypes.c_char),              # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),   # 证件号码
            ("CustType", ctypes.c_char),                # 客户类型
            ("BankAccount", ctypes.c_char * 41),        # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),       # 银行密码
            ("AccountID", ctypes.c_char * 13),          # 投资者帐号
            ("Password", ctypes.c_char * 41),           # 期货密码
            ("FutureSerial", ctypes.c_int),             # 期货公司流水号
            ("InstallID", ctypes.c_int),                # 安装编号
            ("UserID", ctypes.c_char * 16),             # 用户标识
            ("VerifyCertNoFlag", ctypes.c_char),        # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),          # 币种代码
            ("Digest", ctypes.c_char * 36),             # 摘要
            ("BankAccType", ctypes.c_char),             # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),            # 渠道标志
            ("BankSecuAccType", ctypes.c_char),         # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),     # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),        # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),             # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),             # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),             # 交易柜员
            ("RequestID", ctypes.c_int),                # 请求编号
            ("TID", ctypes.c_int),                      # 交易ID
            ("BankUseAmount", ctypes.c_double),         # 银行可用金额
            ("BankFetchAmount", ctypes.c_double),       # 银行可取金额
            ("ErrorID", ctypes.c_int),                  # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),           # 错误信息
            ("LongCustomerName", ctypes.c_char * 161),  # 长客户姓名
        ]

    _capsule_name = "NotifyQueryAccount"

    _field_mappings = {
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "cust_type": "CustType",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "future_serial": "FutureSerial",
        "install_id": "InstallID",
        "user_id": "UserID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "request_id": "RequestID",
        "tid": "TID",
        "bank_use_amount": "BankUseAmount",
        "bank_fetch_amount": "BankFetchAmount",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
        "long_customer_name": "LongCustomerName",
    }

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
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
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

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
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def bank_use_amount(self) -> float:
        """银行可用金额"""
        return self._struct.BankUseAmount

    @bank_use_amount.setter
    def bank_use_amount(self, value: float):
        """设置银行可用金额"""
        self._struct.BankUseAmount = value

    @property
    def bank_fetch_amount(self) -> float:
        """银行可取金额"""
        return self._struct.BankFetchAmount

    @bank_fetch_amount.setter
    def bank_fetch_amount(self, value: float):
        """设置银行可取金额"""
        self._struct.BankFetchAmount = value

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
        """错误信息"""
        if 'error_msg' not in self._cache:
            value = self._struct.ErrorMsg.rstrip(b'\x00').decode('gbk')
            self._cache['error_msg'] = value
        return self._cache['error_msg']

    @error_msg.setter
    def error_msg(self, value: str):
        """设置错误信息"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.ErrorMsg = encoded
        self._cache['error_msg'] = value

    @property
    def long_customer_name(self) -> str:
        """长客户姓名"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名"""
        encoded = value.encode('gbk')[:160].ljust(161, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class AccountRegister(CapsuleStruct):
    """账户注册"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeDay", ctypes.c_char * 9),            # 交易日期
            ("BankID", ctypes.c_char * 4),              # 银行编码
            ("BankBranchID", ctypes.c_char * 5),        # 银行分支机构编码
            ("BankAccount", ctypes.c_char * 41),        # 银行帐号
            ("BrokerID", ctypes.c_char * 11),           # 期货公司编码
            ("BrokerBranchID", ctypes.c_char * 31),     # 期货公司分支机构编码
            ("AccountID", ctypes.c_char * 13),          # 投资者帐号
            ("IdCardType", ctypes.c_char),              # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),   # 证件号码
            ("CustomerName", ctypes.c_char * 51),       # 客户姓名
            ("CurrencyID", ctypes.c_char * 4),          # 币种代码
            ("OpenOrDestroy", ctypes.c_char),           # 开销户类别
            ("RegDate", ctypes.c_char * 9),             # 签约日期
            ("OutDate", ctypes.c_char * 9),             # 解约日期
            ("TID", ctypes.c_int),                      # 交易ID
            ("CustType", ctypes.c_char),                # 客户类型
            ("BankAccType", ctypes.c_char),             # 银行帐号类型
            ("LongCustomerName", ctypes.c_char * 161),  # 长客户姓名
        ]

    _capsule_name = "AccountRegister"

    _field_mappings = {
        "trade_day": "TradeDay",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "bank_account": "BankAccount",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "account_id": "AccountID",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "customer_name": "CustomerName",
        "currency_id": "CurrencyID",
        "open_or_destroy": "OpenOrDestroy",
        "reg_date": "RegDate",
        "out_date": "OutDate",
        "tid": "TID",
        "cust_type": "CustType",
        "bank_acc_type": "BankAccType",
        "long_customer_name": "LongCustomerName",
    }

    @property
    def trade_day(self) -> str:
        """交易日期"""
        if 'trade_day' not in self._cache:
            value = self._struct.TradeDay.rstrip(b'\x00').decode('ascii')
            self._cache['trade_day'] = value
        return self._cache['trade_day']

    @trade_day.setter
    def trade_day(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDay = encoded
        self._cache['trade_day'] = value

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
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def broker_id(self) -> str:
        """期货公司编码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期货公司编码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期货公司分支机构编码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期货公司分支机构编码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

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
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def customer_name(self) -> str:
        """客户姓名"""
        if 'customer_name' not in self._cache:
            value = self._struct.CustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['customer_name'] = value
        return self._cache['customer_name']

    @customer_name.setter
    def customer_name(self, value: str):
        """设置客户姓名"""
        encoded = value.encode('gbk')[:50].ljust(51, b'\x00')
        self._struct.CustomerName = encoded
        self._cache['customer_name'] = value

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
    def open_or_destroy(self) -> str:
        """开销户类别"""
        if 'open_or_destroy' not in self._cache:
            value = self._struct.OpenOrDestroy.decode('ascii')
            self._cache['open_or_destroy'] = value
        return self._cache['open_or_destroy']

    @open_or_destroy.setter
    def open_or_destroy(self, value: str):
        """设置开销户类别"""
        self._struct.OpenOrDestroy = value.encode('ascii')[0]
        self._cache['open_or_destroy'] = value

    @property
    def reg_date(self) -> str:
        """签约日期"""
        if 'reg_date' not in self._cache:
            value = self._struct.RegDate.rstrip(b'\x00').decode('ascii')
            self._cache['reg_date'] = value
        return self._cache['reg_date']

    @reg_date.setter
    def reg_date(self, value: str):
        """设置签约日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.RegDate = encoded
        self._cache['reg_date'] = value

    @property
    def out_date(self) -> str:
        """解约日期"""
        if 'out_date' not in self._cache:
            value = self._struct.OutDate.rstrip(b'\x00').decode('ascii')
            self._cache['out_date'] = value
        return self._cache['out_date']

    @out_date.setter
    def out_date(self, value: str):
        """设置解约日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.OutDate = encoded
        self._cache['out_date'] = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def cust_type(self) -> str:
        """客户类型"""
        if 'cust_type' not in self._cache:
            value = self._struct.CustType.decode('ascii')
            self._cache['cust_type'] = value
        return self._cache['cust_type']

    @cust_type.setter
    def cust_type(self, value: str):
        """设置客户类型"""
        self._struct.CustType = value.encode('ascii')[0]
        self._cache['cust_type'] = value

    @property
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def long_customer_name(self) -> str:
        """长客户姓名"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名"""
        encoded = value.encode('gbk')[:160].ljust(161, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class OffsetSetting(CapsuleStruct):
    """对冲设置"""

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
            ("ExchangeInstID", ctypes.c_char * 81),     # 交易所合约代码
            ("ExchangeSerialNo", ctypes.c_char * 81),   # 交易所期权系列号
            ("ExchangeProductID", ctypes.c_char * 41),  # 交易所产品代码
            ("ParticipantID", ctypes.c_char * 11),      # 会员代码
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("TraderID", ctypes.c_char * 21),           # 交易所交易员代码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("OrderSubmitStatus", ctypes.c_char),       # 对冲提交状态
            ("TradingDay", ctypes.c_char * 9),          # 交易日
            ("SettlementID", ctypes.c_int),             # 结算编号
            ("InsertDate", ctypes.c_char * 9),          # 报单日期
            ("InsertTime", ctypes.c_char * 9),          # 插入时间
            ("CancelTime", ctypes.c_char * 9),          # 撤销时间
            ("ExecResult", ctypes.c_char),              # 对冲设置结果
            ("SequenceNo", ctypes.c_int),               # 序号
            ("FrontID", ctypes.c_int),                  # 前置编号
            ("SessionID", ctypes.c_int),                # 会话编号
            ("StatusMsg", ctypes.c_char * 81),          # 状态信息
            ("ActiveUserID", ctypes.c_char * 16),       # 操作用户代码
            ("BrokerOffsetSettingSeq", ctypes.c_int),   # 经纪公司报单编号
            ("ApplySrc", ctypes.c_char),                # 申请来源
        ]

    _capsule_name = "OffsetSetting"

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
        "exchange_inst_id": "ExchangeInstID",
        "exchange_serial_no": "ExchangeSerialNo",
        "exchange_product_id": "ExchangeProductID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "order_submit_status": "OrderSubmitStatus",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "insert_date": "InsertDate",
        "insert_time": "InsertTime",
        "cancel_time": "CancelTime",
        "exec_result": "ExecResult",
        "sequence_no": "SequenceNo",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "status_msg": "StatusMsg",
        "active_user_id": "ActiveUserID",
        "broker_offset_setting_seq": "BrokerOffsetSettingSeq",
        "apply_src": "ApplySrc",
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

    @property
    def exchange_inst_id(self) -> str:
        """交易所合约代码"""
        if 'exchange_inst_id' not in self._cache:
            value = self._struct.ExchangeInstID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_inst_id'] = value
        return self._cache['exchange_inst_id']

    @exchange_inst_id.setter
    def exchange_inst_id(self, value: str):
        """设置交易所合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ExchangeInstID = encoded
        self._cache['exchange_inst_id'] = value

    @property
    def exchange_serial_no(self) -> str:
        """交易所期权系列号"""
        if 'exchange_serial_no' not in self._cache:
            value = self._struct.ExchangeSerialNo.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_serial_no'] = value
        return self._cache['exchange_serial_no']

    @exchange_serial_no.setter
    def exchange_serial_no(self, value: str):
        """设置交易所期权系列号"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ExchangeSerialNo = encoded
        self._cache['exchange_serial_no'] = value

    @property
    def exchange_product_id(self) -> str:
        """交易所产品代码"""
        if 'exchange_product_id' not in self._cache:
            value = self._struct.ExchangeProductID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_product_id'] = value
        return self._cache['exchange_product_id']

    @exchange_product_id.setter
    def exchange_product_id(self, value: str):
        """设置交易所产品代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ExchangeProductID = encoded
        self._cache['exchange_product_id'] = value

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
        """对冲提交状态"""
        if 'order_submit_status' not in self._cache:
            value = self._struct.OrderSubmitStatus.decode('ascii')
            self._cache['order_submit_status'] = value
        return self._cache['order_submit_status']

    @order_submit_status.setter
    def order_submit_status(self, value: str):
        """设置对冲提交状态"""
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
    def exec_result(self) -> str:
        """对冲设置结果"""
        if 'exec_result' not in self._cache:
            value = self._struct.ExecResult.decode('ascii')
            self._cache['exec_result'] = value
        return self._cache['exec_result']

    @exec_result.setter
    def exec_result(self, value: str):
        """设置对冲设置结果"""
        self._struct.ExecResult = value.encode('ascii')[0]
        self._cache['exec_result'] = value

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
    def status_msg(self) -> str:
        """状态信息"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息"""
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
    def broker_offset_setting_seq(self) -> int:
        """经纪公司报单编号"""
        return self._struct.BrokerOffsetSettingSeq

    @broker_offset_setting_seq.setter
    def broker_offset_setting_seq(self, value: int):
        """设置经纪公司报单编号"""
        self._struct.BrokerOffsetSettingSeq = value

    @property
    def apply_src(self) -> str:
        """申请来源"""
        if 'apply_src' not in self._cache:
            value = self._struct.ApplySrc.decode('ascii')
            self._cache['apply_src'] = value
        return self._cache['apply_src']

    @apply_src.setter
    def apply_src(self, value: str):
        """设置申请来源"""
        self._struct.ApplySrc = value.encode('ascii')[0]
        self._cache['apply_src'] = value



class CancelOffsetSetting(CapsuleStruct):
    """取消对冲设置"""

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
            ("ExchangeInstID", ctypes.c_char * 81),     # 交易所合约代码
            ("ExchangeSerialNo", ctypes.c_char * 81),   # 交易所期权系列号
            ("ExchangeProductID", ctypes.c_char * 41),  # 交易所产品代码
            ("TraderID", ctypes.c_char * 21),           # 交易所交易员代码
            ("InstallID", ctypes.c_int),                # 安装编号
            ("ParticipantID", ctypes.c_char * 11),      # 会员代码
            ("ClientID", ctypes.c_char * 11),           # 客户代码
            ("OrderActionStatus", ctypes.c_char),       # 报单操作状态
            ("StatusMsg", ctypes.c_char * 81),          # 状态信息
            ("ActionLocalID", ctypes.c_char * 13),      # 操作本地编号
            ("ActionDate", ctypes.c_char * 9),          # 操作日期
            ("ActionTime", ctypes.c_char * 9),          # 操作时间
        ]

    _capsule_name = "CancelOffsetSetting"

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
        "exchange_inst_id": "ExchangeInstID",
        "exchange_serial_no": "ExchangeSerialNo",
        "exchange_product_id": "ExchangeProductID",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "order_action_status": "OrderActionStatus",
        "status_msg": "StatusMsg",
        "action_local_id": "ActionLocalID",
        "action_date": "ActionDate",
        "action_time": "ActionTime",
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
    def request_id(self) -> int:
        """请求编号"""
        return self._struct.RequestID

    @request_id.setter
    def request_id(self, value: int):
        """设置请求编号"""
        self._struct.RequestID = value

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
    def status_msg(self) -> str:
        """状态信息"""
        if 'status_msg' not in self._cache:
            value = self._struct.StatusMsg.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['status_msg'] = value
        return self._cache['status_msg']

    @status_msg.setter
    def status_msg(self, value: str):
        """设置状态信息"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.StatusMsg = encoded
        self._cache['status_msg'] = value



class OpenAccount(CapsuleStruct):
    """银期开户信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeCode", ctypes.c_char * 7),              # 业务功能码
            ("BankID", ctypes.c_char * 4),                 # 银行代码
            ("BankBranchID", ctypes.c_char * 5),           # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),              # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),        # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),              # 交易日期
            ("TradeTime", ctypes.c_char * 9),              # 交易时间
            ("BankSerial", ctypes.c_char * 13),            # 银行流水号
            ("TradingDay", ctypes.c_char * 9),             # 交易系统日期
            ("PlateSerial", ctypes.c_int),                 # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),               # 最后分片标志
            ("SessionID", ctypes.c_int),                   # 会话号
            ("CustomerName", ctypes.c_char * 51),          # 客户姓名
            ("IdCardType", ctypes.c_char),                 # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),      # 证件号码
            ("Gender", ctypes.c_char),                     # 性别
            ("CountryCode", ctypes.c_char * 21),           # 国家代码
            ("CustType", ctypes.c_char),                   # 客户类型
            ("Address", ctypes.c_char * 101),              # 地址
            ("ZipCode", ctypes.c_char * 7),                # 邮编
            ("Telephone", ctypes.c_char * 41),             # 电话号码
            ("MobilePhone", ctypes.c_char * 41),           # 手机
            ("Fax", ctypes.c_char * 41),                   # 传真
            ("EMail", ctypes.c_char * 51),                 # 电子邮件
            ("MoneyAccountStatus", ctypes.c_char),         # 资金账户状态
            ("BankAccount", ctypes.c_char * 41),           # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),          # 银行密码
            ("AccountID", ctypes.c_char * 13),             # 投资者帐号
            ("Password", ctypes.c_char * 41),              # 期货密码
            ("InstallID", ctypes.c_int),                   # 安装编号
            ("VerifyCertNoFlag", ctypes.c_char),           # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),             # 币种代码
            ("CashExchangeCode", ctypes.c_char),           # 汇钞标志
            ("Digest", ctypes.c_char * 36),                # 摘要
            ("BankAccType", ctypes.c_char),                # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),               # 渠道标志
            ("BankSecuAccType", ctypes.c_char),            # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),        # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),           # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),                # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),                # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),                # 交易柜员
            ("TID", ctypes.c_int),                         # 交易ID
            ("UserID", ctypes.c_char * 16),                # 用户标识
            ("ErrorID", ctypes.c_int),                     # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),              # 错误信息
            ("LongCustomerName", ctypes.c_char * 161),     # 长客户姓名
        ]

    _capsule_name = "OpenAccount"

    _field_mappings = {
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "gender": "Gender",
        "country_code": "CountryCode",
        "cust_type": "CustType",
        "address": "Address",
        "zip_code": "ZipCode",
        "telephone": "Telephone",
        "mobile_phone": "MobilePhone",
        "fax": "Fax",
        "email": "EMail",
        "money_account_status": "MoneyAccountStatus",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "install_id": "InstallID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "cash_exchange_code": "CashExchangeCode",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "tid": "TID",
        "user_id": "UserID",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
        "long_customer_name": "LongCustomerName",
    }

    # 字符串字段 (ascii)
    @property
    def trade_code(self) -> str:
        """业务功能码"""
        if 'trade_code' not in self._cache:
            value = self._struct.TradeCode.rstrip(b'\x00').decode('ascii')
            self._cache['trade_code'] = value
        return self._cache['trade_code']

    @trade_code.setter
    def trade_code(self, value: str):
        """设置业务功能码"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.TradeCode = encoded
        self._cache['trade_code'] = value

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
    def bank_branch_id(self) -> str:
        """银行分支机构代码"""
        if 'bank_branch_id' not in self._cache:
            value = self._struct.BankBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_branch_id'] = value
        return self._cache['bank_branch_id']

    @bank_branch_id.setter
    def bank_branch_id(self, value: str):
        """设置银行分支机构代码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBranchID = encoded
        self._cache['bank_branch_id'] = value

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期商分支机构代码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期商分支机构代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

    @property
    def trade_date(self) -> str:
        """交易日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trade_time(self) -> str:
        """交易时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置交易时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

    @property
    def bank_serial(self) -> str:
        """银行流水号"""
        if 'bank_serial' not in self._cache:
            value = self._struct.BankSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_serial'] = value
        return self._cache['bank_serial']

    @bank_serial.setter
    def bank_serial(self, value: str):
        """设置银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankSerial = encoded
        self._cache['bank_serial'] = value

    @property
    def trading_day(self) -> str:
        """交易系统日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易系统日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    # 整数字段
    @property
    def plate_serial(self) -> int:
        """银期平台消息流水号"""
        return self._struct.PlateSerial

    @plate_serial.setter
    def plate_serial(self, value: int):
        """设置银期平台消息流水号"""
        self._struct.PlateSerial = value

    # 单字符字段
    @property
    def last_fragment(self) -> str:
        """最后分片标志"""
        if 'last_fragment' not in self._cache:
            value = self._struct.LastFragment.decode('ascii')
            self._cache['last_fragment'] = value
        return self._cache['last_fragment']

    @last_fragment.setter
    def last_fragment(self, value: str):
        """设置最后分片标志"""
        self._struct.LastFragment = value.encode('ascii')[0]
        self._cache['last_fragment'] = value

    @property
    def session_id(self) -> int:
        """会话号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话号"""
        self._struct.SessionID = value

    # GBK 编码字段
    @property
    def customer_name(self) -> str:
        """客户姓名（GBK 编码）"""
        if 'customer_name' not in self._cache:
            value = self._struct.CustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['customer_name'] = value
        return self._cache['customer_name']

    @customer_name.setter
    def customer_name(self, value: str):
        """设置客户姓名（GBK 编码）"""
        encoded = value.encode('gbk')[:50].ljust(51, b'\x00')
        self._struct.CustomerName = encoded
        self._cache['customer_name'] = value

    @property
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def gender(self) -> str:
        """性别"""
        if 'gender' not in self._cache:
            value = self._struct.Gender.decode('ascii')
            self._cache['gender'] = value
        return self._cache['gender']

    @gender.setter
    def gender(self, value: str):
        """设置性别"""
        self._struct.Gender = value.encode('ascii')[0]
        self._cache['gender'] = value

    @property
    def country_code(self) -> str:
        """国家代码"""
        if 'country_code' not in self._cache:
            value = self._struct.CountryCode.rstrip(b'\x00').decode('ascii')
            self._cache['country_code'] = value
        return self._cache['country_code']

    @country_code.setter
    def country_code(self, value: str):
        """设置国家代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.CountryCode = encoded
        self._cache['country_code'] = value

    @property
    def cust_type(self) -> str:
        """客户类型"""
        if 'cust_type' not in self._cache:
            value = self._struct.CustType.decode('ascii')
            self._cache['cust_type'] = value
        return self._cache['cust_type']

    @cust_type.setter
    def cust_type(self, value: str):
        """设置客户类型"""
        self._struct.CustType = value.encode('ascii')[0]
        self._cache['cust_type'] = value

    @property
    def address(self) -> str:
        """地址（GBK 编码）"""
        if 'address' not in self._cache:
            value = self._struct.Address.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['address'] = value
        return self._cache['address']

    @address.setter
    def address(self, value: str):
        """设置地址（GBK 编码）"""
        encoded = value.encode('gbk')[:100].ljust(101, b'\x00')
        self._struct.Address = encoded
        self._cache['address'] = value

    @property
    def zip_code(self) -> str:
        """邮编"""
        if 'zip_code' not in self._cache:
            value = self._struct.ZipCode.rstrip(b'\x00').decode('ascii')
            self._cache['zip_code'] = value
        return self._cache['zip_code']

    @zip_code.setter
    def zip_code(self, value: str):
        """设置邮编"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.ZipCode = encoded
        self._cache['zip_code'] = value

    @property
    def telephone(self) -> str:
        """电话号码"""
        if 'telephone' not in self._cache:
            value = self._struct.Telephone.rstrip(b'\x00').decode('ascii')
            self._cache['telephone'] = value
        return self._cache['telephone']

    @telephone.setter
    def telephone(self, value: str):
        """设置电话号码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Telephone = encoded
        self._cache['telephone'] = value

    @property
    def mobile_phone(self) -> str:
        """手机"""
        if 'mobile_phone' not in self._cache:
            value = self._struct.MobilePhone.rstrip(b'\x00').decode('ascii')
            self._cache['mobile_phone'] = value
        return self._cache['mobile_phone']

    @mobile_phone.setter
    def mobile_phone(self, value: str):
        """设置手机"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.MobilePhone = encoded
        self._cache['mobile_phone'] = value

    @property
    def fax(self) -> str:
        """传真"""
        if 'fax' not in self._cache:
            value = self._struct.Fax.rstrip(b'\x00').decode('ascii')
            self._cache['fax'] = value
        return self._cache['fax']

    @fax.setter
    def fax(self, value: str):
        """设置传真"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Fax = encoded
        self._cache['fax'] = value

    @property
    def email(self) -> str:
        """电子邮件"""
        if 'email' not in self._cache:
            value = self._struct.EMail.rstrip(b'\x00').decode('ascii')
            self._cache['email'] = value
        return self._cache['email']

    @email.setter
    def email(self, value: str):
        """设置电子邮件"""
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.EMail = encoded
        self._cache['email'] = value

    @property
    def money_account_status(self) -> str:
        """资金账户状态"""
        if 'money_account_status' not in self._cache:
            value = self._struct.MoneyAccountStatus.decode('ascii')
            self._cache['money_account_status'] = value
        return self._cache['money_account_status']

    @money_account_status.setter
    def money_account_status(self, value: str):
        """设置资金账户状态"""
        self._struct.MoneyAccountStatus = value.encode('ascii')[0]
        self._cache['money_account_status'] = value

    @property
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def bank_pass_word(self) -> str:
        """银行密码"""
        if 'bank_pass_word' not in self._cache:
            value = self._struct.BankPassWord.rstrip(b'\x00').decode('ascii')
            self._cache['bank_pass_word'] = value
        return self._cache['bank_pass_word']

    @bank_pass_word.setter
    def bank_pass_word(self, value: str):
        """设置银行密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankPassWord = encoded
        self._cache['bank_pass_word'] = value

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
    def password(self) -> str:
        """期货密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置期货密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def verify_cert_no_flag(self) -> str:
        """验证客户证件号码标志"""
        if 'verify_cert_no_flag' not in self._cache:
            value = self._struct.VerifyCertNoFlag.decode('ascii')
            self._cache['verify_cert_no_flag'] = value
        return self._cache['verify_cert_no_flag']

    @verify_cert_no_flag.setter
    def verify_cert_no_flag(self, value: str):
        """设置验证客户证件号码标志"""
        self._struct.VerifyCertNoFlag = value.encode('ascii')[0]
        self._cache['verify_cert_no_flag'] = value

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
    def cash_exchange_code(self) -> str:
        """汇钞标志"""
        if 'cash_exchange_code' not in self._cache:
            value = self._struct.CashExchangeCode.decode('ascii')
            self._cache['cash_exchange_code'] = value
        return self._cache['cash_exchange_code']

    @cash_exchange_code.setter
    def cash_exchange_code(self, value: str):
        """设置汇钞标志"""
        self._struct.CashExchangeCode = value.encode('ascii')[0]
        self._cache['cash_exchange_code'] = value

    @property
    def digest(self) -> str:
        """摘要"""
        if 'digest' not in self._cache:
            value = self._struct.Digest.rstrip(b'\x00').decode('ascii')
            self._cache['digest'] = value
        return self._cache['digest']

    @digest.setter
    def digest(self, value: str):
        """设置摘要"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.Digest = encoded
        self._cache['digest'] = value

    @property
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def device_id(self) -> str:
        """渠道标志"""
        if 'device_id' not in self._cache:
            value = self._struct.DeviceID.rstrip(b'\x00').decode('ascii')
            self._cache['device_id'] = value
        return self._cache['device_id']

    @device_id.setter
    def device_id(self, value: str):
        """设置渠道标志"""
        encoded = value.encode('ascii')[:2].ljust(3, b'\x00')
        self._struct.DeviceID = encoded
        self._cache['device_id'] = value

    @property
    def bank_secu_acc_type(self) -> str:
        """期货单位帐号类型"""
        if 'bank_secu_acc_type' not in self._cache:
            value = self._struct.BankSecuAccType.decode('ascii')
            self._cache['bank_secu_acc_type'] = value
        return self._cache['bank_secu_acc_type']

    @bank_secu_acc_type.setter
    def bank_secu_acc_type(self, value: str):
        """设置期货单位帐号类型"""
        self._struct.BankSecuAccType = value.encode('ascii')[0]
        self._cache['bank_secu_acc_type'] = value

    @property
    def broker_id_by_bank(self) -> str:
        """期货公司银行编码"""
        if 'broker_id_by_bank' not in self._cache:
            value = self._struct.BrokerIDByBank.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id_by_bank'] = value
        return self._cache['broker_id_by_bank']

    @broker_id_by_bank.setter
    def broker_id_by_bank(self, value: str):
        """设置期货公司银行编码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.BrokerIDByBank = encoded
        self._cache['broker_id_by_bank'] = value

    @property
    def bank_secu_acc(self) -> str:
        """期货单位帐号"""
        if 'bank_secu_acc' not in self._cache:
            value = self._struct.BankSecuAcc.rstrip(b'\x00').decode('ascii')
            self._cache['bank_secu_acc'] = value
        return self._cache['bank_secu_acc']

    @bank_secu_acc.setter
    def bank_secu_acc(self, value: str):
        """设置期货单位帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankSecuAcc = encoded
        self._cache['bank_secu_acc'] = value

    @property
    def bank_pwd_flag(self) -> str:
        """银行密码标志"""
        if 'bank_pwd_flag' not in self._cache:
            value = self._struct.BankPwdFlag.decode('ascii')
            self._cache['bank_pwd_flag'] = value
        return self._cache['bank_pwd_flag']

    @bank_pwd_flag.setter
    def bank_pwd_flag(self, value: str):
        """设置银行密码标志"""
        self._struct.BankPwdFlag = value.encode('ascii')[0]
        self._cache['bank_pwd_flag'] = value

    @property
    def secu_pwd_flag(self) -> str:
        """期货资金密码核对标志"""
        if 'secu_pwd_flag' not in self._cache:
            value = self._struct.SecuPwdFlag.decode('ascii')
            self._cache['secu_pwd_flag'] = value
        return self._cache['secu_pwd_flag']

    @secu_pwd_flag.setter
    def secu_pwd_flag(self, value: str):
        """设置期货资金密码核对标志"""
        self._struct.SecuPwdFlag = value.encode('ascii')[0]
        self._cache['secu_pwd_flag'] = value

    @property
    def oper_no(self) -> str:
        """交易柜员"""
        if 'oper_no' not in self._cache:
            value = self._struct.OperNo.rstrip(b'\x00').decode('ascii')
            self._cache['oper_no'] = value
        return self._cache['oper_no']

    @oper_no.setter
    def oper_no(self, value: str):
        """设置交易柜员"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.OperNo = encoded
        self._cache['oper_no'] = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def user_id(self) -> str:
        """用户标识"""
        if 'user_id' not in self._cache:
            value = self._struct.UserID.rstrip(b'\x00').decode('ascii')
            self._cache['user_id'] = value
        return self._cache['user_id']

    @user_id.setter
    def user_id(self, value: str):
        """设置用户标识"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.UserID = encoded
        self._cache['user_id'] = value

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
    def long_customer_name(self) -> str:
        """长客户姓名（GBK 编码）"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名（GBK 编码）"""
        encoded = value.encode('gbk')[:150].ljust(151, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class CancelAccount(CapsuleStruct):
    """银期销户信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeCode", ctypes.c_char * 7),              # 业务功能码
            ("BankID", ctypes.c_char * 4),                 # 银行代码
            ("BankBranchID", ctypes.c_char * 5),           # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),              # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),        # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),              # 交易日期
            ("TradeTime", ctypes.c_char * 9),              # 交易时间
            ("BankSerial", ctypes.c_char * 13),            # 银行流水号
            ("TradingDay", ctypes.c_char * 9),             # 交易系统日期
            ("PlateSerial", ctypes.c_int),                 # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),               # 最后分片标志
            ("SessionID", ctypes.c_int),                   # 会话号
            ("CustomerName", ctypes.c_char * 51),          # 客户姓名
            ("IdCardType", ctypes.c_char),                 # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),      # 证件号码
            ("Gender", ctypes.c_char),                     # 性别
            ("CountryCode", ctypes.c_char * 21),           # 国家代码
            ("CustType", ctypes.c_char),                   # 客户类型
            ("Address", ctypes.c_char * 101),              # 地址
            ("ZipCode", ctypes.c_char * 7),                # 邮编
            ("Telephone", ctypes.c_char * 41),             # 电话号码
            ("MobilePhone", ctypes.c_char * 41),           # 手机
            ("Fax", ctypes.c_char * 41),                   # 传真
            ("EMail", ctypes.c_char * 51),                 # 电子邮件
            ("MoneyAccountStatus", ctypes.c_char),         # 资金账户状态
            ("BankAccount", ctypes.c_char * 41),           # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),          # 银行密码
            ("AccountID", ctypes.c_char * 13),             # 投资者帐号
            ("Password", ctypes.c_char * 41),              # 期货密码
            ("InstallID", ctypes.c_int),                   # 安装编号
            ("VerifyCertNoFlag", ctypes.c_char),           # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),             # 币种代码
            ("CashExchangeCode", ctypes.c_char),           # 汇钞标志
            ("Digest", ctypes.c_char * 36),                # 摘要
            ("BankAccType", ctypes.c_char),                # 银行帐号类型
            ("DeviceID", ctypes.c_char * 3),               # 渠道标志
            ("BankSecuAccType", ctypes.c_char),            # 期货单位帐号类型
            ("BrokerIDByBank", ctypes.c_char * 33),        # 期货公司银行编码
            ("BankSecuAcc", ctypes.c_char * 41),           # 期货单位帐号
            ("BankPwdFlag", ctypes.c_char),                # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),                # 期货资金密码核对标志
            ("OperNo", ctypes.c_char * 17),                # 交易柜员
            ("TID", ctypes.c_int),                         # 交易ID
            ("UserID", ctypes.c_char * 16),                # 用户标识
            ("ErrorID", ctypes.c_int),                     # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),              # 错误信息
            ("LongCustomerName", ctypes.c_char * 161),     # 长客户姓名
        ]

    _capsule_name = "CancelAccount"

    _field_mappings = {
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "gender": "Gender",
        "country_code": "CountryCode",
        "cust_type": "CustType",
        "address": "Address",
        "zip_code": "ZipCode",
        "telephone": "Telephone",
        "mobile_phone": "MobilePhone",
        "fax": "Fax",
        "email": "EMail",
        "money_account_status": "MoneyAccountStatus",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "install_id": "InstallID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "cash_exchange_code": "CashExchangeCode",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "tid": "TID",
        "user_id": "UserID",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
        "long_customer_name": "LongCustomerName",
    }

    @property
    def trade_code(self) -> str:
        """业务功能码"""
        if 'trade_code' not in self._cache:
            value = self._struct.TradeCode.rstrip(b'\x00').decode('ascii')
            self._cache['trade_code'] = value
        return self._cache['trade_code']

    @trade_code.setter
    def trade_code(self, value: str):
        """设置业务功能码"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.TradeCode = encoded
        self._cache['trade_code'] = value

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
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期商分支机构代码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期商分支机构代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

    @property
    def trade_date(self) -> str:
        """交易日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trade_time(self) -> str:
        """交易时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置交易时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

    @property
    def bank_serial(self) -> str:
        """银行流水号"""
        if 'bank_serial' not in self._cache:
            value = self._struct.BankSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_serial'] = value
        return self._cache['bank_serial']

    @bank_serial.setter
    def bank_serial(self, value: str):
        """设置银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankSerial = encoded
        self._cache['bank_serial'] = value

    @property
    def trading_day(self) -> str:
        """交易系统日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易系统日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    @property
    def plate_serial(self) -> int:
        """银期平台消息流水号"""
        return self._struct.PlateSerial

    @plate_serial.setter
    def plate_serial(self, value: int):
        """设置银期平台消息流水号"""
        self._struct.PlateSerial = value

    @property
    def last_fragment(self) -> str:
        """最后分片标志"""
        if 'last_fragment' not in self._cache:
            value = self._struct.LastFragment.decode('ascii')
            self._cache['last_fragment'] = value
        return self._cache['last_fragment']

    @last_fragment.setter
    def last_fragment(self, value: str):
        """设置最后分片标志"""
        self._struct.LastFragment = value.encode('ascii')[0]
        self._cache['last_fragment'] = value

    @property
    def session_id(self) -> int:
        """会话号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话号"""
        self._struct.SessionID = value

    @property
    def customer_name(self) -> str:
        """客户姓名（GBK 编码）"""
        if 'customer_name' not in self._cache:
            value = self._struct.CustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['customer_name'] = value
        return self._cache['customer_name']

    @customer_name.setter
    def customer_name(self, value: str):
        """设置客户姓名（GBK 编码）"""
        encoded = value.encode('gbk')[:50].ljust(51, b'\x00')
        self._struct.CustomerName = encoded
        self._cache['customer_name'] = value

    @property
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def gender(self) -> str:
        """性别"""
        if 'gender' not in self._cache:
            value = self._struct.Gender.decode('ascii')
            self._cache['gender'] = value
        return self._cache['gender']

    @gender.setter
    def gender(self, value: str):
        """设置性别"""
        self._struct.Gender = value.encode('ascii')[0]
        self._cache['gender'] = value

    @property
    def country_code(self) -> str:
        """国家代码"""
        if 'country_code' not in self._cache:
            value = self._struct.CountryCode.rstrip(b'\x00').decode('ascii')
            self._cache['country_code'] = value
        return self._cache['country_code']

    @country_code.setter
    def country_code(self, value: str):
        """设置国家代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.CountryCode = encoded
        self._cache['country_code'] = value

    @property
    def cust_type(self) -> str:
        """客户类型"""
        if 'cust_type' not in self._cache:
            value = self._struct.CustType.decode('ascii')
            self._cache['cust_type'] = value
        return self._cache['cust_type']

    @cust_type.setter
    def cust_type(self, value: str):
        """设置客户类型"""
        self._struct.CustType = value.encode('ascii')[0]
        self._cache['cust_type'] = value

    @property
    def address(self) -> str:
        """地址（GBK 编码）"""
        if 'address' not in self._cache:
            value = self._struct.Address.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['address'] = value
        return self._cache['address']

    @address.setter
    def address(self, value: str):
        """设置地址（GBK 编码）"""
        encoded = value.encode('gbk')[:100].ljust(101, b'\x00')
        self._struct.Address = encoded
        self._cache['address'] = value

    @property
    def zip_code(self) -> str:
        """邮编"""
        if 'zip_code' not in self._cache:
            value = self._struct.ZipCode.rstrip(b'\x00').decode('ascii')
            self._cache['zip_code'] = value
        return self._cache['zip_code']

    @zip_code.setter
    def zip_code(self, value: str):
        """设置邮编"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.ZipCode = encoded
        self._cache['zip_code'] = value

    @property
    def telephone(self) -> str:
        """电话号码"""
        if 'telephone' not in self._cache:
            value = self._struct.Telephone.rstrip(b'\x00').decode('ascii')
            self._cache['telephone'] = value
        return self._cache['telephone']

    @telephone.setter
    def telephone(self, value: str):
        """设置电话号码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Telephone = encoded
        self._cache['telephone'] = value

    @property
    def mobile_phone(self) -> str:
        """手机"""
        if 'mobile_phone' not in self._cache:
            value = self._struct.MobilePhone.rstrip(b'\x00').decode('ascii')
            self._cache['mobile_phone'] = value
        return self._cache['mobile_phone']

    @mobile_phone.setter
    def mobile_phone(self, value: str):
        """设置手机"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.MobilePhone = encoded
        self._cache['mobile_phone'] = value

    @property
    def fax(self) -> str:
        """传真"""
        if 'fax' not in self._cache:
            value = self._struct.Fax.rstrip(b'\x00').decode('ascii')
            self._cache['fax'] = value
        return self._cache['fax']

    @fax.setter
    def fax(self, value: str):
        """设置传真"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Fax = encoded
        self._cache['fax'] = value

    @property
    def email(self) -> str:
        """电子邮件"""
        if 'email' not in self._cache:
            value = self._struct.EMail.rstrip(b'\x00').decode('ascii')
            self._cache['email'] = value
        return self._cache['email']

    @email.setter
    def email(self, value: str):
        """设置电子邮件"""
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.EMail = encoded
        self._cache['email'] = value

    @property
    def money_account_status(self) -> str:
        """资金账户状态"""
        if 'money_account_status' not in self._cache:
            value = self._struct.MoneyAccountStatus.decode('ascii')
            self._cache['money_account_status'] = value
        return self._cache['money_account_status']

    @money_account_status.setter
    def money_account_status(self, value: str):
        """设置资金账户状态"""
        self._struct.MoneyAccountStatus = value.encode('ascii')[0]
        self._cache['money_account_status'] = value

    @property
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def bank_pass_word(self) -> str:
        """银行密码"""
        if 'bank_pass_word' not in self._cache:
            value = self._struct.BankPassWord.rstrip(b'\x00').decode('ascii')
            self._cache['bank_pass_word'] = value
        return self._cache['bank_pass_word']

    @bank_pass_word.setter
    def bank_pass_word(self, value: str):
        """设置银行密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankPassWord = encoded
        self._cache['bank_pass_word'] = value

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
    def password(self) -> str:
        """期货密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置期货密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def verify_cert_no_flag(self) -> str:
        """验证客户证件号码标志"""
        if 'verify_cert_no_flag' not in self._cache:
            value = self._struct.VerifyCertNoFlag.decode('ascii')
            self._cache['verify_cert_no_flag'] = value
        return self._cache['verify_cert_no_flag']

    @verify_cert_no_flag.setter
    def verify_cert_no_flag(self, value: str):
        """设置验证客户证件号码标志"""
        self._struct.VerifyCertNoFlag = value.encode('ascii')[0]
        self._cache['verify_cert_no_flag'] = value

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
    def cash_exchange_code(self) -> str:
        """汇钞标志"""
        if 'cash_exchange_code' not in self._cache:
            value = self._struct.CashExchangeCode.decode('ascii')
            self._cache['cash_exchange_code'] = value
        return self._cache['cash_exchange_code']

    @cash_exchange_code.setter
    def cash_exchange_code(self, value: str):
        """设置汇钞标志"""
        self._struct.CashExchangeCode = value.encode('ascii')[0]
        self._cache['cash_exchange_code'] = value

    @property
    def digest(self) -> str:
        """摘要"""
        if 'digest' not in self._cache:
            value = self._struct.Digest.rstrip(b'\x00').decode('ascii')
            self._cache['digest'] = value
        return self._cache['digest']

    @digest.setter
    def digest(self, value: str):
        """设置摘要"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.Digest = encoded
        self._cache['digest'] = value

    @property
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def device_id(self) -> str:
        """渠道标志"""
        if 'device_id' not in self._cache:
            value = self._struct.DeviceID.rstrip(b'\x00').decode('ascii')
            self._cache['device_id'] = value
        return self._cache['device_id']

    @device_id.setter
    def device_id(self, value: str):
        """设置渠道标志"""
        encoded = value.encode('ascii')[:2].ljust(3, b'\x00')
        self._struct.DeviceID = encoded
        self._cache['device_id'] = value

    @property
    def bank_secu_acc_type(self) -> str:
        """期货单位帐号类型"""
        if 'bank_secu_acc_type' not in self._cache:
            value = self._struct.BankSecuAccType.decode('ascii')
            self._cache['bank_secu_acc_type'] = value
        return self._cache['bank_secu_acc_type']

    @bank_secu_acc_type.setter
    def bank_secu_acc_type(self, value: str):
        """设置期货单位帐号类型"""
        self._struct.BankSecuAccType = value.encode('ascii')[0]
        self._cache['bank_secu_acc_type'] = value

    @property
    def broker_id_by_bank(self) -> str:
        """期货公司银行编码"""
        if 'broker_id_by_bank' not in self._cache:
            value = self._struct.BrokerIDByBank.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id_by_bank'] = value
        return self._cache['broker_id_by_bank']

    @broker_id_by_bank.setter
    def broker_id_by_bank(self, value: str):
        """设置期货公司银行编码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.BrokerIDByBank = encoded
        self._cache['broker_id_by_bank'] = value

    @property
    def bank_secu_acc(self) -> str:
        """期货单位帐号"""
        if 'bank_secu_acc' not in self._cache:
            value = self._struct.BankSecuAcc.rstrip(b'\x00').decode('ascii')
            self._cache['bank_secu_acc'] = value
        return self._cache['bank_secu_acc']

    @bank_secu_acc.setter
    def bank_secu_acc(self, value: str):
        """设置期货单位帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankSecuAcc = encoded
        self._cache['bank_secu_acc'] = value

    @property
    def bank_pwd_flag(self) -> str:
        """银行密码标志"""
        if 'bank_pwd_flag' not in self._cache:
            value = self._struct.BankPwdFlag.decode('ascii')
            self._cache['bank_pwd_flag'] = value
        return self._cache['bank_pwd_flag']

    @bank_pwd_flag.setter
    def bank_pwd_flag(self, value: str):
        """设置银行密码标志"""
        self._struct.BankPwdFlag = value.encode('ascii')[0]
        self._cache['bank_pwd_flag'] = value

    @property
    def secu_pwd_flag(self) -> str:
        """期货资金密码核对标志"""
        if 'secu_pwd_flag' not in self._cache:
            value = self._struct.SecuPwdFlag.decode('ascii')
            self._cache['secu_pwd_flag'] = value
        return self._cache['secu_pwd_flag']

    @secu_pwd_flag.setter
    def secu_pwd_flag(self, value: str):
        """设置期货资金密码核对标志"""
        self._struct.SecuPwdFlag = value.encode('ascii')[0]
        self._cache['secu_pwd_flag'] = value

    @property
    def oper_no(self) -> str:
        """交易柜员"""
        if 'oper_no' not in self._cache:
            value = self._struct.OperNo.rstrip(b'\x00').decode('ascii')
            self._cache['oper_no'] = value
        return self._cache['oper_no']

    @oper_no.setter
    def oper_no(self, value: str):
        """设置交易柜员"""
        encoded = value.encode('ascii')[:16].ljust(17, b'\x00')
        self._struct.OperNo = encoded
        self._cache['oper_no'] = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def user_id(self) -> str:
        """用户标识"""
        if 'user_id' not in self._cache:
            value = self._struct.UserID.rstrip(b'\x00').decode('ascii')
            self._cache['user_id'] = value
        return self._cache['user_id']

    @user_id.setter
    def user_id(self, value: str):
        """设置用户标识"""
        encoded = value.encode('ascii')[:15].ljust(16, b'\x00')
        self._struct.UserID = encoded
        self._cache['user_id'] = value

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
    def long_customer_name(self) -> str:
        """长客户姓名（GBK 编码）"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名（GBK 编码）"""
        encoded = value.encode('gbk')[:150].ljust(151, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class ChangeAccount(CapsuleStruct):
    """银期变更银行账号信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradeCode", ctypes.c_char * 7),              # 业务功能码
            ("BankID", ctypes.c_char * 4),                 # 银行代码
            ("BankBranchID", ctypes.c_char * 5),           # 银行分支机构代码
            ("BrokerID", ctypes.c_char * 11),              # 期商代码
            ("BrokerBranchID", ctypes.c_char * 31),        # 期商分支机构代码
            ("TradeDate", ctypes.c_char * 9),              # 交易日期
            ("TradeTime", ctypes.c_char * 9),              # 交易时间
            ("BankSerial", ctypes.c_char * 13),            # 银行流水号
            ("TradingDay", ctypes.c_char * 9),             # 交易系统日期
            ("PlateSerial", ctypes.c_int),                 # 银期平台消息流水号
            ("LastFragment", ctypes.c_char),               # 最后分片标志
            ("SessionID", ctypes.c_int),                   # 会话号
            ("CustomerName", ctypes.c_char * 51),          # 客户姓名
            ("IdCardType", ctypes.c_char),                 # 证件类型
            ("IdentifiedCardNo", ctypes.c_char * 51),      # 证件号码
            ("Gender", ctypes.c_char),                     # 性别
            ("CountryCode", ctypes.c_char * 21),           # 国家代码
            ("CustType", ctypes.c_char),                   # 客户类型
            ("Address", ctypes.c_char * 101),              # 地址
            ("ZipCode", ctypes.c_char * 7),                # 邮编
            ("Telephone", ctypes.c_char * 41),             # 电话号码
            ("MobilePhone", ctypes.c_char * 41),           # 手机
            ("Fax", ctypes.c_char * 41),                   # 传真
            ("EMail", ctypes.c_char * 51),                 # 电子邮件
            ("MoneyAccountStatus", ctypes.c_char),         # 资金账户状态
            ("BankAccount", ctypes.c_char * 41),           # 银行帐号
            ("BankPassWord", ctypes.c_char * 41),          # 银行密码
            ("NewBankAccount", ctypes.c_char * 41),        # 新银行帐号
            ("NewBankPassWord", ctypes.c_char * 41),       # 新银行密码
            ("AccountID", ctypes.c_char * 13),             # 投资者帐号
            ("Password", ctypes.c_char * 41),              # 期货密码
            ("BankAccType", ctypes.c_char),                # 银行帐号类型
            ("InstallID", ctypes.c_int),                   # 安装编号
            ("VerifyCertNoFlag", ctypes.c_char),           # 验证客户证件号码标志
            ("CurrencyID", ctypes.c_char * 4),             # 币种代码
            ("BrokerIDByBank", ctypes.c_char * 33),        # 期货公司银行编码
            ("BankPwdFlag", ctypes.c_char),                # 银行密码标志
            ("SecuPwdFlag", ctypes.c_char),                # 期货资金密码核对标志
            ("TID", ctypes.c_int),                         # 交易ID
            ("Digest", ctypes.c_char * 36),                # 摘要
            ("ErrorID", ctypes.c_int),                     # 错误代码
            ("ErrorMsg", ctypes.c_char * 81),              # 错误信息
            ("LongCustomerName", ctypes.c_char * 161),     # 长客户姓名
        ]

    _capsule_name = "ChangeAccount"

    _field_mappings = {
        "trade_code": "TradeCode",
        "bank_id": "BankID",
        "bank_branch_id": "BankBranchID",
        "broker_id": "BrokerID",
        "broker_branch_id": "BrokerBranchID",
        "trade_date": "TradeDate",
        "trade_time": "TradeTime",
        "bank_serial": "BankSerial",
        "trading_day": "TradingDay",
        "plate_serial": "PlateSerial",
        "last_fragment": "LastFragment",
        "session_id": "SessionID",
        "customer_name": "CustomerName",
        "id_card_type": "IdCardType",
        "identified_card_no": "IdentifiedCardNo",
        "gender": "Gender",
        "country_code": "CountryCode",
        "cust_type": "CustType",
        "address": "Address",
        "zip_code": "ZipCode",
        "telephone": "Telephone",
        "mobile_phone": "MobilePhone",
        "fax": "Fax",
        "email": "EMail",
        "money_account_status": "MoneyAccountStatus",
        "bank_account": "BankAccount",
        "bank_pass_word": "BankPassWord",
        "account_id": "AccountID",
        "password": "Password",
        "install_id": "InstallID",
        "verify_cert_no_flag": "VerifyCertNoFlag",
        "currency_id": "CurrencyID",
        "cash_exchange_code": "CashExchangeCode",
        "digest": "Digest",
        "bank_acc_type": "BankAccType",
        "device_id": "DeviceID",
        "bank_secu_acc_type": "BankSecuAccType",
        "broker_id_by_bank": "BrokerIDByBank",
        "bank_secu_acc": "BankSecuAcc",
        "bank_pwd_flag": "BankPwdFlag",
        "secu_pwd_flag": "SecuPwdFlag",
        "oper_no": "OperNo",
        "tid": "TID",
        "user_id": "UserID",
        "error_id": "ErrorID",
        "error_msg": "ErrorMsg",
        "long_customer_name": "LongCustomerName",
    }

    # 字段属性定义
    @property
    def trade_code(self) -> str:
        """业务功能码"""
        if 'trade_code' not in self._cache:
            value = self._struct.TradeCode.rstrip(b'\x00').decode('ascii')
            self._cache['trade_code'] = value
        return self._cache['trade_code']

    @trade_code.setter
    def trade_code(self, value: str):
        """设置业务功能码"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.TradeCode = encoded
        self._cache['trade_code'] = value

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
    def bank_branch_id(self) -> str:
        """银行分支机构代码"""
        if 'bank_branch_id' not in self._cache:
            value = self._struct.BankBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['bank_branch_id'] = value
        return self._cache['bank_branch_id']

    @bank_branch_id.setter
    def bank_branch_id(self, value: str):
        """设置银行分支机构代码"""
        encoded = value.encode('ascii')[:4].ljust(5, b'\x00')
        self._struct.BankBranchID = encoded
        self._cache['bank_branch_id'] = value

    @property
    def broker_id(self) -> str:
        """期商代码"""
        if 'broker_id' not in self._cache:
            value = self._struct.BrokerID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id'] = value
        return self._cache['broker_id']

    @broker_id.setter
    def broker_id(self, value: str):
        """设置期商代码"""
        encoded = value.encode('ascii')[:10].ljust(11, b'\x00')
        self._struct.BrokerID = encoded
        self._cache['broker_id'] = value

    @property
    def broker_branch_id(self) -> str:
        """期商分支机构代码"""
        if 'broker_branch_id' not in self._cache:
            value = self._struct.BrokerBranchID.rstrip(b'\x00').decode('ascii')
            self._cache['broker_branch_id'] = value
        return self._cache['broker_branch_id']

    @broker_branch_id.setter
    def broker_branch_id(self, value: str):
        """设置期商分支机构代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.BrokerBranchID = encoded
        self._cache['broker_branch_id'] = value

    @property
    def trade_date(self) -> str:
        """交易日期"""
        if 'trade_date' not in self._cache:
            value = self._struct.TradeDate.rstrip(b'\x00').decode('ascii')
            self._cache['trade_date'] = value
        return self._cache['trade_date']

    @trade_date.setter
    def trade_date(self, value: str):
        """设置交易日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeDate = encoded
        self._cache['trade_date'] = value

    @property
    def trade_time(self) -> str:
        """交易时间"""
        if 'trade_time' not in self._cache:
            value = self._struct.TradeTime.rstrip(b'\x00').decode('ascii')
            self._cache['trade_time'] = value
        return self._cache['trade_time']

    @trade_time.setter
    def trade_time(self, value: str):
        """设置交易时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradeTime = encoded
        self._cache['trade_time'] = value

    @property
    def bank_serial(self) -> str:
        """银行流水号"""
        if 'bank_serial' not in self._cache:
            value = self._struct.BankSerial.rstrip(b'\x00').decode('ascii')
            self._cache['bank_serial'] = value
        return self._cache['bank_serial']

    @bank_serial.setter
    def bank_serial(self, value: str):
        """设置银行流水号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.BankSerial = encoded
        self._cache['bank_serial'] = value

    @property
    def trading_day(self) -> str:
        """交易系统日期"""
        if 'trading_day' not in self._cache:
            value = self._struct.TradingDay.rstrip(b'\x00').decode('ascii')
            self._cache['trading_day'] = value
        return self._cache['trading_day']

    @trading_day.setter
    def trading_day(self, value: str):
        """设置交易系统日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.TradingDay = encoded
        self._cache['trading_day'] = value

    @property
    def plate_serial(self) -> int:
        """银期平台消息流水号"""
        return self._struct.PlateSerial

    @plate_serial.setter
    def plate_serial(self, value: int):
        """设置银期平台消息流水号"""
        self._struct.PlateSerial = value

    @property
    def last_fragment(self) -> str:
        """最后分片标志"""
        if 'last_fragment' not in self._cache:
            value = self._struct.LastFragment.decode('ascii')
            self._cache['last_fragment'] = value
        return self._cache['last_fragment']

    @last_fragment.setter
    def last_fragment(self, value: str):
        """设置最后分片标志"""
        self._struct.LastFragment = value.encode('ascii')[0]
        self._cache['last_fragment'] = value

    @property
    def session_id(self) -> int:
        """会话号"""
        return self._struct.SessionID

    @session_id.setter
    def session_id(self, value: int):
        """设置会话号"""
        self._struct.SessionID = value

    @property
    def customer_name(self) -> str:
        """客户姓名（GBK 编码）"""
        if 'customer_name' not in self._cache:
            value = self._struct.CustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['customer_name'] = value
        return self._cache['customer_name']

    @customer_name.setter
    def customer_name(self, value: str):
        """设置客户姓名（GBK 编码）"""
        encoded = value.encode('gbk')[:50].ljust(51, b'\x00')
        self._struct.CustomerName = encoded
        self._cache['customer_name'] = value

    @property
    def id_card_type(self) -> str:
        """证件类型"""
        if 'id_card_type' not in self._cache:
            value = self._struct.IdCardType.decode('ascii')
            self._cache['id_card_type'] = value
        return self._cache['id_card_type']

    @id_card_type.setter
    def id_card_type(self, value: str):
        """设置证件类型"""
        self._struct.IdCardType = value.encode('ascii')[0]
        self._cache['id_card_type'] = value

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
    def gender(self) -> str:
        """性别"""
        if 'gender' not in self._cache:
            value = self._struct.Gender.decode('ascii')
            self._cache['gender'] = value
        return self._cache['gender']

    @gender.setter
    def gender(self, value: str):
        """设置性别"""
        self._struct.Gender = value.encode('ascii')[0]
        self._cache['gender'] = value

    @property
    def country_code(self) -> str:
        """国家代码"""
        if 'country_code' not in self._cache:
            value = self._struct.CountryCode.rstrip(b'\x00').decode('ascii')
            self._cache['country_code'] = value
        return self._cache['country_code']

    @country_code.setter
    def country_code(self, value: str):
        """设置国家代码"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.CountryCode = encoded
        self._cache['country_code'] = value

    @property
    def cust_type(self) -> str:
        """客户类型"""
        if 'cust_type' not in self._cache:
            value = self._struct.CustType.decode('ascii')
            self._cache['cust_type'] = value
        return self._cache['cust_type']

    @cust_type.setter
    def cust_type(self, value: str):
        """设置客户类型"""
        self._struct.CustType = value.encode('ascii')[0]
        self._cache['cust_type'] = value

    @property
    def address(self) -> str:
        """地址（GBK 编码）"""
        if 'address' not in self._cache:
            value = self._struct.Address.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['address'] = value
        return self._cache['address']

    @address.setter
    def address(self, value: str):
        """设置地址（GBK 编码）"""
        encoded = value.encode('gbk')[:100].ljust(101, b'\x00')
        self._struct.Address = encoded
        self._cache['address'] = value

    @property
    def zip_code(self) -> str:
        """邮编"""
        if 'zip_code' not in self._cache:
            value = self._struct.ZipCode.rstrip(b'\x00').decode('ascii')
            self._cache['zip_code'] = value
        return self._cache['zip_code']

    @zip_code.setter
    def zip_code(self, value: str):
        """设置邮编"""
        encoded = value.encode('ascii')[:6].ljust(7, b'\x00')
        self._struct.ZipCode = encoded
        self._cache['zip_code'] = value

    @property
    def telephone(self) -> str:
        """电话号码"""
        if 'telephone' not in self._cache:
            value = self._struct.Telephone.rstrip(b'\x00').decode('ascii')
            self._cache['telephone'] = value
        return self._cache['telephone']

    @telephone.setter
    def telephone(self, value: str):
        """设置电话号码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Telephone = encoded
        self._cache['telephone'] = value

    @property
    def mobile_phone(self) -> str:
        """手机"""
        if 'mobile_phone' not in self._cache:
            value = self._struct.MobilePhone.rstrip(b'\x00').decode('ascii')
            self._cache['mobile_phone'] = value
        return self._cache['mobile_phone']

    @mobile_phone.setter
    def mobile_phone(self, value: str):
        """设置手机"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.MobilePhone = encoded
        self._cache['mobile_phone'] = value

    @property
    def fax(self) -> str:
        """传真"""
        if 'fax' not in self._cache:
            value = self._struct.Fax.rstrip(b'\x00').decode('ascii')
            self._cache['fax'] = value
        return self._cache['fax']

    @fax.setter
    def fax(self, value: str):
        """设置传真"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Fax = encoded
        self._cache['fax'] = value

    @property
    def email(self) -> str:
        """电子邮件"""
        if 'email' not in self._cache:
            value = self._struct.EMail.rstrip(b'\x00').decode('ascii')
            self._cache['email'] = value
        return self._cache['email']

    @email.setter
    def email(self, value: str):
        """设置电子邮件"""
        encoded = value.encode('ascii')[:50].ljust(51, b'\x00')
        self._struct.EMail = encoded
        self._cache['email'] = value

    @property
    def money_account_status(self) -> str:
        """资金账户状态"""
        if 'money_account_status' not in self._cache:
            value = self._struct.MoneyAccountStatus.decode('ascii')
            self._cache['money_account_status'] = value
        return self._cache['money_account_status']

    @money_account_status.setter
    def money_account_status(self, value: str):
        """设置资金账户状态"""
        self._struct.MoneyAccountStatus = value.encode('ascii')[0]
        self._cache['money_account_status'] = value

    @property
    def bank_account(self) -> str:
        """银行帐号"""
        if 'bank_account' not in self._cache:
            value = self._struct.BankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['bank_account'] = value
        return self._cache['bank_account']

    @bank_account.setter
    def bank_account(self, value: str):
        """设置银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankAccount = encoded
        self._cache['bank_account'] = value

    @property
    def bank_pass_word(self) -> str:
        """银行密码"""
        if 'bank_pass_word' not in self._cache:
            value = self._struct.BankPassWord.rstrip(b'\x00').decode('ascii')
            self._cache['bank_pass_word'] = value
        return self._cache['bank_pass_word']

    @bank_pass_word.setter
    def bank_pass_word(self, value: str):
        """设置银行密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.BankPassWord = encoded
        self._cache['bank_pass_word'] = value

    @property
    def new_bank_account(self) -> str:
        """新银行帐号"""
        if 'new_bank_account' not in self._cache:
            value = self._struct.NewBankAccount.rstrip(b'\x00').decode('ascii')
            self._cache['new_bank_account'] = value
        return self._cache['new_bank_account']

    @new_bank_account.setter
    def new_bank_account(self, value: str):
        """设置新银行帐号"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.NewBankAccount = encoded
        self._cache['new_bank_account'] = value

    @property
    def new_bank_pass_word(self) -> str:
        """新银行密码"""
        if 'new_bank_pass_word' not in self._cache:
            value = self._struct.NewBankPassWord.rstrip(b'\x00').decode('ascii')
            self._cache['new_bank_pass_word'] = value
        return self._cache['new_bank_pass_word']

    @new_bank_pass_word.setter
    def new_bank_pass_word(self, value: str):
        """设置新银行密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.NewBankPassWord = encoded
        self._cache['new_bank_pass_word'] = value

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
    def password(self) -> str:
        """期货密码"""
        if 'password' not in self._cache:
            value = self._struct.Password.rstrip(b'\x00').decode('ascii')
            self._cache['password'] = value
        return self._cache['password']

    @password.setter
    def password(self, value: str):
        """设置期货密码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.Password = encoded
        self._cache['password'] = value

    @property
    def bank_acc_type(self) -> str:
        """银行帐号类型"""
        if 'bank_acc_type' not in self._cache:
            value = self._struct.BankAccType.decode('ascii')
            self._cache['bank_acc_type'] = value
        return self._cache['bank_acc_type']

    @bank_acc_type.setter
    def bank_acc_type(self, value: str):
        """设置银行帐号类型"""
        self._struct.BankAccType = value.encode('ascii')[0]
        self._cache['bank_acc_type'] = value

    @property
    def install_id(self) -> int:
        """安装编号"""
        return self._struct.InstallID

    @install_id.setter
    def install_id(self, value: int):
        """设置安装编号"""
        self._struct.InstallID = value

    @property
    def verify_cert_no_flag(self) -> str:
        """验证客户证件号码标志"""
        if 'verify_cert_no_flag' not in self._cache:
            value = self._struct.VerifyCertNoFlag.decode('ascii')
            self._cache['verify_cert_no_flag'] = value
        return self._cache['verify_cert_no_flag']

    @verify_cert_no_flag.setter
    def verify_cert_no_flag(self, value: str):
        """设置验证客户证件号码标志"""
        self._struct.VerifyCertNoFlag = value.encode('ascii')[0]
        self._cache['verify_cert_no_flag'] = value

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
    def broker_id_by_bank(self) -> str:
        """期货公司银行编码"""
        if 'broker_id_by_bank' not in self._cache:
            value = self._struct.BrokerIDByBank.rstrip(b'\x00').decode('ascii')
            self._cache['broker_id_by_bank'] = value
        return self._cache['broker_id_by_bank']

    @broker_id_by_bank.setter
    def broker_id_by_bank(self, value: str):
        """设置期货公司银行编码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.BrokerIDByBank = encoded
        self._cache['broker_id_by_bank'] = value

    @property
    def bank_pwd_flag(self) -> str:
        """银行密码标志"""
        if 'bank_pwd_flag' not in self._cache:
            value = self._struct.BankPwdFlag.decode('ascii')
            self._cache['bank_pwd_flag'] = value
        return self._cache['bank_pwd_flag']

    @bank_pwd_flag.setter
    def bank_pwd_flag(self, value: str):
        """设置银行密码标志"""
        self._struct.BankPwdFlag = value.encode('ascii')[0]
        self._cache['bank_pwd_flag'] = value

    @property
    def secu_pwd_flag(self) -> str:
        """期货资金密码核对标志"""
        if 'secu_pwd_flag' not in self._cache:
            value = self._struct.SecuPwdFlag.decode('ascii')
            self._cache['secu_pwd_flag'] = value
        return self._cache['secu_pwd_flag']

    @secu_pwd_flag.setter
    def secu_pwd_flag(self, value: str):
        """设置期货资金密码核对标志"""
        self._struct.SecuPwdFlag = value.encode('ascii')[0]
        self._cache['secu_pwd_flag'] = value

    @property
    def tid(self) -> int:
        """交易ID"""
        return self._struct.TID

    @tid.setter
    def tid(self, value: int):
        """设置交易ID"""
        self._struct.TID = value

    @property
    def digest(self) -> str:
        """摘要"""
        if 'digest' not in self._cache:
            value = self._struct.Digest.rstrip(b'\x00').decode('ascii')
            self._cache['digest'] = value
        return self._cache['digest']

    @digest.setter
    def digest(self, value: str):
        """设置摘要"""
        encoded = value.encode('ascii')[:35].ljust(36, b'\x00')
        self._struct.Digest = encoded
        self._cache['digest'] = value

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
    def long_customer_name(self) -> str:
        """长客户姓名（GBK 编码）"""
        if 'long_customer_name' not in self._cache:
            value = self._struct.LongCustomerName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['long_customer_name'] = value
        return self._cache['long_customer_name']

    @long_customer_name.setter
    def long_customer_name(self, value: str):
        """设置长客户姓名（GBK 编码）"""
        encoded = value.encode('gbk')[:150].ljust(151, b'\x00')
        self._struct.LongCustomerName = encoded
        self._cache['long_customer_name'] = value



class CombPromotionParam(CapsuleStruct):
    """期权组合保证金参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
            ("CombHedgeFlag", ctypes.c_char),              # 投机套保标志
            ("Xparameter", ctypes.c_double),               # 期权组合保证金比例
        ]

    _capsule_name = "CombPromotionParam"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "comb_hedge_flag": "CombHedgeFlag",
        "xparameter": "Xparameter",
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
    def comb_hedge_flag(self) -> str:
        """投机套保标志"""
        if 'comb_hedge_flag' not in self._cache:
            value = self._struct.CombHedgeFlag.decode('ascii')
            self._cache['comb_hedge_flag'] = value
        return self._cache['comb_hedge_flag']

    @comb_hedge_flag.setter
    def comb_hedge_flag(self, value: str):
        """设置投机套保标志"""
        self._struct.CombHedgeFlag = value.encode('ascii')[0]
        self._cache['comb_hedge_flag'] = value

    @property
    def xparameter(self) -> float:
        """期权组合保证金比例"""
        return self._struct.Xparameter

    @xparameter.setter
    def xparameter(self, value: float):
        """设置期权组合保证金比例"""
        self._struct.Xparameter = value



class InvestorInfoCommRec(CapsuleStruct):
    """投资者信息通信记录"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("InstrumentID", ctypes.c_char * 81),          # 商品代码
            ("OrderCount", ctypes.c_int),                  # 报单总笔数
            ("OrderActionCount", ctypes.c_int),            # 撤单总笔数
            ("ForQuoteCnt", ctypes.c_int),                 # 询价总次数
            ("InfoComm", ctypes.c_double),                 # 申报费
            ("IsOptSeries", ctypes.c_int),                 # 是否期权系列
            ("ProductID", ctypes.c_char * 41),             # 品种代码
            ("InfoCnt", ctypes.c_int),                     # 信息量总量
        ]

    _capsule_name = "InvestorInfoCommRec"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "instrument_id": "InstrumentID",
        "order_count": "OrderCount",
        "order_action_count": "OrderActionCount",
        "for_quote_cnt": "ForQuoteCnt",
        "info_comm": "InfoComm",
        "is_opt_series": "IsOptSeries",
        "product_id": "ProductID",
        "info_cnt": "InfoCnt",
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
    def order_count(self) -> int:
        """报单总笔数"""
        return self._struct.OrderCount

    @order_count.setter
    def order_count(self, value: int):
        """设置报单总笔数"""
        self._struct.OrderCount = value

    @property
    def order_action_count(self) -> int:
        """撤单总笔数"""
        return self._struct.OrderActionCount

    @order_action_count.setter
    def order_action_count(self, value: int):
        """设置撤单总笔数"""
        self._struct.OrderActionCount = value

    @property
    def for_quote_cnt(self) -> int:
        """询价总次数"""
        return self._struct.ForQuoteCnt

    @for_quote_cnt.setter
    def for_quote_cnt(self, value: int):
        """设置询价总次数"""
        self._struct.ForQuoteCnt = value

    @property
    def info_comm(self) -> float:
        """申报费"""
        return self._struct.InfoComm

    @info_comm.setter
    def info_comm(self, value: float):
        """设置申报费"""
        self._struct.InfoComm = value

    @property
    def is_opt_series(self) -> int:
        """是否期权系列"""
        return self._struct.IsOptSeries

    @is_opt_series.setter
    def is_opt_series(self, value: int):
        """设置是否期权系列"""
        self._struct.IsOptSeries = value

    @property
    def product_id(self) -> str:
        """品种代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置品种代码"""
        encoded = value.encode('ascii')[:40].ljust(41, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value

    @property
    def info_cnt(self) -> int:
        """信息量总量"""
        return self._struct.InfoCnt

    @info_cnt.setter
    def info_cnt(self, value: int):
        """设置信息量总量"""
        self._struct.InfoCnt = value



class TraderOffer(CapsuleStruct):
    """交易员报盘"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),              # 交易所代码
            ("TraderID", ctypes.c_char * 21),               # 交易所交易员代码
            ("ParticipantID", ctypes.c_char * 11),          # 会员代码
            ("Password", ctypes.c_char * 41),               # 密码
            ("InstallID", ctypes.c_int),                    # 安装编号
            ("OrderLocalID", ctypes.c_char * 13),           # 本地报单编号
            ("TraderConnectStatus", ctypes.c_char),         # 交易所交易员连接状态
            ("ConnectRequestDate", ctypes.c_char * 9),      # 发出连接请求的日期
            ("ConnectRequestTime", ctypes.c_char * 9),      # 发出连接请求的时间
            ("LastReportDate", ctypes.c_char * 9),          # 上次报告日期
            ("LastReportTime", ctypes.c_char * 9),          # 上次报告时间
            ("ConnectDate", ctypes.c_char * 9),             # 完成连接日期
            ("ConnectTime", ctypes.c_char * 9),             # 完成连接时间
            ("StartDate", ctypes.c_char * 9),               # 启动日期
            ("StartTime", ctypes.c_char * 9),               # 启动时间
            ("TradingDay", ctypes.c_char * 9),              # 交易日
            ("BrokerID", ctypes.c_char * 11),               # 经纪公司代码
        ]

    _capsule_name = "TraderOffer"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "trader_id": "TraderID",
        "participant_id": "ParticipantID",
        "password": "Password",
        "install_id": "InstallID",
        "order_local_id": "OrderLocalID",
        "trader_connect_status": "TraderConnectStatus",
        "connect_request_date": "ConnectRequestDate",
        "connect_request_time": "ConnectRequestTime",
        "last_report_date": "LastReportDate",
        "last_report_time": "LastReportTime",
        "connect_date": "ConnectDate",
        "connect_time": "ConnectTime",
        "start_date": "StartDate",
        "start_time": "StartTime",
        "trading_day": "TradingDay",
        "broker_id": "BrokerID",
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
    def trader_connect_status(self) -> str:
        """交易所交易员连接状态"""
        if 'trader_connect_status' not in self._cache:
            value = self._struct.TraderConnectStatus.decode('ascii')
            self._cache['trader_connect_status'] = value
        return self._cache['trader_connect_status']

    @trader_connect_status.setter
    def trader_connect_status(self, value: str):
        """设置交易所交易员连接状态"""
        self._struct.TraderConnectStatus = value.encode('ascii')[0]
        self._cache['trader_connect_status'] = value

    @property
    def connect_request_date(self) -> str:
        """发出连接请求的日期"""
        if 'connect_request_date' not in self._cache:
            value = self._struct.ConnectRequestDate.rstrip(b'\x00').decode('ascii')
            self._cache['connect_request_date'] = value
        return self._cache['connect_request_date']

    @connect_request_date.setter
    def connect_request_date(self, value: str):
        """设置发出连接请求的日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ConnectRequestDate = encoded
        self._cache['connect_request_date'] = value

    @property
    def connect_request_time(self) -> str:
        """发出连接请求的时间"""
        if 'connect_request_time' not in self._cache:
            value = self._struct.ConnectRequestTime.rstrip(b'\x00').decode('ascii')
            self._cache['connect_request_time'] = value
        return self._cache['connect_request_time']

    @connect_request_time.setter
    def connect_request_time(self, value: str):
        """设置发出连接请求的时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ConnectRequestTime = encoded
        self._cache['connect_request_time'] = value

    @property
    def last_report_date(self) -> str:
        """上次报告日期"""
        if 'last_report_date' not in self._cache:
            value = self._struct.LastReportDate.rstrip(b'\x00').decode('ascii')
            self._cache['last_report_date'] = value
        return self._cache['last_report_date']

    @last_report_date.setter
    def last_report_date(self, value: str):
        """设置上次报告日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.LastReportDate = encoded
        self._cache['last_report_date'] = value

    @property
    def last_report_time(self) -> str:
        """上次报告时间"""
        if 'last_report_time' not in self._cache:
            value = self._struct.LastReportTime.rstrip(b'\x00').decode('ascii')
            self._cache['last_report_time'] = value
        return self._cache['last_report_time']

    @last_report_time.setter
    def last_report_time(self, value: str):
        """设置上次报告时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.LastReportTime = encoded
        self._cache['last_report_time'] = value

    @property
    def connect_date(self) -> str:
        """完成连接日期"""
        if 'connect_date' not in self._cache:
            value = self._struct.ConnectDate.rstrip(b'\x00').decode('ascii')
            self._cache['connect_date'] = value
        return self._cache['connect_date']

    @connect_date.setter
    def connect_date(self, value: str):
        """设置完成连接日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ConnectDate = encoded
        self._cache['connect_date'] = value

    @property
    def connect_time(self) -> str:
        """完成连接时间"""
        if 'connect_time' not in self._cache:
            value = self._struct.ConnectTime.rstrip(b'\x00').decode('ascii')
            self._cache['connect_time'] = value
        return self._cache['connect_time']

    @connect_time.setter
    def connect_time(self, value: str):
        """设置完成连接时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ConnectTime = encoded
        self._cache['connect_time'] = value

    @property
    def start_date(self) -> str:
        """启动日期"""
        if 'start_date' not in self._cache:
            value = self._struct.StartDate.rstrip(b'\x00').decode('ascii')
            self._cache['start_date'] = value
        return self._cache['start_date']

    @start_date.setter
    def start_date(self, value: str):
        """设置启动日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.StartDate = encoded
        self._cache['start_date'] = value

    @property
    def start_time(self) -> str:
        """启动时间"""
        if 'start_time' not in self._cache:
            value = self._struct.StartTime.rstrip(b'\x00').decode('ascii')
            self._cache['start_time'] = value
        return self._cache['start_time']

    @start_time.setter
    def start_time(self, value: str):
        """设置启动时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.StartTime = encoded
        self._cache['start_time'] = value

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




class RiskSettleInvestPosition(CapsuleStruct):
    """投资者风险结算持仓"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
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
            ("CombPosition", ctypes.c_int),              # 组合成交形成的持仓
            ("CombLongFrozen", ctypes.c_int),            # 组合多头冻结
            ("CombShortFrozen", ctypes.c_int),           # 组合空头冻结
            ("CloseProfitByDate", ctypes.c_double),      # 逐日盯市平仓盈亏
            ("CloseProfitByTrade", ctypes.c_double),     # 逐笔对冲平仓盈亏
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
            ("TasPosition", ctypes.c_int),               # tas持仓手数
            ("TasPositionCost", ctypes.c_double),        # tas持仓成本
        ]

    _capsule_name = "RiskSettleInvstPosition"  # 注意: C++ 使用 Invst 而非 Invest

    _field_mappings = {
        "instrument_id": "InstrumentID",
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
        """持仓多空方向"""
        if 'posi_direction' not in self._cache:
            value = self._struct.PosiDirection.decode('ascii')
            self._cache['posi_direction'] = value
        return self._cache['posi_direction']

    @posi_direction.setter
    def posi_direction(self, value: str):
        """设置持仓多空方向"""
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
        """上日持仓"""
        return self._struct.YdPosition

    @yd_position.setter
    def yd_position(self, value: int):
        """设置上日持仓"""
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
        """多头冻结"""
        return self._struct.LongFrozen

    @long_frozen.setter
    def long_frozen(self, value: int):
        """设置多头冻结"""
        self._struct.LongFrozen = value

    @property
    def short_frozen(self) -> int:
        """空头冻结"""
        return self._struct.ShortFrozen

    @short_frozen.setter
    def short_frozen(self, value: int):
        """设置空头冻结"""
        self._struct.ShortFrozen = value

    @property
    def long_frozen_amount(self) -> float:
        """开仓冻结金额"""
        return self._struct.LongFrozenAmount

    @long_frozen_amount.setter
    def long_frozen_amount(self, value: float):
        """设置开仓冻结金额"""
        self._struct.LongFrozenAmount = value

    @property
    def short_frozen_amount(self) -> float:
        """开仓冻结金额"""
        return self._struct.ShortFrozenAmount

    @short_frozen_amount.setter
    def short_frozen_amount(self, value: float):
        """设置开仓冻结金额"""
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
    def today_position(self) -> int:
        """今日持仓"""
        return self._struct.TodayPosition

    @today_position.setter
    def today_position(self, value: int):
        """设置今日持仓"""
        self._struct.TodayPosition = value

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



class RiskSettleProductStatus(CapsuleStruct):
    """风险结算产品状态"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),      # 交易所代码
            ("ProductID", ctypes.c_char * 81),      # 产品编号
            ("ProductStatus", ctypes.c_char),       # 产品结算状态
        ]

    _capsule_name = "RiskSettleProductStatus"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
        "product_status": "ProductStatus",
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
        """产品编号"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品编号"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value

    @property
    def product_status(self) -> str:
        """产品结算状态"""
        if 'product_status' not in self._cache:
            value = self._struct.ProductStatus.decode('ascii')
            self._cache['product_status'] = value
        return self._cache['product_status']

    @product_status.setter
    def product_status(self, value: str):
        """设置产品结算状态"""
        self._struct.ProductStatus = value.encode('ascii')[0]
        self._cache['product_status'] = value






class UserSystemInfo(CapsuleStruct):
    """用户系统信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("UserID", ctypes.c_char * 16),               # 用户代码
            ("ClientSystemInfoLen", ctypes.c_int),        # 用户端系统内部信息长度
            ("ClientSystemInfo", ctypes.c_char * 273),    # 用户端系统内部信息
            ("reserve1", ctypes.c_char * 16),             # 保留的无效字段
            ("ClientIPPort", ctypes.c_int),               # 终端IP端口
            ("ClientLoginTime", ctypes.c_char * 9),       # 登录成功时间
            ("ClientAppID", ctypes.c_char * 33),          # App代码
            ("ClientPublicIP", ctypes.c_char * 33),       # 用户公网IP
            ("ClientLoginRemark", ctypes.c_char * 151),   # 客户登录备注2
        ]

    _capsule_name = "UserSystemInfo"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "client_system_info_len": "ClientSystemInfoLen",
        "client_system_info": "ClientSystemInfo",
        "reserve1": "reserve1",
        "client_ip_port": "ClientIPPort",
        "client_login_time": "ClientLoginTime",
        "client_app_id": "ClientAppID",
        "client_public_ip": "ClientPublicIP",
        "client_login_remark": "ClientLoginRemark",
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
    def client_system_info_len(self) -> int:
        """用户端系统内部信息长度"""
        return self._struct.ClientSystemInfoLen

    @client_system_info_len.setter
    def client_system_info_len(self, value: int):
        """设置用户端系统内部信息长度"""
        self._struct.ClientSystemInfoLen = value

    @property
    def client_system_info(self) -> str:
        """用户端系统内部信息"""
        if 'client_system_info' not in self._cache:
            value = self._struct.ClientSystemInfo.rstrip(b'\x00').decode('ascii')
            self._cache['client_system_info'] = value
        return self._cache['client_system_info']

    @client_system_info.setter
    def client_system_info(self, value: str):
        """设置用户端系统内部信息"""
        encoded = value.encode('ascii')[:272].ljust(273, b'\x00')
        self._struct.ClientSystemInfo = encoded
        self._cache['client_system_info'] = value

    @property
    def client_ip_port(self) -> int:
        """终端IP端口"""
        return self._struct.ClientIPPort

    @client_ip_port.setter
    def client_ip_port(self, value: int):
        """设置终端IP端口"""
        self._struct.ClientIPPort = value

    @property
    def client_login_time(self) -> str:
        """登录成功时间"""
        if 'client_login_time' not in self._cache:
            value = self._struct.ClientLoginTime.rstrip(b'\x00').decode('ascii')
            self._cache['client_login_time'] = value
        return self._cache['client_login_time']

    @client_login_time.setter
    def client_login_time(self, value: str):
        """设置登录成功时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ClientLoginTime = encoded
        self._cache['client_login_time'] = value

    @property
    def client_app_id(self) -> str:
        """App代码"""
        if 'client_app_id' not in self._cache:
            value = self._struct.ClientAppID.rstrip(b'\x00').decode('ascii')
            self._cache['client_app_id'] = value
        return self._cache['client_app_id']

    @client_app_id.setter
    def client_app_id(self, value: str):
        """设置App代码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientAppID = encoded
        self._cache['client_app_id'] = value

    @property
    def client_public_ip(self) -> str:
        """用户公网IP"""
        if 'client_public_ip' not in self._cache:
            value = self._struct.ClientPublicIP.rstrip(b'\x00').decode('ascii')
            self._cache['client_public_ip'] = value
        return self._cache['client_public_ip']

    @client_public_ip.setter
    def client_public_ip(self, value: str):
        """设置用户公网IP"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientPublicIP = encoded
        self._cache['client_public_ip'] = value

    @property
    def client_login_remark(self) -> str:
        """客户登录备注2"""
        if 'client_login_remark' not in self._cache:
            value = self._struct.ClientLoginRemark.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['client_login_remark'] = value
        return self._cache['client_login_remark']

    @client_login_remark.setter
    def client_login_remark(self, value: str):
        """设置客户登录备注2（GBK 编码）"""
        encoded = value.encode('gbk')[:150].ljust(151, b'\x00')
        self._struct.ClientLoginRemark = encoded
        self._cache['client_login_remark'] = value



class WechatUserSystemInfo(CapsuleStruct):
    """微信小程序等用户系统信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),             # 经纪公司代码
            ("UserID", ctypes.c_char * 16),               # 用户代码
            ("WechatCltSysInfoLen", ctypes.c_int),        # 微信小程序等用户端系统内部信息长度
            ("WechatCltSysInfo", ctypes.c_char * 273),    # 微信小程序等用户端系统内部信息
            ("ClientIPPort", ctypes.c_int),               # 终端IP端口
            ("ClientLoginTime", ctypes.c_char * 9),       # 登录成功时间
            ("ClientAppID", ctypes.c_char * 33),          # App代码
            ("ClientPublicIP", ctypes.c_char * 33),       # 用户公网IP
            ("ClientLoginRemark", ctypes.c_char * 151),   # 客户登录备注2
        ]

    _capsule_name = "WechatUserSystemInfo"

    _field_mappings = {
        "broker_id": "BrokerID",
        "user_id": "UserID",
        "wechat_clt_sys_info_len": "WechatCltSysInfoLen",
        "wechat_clt_sys_info": "WechatCltSysInfo",
        "client_ip_port": "ClientIPPort",
        "client_login_time": "ClientLoginTime",
        "client_app_id": "ClientAppID",
        "client_public_ip": "ClientPublicIP",
        "client_login_remark": "ClientLoginRemark",
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
    def wechat_clt_sys_info_len(self) -> int:
        """微信小程序等用户端系统内部信息长度"""
        return self._struct.WechatCltSysInfoLen

    @wechat_clt_sys_info_len.setter
    def wechat_clt_sys_info_len(self, value: int):
        """设置微信小程序等用户端系统内部信息长度"""
        self._struct.WechatCltSysInfoLen = value

    @property
    def wechat_clt_sys_info(self) -> str:
        """微信小程序等用户端系统内部信息"""
        if 'wechat_clt_sys_info' not in self._cache:
            value = self._struct.WechatCltSysInfo.rstrip(b'\x00').decode('ascii')
            self._cache['wechat_clt_sys_info'] = value
        return self._cache['wechat_clt_sys_info']

    @wechat_clt_sys_info.setter
    def wechat_clt_sys_info(self, value: str):
        """设置微信小程序等用户端系统内部信息"""
        encoded = value.encode('ascii')[:272].ljust(273, b'\x00')
        self._struct.WechatCltSysInfo = encoded
        self._cache['wechat_clt_sys_info'] = value

    @property
    def client_ip_port(self) -> int:
        """终端IP端口"""
        return self._struct.ClientIPPort

    @client_ip_port.setter
    def client_ip_port(self, value: int):
        """设置终端IP端口"""
        self._struct.ClientIPPort = value

    @property
    def client_login_time(self) -> str:
        """登录成功时间"""
        if 'client_login_time' not in self._cache:
            value = self._struct.ClientLoginTime.rstrip(b'\x00').decode('ascii')
            self._cache['client_login_time'] = value
        return self._cache['client_login_time']

    @client_login_time.setter
    def client_login_time(self, value: str):
        """设置登录成功时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ClientLoginTime = encoded
        self._cache['client_login_time'] = value

    @property
    def client_app_id(self) -> str:
        """App代码"""
        if 'client_app_id' not in self._cache:
            value = self._struct.ClientAppID.rstrip(b'\x00').decode('ascii')
            self._cache['client_app_id'] = value
        return self._cache['client_app_id']

    @client_app_id.setter
    def client_app_id(self, value: str):
        """设置App代码"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientAppID = encoded
        self._cache['client_app_id'] = value

    @property
    def client_public_ip(self) -> str:
        """用户公网IP"""
        if 'client_public_ip' not in self._cache:
            value = self._struct.ClientPublicIP.rstrip(b'\x00').decode('ascii')
            self._cache['client_public_ip'] = value
        return self._cache['client_public_ip']

    @client_public_ip.setter
    def client_public_ip(self, value: str):
        """设置用户公网IP"""
        encoded = value.encode('ascii')[:32].ljust(33, b'\x00')
        self._struct.ClientPublicIP = encoded
        self._cache['client_public_ip'] = value

    @property
    def client_login_remark(self) -> str:
        """客户登录备注2"""
        if 'client_login_remark' not in self._cache:
            value = self._struct.ClientLoginRemark.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['client_login_remark'] = value
        return self._cache['client_login_remark']

    @client_login_remark.setter
    def client_login_remark(self, value: str):
        """设置客户登录备注2（GBK 编码）"""
        encoded = value.encode('gbk')[:150].ljust(151, b'\x00')
        self._struct.ClientLoginRemark = encoded
        self._cache['client_login_remark'] = value




