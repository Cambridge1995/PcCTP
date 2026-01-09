"""
Param
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class InvestorPortfSetting(CapsuleStruct):
    """投资者投资组合设置"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者编号
            ("HedgeFlag", ctypes.c_char),                  # 投机套保标志
            ("UsePortf", ctypes.c_int),                    # 是否开启新组保
        ]

    _capsule_name = "InvestorPortfSetting"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "hedge_flag": "HedgeFlag",
        "use_portf": "UsePortf",
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
    def use_portf(self) -> int:
        """是否开启新组保"""
        return self._struct.UsePortf

    @use_portf.setter
    def use_portf(self, value: int):
        """设置是否开启新组保"""
        self._struct.UsePortf = value



class InvestorProdRULEMargin(CapsuleStruct):
    """投资者产品RULE保证金"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("ProdFamilyCode", ctypes.c_char * 81),        # 品种代码
            ("InstrumentClass", ctypes.c_char),            # 合约类型
            ("CommodityGroupID", ctypes.c_int),            # 商品群号
            ("BStdPosition", ctypes.c_double),             # 买标准持仓
            ("SStdPosition", ctypes.c_double),             # 卖标准持仓
            ("BStdOpenFrozen", ctypes.c_double),           # 买标准开仓冻结
            ("SStdOpenFrozen", ctypes.c_double),           # 卖标准开仓冻结
            ("BStdCloseFrozen", ctypes.c_double),          # 买标准平仓冻结
            ("SStdCloseFrozen", ctypes.c_double),          # 卖标准平仓冻结
            ("IntraProdStdPosition", ctypes.c_double),    # 品种内对冲标准持仓
            ("NetStdPosition", ctypes.c_double),          # 品种内单腿标准持仓
            ("InterProdStdPosition", ctypes.c_double),    # 品种间对冲标准持仓
            ("SingleStdPosition", ctypes.c_double),       # 单腿标准持仓
            ("IntraProdMargin", ctypes.c_double),         # 品种内对锁保证金
            ("InterProdMargin", ctypes.c_double),         # 品种间对锁保证金
            ("SingleMargin", ctypes.c_double),            # 跨品种单腿保证金
            ("NonCombMargin", ctypes.c_double),           # 非组合合约保证金
            ("AddOnMargin", ctypes.c_double),             # 附加保证金
            ("ExchMargin", ctypes.c_double),              # 交易所保证金
            ("AddOnFrozenMargin", ctypes.c_double),       # 附加冻结保证金
            ("OpenFrozenMargin", ctypes.c_double),        # 开仓冻结保证金
            ("CloseFrozenMargin", ctypes.c_double),       # 平仓冻结保证金
            ("Margin", ctypes.c_double),                  # 品种保证金
            ("FrozenMargin", ctypes.c_double),            # 冻结保证金
        ]

    _capsule_name = "InvestorProdRULEMargin"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "prod_family_code": "ProdFamilyCode",
        "instrument_class": "InstrumentClass",
        "commodity_group_id": "CommodityGroupID",
        "b_std_position": "BStdPosition",
        "s_std_position": "SStdPosition",
        "b_std_open_frozen": "BStdOpenFrozen",
        "s_std_open_frozen": "SStdOpenFrozen",
        "b_std_close_frozen": "BStdCloseFrozen",
        "s_std_close_frozen": "SStdCloseFrozen",
        "intra_prod_std_position": "IntraProdStdPosition",
        "net_std_position": "NetStdPosition",
        "inter_prod_std_position": "InterProdStdPosition",
        "single_std_position": "SingleStdPosition",
        "intra_prod_margin": "IntraProdMargin",
        "inter_prod_margin": "InterProdMargin",
        "single_margin": "SingleMargin",
        "non_comb_margin": "NonCombMargin",
        "add_on_margin": "AddOnMargin",
        "exch_margin": "ExchMargin",
        "add_on_frozen_margin": "AddOnFrozenMargin",
        "open_frozen_margin": "OpenFrozenMargin",
        "close_frozen_margin": "CloseFrozenMargin",
        "margin": "Margin",
        "frozen_margin": "FrozenMargin",
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
    def instrument_class(self) -> str:
        """合约类型"""
        if 'instrument_class' not in self._cache:
            value = self._struct.InstrumentClass.decode('ascii')
            self._cache['instrument_class'] = value
        return self._cache['instrument_class']

    @instrument_class.setter
    def instrument_class(self, value: str):
        """设置合约类型"""
        self._struct.InstrumentClass = value.encode('ascii')[0]
        self._cache['instrument_class'] = value

    @property
    def commodity_group_id(self) -> int:
        """商品群号"""
        return self._struct.CommodityGroupID

    @commodity_group_id.setter
    def commodity_group_id(self, value: int):
        """设置商品群号"""
        self._struct.CommodityGroupID = value

    @property
    def b_std_position(self) -> float:
        """买标准持仓"""
        return self._struct.BStdPosition

    @b_std_position.setter
    def b_std_position(self, value: float):
        """设置买标准持仓"""
        self._struct.BStdPosition = value

    @property
    def s_std_position(self) -> float:
        """卖标准持仓"""
        return self._struct.SStdPosition

    @s_std_position.setter
    def s_std_position(self, value: float):
        """设置卖标准持仓"""
        self._struct.SStdPosition = value

    @property
    def b_std_open_frozen(self) -> float:
        """买标准开仓冻结"""
        return self._struct.BStdOpenFrozen

    @b_std_open_frozen.setter
    def b_std_open_frozen(self, value: float):
        """设置买标准开仓冻结"""
        self._struct.BStdOpenFrozen = value

    @property
    def s_std_open_frozen(self) -> float:
        """卖标准开仓冻结"""
        return self._struct.SStdOpenFrozen

    @s_std_open_frozen.setter
    def s_std_open_frozen(self, value: float):
        """设置卖标准开仓冻结"""
        self._struct.SStdOpenFrozen = value

    @property
    def b_std_close_frozen(self) -> float:
        """买标准平仓冻结"""
        return self._struct.BStdCloseFrozen

    @b_std_close_frozen.setter
    def b_std_close_frozen(self, value: float):
        """设置买标准平仓冻结"""
        self._struct.BStdCloseFrozen = value

    @property
    def s_std_close_frozen(self) -> float:
        """卖标准平仓冻结"""
        return self._struct.SStdCloseFrozen

    @s_std_close_frozen.setter
    def s_std_close_frozen(self, value: float):
        """设置卖标准平仓冻结"""
        self._struct.SStdCloseFrozen = value

    @property
    def intra_prod_std_position(self) -> float:
        """品种内对冲标准持仓"""
        return self._struct.IntraProdStdPosition

    @intra_prod_std_position.setter
    def intra_prod_std_position(self, value: float):
        """设置品种内对冲标准持仓"""
        self._struct.IntraProdStdPosition = value

    @property
    def net_std_position(self) -> float:
        """品种内单腿标准持仓"""
        return self._struct.NetStdPosition

    @net_std_position.setter
    def net_std_position(self, value: float):
        """设置品种内单腿标准持仓"""
        self._struct.NetStdPosition = value

    @property
    def inter_prod_std_position(self) -> float:
        """品种间对冲标准持仓"""
        return self._struct.InterProdStdPosition

    @inter_prod_std_position.setter
    def inter_prod_std_position(self, value: float):
        """设置品种间对冲标准持仓"""
        self._struct.InterProdStdPosition = value

    @property
    def single_std_position(self) -> float:
        """单腿标准持仓"""
        return self._struct.SingleStdPosition

    @single_std_position.setter
    def single_std_position(self, value: float):
        """设置单腿标准持仓"""
        self._struct.SingleStdPosition = value

    @property
    def intra_prod_margin(self) -> float:
        """品种内对锁保证金"""
        return self._struct.IntraProdMargin

    @intra_prod_margin.setter
    def intra_prod_margin(self, value: float):
        """设置品种内对锁保证金"""
        self._struct.IntraProdMargin = value

    @property
    def inter_prod_margin(self) -> float:
        """品种间对锁保证金"""
        return self._struct.InterProdMargin

    @inter_prod_margin.setter
    def inter_prod_margin(self, value: float):
        """设置品种间对锁保证金"""
        self._struct.InterProdMargin = value

    @property
    def single_margin(self) -> float:
        """跨品种单腿保证金"""
        return self._struct.SingleMargin

    @single_margin.setter
    def single_margin(self, value: float):
        """设置跨品种单腿保证金"""
        self._struct.SingleMargin = value

    @property
    def non_comb_margin(self) -> float:
        """非组合合约保证金"""
        return self._struct.NonCombMargin

    @non_comb_margin.setter
    def non_comb_margin(self, value: float):
        """设置非组合合约保证金"""
        self._struct.NonCombMargin = value

    @property
    def add_on_margin(self) -> float:
        """附加保证金"""
        return self._struct.AddOnMargin

    @add_on_margin.setter
    def add_on_margin(self, value: float):
        """设置附加保证金"""
        self._struct.AddOnMargin = value

    @property
    def exch_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchMargin = value

    @property
    def add_on_frozen_margin(self) -> float:
        """附加冻结保证金"""
        return self._struct.AddOnFrozenMargin

    @add_on_frozen_margin.setter
    def add_on_frozen_margin(self, value: float):
        """设置附加冻结保证金"""
        self._struct.AddOnFrozenMargin = value

    @property
    def open_frozen_margin(self) -> float:
        """开仓冻结保证金"""
        return self._struct.OpenFrozenMargin

    @open_frozen_margin.setter
    def open_frozen_margin(self, value: float):
        """设置开仓冻结保证金"""
        self._struct.OpenFrozenMargin = value

    @property
    def close_frozen_margin(self) -> float:
        """平仓冻结保证金"""
        return self._struct.CloseFrozenMargin

    @close_frozen_margin.setter
    def close_frozen_margin(self, value: float):
        """设置平仓冻结保证金"""
        self._struct.CloseFrozenMargin = value

    @property
    def margin(self) -> float:
        """品种保证金"""
        return self._struct.Margin

    @margin.setter
    def margin(self, value: float):
        """设置品种保证金"""
        self._struct.Margin = value

    @property
    def frozen_margin(self) -> float:
        """冻结保证金"""
        return self._struct.FrozenMargin

    @frozen_margin.setter
    def frozen_margin(self, value: float):
        """设置冻结保证金"""
        self._struct.FrozenMargin = value



class RULEInterParameter(CapsuleStruct):
    """RULE跨品种参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("SpreadId", ctypes.c_int),                    # 优先级
            ("InterRate", ctypes.c_double),                # 品种间对锁仓费率折扣比例
            ("Leg1ProdFamilyCode", ctypes.c_char * 81),    # 第一腿构成品种
            ("Leg2ProdFamilyCode", ctypes.c_char * 81),    # 第二腿构成品种
            ("Leg1PropFactor", ctypes.c_int),              # 腿1比例系数
            ("Leg2PropFactor", ctypes.c_int),              # 腿2比例系数
            ("CommodityGroupID", ctypes.c_int),            # 商品群号
            ("CommodityGroupName", ctypes.c_char * 81),    # 商品群名称
        ]

    _capsule_name = "RULEInterParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "spread_id": "SpreadId",
        "inter_rate": "InterRate",
        "leg1_prod_family_code": "Leg1ProdFamilyCode",
        "leg2_prod_family_code": "Leg2ProdFamilyCode",
        "leg1_prop_factor": "Leg1PropFactor",
        "leg2_prop_factor": "Leg2PropFactor",
        "commodity_group_id": "CommodityGroupID",
        "commodity_group_name": "CommodityGroupName",
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
    def spread_id(self) -> int:
        """优先级"""
        return self._struct.SpreadId

    @spread_id.setter
    def spread_id(self, value: int):
        """设置优先级"""
        self._struct.SpreadId = value

    @property
    def inter_rate(self) -> float:
        """品种间对锁仓费率折扣比例"""
        return self._struct.InterRate

    @inter_rate.setter
    def inter_rate(self, value: float):
        """设置品种间对锁仓费率折扣比例"""
        self._struct.InterRate = value

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
    def leg1_prop_factor(self) -> int:
        """腿1比例系数"""
        return self._struct.Leg1PropFactor

    @leg1_prop_factor.setter
    def leg1_prop_factor(self, value: int):
        """设置腿1比例系数"""
        self._struct.Leg1PropFactor = value

    @property
    def leg2_prop_factor(self) -> int:
        """腿2比例系数"""
        return self._struct.Leg2PropFactor

    @leg2_prop_factor.setter
    def leg2_prop_factor(self, value: int):
        """设置腿2比例系数"""
        self._struct.Leg2PropFactor = value

    @property
    def commodity_group_id(self) -> int:
        """商品群号"""
        return self._struct.CommodityGroupID

    @commodity_group_id.setter
    def commodity_group_id(self, value: int):
        """设置商品群号"""
        self._struct.CommodityGroupID = value

    @property
    def commodity_group_name(self) -> str:
        """商品群名称（GBK 编码）"""
        if 'commodity_group_name' not in self._cache:
            value = self._struct.CommodityGroupName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['commodity_group_name'] = value
        return self._cache['commodity_group_name']

    @commodity_group_name.setter
    def commodity_group_name(self, value: str):
        """设置商品群名称（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.CommodityGroupName = encoded
        self._cache['commodity_group_name'] = value



class RULEIntraParameter(CapsuleStruct):
    """RULE品种内参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("ProdFamilyCode", ctypes.c_char * 81),        # 品种代码
            ("StdInstrumentID", ctypes.c_char * 81),       # 标准合约
            ("StdInstrMargin", ctypes.c_double),           # 标准合约保证金
            ("UsualIntraRate", ctypes.c_double),           # 一般月份合约组合保证金系数
            ("DeliveryIntraRate", ctypes.c_double),        # 临近交割合约组合保证金系数
        ]

    _capsule_name = "RULEIntraParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "prod_family_code": "ProdFamilyCode",
        "std_instrument_id": "StdInstrumentID",
        "std_instr_margin": "StdInstrMargin",
        "usual_intra_rate": "UsualIntraRate",
        "delivery_intra_rate": "DeliveryIntraRate",
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

    @property
    def std_instrument_id(self) -> str:
        """标准合约"""
        if 'std_instrument_id' not in self._cache:
            value = self._struct.StdInstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['std_instrument_id'] = value
        return self._cache['std_instrument_id']

    @std_instrument_id.setter
    def std_instrument_id(self, value: str):
        """设置标准合约"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.StdInstrumentID = encoded
        self._cache['std_instrument_id'] = value

    @property
    def std_instr_margin(self) -> float:
        """标准合约保证金"""
        return self._struct.StdInstrMargin

    @std_instr_margin.setter
    def std_instr_margin(self, value: float):
        """设置标准合约保证金"""
        self._struct.StdInstrMargin = value

    @property
    def usual_intra_rate(self) -> float:
        """一般月份合约组合保证金系数"""
        return self._struct.UsualIntraRate

    @usual_intra_rate.setter
    def usual_intra_rate(self, value: float):
        """设置一般月份合约组合保证金系数"""
        self._struct.UsualIntraRate = value

    @property
    def delivery_intra_rate(self) -> float:
        """临近交割合约组合保证金系数"""
        return self._struct.DeliveryIntraRate

    @delivery_intra_rate.setter
    def delivery_intra_rate(self, value: float):
        """设置临近交割合约组合保证金系数"""
        self._struct.DeliveryIntraRate = value



class RULEInstrParameter(CapsuleStruct):
    """RULE合约参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
            ("InstrumentClass", ctypes.c_char),            # 合约类型
            ("StdInstrumentID", ctypes.c_char * 81),       # 标准合约
            ("BSpecRatio", ctypes.c_double),               # 投机买折算系数
            ("SSpecRatio", ctypes.c_double),               # 投机卖折算系数
            ("BHedgeRatio", ctypes.c_double),              # 套保买折算系数
            ("SHedgeRatio", ctypes.c_double),              # 套保卖折算系数
            ("BAddOnMargin", ctypes.c_double),             # 买附加风险保证金
            ("SAddOnMargin", ctypes.c_double),             # 卖附加风险保证金
            ("CommodityGroupID", ctypes.c_int),            # 商品群号
        ]

    _capsule_name = "RULEInstrParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "instrument_class": "InstrumentClass",
        "std_instrument_id": "StdInstrumentID",
        "b_spec_ratio": "BSpecRatio",
        "s_spec_ratio": "SSpecRatio",
        "b_hedge_ratio": "BHedgeRatio",
        "s_hedge_ratio": "SHedgeRatio",
        "b_add_on_margin": "BAddOnMargin",
        "s_add_on_margin": "SAddOnMargin",
        "commodity_group_id": "CommodityGroupID",
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
    def instrument_class(self) -> str:
        """合约类型"""
        if 'instrument_class' not in self._cache:
            value = self._struct.InstrumentClass.decode('ascii')
            self._cache['instrument_class'] = value
        return self._cache['instrument_class']

    @instrument_class.setter
    def instrument_class(self, value: str):
        """设置合约类型"""
        self._struct.InstrumentClass = value.encode('ascii')[0]
        self._cache['instrument_class'] = value

    @property
    def std_instrument_id(self) -> str:
        """标准合约"""
        if 'std_instrument_id' not in self._cache:
            value = self._struct.StdInstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['std_instrument_id'] = value
        return self._cache['std_instrument_id']

    @std_instrument_id.setter
    def std_instrument_id(self, value: str):
        """设置标准合约"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.StdInstrumentID = encoded
        self._cache['std_instrument_id'] = value

    @property
    def b_spec_ratio(self) -> float:
        """投机买折算系数"""
        return self._struct.BSpecRatio

    @b_spec_ratio.setter
    def b_spec_ratio(self, value: float):
        """设置投机买折算系数"""
        self._struct.BSpecRatio = value

    @property
    def s_spec_ratio(self) -> float:
        """投机卖折算系数"""
        return self._struct.SSpecRatio

    @s_spec_ratio.setter
    def s_spec_ratio(self, value: float):
        """设置投机卖折算系数"""
        self._struct.SSpecRatio = value

    @property
    def b_hedge_ratio(self) -> float:
        """套保买折算系数"""
        return self._struct.BHedgeRatio

    @b_hedge_ratio.setter
    def b_hedge_ratio(self, value: float):
        """设置套保买折算系数"""
        self._struct.BHedgeRatio = value

    @property
    def s_hedge_ratio(self) -> float:
        """套保卖折算系数"""
        return self._struct.SHedgeRatio

    @s_hedge_ratio.setter
    def s_hedge_ratio(self, value: float):
        """设置套保卖折算系数"""
        self._struct.SHedgeRatio = value

    @property
    def b_add_on_margin(self) -> float:
        """买附加风险保证金"""
        return self._struct.BAddOnMargin

    @b_add_on_margin.setter
    def b_add_on_margin(self, value: float):
        """设置买附加风险保证金"""
        self._struct.BAddOnMargin = value

    @property
    def s_add_on_margin(self) -> float:
        """卖附加风险保证金"""
        return self._struct.SAddOnMargin

    @s_add_on_margin.setter
    def s_add_on_margin(self, value: float):
        """设置卖附加风险保证金"""
        self._struct.SAddOnMargin = value

    @property
    def commodity_group_id(self) -> int:
        """商品群号"""
        return self._struct.CommodityGroupID

    @commodity_group_id.setter
    def commodity_group_id(self, value: int):
        """设置商品群号"""
        self._struct.CommodityGroupID = value



