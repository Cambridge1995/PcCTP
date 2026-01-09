"""
Param
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class SPBMFutureParameter(CapsuleStruct):
    """SPBM期货合约保证金参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
            ("ProdFamilyCode", ctypes.c_char * 81),        # 品种代码
            ("Cvf", ctypes.c_int),                         # 期货合约因子
            ("TimeRange", ctypes.c_char),                  # 阶段标识
            ("MarginRate", ctypes.c_double),               # 品种保证金标准
            ("LockRateX", ctypes.c_double),                # 期货合约内部对锁仓费率折扣比例
            ("AddOnRate", ctypes.c_double),                # 提高保证金标准
            ("PreSettlementPrice", ctypes.c_double),       # 昨结算价
            ("AddOnLockRateX2", ctypes.c_double),          # 期货合约内部对锁仓附加费率折扣比例
        ]

    _capsule_name = "SPBMFutureParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "prod_family_code": "ProdFamilyCode",
        "cvf": "Cvf",
        "time_range": "TimeRange",
        "margin_rate": "MarginRate",
        "lock_rate_x": "LockRateX",
        "add_on_rate": "AddOnRate",
        "pre_settlement_price": "PreSettlementPrice",
        "add_on_lock_rate_x2": "AddOnLockRateX2",
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
    def cvf(self) -> int:
        """期货合约因子"""
        return self._struct.Cvf

    @cvf.setter
    def cvf(self, value: int):
        """设置期货合约因子"""
        self._struct.Cvf = value

    @property
    def time_range(self) -> str:
        """阶段标识"""
        if 'time_range' not in self._cache:
            value = self._struct.TimeRange.decode('ascii')
            self._cache['time_range'] = value
        return self._cache['time_range']

    @time_range.setter
    def time_range(self, value: str):
        """设置阶段标识"""
        self._struct.TimeRange = value.encode('ascii')[0]
        self._cache['time_range'] = value

    @property
    def margin_rate(self) -> float:
        """品种保证金标准"""
        return self._struct.MarginRate

    @margin_rate.setter
    def margin_rate(self, value: float):
        """设置品种保证金标准"""
        self._struct.MarginRate = value

    @property
    def lock_rate_x(self) -> float:
        """期货合约内部对锁仓费率折扣比例"""
        return self._struct.LockRateX

    @lock_rate_x.setter
    def lock_rate_x(self, value: float):
        """设置期货合约内部对锁仓费率折扣比例"""
        self._struct.LockRateX = value

    @property
    def add_on_rate(self) -> float:
        """提高保证金标准"""
        return self._struct.AddOnRate

    @add_on_rate.setter
    def add_on_rate(self, value: float):
        """设置提高保证金标准"""
        self._struct.AddOnRate = value

    @property
    def pre_settlement_price(self) -> float:
        """昨结算价"""
        return self._struct.PreSettlementPrice

    @pre_settlement_price.setter
    def pre_settlement_price(self, value: float):
        """设置昨结算价"""
        self._struct.PreSettlementPrice = value

    @property
    def add_on_lock_rate_x2(self) -> float:
        """期货合约内部对锁仓附加费率折扣比例"""
        return self._struct.AddOnLockRateX2

    @add_on_lock_rate_x2.setter
    def add_on_lock_rate_x2(self, value: float):
        """设置期货合约内部对锁仓附加费率折扣比例"""
        self._struct.AddOnLockRateX2 = value



class SPBMOptionParameter(CapsuleStruct):
    """SPBM期权合约保证金参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
            ("ProdFamilyCode", ctypes.c_char * 81),        # 品种代码
            ("Cvf", ctypes.c_int),                         # 期权合约因子
            ("DownPrice", ctypes.c_double),                # 期权冲抵价格
            ("Delta", ctypes.c_double),                    # Delta值
            ("SlimiDelta", ctypes.c_double),               # 卖方期权风险转换最低值
            ("PreSettlementPrice", ctypes.c_double),       # 昨结算价
        ]

    _capsule_name = "SPBMOptionParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "prod_family_code": "ProdFamilyCode",
        "cvf": "Cvf",
        "down_price": "DownPrice",
        "delta": "Delta",
        "slimi_delta": "SlimiDelta",
        "pre_settlement_price": "PreSettlementPrice",
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
    def cvf(self) -> int:
        """期权合约因子"""
        return self._struct.Cvf

    @cvf.setter
    def cvf(self, value: int):
        """设置期权合约因子"""
        self._struct.Cvf = value

    @property
    def down_price(self) -> float:
        """期权冲抵价格"""
        return self._struct.DownPrice

    @down_price.setter
    def down_price(self, value: float):
        """设置期权冲抵价格"""
        self._struct.DownPrice = value

    @property
    def delta(self) -> float:
        """Delta值"""
        return self._struct.Delta

    @delta.setter
    def delta(self, value: float):
        """设置Delta值"""
        self._struct.Delta = value

    @property
    def slimi_delta(self) -> float:
        """卖方期权风险转换最低值"""
        return self._struct.SlimiDelta

    @slimi_delta.setter
    def slimi_delta(self, value: float):
        """设置卖方期权风险转换最低值"""
        self._struct.SlimiDelta = value

    @property
    def pre_settlement_price(self) -> float:
        """昨结算价"""
        return self._struct.PreSettlementPrice

    @pre_settlement_price.setter
    def pre_settlement_price(self, value: float):
        """设置昨结算价"""
        self._struct.PreSettlementPrice = value



class SPBMIntraParameter(CapsuleStruct):
    """SPBM品种内对锁仓折扣参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("ProdFamilyCode", ctypes.c_char * 81),        # 品种代码
            ("IntraRateY", ctypes.c_double),               # 品种内合约间对锁仓费率折扣比例
            ("AddOnIntraRateY2", ctypes.c_double),         # 品种内合约间对锁仓附加费率折扣比例
        ]

    _capsule_name = "SPBMIntraParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "prod_family_code": "ProdFamilyCode",
        "intra_rate_y": "IntraRateY",
        "add_on_intra_rate_y2": "AddOnIntraRateY2",
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
    def intra_rate_y(self) -> float:
        """品种内合约间对锁仓费率折扣比例"""
        return self._struct.IntraRateY

    @intra_rate_y.setter
    def intra_rate_y(self, value: float):
        """设置品种内合约间对锁仓费率折扣比例"""
        self._struct.IntraRateY = value

    @property
    def add_on_intra_rate_y2(self) -> float:
        """品种内合约间对锁仓附加费率折扣比例"""
        return self._struct.AddOnIntraRateY2

    @add_on_intra_rate_y2.setter
    def add_on_intra_rate_y2(self, value: float):
        """设置品种内合约间对锁仓附加费率折扣比例"""
        self._struct.AddOnIntraRateY2 = value



class SPBMInterParameter(CapsuleStruct):
    """SPBM跨品种抵扣参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("SpreadId", ctypes.c_int),                    # 优先级
            ("InterRateZ", ctypes.c_double),               # 品种间对锁仓费率折扣比例
            ("Leg1ProdFamilyCode", ctypes.c_char * 81),    # 第一腿构成品种
            ("Leg2ProdFamilyCode", ctypes.c_char * 81),    # 第二腿构成品种
        ]

    _capsule_name = "SPBMInterParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "spread_id": "SpreadId",
        "inter_rate_z": "InterRateZ",
        "leg1_prod_family_code": "Leg1ProdFamilyCode",
        "leg2_prod_family_code": "Leg2ProdFamilyCode",
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
    def inter_rate_z(self) -> float:
        """品种间对锁仓费率折扣比例"""
        return self._struct.InterRateZ

    @inter_rate_z.setter
    def inter_rate_z(self, value: float):
        """设置品种间对锁仓费率折扣比例"""
        self._struct.InterRateZ = value

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



class SPBMPortfDefinition(CapsuleStruct):
    """组合保证金套餐定义"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("PortfolioDefID", ctypes.c_char * 21),        # 组合保证金套餐代码
            ("ProdFamilyCode", ctypes.c_char * 81),        # 品种代码
            ("IsSPBM", ctypes.c_int),                      # 是否启用SPBM
        ]

    _capsule_name = "SPBMPortfDefinition"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "portfolio_def_id": "PortfolioDefID",
        "prod_family_code": "ProdFamilyCode",
        "is_spbm": "IsSPBM",
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
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
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

    @property
    def is_spbm(self) -> int:
        """是否启用SPBM"""
        return self._struct.IsSPBM

    @is_spbm.setter
    def is_spbm(self, value: int):
        """设置是否启用SPBM"""
        self._struct.IsSPBM = value



class SPBMInvestorPortfDef(CapsuleStruct):
    """投资者套餐选择"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("PortfolioDefID", ctypes.c_int),             # 组合保证金套餐代码
        ]

    _capsule_name = "SPBMInvestorPortfDef"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "portfolio_def_id": "PortfolioDefID",
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
    def portfolio_def_id(self) -> int:
        """组合保证金套餐代码"""
        return self._struct.PortfolioDefID

    @portfolio_def_id.setter
    def portfolio_def_id(self, value: int):
        """设置组合保证金套餐代码"""
        self._struct.PortfolioDefID = value



class InvestorPortfMarginRatio(CapsuleStruct):
    """投资者新型组合保证金系数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("InvestorRange", ctypes.c_char),              # 投资者范围
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("MarginRatio", ctypes.c_double),              # 会员对投资者收取的保证金和交易所对投资者收取的保证金的比例
            ("ProductGroupID", ctypes.c_char * 41),        # 产品群代码
        ]

    _capsule_name = "InvestorPortfMarginRatio"

    _field_mappings = {
        "investor_range": "InvestorRange",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "exchange_id": "ExchangeID",
        "margin_ratio": "MarginRatio",
        "product_group_id": "ProductGroupID",
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
    def margin_ratio(self) -> float:
        """会员对投资者收取的保证金和交易所对投资者收取的保证金的比例"""
        return self._struct.MarginRatio

    @margin_ratio.setter
    def margin_ratio(self, value: float):
        """设置会员对投资者收取的保证金和交易所对投资者收取的保证金的比例"""
        self._struct.MarginRatio = value

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
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.ProductGroupID = encoded
        self._cache['product_group_id'] = value



class InvestorProdSPBMDetail(CapsuleStruct):
    """投资者产品SPBM明细"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("ProdFamilyCode", ctypes.c_char * 81),        # 品种代码
            ("IntraInstrMargin", ctypes.c_double),         # 合约内对锁保证金
            ("BCollectingMargin", ctypes.c_double),        # 买归集保证金
            ("SCollectingMargin", ctypes.c_double),        # 卖归集保证金
            ("IntraProdMargin", ctypes.c_double),          # 品种内合约间对锁保证金
            ("NetMargin", ctypes.c_double),                # 净保证金
            ("InterProdMargin", ctypes.c_double),          # 产品间对锁保证金
            ("SingleMargin", ctypes.c_double),             # 裸保证金
            ("AddOnMargin", ctypes.c_double),              # 附加保证金
            ("DeliveryMargin", ctypes.c_double),           # 交割月保证金
            ("CallOptionMinRisk", ctypes.c_double),        # 看涨期权最低风险
            ("PutOptionMinRisk", ctypes.c_double),         # 看跌期权最低风险
            ("OptionMinRisk", ctypes.c_double),            # 卖方期权最低风险
            ("OptionValueOffset", ctypes.c_double),        # 买方期权冲抵价值
            ("OptionRoyalty", ctypes.c_double),            # 卖方期权权利金
            ("RealOptionValueOffset", ctypes.c_double),    # 价值冲抵
            ("Margin", ctypes.c_double),                   # 保证金
            ("ExchMargin", ctypes.c_double),               # 交易所保证金
        ]

    _capsule_name = "InvestorProdSPBMDetail"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "prod_family_code": "ProdFamilyCode",
        "intra_instr_margin": "IntraInstrMargin",
        "b_collecting_margin": "BCollectingMargin",
        "s_collecting_margin": "SCollectingMargin",
        "intra_prod_margin": "IntraProdMargin",
        "net_margin": "NetMargin",
        "inter_prod_margin": "InterProdMargin",
        "single_margin": "SingleMargin",
        "add_on_margin": "AddOnMargin",
        "delivery_margin": "DeliveryMargin",
        "call_option_min_risk": "CallOptionMinRisk",
        "put_option_min_risk": "PutOptionMinRisk",
        "option_min_risk": "OptionMinRisk",
        "option_value_offset": "OptionValueOffset",
        "option_royalty": "OptionRoyalty",
        "real_option_value_offset": "RealOptionValueOffset",
        "margin": "Margin",
        "exch_margin": "ExchMargin",
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
    def intra_instr_margin(self) -> float:
        """合约内对锁保证金"""
        return self._struct.IntraInstrMargin

    @intra_instr_margin.setter
    def intra_instr_margin(self, value: float):
        """设置合约内对锁保证金"""
        self._struct.IntraInstrMargin = value

    @property
    def b_collecting_margin(self) -> float:
        """买归集保证金"""
        return self._struct.BCollectingMargin

    @b_collecting_margin.setter
    def b_collecting_margin(self, value: float):
        """设置买归集保证金"""
        self._struct.BCollectingMargin = value

    @property
    def s_collecting_margin(self) -> float:
        """卖归集保证金"""
        return self._struct.SCollectingMargin

    @s_collecting_margin.setter
    def s_collecting_margin(self, value: float):
        """设置卖归集保证金"""
        self._struct.SCollectingMargin = value

    @property
    def intra_prod_margin(self) -> float:
        """品种内合约间对锁保证金"""
        return self._struct.IntraProdMargin

    @intra_prod_margin.setter
    def intra_prod_margin(self, value: float):
        """设置品种内合约间对锁保证金"""
        self._struct.IntraProdMargin = value

    @property
    def net_margin(self) -> float:
        """净保证金"""
        return self._struct.NetMargin

    @net_margin.setter
    def net_margin(self, value: float):
        """设置净保证金"""
        self._struct.NetMargin = value

    @property
    def inter_prod_margin(self) -> float:
        """产品间对锁保证金"""
        return self._struct.InterProdMargin

    @inter_prod_margin.setter
    def inter_prod_margin(self, value: float):
        """设置产品间对锁保证金"""
        self._struct.InterProdMargin = value

    @property
    def single_margin(self) -> float:
        """裸保证金"""
        return self._struct.SingleMargin

    @single_margin.setter
    def single_margin(self, value: float):
        """设置裸保证金"""
        self._struct.SingleMargin = value

    @property
    def add_on_margin(self) -> float:
        """附加保证金"""
        return self._struct.AddOnMargin

    @add_on_margin.setter
    def add_on_margin(self, value: float):
        """设置附加保证金"""
        self._struct.AddOnMargin = value

    @property
    def delivery_margin(self) -> float:
        """交割月保证金"""
        return self._struct.DeliveryMargin

    @delivery_margin.setter
    def delivery_margin(self, value: float):
        """设置交割月保证金"""
        self._struct.DeliveryMargin = value

    @property
    def call_option_min_risk(self) -> float:
        """看涨期权最低风险"""
        return self._struct.CallOptionMinRisk

    @call_option_min_risk.setter
    def call_option_min_risk(self, value: float):
        """设置看涨期权最低风险"""
        self._struct.CallOptionMinRisk = value

    @property
    def put_option_min_risk(self) -> float:
        """看跌期权最低风险"""
        return self._struct.PutOptionMinRisk

    @put_option_min_risk.setter
    def put_option_min_risk(self, value: float):
        """设置看跌期权最低风险"""
        self._struct.PutOptionMinRisk = value

    @property
    def option_min_risk(self) -> float:
        """卖方期权最低风险"""
        return self._struct.OptionMinRisk

    @option_min_risk.setter
    def option_min_risk(self, value: float):
        """设置卖方期权最低风险"""
        self._struct.OptionMinRisk = value

    @property
    def option_value_offset(self) -> float:
        """买方期权冲抵价值"""
        return self._struct.OptionValueOffset

    @option_value_offset.setter
    def option_value_offset(self, value: float):
        """设置买方期权冲抵价值"""
        self._struct.OptionValueOffset = value

    @property
    def option_royalty(self) -> float:
        """卖方期权权利金"""
        return self._struct.OptionRoyalty

    @option_royalty.setter
    def option_royalty(self, value: float):
        """设置卖方期权权利金"""
        self._struct.OptionRoyalty = value

    @property
    def real_option_value_offset(self) -> float:
        """价值冲抵"""
        return self._struct.RealOptionValueOffset

    @real_option_value_offset.setter
    def real_option_value_offset(self, value: float):
        """设置价值冲抵"""
        self._struct.RealOptionValueOffset = value

    @property
    def margin(self) -> float:
        """保证金"""
        return self._struct.Margin

    @margin.setter
    def margin(self, value: float):
        """设置保证金"""
        self._struct.Margin = value

    @property
    def exch_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchMargin = value



class SPBMAddOnInterParameter(CapsuleStruct):
    """SPBM附加跨品种参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("SpreadId", ctypes.c_int),                    # 优先级
            ("AddOnInterRateZ2", ctypes.c_double),         # 品种间对锁仓附加费率折扣比例
            ("Leg1ProdFamilyCode", ctypes.c_char * 81),    # 第一腿构成品种
            ("Leg2ProdFamilyCode", ctypes.c_char * 81),    # 第二腿构成品种
        ]

    _capsule_name = "SPBMAddOnInterParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "spread_id": "SpreadId",
        "add_on_inter_rate_z2": "AddOnInterRateZ2",
        "leg1_prod_family_code": "Leg1ProdFamilyCode",
        "leg2_prod_family_code": "Leg2ProdFamilyCode",
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
    def add_on_inter_rate_z2(self) -> float:
        """品种间对锁仓附加费率折扣比例"""
        return self._struct.AddOnInterRateZ2

    @add_on_inter_rate_z2.setter
    def add_on_inter_rate_z2(self, value: float):
        """设置品种间对锁仓附加费率折扣比例"""
        self._struct.AddOnInterRateZ2 = value

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



