"""
Param
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class InvestorCommoditySPMMMargin(CapsuleStruct):
    """投资者商品SPMM记录"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("CommodityID", ctypes.c_char * 41),           # 商品组代码
            ("MarginBeforeDiscount", ctypes.c_double),     # 优惠仓位应收保证金
            ("MarginNoDiscount", ctypes.c_double),         # 不优惠仓位应收保证金
            ("LongPosRisk", ctypes.c_double),              # 多头实仓风险
            ("LongOpenFrozenRisk", ctypes.c_double),       # 多头开仓冻结风险
            ("LongCloseFrozenRisk", ctypes.c_double),      # 多头被平冻结风险
            ("ShortPosRisk", ctypes.c_double),             # 空头实仓风险
            ("ShortOpenFrozenRisk", ctypes.c_double),      # 空头开仓冻结风险
            ("ShortCloseFrozenRisk", ctypes.c_double),     # 空头被平冻结风险
            ("IntraCommodityRate", ctypes.c_double),       # SPMM品种内跨期优惠系数
            ("OptionDiscountRate", ctypes.c_double),       # SPMM期权优惠系数
            ("PosDiscount", ctypes.c_double),              # 实仓对冲优惠金额
            ("OpenFrozenDiscount", ctypes.c_double),       # 开仓报单对冲优惠金额
            ("NetRisk", ctypes.c_double),                  # 品种风险净头
            ("CloseFrozenMargin", ctypes.c_double),        # 平仓冻结保证金
            ("FrozenCommission", ctypes.c_double),         # 冻结的手续费
            ("Commission", ctypes.c_double),               # 手续费
            ("FrozenCash", ctypes.c_double),               # 冻结的资金
            ("CashIn", ctypes.c_double),                   # 资金差额
            ("StrikeFrozenMargin", ctypes.c_double),       # 行权冻结资金
        ]

    _capsule_name = "InvestorCommoditySPMMMargin"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "commodity_id": "CommodityID",
        "margin_before_discount": "MarginBeforeDiscount",
        "margin_no_discount": "MarginNoDiscount",
        "long_pos_risk": "LongPosRisk",
        "long_open_frozen_risk": "LongOpenFrozenRisk",
        "long_close_frozen_risk": "LongCloseFrozenRisk",
        "short_pos_risk": "ShortPosRisk",
        "short_open_frozen_risk": "ShortOpenFrozenRisk",
        "short_close_frozen_risk": "ShortCloseFrozenRisk",
        "intra_commodity_rate": "IntraCommodityRate",
        "option_discount_rate": "OptionDiscountRate",
        "pos_discount": "PosDiscount",
        "open_frozen_discount": "OpenFrozenDiscount",
        "net_risk": "NetRisk",
        "close_frozen_margin": "CloseFrozenMargin",
        "frozen_commission": "FrozenCommission",
        "commission": "Commission",
        "frozen_cash": "FrozenCash",
        "cash_in": "CashIn",
        "strike_frozen_margin": "StrikeFrozenMargin",
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
    def commodity_id(self) -> str:
        """商品组代码"""
        if 'commodity_id' not in self._cache:
            value = self._struct.CommodityID.rstrip(b'\x00').decode('ascii')
            self._cache['commodity_id'] = value
        return self._cache['commodity_id']

    @commodity_id.setter
    def commodity_id(self, value: str):
        """设置商品组代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.CommodityID = encoded
        self._cache['commodity_id'] = value

    @property
    def margin_before_discount(self) -> float:
        """优惠仓位应收保证金"""
        return self._struct.MarginBeforeDiscount

    @margin_before_discount.setter
    def margin_before_discount(self, value: float):
        """设置优惠仓位应收保证金"""
        self._struct.MarginBeforeDiscount = value

    @property
    def margin_no_discount(self) -> float:
        """不优惠仓位应收保证金"""
        return self._struct.MarginNoDiscount

    @margin_no_discount.setter
    def margin_no_discount(self, value: float):
        """设置不优惠仓位应收保证金"""
        self._struct.MarginNoDiscount = value

    @property
    def long_pos_risk(self) -> float:
        """多头实仓风险"""
        return self._struct.LongPosRisk

    @long_pos_risk.setter
    def long_pos_risk(self, value: float):
        """设置多头实仓风险"""
        self._struct.LongPosRisk = value

    @property
    def long_open_frozen_risk(self) -> float:
        """多头开仓冻结风险"""
        return self._struct.LongOpenFrozenRisk

    @long_open_frozen_risk.setter
    def long_open_frozen_risk(self, value: float):
        """设置多头开仓冻结风险"""
        self._struct.LongOpenFrozenRisk = value

    @property
    def long_close_frozen_risk(self) -> float:
        """多头被平冻结风险"""
        return self._struct.LongCloseFrozenRisk

    @long_close_frozen_risk.setter
    def long_close_frozen_risk(self, value: float):
        """设置多头被平冻结风险"""
        self._struct.LongCloseFrozenRisk = value

    @property
    def short_pos_risk(self) -> float:
        """空头实仓风险"""
        return self._struct.ShortPosRisk

    @short_pos_risk.setter
    def short_pos_risk(self, value: float):
        """设置空头实仓风险"""
        self._struct.ShortPosRisk = value

    @property
    def short_open_frozen_risk(self) -> float:
        """空头开仓冻结风险"""
        return self._struct.ShortOpenFrozenRisk

    @short_open_frozen_risk.setter
    def short_open_frozen_risk(self, value: float):
        """设置空头开仓冻结风险"""
        self._struct.ShortOpenFrozenRisk = value

    @property
    def short_close_frozen_risk(self) -> float:
        """空头被平冻结风险"""
        return self._struct.ShortCloseFrozenRisk

    @short_close_frozen_risk.setter
    def short_close_frozen_risk(self, value: float):
        """设置空头被平冻结风险"""
        self._struct.ShortCloseFrozenRisk = value

    @property
    def intra_commodity_rate(self) -> float:
        """SPMM品种内跨期优惠系数"""
        return self._struct.IntraCommodityRate

    @intra_commodity_rate.setter
    def intra_commodity_rate(self, value: float):
        """设置SPMM品种内跨期优惠系数"""
        self._struct.IntraCommodityRate = value

    @property
    def option_discount_rate(self) -> float:
        """SPMM期权优惠系数"""
        return self._struct.OptionDiscountRate

    @option_discount_rate.setter
    def option_discount_rate(self, value: float):
        """设置SPMM期权优惠系数"""
        self._struct.OptionDiscountRate = value

    @property
    def pos_discount(self) -> float:
        """实仓对冲优惠金额"""
        return self._struct.PosDiscount

    @pos_discount.setter
    def pos_discount(self, value: float):
        """设置实仓对冲优惠金额"""
        self._struct.PosDiscount = value

    @property
    def open_frozen_discount(self) -> float:
        """开仓报单对冲优惠金额"""
        return self._struct.OpenFrozenDiscount

    @open_frozen_discount.setter
    def open_frozen_discount(self, value: float):
        """设置开仓报单对冲优惠金额"""
        self._struct.OpenFrozenDiscount = value

    @property
    def net_risk(self) -> float:
        """品种风险净头"""
        return self._struct.NetRisk

    @net_risk.setter
    def net_risk(self, value: float):
        """设置品种风险净头"""
        self._struct.NetRisk = value

    @property
    def close_frozen_margin(self) -> float:
        """平仓冻结保证金"""
        return self._struct.CloseFrozenMargin

    @close_frozen_margin.setter
    def close_frozen_margin(self, value: float):
        """设置平仓冻结保证金"""
        self._struct.CloseFrozenMargin = value

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
    def strike_frozen_margin(self) -> float:
        """行权冻结资金"""
        return self._struct.StrikeFrozenMargin

    @strike_frozen_margin.setter
    def strike_frozen_margin(self, value: float):
        """设置行权冻结资金"""
        self._struct.StrikeFrozenMargin = value



class InvestorCommodityGroupSPMMMargin(CapsuleStruct):
    """投资者商品群SPMM记录"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("CommodityGroupID", ctypes.c_char * 41),      # 商品群代码
            ("MarginBeforeDiscount", ctypes.c_double),     # 优惠仓位应收保证金
            ("MarginNoDiscount", ctypes.c_double),         # 不优惠仓位应收保证金
            ("LongRisk", ctypes.c_double),                 # 多头风险
            ("ShortRisk", ctypes.c_double),                # 空头风险
            ("CloseFrozenMargin", ctypes.c_double),        # 商品群平仓冻结保证金
            ("InterCommodityRate", ctypes.c_double),       # SPMM跨品种优惠系数
            ("MiniMarginRatio", ctypes.c_double),          # 商品群最小保证金比例
            ("AdjustRatio", ctypes.c_double),              # 投资者保证金和交易所保证金的比例
            ("IntraCommodityDiscount", ctypes.c_double),   # SPMM品种内优惠汇总
            ("InterCommodityDiscount", ctypes.c_double),   # SPMM跨品种优惠
            ("ExchMargin", ctypes.c_double),               # 交易所保证金
            ("InvestorMargin", ctypes.c_double),           # 投资者保证金
            ("FrozenCommission", ctypes.c_double),         # 冻结的手续费
            ("Commission", ctypes.c_double),               # 手续费
            ("FrozenCash", ctypes.c_double),               # 冻结的资金
            ("CashIn", ctypes.c_double),                   # 资金差额
            ("StrikeFrozenMargin", ctypes.c_double),       # 行权冻结资金
        ]

    _capsule_name = "InvestorCommodityGroupSPMMMargin"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "commodity_group_id": "CommodityGroupID",
        "margin_before_discount": "MarginBeforeDiscount",
        "margin_no_discount": "MarginNoDiscount",
        "long_risk": "LongRisk",
        "short_risk": "ShortRisk",
        "close_frozen_margin": "CloseFrozenMargin",
        "inter_commodity_rate": "InterCommodityRate",
        "mini_margin_ratio": "MiniMarginRatio",
        "adjust_ratio": "AdjustRatio",
        "intra_commodity_discount": "IntraCommodityDiscount",
        "inter_commodity_discount": "InterCommodityDiscount",
        "exch_margin": "ExchMargin",
        "investor_margin": "InvestorMargin",
        "frozen_commission": "FrozenCommission",
        "commission": "Commission",
        "frozen_cash": "FrozenCash",
        "cash_in": "CashIn",
        "strike_frozen_margin": "StrikeFrozenMargin",
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
    def commodity_group_id(self) -> str:
        """商品群代码"""
        if 'commodity_group_id' not in self._cache:
            value = self._struct.CommodityGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['commodity_group_id'] = value
        return self._cache['commodity_group_id']

    @commodity_group_id.setter
    def commodity_group_id(self, value: str):
        """设置商品群代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.CommodityGroupID = encoded
        self._cache['commodity_group_id'] = value

    @property
    def margin_before_discount(self) -> float:
        """优惠仓位应收保证金"""
        return self._struct.MarginBeforeDiscount

    @margin_before_discount.setter
    def margin_before_discount(self, value: float):
        """设置优惠仓位应收保证金"""
        self._struct.MarginBeforeDiscount = value

    @property
    def margin_no_discount(self) -> float:
        """不优惠仓位应收保证金"""
        return self._struct.MarginNoDiscount

    @margin_no_discount.setter
    def margin_no_discount(self, value: float):
        """设置不优惠仓位应收保证金"""
        self._struct.MarginNoDiscount = value

    @property
    def long_risk(self) -> float:
        """多头风险"""
        return self._struct.LongRisk

    @long_risk.setter
    def long_risk(self, value: float):
        """设置多头风险"""
        self._struct.LongRisk = value

    @property
    def short_risk(self) -> float:
        """空头风险"""
        return self._struct.ShortRisk

    @short_risk.setter
    def short_risk(self, value: float):
        """设置空头风险"""
        self._struct.ShortRisk = value

    @property
    def close_frozen_margin(self) -> float:
        """商品群平仓冻结保证金"""
        return self._struct.CloseFrozenMargin

    @close_frozen_margin.setter
    def close_frozen_margin(self, value: float):
        """设置商品群平仓冻结保证金"""
        self._struct.CloseFrozenMargin = value

    @property
    def inter_commodity_rate(self) -> float:
        """SPMM跨品种优惠系数"""
        return self._struct.InterCommodityRate

    @inter_commodity_rate.setter
    def inter_commodity_rate(self, value: float):
        """设置SPMM跨品种优惠系数"""
        self._struct.InterCommodityRate = value

    @property
    def mini_margin_ratio(self) -> float:
        """商品群最小保证金比例"""
        return self._struct.MiniMarginRatio

    @mini_margin_ratio.setter
    def mini_margin_ratio(self, value: float):
        """设置商品群最小保证金比例"""
        self._struct.MiniMarginRatio = value

    @property
    def adjust_ratio(self) -> float:
        """投资者保证金和交易所保证金的比例"""
        return self._struct.AdjustRatio

    @adjust_ratio.setter
    def adjust_ratio(self, value: float):
        """设置投资者保证金和交易所保证金的比例"""
        self._struct.AdjustRatio = value

    @property
    def intra_commodity_discount(self) -> float:
        """SPMM品种内优惠汇总"""
        return self._struct.IntraCommodityDiscount

    @intra_commodity_discount.setter
    def intra_commodity_discount(self, value: float):
        """设置SPMM品种内优惠汇总"""
        self._struct.IntraCommodityDiscount = value

    @property
    def inter_commodity_discount(self) -> float:
        """SPMM跨品种优惠"""
        return self._struct.InterCommodityDiscount

    @inter_commodity_discount.setter
    def inter_commodity_discount(self, value: float):
        """设置SPMM跨品种优惠"""
        self._struct.InterCommodityDiscount = value

    @property
    def exch_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchMargin = value

    @property
    def investor_margin(self) -> float:
        """投资者保证金"""
        return self._struct.InvestorMargin

    @investor_margin.setter
    def investor_margin(self, value: float):
        """设置投资者保证金"""
        self._struct.InvestorMargin = value

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
    def strike_frozen_margin(self) -> float:
        """行权冻结资金"""
        return self._struct.StrikeFrozenMargin

    @strike_frozen_margin.setter
    def strike_frozen_margin(self, value: float):
        """设置行权冻结资金"""
        self._struct.StrikeFrozenMargin = value



class SPMMInstParam(CapsuleStruct):
    """SPMM合约参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
            ("InstMarginCalID", ctypes.c_char),            # SPMM合约保证金算法
            ("CommodityID", ctypes.c_char * 41),           # 商品组代码
            ("CommodityGroupID", ctypes.c_char * 41),      # 商品群代码
        ]

    _capsule_name = "SPMMInstParam"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "instrument_id": "InstrumentID",
        "inst_margin_cal_id": "InstMarginCalID",
        "commodity_id": "CommodityID",
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
    def inst_margin_cal_id(self) -> str:
        """SPMM合约保证金算法"""
        if 'inst_margin_cal_id' not in self._cache:
            value = self._struct.InstMarginCalID.decode('ascii')
            self._cache['inst_margin_cal_id'] = value
        return self._cache['inst_margin_cal_id']

    @inst_margin_cal_id.setter
    def inst_margin_cal_id(self, value: str):
        """设置SPMM合约保证金算法"""
        self._struct.InstMarginCalID = value.encode('ascii')[0]
        self._cache['inst_margin_cal_id'] = value

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
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.CommodityID = encoded
        self._cache['commodity_id'] = value

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
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.CommodityGroupID = encoded
        self._cache['commodity_group_id'] = value



class SPMMProductParam(CapsuleStruct):
    """SPMM商品参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),        # 交易所代码
            ("ProductID", ctypes.c_char * 41),         # 产品代码
            ("CommodityID", ctypes.c_char * 41),       # 商品组代码
            ("CommodityGroupID", ctypes.c_char * 41),  # 商品群代码
        ]

    _capsule_name = "SPMMProductParam"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
        "commodity_id": "CommodityID",
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




