"""
Param
"""

import ctypes
from PcCTP.types.base import CapsuleStruct


class InvestorProdRCAMSMargin(CapsuleStruct):
    """投资者产品RCAMS保证金"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("CombProductID", ctypes.c_char * 41),         # 产品组合代码
            ("HedgeFlag", ctypes.c_char),                  # 投套标志
            ("ProductGroupID", ctypes.c_char * 41),        # 商品群代码
            ("RiskBeforeDiscount", ctypes.c_double),       # 品种组合前风险
            ("IntraInstrRisk", ctypes.c_double),           # 同合约对冲风险
            ("BPosRisk", ctypes.c_double),                 # 品种买持仓风险
            ("SPosRisk", ctypes.c_double),                 # 品种卖持仓风险
            ("IntraProdRisk", ctypes.c_double),            # 品种内对冲风险
            ("NetRisk", ctypes.c_double),                  # 品种净持仓风险
            ("InterProdRisk", ctypes.c_double),            # 品种间对冲风险
            ("ShortOptRiskAdj", ctypes.c_double),          # 空头期权风险调整
            ("OptionRoyalty", ctypes.c_double),            # 空头期权权利金
            ("MMSACloseFrozenMargin", ctypes.c_double),   # 大边组合平仓冻结保证金
            ("CloseCombFrozenMargin", ctypes.c_double),   # 策略组合平仓/行权冻结保证金
            ("CloseFrozenMargin", ctypes.c_double),       # 平仓/行权冻结保证金
            ("MMSAOpenFrozenMargin", ctypes.c_double),    # 大边组合开仓冻结保证金
            ("DeliveryOpenFrozenMargin", ctypes.c_double),# 交割月期货开仓冻结保证金
            ("OpenFrozenMargin", ctypes.c_double),        # 开仓冻结保证金
            ("UseFrozenMargin", ctypes.c_double),         # 投资者冻结保证金
            ("MMSAExchMargin", ctypes.c_double),         # 大边组合交易所持仓保证金
            ("DeliveryExchMargin", ctypes.c_double),     # 交割月期货交易所持仓保证金
            ("CombExchMargin", ctypes.c_double),         # 策略组合交易所保证金
            ("ExchMargin", ctypes.c_double),             # 交易所持仓保证金
            ("UseMargin", ctypes.c_double),              # 投资者持仓保证金
        ]

    _capsule_name = "InvestorProdRCAMSMargin"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "comb_product_id": "CombProductID",
        "hedge_flag": "HedgeFlag",
        "product_group_id": "ProductGroupID",
        "risk_before_discount": "RiskBeforeDiscount",
        "intra_instr_risk": "IntraInstrRisk",
        "b_pos_risk": "BPosRisk",
        "s_pos_risk": "SPosRisk",
        "intra_prod_risk": "IntraProdRisk",
        "net_risk": "NetRisk",
        "inter_prod_risk": "InterProdRisk",
        "short_opt_risk_adj": "ShortOptRiskAdj",
        "option_royalty": "OptionRoyalty",
        "mmsa_close_frozen_margin": "MMSACloseFrozenMargin",
        "close_comb_frozen_margin": "CloseCombFrozenMargin",
        "close_frozen_margin": "CloseFrozenMargin",
        "mmsa_open_frozen_margin": "MMSAOpenFrozenMargin",
        "delivery_open_frozen_margin": "DeliveryOpenFrozenMargin",
        "open_frozen_margin": "OpenFrozenMargin",
        "use_frozen_margin": "UseFrozenMargin",
        "mmsa_exch_margin": "MMSAExchMargin",
        "delivery_exch_margin": "DeliveryExchMargin",
        "comb_exch_margin": "CombExchMargin",
        "exch_margin": "ExchMargin",
        "use_margin": "UseMargin",
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
    def hedge_flag(self) -> str:
        """投套标志"""
        if 'hedge_flag' not in self._cache:
            value = self._struct.HedgeFlag.decode('ascii')
            self._cache['hedge_flag'] = value
        return self._cache['hedge_flag']

    @hedge_flag.setter
    def hedge_flag(self, value: str):
        """设置投套标志"""
        self._struct.HedgeFlag = value.encode('ascii')[0]
        self._cache['hedge_flag'] = value

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
    def risk_before_discount(self) -> float:
        """品种组合前风险"""
        return self._struct.RiskBeforeDiscount

    @risk_before_discount.setter
    def risk_before_discount(self, value: float):
        """设置品种组合前风险"""
        self._struct.RiskBeforeDiscount = value

    @property
    def intra_instr_risk(self) -> float:
        """同合约对冲风险"""
        return self._struct.IntraInstrRisk

    @intra_instr_risk.setter
    def intra_instr_risk(self, value: float):
        """设置同合约对冲风险"""
        self._struct.IntraInstrRisk = value

    @property
    def b_pos_risk(self) -> float:
        """品种买持仓风险"""
        return self._struct.BPosRisk

    @b_pos_risk.setter
    def b_pos_risk(self, value: float):
        """设置品种买持仓风险"""
        self._struct.BPosRisk = value

    @property
    def s_pos_risk(self) -> float:
        """品种卖持仓风险"""
        return self._struct.SPosRisk

    @s_pos_risk.setter
    def s_pos_risk(self, value: float):
        """设置品种卖持仓风险"""
        self._struct.SPosRisk = value

    @property
    def intra_prod_risk(self) -> float:
        """品种内对冲风险"""
        return self._struct.IntraProdRisk

    @intra_prod_risk.setter
    def intra_prod_risk(self, value: float):
        """设置品种内对冲风险"""
        self._struct.IntraProdRisk = value

    @property
    def net_risk(self) -> float:
        """品种净持仓风险"""
        return self._struct.NetRisk

    @net_risk.setter
    def net_risk(self, value: float):
        """设置品种净持仓风险"""
        self._struct.NetRisk = value

    @property
    def inter_prod_risk(self) -> float:
        """品种间对冲风险"""
        return self._struct.InterProdRisk

    @inter_prod_risk.setter
    def inter_prod_risk(self, value: float):
        """设置品种间对冲风险"""
        self._struct.InterProdRisk = value

    @property
    def short_opt_risk_adj(self) -> float:
        """空头期权风险调整"""
        return self._struct.ShortOptRiskAdj

    @short_opt_risk_adj.setter
    def short_opt_risk_adj(self, value: float):
        """设置空头期权风险调整"""
        self._struct.ShortOptRiskAdj = value

    @property
    def option_royalty(self) -> float:
        """空头期权权利金"""
        return self._struct.OptionRoyalty

    @option_royalty.setter
    def option_royalty(self, value: float):
        """设置空头期权权利金"""
        self._struct.OptionRoyalty = value

    @property
    def mmsa_close_frozen_margin(self) -> float:
        """大边组合平仓冻结保证金"""
        return self._struct.MMSACloseFrozenMargin

    @mmsa_close_frozen_margin.setter
    def mmsa_close_frozen_margin(self, value: float):
        """设置大边组合平仓冻结保证金"""
        self._struct.MMSACloseFrozenMargin = value

    @property
    def close_comb_frozen_margin(self) -> float:
        """策略组合平仓/行权冻结保证金"""
        return self._struct.CloseCombFrozenMargin

    @close_comb_frozen_margin.setter
    def close_comb_frozen_margin(self, value: float):
        """设置策略组合平仓/行权冻结保证金"""
        self._struct.CloseCombFrozenMargin = value

    @property
    def close_frozen_margin(self) -> float:
        """平仓/行权冻结保证金"""
        return self._struct.CloseFrozenMargin

    @close_frozen_margin.setter
    def close_frozen_margin(self, value: float):
        """设置平仓/行权冻结保证金"""
        self._struct.CloseFrozenMargin = value

    @property
    def mmsa_open_frozen_margin(self) -> float:
        """大边组合开仓冻结保证金"""
        return self._struct.MMSAOpenFrozenMargin

    @mmsa_open_frozen_margin.setter
    def mmsa_open_frozen_margin(self, value: float):
        """设置大边组合开仓冻结保证金"""
        self._struct.MMSAOpenFrozenMargin = value

    @property
    def delivery_open_frozen_margin(self) -> float:
        """交割月期货开仓冻结保证金"""
        return self._struct.DeliveryOpenFrozenMargin

    @delivery_open_frozen_margin.setter
    def delivery_open_frozen_margin(self, value: float):
        """设置交割月期货开仓冻结保证金"""
        self._struct.DeliveryOpenFrozenMargin = value

    @property
    def open_frozen_margin(self) -> float:
        """开仓冻结保证金"""
        return self._struct.OpenFrozenMargin

    @open_frozen_margin.setter
    def open_frozen_margin(self, value: float):
        """设置开仓冻结保证金"""
        self._struct.OpenFrozenMargin = value

    @property
    def use_frozen_margin(self) -> float:
        """投资者冻结保证金"""
        return self._struct.UseFrozenMargin

    @use_frozen_margin.setter
    def use_frozen_margin(self, value: float):
        """设置投资者冻结保证金"""
        self._struct.UseFrozenMargin = value

    @property
    def mmsa_exch_margin(self) -> float:
        """大边组合交易所持仓保证金"""
        return self._struct.MMSAExchMargin

    @mmsa_exch_margin.setter
    def mmsa_exch_margin(self, value: float):
        """设置大边组合交易所持仓保证金"""
        self._struct.MMSAExchMargin = value

    @property
    def delivery_exch_margin(self) -> float:
        """交割月期货交易所持仓保证金"""
        return self._struct.DeliveryExchMargin

    @delivery_exch_margin.setter
    def delivery_exch_margin(self, value: float):
        """设置交割月期货交易所持仓保证金"""
        self._struct.DeliveryExchMargin = value

    @property
    def comb_exch_margin(self) -> float:
        """策略组合交易所保证金"""
        return self._struct.CombExchMargin

    @comb_exch_margin.setter
    def comb_exch_margin(self, value: float):
        """设置策略组合交易所保证金"""
        self._struct.CombExchMargin = value

    @property
    def exch_margin(self) -> float:
        """交易所持仓保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所持仓保证金"""
        self._struct.ExchMargin = value

    @property
    def use_margin(self) -> float:
        """投资者持仓保证金"""
        return self._struct.UseMargin

    @use_margin.setter
    def use_margin(self, value: float):
        """设置投资者持仓保证金"""
        self._struct.UseMargin = value



class RCAMSInvestorCombPosition(CapsuleStruct):
    """RCAMS投资者组合持仓"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("BrokerID", ctypes.c_char * 11),              # 经纪公司代码
            ("InvestorID", ctypes.c_char * 13),            # 投资者代码
            ("InstrumentID", ctypes.c_char * 81),          # 合约代码
            ("HedgeFlag", ctypes.c_char),                  # 投套标志
            ("PosiDirection", ctypes.c_char),              # 持仓多空方向
            ("CombInstrumentID", ctypes.c_char * 81),      # 组合合约代码
            ("LegID", ctypes.c_int),                       # 单腿编号
            ("ExchangeInstID", ctypes.c_char * 81),        # 交易所组合合约代码
            ("TotalAmt", ctypes.c_int),                    # 持仓量
            ("ExchMargin", ctypes.c_double),               # 交易所保证金
            ("Margin", ctypes.c_double),                   # 投资者保证金
        ]

    _capsule_name = "RCAMSInvestorCombPosition"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "broker_id": "BrokerID",
        "investor_id": "InvestorID",
        "instrument_id": "InstrumentID",
        "hedge_flag": "HedgeFlag",
        "posi_direction": "PosiDirection",
        "comb_instrument_id": "CombInstrumentID",
        "leg_id": "LegID",
        "exchange_inst_id": "ExchangeInstID",
        "total_amt": "TotalAmt",
        "exch_margin": "ExchMargin",
        "margin": "Margin",
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
    def hedge_flag(self) -> str:
        """投套标志"""
        if 'hedge_flag' not in self._cache:
            value = self._struct.HedgeFlag.decode('ascii')
            self._cache['hedge_flag'] = value
        return self._cache['hedge_flag']

    @hedge_flag.setter
    def hedge_flag(self, value: str):
        """设置投套标志"""
        self._struct.HedgeFlag = value.encode('ascii')[0]
        self._cache['hedge_flag'] = value

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

    @property
    def leg_id(self) -> int:
        """单腿编号"""
        return self._struct.LegID

    @leg_id.setter
    def leg_id(self, value: int):
        """设置单腿编号"""
        self._struct.LegID = value

    @property
    def exchange_inst_id(self) -> str:
        """交易所组合合约代码"""
        if 'exchange_inst_id' not in self._cache:
            value = self._struct.ExchangeInstID.rstrip(b'\x00').decode('ascii')
            self._cache['exchange_inst_id'] = value
        return self._cache['exchange_inst_id']

    @exchange_inst_id.setter
    def exchange_inst_id(self, value: str):
        """设置交易所组合合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.ExchangeInstID = encoded
        self._cache['exchange_inst_id'] = value

    @property
    def total_amt(self) -> int:
        """持仓量"""
        return self._struct.TotalAmt

    @total_amt.setter
    def total_amt(self, value: int):
        """设置持仓量"""
        self._struct.TotalAmt = value

    @property
    def exch_margin(self) -> float:
        """交易所保证金"""
        return self._struct.ExchMargin

    @exch_margin.setter
    def exch_margin(self, value: float):
        """设置交易所保证金"""
        self._struct.ExchMargin = value

    @property
    def margin(self) -> float:
        """投资者保证金"""
        return self._struct.Margin

    @margin.setter
    def margin(self, value: float):
        """设置投资者保证金"""
        self._struct.Margin = value



class RCAMSShortOptAdjustParam(CapsuleStruct):
    """RCAMS卖方期权调整参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("CombProductID", ctypes.c_char * 41),         # 产品组合代码
            ("HedgeFlag", ctypes.c_char),                  # 投套标志
            ("AdjustValue", ctypes.c_double),              # 空头期权风险调整标准
        ]

    _capsule_name = "RCAMSShortOptAdjustParam"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "comb_product_id": "CombProductID",
        "hedge_flag": "HedgeFlag",
        "adjust_value": "AdjustValue",
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
    def hedge_flag(self) -> str:
        """投套标志"""
        if 'hedge_flag' not in self._cache:
            value = self._struct.HedgeFlag.decode('ascii')
            self._cache['hedge_flag'] = value
        return self._cache['hedge_flag']

    @hedge_flag.setter
    def hedge_flag(self, value: str):
        """设置投套标志"""
        self._struct.HedgeFlag = value.encode('ascii')[0]
        self._cache['hedge_flag'] = value

    @property
    def adjust_value(self) -> float:
        """空头期权风险调整标准"""
        return self._struct.AdjustValue

    @adjust_value.setter
    def adjust_value(self, value: float):
        """设置空头期权风险调整标准"""
        self._struct.AdjustValue = value



class RCAMSInterParameter(CapsuleStruct):
    """RCAMS跨品种参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("ProductGroupID", ctypes.c_char * 41),        # 商品群代码
            ("Priority", ctypes.c_int),                    # 优先级
            ("CreditRate", ctypes.c_double),               # 折抵率
            ("CombProduct1", ctypes.c_char * 41),          # 产品组合代码1
            ("CombProduct2", ctypes.c_char * 41),          # 产品组合代码2
        ]

    _capsule_name = "RCAMSInterParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "product_group_id": "ProductGroupID",
        "priority": "Priority",
        "credit_rate": "CreditRate",
        "comb_product1": "CombProduct1",
        "comb_product2": "CombProduct2",
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
    def priority(self) -> int:
        """优先级"""
        return self._struct.Priority

    @priority.setter
    def priority(self, value: int):
        """设置优先级"""
        self._struct.Priority = value

    @property
    def credit_rate(self) -> float:
        """折抵率"""
        return self._struct.CreditRate

    @credit_rate.setter
    def credit_rate(self, value: float):
        """设置折抵率"""
        self._struct.CreditRate = value

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



class RCAMSIntraParameter(CapsuleStruct):
    """RCAMS品种内参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("CombProductID", ctypes.c_char * 41),         # 产品组合代码
            ("HedgeRate", ctypes.c_double),                # 品种内对冲比率
        ]

    _capsule_name = "RCAMSIntraParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "comb_product_id": "CombProductID",
        "hedge_rate": "HedgeRate",
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
    def hedge_rate(self) -> float:
        """品种内对冲比率"""
        return self._struct.HedgeRate

    @hedge_rate.setter
    def hedge_rate(self, value: float):
        """设置品种内对冲比率"""
        self._struct.HedgeRate = value



class RCAMSInstrParameter(CapsuleStruct):
    """RCAMS合约参数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("ProductID", ctypes.c_char * 41),             # 产品代码
            ("HedgeRate", ctypes.c_double),                # 同合约风险对冲比率
        ]

    _capsule_name = "RCAMSInstrParameter"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
        "hedge_rate": "HedgeRate",
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
    def hedge_rate(self) -> float:
        """同合约风险对冲比率"""
        return self._struct.HedgeRate

    @hedge_rate.setter
    def hedge_rate(self, value: float):
        """设置同合约风险对冲比率"""
        self._struct.HedgeRate = value



class RCAMSCombProductInfo(CapsuleStruct):
    """RCAMS组合产品信息"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TradingDay", ctypes.c_char * 9),             # 交易日
            ("ExchangeID", ctypes.c_char * 9),             # 交易所代码
            ("ProductID", ctypes.c_char * 41),             # 产品代码
            ("CombProductID", ctypes.c_char * 41),         # 商品组代码
            ("ProductGroupID", ctypes.c_char * 41),        # 商品群代码
        ]

    _capsule_name = "RCAMSCombProductInfo"

    _field_mappings = {
        "trading_day": "TradingDay",
        "exchange_id": "ExchangeID",
        "product_id": "ProductID",
        "comb_product_id": "CombProductID",
        "product_group_id": "ProductGroupID",
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



