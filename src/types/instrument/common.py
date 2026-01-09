"""
Instrument
"""
import ctypes
from PcCTP.types.base import CapsuleStruct


class Product(CapsuleStruct):
    """产品"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ProductName", ctypes.c_char * 31),         # 产品名称
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ProductClass", ctypes.c_char),             # 产品类型
            ("VolumeMultiple", ctypes.c_int),            # 合约数量乘数
            ("PriceTick", ctypes.c_double),              # 最小变动价位
            ("MaxMarketOrderVolume", ctypes.c_int),      # 市价单最大下单量
            ("MinMarketOrderVolume", ctypes.c_int),      # 市价单最小下单量
            ("MaxLimitOrderVolume", ctypes.c_int),       # 限价单最大下单量
            ("MinLimitOrderVolume", ctypes.c_int),       # 限价单最小下单量
            ("PositionType", ctypes.c_char),             # 持仓类型
            ("PositionDateType", ctypes.c_char),         # 持仓日期类型
            ("CloseDealType", ctypes.c_char),            # 平仓处理类型
            ("TradeCurrencyID", ctypes.c_char * 4),      # 交易币种
            ("MortgageFundUseRange", ctypes.c_char),     # 质押资金可用范围
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("UnderlyingMultiple", ctypes.c_double),     # 合约基础商品乘数
            ("ProductID", ctypes.c_char * 31),           # 产品代码
            ("ExchangeProductID", ctypes.c_char * 31),   # 交易所产品代码
            ("OpenLimitControlLevel", ctypes.c_char),    # 开仓量限制粒度
            ("OrderFreqControlLevel", ctypes.c_char),    # 报单频率控制粒度
        ]

    _capsule_name = "Product"

    _field_mappings = {
        "reserve1": "reserve1",
        "product_name": "ProductName",
        "exchange_id": "ExchangeID",
        "product_class": "ProductClass",
        "volume_multiple": "VolumeMultiple",
        "price_tick": "PriceTick",
        "max_market_order_volume": "MaxMarketOrderVolume",
        "min_market_order_volume": "MinMarketOrderVolume",
        "max_limit_order_volume": "MaxLimitOrderVolume",
        "min_limit_order_volume": "MinLimitOrderVolume",
        "position_type": "PositionType",
        "position_date_type": "PositionDateType",
        "close_deal_type": "CloseDealType",
        "trade_currency_id": "TradeCurrencyID",
        "mortgage_fund_use_range": "MortgageFundUseRange",
        "reserve2": "reserve2",
        "underlying_multiple": "UnderlyingMultiple",
        "product_id": "ProductID",
        "exchange_product_id": "ExchangeProductID",
        "open_limit_control_level": "OpenLimitControlLevel",
        "order_freq_control_level": "OrderFreqControlLevel",
    }

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
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value

    @property
    def product_name(self) -> str:
        """产品名称（GBK 编码）"""
        if 'product_name' not in self._cache:
            value = self._struct.ProductName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['product_name'] = value
        return self._cache['product_name']

    @product_name.setter
    def product_name(self, value: str):
        """设置产品名称（GBK 编码）"""
        encoded = value.encode('gbk')[:30].ljust(31, b'\x00')
        self._struct.ProductName = encoded
        self._cache['product_name'] = value

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
    def product_class(self) -> str:
        """产品类型"""
        if 'product_class' not in self._cache:
            value = self._struct.ProductClass.decode('ascii')
            self._cache['product_class'] = value
        return self._cache['product_class']

    @product_class.setter
    def product_class(self, value: str):
        """设置产品类型"""
        self._struct.ProductClass = value.encode('ascii')[0]
        self._cache['product_class'] = value

    @property
    def volume_multiple(self) -> int:
        """合约数量乘数"""
        return self._struct.VolumeMultiple

    @volume_multiple.setter
    def volume_multiple(self, value: int):
        """设置合约数量乘数"""
        self._struct.VolumeMultiple = value

    @property
    def price_tick(self) -> float:
        """最小变动价位"""
        return self._struct.PriceTick

    @price_tick.setter
    def price_tick(self, value: float):
        """设置最小变动价位"""
        self._struct.PriceTick = value

    @property
    def max_market_order_volume(self) -> int:
        """市价单最大下单量"""
        return self._struct.MaxMarketOrderVolume

    @max_market_order_volume.setter
    def max_market_order_volume(self, value: int):
        """设置市价单最大下单量"""
        self._struct.MaxMarketOrderVolume = value

    @property
    def min_market_order_volume(self) -> int:
        """市价单最小下单量"""
        return self._struct.MinMarketOrderVolume

    @min_market_order_volume.setter
    def min_market_order_volume(self, value: int):
        """设置市价单最小下单量"""
        self._struct.MinMarketOrderVolume = value

    @property
    def max_limit_order_volume(self) -> int:
        """限价单最大下单量"""
        return self._struct.MaxLimitOrderVolume

    @max_limit_order_volume.setter
    def max_limit_order_volume(self, value: int):
        """设置限价单最大下单量"""
        self._struct.MaxLimitOrderVolume = value

    @property
    def min_limit_order_volume(self) -> int:
        """限价单最小下单量"""
        return self._struct.MinLimitOrderVolume

    @min_limit_order_volume.setter
    def min_limit_order_volume(self, value: int):
        """设置限价单最小下单量"""
        self._struct.MinLimitOrderVolume = value

    @property
    def position_type(self) -> str:
        """持仓类型"""
        if 'position_type' not in self._cache:
            value = self._struct.PositionType.decode('ascii')
            self._cache['position_type'] = value
        return self._cache['position_type']

    @position_type.setter
    def position_type(self, value: str):
        """设置持仓类型"""
        self._struct.PositionType = value.encode('ascii')[0]
        self._cache['position_type'] = value

    @property
    def position_date_type(self) -> str:
        """持仓日期类型"""
        if 'position_date_type' not in self._cache:
            value = self._struct.PositionDateType.decode('ascii')
            self._cache['position_date_type'] = value
        return self._cache['position_date_type']

    @position_date_type.setter
    def position_date_type(self, value: str):
        """设置持仓日期类型"""
        self._struct.PositionDateType = value.encode('ascii')[0]
        self._cache['position_date_type'] = value

    @property
    def close_deal_type(self) -> str:
        """平仓处理类型"""
        if 'close_deal_type' not in self._cache:
            value = self._struct.CloseDealType.decode('ascii')
            self._cache['close_deal_type'] = value
        return self._cache['close_deal_type']

    @close_deal_type.setter
    def close_deal_type(self, value: str):
        """设置平仓处理类型"""
        self._struct.CloseDealType = value.encode('ascii')[0]
        self._cache['close_deal_type'] = value

    @property
    def trade_currency_id(self) -> str:
        """交易币种"""
        if 'trade_currency_id' not in self._cache:
            value = self._struct.TradeCurrencyID.rstrip(b'\x00').decode('ascii')
            self._cache['trade_currency_id'] = value
        return self._cache['trade_currency_id']

    @trade_currency_id.setter
    def trade_currency_id(self, value: str):
        """设置交易币种"""
        encoded = value.encode('ascii')[:3].ljust(4, b'\x00')
        self._struct.TradeCurrencyID = encoded
        self._cache['trade_currency_id'] = value

    @property
    def mortgage_fund_use_range(self) -> str:
        """质押资金可用范围"""
        if 'mortgage_fund_use_range' not in self._cache:
            value = self._struct.MortgageFundUseRange.decode('ascii')
            self._cache['mortgage_fund_use_range'] = value
        return self._cache['mortgage_fund_use_range']

    @mortgage_fund_use_range.setter
    def mortgage_fund_use_range(self, value: str):
        """设置质押资金可用范围"""
        self._struct.MortgageFundUseRange = value.encode('ascii')[0]
        self._cache['mortgage_fund_use_range'] = value

    @property
    def underlying_multiple(self) -> float:
        """合约基础商品乘数"""
        return self._struct.UnderlyingMultiple

    @underlying_multiple.setter
    def underlying_multiple(self, value: float):
        """设置合约基础商品乘数"""
        self._struct.UnderlyingMultiple = value

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
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.ExchangeProductID = encoded
        self._cache['exchange_product_id'] = value

    @property
    def open_limit_control_level(self) -> str:
        """开仓量限制粒度"""
        if 'open_limit_control_level' not in self._cache:
            value = self._struct.OpenLimitControlLevel.decode('ascii')
            self._cache['open_limit_control_level'] = value
        return self._cache['open_limit_control_level']

    @open_limit_control_level.setter
    def open_limit_control_level(self, value: str):
        """设置开仓量限制粒度"""
        self._struct.OpenLimitControlLevel = value.encode('ascii')[0]
        self._cache['open_limit_control_level'] = value

    @property
    def order_freq_control_level(self) -> str:
        """报单频率控制粒度"""
        if 'order_freq_control_level' not in self._cache:
            value = self._struct.OrderFreqControlLevel.decode('ascii')
            self._cache['order_freq_control_level'] = value
        return self._cache['order_freq_control_level']

    @order_freq_control_level.setter
    def order_freq_control_level(self, value: str):
        """设置报单频率控制粒度"""
        self._struct.OrderFreqControlLevel = value.encode('ascii')[0]
        self._cache['order_freq_control_level'] = value


# =============================================================================
# Instrument - 合约
# =============================================================================


class Instrument(CapsuleStruct):
    """合约"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InstrumentName", ctypes.c_char * 81),      # 合约名称
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("reserve3", ctypes.c_char * 31),            # 保留的无效字段
            ("ProductClass", ctypes.c_char),             # 产品类型
            ("DeliveryYear", ctypes.c_int),              # 交割年份
            ("DeliveryMonth", ctypes.c_int),             # 交割月
            ("MaxMarketOrderVolume", ctypes.c_int),      # 市价单最大下单量
            ("MinMarketOrderVolume", ctypes.c_int),      # 市价单最小下单量
            ("MaxLimitOrderVolume", ctypes.c_int),       # 限价单最大下单量
            ("MinLimitOrderVolume", ctypes.c_int),       # 限价单最小下单量
            ("VolumeMultiple", ctypes.c_int),            # 合约数量乘数
            ("PriceTick", ctypes.c_double),              # 最小变动价位
            ("CreateDate", ctypes.c_char * 9),           # 创建日
            ("OpenDate", ctypes.c_char * 9),             # 上市日期
            ("ExpireDate", ctypes.c_char * 9),           # 到期日
            ("StartDelivDate", ctypes.c_char * 9),       # 开始交割日
            ("EndDelivDate", ctypes.c_char * 9),         # 结束交割日
            ("InstLifePhase", ctypes.c_char),            # 合约生命周期状态
            ("IsTrading", ctypes.c_int),                 # 当前是否交易
            ("PositionType", ctypes.c_char),             # 持仓类型
            ("PositionDateType", ctypes.c_char),         # 持仓日期类型
            ("LongMarginRatio", ctypes.c_double),        # 多头保证金率
            ("ShortMarginRatio", ctypes.c_double),       # 空头保证金率
            ("MaxMarginSideAlgorithm", ctypes.c_char),   # 是否使用大额单边保证金算法
            ("reserve4", ctypes.c_char * 31),            # 保留的无效字段
            ("StrikePrice", ctypes.c_double),            # 执行价
            ("OptionsType", ctypes.c_char),              # 期权类型
            ("UnderlyingMultiple", ctypes.c_double),     # 合约基础商品乘数
            ("CombinationType", ctypes.c_char),          # 组合类型
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 81),      # 合约在交易所的代码
            ("ProductID", ctypes.c_char * 31),           # 产品代码
            ("UnderlyingInstrID", ctypes.c_char * 81),   # 基础商品代码
        ]

    _capsule_name = "Instrument"

    _field_mappings = {
        "reserve1": "reserve1",
        "exchange_id": "ExchangeID",
        "instrument_name": "InstrumentName",
        "reserve2": "reserve2",
        "reserve3": "reserve3",
        "product_class": "ProductClass",
        "delivery_year": "DeliveryYear",
        "delivery_month": "DeliveryMonth",
        "max_market_order_volume": "MaxMarketOrderVolume",
        "min_market_order_volume": "MinMarketOrderVolume",
        "max_limit_order_volume": "MaxLimitOrderVolume",
        "min_limit_order_volume": "MinLimitOrderVolume",
        "volume_multiple": "VolumeMultiple",
        "price_tick": "PriceTick",
        "create_date": "CreateDate",
        "open_date": "OpenDate",
        "expire_date": "ExpireDate",
        "start_deliv_date": "StartDelivDate",
        "end_deliv_date": "EndDelivDate",
        "inst_life_phase": "InstLifePhase",
        "is_trading": "IsTrading",
        "position_type": "PositionType",
        "position_date_type": "PositionDateType",
        "long_margin_ratio": "LongMarginRatio",
        "short_margin_ratio": "ShortMarginRatio",
        "max_margin_side_algorithm": "MaxMarginSideAlgorithm",
        "reserve4": "reserve4",
        "strike_price": "StrikePrice",
        "options_type": "OptionsType",
        "underlying_multiple": "UnderlyingMultiple",
        "combination_type": "CombinationType",
        "instrument_id": "InstrumentID",
        "exchange_inst_id": "ExchangeInstID",
        "product_id": "ProductID",
        "underlying_instr_id": "UnderlyingInstrID",
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
    def instrument_name(self) -> str:
        """合约名称（GBK 编码）"""
        if 'instrument_name' not in self._cache:
            value = self._struct.InstrumentName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['instrument_name'] = value
        return self._cache['instrument_name']

    @instrument_name.setter
    def instrument_name(self, value: str):
        """设置合约名称（GBK 编码）"""
        encoded = value.encode('gbk')[:80].ljust(81, b'\x00')
        self._struct.InstrumentName = encoded
        self._cache['instrument_name'] = value

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
    def product_id(self) -> str:
        """产品代码"""
        if 'product_id' not in self._cache:
            value = self._struct.ProductID.rstrip(b'\x00').decode('ascii')
            self._cache['product_id'] = value
        return self._cache['product_id']

    @product_id.setter
    def product_id(self, value: str):
        """设置产品代码"""
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
        self._struct.ProductID = encoded
        self._cache['product_id'] = value

    @property
    def product_class(self) -> str:
        """产品类型"""
        if 'product_class' not in self._cache:
            value = self._struct.ProductClass.decode('ascii')
            self._cache['product_class'] = value
        return self._cache['product_class']

    @product_class.setter
    def product_class(self, value: str):
        """设置产品类型"""
        self._struct.ProductClass = value.encode('ascii')[0]
        self._cache['product_class'] = value

    @property
    def delivery_year(self) -> int:
        """交割年份"""
        return self._struct.DeliveryYear

    @delivery_year.setter
    def delivery_year(self, value: int):
        """设置交割年份"""
        self._struct.DeliveryYear = value

    @property
    def delivery_month(self) -> int:
        """交割月"""
        return self._struct.DeliveryMonth

    @delivery_month.setter
    def delivery_month(self, value: int):
        """设置交割月"""
        self._struct.DeliveryMonth = value

    @property
    def max_market_order_volume(self) -> int:
        """市价单最大下单量"""
        return self._struct.MaxMarketOrderVolume

    @max_market_order_volume.setter
    def max_market_order_volume(self, value: int):
        """设置市价单最大下单量"""
        self._struct.MaxMarketOrderVolume = value

    @property
    def min_market_order_volume(self) -> int:
        """市价单最小下单量"""
        return self._struct.MinMarketOrderVolume

    @min_market_order_volume.setter
    def min_market_order_volume(self, value: int):
        """设置市价单最小下单量"""
        self._struct.MinMarketOrderVolume = value

    @property
    def max_limit_order_volume(self) -> int:
        """限价单最大下单量"""
        return self._struct.MaxLimitOrderVolume

    @max_limit_order_volume.setter
    def max_limit_order_volume(self, value: int):
        """设置限价单最大下单量"""
        self._struct.MaxLimitOrderVolume = value

    @property
    def min_limit_order_volume(self) -> int:
        """限价单最小下单量"""
        return self._struct.MinLimitOrderVolume

    @min_limit_order_volume.setter
    def min_limit_order_volume(self, value: int):
        """设置限价单最小下单量"""
        self._struct.MinLimitOrderVolume = value

    @property
    def volume_multiple(self) -> int:
        """合约数量乘数"""
        return self._struct.VolumeMultiple

    @volume_multiple.setter
    def volume_multiple(self, value: int):
        """设置合约数量乘数"""
        self._struct.VolumeMultiple = value

    @property
    def price_tick(self) -> float:
        """最小变动价位"""
        return self._struct.PriceTick

    @price_tick.setter
    def price_tick(self, value: float):
        """设置最小变动价位"""
        self._struct.PriceTick = value

    @property
    def create_date(self) -> str:
        """创建日"""
        if 'create_date' not in self._cache:
            value = self._struct.CreateDate.rstrip(b'\x00').decode('ascii')
            self._cache['create_date'] = value
        return self._cache['create_date']

    @create_date.setter
    def create_date(self, value: str):
        """设置创建日"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.CreateDate = encoded
        self._cache['create_date'] = value

    @property
    def open_date(self) -> str:
        """上市日期"""
        if 'open_date' not in self._cache:
            value = self._struct.OpenDate.rstrip(b'\x00').decode('ascii')
            self._cache['open_date'] = value
        return self._cache['open_date']

    @open_date.setter
    def open_date(self, value: str):
        """设置上市日期"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.OpenDate = encoded
        self._cache['open_date'] = value

    @property
    def expire_date(self) -> str:
        """到期日"""
        if 'expire_date' not in self._cache:
            value = self._struct.ExpireDate.rstrip(b'\x00').decode('ascii')
            self._cache['expire_date'] = value
        return self._cache['expire_date']

    @expire_date.setter
    def expire_date(self, value: str):
        """设置到期日"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.ExpireDate = encoded
        self._cache['expire_date'] = value

    @property
    def start_deliv_date(self) -> str:
        """开始交割日"""
        if 'start_deliv_date' not in self._cache:
            value = self._struct.StartDelivDate.rstrip(b'\x00').decode('ascii')
            self._cache['start_deliv_date'] = value
        return self._cache['start_deliv_date']

    @start_deliv_date.setter
    def start_deliv_date(self, value: str):
        """设置开始交割日"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.StartDelivDate = encoded
        self._cache['start_deliv_date'] = value

    @property
    def end_deliv_date(self) -> str:
        """结束交割日"""
        if 'end_deliv_date' not in self._cache:
            value = self._struct.EndDelivDate.rstrip(b'\x00').decode('ascii')
            self._cache['end_deliv_date'] = value
        return self._cache['end_deliv_date']

    @end_deliv_date.setter
    def end_deliv_date(self, value: str):
        """设置结束交割日"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.EndDelivDate = encoded
        self._cache['end_deliv_date'] = value

    @property
    def inst_life_phase(self) -> str:
        """合约生命周期状态"""
        if 'inst_life_phase' not in self._cache:
            value = self._struct.InstLifePhase.decode('ascii')
            self._cache['inst_life_phase'] = value
        return self._cache['inst_life_phase']

    @inst_life_phase.setter
    def inst_life_phase(self, value: str):
        """设置合约生命周期状态"""
        self._struct.InstLifePhase = value.encode('ascii')[0]
        self._cache['inst_life_phase'] = value

    @property
    def is_trading(self) -> int:
        """当前是否交易"""
        return self._struct.IsTrading

    @is_trading.setter
    def is_trading(self, value: int):
        """设置当前是否交易"""
        self._struct.IsTrading = value

    @property
    def position_type(self) -> str:
        """持仓类型"""
        if 'position_type' not in self._cache:
            value = self._struct.PositionType.decode('ascii')
            self._cache['position_type'] = value
        return self._cache['position_type']

    @position_type.setter
    def position_type(self, value: str):
        """设置持仓类型"""
        self._struct.PositionType = value.encode('ascii')[0]
        self._cache['position_type'] = value

    @property
    def position_date_type(self) -> str:
        """持仓日期类型"""
        if 'position_date_type' not in self._cache:
            value = self._struct.PositionDateType.decode('ascii')
            self._cache['position_date_type'] = value
        return self._cache['position_date_type']

    @position_date_type.setter
    def position_date_type(self, value: str):
        """设置持仓日期类型"""
        self._struct.PositionDateType = value.encode('ascii')[0]
        self._cache['position_date_type'] = value

    @property
    def long_margin_ratio(self) -> float:
        """多头保证金率"""
        return self._struct.LongMarginRatio

    @long_margin_ratio.setter
    def long_margin_ratio(self, value: float):
        """设置多头保证金率"""
        self._struct.LongMarginRatio = value

    @property
    def short_margin_ratio(self) -> float:
        """空头保证金率"""
        return self._struct.ShortMarginRatio

    @short_margin_ratio.setter
    def short_margin_ratio(self, value: float):
        """设置空头保证金率"""
        self._struct.ShortMarginRatio = value

    @property
    def max_margin_side_algorithm(self) -> str:
        """最大保证金侧算法"""
        if 'max_margin_side_algorithm' not in self._cache:
            value = self._struct.MaxMarginSideAlgorithm.decode('ascii')
            self._cache['max_margin_side_algorithm'] = value
        return self._cache['max_margin_side_algorithm']

    @max_margin_side_algorithm.setter
    def max_margin_side_algorithm(self, value: str):
        """设置最大保证金侧算法"""
        self._struct.MaxMarginSideAlgorithm = value.encode('ascii')[0]
        self._cache['max_margin_side_algorithm'] = value

    @property
    def underlying_instr_id(self) -> str:
        """基础合约代码"""
        if 'underlying_instr_id' not in self._cache:
            value = self._struct.UnderlyingInstrID.rstrip(b'\x00').decode('ascii')
            self._cache['underlying_instr_id'] = value
        return self._cache['underlying_instr_id']

    @underlying_instr_id.setter
    def underlying_instr_id(self, value: str):
        """设置基础合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.UnderlyingInstrID = encoded
        self._cache['underlying_instr_id'] = value

    @property
    def strike_price(self) -> float:
        """执行价"""
        return self._struct.StrikePrice

    @strike_price.setter
    def strike_price(self, value: float):
        """设置执行价"""
        self._struct.StrikePrice = value

    @property
    def options_type(self) -> str:
        """期权类型"""
        if 'options_type' not in self._cache:
            value = self._struct.OptionsType.decode('ascii')
            self._cache['options_type'] = value
        return self._cache['options_type']

    @options_type.setter
    def options_type(self, value: str):
        """设置期权类型"""
        self._struct.OptionsType = value.encode('ascii')[0]
        self._cache['options_type'] = value

    @property
    def underlying_multiple(self) -> float:
        """基础合约乘数"""
        return self._struct.UnderlyingMultiple

    @underlying_multiple.setter
    def underlying_multiple(self, value: float):
        """设置基础合约乘数"""
        self._struct.UnderlyingMultiple = value

    @property
    def combination_type(self) -> str:
        """组合类型"""
        if 'combination_type' not in self._cache:
            value = self._struct.CombinationType.decode('ascii')
            self._cache['combination_type'] = value
        return self._cache['combination_type']

    @combination_type.setter
    def combination_type(self, value: str):
        """设置组合类型"""
        self._struct.CombinationType = value.encode('ascii')[0]
        self._cache['combination_type'] = value



class Exchange(CapsuleStruct):
    """交易所"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ExchangeName", ctypes.c_char * 31),        # 交易所名称
            ("ExchangeProperty", ctypes.c_char),         # 交易所属性
        ]

    _capsule_name = "Exchange"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "exchange_name": "ExchangeName",
        "exchange_property": "ExchangeProperty",
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
    def exchange_name(self) -> str:
        """交易所名称（GBK 编码）"""
        if 'exchange_name' not in self._cache:
            value = self._struct.ExchangeName.rstrip(b'\x00').decode('gbk', errors='ignore')
            self._cache['exchange_name'] = value
        return self._cache['exchange_name']

    @exchange_name.setter
    def exchange_name(self, value: str):
        """设置交易所名称（GBK 编码）"""
        encoded = value.encode('gbk')[:30].ljust(31, b'\x00')
        self._struct.ExchangeName = encoded
        self._cache['exchange_name'] = value

    @property
    def exchange_property(self) -> str:
        """交易所属性"""
        if 'exchange_property' not in self._cache:
            value = self._struct.ExchangeProperty.decode('ascii')
            self._cache['exchange_property'] = value
        return self._cache['exchange_property']

    @exchange_property.setter
    def exchange_property(self, value: str):
        """设置交易所属性"""
        self._struct.ExchangeProperty = value.encode('ascii')[0]
        self._cache['exchange_property'] = value


# =============================================================================
# Product - 产品
# =============================================================================


class MulticastInstrument(CapsuleStruct):
    """组播行情合约"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("TopicID", ctypes.c_int),                 # 主题号
            ("reserve1", ctypes.c_char * 31),          # 保留字段
            ("InstrumentNo", ctypes.c_int),            # 合约编号
            ("CodePrice", ctypes.c_double),            # 基准价
            ("VolumeMultiple", ctypes.c_int),          # 合约数量乘数
            ("PriceTick", ctypes.c_double),            # 最小变动价位
            ("InstrumentID", ctypes.c_char * 81),      # 合约代码
        ]

    _capsule_name = "MulticastInstrument"

    _field_mappings = {
        "topic_id": "TopicID",
        "instrument_no": "InstrumentNo",
        "code_price": "CodePrice",
        "volume_multiple": "VolumeMultiple",
        "price_tick": "PriceTick",
        "instrument_id": "InstrumentID",
    }

    @property
    def topic_id(self) -> int:
        """主题号"""
        return self._struct.TopicID

    @topic_id.setter
    def topic_id(self, value: int):
        """设置主题号"""
        self._struct.TopicID = value
        self._cache.pop('topic_id', None)

    @property
    def instrument_no(self) -> int:
        """合约编号"""
        return self._struct.InstrumentNo

    @instrument_no.setter
    def instrument_no(self, value: int):
        """设置合约编号"""
        self._struct.InstrumentNo = value
        self._cache.pop('instrument_no', None)

    @property
    def code_price(self) -> float:
        """基准价"""
        return self._struct.CodePrice

    @code_price.setter
    def code_price(self, value: float):
        """设置基准价"""
        self._struct.CodePrice = value
        self._cache.pop('code_price', None)

    @property
    def volume_multiple(self) -> int:
        """合约数量乘数"""
        return self._struct.VolumeMultiple

    @volume_multiple.setter
    def volume_multiple(self, value: int):
        """设置合约数量乘数"""
        self._struct.VolumeMultiple = value
        self._cache.pop('volume_multiple', None)

    @property
    def price_tick(self) -> float:
        """最小变动价位"""
        return self._struct.PriceTick

    @price_tick.setter
    def price_tick(self, value: float):
        """设置最小变动价位"""
        self._struct.PriceTick = value
        self._cache.pop('price_tick', None)

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
# QryMulticastInstrument - 查询组播行情合约
# =============================================================================


class InstrumentStatus(CapsuleStruct):
    """合约状态"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("SettlementGroupID", ctypes.c_char * 9),    # 结算组代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("InstrumentStatus", ctypes.c_char),          # 合约交易状态
            ("TradingSegmentSN", ctypes.c_int),          # 交易阶段编号
            ("EnterTime", ctypes.c_char * 9),            # 进入本状态时间
            ("EnterReason", ctypes.c_char),              # 进入本状态原因
            ("ExchangeInstID", ctypes.c_char * 81),      # 合约在交易所的代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "InstrumentStatus"

    _field_mappings = {
        "exchange_id": "ExchangeID",
        "reserve1": "reserve1",
        "settlement_group_id": "SettlementGroupID",
        "reserve2": "reserve2",
        "instrument_status": "InstrumentStatus",
        "trading_segment_sn": "TradingSegmentSN",
        "enter_time": "EnterTime",
        "enter_reason": "EnterReason",
        "exchange_inst_id": "ExchangeInstID",
        "instrument_id": "InstrumentID",
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
    def settlement_group_id(self) -> str:
        """结算组代码"""
        if 'settlement_group_id' not in self._cache:
            value = self._struct.SettlementGroupID.rstrip(b'\x00').decode('ascii')
            self._cache['settlement_group_id'] = value
        return self._cache['settlement_group_id']

    @settlement_group_id.setter
    def settlement_group_id(self, value: str):
        """设置结算组代码"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.SettlementGroupID = encoded
        self._cache['settlement_group_id'] = value

    @property
    def instrument_status(self) -> str:
        """合约交易状态"""
        if 'instrument_status' not in self._cache:
            value = self._struct.InstrumentStatus.decode('ascii')
            self._cache['instrument_status'] = value
        return self._cache['instrument_status']

    @instrument_status.setter
    def instrument_status(self, value: str):
        """设置合约交易状态"""
        self._struct.InstrumentStatus = value.encode('ascii')[0]
        self._cache['instrument_status'] = value

    @property
    def trading_segment_sn(self) -> int:
        """交易阶段编号"""
        return self._struct.TradingSegmentSN

    @trading_segment_sn.setter
    def trading_segment_sn(self, value: int):
        """设置交易阶段编号"""
        self._struct.TradingSegmentSN = value

    @property
    def enter_time(self) -> str:
        """进入本状态时间"""
        if 'enter_time' not in self._cache:
            value = self._struct.EnterTime.rstrip(b'\x00').decode('ascii')
            self._cache['enter_time'] = value
        return self._cache['enter_time']

    @enter_time.setter
    def enter_time(self, value: str):
        """设置进入本状态时间"""
        encoded = value.encode('ascii')[:8].ljust(9, b'\x00')
        self._struct.EnterTime = encoded
        self._cache['enter_time'] = value

    @property
    def enter_reason(self) -> str:
        """进入本状态原因"""
        if 'enter_reason' not in self._cache:
            value = self._struct.EnterReason.decode('ascii')
            self._cache['enter_reason'] = value
        return self._cache['enter_reason']

    @enter_reason.setter
    def enter_reason(self, value: str):
        """设置进入本状态原因"""
        self._struct.EnterReason = value.encode('ascii')[0]
        self._cache['enter_reason'] = value

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



class CombInstrumentGuard(CapsuleStruct):
    """组合合约安全系数"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("BrokerID", ctypes.c_char * 11),            # 经纪公司代码
            ("reserve1", ctypes.c_char * 31),            # 保留的无效字段
            ("GuarantRatio", ctypes.c_double),           # 安全系数
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
        ]

    _capsule_name = "CombInstrumentGuard"

    _field_mappings = {
        "broker_id": "BrokerID",
        "reserve1": "reserve1",
        "guarant_ratio": "GuarantRatio",
        "exchange_id": "ExchangeID",
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
    def guarant_ratio(self) -> float:
        """安全系数"""
        return self._struct.GuarantRatio

    @guarant_ratio.setter
    def guarant_ratio(self, value: float):
        """设置安全系数"""
        self._struct.GuarantRatio = value

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



class CombAction(CapsuleStruct):
    """组合申请"""

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
            ("ActionLocalID", ctypes.c_char * 13),       # 本地申请组合编号
            ("ExchangeID", ctypes.c_char * 9),           # 交易所代码
            ("ParticipantID", ctypes.c_char * 11),       # 会员代码
            ("ClientID", ctypes.c_char * 11),            # 客户代码
            ("reserve2", ctypes.c_char * 31),            # 保留的无效字段
            ("TraderID", ctypes.c_char * 21),            # 交易所交易员代码
            ("InstallID", ctypes.c_int),                 # 安装编号
            ("ActionStatus", ctypes.c_char),             # 组合状态
            ("NotifySequence", ctypes.c_int),            # 报单提示序号
            ("TradingDay", ctypes.c_char * 9),           # 交易日
            ("SettlementID", ctypes.c_int),              # 结算编号
            ("SequenceNo", ctypes.c_int),                # 序号
            ("FrontID", ctypes.c_int),                   # 前置编号
            ("SessionID", ctypes.c_int),                 # 会话编号
            ("UserProductInfo", ctypes.c_char * 11),     # 用户端产品信息
            ("StatusMsg", ctypes.c_char * 81),           # 状态信息
            ("reserve3", ctypes.c_char * 21),            # 保留的无效字段
            ("MacAddress", ctypes.c_char * 21),          # Mac地址
            ("ComTradeID", ctypes.c_char * 21),          # 组合编号
            ("BranchID", ctypes.c_char * 31),            # 营业部编号
            ("InvestUnitID", ctypes.c_char * 17),        # 投资单元代码
            ("InstrumentID", ctypes.c_char * 81),        # 合约代码
            ("ExchangeInstID", ctypes.c_char * 81),      # 合约在交易所的代码
            ("IPAddress", ctypes.c_char * 33),           # IP地址
        ]

    _capsule_name = "CombAction"

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
        "action_local_id": "ActionLocalID",
        "exchange_id": "ExchangeID",
        "participant_id": "ParticipantID",
        "client_id": "ClientID",
        "reserve2": "reserve2",
        "trader_id": "TraderID",
        "install_id": "InstallID",
        "action_status": "ActionStatus",
        "notify_sequence": "NotifySequence",
        "trading_day": "TradingDay",
        "settlement_id": "SettlementID",
        "sequence_no": "SequenceNo",
        "front_id": "FrontID",
        "session_id": "SessionID",
        "user_product_info": "UserProductInfo",
        "status_msg": "StatusMsg",
        "reserve3": "reserve3",
        "mac_address": "MacAddress",
        "com_trade_id": "ComTradeID",
        "branch_id": "BranchID",
        "invest_unit_id": "InvestUnitID",
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
    def action_local_id(self) -> str:
        """本地申请组合编号"""
        if 'action_local_id' not in self._cache:
            value = self._struct.ActionLocalID.rstrip(b'\x00').decode('ascii')
            self._cache['action_local_id'] = value
        return self._cache['action_local_id']

    @action_local_id.setter
    def action_local_id(self, value: str):
        """设置本地申请组合编号"""
        encoded = value.encode('ascii')[:12].ljust(13, b'\x00')
        self._struct.ActionLocalID = encoded
        self._cache['action_local_id'] = value

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
    def action_status(self) -> str:
        """组合状态"""
        if 'action_status' not in self._cache:
            value = self._struct.ActionStatus.decode('ascii')
            self._cache['action_status'] = value
        return self._cache['action_status']

    @action_status.setter
    def action_status(self, value: str):
        """设置组合状态"""
        self._struct.ActionStatus = value.encode('ascii')[0]
        self._cache['action_status'] = value

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
    def reserve3(self) -> str:
        """保留的无效字段"""
        if 'reserve3' not in self._cache:
            value = self._struct.reserve3.rstrip(b'\x00').decode('ascii')
            self._cache['reserve3'] = value
        return self._cache['reserve3']

    @reserve3.setter
    def reserve3(self, value: str):
        """设置保留的无效字段"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
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
    def com_trade_id(self) -> str:
        """组合编号"""
        if 'com_trade_id' not in self._cache:
            value = self._struct.ComTradeID.rstrip(b'\x00').decode('ascii')
            self._cache['com_trade_id'] = value
        return self._cache['com_trade_id']

    @com_trade_id.setter
    def com_trade_id(self, value: str):
        """设置组合编号"""
        encoded = value.encode('ascii')[:20].ljust(21, b'\x00')
        self._struct.ComTradeID = encoded
        self._cache['com_trade_id'] = value

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
        encoded = value.encode('ascii')[:30].ljust(31, b'\x00')
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



class CombLeg(CapsuleStruct):
    """组合腿"""

    class _Struct(ctypes.Structure):
        _fields_ = [
            ("CombInstrumentID", ctypes.c_char * 81),    # 组合合约代码
            ("LegID", ctypes.c_int),                      # 单腿编号
            ("LegInstrumentID", ctypes.c_char * 81),      # 单腿合约代码
            ("Direction", ctypes.c_char),                 # 买卖方向
            ("LegMultiple", ctypes.c_int),                # 单腿乘数
            ("ImplyLevel", ctypes.c_int),                 # 派生层数
        ]

    _capsule_name = "CombLeg"

    _field_mappings = {
        "comb_instrument_id": "CombInstrumentID",
        "leg_id": "LegID",
        "leg_instrument_id": "LegInstrumentID",
        "direction": "Direction",
        "leg_multiple": "LegMultiple",
        "imply_level": "ImplyLevel",
    }

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
    def leg_instrument_id(self) -> str:
        """单腿合约代码"""
        if 'leg_instrument_id' not in self._cache:
            value = self._struct.LegInstrumentID.rstrip(b'\x00').decode('ascii')
            self._cache['leg_instrument_id'] = value
        return self._cache['leg_instrument_id']

    @leg_instrument_id.setter
    def leg_instrument_id(self, value: str):
        """设置单腿合约代码"""
        encoded = value.encode('ascii')[:80].ljust(81, b'\x00')
        self._struct.LegInstrumentID = encoded
        self._cache['leg_instrument_id'] = value

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
    def leg_multiple(self) -> int:
        """单腿乘数"""
        return self._struct.LegMultiple

    @leg_multiple.setter
    def leg_multiple(self, value: int):
        """设置单腿乘数"""
        self._struct.LegMultiple = value

    @property
    def imply_level(self) -> int:
        """派生层数"""
        return self._struct.ImplyLevel

    @imply_level.setter
    def imply_level(self, value: int):
        """设置派生层数"""
        self._struct.ImplyLevel = value



