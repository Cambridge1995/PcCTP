> PcCTP API - Python风格的CTP API常量定义

本文档从 `CTP常量定义.md` 转换而来，使用Python风格的命名：
- 枚举类名去掉 `Type` 后缀（如 `DirectionType` → `Direction`）
- 枚举成员去掉共同前缀（如 `THOST_FTDC_D_Buy` → `Buy`）

#### 1. ExchangeProperty

交易所属性类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `ExchangeProperty.Normal` | `0` |
| 根据成交生成报单 | `ExchangeProperty.GenOrderByTrade` | `1` |

#### 2. IdCardType

证件类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 组织机构代码 | `IdCardType.EID` | `0` |
| 中国公民身份证 | `IdCardType.IDCard` | `1` |
| 军官证 | `IdCardType.OfficerIDCard` | `2` |
| 警官证 | `IdCardType.PoliceIDCard` | `3` |
| 士兵证 | `IdCardType.SoldierIDCard` | `4` |
| 护照 | `IdCardType.Passport` | `6` |
| 港澳同胞回乡证 | `IdCardType.HomeComingCard` | `8` |
| 营业执照号 | `IdCardType.LicenseNo` | `9` |
| 税务登记号 | `IdCardType.TaxNo` | `A` |
| 台湾居民来往大陆通行证 | `IdCardType.TwMainlandTravelPermit` | `C` |
| 机动车驾驶证 | `IdCardType.DrivingLicense` | `D` |
| 当地社保ID(社会保障号) | `IdCardType.SocialID` | `F` |
| 当地身份证 | `IdCardType.LocalID` | `G` |
| 港澳永久性居民身份证 | `IdCardType.HKMCIDCard` | `I` |
| 人行开户许可证 | `IdCardType.AccountsPermits` | `J` |
| 外国人永久居留证 | `IdCardType.FrgPrmtRdCard` | `K` |
| 资管产品备案函(船员证) | `IdCardType.CptMngPrdLetter` | `L` |
| 港澳台居民居住证 | `IdCardType.HKMCTwResidencePermit` | `M` |
| 统一社会信用代码 | `IdCardType.UniformSocialCreditCode` | `N` |
| 机构成立证明文件 | `IdCardType.CorporationCertNo` | `O` |
| 其他证件 | `IdCardType.OtherCard` | `x` |

#### 3. InvestorRange

投资者范围类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有 | `InvestorRange.All` | `1` |
| 投资者组 | `InvestorRange.Group` | `2` |
| 单一投资者 | `InvestorRange.Single` | `3` |

#### 4. DepartmentRange

投资者范围类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有 | `DepartmentRange.All` | `1` |
| 组织架构 | `DepartmentRange.Group` | `2` |
| 单一投资者 | `DepartmentRange.Single` | `3` |

#### 5. DataSyncStatus

数据同步状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未同步 | `DataSyncStatus.Asynchronous` | `1` |
| 同步中 | `DataSyncStatus.Synchronizing` | `2` |
| 已同步 | `DataSyncStatus.Synchronized` | `3` |

#### 6. BrokerDataSyncStatus

经纪公司数据同步状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已同步 | `BrokerDataSyncStatus.Synchronized` | `1` |
| 同步中 | `BrokerDataSyncStatus.Synchronizing` | `2` |

#### 7. ExchangeConnectStatus

交易所连接状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 没有任何连接 | `ExchangeConnectStatus.NoConnection` | `1` |
| 已经发出合约查询请求 | `ExchangeConnectStatus.QryInstrumentSent` | `2` |
| 已经获取信息 | `ExchangeConnectStatus.GotInformation` | `9` |

#### 8. TraderConnectStatus

交易所交易员连接状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 没有任何连接 | `TraderConnectStatus.NotConnected` | `1` |
| 已经连接 | `TraderConnectStatus.Connected` | `2` |
| 已经发出合约查询请求 | `TraderConnectStatus.QryInstrumentSent` | `3` |
| 订阅私有流 | `TraderConnectStatus.SubPrivateFlow` | `4` |

#### 9. FunctionCode

功能代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 数据异步化 | `FunctionCode.DataAsync` | `1` |
| 强制用户登出 | `FunctionCode.ForceUserLogout` | `2` |
| 变更管理用户口令 | `FunctionCode.UserPasswordUpdate` | `3` |
| 变更经纪公司口令 | `FunctionCode.BrokerPasswordUpdate` | `4` |
| 变更投资者口令 | `FunctionCode.InvestorPasswordUpdate` | `5` |
| 报单插入 | `FunctionCode.OrderInsert` | `6` |
| 报单操作 | `FunctionCode.OrderAction` | `7` |
| 同步系统数据 | `FunctionCode.SyncSystemData` | `8` |
| 同步经纪公司数据 | `FunctionCode.SyncBrokerData` | `9` |
| 批量同步经纪公司数据 | `FunctionCode.BachSyncBrokerData` | `A` |
| 超级查询 | `FunctionCode.SuperQuery` | `B` |
| 预埋报单插入 | `FunctionCode.ParkedOrderInsert` | `C` |
| 预埋报单操作 | `FunctionCode.ParkedOrderAction` | `D` |
| 同步动态令牌 | `FunctionCode.SyncOTP` | `E` |
| 删除未知单 | `FunctionCode.DeleteOrder` | `F` |
| 退出紧急状态 | `FunctionCode.ExitEmergency` | `G` |

#### 10. BrokerFunctionCode

经纪公司功能代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 强制用户登出 | `BrokerFunctionCode.ForceUserLogout` | `1` |
| 变更用户口令 | `BrokerFunctionCode.UserPasswordUpdate` | `2` |
| 同步经纪公司数据 | `BrokerFunctionCode.SyncBrokerData` | `3` |
| 批量同步经纪公司数据 | `BrokerFunctionCode.BachSyncBrokerData` | `4` |
| 报单插入 | `BrokerFunctionCode.OrderInsert` | `5` |
| 报单操作 | `BrokerFunctionCode.OrderAction` | `6` |
| 全部查询 | `BrokerFunctionCode.AllQuery` | `7` |
| 系统功能：登入 | `BrokerFunctionCode.log` | `a` |
| 基本查询：查询基础数据，如合约，交易所等常量 | `BrokerFunctionCode.BaseQry` | `b` |
| 交易查询：如查成交，委托 | `BrokerFunctionCode.TradeQry` | `c` |
| 交易功能：报单，撤单 | `BrokerFunctionCode.Trade` | `d` |
| 银期转账 | `BrokerFunctionCode.Virement` | `e` |
| 风险监控 | `BrokerFunctionCode.Risk` | `f` |
| 查询 | `BrokerFunctionCode.Session` | `g` |
| 风控通知控制 | `BrokerFunctionCode.RiskNoticeCtl` | `h` |
| 风控通知发送 | `BrokerFunctionCode.RiskNotice` | `i` |
| 察看经纪公司资金权限 | `BrokerFunctionCode.BrokerDeposit` | `j` |
| 资金查询 | `BrokerFunctionCode.QueryFund` | `k` |
| 报单查询 | `BrokerFunctionCode.QueryOrder` | `l` |
| 成交查询 | `BrokerFunctionCode.QueryTrade` | `m` |
| 持仓查询 | `BrokerFunctionCode.QueryPosition` | `n` |
| 行情查询 | `BrokerFunctionCode.QueryMarketData` | `o` |
| 用户事件查询 | `BrokerFunctionCode.QueryUserEvent` | `p` |
| 风险通知查询 | `BrokerFunctionCode.QueryRiskNotify` | `q` |
| 出入金查询 | `BrokerFunctionCode.QueryFundChange` | `r` |
| 投资者信息查询 | `BrokerFunctionCode.QueryInvestor` | `s` |
| 交易编码查询 | `BrokerFunctionCode.QueryTradingCode` | `t` |
| 强平 | `BrokerFunctionCode.ForceClose` | `u` |
| 压力测试 | `BrokerFunctionCode.PressTest` | `v` |
| 权益反算 | `BrokerFunctionCode.RemainCalc` | `w` |
| 净持仓保证金指标 | `BrokerFunctionCode.NetPositionInd` | `x` |
| 风险预算 | `BrokerFunctionCode.RiskPredict` | `y` |
| 数据导出 | `BrokerFunctionCode.DataExport` | `z` |
| 风控指标设置 | `BrokerFunctionCode.RiskTargetSetup` | `A` |
| 行情预警 | `BrokerFunctionCode.MarketDataWarn` | `B` |
| 业务通知查询 | `BrokerFunctionCode.QryBizNotice` | `C` |
| 业务通知模板设置 | `BrokerFunctionCode.CfgBizNotice` | `D` |
| 同步动态令牌 | `BrokerFunctionCode.SyncOTP` | `E` |
| 发送业务通知 | `BrokerFunctionCode.SendBizNotice` | `F` |
| 风险级别标准设置 | `BrokerFunctionCode.CfgRiskLevelStd` | `G` |
| 交易终端应急功能 | `BrokerFunctionCode.TbCommand` | `H` |
| 删除未知单 | `BrokerFunctionCode.DeleteOrder` | `J` |
| 预埋报单插入 | `BrokerFunctionCode.ParkedOrderInsert` | `K` |
| 预埋报单操作 | `BrokerFunctionCode.ParkedOrderAction` | `L` |
| 资金不够仍允许行权 | `BrokerFunctionCode.ExecOrderNoCheck` | `M` |
| 指定 | `BrokerFunctionCode.Designate` | `N` |
| 证券处置 | `BrokerFunctionCode.StockDisposal` | `O` |
| 席位资金预警 | `BrokerFunctionCode.BrokerDepositWarn` | `Q` |
| 备兑不足预警 | `BrokerFunctionCode.CoverWarn` | `S` |
| 行权试算 | `BrokerFunctionCode.PreExecOrder` | `T` |
| 行权交收风险 | `BrokerFunctionCode.ExecOrderRisk` | `P` |
| 持仓限额预警 | `BrokerFunctionCode.PosiLimitWarn` | `U` |
| 持仓限额查询 | `BrokerFunctionCode.QryPosiLimit` | `V` |
| 银期签到签退 | `BrokerFunctionCode.FBSign` | `W` |
| 银期签约解约 | `BrokerFunctionCode.FBAccount` | `X` |

#### 11. OrderActionStatus

报单操作状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已经提交 | `OrderActionStatus.Submitted` | `a` |
| 已经接受 | `OrderActionStatus.Accepted` | `b` |
| 已经被拒绝 | `OrderActionStatus.Rejected` | `c` |

#### 12. OrderStatus

报单状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 全部成交 | `OrderStatus.AllTraded` | `0` |
| 部分成交还在队列中 | `OrderStatus.PartTradedQueueing` | `1` |
| 部分成交不在队列中 | `OrderStatus.PartTradedNotQueueing` | `2` |
| 未成交还在队列中 | `OrderStatus.NoTradeQueueing` | `3` |
| 未成交不在队列中 | `OrderStatus.NoTradeNotQueueing` | `4` |
| 撤单 | `OrderStatus.Canceled` | `5` |
| 未知 | `OrderStatus.Unknown` | `a` |
| 尚未触发 | `OrderStatus.NotTouched` | `b` |
| 已触发 | `OrderStatus.Touched` | `c` |

#### 13. OrderSubmitStatus

报单提交状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已经提交 | `OrderSubmitStatus.InsertSubmitted` | `0` |
| 撤单已经提交 | `OrderSubmitStatus.CancelSubmitted` | `1` |
| 修改已经提交 | `OrderSubmitStatus.ModifySubmitted` | `2` |
| 已经接受 | `OrderSubmitStatus.Accepted` | `3` |
| 报单已经被拒绝 | `OrderSubmitStatus.InsertRejected` | `4` |
| 撤单已经被拒绝 | `OrderSubmitStatus.CancelRejected` | `5` |
| 改单已经被拒绝 | `OrderSubmitStatus.ModifyRejected` | `6` |

#### 14. PositionDate

持仓日期类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 今日持仓 | `PositionDate.Today` | `1` |
| 历史持仓 | `PositionDate.History` | `2` |

#### 15. PositionDateType

持仓日期类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 使用历史持仓 | `PositionDateType.UseHistory` | `1` |
| 不使用历史持仓 | `PositionDateType.NoUseHistory` | `2` |

#### 16. TradingRole

交易角色类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 代理 | `TradingRole.Broker` | `1` |
| 自营 | `TradingRole.Host` | `2` |
| 做市商 | `TradingRole.Maker` | `3` |

#### 17. ProductClass

产品类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货 | `ProductClass.Futures` | `1` |
| 期货期权 | `ProductClass.Options` | `2` |
| 组合 | `ProductClass.Combination` | `3` |
| 即期 | `ProductClass.Spot` | `4` |
| 期转现 | `ProductClass.EFP` | `5` |
| 现货期权 | `ProductClass.SpotOption` | `6` |
| TAS合约 | `ProductClass.TAS` | `7` |
| 金属指数 | `ProductClass.MI` | `I` |

#### 18. APIProductClass

产品类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货单一合约 | `APIProductClass.FutureSingle` | `1` |
| 期权单一合约 | `APIProductClass.OptionSingle` | `2` |
| 可交易期货(含期货组合和期货单一合约) | `APIProductClass.Futures` | `3` |
| 可交易期权(含期权组合和期权单一合约) | `APIProductClass.Options` | `4` |
| 可下单套利组合 | `APIProductClass.TradingComb` | `5` |
| 可申请的组合（可以申请的组合合约 包含可以交易的合约） | `APIProductClass.UnTradingComb` | `6` |
| 所有可以交易合约 | `APIProductClass.AllTrading` | `7` |
| 所有合约（包含不能交易合约 慎用） | `APIProductClass.All` | `8` |

#### 19. InstLifePhase

合约生命周期状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未上市 | `InstLifePhase.NotStart` | `0` |
| 上市 | `InstLifePhase.Started` | `1` |
| 停牌 | `InstLifePhase.Pause` | `2` |
| 到期 | `InstLifePhase.Expired` | `3` |

#### 20. Direction

买卖方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 买 | `Direction.D_Buy` | `0` |
| 卖 | `Direction.D_Sell` | `1` |

#### 21. PositionType

持仓类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 净持仓 | `PositionType.Net` | `1` |
| 综合持仓 | `PositionType.Gross` | `2` |

#### 22. PosiDirection

持仓多空方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 净 | `PosiDirection.Net` | `1` |
| 多头 | `PosiDirection.Long` | `2` |
| 空头 | `PosiDirection.Short` | `3` |

#### 23. SysSettlementStatus

系统结算状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不活跃 | `SysSettlementStatus.NonActive` | `1` |
| 启动 | `SysSettlementStatus.Startup` | `2` |
| 操作 | `SysSettlementStatus.Operating` | `3` |
| 结算 | `SysSettlementStatus.Settlement` | `4` |
| 结算完成 | `SysSettlementStatus.SettlementFinished` | `5` |

#### 24. RatioAttr

费率属性类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易费率 | `RatioAttr.Trade` | `0` |
| 结算费率 | `RatioAttr.Settlement` | `1` |

#### 25. HedgeFlag

投机套保标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 投机 | `HedgeFlag.Speculation` | `1` |
| 套利 | `HedgeFlag.Arbitrage` | `2` |
| 套保 | `HedgeFlag.Hedge` | `3` |
| 做市商 | `HedgeFlag.MarketMaker` | `5` |
| 第一腿投机第二腿套保 | `HedgeFlag.SpecHedge` | `6` |
| 第一腿套保第二腿投机 | `HedgeFlag.HedgeSpec` | `7` |

#### 26. BillHedgeFlag

投机套保标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 投机 | `BillHedgeFlag.Speculation` | `1` |
| 套利 | `BillHedgeFlag.Arbitrage` | `2` |
| 套保 | `BillHedgeFlag.Hedge` | `3` |

#### 27. ClientIDType

交易编码类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 投机 | `ClientIDType.Speculation` | `1` |
| 套利 | `ClientIDType.Arbitrage` | `2` |
| 套保 | `ClientIDType.Hedge` | `3` |
| 做市商 | `ClientIDType.MarketMaker` | `5` |

#### 28. OrderPriceType

报单价格条件类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 任意价 | `OrderPriceType.AnyPrice` | `1` |
| 限价 | `OrderPriceType.LimitPrice` | `2` |
| 最优价 | `OrderPriceType.BestPrice` | `3` |
| 最新价 | `OrderPriceType.LastPrice` | `4` |
| 最新价浮动上浮1个ticks | `OrderPriceType.LastPricePlusOneTicks` | `5` |
| 最新价浮动上浮2个ticks | `OrderPriceType.LastPricePlusTwoTicks` | `6` |
| 最新价浮动上浮3个ticks | `OrderPriceType.LastPricePlusThreeTicks` | `7` |
| 卖一价 | `OrderPriceType.AskPrice1` | `8` |
| 卖一价浮动上浮1个ticks | `OrderPriceType.AskPrice1PlusOneTicks` | `9` |
| 卖一价浮动上浮2个ticks | `OrderPriceType.AskPrice1PlusTwoTicks` | `A` |
| 卖一价浮动上浮3个ticks | `OrderPriceType.AskPrice1PlusThreeTicks` | `B` |
| 买一价 | `OrderPriceType.BidPrice1` | `C` |
| 买一价浮动上浮1个ticks | `OrderPriceType.BidPrice1PlusOneTicks` | `D` |
| 买一价浮动上浮2个ticks | `OrderPriceType.BidPrice1PlusTwoTicks` | `E` |
| 买一价浮动上浮3个ticks | `OrderPriceType.BidPrice1PlusThreeTicks` | `F` |
| 五档价 | `OrderPriceType.FiveLevelPrice` | `G` |

#### 29. OffsetFlag

开平标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 开仓 | `OffsetFlag.Open` | `0` |
| 平仓 | `OffsetFlag.Close` | `1` |
| 强平 | `OffsetFlag.ForceClose` | `2` |
| 平今 | `OffsetFlag.CloseToday` | `3` |
| 平昨 | `OffsetFlag.CloseYesterday` | `4` |
| 强减 | `OffsetFlag.ForceOff` | `5` |
| 本地强平 | `OffsetFlag.LocalForceClose` | `6` |

#### 30. ForceCloseReason

强平原因类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 非强平 | `ForceCloseReason.NotForceClose` | `0` |
| 资金不足 | `ForceCloseReason.LackDeposit` | `1` |
| 客户超仓 | `ForceCloseReason.ClientOverPositionLimit` | `2` |
| 会员超仓 | `ForceCloseReason.MemberOverPositionLimit` | `3` |
| 持仓非整数倍 | `ForceCloseReason.NotMultiple` | `4` |
| 违规 | `ForceCloseReason.Violation` | `5` |
| 其它 | `ForceCloseReason.Other` | `6` |
| 自然人临近交割 | `ForceCloseReason.PersonDeliv` | `7` |
| 本地强平资金不足忽略敞口 | `ForceCloseReason.Notverifycapital` | `8` |
| 本地强平资金不足 | `ForceCloseReason.LocalLackDeposit` | `9` |
| 本地强平违规持仓忽略敞口 | `ForceCloseReason.LocalViolationNocheck` | `a` |
| 本地强平违规持仓 | `ForceCloseReason.LocalViolation` | `b` |

#### 31. OrderType

报单类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `OrderType.Normal` | `0` |
| 报价衍生 | `OrderType.DeriveFromQuote` | `1` |
| 组合衍生 | `OrderType.DeriveFromCombination` | `2` |
| 组合报单 | `OrderType.Combination` | `3` |
| 条件单 | `OrderType.ConditionalOrder` | `4` |
| 互换单 | `OrderType.Swap` | `5` |
| 大宗交易成交衍生 | `OrderType.DeriveFromBlockTrade` | `6` |
| 期转现成交衍生 | `OrderType.DeriveFromEFPTrade` | `7` |

#### 32. TimeCondition

有效期类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 立即完成，否则撤销 | `TimeCondition.IOC` | `1` |
| 本节有效 | `TimeCondition.GFS` | `2` |
| 当日有效 | `TimeCondition.GFD` | `3` |
| 指定日期前有效 | `TimeCondition.GTD` | `4` |
| 撤销前有效 | `TimeCondition.GTC` | `5` |
| 集合竞价有效 | `TimeCondition.GFA` | `6` |

#### 33. VolumeCondition

成交量类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 任何数量 | `VolumeCondition.AV` | `1` |
| 最小数量 | `VolumeCondition.MV` | `2` |
| 全部数量 | `VolumeCondition.CV` | `3` |

#### 34. ContingentCondition

触发条件类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 立即 | `ContingentCondition.Immediately` | `1` |
| 止损 | `ContingentCondition.Touch` | `2` |
| 止赢 | `ContingentCondition.TouchProfit` | `3` |
| 预埋单 | `ContingentCondition.ParkedOrder` | `4` |
| 最新价大于条件价 | `ContingentCondition.LastPriceGreaterThanStopPrice` | `5` |
| 最新价大于等于条件价 | `ContingentCondition.LastPriceGreaterEqualStopPrice` | `6` |
| 最新价小于条件价 | `ContingentCondition.LastPriceLesserThanStopPrice` | `7` |
| 最新价小于等于条件价 | `ContingentCondition.LastPriceLesserEqualStopPrice` | `8` |
| 卖一价大于条件价 | `ContingentCondition.AskPriceGreaterThanStopPrice` | `9` |
| 卖一价大于等于条件价 | `ContingentCondition.AskPriceGreaterEqualStopPrice` | `A` |
| 卖一价小于条件价 | `ContingentCondition.AskPriceLesserThanStopPrice` | `B` |
| 卖一价小于等于条件价 | `ContingentCondition.AskPriceLesserEqualStopPrice` | `C` |
| 买一价大于条件价 | `ContingentCondition.BidPriceGreaterThanStopPrice` | `D` |
| 买一价大于等于条件价 | `ContingentCondition.BidPriceGreaterEqualStopPrice` | `E` |
| 买一价小于条件价 | `ContingentCondition.BidPriceLesserThanStopPrice` | `F` |
| 买一价小于等于条件价 | `ContingentCondition.BidPriceLesserEqualStopPrice` | `H` |

#### 35. ActionFlag

操作标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 删除 | `ActionFlag.Delete` | `0` |
| 修改 | `ActionFlag.Modify` | `3` |

#### 36. TradingRight

交易权限类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 可以交易 | `TradingRight.Allow` | `0` |
| 只能平仓 | `TradingRight.CloseOnly` | `1` |
| 不能交易 | `TradingRight.Forbidden` | `2` |

#### 37. OrderSource

报单来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 来自参与者 | `OrderSource.Participant` | `0` |
| 来自管理员 | `OrderSource.Administrator` | `1` |

#### 38. TradeType

成交类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 组合持仓拆分为单一持仓,初始化不应包含该类型的持仓 | `TradeType.SplitCombination` | `#` |
| 普通成交 | `TradeType.Common` | `0` |
| 期权执行 | `TradeType.OptionsExecution` | `1` |
| OTC成交 | `TradeType.OTC` | `2` |
| 期转现衍生成交 | `TradeType.EFPDerived` | `3` |
| 组合衍生成交 | `TradeType.CombinationDerived` | `4` |
| 大宗交易成交 | `TradeType.BlockTrade` | `5` |

#### 39. SpecPosiType

特殊持仓明细标识类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 普通持仓明细 | `SpecPosiType.Common` | `#` |
| TAS合约成交产生的标的合约持仓明细 | `SpecPosiType.Tas` | `0` |

#### 40. PriceSource

成交价来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 前成交价 | `PriceSource.LastPrice` | `0` |
| 买委托价 | `PriceSource.Buy` | `1` |
| 卖委托价 | `PriceSource.Sell` | `2` |
| 场外成交价 | `PriceSource.OTC` | `3` |

#### 41. InstrumentStatus

合约交易状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 开盘前 | `InstrumentStatus.BeforeTrading` | `0` |
| 非交易 | `InstrumentStatus.NoTrading` | `1` |
| 连续交易 | `InstrumentStatus.Continous` | `2` |
| 集合竞价报单 | `InstrumentStatus.AuctionOrdering` | `3` |
| 集合竞价价格平衡 | `InstrumentStatus.AuctionBalance` | `4` |
| 集合竞价撮合 | `InstrumentStatus.AuctionMatch` | `5` |
| 收盘 | `InstrumentStatus.Closed` | `6` |
| 交易业务处理 | `InstrumentStatus.TransactionProcessing` | `7` |

#### 42. InstStatusEnterReason

品种进入交易状态原因类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自动切换 | `InstStatusEnterReason.Automatic` | `1` |
| 手动切换 | `InstStatusEnterReason.Manual` | `2` |
| 熔断 | `InstStatusEnterReason.Fuse` | `3` |

#### 43. BatchStatus

处理状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未上传 | `BatchStatus.NoUpload` | `1` |
| 已上传 | `BatchStatus.Uploaded` | `2` |
| 审核失败 | `BatchStatus.Failed` | `3` |

#### 44. ReturnStyle

按品种返还方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 按所有品种 | `ReturnStyle.All` | `1` |
| 按品种 | `ReturnStyle.ByProduct` | `2` |

#### 45. ReturnPattern

返还模式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 按成交手数 | `ReturnPattern.ByVolume` | `1` |
| 按留存手续费 | `ReturnPattern.ByFeeOnHand` | `2` |

#### 46. ReturnLevel

返还级别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 级别1 | `ReturnLevel.Level1` | `1` |
| 级别2 | `ReturnLevel.Level2` | `2` |
| 级别3 | `ReturnLevel.Level3` | `3` |
| 级别4 | `ReturnLevel.Level4` | `4` |
| 级别5 | `ReturnLevel.Level5` | `5` |
| 级别6 | `ReturnLevel.Level6` | `6` |
| 级别7 | `ReturnLevel.Level7` | `7` |
| 级别8 | `ReturnLevel.Level8` | `8` |
| 级别9 | `ReturnLevel.Level9` | `9` |

#### 47. ReturnStandard

返还标准类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 分阶段返还 | `ReturnStandard.ByPeriod` | `1` |
| 按某一标准 | `ReturnStandard.ByStandard` | `2` |

#### 48. MortgageType

质押类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 质出 | `MortgageType.Out` | `0` |
| 质入 | `MortgageType.In` | `1` |

#### 49. InvestorSettlementParamID

投资者结算参数代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 质押比例 | `InvestorSettlementParamID.MortgageRatio` | `4` |
| 保证金算法 | `InvestorSettlementParamID.MarginWay` | `5` |
| 结算单结存是否包含质押 | `InvestorSettlementParamID.BillDeposit` | `9` |

#### 50. ExchangeSettlementParamID

交易所结算参数代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 质押比例 | `ExchangeSettlementParamID.MortgageRatio` | `1` |
| 分项资金导入项 | `ExchangeSettlementParamID.OtherFundItem` | `2` |
| 分项资金入交易所出入金 | `ExchangeSettlementParamID.OtherFundImport` | `3` |
| 中金所开户最低可用金额 | `ExchangeSettlementParamID.CFFEXMinPrepa` | `6` |
| 郑商所结算方式 | `ExchangeSettlementParamID.CZCESettlementType` | `7` |
| 交易所交割手续费收取方式 | `ExchangeSettlementParamID.ExchDelivFeeMode` | `9` |
| 投资者交割手续费收取方式 | `ExchangeSettlementParamID.DelivFeeMode` | `0` |
| 郑商所组合持仓保证金收取方式 | `ExchangeSettlementParamID.CZCEComMarginType` | `A` |
| 大商所套利保证金是否优惠 | `ExchangeSettlementParamID.DceComMarginType` | `B` |
| 虚值期权保证金优惠比率 | `ExchangeSettlementParamID.OptOutDisCountRate` | `a` |
| 最低保障系数 | `ExchangeSettlementParamID.OptMiniGuarantee` | `b` |

#### 51. SystemParamID

系统参数代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 投资者代码最小长度 | `SystemParamID.InvestorIDMinLength` | `1` |
| 投资者账号代码最小长度 | `SystemParamID.AccountIDMinLength` | `2` |
| 投资者开户默认登录权限 | `SystemParamID.UserRightLogon` | `3` |
| 投资者交易结算单成交汇总方式 | `SystemParamID.SettlementBillTrade` | `4` |
| 统一开户更新交易编码方式 | `SystemParamID.TradingCode` | `5` |
| 结算是否判断存在未复核的出入金和分项资金 | `SystemParamID.CheckFund` | `6` |
| 是否启用手续费模板数据权限 | `SystemParamID.CommModelRight` | `7` |
| 是否启用保证金率模板数据权限 | `SystemParamID.MarginModelRight` | `9` |
| 是否规范用户才能激活 | `SystemParamID.IsStandardActive` | `8` |
| 上传的交易所结算文件路径 | `SystemParamID.UploadSettlementFile` | `U` |
| 上报保证金监控中心文件路径 | `SystemParamID.DownloadCSRCFile` | `D` |
| 生成的结算单文件路径 | `SystemParamID.SettlementBillFile` | `S` |
| 证监会文件标识 | `SystemParamID.CSRCOthersFile` | `C` |
| 投资者照片路径 | `SystemParamID.InvestorPhoto` | `P` |
| 全结经纪公司上传文件路径 | `SystemParamID.CSRCData` | `R` |
| 开户密码录入方式 | `SystemParamID.InvestorPwdModel` | `I` |
| 投资者中金所结算文件下载路径 | `SystemParamID.CFFEXInvestorSettleFile` | `F` |
| 投资者代码编码方式 | `SystemParamID.InvestorIDType` | `a` |
| 休眠户最高权益 | `SystemParamID.FreezeMaxReMain` | `r` |
| 手续费相关操作实时上场开关 | `SystemParamID.IsSync` | `A` |
| 解除开仓权限限制 | `SystemParamID.RelieveOpenLimit` | `O` |
| 是否规范用户才能休眠 | `SystemParamID.IsStandardFreeze` | `X` |
| 郑商所是否开放所有品种套保交易 | `SystemParamID.CZCENormalProductHedge` | `B` |

#### 52. TradeParamID

交易系统参数代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 系统加密算法 | `TradeParamID.EncryptionStandard` | `E` |
| 系统风险算法 | `TradeParamID.RiskMode` | `R` |
| 系统风险算法是否全局 0-否 1-是 | `TradeParamID.RiskModeGlobal` | `G` |
| 密码加密算法 | `TradeParamID.modeEncode` | `P` |
| 价格小数位数参数 | `TradeParamID.tickMode` | `T` |
| 用户最大会话数 | `TradeParamID.SingleUserSessionMaxNum` | `S` |
| 最大连续登录失败数 | `TradeParamID.LoginFailMaxNum` | `L` |
| 是否强制认证 | `TradeParamID.IsAuthForce` | `A` |
| 是否冻结证券持仓 | `TradeParamID.IsPosiFreeze` | `F` |
| 是否限仓 | `TradeParamID.IsPosiLimit` | `M` |
| 郑商所询价时间间隔 | `TradeParamID.ForQuoteTimeInterval` | `Q` |
| 是否期货限仓 | `TradeParamID.IsFuturePosiLimit` | `B` |
| 是否期货下单频率限制 | `TradeParamID.IsFutureOrderFreq` | `C` |
| 行权冻结是否计算盈利 | `TradeParamID.IsExecOrderProfit` | `H` |
| 银期开户是否验证开户银行卡号是否是预留银行账户 | `TradeParamID.IsCheckBankAcc` | `I` |
| 弱密码最后修改日期 | `TradeParamID.PasswordDeadLine` | `J` |
| 强密码校验 | `TradeParamID.IsStrongPassword` | `K` |
| 自有资金质押比 | `TradeParamID.BalanceMorgage` | `a` |
| 最小密码长度 | `TradeParamID.MinPwdLen` | `O` |
| IP当日最大登陆失败次数 | `TradeParamID.LoginFailMaxNumForIP` | `U` |
| 密码有效期 | `TradeParamID.PasswordPeriod` | `V` |
| 历史密码重复限制次数 | `TradeParamID.PwdHistoryCmp` | `X` |

#### 53. FileID

文件标识类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 资金数据 | `FileID.SettlementFund` | `F` |
| 成交数据 | `FileID.Trade` | `T` |
| 投资者持仓数据 | `FileID.InvestorPosition` | `P` |
| 投资者分项资金数据 | `FileID.SubEntryFund` | `O` |
| 组合持仓数据 | `FileID.CZCECombinationPos` | `C` |
| 上报保证金监控中心数据 | `FileID.CSRCData` | `R` |
| 郑商所平仓了结数据 | `FileID.CZCEClose` | `L` |
| 郑商所非平仓了结数据 | `FileID.CZCENoClose` | `N` |
| 持仓明细数据 | `FileID.PositionDtl` | `D` |
| 期权执行文件 | `FileID.OptionStrike` | `S` |
| 结算价比对文件 | `FileID.SettlementPriceComparison` | `M` |
| 上期所非持仓变动明细 | `FileID.NonTradePosChange` | `B` |

#### 54. FileType

文件上传类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 结算 | `FileType.Settlement` | `0` |
| 核对 | `FileType.Check` | `1` |

#### 55. FileFormat

文件格式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 文本文件(.txt) | `FileFormat.Txt` | `0` |
| 压缩文件(.zip) | `FileFormat.Zip` | `1` |
| DBF文件(.dbf) | `FileFormat.DBF` | `2` |

#### 56. FileUploadStatus

文件状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 上传成功 | `FileUploadStatus.SucceedUpload` | `1` |
| 上传失败 | `FileUploadStatus.FailedUpload` | `2` |
| 导入成功 | `FileUploadStatus.SucceedLoad` | `3` |
| 导入部分成功 | `FileUploadStatus.PartSucceedLoad` | `4` |
| 导入失败 | `FileUploadStatus.FailedLoad` | `5` |

#### 57. TransferDirection

移仓方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 移出 | `TransferDirection.Out` | `0` |
| 移入 | `TransferDirection.In` | `1` |

#### 58. SpecialCreateRule

特殊的创建规则类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 没有特殊创建规则 | `SpecialCreateRule.NoSpecialRule` | `0` |
| 不包含春节 | `SpecialCreateRule.NoSpringFestival` | `1` |

#### 59. BasisPriceType

挂牌基准价类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 上一合约结算价 | `BasisPriceType.LastSettlement` | `1` |
| 上一合约收盘价 | `BasisPriceType.LaseClose` | `2` |

#### 60. ProductLifePhase

产品生命周期状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 活跃 | `ProductLifePhase.Active` | `1` |
| 不活跃 | `ProductLifePhase.NonActive` | `2` |
| 注销 | `ProductLifePhase.Canceled` | `3` |

#### 61. DeliveryMode

交割方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 现金交割 | `DeliveryMode.CashDeliv` | `1` |
| 实物交割 | `DeliveryMode.CommodityDeliv` | `2` |

#### 62. FundIOType

出入金类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 出入金 | `FundIOType.FundIO` | `1` |
| 银期转帐 | `FundIOType.Transfer` | `2` |
| 银期换汇 | `FundIOType.SwapCurrency` | `3` |

#### 63. FundType

资金类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行存款 | `FundType.Deposite` | `1` |
| 分项资金 | `FundType.ItemFund` | `2` |
| 公司调整 | `FundType.Company` | `3` |
| 资金内转 | `FundType.InnerTransfer` | `4` |

#### 64. FundDirection

出入金方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 入金 | `FundDirection.In` | `1` |
| 出金 | `FundDirection.Out` | `2` |

#### 65. FundStatus

资金状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已录入 | `FundStatus.Record` | `1` |
| 已复核 | `FundStatus.Check` | `2` |
| 已冲销 | `FundStatus.Charge` | `3` |

#### 66. PublishStatus

发布状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未发布 | `PublishStatus.NONE` | `1` |
| 正在发布 | `PublishStatus.Publishing` | `2` |
| 已发布 | `PublishStatus.Published` | `3` |

#### 67. SystemStatus

系统状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不活跃 | `SystemStatus.NonActive` | `1` |
| 启动 | `SystemStatus.Startup` | `2` |
| 交易开始初始化 | `SystemStatus.Initialize` | `3` |
| 交易完成初始化 | `SystemStatus.Initialized` | `4` |
| 收市开始 | `SystemStatus.Close` | `5` |
| 收市完成 | `SystemStatus.Closed` | `6` |
| 结算 | `SystemStatus.Settlement` | `7` |

#### 68. SettlementStatus

结算状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 初始 | `SettlementStatus.Initialize` | `0` |
| 结算中 | `SettlementStatus.Settlementing` | `1` |
| 已结算 | `SettlementStatus.Settlemented` | `2` |
| 结算完成 | `SettlementStatus.Finished` | `3` |

#### 69. InvestorType

投资者类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自然人 | `InvestorType.Person` | `0` |
| 法人 | `InvestorType.Company` | `1` |
| 投资基金 | `InvestorType.Fund` | `2` |
| 特殊法人 | `InvestorType.SpecialOrgan` | `3` |
| 资管户 | `InvestorType.Asset` | `4` |

#### 70. BrokerType

经纪公司类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易会员 | `BrokerType.Trade` | `0` |
| 交易结算会员 | `BrokerType.TradeSettle` | `1` |

#### 71. RiskLevel

风险等级类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 低风险客户 | `RiskLevel.Low` | `1` |
| 普通客户 | `RiskLevel.Normal` | `2` |
| 关注客户 | `RiskLevel.Focus` | `3` |
| 风险客户 | `RiskLevel.Risk` | `4` |

#### 72. FeeAcceptStyle

手续费收取方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 按交易收取 | `FeeAcceptStyle.ByTrade` | `1` |
| 按交割收取 | `FeeAcceptStyle.ByDeliv` | `2` |
| 不收 | `FeeAcceptStyle.NONE` | `3` |
| 按指定手续费收取 | `FeeAcceptStyle.FixFee` | `4` |

#### 73. PasswordType

密码类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易密码 | `PasswordType.Trade` | `1` |
| 资金密码 | `PasswordType.Account` | `2` |

#### 74. Algorithm

盈亏算法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 浮盈浮亏都计算 | `Algorithm.All` | `1` |
| 浮盈不计，浮亏计 | `Algorithm.OnlyLost` | `2` |
| 浮盈计，浮亏不计 | `Algorithm.OnlyGain` | `3` |
| 浮盈浮亏都不计算 | `Algorithm.NONE` | `4` |

#### 75. IncludeCloseProfit

是否包含平仓盈利类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 包含平仓盈利 | `IncludeCloseProfit.Include` | `0` |
| 不包含平仓盈利 | `IncludeCloseProfit.NotInclude` | `2` |

#### 76. AllWithoutTrade

是否受可提比例限制类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 无仓无成交不受可提比例限制 | `AllWithoutTrade.Enable` | `0` |
| 受可提比例限制 | `AllWithoutTrade.Disable` | `2` |
| 无仓不受可提比例限制 | `AllWithoutTrade.NoHoldEnable` | `3` |

#### 77. FuturePwdFlag

资金密码核对标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不核对 | `FuturePwdFlag.UnCheck` | `0` |
| 核对 | `FuturePwdFlag.Check` | `1` |

#### 78. TransferType

银期转账类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行转期货 | `TransferType.BankToFuture` | `0` |
| 期货转银行 | `TransferType.FutureToBank` | `1` |

#### 79. TransferValidFlag

转账有效标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 无效或失败 | `TransferValidFlag.Invalid` | `0` |
| 有效 | `TransferValidFlag.Valid` | `1` |
| 冲正 | `TransferValidFlag.Reverse` | `2` |

#### 80. Reason

事由类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 错单 | `Reason.CD` | `0` |
| 资金在途 | `Reason.ZT` | `1` |
| 其它 | `Reason.QT` | `2` |

#### 81. Sex

性别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未知 | `Sex.NONE` | `0` |
| 男 | `Sex.Man` | `1` |
| 女 | `Sex.Woman` | `2` |

#### 82. UserType

用户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 投资者 | `UserType.Investor` | `0` |
| 操作员 | `UserType.Operator` | `1` |
| 管理员 | `UserType.SuperUser` | `2` |

#### 83. RateType

费率类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 保证金率 | `RateType.MarginRate` | `2` |

#### 84. NoteType

通知类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易结算单 | `NoteType.TradeSettleBill` | `1` |
| 交易结算月报 | `NoteType.TradeSettleMonth` | `2` |
| 追加保证金通知书 | `NoteType.CallMarginNotes` | `3` |
| 强行平仓通知书 | `NoteType.ForceCloseNotes` | `4` |
| 成交通知书 | `NoteType.TradeNotes` | `5` |
| 交割通知书 | `NoteType.DelivNotes` | `6` |

#### 85. SettlementStyle

结算单方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 逐日盯市 | `SettlementStyle.Day` | `1` |
| 逐笔对冲 | `SettlementStyle.Volume` | `2` |

#### 86. SettlementBillType

结算单类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 日报 | `SettlementBillType.Day` | `0` |
| 月报 | `SettlementBillType.Month` | `1` |

#### 87. UserRightType

客户权限类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 登录 | `UserRightType.Logon` | `1` |
| 银期转帐 | `UserRightType.Transfer` | `2` |
| 邮寄结算单 | `UserRightType.EMail` | `3` |
| 传真结算单 | `UserRightType.Fax` | `4` |
| 条件单 | `UserRightType.ConditionOrder` | `5` |

#### 88. MarginPriceType

保证金价格类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 昨结算价 | `MarginPriceType.PreSettlementPrice` | `1` |
| 最新价 | `MarginPriceType.SettlementPrice` | `2` |
| 成交均价 | `MarginPriceType.AveragePrice` | `3` |
| 开仓价 | `MarginPriceType.OpenPrice` | `4` |

#### 89. BillGenStatus

结算单生成状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未生成 | `BillGenStatus.NONE` | `0` |
| 生成中 | `BillGenStatus.NoGenerated` | `1` |
| 已生成 | `BillGenStatus.Generated` | `2` |

#### 90. AlgoType

算法类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 持仓处理算法 | `AlgoType.HandlePositionAlgo` | `1` |
| 寻找保证金率算法 | `AlgoType.FindMarginRateAlgo` | `2` |

#### 91. HandlePositionAlgoID

持仓处理算法编号类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 基本 | `HandlePositionAlgoID.Base` | `1` |
| 大连商品交易所 | `HandlePositionAlgoID.DCE` | `2` |
| 郑州商品交易所 | `HandlePositionAlgoID.CZCE` | `3` |

#### 92. FindMarginRateAlgoID

寻找保证金率算法编号类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 基本 | `FindMarginRateAlgoID.Base` | `1` |
| 大连商品交易所 | `FindMarginRateAlgoID.DCE` | `2` |
| 郑州商品交易所 | `FindMarginRateAlgoID.CZCE` | `3` |

#### 93. HandleTradingAccountAlgoID

资金处理算法编号类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 基本 | `HandleTradingAccountAlgoID.Base` | `1` |
| 大连商品交易所 | `HandleTradingAccountAlgoID.DCE` | `2` |
| 郑州商品交易所 | `HandleTradingAccountAlgoID.CZCE` | `3` |

#### 94. PersonType

联系人类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 指定下单人 | `PersonType.Order` | `1` |
| 开户授权人 | `PersonType.Open` | `2` |
| 资金调拨人 | `PersonType.Fund` | `3` |
| 结算单确认人 | `PersonType.Settlement` | `4` |
| 法人 | `PersonType.Company` | `5` |
| 法人代表 | `PersonType.Corporation` | `6` |
| 投资者联系人 | `PersonType.LinkMan` | `7` |
| 分户管理资产负责人 | `PersonType.Ledger` | `8` |
| 托（保）管人 | `PersonType.Trustee` | `9` |
| 托（保）管机构法人代表 | `PersonType.TrusteeCorporation` | `A` |
| 托（保）管机构开户授权人 | `PersonType.TrusteeOpen` | `B` |
| 托（保）管机构联系人 | `PersonType.TrusteeContact` | `C` |
| 境外自然人参考证件 | `PersonType.ForeignerRefer` | `D` |
| 法人代表参考证件 | `PersonType.CorporationRefer` | `E` |

#### 95. QueryInvestorRange

查询范围类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有 | `QueryInvestorRange.All` | `1` |
| 查询分类 | `QueryInvestorRange.Group` | `2` |
| 单一投资者 | `QueryInvestorRange.Single` | `3` |

#### 96. InvestorRiskStatus

投资者风险状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `InvestorRiskStatus.Normal` | `1` |
| 警告 | `InvestorRiskStatus.Warn` | `2` |
| 追保 | `InvestorRiskStatus.Call` | `3` |
| 强平 | `InvestorRiskStatus.Force` | `4` |
| 异常 | `InvestorRiskStatus.Exception` | `5` |

#### 97. UserEventType

用户事件类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 登录 | `UserEventType.Login` | `1` |
| 登出 | `UserEventType.Logout` | `2` |
| CTP校验通过 | `UserEventType.Trading` | `3` |
| CTP校验失败 | `UserEventType.TradingError` | `4` |
| 修改密码 | `UserEventType.UpdatePassword` | `5` |
| 客户端认证 | `UserEventType.Authenticate` | `6` |
| 终端信息上报 | `UserEventType.SubmitSysInfo` | `7` |
| 转账 | `UserEventType.Transfer` | `8` |
| 其他 | `UserEventType.Other` | `9` |
| 修改资金密码 | `UserEventType.UpdateTradingAccountPassword` | `a` |

#### 98. CloseStyle

平仓方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 先开先平 | `CloseStyle.Close` | `0` |
| 先平今再平昨 | `CloseStyle.CloseToday` | `1` |

#### 99. StatMode

统计方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| ---- | `StatMode.Non` | `0` |
| 按合约统计 | `StatMode.Instrument` | `1` |
| 按产品统计 | `StatMode.Product` | `2` |
| 按投资者统计 | `StatMode.Investor` | `3` |

#### 100. ParkedOrderStatus

预埋单状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未发送 | `ParkedOrderStatus.NotSend` | `1` |
| 已发送 | `ParkedOrderStatus.Send` | `2` |
| 已删除 | `ParkedOrderStatus.Deleted` | `3` |

#### 101. VirDealStatus

处理状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正在处理 | `VirDealStatus.Dealing` | `1` |
| 处理成功 | `VirDealStatus.DeaclSucceed` | `2` |

#### 102. OrgSystemID

原有系统代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 综合交易平台 | `OrgSystemID.Standard` | `0` |
| 易盛系统 | `OrgSystemID.ESunny` | `1` |
| 金仕达V6系统 | `OrgSystemID.KingStarV6` | `2` |

#### 103. VirTradeStatus

交易状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常处理中 | `VirTradeStatus.NaturalDeal` | `0` |
| 成功结束 | `VirTradeStatus.SucceedEnd` | `1` |
| 失败结束 | `VirTradeStatus.FailedEND` | `2` |
| 异常中 | `VirTradeStatus.Exception` | `3` |
| 已人工异常处理 | `VirTradeStatus.ManualDeal` | `4` |
| 通讯异常 ，请人工处理 | `VirTradeStatus.MesException` | `5` |
| 系统出错，请人工处理 | `VirTradeStatus.SysException` | `6` |

#### 104. VirBankAccType

银行帐户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 存折 | `VirBankAccType.BankBook` | `1` |
| 储蓄卡 | `VirBankAccType.BankCard` | `2` |
| 信用卡 | `VirBankAccType.CreditCard` | `3` |

#### 105. VirementStatus

银行帐户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `VirementStatus.Natural` | `0` |
| 销户 | `VirementStatus.Canceled` | `9` |

#### 106. VirementAvailAbility

有效标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未确认 | `VirementAvailAbility.NoAvailAbility` | `0` |
| 有效 | `VirementAvailAbility.AvailAbility` | `1` |
| 冲正 | `VirementAvailAbility.Repeal` | `2` |

#### 107. VirementTradeCode

交易代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行发起银行资金转期货 | `VirementTradeCode.BankBankToFuture` | `102001` |
| 银行发起期货资金转银行 | `VirementTradeCode.BankFutureToBank` | `102002` |
| 期货发起银行资金转期货 | `VirementTradeCode.FutureBankToFuture` | `202001` |
| 期货发起期货资金转银行 | `VirementTradeCode.FutureFutureToBank` | `202002` |

#### 108. AMLGenStatus

Aml生成方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 程序生成 | `AMLGenStatus.Program` | `0` |
| 人工生成 | `AMLGenStatus.HandWork` | `1` |

#### 109. CFMMCKeyKind

动态密钥类别(保证金监管)类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 主动请求更新 | `CFMMCKeyKind.REQUEST` | `R` |
| CFMMC自动更新 | `CFMMCKeyKind.AUTO` | `A` |
| CFMMC手动更新 | `CFMMCKeyKind.MANUAL` | `M` |

#### 110. CertificationType

证件类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 身份证 | `CertificationType.IDCard` | `0` |
| 护照 | `CertificationType.Passport` | `1` |
| 军官证 | `CertificationType.OfficerIDCard` | `2` |
| 士兵证 | `CertificationType.SoldierIDCard` | `3` |
| 回乡证 | `CertificationType.HomeComingCard` | `4` |
| 营业执照号 | `CertificationType.LicenseNo` | `6` |
| 组织机构代码证 | `CertificationType.InstitutionCodeCard` | `7` |
| 临时营业执照号 | `CertificationType.TempLicenseNo` | `8` |
| 民办非企业登记证书 | `CertificationType.NoEnterpriseLicenseNo` | `9` |
| 其他证件 | `CertificationType.OtherCard` | `x` |
| 主管部门批文 | `CertificationType.SuperDepAgree` | `a` |

#### 111. FileBusinessCode

文件业务功能类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 其他 | `FileBusinessCode.Others` | `0` |
| 转账交易明细对账 | `FileBusinessCode.TransferDetails` | `1` |
| 客户账户状态对账 | `FileBusinessCode.CustAccStatus` | `2` |
| 账户类交易明细对账 | `FileBusinessCode.AccountTradeDetails` | `3` |
| 期货账户信息变更明细对账 | `FileBusinessCode.FutureAccountChangeInfoDetails` | `4` |
| 客户资金台账余额明细对账 | `FileBusinessCode.CustMoneyDetail` | `5` |
| 客户销户结息明细对账 | `FileBusinessCode.CustCancelAccountInfo` | `6` |
| 客户资金余额对账结果 | `FileBusinessCode.CustMoneyResult` | `7` |
| 其它对账异常结果文件 | `FileBusinessCode.OthersExceptionResult` | `8` |
| 客户结息净额明细 | `FileBusinessCode.CustInterestNetMoneyDetails` | `9` |
| 客户资金交收明细 | `FileBusinessCode.CustMoneySendAndReceiveDetails` | `a` |
| 法人存管银行资金交收汇总 | `FileBusinessCode.CorporationMoneyTotal` | `b` |
| 主体间资金交收汇总 | `FileBusinessCode.MainbodyMoneyTotal` | `c` |
| 总分平衡监管数据 | `FileBusinessCode.MainPartMonitorData` | `d` |
| 存管银行备付金余额 | `FileBusinessCode.PreparationMoney` | `e` |
| 协办存管银行资金监管数据 | `FileBusinessCode.BankMoneyMonitorData` | `f` |

#### 112. CashExchangeCode

汇钞标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 汇 | `CashExchangeCode.Exchange` | `1` |
| 钞 | `CashExchangeCode.Cash` | `2` |

#### 113. YesNoIndicator

是或否标识类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 是 | `YesNoIndicator.Yes` | `0` |
| 否 | `YesNoIndicator.No` | `1` |

#### 114. BanlanceType

余额类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 当前余额 | `BanlanceType.CurrentMoney` | `0` |
| 可用余额 | `BanlanceType.UsableMoney` | `1` |
| 可取余额 | `BanlanceType.FetchableMoney` | `2` |
| 冻结余额 | `BanlanceType.FreezeMoney` | `3` |

#### 115. Gender

性别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未知状态 | `Gender.Unknown` | `0` |
| 男 | `Gender.Male` | `1` |
| 女 | `Gender.Female` | `2` |

#### 116. FeePayFlag

费用支付标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 由受益方支付费用 | `FeePayFlag.BEN` | `0` |
| 由发送方支付费用 | `FeePayFlag.OUR` | `1` |
| 由发送方支付发起的费用，受益方支付接受的费用 | `FeePayFlag.SHA` | `2` |

#### 117. PassWordKeyType

密钥类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交换密钥 | `PassWordKeyType.ExchangeKey` | `0` |
| 密码密钥 | `PassWordKeyType.PassWordKey` | `1` |
| MAC密钥 | `PassWordKeyType.MACKey` | `2` |
| 报文密钥 | `PassWordKeyType.MessageKey` | `3` |

#### 118. FBTPassWordType

密码类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 查询 | `FBTPassWordType.Query` | `0` |
| 取款 | `FBTPassWordType.Fetch` | `1` |
| 转帐 | `FBTPassWordType.Transfer` | `2` |
| 交易 | `FBTPassWordType.Trade` | `3` |

#### 119. FBTEncryMode

加密方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不加密 | `FBTEncryMode.NoEncry` | `0` |
| DES | `FBTEncryMode.DES` | `1` |
| 3DES | `FBTEncryMode.DES3` | `2` |

#### 120. BankRepealFlag

银行冲正标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行无需自动冲正 | `BankRepealFlag.BankNotNeedRepeal` | `0` |
| 银行待自动冲正 | `BankRepealFlag.BankWaitingRepeal` | `1` |
| 银行已自动冲正 | `BankRepealFlag.BankBeenRepealed` | `2` |

#### 121. BrokerRepealFlag

期商冲正标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期商无需自动冲正 | `BrokerRepealFlag.BrokerNotNeedRepeal` | `0` |
| 期商待自动冲正 | `BrokerRepealFlag.BrokerWaitingRepeal` | `1` |
| 期商已自动冲正 | `BrokerRepealFlag.BrokerBeenRepealed` | `2` |

#### 122. InstitutionType

机构类别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行 | `InstitutionType.Bank` | `0` |
| 期商 | `InstitutionType.Future` | `1` |
| 券商 | `InstitutionType.Store` | `2` |

#### 123. LastFragment

最后分片标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 是最后分片 | `LastFragment.Yes` | `0` |
| 不是最后分片 | `LastFragment.No` | `1` |

#### 124. BankAccStatus

银行账户状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `BankAccStatus.Normal` | `0` |
| 冻结 | `BankAccStatus.Freeze` | `1` |
| 挂失 | `BankAccStatus.ReportLoss` | `2` |

#### 125. MoneyAccountStatus

资金账户状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `MoneyAccountStatus.Normal` | `0` |
| 销户 | `MoneyAccountStatus.Cancel` | `1` |

#### 126. ManageStatus

存管状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 指定存管 | `ManageStatus.Point` | `0` |
| 预指定 | `ManageStatus.PrePoint` | `1` |
| 撤销指定 | `ManageStatus.CancelPoint` | `2` |

#### 127. SystemType

应用系统类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银期转帐 | `SystemType.FutureBankTransfer` | `0` |
| 银证转帐 | `SystemType.StockBankTransfer` | `1` |
| 第三方存管 | `SystemType.TheThirdPartStore` | `2` |

#### 128. TxnEndFlag

银期转帐划转结果标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常处理中 | `TxnEndFlag.NormalProcessing` | `0` |
| 成功结束 | `TxnEndFlag.Success` | `1` |
| 失败结束 | `TxnEndFlag.Failed` | `2` |
| 异常中 | `TxnEndFlag.Abnormal` | `3` |
| 已人工异常处理 | `TxnEndFlag.ManualProcessedForException` | `4` |
| 通讯异常 ，请人工处理 | `TxnEndFlag.CommuFailedNeedManualProcess` | `5` |
| 系统出错，请人工处理 | `TxnEndFlag.SysErrorNeedManualProcess` | `6` |

#### 129. ProcessStatus

银期转帐服务处理状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未处理 | `ProcessStatus.NotProcess` | `0` |
| 开始处理 | `ProcessStatus.StartProcess` | `1` |
| 处理完成 | `ProcessStatus.Finished` | `2` |

#### 130. CustType

客户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自然人 | `CustType.Person` | `0` |
| 机构户 | `CustType.Institution` | `1` |

#### 131. FBTTransferDirection

银期转帐方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 入金，银行转期货 | `FBTTransferDirection.FromBankToFuture` | `1` |
| 出金，期货转银行 | `FBTTransferDirection.FromFutureToBank` | `2` |

#### 132. OpenOrDestroy

开销户类别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 开户 | `OpenOrDestroy.Open` | `1` |
| 销户 | `OpenOrDestroy.Destroy` | `0` |

#### 133. AvailabilityFlag

有效标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未确认 | `AvailabilityFlag.Invalid` | `0` |
| 有效 | `AvailabilityFlag.Valid` | `1` |
| 冲正 | `AvailabilityFlag.Repeal` | `2` |

#### 134. OrganType

机构类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行代理 | `OrganType.Bank` | `1` |
| 交易前置 | `OrganType.Future` | `2` |
| 银期转帐平台管理 | `OrganType.PlateForm` | `9` |

#### 135. OrganLevel

机构级别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行总行或期商总部 | `OrganLevel.HeadQuarters` | `1` |
| 银行分中心或期货公司营业部 | `OrganLevel.Branch` | `2` |

#### 136. ProtocalID

协议类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期商协议 | `ProtocalID.FutureProtocal` | `0` |
| 工行协议 | `ProtocalID.ICBCProtocal` | `1` |
| 农行协议 | `ProtocalID.ABCProtocal` | `2` |
| 中国银行协议 | `ProtocalID.CBCProtocal` | `3` |
| 建行协议 | `ProtocalID.CCBProtocal` | `4` |
| 交行协议 | `ProtocalID.BOCOMProtocal` | `5` |
| 银期转帐平台协议 | `ProtocalID.FBTPlateFormProtocal` | `X` |

#### 137. ConnectMode

套接字连接方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 短连接 | `ConnectMode.ShortConnect` | `0` |
| 长连接 | `ConnectMode.LongConnect` | `1` |

#### 138. SyncMode

套接字通信方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 异步 | `SyncMode.ASync` | `0` |
| 同步 | `SyncMode.Sync` | `1` |

#### 139. BankAccType

银行帐号类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行存折 | `BankAccType.BankBook` | `1` |
| 储蓄卡 | `BankAccType.SavingCard` | `2` |
| 信用卡 | `BankAccType.CreditCard` | `3` |

#### 140. FutureAccType

期货公司帐号类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行存折 | `FutureAccType.BankBook` | `1` |
| 储蓄卡 | `FutureAccType.SavingCard` | `2` |
| 信用卡 | `FutureAccType.CreditCard` | `3` |

#### 141. OrganStatus

接入机构状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 启用 | `OrganStatus.Ready` | `0` |
| 签到 | `OrganStatus.CheckIn` | `1` |
| 签退 | `OrganStatus.CheckOut` | `2` |
| 对帐文件到达 | `OrganStatus.CheckFileArrived` | `3` |
| 对帐 | `OrganStatus.CheckDetail` | `4` |
| 日终清理 | `OrganStatus.DayEndClean` | `5` |
| 注销 | `OrganStatus.Invalid` | `9` |

#### 142. CCBFeeMode

建行收费模式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 按金额扣收 | `CCBFeeMode.ByAmount` | `1` |
| 按月扣收 | `CCBFeeMode.ByMonth` | `2` |

#### 143. CommApiType

通讯API类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 客户端 | `CommApiType.Client` | `1` |
| 服务端 | `CommApiType.Server` | `2` |
| 交易系统的UserApi | `CommApiType.UserApi` | `3` |

#### 144. LinkStatus

连接状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已经连接 | `LinkStatus.Connected` | `1` |
| 没有连接 | `LinkStatus.Disconnected` | `2` |

#### 145. PwdFlag

密码核对标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不核对 | `PwdFlag.NoCheck` | `0` |
| 明文核对 | `PwdFlag.BlankCheck` | `1` |
| 密文核对 | `PwdFlag.EncryptCheck` | `2` |

#### 146. SecuAccType

期货帐号类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 资金账号 | `SecuAccType.AccountID` | `1` |
| 资金卡号 | `SecuAccType.CardID` | `2` |
| 上海股东账号 | `SecuAccType.SHStockholderID` | `3` |
| 深圳股东账号 | `SecuAccType.SZStockholderID` | `4` |

#### 147. TransferStatus

转账交易状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `TransferStatus.Normal` | `0` |
| 被冲正 | `TransferStatus.Repealed` | `1` |

#### 148. SponsorType

发起方类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期商 | `SponsorType.Broker` | `0` |
| 银行 | `SponsorType.Bank` | `1` |

#### 149. ReqRspType

请求响应类别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 请求 | `ReqRspType.Request` | `0` |
| 响应 | `ReqRspType.Response` | `1` |

#### 150. FBTUserEventType

银期转帐用户事件类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 签到 | `FBTUserEventType.SignIn` | `0` |
| 银行转期货 | `FBTUserEventType.FromBankToFuture` | `1` |
| 期货转银行 | `FBTUserEventType.FromFutureToBank` | `2` |
| 开户 | `FBTUserEventType.OpenAccount` | `3` |
| 销户 | `FBTUserEventType.CancelAccount` | `4` |
| 变更银行账户 | `FBTUserEventType.ChangeAccount` | `5` |
| 冲正银行转期货 | `FBTUserEventType.RepealFromBankToFuture` | `6` |
| 冲正期货转银行 | `FBTUserEventType.RepealFromFutureToBank` | `7` |
| 查询银行账户 | `FBTUserEventType.QueryBankAccount` | `8` |
| 查询期货账户 | `FBTUserEventType.QueryFutureAccount` | `9` |
| 签退 | `FBTUserEventType.SignOut` | `A` |
| 密钥同步 | `FBTUserEventType.SyncKey` | `B` |
| 预约开户 | `FBTUserEventType.ReserveOpenAccount` | `C` |
| 撤销预约开户 | `FBTUserEventType.CancelReserveOpenAccount` | `D` |
| 预约开户确认 | `FBTUserEventType.ReserveOpenAccountConfirm` | `E` |
| 其他 | `FBTUserEventType.Other` | `Z` |

#### 151. DBOperation

记录操作类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 插入 | `DBOperation.Insert` | `0` |
| 更新 | `DBOperation.Update` | `1` |
| 删除 | `DBOperation.Delete` | `2` |

#### 152. SyncFlag

同步标记类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已同步 | `SyncFlag.Yes` | `0` |
| 未同步 | `SyncFlag.No` | `1` |

#### 153. SyncType

同步类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 一次同步 | `SyncType.OneOffSync` | `0` |
| 定时同步 | `SyncType.TimerSync` | `1` |
| 定时完全同步 | `SyncType.TimerFullSync` | `2` |

#### 154. ExDirection

换汇方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 结汇 | `ExDirection.Settlement` | `0` |
| 售汇 | `ExDirection.Sale` | `1` |

#### 155. FBEResultFlag

换汇成功标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 成功 | `FBEResultFlag.Success` | `0` |
| 账户余额不足 | `FBEResultFlag.InsufficientBalance` | `1` |
| 交易结果未知 | `FBEResultFlag.UnknownTrading` | `8` |
| 失败 | `FBEResultFlag.Fail` | `x` |

#### 156. FBEExchStatus

换汇交易状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `FBEExchStatus.Normal` | `0` |
| 交易重发 | `FBEExchStatus.ReExchange` | `1` |

#### 157. FBEFileFlag

换汇文件标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 数据包 | `FBEFileFlag.DataPackage` | `0` |
| 文件 | `FBEFileFlag.File` | `1` |

#### 158. FBEAlreadyTrade

换汇已交易标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未交易 | `FBEAlreadyTrade.NotTrade` | `0` |
| 已交易 | `FBEAlreadyTrade.Trade` | `1` |

#### 159. FBEUserEventType

银期换汇用户事件类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 签到 | `FBEUserEventType.SignIn` | `0` |
| 换汇 | `FBEUserEventType.Exchange` | `1` |
| 换汇重发 | `FBEUserEventType.ReExchange` | `2` |
| 银行账户查询 | `FBEUserEventType.QueryBankAccount` | `3` |
| 换汇明细查询 | `FBEUserEventType.QueryExchDetial` | `4` |
| 换汇汇总查询 | `FBEUserEventType.QueryExchSummary` | `5` |
| 换汇汇率查询 | `FBEUserEventType.QueryExchRate` | `6` |
| 对账文件通知 | `FBEUserEventType.CheckBankAccount` | `7` |
| 签退 | `FBEUserEventType.SignOut` | `8` |
| 其他 | `FBEUserEventType.Other` | `Z` |

#### 160. FBEReqFlag

换汇发送标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未处理 | `FBEReqFlag.UnProcessed` | `0` |
| 等待发送 | `FBEReqFlag.WaitSend` | `1` |
| 发送成功 | `FBEReqFlag.SendSuccess` | `2` |
| 发送失败 | `FBEReqFlag.SendFailed` | `3` |
| 等待重发 | `FBEReqFlag.WaitReSend` | `4` |

#### 161. NotifyClass

风险通知类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `NotifyClass.NOERROR` | `0` |
| 警示 | `NotifyClass.Warn` | `1` |
| 追保 | `NotifyClass.Call` | `2` |
| 强平 | `NotifyClass.Force` | `3` |
| 穿仓 | `NotifyClass.CHUANCANG` | `4` |
| 异常 | `NotifyClass.Exception` | `5` |

#### 162. ForceCloseType

强平单类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 手工强平 | `ForceCloseType.Manual` | `0` |
| 单一投资者辅助强平 | `ForceCloseType.Single` | `1` |
| 批量投资者辅助强平 | `ForceCloseType.Group` | `2` |

#### 163. RiskNotifyMethod

风险通知途径类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 系统通知 | `RiskNotifyMethod.System` | `0` |
| 短信通知 | `RiskNotifyMethod.SMS` | `1` |
| 邮件通知 | `RiskNotifyMethod.EMail` | `2` |
| 人工通知 | `RiskNotifyMethod.Manual` | `3` |

#### 164. RiskNotifyStatus

风险通知状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未生成 | `RiskNotifyStatus.NotGen` | `0` |
| 已生成未发送 | `RiskNotifyStatus.Generated` | `1` |
| 发送失败 | `RiskNotifyStatus.SendError` | `2` |
| 已发送未接收 | `RiskNotifyStatus.SendOk` | `3` |
| 已接收未确认 | `RiskNotifyStatus.Received` | `4` |
| 已确认 | `RiskNotifyStatus.Confirmed` | `5` |

#### 165. RiskUserEvent

风控用户操作事件类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 导出数据 | `RiskUserEvent.ExportData` | `0` |

#### 166. ConditionalOrderSortType

条件单索引条件类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 使用最新价升序 | `ConditionalOrderSortType.LastPriceAsc` | `0` |
| 使用最新价降序 | `ConditionalOrderSortType.LastPriceDesc` | `1` |
| 使用卖价升序 | `ConditionalOrderSortType.AskPriceAsc` | `2` |
| 使用卖价降序 | `ConditionalOrderSortType.AskPriceDesc` | `3` |
| 使用买价升序 | `ConditionalOrderSortType.BidPriceAsc` | `4` |
| 使用买价降序 | `ConditionalOrderSortType.BidPriceDesc` | `5` |

#### 167. SendType

报送状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未发送 | `SendType.NoSend` | `0` |
| 已发送 | `SendType.Sended` | `1` |
| 已生成 | `SendType.Generated` | `2` |
| 报送失败 | `SendType.SendFail` | `3` |
| 接收成功 | `SendType.Success` | `4` |
| 接收失败 | `SendType.Fail` | `5` |
| 取消报送 | `SendType.Cancel` | `6` |

#### 168. ClientIDStatus

交易编码状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未申请 | `ClientIDStatus.NoApply` | `1` |
| 已提交申请 | `ClientIDStatus.Submited` | `2` |
| 已发送申请 | `ClientIDStatus.Sended` | `3` |
| 完成 | `ClientIDStatus.Success` | `4` |
| 拒绝 | `ClientIDStatus.Refuse` | `5` |
| 已撤销编码 | `ClientIDStatus.Cancel` | `6` |

#### 169. QuestionType

特有信息类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 单选 | `QuestionType.Radio` | `1` |
| 多选 | `QuestionType.Option` | `2` |
| 填空 | `QuestionType.Blank` | `3` |

#### 170. BusinessType

业务类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 请求 | `BusinessType.Request` | `1` |
| 应答 | `BusinessType.Response` | `2` |
| 通知 | `BusinessType.Notice` | `3` |

#### 171. CfmmcReturnCode

监控中心返回码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 成功 | `CfmmcReturnCode.Success` | `0` |
| 该客户已经有流程在处理中 | `CfmmcReturnCode.Working` | `1` |
| 监控中客户资料检查失败 | `CfmmcReturnCode.InfoFail` | `2` |
| 监控中实名制检查失败 | `CfmmcReturnCode.IDCardFail` | `3` |
| 其他错误 | `CfmmcReturnCode.OtherFail` | `4` |

#### 172. ClientType

客户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有 | `ClientType.All` | `0` |
| 个人 | `ClientType.Person` | `1` |
| 单位 | `ClientType.Company` | `2` |
| 其他 | `ClientType.Other` | `3` |
| 特殊法人 | `ClientType.SpecialOrgan` | `4` |
| 资管户 | `ClientType.Asset` | `5` |

#### 173. ExchangeIDType

交易所编号类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 上海期货交易所(上期所) | `ExchangeIDType.SHFE` | `S` |
| 郑州商品交易所(郑商所) | `ExchangeIDType.CZCE` | `Z` |
| 大连商品交易所(大商所) | `ExchangeIDType.DCE` | `D` |
| 中国金融期货交易所(中金所) | `ExchangeIDType.CFFEX` | `J` |
| 上海国际能源交易中心股份有限公司(能源中心) | `ExchangeIDType.INE` | `N` |

#### 174. ExClientIDType

交易编码类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 套保 | `ExClientIDType.Hedge` | `1` |
| 套利 | `ExClientIDType.Arbitrage` | `2` |
| 投机 | `ExClientIDType.Speculation` | `3` |

#### 175. UpdateFlag

更新状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未更新 | `UpdateFlag.NoUpdate` | `0` |
| 更新全部信息成功 | `UpdateFlag.Success` | `1` |
| 更新全部信息失败 | `UpdateFlag.Fail` | `2` |
| 更新交易编码成功 | `UpdateFlag.TCSuccess` | `3` |
| 更新交易编码失败 | `UpdateFlag.TCFail` | `4` |
| 已丢弃 | `UpdateFlag.Cancel` | `5` |

#### 176. ApplyOperateID

申请动作类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 开户 | `ApplyOperateID.OpenInvestor` | `1` |
| 修改身份信息 | `ApplyOperateID.ModifyIDCard` | `2` |
| 修改一般信息 | `ApplyOperateID.ModifyNoIDCard` | `3` |
| 申请交易编码 | `ApplyOperateID.ApplyTradingCode` | `4` |
| 撤销交易编码 | `ApplyOperateID.CancelTradingCode` | `5` |
| 销户 | `ApplyOperateID.CancelInvestor` | `6` |
| 账户休眠 | `ApplyOperateID.FreezeAccount` | `8` |
| 激活休眠账户 | `ApplyOperateID.ActiveFreezeAccount` | `9` |

#### 177. ApplyStatusID

申请状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未补全 | `ApplyStatusID.NoComplete` | `1` |
| 已提交 | `ApplyStatusID.Submited` | `2` |
| 已审核 | `ApplyStatusID.Checked` | `3` |
| 已拒绝 | `ApplyStatusID.Refused` | `4` |
| 已删除 | `ApplyStatusID.Deleted` | `5` |

#### 178. SendMethod

发送方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 文件发送 | `SendMethod.ByAPI` | `1` |
| 电子发送 | `SendMethod.ByFile` | `2` |

#### 179. EventMode

操作方法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 增加 | `EventMode.ADD` | `1` |
| 修改 | `EventMode.UPDATE` | `2` |
| 删除 | `EventMode.DELETE` | `3` |
| 复核 | `EventMode.CHECK` | `4` |
| 复制 | `EventMode.COPY` | `5` |
| 注销 | `EventMode.CANCEL` | `6` |
| 冲销 | `EventMode.Reverse` | `7` |

#### 180. UOAAutoSend

统一开户申请自动发送类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自动发送并接收 | `UOAAutoSend.ASR` | `1` |
| 自动发送，不自动接收 | `UOAAutoSend.ASNR` | `2` |
| 不自动发送，自动接收 | `UOAAutoSend.NSAR` | `3` |
| 不自动发送，也不自动接收 | `UOAAutoSend.NSR` | `4` |

#### 181. FlowID

流程ID类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 投资者对应投资者组设置 | `FlowID.InvestorGroupFlow` | `1` |
| 投资者手续费率设置 | `FlowID.InvestorRate` | `2` |
| 投资者手续费率模板关系设置 | `FlowID.InvestorCommRateModel` | `3` |

#### 182. CheckLevel

复核级别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 零级复核 | `CheckLevel.Zero` | `0` |
| 一级复核 | `CheckLevel.One` | `1` |
| 二级复核 | `CheckLevel.Two` | `2` |

#### 183. CheckStatus

复核级别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未复核 | `CheckStatus.Init` | `0` |
| 复核中 | `CheckStatus.Checking` | `1` |
| 已复核 | `CheckStatus.Checked` | `2` |
| 拒绝 | `CheckStatus.Refuse` | `3` |
| 作废 | `CheckStatus.Cancel` | `4` |

#### 184. UsedStatus

生效状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未生效 | `UsedStatus.Unused` | `0` |
| 已生效 | `UsedStatus.Used` | `1` |
| 生效失败 | `UsedStatus.Fail` | `2` |

#### 185. BankAcountOrigin

账户来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 手工录入 | `BankAcountOrigin.ByAccProperty` | `0` |
| 银期转账 | `BankAcountOrigin.ByFBTransfer` | `1` |

#### 186. MonthBillTradeSum

结算单月报成交汇总方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 同日同合约 | `MonthBillTradeSum.ByInstrument` | `0` |
| 同日同合约同价格 | `MonthBillTradeSum.ByDayInsPrc` | `1` |
| 同合约 | `MonthBillTradeSum.ByDayIns` | `2` |

#### 187. FBTTradeCodeEnum

银期交易代码枚举类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行发起银行转期货 | `FBTTradeCodeEnum.BankLaunchBankToBroker` | `102001` |
| 期货发起银行转期货 | `FBTTradeCodeEnum.BrokerLaunchBankToBroker` | `202001` |
| 银行发起期货转银行 | `FBTTradeCodeEnum.BankLaunchBrokerToBank` | `102002` |
| 期货发起期货转银行 | `FBTTradeCodeEnum.BrokerLaunchBrokerToBank` | `202002` |

#### 188. OTPType

动态令牌类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 无动态令牌 | `OTPType.NONE` | `0` |
| 时间令牌 | `OTPType.TOTP` | `1` |

#### 189. OTPStatus

动态令牌状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未使用 | `OTPStatus.Unused` | `0` |
| 已使用 | `OTPStatus.Used` | `1` |
| 注销 | `OTPStatus.Disuse` | `2` |

#### 190. BrokerUserType

经济公司用户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 投资者 | `BrokerUserType.Investor` | `1` |
| 操作员 | `BrokerUserType.BrokerUser` | `2` |

#### 191. FutureType

期货类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 商品期货 | `FutureType.Commodity` | `1` |
| 金融期货 | `FutureType.Financial` | `2` |

#### 192. FundEventType

资金管理操作类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 转账限额 | `FundEventType.Restriction` | `0` |
| 当日转账限额 | `FundEventType.TodayRestriction` | `1` |
| 期商流水 | `FundEventType.Transfer` | `2` |
| 资金冻结 | `FundEventType.Credit` | `3` |
| 投资者可提资金比例 | `FundEventType.InvestorWithdrawAlm` | `4` |
| 单个银行账户转账限额 | `FundEventType.BankRestriction` | `5` |
| 银期签约账户 | `FundEventType.Accountregister` | `6` |
| 交易所出入金 | `FundEventType.ExchangeFundIO` | `7` |
| 投资者出入金 | `FundEventType.InvestorFundIO` | `8` |

#### 193. AccountSourceType

资金账户来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银期同步 | `AccountSourceType.FBTransfer` | `0` |
| 手工录入 | `AccountSourceType.ManualEntry` | `1` |

#### 194. CodeSourceType

交易编码来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 统一开户(已规范) | `CodeSourceType.UnifyAccount` | `0` |
| 手工录入(未规范) | `CodeSourceType.ManualEntry` | `1` |

#### 195. UserRange

操作员范围类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有 | `UserRange.All` | `0` |
| 单一操作员 | `UserRange.Single` | `1` |

#### 196. ByGroup

交易统计表按客户统计方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 按投资者统计 | `ByGroup.Investor` | `2` |
| 按类统计 | `ByGroup.Group` | `1` |

#### 197. TradeSumStatMode

交易统计表按范围统计方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 按合约统计 | `TradeSumStatMode.Instrument` | `1` |
| 按产品统计 | `TradeSumStatMode.Product` | `2` |
| 按交易所统计 | `TradeSumStatMode.Exchange` | `3` |

#### 198. ExprSetMode

日期表达式设置类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 相对已有规则设置 | `ExprSetMode.Relative` | `1` |
| 典型设置 | `ExprSetMode.Typical` | `2` |

#### 199. RateInvestorRange

投资者范围类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 公司标准 | `RateInvestorRange.All` | `1` |
| 模板 | `RateInvestorRange.Model` | `2` |
| 单一投资者 | `RateInvestorRange.Single` | `3` |

#### 200. SyncDataStatus

主次用系统数据同步状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未同步 | `SyncDataStatus.Initialize` | `0` |
| 同步中 | `SyncDataStatus.Settlementing` | `1` |
| 已同步 | `SyncDataStatus.Settlemented` | `2` |

#### 201. TradeSource

成交来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 来自交易所普通回报 | `TradeSource.NORMAL` | `0` |
| 来自查询 | `TradeSource.QUERY` | `1` |

#### 202. FlexStatMode

产品合约统计方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 产品统计 | `FlexStatMode.Product` | `1` |
| 交易所统计 | `FlexStatMode.Exchange` | `2` |
| 统计所有 | `FlexStatMode.All` | `3` |

#### 203. ByInvestorRange

投资者范围统计方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 属性统计 | `ByInvestorRange.Property` | `1` |
| 统计所有 | `ByInvestorRange.All` | `2` |

#### 204. PropertyInvestorRange

投资者范围类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有 | `PropertyInvestorRange.All` | `1` |
| 投资者属性 | `PropertyInvestorRange.Property` | `2` |
| 单一投资者 | `PropertyInvestorRange.Single` | `3` |

#### 205. FileStatus

文件状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未生成 | `FileStatus.NoCreate` | `0` |
| 已生成 | `FileStatus.Created` | `1` |
| 生成失败 | `FileStatus.Failed` | `2` |

#### 206. FileGenStyle

文件生成方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 下发 | `FileGenStyle.FileTransmit` | `0` |
| 生成 | `FileGenStyle.FileGen` | `1` |

#### 207. SysOperMode

系统日志操作方法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 增加 | `SysOperMode.Add` | `1` |
| 修改 | `SysOperMode.Update` | `2` |
| 删除 | `SysOperMode.Delete` | `3` |
| 复制 | `SysOperMode.Copy` | `4` |
| 激活 | `SysOperMode.AcTive` | `5` |
| 注销 | `SysOperMode.CanCel` | `6` |
| 重置 | `SysOperMode.ReSet` | `7` |

#### 208. SysOperType

系统日志操作类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 修改操作员密码 | `SysOperType.UpdatePassword` | `0` |
| 操作员组织架构关系 | `SysOperType.UserDepartment` | `1` |
| 角色管理 | `SysOperType.RoleManager` | `2` |
| 角色功能设置 | `SysOperType.RoleFunction` | `3` |
| 基础参数设置 | `SysOperType.BaseParam` | `4` |
| 设置操作员 | `SysOperType.SetUserID` | `5` |
| 用户角色设置 | `SysOperType.SetUserRole` | `6` |
| 用户IP限制 | `SysOperType.UserIpRestriction` | `7` |
| 组织架构管理 | `SysOperType.DepartmentManager` | `8` |
| 组织架构向查询分类复制 | `SysOperType.DepartmentCopy` | `9` |
| 交易编码管理 | `SysOperType.Tradingcode` | `A` |
| 投资者状态维护 | `SysOperType.InvestorStatus` | `B` |
| 投资者权限管理 | `SysOperType.InvestorAuthority` | `C` |
| 属性设置 | `SysOperType.PropertySet` | `D` |
| 重置投资者密码 | `SysOperType.ReSetInvestorPasswd` | `E` |
| 投资者个性信息维护 | `SysOperType.InvestorPersonalityInfo` | `F` |

#### 209. CSRCDataQueyType

上报数据查询类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 查询当前交易日报送的数据 | `CSRCDataQueyType.Current` | `0` |
| 查询历史报送的代理经纪公司的数据 | `CSRCDataQueyType.History` | `1` |

#### 210. FreezeStatus

休眠状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 活跃 | `FreezeStatus.Normal` | `1` |
| 休眠 | `FreezeStatus.Freeze` | `0` |

#### 211. StandardStatus

规范状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已规范 | `StandardStatus.Standard` | `0` |
| 未规范 | `StandardStatus.NonStandard` | `1` |

#### 212. RightParamType

配置类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 休眠户 | `RightParamType.Freeze` | `1` |
| 激活休眠户 | `RightParamType.FreezeActive` | `2` |
| 开仓权限限制 | `RightParamType.OpenLimit` | `3` |
| 解除开仓权限限制 | `RightParamType.RelieveOpenLimit` | `4` |

#### 213. DataStatus

反洗钱审核表数据状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `DataStatus.Normal` | `0` |
| 已删除 | `DataStatus.Deleted` | `1` |

#### 214. AMLCheckStatus

审核状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未复核 | `AMLCheckStatus.Init` | `0` |
| 复核中 | `AMLCheckStatus.Checking` | `1` |
| 已复核 | `AMLCheckStatus.Checked` | `2` |
| 拒绝上报 | `AMLCheckStatus.RefuseReport` | `3` |

#### 215. AmlDateType

日期类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 检查日期 | `AmlDateType.DrawDay` | `0` |
| 发生日期 | `AmlDateType.TouchDay` | `1` |

#### 216. AmlCheckLevel

审核级别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 零级审核 | `AmlCheckLevel.CheckLevel0` | `0` |
| 一级审核 | `AmlCheckLevel.CheckLevel1` | `1` |
| 二级审核 | `AmlCheckLevel.CheckLevel2` | `2` |
| 三级审核 | `AmlCheckLevel.CheckLevel3` | `3` |

#### 217. ExportFileType

导出文件类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| CSV文件 | `ExportFileType.CSV` | `0` |
| Excel文件 | `ExportFileType.EXCEL` | `1` |
| DBF文件 | `ExportFileType.DBF` | `2` |

#### 218. SettleManagerType

结算配置类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 结算前准备 | `SettleManagerType.Before` | `1` |
| 结算 | `SettleManagerType.Settlement` | `2` |
| 结算后核对 | `SettleManagerType.After` | `3` |
| 结算后处理 | `SettleManagerType.Settlemented` | `4` |

#### 219. SettleManagerLevel

结算配置等级类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 必要 | `SettleManagerLevel.Must` | `1` |
| 警告 | `SettleManagerLevel.Alarm` | `2` |
| 提示 | `SettleManagerLevel.Prompt` | `3` |
| 不检查 | `SettleManagerLevel.Ignore` | `4` |

#### 220. SettleManagerGroup

模块分组类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易所核对 | `SettleManagerGroup.Exhcange` | `1` |
| 内部核对 | `SettleManagerGroup.ASP` | `2` |
| 上报数据核对 | `SettleManagerGroup.CSRC` | `3` |

#### 221. LimitUseType

保值额度使用类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 可重复使用 | `LimitUseType.Repeatable` | `1` |
| 不可重复使用 | `LimitUseType.Unrepeatable` | `2` |

#### 222. DataResource

数据来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 本系统 | `DataResource.Settle` | `1` |
| 交易所 | `DataResource.Exchange` | `2` |
| 报送数据 | `DataResource.CSRC` | `3` |

#### 223. MarginType

保证金类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易所保证金率 | `MarginType.ExchMarginRate` | `0` |
| 投资者保证金率 | `MarginType.InstrMarginRate` | `1` |
| 投资者交易保证金率 | `MarginType.InstrMarginRateTrade` | `2` |

#### 224. ActiveType

生效类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 仅当日生效 | `ActiveType.Intraday` | `1` |
| 长期生效 | `ActiveType.Long` | `2` |

#### 225. MarginRateType

冲突保证金率类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易所保证金率 | `MarginRateType.Exchange` | `1` |
| 投资者保证金率 | `MarginRateType.Investor` | `2` |
| 投资者交易保证金率 | `MarginRateType.InvestorTrade` | `3` |

#### 226. BackUpStatus

备份数据状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未生成备份数据 | `BackUpStatus.UnBak` | `0` |
| 备份数据生成中 | `BackUpStatus.BakUp` | `1` |
| 已生成备份数据 | `BackUpStatus.BakUped` | `2` |
| 备份数据失败 | `BackUpStatus.BakFail` | `3` |

#### 227. InitSettlement

结算初始化状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 结算初始化未开始 | `InitSettlement.UnInitialize` | `0` |
| 结算初始化中 | `InitSettlement.Initialize` | `1` |
| 结算初始化完成 | `InitSettlement.Initialized` | `2` |

#### 228. ReportStatus

报表数据生成状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未生成报表数据 | `ReportStatus.NoCreate` | `0` |
| 报表数据生成中 | `ReportStatus.Create` | `1` |
| 已生成报表数据 | `ReportStatus.Created` | `2` |
| 生成报表数据失败 | `ReportStatus.CreateFail` | `3` |

#### 229. SaveStatus

数据归档状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 归档未完成 | `SaveStatus.UnSaveData` | `0` |
| 归档完成 | `SaveStatus.SaveDatad` | `1` |

#### 230. SettArchiveStatus

结算确认数据归档状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未归档数据 | `SettArchiveStatus.UnArchived` | `0` |
| 数据归档中 | `SettArchiveStatus.Archiving` | `1` |
| 已归档数据 | `SettArchiveStatus.Archived` | `2` |
| 归档数据失败 | `SettArchiveStatus.ArchiveFail` | `3` |

#### 231. CTPType

CTP交易系统类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未知类型 | `CTPType.Unkown` | `0` |
| 主中心 | `CTPType.MainCenter` | `1` |
| 备中心 | `CTPType.BackUp` | `2` |

#### 232. CloseDealType

平仓处理类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `CloseDealType.Normal` | `0` |
| 投机平仓优先 | `CloseDealType.SpecFirst` | `1` |

#### 233. MortgageFundUseRange

货币质押资金可用范围类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不能使用 | `MortgageFundUseRange.NONE` | `0` |
| 用于保证金 | `MortgageFundUseRange.Margin` | `1` |
| 用于手续费、盈亏、保证金 | `MortgageFundUseRange.All` | `2` |
| 人民币方案3 | `MortgageFundUseRange.CNY3` | `3` |

#### 234. SpecProductType

特殊产品类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 郑商所套保产品 | `SpecProductType.CzceHedge` | `1` |
| 货币质押产品 | `SpecProductType.IneForeignCurrency` | `2` |
| 大连短线开平仓产品 | `SpecProductType.DceOpenClose` | `3` |

#### 235. FundMortgageType

货币质押类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 质押 | `FundMortgageType.Mortgage` | `1` |
| 解质 | `FundMortgageType.Redemption` | `2` |

#### 236. AccountSettlementParamID

投资者账户结算参数代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 基础保证金 | `AccountSettlementParamID.BaseMargin` | `1` |
| 最低权益标准 | `AccountSettlementParamID.LowestInterest` | `2` |

#### 237. FundMortDirection

货币质押方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 货币质入 | `FundMortDirection.In` | `1` |
| 货币质出 | `FundMortDirection.Out` | `2` |

#### 238. BusinessClass

换汇类别类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 盈利 | `BusinessClass.Profit` | `0` |
| 亏损 | `BusinessClass.Loss` | `1` |
| 其他 | `BusinessClass.Other` | `Z` |

#### 239. SwapSourceType

换汇数据来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 手工 | `SwapSourceType.Manual` | `0` |
| 自动生成 | `SwapSourceType.Automatic` | `1` |

#### 240. CurrExDirection

换汇类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 结汇 | `CurrExDirection.Settlement` | `0` |
| 售汇 | `CurrExDirection.Sale` | `1` |

#### 241. CurrencySwapStatus

申请状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已录入 | `CurrencySwapStatus.Entry` | `1` |
| 已审核 | `CurrencySwapStatus.Approve` | `2` |
| 已拒绝 | `CurrencySwapStatus.Refuse` | `3` |
| 已撤销 | `CurrencySwapStatus.Revoke` | `4` |
| 已发送 | `CurrencySwapStatus.Send` | `5` |
| 换汇成功 | `CurrencySwapStatus.Success` | `6` |
| 换汇失败 | `CurrencySwapStatus.Failure` | `7` |

#### 242. ReqFlag

换汇发送标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 未发送 | `ReqFlag.NoSend` | `0` |
| 发送成功 | `ReqFlag.SendSuccess` | `1` |
| 发送失败 | `ReqFlag.SendFailed` | `2` |
| 等待重发 | `ReqFlag.WaitReSend` | `3` |

#### 243. ResFlag

换汇返回成功标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 成功 | `ResFlag.Success` | `0` |
| 账户余额不足 | `ResFlag.InsuffiCient` | `1` |
| 交易结果未知 | `ResFlag.UnKnown` | `8` |

#### 244. ExStatus

修改状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 修改前 | `ExStatus.Before` | `0` |
| 修改后 | `ExStatus.After` | `1` |

#### 245. ClientRegion

开户客户地域类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 国内客户 | `ClientRegion.Domestic` | `1` |
| 港澳台客户 | `ClientRegion.GMT` | `2` |
| 国外客户 | `ClientRegion.Foreign` | `3` |

#### 246. HasBoard

是否有董事会类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 没有 | `HasBoard.No` | `0` |
| 有 | `HasBoard.Yes` | `1` |

#### 247. StartMode

启动模式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 正常 | `StartMode.Normal` | `1` |
| 应急 | `StartMode.Emerge` | `2` |
| 恢复 | `StartMode.Restore` | `3` |

#### 248. TemplateType

模型类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 全量 | `TemplateType.Full` | `1` |
| 增量 | `TemplateType.Increment` | `2` |
| 备份 | `TemplateType.BackUp` | `3` |

#### 249. LoginMode

登录模式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易 | `LoginMode.Trade` | `0` |
| 转账 | `LoginMode.Transfer` | `1` |

#### 250. PromptType

日历提示类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 合约上下市 | `PromptType.Instrument` | `1` |
| 保证金分段生效 | `PromptType.Margin` | `2` |

#### 251. HasTrustee

是否有托管人类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 有 | `HasTrustee.Yes` | `1` |
| 没有 | `HasTrustee.No` | `0` |

#### 252. AmType

机构类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 银行 | `AmType.Bank` | `1` |
| 证券公司 | `AmType.Securities` | `2` |
| 基金公司 | `AmType.Fund` | `3` |
| 保险公司 | `AmType.Insurance` | `4` |
| 信托公司 | `AmType.Trust` | `5` |
| 其他 | `AmType.Other` | `9` |

#### 253. CSRCFundIOType

出入金类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 出入金 | `CSRCFundIOType.FundIO` | `0` |
| 银期换汇 | `CSRCFundIOType.SwapCurrency` | `1` |

#### 254. CusAccountType

结算账户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货结算账户 | `CusAccountType.Futures` | `1` |
| 纯期货资管业务下的资管结算账户 | `CusAccountType.AssetmgrFuture` | `2` |
| 综合类资管业务下的期货资管托管账户 | `CusAccountType.AssetmgrTrustee` | `3` |
| 综合类资管业务下的资金中转账户 | `CusAccountType.AssetmgrTransfer` | `4` |

#### 255. LanguageType

通知语言类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 中文 | `LanguageType.Chinese` | `1` |
| 英文 | `LanguageType.English` | `2` |

#### 256. AssetmgrClientType

资产管理客户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 个人资管客户 | `AssetmgrClientType.Person` | `1` |
| 单位资管客户 | `AssetmgrClientType.Organ` | `2` |
| 特殊单位资管客户 | `AssetmgrClientType.SpecialOrgan` | `4` |

#### 257. AssetmgrType

投资类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货类 | `AssetmgrType.Futures` | `3` |
| 综合类 | `AssetmgrType.SpecialOrgan` | `4` |

#### 258. CheckInstrType

合约比较类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 合约交易所不存在 | `CheckInstrType.HasExch` | `0` |
| 合约本系统不存在 | `CheckInstrType.HasATP` | `1` |
| 合约比较不一致 | `CheckInstrType.HasDiff` | `2` |

#### 259. DeliveryType

交割类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 手工交割 | `DeliveryType.HandDeliv` | `1` |
| 到期交割 | `DeliveryType.PersonDeliv` | `2` |

#### 260. MaxMarginSideAlgorithm

大额单边保证金算法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不使用大额单边保证金算法 | `MaxMarginSideAlgorithm.NO` | `0` |
| 使用大额单边保证金算法 | `MaxMarginSideAlgorithm.YES` | `1` |

#### 261. DAClientType

资产管理客户类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自然人 | `DAClientType.Person` | `0` |
| 法人 | `DAClientType.Company` | `1` |
| 其他 | `DAClientType.Other` | `2` |

#### 262. UOAAssetmgrType

投资类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货类 | `UOAAssetmgrType.Futures` | `1` |
| 综合类 | `UOAAssetmgrType.SpecialOrgan` | `2` |

#### 263. DirectionEn

买卖方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| Buy | `DirectionEn.Buy` | `0` |
| Sell | `DirectionEn.Sell` | `1` |

#### 264. OffsetType

平仓类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期权对冲 | `OffsetType.OPT_OFFSET` | `0` |
| 期货对冲 | `OffsetType.FUT_OFFSET` | `1` |
| 行权后期货对冲 | `OffsetType.EXEC_OFFSET` | `2` |
| 履约后期货对冲 | `OffsetType.PERFORM_OFFSET` | `3` |

#### 265. OffsetFlagEn

开平标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| Position Opening | `OffsetFlagEn.Open` | `0` |
| Position Closing | `OffsetFlagEn.Close` | `1` |
| Forced Liquidation | `OffsetFlagEn.ForceClose` | `2` |
| Close Today | `OffsetFlagEn.CloseToday` | `3` |
| Close Prev | `OffsetFlagEn.CloseYesterday` | `4` |
| Forced Reduction | `OffsetFlagEn.ForceOff` | `5` |
| Local Forced Liquidation | `OffsetFlagEn.LocalForceClose` | `6` |

#### 266. HedgeFlagEn

投机套保标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| Speculation | `HedgeFlagEn.Speculation` | `1` |
| Arbitrage | `HedgeFlagEn.Arbitrage` | `2` |
| Hedge | `HedgeFlagEn.Hedge` | `3` |

#### 267. FundIOTypeEn

出入金类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| Deposit | `FundIOTypeEn.FundIO` | `1` |
| Bank-Futures Transfer | `FundIOTypeEn.Transfer` | `2` |
| Bank-Futures FX Exchange | `FundIOTypeEn.SwapCurrency` | `3` |

#### 268. FundTypeEn

资金类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| Bank Deposit | `FundTypeEn.Deposite` | `1` |
| Payment | `FundTypeEn.ItemFund` | `2` |
| Brokerage Adj | `FundTypeEn.Company` | `3` |
| Internal Transfer | `FundTypeEn.InnerTransfer` | `4` |

#### 269. FundDirectionEn

出入金方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| Deposit | `FundDirectionEn.In` | `1` |
| Withdrawal | `FundDirectionEn.Out` | `2` |

#### 270. FundMortDirectionEn

货币质押方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| Pledge | `FundMortDirectionEn.In` | `1` |
| Redemption | `FundMortDirectionEn.Out` | `2` |

#### 271. OptionsType

期权类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 看涨 | `OptionsType.CallOptions` | `1` |
| 看跌 | `OptionsType.PutOptions` | `2` |

#### 272. StrikeMode

执行方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 欧式 | `StrikeMode.Continental` | `0` |
| 美式 | `StrikeMode.American` | `1` |
| 百慕大 | `StrikeMode.Bermuda` | `2` |

#### 273. StrikeType

执行类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自身对冲 | `StrikeType.Hedge` | `0` |
| 匹配执行 | `StrikeType.Match` | `1` |

#### 274. ApplyType

中金所期权放弃执行申请类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不执行数量 | `ApplyType.NotStrikeNum` | `4` |

#### 275. GiveUpDataSource

放弃执行申请数据来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 系统生成 | `GiveUpDataSource.Gen` | `0` |
| 手工添加 | `GiveUpDataSource.Hand` | `1` |

#### 276. ExecResult

执行结果类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 没有执行 | `ExecResult.NoExec` | `n` |
| 已经取消 | `ExecResult.Canceled` | `c` |
| 执行成功 | `ExecResult.OK` | `0` |
| 期权持仓不够 | `ExecResult.NoPosition` | `1` |
| 资金不够 | `ExecResult.NoDeposit` | `2` |
| 会员不存在 | `ExecResult.NoParticipant` | `3` |
| 客户不存在 | `ExecResult.NoClient` | `4` |
| 合约不存在 | `ExecResult.NoInstrument` | `6` |
| 没有执行权限 | `ExecResult.NoRight` | `7` |
| 不合理的数量 | `ExecResult.InvalidVolume` | `8` |
| 没有足够的历史成交 | `ExecResult.NoEnoughHistoryTrade` | `9` |
| 未知 | `ExecResult.Unknown` | `a` |

#### 277. CombinationType

组合类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货组合 | `CombinationType.Future` | `0` |
| 垂直价差BUL | `CombinationType.BUL` | `1` |
| 垂直价差BER | `CombinationType.BER` | `2` |
| 跨式组合 | `CombinationType.STD` | `3` |
| 宽跨式组合 | `CombinationType.STG` | `4` |
| 备兑组合 | `CombinationType.PRT` | `5` |
| 时间价差组合 | `CombinationType.CAS` | `6` |
| 期权对锁组合 | `CombinationType.OPL` | `7` |
| 买备兑组合 | `CombinationType.BFO` | `8` |
| 买入期权垂直价差组合 | `CombinationType.BLS` | `9` |
| 卖出期权垂直价差组合 | `CombinationType.BES` | `a` |

#### 278. DceCombinationType

组合类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货对锁组合 | `DceCombinationType.SPL` | `0` |
| 期权对锁组合 | `DceCombinationType.OPL` | `1` |
| 期货跨期组合 | `DceCombinationType.SP` | `2` |
| 期货跨品种组合 | `DceCombinationType.SPC` | `3` |
| 买入期权垂直价差组合 | `DceCombinationType.BLS` | `4` |
| 卖出期权垂直价差组合 | `DceCombinationType.BES` | `5` |
| 期权日历价差组合 | `DceCombinationType.CAS` | `6` |
| 期权跨式组合 | `DceCombinationType.STD` | `7` |
| 期权宽跨式组合 | `DceCombinationType.STG` | `8` |
| 买入期货期权组合 | `DceCombinationType.BFO` | `9` |
| 卖出期货期权组合 | `DceCombinationType.SFO` | `a` |

#### 279. OptionRoyaltyPriceType

期权权利金价格类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 昨结算价 | `OptionRoyaltyPriceType.PreSettlementPrice` | `1` |
| 开仓价 | `OptionRoyaltyPriceType.OpenPrice` | `4` |
| 最新价与昨结算价较大值 | `OptionRoyaltyPriceType.MaxPreSettlementPrice` | `5` |

#### 280. BalanceAlgorithm

权益算法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不计算期权市值盈亏 | `BalanceAlgorithm.Default` | `1` |
| 计算期权市值亏损 | `BalanceAlgorithm.IncludeOptValLost` | `2` |

#### 281. ActionType

执行类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 执行 | `ActionType.Exec` | `1` |
| 放弃 | `ActionType.Abandon` | `2` |

#### 282. ForQuoteStatus

询价状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 已经提交 | `ForQuoteStatus.Submitted` | `a` |
| 已经接受 | `ForQuoteStatus.Accepted` | `b` |
| 已经被拒绝 | `ForQuoteStatus.Rejected` | `c` |

#### 283. ValueMethod

取值方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 按绝对值 | `ValueMethod.Absolute` | `0` |
| 按比率 | `ValueMethod.Ratio` | `1` |

#### 284. ExecOrderPositionFlag

期权行权后是否保留期货头寸的标记类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 保留 | `ExecOrderPositionFlag.Reserve` | `0` |
| 不保留 | `ExecOrderPositionFlag.UnReserve` | `1` |

#### 285. ExecOrderCloseFlag

期权行权后生成的头寸是否自动平仓类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自动平仓 | `ExecOrderCloseFlag.AutoClose` | `0` |
| 免于自动平仓 | `ExecOrderCloseFlag.NotToClose` | `1` |

#### 286. ProductType

产品类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货 | `ProductType.Futures` | `1` |
| 期权 | `ProductType.Options` | `2` |

#### 287. CZCEUploadFileName

郑商所结算文件名类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| ^\d{8}*zz*\d{4} | `CZCEUploadFileName.O` | `O` |
| ^\d{8}成交表 | `CZCEUploadFileName.T` | `T` |
| ^\d{8}单腿持仓表new | `CZCEUploadFileName.P` | `P` |
| ^\d{8}非平仓了结表 | `CZCEUploadFileName.N` | `N` |
| ^\d{8}平仓表 | `CZCEUploadFileName.L` | `L` |
| ^\d{8}资金表 | `CZCEUploadFileName.F` | `F` |
| ^\d{8}组合持仓表 | `CZCEUploadFileName.C` | `C` |
| ^\d{8}保证金参数表 | `CZCEUploadFileName.M` | `M` |

#### 288. DCEUploadFileName

大商所结算文件名类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| ^\d{8}*dl*\d{3} | `DCEUploadFileName.O` | `O` |
| ^\d{8}_成交表 | `DCEUploadFileName.T` | `T` |
| ^\d{8}_持仓表 | `DCEUploadFileName.P` | `P` |
| ^\d{8}_资金结算表 | `DCEUploadFileName.F` | `F` |
| ^\d{8}_优惠组合持仓明细表 | `DCEUploadFileName.C` | `C` |
| ^\d{8}_持仓明细表 | `DCEUploadFileName.D` | `D` |
| ^\d{8}_保证金参数表 | `DCEUploadFileName.M` | `M` |
| ^\d{8}_期权执行表 | `DCEUploadFileName.S` | `S` |

#### 289. SHFEUploadFileName

上期所结算文件名类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| ^\d{4}*\d{8}*\d{8}_DailyFundChg | `SHFEUploadFileName.O` | `O` |
| ^\d{4}*\d{8}*\d{8}_Trade | `SHFEUploadFileName.T` | `T` |
| ^\d{4}*\d{8}*\d{8}_SettlementDetail | `SHFEUploadFileName.P` | `P` |
| ^\d{4}*\d{8}*\d{8}_Capital | `SHFEUploadFileName.F` | `F` |

#### 290. CFFEXUploadFileName

中金所结算文件名类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| ^\d{4}*SG\d{1}*\d{8}_\d{1}_Trade | `CFFEXUploadFileName.T` | `T` |
| ^\d{4}*SG\d{1}*\d{8}_\d{1}_SettlementDetail | `CFFEXUploadFileName.P` | `P` |
| ^\d{4}*SG\d{1}*\d{8}_\d{1}_Capital | `CFFEXUploadFileName.F` | `F` |
| ^\d{4}*SG\d{1}*\d{8}_\d{1}_OptionExec | `CFFEXUploadFileName.S` | `S` |

#### 291. CombDirection

组合指令方向类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 申请组合 | `CombDirection.Comb` | `0` |
| 申请拆分 | `CombDirection.UnComb` | `1` |
| 操作员删组合单 | `CombDirection.DelComb` | `2` |

#### 292. StrikeOffsetType

行权偏移类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 实值额 | `StrikeOffsetType.RealValue` | `1` |
| 盈利额 | `StrikeOffsetType.ProfitValue` | `2` |
| 实值比例 | `StrikeOffsetType.RealRatio` | `3` |
| 盈利比例 | `StrikeOffsetType.ProfitRatio` | `4` |

#### 293. ReserveOpenAccStas

预约开户状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 等待处理中 | `ReserveOpenAccStas.Processing` | `0` |
| 已撤销 | `ReserveOpenAccStas.Cancelled` | `1` |
| 已开户 | `ReserveOpenAccStas.Opened` | `2` |
| 无效请求 | `ReserveOpenAccStas.Invalid` | `3` |

#### 294. WeakPasswordSource

弱密码来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 弱密码库 | `WeakPasswordSource.Lib` | `1` |
| 手工录入 | `WeakPasswordSource.Manual` | `2` |

#### 295. OptSelfCloseFlag

期权行权的头寸是否自对冲类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 自对冲期权仓位 | `OptSelfCloseFlag.CloseSelfOptionPosition` | `1` |
| 保留期权仓位 | `OptSelfCloseFlag.ReserveOptionPosition` | `2` |
| 自对冲卖方履约后的期货仓位 | `OptSelfCloseFlag.SellCloseSelfFuturePosition` | `3` |
| 保留卖方履约后的期货仓位 | `OptSelfCloseFlag.ReserveFuturePosition` | `4` |

#### 296. BizType

业务类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 期货 | `BizType.Future` | `1` |
| 证券 | `BizType.Stock` | `2` |

#### 297. AppType

用户App类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 直连的投资者 | `AppType.Investor` | `1` |
| 为每个投资者都创建连接的中继 | `AppType.InvestorRelay` | `2` |
| 所有投资者共享一个操作员连接的中继 | `AppType.OperatorRelay` | `3` |
| 未知 | `AppType.UnKnown` | `4` |

#### 298. ResponseValue

应答类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 检查成功 | `ResponseValue.Right` | `0` |
| 检查失败 | `ResponseValue.Refuse` | `1` |

#### 299. OTCTradeType

OTC成交类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 大宗交易 | `OTCTradeType.Block` | `0` |
| 期转现 | `OTCTradeType.EFP` | `1` |

#### 300. MatchType

期现风险匹配方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 基点价值 | `MatchType.DV01` | `1` |
| 面值 | `MatchType.ParValue` | `2` |

#### 301. AuthType

用户终端认证方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 白名单校验 | `AuthType.WHITE` | `0` |
| 黑名单校验 | `AuthType.BLACK` | `1` |

#### 302. ClassType

合约分类方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有合约 | `ClassType.ALL` | `0` |
| 期货、即期、期转现、Tas、金属指数合约 | `ClassType.FUTURE` | `1` |
| 期货、现货期权合约 | `ClassType.OPTION` | `2` |
| 组合合约 | `ClassType.COMB` | `3` |

#### 303. TradingType

合约交易状态分类方式类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 所有状态 | `TradingType.ALL` | `0` |
| 交易 | `TradingType.TRADE` | `1` |
| 非交易 | `TradingType.UNTRADE` | `2` |

#### 304. ProductStatus

产品状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 可交易 | `ProductStatus.tradeable` | `1` |
| 不可交易 | `ProductStatus.untradeable` | `2` |

#### 305. SyncDeltaStatus

追平状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 交易可读 | `SyncDeltaStatus.Readable` | `1` |
| 交易在读 | `SyncDeltaStatus.Reading` | `2` |
| 交易读取完成 | `SyncDeltaStatus.Readend` | `3` |
| 追平失败 交易本地状态结算不存在 | `SyncDeltaStatus.OptErr` | `e` |

#### 306. ActionDirection

操作标志类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 增加 | `ActionDirection.Add` | `1` |
| 删除 | `ActionDirection.Del` | `2` |
| 更新 | `ActionDirection.Upd` | `3` |

#### 307. OrderCancelAlg

撤单时选择席位算法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 轮询席位撤单 | `OrderCancelAlg.Balance` | `1` |
| 优先原报单席位撤单 | `OrderCancelAlg.OrigFirst` | `2` |

#### 308. OpenLimitControlLevel

开仓量限制粒度类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不控制 | `OpenLimitControlLevel.NONE` | `0` |
| 产品级别 | `OpenLimitControlLevel.Product` | `1` |
| 合约级别 | `OpenLimitControlLevel.Inst` | `2` |

#### 309. OrderFreqControlLevel

报单频率控制粒度类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不控制 | `OrderFreqControlLevel.NONE` | `0` |
| 产品级别 | `OrderFreqControlLevel.Product` | `1` |
| 合约级别 | `OrderFreqControlLevel.Inst` | `2` |

#### 310. EnumBool

枚举bool类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| false | `EnumBool._False` | `0` |
| true | `EnumBool._True` | `1` |

#### 311. TimeRange

期货合约阶段标识类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 一般月份 | `TimeRange.USUAL` | `1` |
| 交割月前一个月上半月 | `TimeRange.FNSP` | `2` |
| 交割月前一个月下半月 | `TimeRange.BNSP` | `3` |
| 交割月份 | `TimeRange.SPOT` | `4` |

#### 312. Portfolio

新型组保算法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 不使用新型组保算法 | `Portfolio.NONE` | `0` |
| SPBM算法 | `Portfolio.SPBM` | `1` |
| RULE算法 | `Portfolio.RULE` | `2` |
| SPMM算法 | `Portfolio.SPMM` | `3` |
| RCAMS算法 | `Portfolio.RCAMS` | `4` |

#### 313. WithDrawParamID

可提参数代码类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 权利金收支是否可提 1 代表可提 0 不可提 | `WithDrawParamID.CashIn` | `C` |

#### 314. InvstTradingRight

投资者交易权限类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 只能平仓 | `InvstTradingRight.CloseOnly` | `1` |
| 不能交易 | `InvstTradingRight.Forbidden` | `2` |

#### 315. InstMarginCalID

SPMM合约保证金算法类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 标准算法收取双边 | `InstMarginCalID.BothSide` | `1` |
| 单向大边 | `InstMarginCalID.MMSA` | `2` |
| 新组保SPMM | `InstMarginCalID.SPMM` | `3` |

#### 316. RCAMSCombinationType

RCAMS组合类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 牛市看涨价差组合 | `RCAMSCombinationType.BUC` | `0` |
| 熊市看涨价差组合 | `RCAMSCombinationType.BEC` | `1` |
| 熊市看跌价差组合 | `RCAMSCombinationType.BEP` | `2` |
| 牛市看跌价差组合 | `RCAMSCombinationType.BUP` | `3` |
| 日历价差组合 | `RCAMSCombinationType.CAS` | `4` |

#### 317. PortfType

新组保算法启用类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 使用初版交易所算法 | `PortfType.NONE` | `0` |
| SPBM算法V1.1.0_附加保证金调整 | `PortfType.SPBM_AddOnHedge` | `1` |

#### 318. InstrumentClass

合约类型类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 一般月份合约 | `InstrumentClass.Usual` | `1` |
| 临近交割合约 | `InstrumentClass.Delivery` | `2` |
| 非组合合约 | `InstrumentClass.NonComb` | `3` |

#### 319. ProdChangeFlag

品种记录改变状态类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 持仓量和冻结量均无变化 | `ProdChangeFlag.NONE` | `0` |
| 持仓量无变化，冻结量有变化 | `ProdChangeFlag.OnlyFrozen` | `1` |
| 持仓量有变化 | `ProdChangeFlag.PositionChange` | `2` |

#### 320. MarketMakeState

做市状态

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 否 | `MarketMakeState.NO` | `0` |
| 是 | `MarketMakeState.YES` | `1` |

#### 321. PwdRcdSrc

历史密码来源类型

| 名称 | Python常量 | 值 |
| :--- | :--- | :--- |
| 来源于Sync初始化数据 | `PwdRcdSrc.Init` | `0` |
| 来源于实时上场数据 | `PwdRcdSrc.Sync` | `1` |
| 来源于用户修改 | `PwdRcdSrc.UserUpd` | `2` |
| 来源于超户修改，很可能来自主席同步数据 | `PwdRcdSrc.SuperUserUpd` | `3` |

