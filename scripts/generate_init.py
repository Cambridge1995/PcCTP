#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 __init__.py 文件的脚本

支持两种模式：
1. 平台特定模式：生成 PcCTP/{platform}/__init__.py（简化版）
2. 主入口模式：生成 PcCTP/__init__.py（完整版，带平台检测）

支持两种绑定类型：
1. nanobind：使用 PyMdSpi 类
2. Python C API：不使用 PyMdSpi 类
"""
import os
import sys
import re
from pathlib import Path

# =============================================================================
# nanobind 版本的模板
# =============================================================================

# nanobind 平台特定模板
PLATFORM_INIT_TEMPLATE_NANOBIND = '''# {module_name} - {platform} 平台特定模块
# 自动生成文件，请勿手动修改

# 导入模块的所有内容（包含 PyMdSpi, MdApi 等）
from .{module_name} import *

# 定义__all__列表
__all__ = [
    # 字符串池监控和清理函数
    "cleanup_temporal_pools", "cleanup_instruments",
    "check_instrument_pool_size", "get_pool_sizes",
    # 核心类（2个）
    "PyMdSpi", "MdApi",
]
'''

# nanobind 主入口模板
MAIN_INIT_TEMPLATE_NANOBIND = '''# {module_name} - CTP API Python 绑定
# 自动生成文件，请勿手动修改
# 模块常量
# =============================================================================

# PcCTP 模块版本号
__version__ = 'v{pcctp_version}'
# CTP 接口版本号
__ctp_version__ = '{version}'
# 采集库fix版本号
__fix_version__ = '{fix_version}'
# 运行平台
__platform__ = '{platform}'
# 版本类型
__version_type__ = '{version_type}'

__full_version__ = '{module_name} v{pcctp_version} | CTP {ctp_version} | FIX {fix_version} | {version_type} | {ctp_platform}'
import platform
import os
# 获取操作系统类型
def get_os_type():
    """获取操作系统类型: win64, win32, linux, macos, 或其他"""
    system = platform.system().lower()
    if system == "windows":
        # 通过环境变量检测系统架构
        arch_env = os.environ.get('PROCESSOR_ARCHITECTURE', '')
        arch_env_w6432 = os.environ.get('PROCESSOR_ARCHITEW6432', '')
        if 'AMD64' in (arch_env, arch_env_w6432):
            return 'win64'
        else:
            return 'win32'
    elif system == "linux":
        return 'linux'
    elif system == "darwin":
        return 'macos'
    else:
        return 'other'

# 调用方法获取当前系统类型
env = get_os_type()

# 通过系统类型导入对应的接口
if env == 'win64':  # windows64位
    from PcCTP.win64 import *
    print('导入PyCTP')
elif env == 'win32':  # windows32位
    from PcCTP.win32 import *
elif env == 'linux':  # linux
    from PcCTP.linux import *
else:  # 其他系统 暂不支持,如：macOS
    raise EnvironmentError('本CTP版本与当前系统不匹配')

# 导入所有枚举类（支持 from win64 import Direction, OffsetFlag 等）
from .enums import *
from .types import *

# 定义__all__列表，包含所有导出的名称
__all__ = [
    # 枚举类
{enum_names},
    # 方法
    "validate_direction", "validate_offset_flag",
    "validate_order_price_type", "get_direction_name",
    "get_offset_flag_name",
    # 错误码映射
    "reason_map",
    # 字符串池监控和清理函数
    "cleanup_temporal_pools", "cleanup_instruments",
    "check_instrument_pool_size", "get_pool_sizes",
    # 核心类（2个）
    "PyMdSpi", "MdApi",
    # TypedDict（{type_count}个）
{type_names},
]
'''

# =============================================================================
# Python C API 版本的模板
# =============================================================================

# Python C API 平台特定模板
PLATFORM_INIT_TEMPLATE_PYTHON_C_API = '''# {module_name} - {platform} 平台特定模块
# 自动生成文件，请勿手动修改

# 导入模块的所有内容（包含 MdApi 等）
from .{module_name} import *

# 定义__all__列表
__all__ = [
    # 字符串池监控和清理函数
    "cleanup_temporal_pools", "cleanup_instruments",
    "check_instrument_pool_size", "get_pool_sizes",
    # 核心类（1个）
    "MdApi",
]
'''

# Python C API 主入口模板
MAIN_INIT_TEMPLATE_PYTHON_C_API = '''# {module_name} - CTP API Python 绑定
# 自动生成文件，请勿手动修改
# 模块常量
# =============================================================================

# PcCTP 模块版本号
__version__ = 'v{pcctp_version}'
# CTP 接口版本号
__ctp_version__ = '{version}'
# 采集库fix版本号
__fix_version__ = '{fix_version}'
# 运行平台
__platform__ = '{platform}'
# 版本类型
__version_type__ = '{version_type}'

__full_version__ = '{module_name} v{pcctp_version} | CTP {ctp_version} | FIX {fix_version} | {version_type} | {ctp_platform}'
import platform
import os
from PcCTP.interface import *
# 获取操作系统类型
def get_os_type():
    """获取操作系统类型: win64, win32, linux, macos, 或其他"""
    system = platform.system().lower()
    if system == "windows":
        # 通过环境变量检测系统架构
        arch_env = os.environ.get('PROCESSOR_ARCHITECTURE', '')
        arch_env_w6432 = os.environ.get('PROCESSOR_ARCHITEW6432', '')
        if 'AMD64' in (arch_env, arch_env_w6432):
            return 'win64'
        else:
            return 'win32'
    elif system == "linux":
        return 'linux'
    elif system == "darwin":
        return 'macos'
    else:
        return 'other'

# 调用方法获取当前系统类型
env = get_os_type()

# 通过系统类型导入对应的接口
if env == 'win64':  # windows64位
    from PcCTP.win64 import *
    print('导入PyCTP')
elif env == 'win32':  # windows32位
    from PcCTP.win32 import *
elif env == 'linux':  # linux
    from PcCTP.linux import *
else:  # 其他系统 暂不支持,如：macOS
    raise EnvironmentError('本CTP版本与当前系统不匹配')

# 导入所有枚举类（支持 from win64 import Direction, OffsetFlag 等）
from PcCTP.enums import *
from PcCTP.types import *

# 定义__all__列表，包含所有导出的名称
__all__ = [
    # 枚举类
{enum_names},
    # 方法
    "validate_direction", "validate_offset_flag",
    "validate_order_price_type", "get_direction_name",
    "get_offset_flag_name",
    # 错误码映射
    "reason_map",
    # 字符串池监控和清理函数
    "cleanup_temporal_pools", "cleanup_instruments",
    "check_instrument_pool_size", "get_pool_sizes",
    # 核心类（1个）
    "MdApi",
    # TypedDict（{type_count}个）
{type_names},
]
'''


def extract_enum_names(enums_file: Path) -> list[str]:
    """从 enums.py 文件中提取所有枚举类名"""
    content = Path(enums_file).read_text(encoding='utf-8')
    # 匹配 class XxxName(StrEnum):
    pattern = r'^class (\w+)\(StrEnum\):'
    matches = re.findall(pattern, content, flags=re.MULTILINE)
    return sorted(matches)


def extract_type_names(types_file: Path) -> list[str]:
    """从 types.py 文件中提取所有 TypedDict 类名"""
    content = Path(types_file).read_text(encoding='utf-8')
    # 匹配 class XxxName(TypedDict): 或 class XxxName(TypedDict, ...):
    pattern = r'^class (\w+)\(TypedDict\b'
    matches = re.findall(pattern, content, flags=re.MULTILINE)
    return sorted(matches)


def format_list(items: list[str], indent: str = '    ') -> str:
    """将类名列表格式化为 __all__ 列表格式"""
    if not items:
        return ''

    lines = []
    for i, name in enumerate(items):
        if i == len(items) - 1:
            # 最后一个不加逗号
            lines.append(f'{indent}"{name}"')
        else:
            lines.append(f'{indent}"{name}",')
    return '\n'.join(lines)


def get_binding_type():
    """获取绑定类型：nanobind 或 Python C API"""
    use_nanobind = os.environ.get('USE_NANOBIND', 'OFF')
    return 'nanobind' if use_nanobind.upper() == 'ON' or use_nanobind.upper() == '1' or use_nanobind.upper() == 'TRUE' else 'python_c_api'


def generate_platform_init():
    """生成平台特定的 __init__.py 文件（PcCTP/{platform}/__init__.py）"""
    ctp_output_dir = os.environ.get('CTP_OUTPUT_DIR')
    ctp_platform = os.environ.get('CTP_PLATFORM', 'unknown')
    module_name = os.environ.get('MODULE_NAME', 'PcCTP')
    binding_type = get_binding_type()

    if not ctp_output_dir:
        print("错误: CTP_OUTPUT_DIR 环境变量未设置", file=sys.stderr)
        return 1

    # 根据绑定类型选择模板
    if binding_type == 'nanobind':
        template = PLATFORM_INIT_TEMPLATE_NANOBIND
        print(f"[信息] 使用 nanobind 平台特定模板")
    else:
        template = PLATFORM_INIT_TEMPLATE_PYTHON_C_API
        print(f"[信息] 使用 Python C API 平台特定模板")

    # 使用模板
    content = template.format(
        module_name=module_name,
        platform=ctp_platform,
    )

    output_file = os.path.join(ctp_output_dir, '__init__.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"生成 {output_file}")
    return 0


def generate_main_init():
    """生成主入口的 __init__.py 文件（PcCTP/__init__.py）"""
    ctp_base_dir = os.environ.get('CTP_BASE_DIR')
    ctp_version = os.environ.get('CTP_VERSION', 'unknown')
    ctp_platform = os.environ.get('CTP_PLATFORM', 'unknown')
    version_type_display = os.environ.get('VERSION_TYPE_DISPLAY', 'PC')
    pcctp_version = os.environ.get('PCCTP_VERSION', '1.0.0')
    fix_version = os.environ.get('FIX_VERSION', 'unknown')
    module_name = os.environ.get('MODULE_NAME', 'PcCTP')
    binding_type = get_binding_type()

    if not ctp_base_dir:
        print("错误: CTP_BASE_DIR 环境变量未设置", file=sys.stderr)
        return 1

    base_dir = Path(__file__).parent.parent

    # 从 enums.py 提取枚举类名
    enums_file = base_dir / 'src/enums.py'
    if not enums_file.exists():
        print(f"警告: 找不到 {enums_file}，使用默认枚举列表", file=sys.stderr)
        enum_names = [
            "AMLCheckStatus", "AMLGenStatus", "APIProductClass", "AccountSettlementParamID",
            "AccountSourceType", "ActionDirection", "ActionFlag", "ActionType", "ActiveType", "AlgoType", "Algorithm",
            "AllWithoutTrade", "AmType", "AmlCheckLevel", "AmlDateType", "AppType", "ApplyOperateID", "ApplyStatusID",
            "ApplyType", "AssetmgrClientType", "AssetmgrType", "AuthType", "AvailabilityFlag", "BackUpStatus", "BalanceAlgorithm",
            "BalanceType", "BankAccStatus", "BankAccType", "BankAccountOrigin", "BankRepealFlag", "BasisPriceType", "BatchStatus",
            "BillGenStatus", "BillHedgeFlag", "BizType", "BrokerDataSyncStatus", "BrokerFunctionCode", "BrokerRepealFlag",
            "BrokerType", "BrokerUserType", "BusinessClass", "BusinessType", "ByGroup", "ByInvestorRange", "CCBFeeMode",
            "CFFEXUploadFileName", "CFMMCKeyKind", "CSRCDataQueryType", "CSRCFundIOType", "CTPType", "CZCEUploadFileName",
            "CashExchangeCode", "CertificationType", "CfmmcReturnCode", "CheckInstrType", "CheckLevel", "CheckStatus",
            "ClassType", "ClientIDStatus", "ClientIDType", "ClientRegion", "ClientType", "CloseDealType", "CloseStyle", "CodeSourceType",
            "CombDirection", "CombinationType", "CommApiType", "ConditionalOrderSortType", "ConnectMode", "ContingentCondition",
            "CurrExDirection", "CurrencySwapStatus", "CusAccountType", "CustType", "DAClientType", "DBOperation", "DCEUploadFileName",
            "DataResource", "DataStatus", "DataSyncStatus", "DceCombinationType", "DeliveryMode", "DeliveryType", "DepartmentRange",
            "Direction", "DirectionEn", "EnumBool", "EventMode", "ExClientIDType", "ExDirection", "ExStatus", "ExchangeConnectStatus",
            "ExchangeIDType", "ExchangeProperty", "ExchangeSettlementParamID", "ExecOrderCloseFlag", "ExecOrderPositionFlag",
            "ExecResult", "ExportFileType", "ExprSetMode", "FBEAlreadyTrade", "FBEExchStatus", "FBEFileFlag", "FBEReqFlag",
            "FBEResultFlag", "FBEUserEventType", "FBTEncryMode", "FBTPassWordType", "FBTTradeCodeEnum", "FBTTransferDirection",
            "FBTUserEventType", "FeeAcceptStyle", "FeePayFlag", "FileBusinessCode", "FileFormat", "FileGenStyle", "FileID",
            "FileStatus", "FileType", "FileUploadStatus", "FindMarginRateAlgoID", "FlexStatMode", "FlowID", "ForQuoteStatus",
            "ForceCloseReason", "ForceCloseType", "FreezeStatus", "FunctionCode", "FundDirection", "FundDirectionEn",
            "FundEventType", "FundIOType", "FundIOTypeEn", "FundMortDirection", "FundMortDirectionEn", "FundMortgageType",
            "FundStatus", "FundType", "FundTypeEn", "FutureAccType", "FuturePwdFlag", "FutureType", "Gender", "GiveUpDataSource",
            "HandlePositionAlgoID", "HandleTradingAccountAlgoID", "HasBoard", "HasTrustee", "HedgeFlag", "HedgeFlagEn",
            "IdCardType", "IncludeCloseProfit", "InitSettlement", "InstLifePhase", "InstMarginCalID", "InstStatusEnterReason",
            "InstitutionType", "InstrumentClass", "InstrumentStatus", "InvestTradingRight", "InvestorRange", "InvestorRiskStatus",
            "InvestorSettlementParamID", "InvestorType", "LanguageType", "LastFragment", "LimitUseType", "LinkStatus", "LoginMode",
            "ManageStatus", "MarginPriceType", "MarginRateType", "MarginType", "MarketMakeState", "MatchType", "MaxMarginSideAlgorithm",
            "MoneyAccountStatus", "MonthBillTradeSum", "MortgageFundUseRange", "MortgageType", "NoteType", "NotifyClass", "OTCTradeType",
            "OTPStatus", "OTPType", "OffsetFlag", "OffsetType", "OffsetFlagEn", "OpenLimitControlLevel", "OpenOrDestroy", "OptSelfCloseFlag",
            "OptionRoyaltyPriceType", "OptionsType", "OrderActionStatus", "OrderCancelAlg", "OrderFreqControlLevel", "OrderPriceType", "OrderSource",
            "OrderStatus", "OrderSubmitStatus", "OrderType", "OrgSystemID", "OrganLevel", "OrganStatus",
            "OrganType", "ParkedOrderStatus", "PassWordKeyType", "PasswordType", "PersonType", "PortfType", "Portfolio", "PositionDate",
            "PositionDateType", "PositionDirection", "PositionType", "PriceSource", "ProcessStatus", "ProdChangeFlag", "ProductClass", "ProductLifePhase",
            "ProductStatus", "ProductType", "PromptType", "PropertyInvestorRange", "ProtocolID", "PublishStatus", "PwdFlag",
            "PwdRcdSrc", "QueryInvestorRange", "QuestionType", "RCAMSCombinationType", "RateInvestorRange", "RateType",
            "RatioAttr", "Reason", "ReportStatus", "ReqFlag", "ReqRspType", "ResFlag", "ReserveOpenAccStas", "ResponseValue", "ReturnLevel",
            "ReturnPattern", "ReturnStandard", "ReturnStyle", "RightParamType", "RiskLevel", "RiskNotifyMethod", "RiskNotifyStatus",
            "RiskUserEvent", "SHFEUploadFileName", "SaveStatus", "SecuAccType", "SendMethod", "SendType", "SettArchiveStatus", "SettleManagerGroup", "SettleManagerLevel",
            "SettleManagerType", "SettlementBillType", "SettlementStatus", "SettlementStyle", "Sex", "SpecPosiType",
            "SpecProductType", "SpecialCreateRule", "SponsorType", "StandardStatus", "StartMode", "StatMode", "StrikeMode", "StrikeOffsetType", "StrikeType",
            "SwapSourceType", "SyncDataStatus", "SyncDeltaStatus", "SyncFlag", "SyncMode", "SyncType", "SysOperMode", "SysOperType",
            "SysSettlementStatus", "SystemParamID", "SystemStatus", "SystemType", "TemplateType", "TimeCondition", "TimeRange",
            "TradeParamID", "TradeSource", "TradeSumStatMode", "TradeType", "TraderConnectStatus", "TradingRight", "TradingRole", "TradingType",
            "TransferDirection", "TransferStatus", "TransferType", "TransferValidFlag", "TxnEndFlag", "UOAAssetmgrType", "UOAAutoSend", "UpdateFlag",
            "UsedStatus", "UserEventType", "UserRange", "UserRightType", "UserType", "ValueMethod", "VirBankAccType", "VirDealStatus", "VirTradeStatus", "VirementAvailAbility",
            "VirementStatus", "VirementTradeCode", "VolumeCondition", "WeakPasswordSource", "WithDrawParamID", "YesNoIndicator",
        ]
    else:
        enum_names = extract_enum_names(enums_file)
        print(f"从 {enums_file} 提取到 {len(enum_names)} 个枚举类")

    # 从 types.py 提取 TypedDict 类名
    types_file = base_dir / 'src/types.py'
    if not types_file.exists():
        print(f"警告: 找不到 {types_file}，TypedDict 列表将为空", file=sys.stderr)
        type_names = []
    else:
        type_names = extract_type_names(types_file)
        print(f"从 {types_file} 提取到 {len(type_names)} 个 TypedDict")

    # 格式化列表
    enum_names_str = format_list(enum_names)
    type_names_str = format_list(type_names)

    # 根据绑定类型选择模板
    if binding_type == 'nanobind':
        template = MAIN_INIT_TEMPLATE_NANOBIND
        print(f"[信息] 使用 nanobind 主入口模板")
    else:
        template = MAIN_INIT_TEMPLATE_PYTHON_C_API
        print(f"[信息] 使用 Python C API 主入口模板")

    # 使用模板生成内容
    content = template.format(
        module_name=module_name,
        version=ctp_version,
        platform=ctp_platform,
        version_type=version_type_display,
        pcctp_version=pcctp_version,
        ctp_version=ctp_version,
        fix_version=fix_version,
        ctp_platform=ctp_platform,
        enum_names=enum_names_str,
        type_count=len(type_names),
        type_names=type_names_str,
    )

    output_file = os.path.join(ctp_base_dir, '__init__.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"生成 {output_file}")
    return 0


def generate_init_file():
    """根据环境变量决定生成哪种 __init__.py 文件"""
    generate_main = os.environ.get('GENERATE_MAIN_INIT', '0')

    if generate_main == '1':
        # 生成主入口文件
        return generate_main_init()
    else:
        # 生成平台特定文件
        return generate_platform_init()


if __name__ == '__main__':
    sys.exit(generate_init_file())
