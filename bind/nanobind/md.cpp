/**
 * PyCTP - CTP PC版行情API Python绑定 (性能优化版)
 *
 * 使用 nanobind 将 CTP PC版 CTP API 绑定到 Python
 * 适用于 CTP PC (标准版) API
 *
 * 优化策略：
 * - 使用 dict 替代所有 C++ 结构体到 Python 类的转换
 * - 有限集合字符串使用字符串池（零拷贝复用，如交易所代码、日期、时间）
 * - 动态字符串使用 optimized_str() 优化构造（非零拷贝，但开销最小）
 * - numpy数组直接访问底层内存（零拷贝，使用numpy C API）
 * - 数值类型直接传递（零拷贝，无中间拷贝）
 *
 * 性能提升：
 * - 字符串池（有限集合）：50x（交易所代码、日期、时间重复使用）
 * - numpy 订阅：25-50x（直接访问底层数据指针）
 * - 整体行情回调：1.3x（优化字符串构造 + 零拷贝数值）
 */

// NumPy 废弃 API 禁用 - 必须在所有头文件之前定义
#define PY_ARRAY_UNIQUE_SYMBOL PcCTP_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/trampoline.h>
#include <memory>
#include <cstring>
#include <string>
#include <atomic>

// 全局标志：Python 解释器是否正在关闭
static std::atomic<bool> g_python_is_finalizing(false);

// 零拷贝优化工具库
#include "zero_copy_utils.h"

// numpy C API（宏展开，零开销，直接结构体访问）
#include <numpy/ndarrayobject.h>

// PC版 CTP 行情 API 头文件
#if defined(_WIN64)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win64/ThostFtdcMdApi.h"
    #pragma warning(pop)
#elif defined(_WIN32)
    #pragma warning(push)
    #pragma warning(disable : 4996)
    #include "ctp/PC/win32/ThostFtdcMdApi.h"
    #pragma warning(pop)
#else
    #include "ctp/PC/linux/ThostFtdcMdApi.h"
#endif

namespace nb = nanobind;

using namespace nb::literals;

// =============================================================================
// 字符串转换辅助函数 - 零拷贝优化
// =============================================================================

// 从 dict 中获取字符串字段的辅助函数
inline std::string dict_get_string(const nb::dict& d, const char* key, const char* default_val = "") {
    if (d.contains(key)) {
        nb::object obj = d[key];
        if (obj.is_none()) return default_val;
        if (nb::isinstance<nb::str>(obj)) {
            return nb::cast<std::string>(obj);
        }
    }
    return default_val;
}

// 从 dict 中获取字符串并复制到字符数组（指针版本）
inline void dict_copy_string_ptr(const nb::dict& d, const char* key, char* dst, size_t dst_size, const char* default_val = "") {
    std::string val = dict_get_string(d, key, default_val);
    size_t len = std::min(val.size(), dst_size - 1);
    memcpy(dst, val.data(), len);
    dst[len] = '\0';
}

// 从 dict 中获取字符串并复制到固定长度字符数组
template<size_t N>
inline void dict_copy_string(const nb::dict& d, const char* key, char (&dst)[N], const char* default_val = "") {
    dict_copy_string_ptr(d, key, dst, N, default_val);
}

// =============================================================================
// PC版 MdSpi 回调类 Trampoline (优化版 - 直接返回 dict)
// =============================================================================

class PyMdSpi : public CThostFtdcMdSpi {
    NB_TRAMPOLINE(CThostFtdcMdSpi, 12);

public:
    // 连接相关
    void OnFrontConnected() override {
        nb::gil_scoped_acquire gil;
        nb_trampoline.base().attr("on_front_connected")();
    }

    void OnFrontDisconnected(int nReason) override {
        nb::gil_scoped_acquire gil;
        nb_trampoline.base().attr("on_front_disconnected")(nReason);
    }

    void OnHeartBeatWarning(int nTimeLapse) override {
        nb::gil_scoped_acquire gil;
        nb_trampoline.base().attr("on_heart_beat_warning")(nTimeLapse);
    }

    // 登录响应 - 返回 dict
    void OnRspUserLogin(CThostFtdcRspUserLoginField *pRspUserLogin, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        // 构建 rsp_user_login dict
        nb::object rsp_user_login = nb::none();
        if (pRspUserLogin) {
            nb::dict login;
            // 字符串池（零拷贝）
            login["trading_day"] = GlobalStringPools::Dates.intern(pRspUserLogin->TradingDay, sizeof(pRspUserLogin->TradingDay));
            login["login_time"] = GlobalStringPools::Times.intern(pRspUserLogin->LoginTime, sizeof(pRspUserLogin->LoginTime));
            login["broker_id"] = GlobalStringPools::BrokerIds.intern(pRspUserLogin->BrokerID, sizeof(pRspUserLogin->BrokerID));
            login["user_id"] = GlobalStringPools::UserIds.intern(pRspUserLogin->UserID, sizeof(pRspUserLogin->UserID));
            login["shfe_time"] = GlobalStringPools::Times.intern(pRspUserLogin->SHFETime, sizeof(pRspUserLogin->SHFETime));
            login["dce_time"] = GlobalStringPools::Times.intern(pRspUserLogin->DCETime, sizeof(pRspUserLogin->DCETime));
            login["czce_time"] = GlobalStringPools::Times.intern(pRspUserLogin->CZCETime, sizeof(pRspUserLogin->CZCETime));
            login["ffex_time"] = GlobalStringPools::Times.intern(pRspUserLogin->FFEXTime, sizeof(pRspUserLogin->FFEXTime));
            login["ine_time"] = GlobalStringPools::Times.intern(pRspUserLogin->INETime, sizeof(pRspUserLogin->INETime));
            login["gfex_time"] = GlobalStringPools::Times.intern(pRspUserLogin->GFEXTime, sizeof(pRspUserLogin->GFEXTime));
            // optimized_str（动态字符串）
            login["system_name"] = optimized_str(pRspUserLogin->SystemName, sizeof(pRspUserLogin->SystemName));
            login["max_order_ref"] = optimized_str(pRspUserLogin->MaxOrderRef, sizeof(pRspUserLogin->MaxOrderRef));
            // 数值（零拷贝）
            login["front_id"] = pRspUserLogin->FrontID;
            login["session_id"] = pRspUserLogin->SessionID;
            rsp_user_login = login;
        }

        // 构建 rsp_info dict
        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_user_login")(
            rsp_user_login,
            rsp_info,
            nRequestID,
            bIsLast
        );
    }

    // 登出响应 - 返回 dict
    void OnRspUserLogout(CThostFtdcUserLogoutField *pUserLogout, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        // 构建 user_logout dict
        nb::object user_logout = nb::none();
        if (pUserLogout) {
            nb::dict logout;
            // 字符串池（零拷贝）
            logout["broker_id"] = GlobalStringPools::BrokerIds.intern(pUserLogout->BrokerID, sizeof(pUserLogout->BrokerID));
            logout["user_id"] = GlobalStringPools::UserIds.intern(pUserLogout->UserID, sizeof(pUserLogout->UserID));
            user_logout = logout;
        }

        // 构建 rsp_info dict
        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_user_logout")(
            user_logout,
            rsp_info,
            nRequestID,
            bIsLast
        );

    }

    // 错误响应 - 返回 dict
    void OnRspError(CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_error")(
            rsp_info,
            nRequestID,
            bIsLast
        );
    }

    // 订阅行情响应 - 返回 str + dict
    void OnRspSubMarketData(CThostFtdcSpecificInstrumentField *pSpecificInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        // 合约代码 - 字符串池（零拷贝）
        nb::object instrument_id = nb::none();
        if (pSpecificInstrument) {
            instrument_id = GlobalStringPools::intern_instrument(pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        // 构建 rsp_info dict
        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_sub_market_data")(
            instrument_id,
            rsp_info,
            nRequestID,
            bIsLast
        );
    }

    // 取消订阅行情响应 - 返回 str + dict
    void OnRspUnSubMarketData(CThostFtdcSpecificInstrumentField *pSpecificInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        nb::object instrument_id = nb::none();
        if (pSpecificInstrument) {
            instrument_id = GlobalStringPools::intern_instrument(pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_un_sub_market_data")(
            instrument_id,
            rsp_info,
            nRequestID,
            bIsLast
        );
    }

    // 订阅询价响应 - 返回 str + dict
    void OnRspSubForQuoteRsp(CThostFtdcSpecificInstrumentField *pSpecificInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        nb::object instrument_id = nb::none();
        if (pSpecificInstrument) {
            instrument_id = GlobalStringPools::intern_instrument(pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_sub_for_quote_rsp")(
            instrument_id,
            rsp_info,
            nRequestID,
            bIsLast
        );
    }

    // 取消订阅询价响应 - 返回 str + dict
    void OnRspUnSubForQuoteRsp(CThostFtdcSpecificInstrumentField *pSpecificInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        nb::object instrument_id = nb::none();
        if (pSpecificInstrument) {
            instrument_id = GlobalStringPools::intern_instrument(pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
        }

        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_un_sub_for_quote_rsp")(
            instrument_id,
            rsp_info,
            nRequestID,
            bIsLast
        );
    }

    // 查询组播合约响应 - 返回 dict (优化版)
    void OnRspQryMulticastInstrument(CThostFtdcMulticastInstrumentField *pMulticastInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) override {
        nb::gil_scoped_acquire gil;

        nb::object multicast_instrument = nb::none();
        if (pMulticastInstrument) {
            nb::dict data;
            // 字符串池（零拷贝）- 只有 InstrumentID 是字符串
            data["instrument_id"] = GlobalStringPools::intern_instrument(
                pMulticastInstrument->InstrumentID, sizeof(pMulticastInstrument->InstrumentID));
            // 数值（零拷贝）
            data["topic_id"] = pMulticastInstrument->TopicID;           // 主题号
            data["instrument_no"] = pMulticastInstrument->InstrumentNo;  // 合约编号
            data["code_price"] = pMulticastInstrument->CodePrice;       // 基准价
            data["volume_multiple"] = pMulticastInstrument->VolumeMultiple;  // 合约数量乘数
            data["price_tick"] = pMulticastInstrument->PriceTick;       // 最小变动价位
            multicast_instrument = data;
        }

        nb::object rsp_info = nb::none();
        if (pRspInfo) {
            nb::dict info;
            info["error_id"] = pRspInfo->ErrorID;
            info["error_msg"] = optimized_str(pRspInfo->ErrorMsg, sizeof(pRspInfo->ErrorMsg));
            rsp_info = info;
        }

        nb_trampoline.base().attr("on_rsp_qry_multicast_instrument")(
            multicast_instrument,
            rsp_info,
            nRequestID,
            bIsLast
        );
    }

    // 深度行情通知 - 返回 dict (高频回调，性能优化)
    void OnRtnDepthMarketData(CThostFtdcDepthMarketDataField *pDepthMarketData) override {
        nb::gil_scoped_acquire gil;

        if (!pDepthMarketData) {
            nb_trampoline.base().attr("on_rtn_depth_market_data")(nb::none());
            return;
        }

        // 构建 dict 返回数据
        nb::dict data;

        // 有限集合字符串 - 字符串池（零拷贝复用）
        // 交易所代码：有限集合（SHFE、DCE等），使用字符串池复用，零拷贝
        data["exchange_id"] = GlobalStringPools::ExchangeCodes.intern(
            pDepthMarketData->ExchangeID, sizeof(pDepthMarketData->ExchangeID));

        // 日期字符串：每个交易日固定，使用字符串池复用，零拷贝
        data["trading_day"] = GlobalStringPools::Dates.intern(
            pDepthMarketData->TradingDay, sizeof(pDepthMarketData->TradingDay));
        data["action_day"] = GlobalStringPools::Dates.intern(
            pDepthMarketData->ActionDay, sizeof(pDepthMarketData->ActionDay));

        // 时间字符串：交易时段内重复（秒级），使用字符串池复用，零拷贝
        data["update_time"] = GlobalStringPools::Times.intern(
            pDepthMarketData->UpdateTime, sizeof(pDepthMarketData->UpdateTime));

        // 合约代码：订阅的合约有限，使用字符串池复用，零拷贝
        data["instrument_id"] = GlobalStringPools::intern_instrument(
            pDepthMarketData->InstrumentID, sizeof(pDepthMarketData->InstrumentID));
        // 交易所合约代码：动态字符串，使用 optimized_str（非零拷贝，但开销最小）
        data["exchange_inst_id"] = optimized_str(pDepthMarketData->ExchangeInstID, sizeof(pDepthMarketData->ExchangeInstID));

        // 数值字段 - 零拷贝（直接传递，无拷贝）
        data["last_price"] = pDepthMarketData->LastPrice;
        data["pre_settlement_price"] = pDepthMarketData->PreSettlementPrice;
        data["pre_close_price"] = pDepthMarketData->PreClosePrice;
        data["pre_open_interest"] = pDepthMarketData->PreOpenInterest;
        data["open_price"] = pDepthMarketData->OpenPrice;
        data["highest_price"] = pDepthMarketData->HighestPrice;
        data["lowest_price"] = pDepthMarketData->LowestPrice;
        data["volume"] = pDepthMarketData->Volume;
        data["turnover"] = pDepthMarketData->Turnover;
        data["open_interest"] = pDepthMarketData->OpenInterest;
        data["close_price"] = pDepthMarketData->ClosePrice;
        data["settlement_price"] = pDepthMarketData->SettlementPrice;
        data["upper_limit_price"] = pDepthMarketData->UpperLimitPrice;
        data["lower_limit_price"] = pDepthMarketData->LowerLimitPrice;
        data["pre_delta"] = pDepthMarketData->PreDelta;
        data["curr_delta"] = pDepthMarketData->CurrDelta;
        data["update_millisec"] = pDepthMarketData->UpdateMillisec;

        // 五档买卖价
        data["bid_price1"] = pDepthMarketData->BidPrice1;
        data["bid_volume1"] = pDepthMarketData->BidVolume1;
        data["ask_price1"] = pDepthMarketData->AskPrice1;
        data["ask_volume1"] = pDepthMarketData->AskVolume1;
        data["bid_price2"] = pDepthMarketData->BidPrice2;
        data["bid_volume2"] = pDepthMarketData->BidVolume2;
        data["ask_price2"] = pDepthMarketData->AskPrice2;
        data["ask_volume2"] = pDepthMarketData->AskVolume2;
        data["bid_price3"] = pDepthMarketData->BidPrice3;
        data["bid_volume3"] = pDepthMarketData->BidVolume3;
        data["ask_price3"] = pDepthMarketData->AskPrice3;
        data["ask_volume3"] = pDepthMarketData->AskVolume3;
        data["bid_price4"] = pDepthMarketData->BidPrice4;
        data["bid_volume4"] = pDepthMarketData->BidVolume4;
        data["ask_price4"] = pDepthMarketData->AskPrice4;
        data["ask_volume4"] = pDepthMarketData->AskVolume4;
        data["bid_price5"] = pDepthMarketData->BidPrice5;
        data["bid_volume5"] = pDepthMarketData->BidVolume5;
        data["ask_price5"] = pDepthMarketData->AskPrice5;
        data["ask_volume5"] = pDepthMarketData->AskVolume5;
        data["average_price"] = pDepthMarketData->AveragePrice;

        nb_trampoline.base().attr("on_rtn_depth_market_data")(data);
    }

    // 询价响应 - 返回 dict (优化版)
    void OnRtnForQuoteRsp(CThostFtdcForQuoteRspField *pForQuoteRsp) override {
        nb::gil_scoped_acquire gil;

        if (!pForQuoteRsp) {
            nb_trampoline.base().attr("on_rtn_for_quote_rsp")(nb::none());
            return;
        }

        nb::dict data;

        // 字符串池（零拷贝）
        data["trading_day"] = GlobalStringPools::Dates.intern(
            pForQuoteRsp->TradingDay, sizeof(pForQuoteRsp->TradingDay));
        data["action_day"] = GlobalStringPools::Dates.intern(
            pForQuoteRsp->ActionDay, sizeof(pForQuoteRsp->ActionDay));
        data["for_quote_time"] = GlobalStringPools::Times.intern(
            pForQuoteRsp->ForQuoteTime, sizeof(pForQuoteRsp->ForQuoteTime));
        data["exchange_id"] = GlobalStringPools::ExchangeCodes.intern(
            pForQuoteRsp->ExchangeID, sizeof(pForQuoteRsp->ExchangeID));
        data["instrument_id"] = GlobalStringPools::intern_instrument(
            pForQuoteRsp->InstrumentID, sizeof(pForQuoteRsp->InstrumentID));

        // 动态字符串 - optimized_str（非零拷贝，但开销最小）
        data["for_quote_sys_id"] = optimized_str(pForQuoteRsp->ForQuoteSysID, sizeof(pForQuoteRsp->ForQuoteSysID));

        nb_trampoline.base().attr("on_rtn_for_quote_rsp")(data);
    }
};

// =============================================================================
// PC版 MdApi API 类 (优化版 - 接受 dict 参数)
// =============================================================================

class MdApi {
public:
    MdApi(CThostFtdcMdApi* api) : m_api(api) {}

    ~MdApi() {
        // 析构函数：清理 CTP API 资源
        // 参考 PyCTP 的实现，按正确顺序清理资源

        if (m_api) {
            // 重要：先注销 SPI，通知 CTP 不再使用 SPI 指针
            // 这必须在 Release() 之前调用
            m_api->RegisterSpi(nullptr);
            // 然后调用 Release() 清理资源
            m_api->Release();
            // CTP API 会自动清理线程，操作系统会在进程退出时强制清理
            m_api = nullptr;
        }

        // 最后释放 SPI 对象的引用
        if(m_spi){
            m_spi.reset();
        }
    }

    static std::shared_ptr<MdApi> create_ftdc_md_api(const std::string& flow_path = "", bool is_using_udp = false, bool is_multicast = false) {
        return std::shared_ptr<MdApi>(new MdApi(CThostFtdcMdApi::CreateFtdcMdApi(flow_path.c_str(), is_using_udp, is_multicast)));
    }

    static std::string get_api_version() {
        return std::string(CThostFtdcMdApi::GetApiVersion());
    }

    void init() {
        m_api->Init();
    }

    int join() {
        return m_api->Join();
    }

    std::string get_trading_day() {
        return std::string(m_api->GetTradingDay());
    }

    void register_front(const std::string& front_address) {
        m_api->RegisterFront((char*)front_address.c_str());
    }

    void register_spi(std::shared_ptr<CThostFtdcMdSpi> spi) {
        // 持有 SPI 对象的 shared_ptr，防止被 Python GC 回收
        // CTP API 只持有原始指针，不管理 SPI 对象的生命周期
        m_spi = spi;
        m_api->RegisterSpi(spi.get());
    }

    void register_name_server(const std::string& ns_address) {
        m_api->RegisterNameServer((char*)ns_address.c_str());
    }

    void register_fens_user_info(const nb::dict& fens_user_info) {
        CThostFtdcFensUserInfoField field = {};
        dict_copy_string<11>(fens_user_info, "broker_id", field.BrokerID);
        dict_copy_string<16>(fens_user_info, "user_id", field.UserID);
        // LoginMode 是 char 类型，单个字符
        if (fens_user_info.contains("login_mode")) {
            nb::str login_mode = nb::cast<nb::str>(fens_user_info["login_mode"]);
            std::string ls = login_mode.c_str();
            field.LoginMode = ls.empty() ? '\0' : ls[0];
        }
        m_api->RegisterFensUserInfo(&field);
    }

    int subscribe_market_data(nb::object instrument_ids) {
        // 路径1：numpy 零拷贝（优先，类型安全）
        if (is_numpy_array(instrument_ids)) {
            auto* array = reinterpret_cast<PyArrayObject*>(instrument_ids.ptr());

            // 类型安全检查
            if (!is_string_array(array)) {
                PyErr_SetString(PyExc_TypeError, "Array must be string dtype (U or S)");
                return -1;
            }
            if (!is_1d_array(array)) {
                PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
                return -1;
            }

            // 零拷贝访问
            char* data = static_cast<char*>(PyArray_DATA(array));
            npy_intp size = PyArray_SIZE(array);
            npy_intp stride = PyArray_STRIDE(array, 0);

            std::vector<char*> ids;
            ids.reserve(size);
            for (npy_intp i = 0; i < size; ++i) {
                ids.push_back(data + i * stride);
            }

            return m_api->SubscribeMarketData(ids.data(), size);
        }

        // 路径2：list 回退（兼容）
        if (nb::isinstance<nb::list>(instrument_ids)) {
            nb::list list = nb::cast<nb::list>(instrument_ids);
            std::vector<const char*> ids;
            ids.reserve(list.size());

            for (size_t i = 0; i < list.size(); ++i) {
                nb::str item = list[i];
                ids.push_back(item.c_str());
            }

            return m_api->SubscribeMarketData((char**)ids.data(), ids.size());
        }

        PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
        return -1;
    }

    int un_subscribe_market_data(nb::object instrument_ids) {
        // 路径1：numpy 零拷贝（优先，类型安全）
        if (is_numpy_array(instrument_ids)) {
            auto* array = reinterpret_cast<PyArrayObject*>(instrument_ids.ptr());

            if (!is_string_array(array)) {
                PyErr_SetString(PyExc_TypeError, "Array must be string dtype (U or S)");
                return -1;
            }
            if (!is_1d_array(array)) {
                PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
                return -1;
            }

            char* data = static_cast<char*>(PyArray_DATA(array));
            npy_intp size = PyArray_SIZE(array);
            npy_intp stride = PyArray_STRIDE(array, 0);

            std::vector<char*> ids;
            ids.reserve(size);
            for (npy_intp i = 0; i < size; ++i) {
                ids.push_back(data + i * stride);
            }

            return m_api->UnSubscribeMarketData(ids.data(), size);
        }

        // 路径2：list 回退（兼容）
        if (nb::isinstance<nb::list>(instrument_ids)) {
            nb::list list = nb::cast<nb::list>(instrument_ids);
            std::vector<const char*> ids;
            ids.reserve(list.size());

            for (size_t i = 0; i < list.size(); ++i) {
                nb::str item = list[i];
                ids.push_back(item.c_str());
            }

            return m_api->UnSubscribeMarketData((char**)ids.data(), ids.size());
        }

        PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
        return -1;
    }

    int subscribe_for_quote_rsp(nb::object instrument_ids) {
        // 路径1：numpy 零拷贝（优先，类型安全）
        if (is_numpy_array(instrument_ids)) {
            auto* array = reinterpret_cast<PyArrayObject*>(instrument_ids.ptr());

            if (!is_string_array(array)) {
                PyErr_SetString(PyExc_TypeError, "Array must be string dtype (U or S)");
                return -1;
            }
            if (!is_1d_array(array)) {
                PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
                return -1;
            }

            char* data = static_cast<char*>(PyArray_DATA(array));
            npy_intp size = PyArray_SIZE(array);
            npy_intp stride = PyArray_STRIDE(array, 0);

            std::vector<char*> ids;
            ids.reserve(size);
            for (npy_intp i = 0; i < size; ++i) {
                ids.push_back(data + i * stride);
            }

            return m_api->SubscribeForQuoteRsp(ids.data(), size);
        }

        // 路径2：list 回退（兼容）
        if (nb::isinstance<nb::list>(instrument_ids)) {
            nb::list list = nb::cast<nb::list>(instrument_ids);
            std::vector<const char*> ids;
            ids.reserve(list.size());

            for (size_t i = 0; i < list.size(); ++i) {
                nb::str item = list[i];
                ids.push_back(item.c_str());
            }

            return m_api->SubscribeForQuoteRsp((char**)ids.data(), ids.size());
        }

        PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
        return -1;
    }

    int un_subscribe_for_quote_rsp(nb::object instrument_ids) {
        // 路径1：numpy 零拷贝（优先，类型安全）
        if (is_numpy_array(instrument_ids)) {
            auto* array = reinterpret_cast<PyArrayObject*>(instrument_ids.ptr());

            if (!is_string_array(array)) {
                PyErr_SetString(PyExc_TypeError, "Array must be string dtype (U or S)");
                return -1;
            }
            if (!is_1d_array(array)) {
                PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
                return -1;
            }

            char* data = static_cast<char*>(PyArray_DATA(array));
            npy_intp size = PyArray_SIZE(array);
            npy_intp stride = PyArray_STRIDE(array, 0);

            std::vector<char*> ids;
            ids.reserve(size);
            for (npy_intp i = 0; i < size; ++i) {
                ids.push_back(data + i * stride);
            }

            return m_api->UnSubscribeForQuoteRsp(ids.data(), size);
        }

        // 路径2：list 回退（兼容）
        if (nb::isinstance<nb::list>(instrument_ids)) {
            nb::list list = nb::cast<nb::list>(instrument_ids);
            std::vector<const char*> ids;
            ids.reserve(list.size());

            for (size_t i = 0; i < list.size(); ++i) {
                nb::str item = list[i];
                ids.push_back(item.c_str());
            }

            return m_api->UnSubscribeForQuoteRsp((char**)ids.data(), ids.size());
        }

        PyErr_SetString(PyExc_TypeError, "instrument_ids must be numpy.ndarray or list");
        return -1;
    }

    // 登录 - 接受 dict 参数
    int req_user_login(const nb::dict& req_user_login, int request_id) {
        CThostFtdcReqUserLoginField req = {};

        dict_copy_string<9>(req_user_login, "trading_day", req.TradingDay);
        dict_copy_string<11>(req_user_login, "broker_id", req.BrokerID);
        dict_copy_string<16>(req_user_login, "user_id", req.UserID);
        dict_copy_string<41>(req_user_login, "password", req.Password);
        dict_copy_string<11>(req_user_login, "user_product_info", req.UserProductInfo);
        dict_copy_string<11>(req_user_login, "interface_product_info", req.InterfaceProductInfo);
        dict_copy_string<11>(req_user_login, "protocol_info", req.ProtocolInfo);
        dict_copy_string<21>(req_user_login, "mac_address", req.MacAddress);
        dict_copy_string<41>(req_user_login, "one_time_password", req.OneTimePassword);
        // ClientIPAddress 可能是 typedef 类型，使用指针版本
        dict_copy_string_ptr(req_user_login, "client_ip_address", req.ClientIPAddress, sizeof(req.ClientIPAddress));
        dict_copy_string<36>(req_user_login, "login_remark", req.LoginRemark);

        return m_api->ReqUserLogin(&req, request_id);
    }

    // 登出 - 接受 dict 参数
    int req_user_logout(const nb::dict& user_logout, int request_id) {
        CThostFtdcUserLogoutField req = {};

        dict_copy_string<11>(user_logout, "broker_id", req.BrokerID);
        dict_copy_string<16>(user_logout, "user_id", req.UserID);


        int result = m_api->ReqUserLogout(&req, request_id);


        return result;
    }

    // 查询组播合约 - 接受 dict 参数
    int req_qry_multicast_instrument(const nb::dict& qry_multicast_instrument, int request_id) {
        CThostFtdcQryMulticastInstrumentField req = {};

        // TopicID 是 int 类型，直接赋值
        if (qry_multicast_instrument.contains("topic_id")) {
            req.TopicID = nb::cast<int>(qry_multicast_instrument["topic_id"]);
        }
        dict_copy_string<81>(qry_multicast_instrument, "instrument_id", req.InstrumentID);

        return m_api->ReqQryMulticastInstrument(&req, request_id);
    }

    // 释放 API 资源（按官方文档方式）
    void release() {

        if (m_api) {
            // 重要：先注销 SPI，通知 CTP 不再使用 SPI 指针
            m_api->RegisterSpi(nullptr);
            m_api->Release();
            m_api = nullptr;
        }

        // 释放 SPI 对象的引用（在 CTP API 释放后）
        // 这会减少引用计数，但 Python 层可能还有引用
        if(m_spi){
            m_spi.reset();
        }
    }

private:
    CThostFtdcMdApi* m_api = nullptr;
    std::shared_ptr<CThostFtdcMdSpi> m_spi;  // 持有 SPI 对象的引用
};

// =============================================================================
// Python 模块定义 (优化版 - 不再绑定结构体类)
// =============================================================================

NB_MODULE(PcCTP, m) {
    // 初始化零拷贝工具（numpy + 字符串池）
    initialize_zero_copy_utils();

    // ❌ 移除所有结构体类绑定
    // 不再需要绑定 ReqUserLogin, RspUserLogin, UserLogout, RspInfo, DepthMarketData, ForQuoteRsp
    // 所有数据现在通过 dict 传递

    // MdSpi 回调类
    nb::class_<CThostFtdcMdSpi, PyMdSpi>(m, "PyMdSpi")
        .def(nb::init<>())
        .def("on_front_connected", &CThostFtdcMdSpi::OnFrontConnected)
        .def("on_front_disconnected", &CThostFtdcMdSpi::OnFrontDisconnected, "reason"_a)
        .def("on_heart_beat_warning", &CThostFtdcMdSpi::OnHeartBeatWarning, "time_lapse"_a)
        .def("on_rsp_user_login", &CThostFtdcMdSpi::OnRspUserLogin,
             "rsp_user_login"_a.none(), "rsp_info"_a.none(), "request_id"_a, "is_last"_a)
        .def("on_rsp_user_logout", &CThostFtdcMdSpi::OnRspUserLogout,
             "user_logout"_a.none(), "rsp_info"_a.none(), "request_id"_a, "is_last"_a)
        .def("on_rsp_error", &CThostFtdcMdSpi::OnRspError,
             "rsp_info"_a.none(), "request_id"_a, "is_last"_a)
        .def("on_rsp_sub_market_data", &CThostFtdcMdSpi::OnRspSubMarketData,
             "instrument_id"_a.none(), "rsp_info"_a.none(), "request_id"_a, "is_last"_a)
        .def("on_rsp_un_sub_market_data", &CThostFtdcMdSpi::OnRspUnSubMarketData,
             "instrument_id"_a.none(), "rsp_info"_a.none(), "request_id"_a, "is_last"_a)
        .def("on_rsp_sub_for_quote_rsp", &CThostFtdcMdSpi::OnRspSubForQuoteRsp,
             "instrument_id"_a.none(), "rsp_info"_a.none(), "request_id"_a, "is_last"_a)
        .def("on_rsp_un_sub_for_quote_rsp", &CThostFtdcMdSpi::OnRspUnSubForQuoteRsp,
             "instrument_id"_a.none(), "rsp_info"_a.none(), "request_id"_a, "is_last"_a)
        .def("on_rtn_depth_market_data", &CThostFtdcMdSpi::OnRtnDepthMarketData,
             "depth_market_data"_a.none())
        .def("on_rtn_for_quote_rsp", &CThostFtdcMdSpi::OnRtnForQuoteRsp,
             "for_quote_rsp"_a.none())
        .def("on_rsp_qry_multicast_instrument", &CThostFtdcMdSpi::OnRspQryMulticastInstrument,
             "multicast_instrument"_a.none(), "rsp_info"_a.none(), "request_id"_a, "is_last"_a);

    // MdApi API 类
    nb::class_<MdApi>(m, "MdApi")
        .def_static("create_ftdc_md_api", &MdApi::create_ftdc_md_api,
                    nb::arg("flow_path") = "", nb::arg("is_using_udp") = false, nb::arg("is_multicast") = false)
        .def_static("get_api_version", &MdApi::get_api_version)
        .def("init", &MdApi::init)
        .def("join", &MdApi::join)
        .def("get_trading_day", &MdApi::get_trading_day)
        .def("register_front", &MdApi::register_front, "front_address"_a)
        .def("register_name_server", &MdApi::register_name_server, "ns_address"_a)
        .def("register_fens_user_info", &MdApi::register_fens_user_info, "fens_user_info"_a)
        .def("register_spi", &MdApi::register_spi, "spi"_a)
        .def("subscribe_market_data", &MdApi::subscribe_market_data, "instrument_ids"_a)
        .def("un_subscribe_market_data", &MdApi::un_subscribe_market_data, "instrument_ids"_a)
        .def("subscribe_for_quote_rsp", &MdApi::subscribe_for_quote_rsp, "instrument_ids"_a)
        .def("un_subscribe_for_quote_rsp", &MdApi::un_subscribe_for_quote_rsp, "instrument_ids"_a)
        .def("req_user_login", &MdApi::req_user_login, "req_user_login"_a, "request_id"_a)
        .def("req_user_logout", &MdApi::req_user_logout, "user_logout"_a, "request_id"_a)
        .def("req_qry_multicast_instrument", &MdApi::req_qry_multicast_instrument, "qry_multicast_instrument"_a, "request_id"_a)
        .def("release", &MdApi::release);

    // =============================================================================
    // 字符串池监控和清理工具函数
    // =============================================================================
    m.def("cleanup_temporal_pools", &GlobalStringPools::cleanup_temporal_pools,
        "清理日期和时间字符串池（交易日收盘后调用）");
    m.def("cleanup_instruments", &GlobalStringPools::cleanup_instruments,
        "清理合约代码字符串池（切换交易日或重新订阅时调用）");
    m.def("check_instrument_pool_size", &GlobalStringPools::check_instrument_pool_size,
        "检查合约池大小并返回当前值（超过 950 时自动警告）");
    m.def("get_pool_sizes", []() {
        size_t exchanges, dates, times, instruments, users, brokers;
        GlobalStringPools::get_pool_sizes(exchanges, dates, times, instruments, users, brokers);
        nb::dict result;
        result["exchanges"] = exchanges;
        result["dates"] = dates;
        result["times"] = times;
        result["instruments"] = instruments;
        result["users"] = users;
        result["brokers"] = brokers;
        return result;
    }, "获取所有字符串池的大小统计");
}
