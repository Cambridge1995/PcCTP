/**
 * PcCTP - CTP PC版行情API Python绑定 - SPI回调实现
 *
 * 本文件包含 PyMdSpi 类的特殊回调实现
 * 大部分回调已在 md.h 中使用宏定义，此处只实现需要特殊处理的回调
 */

// NumPy C API - 使用 NO_IMPORT_ARRAY 来引用由 util.cpp 定义的符号
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PcCTP_CAPI_ARRAY_API
#include <numpy/ndarrayobject.h>

#include "md.h"

// =============================================================================
// SPI回调类实现 (使用下划线命名法)
// =============================================================================

// 虚函数不需要内联，因为必须通过 vtable 调度

// ------------------------------------------------------------------------
// 订阅相关回调 (需要使用字符串池，不能使用通用宏)
// ------------------------------------------------------------------------

void PyMdSpi::OnRspSubMarketData(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                CThostFtdcRspInfoField* pRspInfo,
                                int nRequestID, bool bIsLast) {
    if (!m_api || !m_api->py_spi) return;
    PyGILStateKeeper gil;

    // 使用字符串池返回 str（零拷贝）
    PyObject* py_instrument_id = Py_None;
    Py_INCREF(Py_None);
    if (pSpecificInstrument) {
        py_instrument_id = GlobalStringPools::Instruments.intern(
            pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
    }

    PyObject* py_rsp_info = CREATE_CAPSULE_OR_NONE_OPT(pRspInfo, "RspInfo");

    PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_sub_market_data", "OOii",
        py_instrument_id, py_rsp_info, nRequestID, bIsLast);

    Py_XDECREF(py_instrument_id);
    Py_XDECREF(py_rsp_info);
    if (!result) PyErr_Print();
    Py_XDECREF(result);
}

void PyMdSpi::OnRspUnSubMarketData(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                  CThostFtdcRspInfoField* pRspInfo,
                                  int nRequestID, bool bIsLast) {
    if (!m_api || !m_api->py_spi) return;
    PyGILStateKeeper gil;

    // 使用字符串池返回 str（零拷贝）
    PyObject* py_instrument_id = Py_None;
    Py_INCREF(Py_None);
    if (pSpecificInstrument) {
        py_instrument_id = GlobalStringPools::Instruments.intern(
            pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
    }

    PyObject* py_rsp_info = CREATE_CAPSULE_OR_NONE_OPT(pRspInfo, "RspInfo");

    PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_un_sub_market_data", "OOii",
        py_instrument_id, py_rsp_info, nRequestID, bIsLast);

    Py_XDECREF(py_instrument_id);
    Py_XDECREF(py_rsp_info);
    if (!result) PyErr_Print();
    Py_XDECREF(result);
}

// ------------------------------------------------------------------------
// 询价相关回调 (需要使用字符串池，不能使用通用宏)
// ------------------------------------------------------------------------

void PyMdSpi::OnRspSubForQuoteRsp(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                 CThostFtdcRspInfoField* pRspInfo,
                                 int nRequestID, bool bIsLast) {
    if (!m_api || !m_api->py_spi) return;
    PyGILStateKeeper gil;

    // 使用字符串池返回 str（零拷贝）
    PyObject* py_instrument_id = Py_None;
    Py_INCREF(Py_None);
    if (pSpecificInstrument) {
        py_instrument_id = GlobalStringPools::Instruments.intern(
            pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
    }

    PyObject* py_rsp_info = CREATE_CAPSULE_OR_NONE_OPT(pRspInfo, "RspInfo");

    PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_sub_for_quote_rsp", "OOii",
        py_instrument_id, py_rsp_info, nRequestID, bIsLast);

    Py_XDECREF(py_instrument_id);
    Py_XDECREF(py_rsp_info);
    if (!result) PyErr_Print();
    Py_XDECREF(result);
}

void PyMdSpi::OnRspUnSubForQuoteRsp(CThostFtdcSpecificInstrumentField* pSpecificInstrument,
                                   CThostFtdcRspInfoField* pRspInfo,
                                   int nRequestID, bool bIsLast) {
    if (!m_api || !m_api->py_spi) return;
    PyGILStateKeeper gil;

    // 使用字符串池返回 str（零拷贝）
    PyObject* py_instrument_id = Py_None;
    Py_INCREF(Py_None);
    if (pSpecificInstrument) {
        py_instrument_id = GlobalStringPools::Instruments.intern(
            pSpecificInstrument->InstrumentID, sizeof(pSpecificInstrument->InstrumentID));
    }

    PyObject* py_rsp_info = CREATE_CAPSULE_OR_NONE_OPT(pRspInfo, "RspInfo");

    PyObject* result = PyObject_CallMethod(m_api->py_spi, "on_rsp_un_sub_for_quote_rsp", "OOii",
        py_instrument_id, py_rsp_info, nRequestID, bIsLast);

    Py_XDECREF(py_instrument_id);
    Py_XDECREF(py_rsp_info);
    if (!result) PyErr_Print();
    Py_XDECREF(result);
}
