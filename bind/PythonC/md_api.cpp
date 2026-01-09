/**
 * PcCTP - CTP PC版行情API Python绑定 - API方法实现
 *
 * 本文件包含 MdApi 的所有 Python 方法实现和模块初始化代码
 * 使用 common.h 中的宏来简化重复代码
 */

// NumPy C API - 使用 NO_IMPORT_ARRAY 来引用由 util.cpp 定义的符号
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PcCTP_CAPI_ARRAY_API
#include <numpy/ndarrayobject.h>

#include "md.h"
#include "trade.h"  // 添加 TradeApi 支持

// 声明 FIX 子模块初始化函数（来自 fix.cpp）
extern "C" int initialize_fix_submodule(PyObject* parent_module);

// =============================================================================
// MdApi 方法实现 (使用下划线命名法 + 内联优化)
// =============================================================================

/**
 * @brief 创建MdApi实例 (静态方法)
 * Python命名: create_ftdc_md_api
 */
static PyObject* MdApi_create_ftdc_md_api(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* flow_path = "";
    int is_using_udp = 0;
    int is_multicast = 0;

    static char* kwlist[] = {"flow_path", "is_using_udp", "is_multicast", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sii", kwlist, &flow_path, &is_using_udp, &is_multicast)) {
        return NULL;
    }

    PyTypeObject* type = (PyTypeObject*)self;
    MdApiObject* obj = (MdApiObject*)type->tp_alloc(type, 0);
    if (!obj) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate MdApi object");
        return NULL;
    }

    obj->api = CThostFtdcMdApi::CreateFtdcMdApi(flow_path, is_using_udp != 0, is_multicast != 0);
    obj->spi = nullptr;
    obj->py_spi = nullptr;

    if (!obj->api) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create CTP MdApi");
        return NULL;
    }

    return (PyObject*)obj;
}

// 使用 common.h 中的宏定义基础方法
// 直接传递类名，不使用中间宏（避免宏展开问题）

DEF_API_GET_VERSION(MdApi, CThostFtdcMdApi)
DEF_API_INIT(MdApi, MdApiObject)
DEF_API_JOIN(MdApi, MdApiObject)
DEF_API_GET_TRADING_DAY(MdApi, MdApiObject)
DEF_API_REGISTER_FRONT(MdApi, MdApiObject)
DEF_API_REGISTER_NAME_SERVER(MdApi, MdApiObject)
DEF_API_REGISTER_SPI(MdApi, MdApiObject, PyMdSpi)
DEF_API_RELEASE(MdApi, MdApiObject, PyMdSpi)
DEF_API_DEALLOC(MdApiObject, MdApi, PyMdSpi)
DEF_API_NEW_INIT(MdApi, MdApiObject)

// =============================================================================
// 其他 MdApi 特有方法
// =============================================================================

/**
 * @brief 注册用户信息（零拷贝优化）
 * Python命名: register_fens_user_info
 *
 * 零拷贝实现：
 * - 优先使用 _capsule（零拷贝路径）
 * - 如果 _capsule 不存在，降级到 _struct + memcpy（兼容路径）
 */
static PyObject* MdApi_register_fens_user_info(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* fens_user_info_obj;
    if (!PyArg_ParseTuple(args, "O", &fens_user_info_obj)) {
        return NULL;
    }

    CThostFtdcFensUserInfoField* fens_user_info_ptr = nullptr;

    // 优先使用 _capsule（零拷贝路径）
    PyObject* capsule = PyObject_GetAttrString(fens_user_info_obj, "_capsule");
    if (capsule && PyCapsule_CheckExact(capsule)) {
        fens_user_info_ptr = static_cast<CThostFtdcFensUserInfoField*>(
            PyCapsule_GetPointer(capsule, "FensUserInfo"));
        Py_DECREF(capsule);
    }
    // 降级到 _struct（兼容路径，有拷贝）
    else {
        Py_XDECREF(capsule);
        PyObject* struct_obj = PyObject_GetAttrString(fens_user_info_obj, "_struct");
        if (struct_obj) {
            PyObject* buffer = PyObject_CallMethod(struct_obj, "__bytes__", NULL);
            Py_DECREF(struct_obj);
            if (buffer) {
                static CThostFtdcFensUserInfoField temp_struct;
                char* buffer_ptr;
                Py_ssize_t buffer_len;
                if (PyBytes_AsStringAndSize(buffer, &buffer_ptr, &buffer_len) == 0 &&
                    static_cast<size_t>(buffer_len) >= sizeof(CThostFtdcFensUserInfoField)) {
                    memcpy(&temp_struct, buffer_ptr, sizeof(CThostFtdcFensUserInfoField));
                    fens_user_info_ptr = &temp_struct;
                }
                Py_DECREF(buffer);
            }
        }
    }

    if (!fens_user_info_ptr) {
        PyErr_SetString(PyExc_TypeError,
            "fens_user_info must be a FensUserInfo object with _capsule or _struct attribute");
        return NULL;
    }

    obj->api->RegisterFensUserInfo(fens_user_info_ptr);
    Py_RETURN_NONE;
}

/**
 * @brief 订阅行情（支持 NumPy 零拷贝）
 * Python命名: subscribe_market_data
 */
static PyObject* MdApi_subscribe_market_data(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // 使用通用辅助方法提取ID
    std::vector<char*> ids;
    std::vector<std::string> strs;
    int count = extract_instrument_ids(instrument_ids, ids, strs);
    if (count < 0) {
        return NULL;  // 错误信息已在函数中设置
    }

    int result = obj->api->SubscribeMarketData(ids.data(), count);
    return PyLong_FromLong(result);
}

/**
 * @brief 取消订阅行情（支持 NumPy 零拷贝）
 * Python命名: un_subscribe_market_data
 */
static PyObject* MdApi_un_subscribe_market_data(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // 使用通用辅助方法提取ID
    std::vector<char*> ids;
    std::vector<std::string> strs;
    int count = extract_instrument_ids(instrument_ids, ids, strs);
    if (count < 0) {
        return NULL;  // 错误信息已在函数中设置
    }

    int result = obj->api->UnSubscribeMarketData(ids.data(), count);
    return PyLong_FromLong(result);
}

/**
 * @brief 订阅询价（支持 NumPy 零拷贝）
 * Python命名: subscribe_for_quote_rsp
 */
static PyObject* MdApi_subscribe_for_quote_rsp(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // 使用通用辅助方法提取ID
    std::vector<char*> ids;
    std::vector<std::string> strs;
    int count = extract_instrument_ids(instrument_ids, ids, strs);
    if (count < 0) {
        return NULL;  // 错误信息已在函数中设置
    }

    int result = obj->api->SubscribeForQuoteRsp(ids.data(), count);
    return PyLong_FromLong(result);
}

/**
 * @brief 取消订阅询价（支持 NumPy 零拷贝）
 * Python命名: un_subscribe_for_quote_rsp
 */
static PyObject* MdApi_un_subscribe_for_quote_rsp(PyObject* self, PyObject* args) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* instrument_ids;
    if (!PyArg_ParseTuple(args, "O", &instrument_ids)) {
        return NULL;
    }

    // 使用通用辅助方法提取ID
    std::vector<char*> ids;
    std::vector<std::string> strs;
    int count = extract_instrument_ids(instrument_ids, ids, strs);
    if (count < 0) {
        return NULL;  // 错误信息已在函数中设置
    }

    int result = obj->api->UnSubscribeForQuoteRsp(ids.data(), count);
    return PyLong_FromLong(result);
}

/**
 * @brief 查询组播合约请求（零拷贝优化）
 * Python命名: req_qry_multicast_instrument
 *
 * 零拷贝实现：
 * - 优先使用 _capsule（零拷贝路径）
 * - 如果 _capsule 不存在，降级到 _struct + memcpy（兼容路径）
 */
static PyObject* MdApi_req_qry_multicast_instrument(PyObject* self, PyObject* args, PyObject* kwargs) {
    MdApiObject* obj = (MdApiObject*)self;
    if (!obj->api) {
        PyErr_SetString(PyExc_RuntimeError, "MdApi not initialized");
        return NULL;
    }

    PyObject* req_obj;
    int request_id;

    static char* kwlist[] = {"qry_multicast_instrument", "request_id", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &req_obj, &request_id)) {
        return NULL;
    }

    CThostFtdcQryMulticastInstrumentField* req_ptr = nullptr;

    // 优先使用 _capsule（零拷贝路径）
    PyObject* capsule = PyObject_GetAttrString(req_obj, "_capsule");
    if (capsule && PyCapsule_CheckExact(capsule)) {
        req_ptr = static_cast<CThostFtdcQryMulticastInstrumentField*>(
            PyCapsule_GetPointer(capsule, "QryMulticastInstrument"));
        Py_DECREF(capsule);
    }
    // 降级到 _struct（兼容路径，有拷贝）
    else {
        Py_XDECREF(capsule);
        PyObject* struct_obj = PyObject_GetAttrString(req_obj, "_struct");
        if (struct_obj) {
            PyObject* buffer = PyObject_CallMethod(struct_obj, "__bytes__", NULL);
            Py_DECREF(struct_obj);
            if (buffer) {
                static CThostFtdcQryMulticastInstrumentField temp_struct;
                char* buffer_ptr;
                Py_ssize_t buffer_len;
                if (PyBytes_AsStringAndSize(buffer, &buffer_ptr, &buffer_len) == 0 &&
                    static_cast<size_t>(buffer_len) >= sizeof(CThostFtdcQryMulticastInstrumentField)) {
                    memcpy(&temp_struct, buffer_ptr, sizeof(CThostFtdcQryMulticastInstrumentField));
                    req_ptr = &temp_struct;
                }
                Py_DECREF(buffer);
            }
        }
    }

    if (!req_ptr) {
        PyErr_SetString(PyExc_TypeError,
            "qry_multicast_instrument must be a QryMulticastInstrument object with _capsule or _struct attribute");
        return NULL;
    }

    int result = obj->api->ReqQryMulticastInstrument(req_ptr, request_id);
    return PyLong_FromLong(result);
}

// 使用 DEF_REQ_METHOD_CAPSULE 宏定义请求方法
// 直接传递类名，不使用中间宏（避免宏展开问题）
DEF_REQ_METHOD_CAPSULE(MdApi, req_user_login, ReqUserLogin, CThostFtdcReqUserLoginField)

DEF_REQ_METHOD_CAPSULE(MdApi, req_user_logout, ReqUserLogout, CThostFtdcUserLogoutField)

// =============================================================================
// Python类型定义
// =============================================================================

static PyMethodDef MdApi_methods[] = {
    {"create_ftdc_md_api", (PyCFunction)MdApi_create_ftdc_md_api, METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     "Create CTP MdApi instance"},
    {"get_api_version", MdApi_get_api_version, METH_NOARGS | METH_STATIC,
     "Get CTP API version"},
    {"init", MdApi_init, METH_NOARGS, "Initialize MdApi"},
    {"join", MdApi_join, METH_NOARGS, "Join thread"},
    {"get_trading_day", MdApi_get_trading_day, METH_NOARGS, "Get trading day"},
    {"register_front", MdApi_register_front, METH_VARARGS, "Register front address"},
    {"register_name_server", MdApi_register_name_server, METH_VARARGS, "Register name server"},
    {"register_spi", MdApi_register_spi, METH_VARARGS, "Register SPI callback"},
    {"register_fens_user_info", MdApi_register_fens_user_info, METH_VARARGS, "Register fens user info"},
    {"subscribe_market_data", MdApi_subscribe_market_data, METH_VARARGS, "Subscribe market data"},
    {"un_subscribe_market_data", MdApi_un_subscribe_market_data, METH_VARARGS, "Unsubscribe market data"},
    {"subscribe_for_quote_rsp", MdApi_subscribe_for_quote_rsp, METH_VARARGS, "Subscribe for quote response"},
    {"un_subscribe_for_quote_rsp", MdApi_un_subscribe_for_quote_rsp, METH_VARARGS, "Unsubscribe for quote response"},
    {"req_user_login", (PyCFunction)MdApi_req_user_login, METH_VARARGS | METH_KEYWORDS, "Request user login"},
    {"req_user_logout", (PyCFunction)MdApi_req_user_logout, METH_VARARGS | METH_KEYWORDS, "Request user logout"},
    {"req_qry_multicast_instrument", (PyCFunction)MdApi_req_qry_multicast_instrument, METH_VARARGS | METH_KEYWORDS, "Request query multicast instrument"},
    {"release", MdApi_release, METH_NOARGS, "Release resources"},
    {NULL}
};

PyTypeObject MdApiType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "PcCTP.MdApi",                          /* tp_name */
    sizeof(MdApiObject),                    /* tp_basicsize */
    0,                                      /* tp_itemsize */
    (destructor)MdApi_dealloc,              /* tp_dealloc */
    0,                                      /* tp_print */
    0,                                      /* tp_getattr */
    0,                                      /* tp_setattr */
    0,                                      /* tp_reserved */
    0,                                      /* tp_repr */
    0,                                      /* tp_as_number */
    0,                                      /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash */
    0,                                      /* tp_call */
    0,                                      /* tp_str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "CTP PC Market Data API (Optimized with Force Inline)",  /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    MdApi_methods,                          /* tp_methods */
    0,                                      /* tp_members */
    0,                                      /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    0,                                      /* tp_descr_get */
    0,                                      /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)MdApi_tp_init,                /* tp_init */
    0,                                      /* tp_alloc */
    MdApi_new,                              /* tp_new */
};

// =============================================================================
// 模块级方法
// =============================================================================

static PyMethodDef pcctp_module_methods[] = {
    PCCTP_POOL_METHODS(),
    PCCTP_POOL_CLEANUP_METHODS(),  // 新增：字符串池清理函数
    ADD_PYCAPSULE_METHODS()  // 从 common.h 添加 PyCapsule 辅助函数
    {NULL}
};

static PyModuleDef PcCTP_module = {
    PyModuleDef_HEAD_INIT,
    "PcCTP",
    "CTP PC Market Data API Bindings (Pure Python C API + Force Inline Optimization)",
    -1,
    pcctp_module_methods,
};

PyMODINIT_FUNC PyInit_PcCTP(void) {
    if (initialize_util() < 0) {
        return NULL;
    }

    PyObject* module = PyModule_Create(&PcCTP_module);
    if (!module) {
        return NULL;
    }

    // 准备并添加 MdApi 类型
    if (PyType_Ready(&MdApiType) < 0) {
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&MdApiType);
    if (PyModule_AddObject(module, "MdApi", (PyObject*)&MdApiType) < 0) {
        Py_DECREF(&MdApiType);
        Py_DECREF(module);
        return NULL;
    }

    // 准备并添加 TradeApi 类型（从 trade_api.cpp）
    if (PyType_Ready(&TradeApiType) < 0) {
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&TradeApiType);
    if (PyModule_AddObject(module, "TradeApi", (PyObject*)&TradeApiType) < 0) {
        Py_DECREF(&TradeApiType);
        Py_DECREF(module);
        return NULL;
    }

    // 初始化 FIX 子模块
    if (initialize_fix_submodule(module) < 0) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
