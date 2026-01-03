/**
 * @file demo.cpp
 * @brief Pybind数据传递性能测试 - Python绑定代码
 *
 * 本文件使用pybind11将C++函数暴露给Python：
 * 1. process_list: 基于Python List的方式（涉及拷贝）
 * 2. process_numpy: 基于NumPy数组的方式（零拷贝）
 *
 * 模块名称：demo_pybind
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // for std::vector conversion
#include <pybind11/numpy.h> // for numpy array support
#include "../src/demo.h"

// 使用pybind11命名空间简化代码
namespace py = pybind11;

/**
 * @brief Python模块绑定配置
 *
 * PYBIND11_MODULE宏定义Python模块的入口点
 * 第一个参数：模块名称（demo_pybind）
 * 第二个参数：模块对象（py::module_类型）
 */
PYBIND11_MODULE(demo_pybind, m) {
    // 模块文档字符串（Python help()会显示）
    m.doc() = R"doc(
        Python与C++数据传递性能测试模块（无算法处理版本）

        本模块提供两种数据传递方式用于性能对比：
        1. process_list: 基于Python List（涉及拷贝）
        2. process_numpy: 基于NumPy数组（零拷贝）

        注意: 所有函数内部都不进行任何算法处理，直接返回输入数据
              用于测试纯粹的数据传递性能差异
    )doc";

    // ============================================================
    // 绑定 process_list 函数
    // ============================================================
    m.def(
        "process_list",                      // Python函数名
        &process_list,                       // C++函数指针
        py::arg("input"),                    // 参数名（用于Python帮助信息）
        R"doc(
        基于Python List的数据传递（涉及拷贝）

        注意: 函数内部不做任何处理，直接返回输入数据

        参数:
            input: Python List[float] 或其他可迭代对象

        返回:
            Python List[float]: 与输入相同的数据

        性能特点:
            - 涉及2次内存分配（Python→C++, C++→Python）
            - 涉及2次数据拷贝（每次80KB，对于10000个元素）
            - 适用于测试拷贝传递的开销

        示例:
            >>> import demo_pybind
            >>> demo_pybind.process_list([1.0, 2.0, 3.0])
            [1.0, 2.0, 3.0]
        )doc"
    );

    // ============================================================
    // 绑定 process_numpy 函数
    // ============================================================
    m.def(
        "process_numpy",                     // Python函数名
        &process_numpy,                      // C++函数指针
        py::arg("input"),                    // 参数名（用于Python帮助信息）
        R"doc(
        基于NumPy数组的数据传递（零拷贝）

        注意: 函数内部不做任何处理，直接返回输入数据

        参数:
            input: numpy.ndarray[float64] 一维数组

        返回:
            numpy.ndarray[float64]: 与输入相同的数据（共享内存）

        性能特点:
            - 0次内存分配
            - 0次数据拷贝
            - 直接共享NumPy数组内存
            - 适用于测试零拷贝传递的性能

        示例:
            >>> import demo_pybind
            >>> import numpy as np
            >>> demo_pybind.process_numpy(np.array([1.0, 2.0, 3.0]))
            array([1., 2., 3.])
        )doc"
    );

    // ============================================================
    // 模块属性
    // ============================================================
    m.attr("__version__") = "2.0.0";           // 模块版本
    m.attr("__author__") = "Pybind Demo Team"; // 作者信息

    // 暴露一些常量用于测试（可选）
    py::module_ m_constants = m.def_submodule("constants", "测试常量");
    m_constants.attr("DATA_SIZE") = 10000;     // 默认数据规模
    m_constants.attr("NUM_ITERATIONS") = 1000; // 默认循环次数
    m_constants.attr("WARMUP_RUNS") = 10;      // 默认预热次数
}
