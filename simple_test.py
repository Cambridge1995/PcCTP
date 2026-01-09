#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PcCTP 简单测试脚本 (Capsule 版本)

Capsule 版本特点：
- 零拷贝数据传输（C++ 直接访问 Python 内存）
- 懒加载字段（首次访问时解码）
- 自动缓存（后续访问从缓存读取）
- snake_case 命名（Python 风格）
"""
import time
import os
from typing import Optional

import numpy as np

# 导入 PcCTP 模块
from PcCTP import (
    reason_map, MdApi, RspUserLogin, RspInfo, UserLogout,
    ForQuoteRsp, DepthMarketData, MulticastInstrument, ReqUserLogin, PyMdSpi
)


class MdSpi(PyMdSpi):
    """
    自定义行情回调类（Capsule 版本）

    版本特点：
    - 继承 PyMdSpi 协议获得完整类型提示
    - 回调参数是 CapsuleStruct 对象（不是 dict）
    - 使用属性访问：obj.field_name（不是 obj.get('field_name')）
    - 所有回调方法使用下划线命名 (snake_case)
    """


    def on_front_connected(self) -> None:
        print("\n[SPI回调] ✓ 连接成功")

    def on_front_disconnected(self, reason: int) -> None:
        reason_msg = reason_map.get(reason, f"未知原因(0x{reason:04X})")
        print(f"\n[SPI回调] ✗ 连接断开，原因: {reason_msg}")

    def on_heart_beat_warning(self, time_lapse: int) -> None:
        print(f"\n[心跳警告] ⚠ 距离上次接收报文的时间: {time_lapse}秒")

    def on_rsp_user_login(
            self,
            rsp_user_login: RspUserLogin,
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        print(f"\n[登录响应] RequestID={request_id}, IsLast={is_last}")

        if rsp_info is not None:
            error_id = rsp_info.error_id
            error_msg = rsp_info.error_msg
            print(f"  ErrorID={error_id}, ErrorMsg={error_msg}")

            if error_id == 0 and rsp_user_login is not None:
                print("  ✓ 登录成功!")
                print(f"  交易日: {rsp_user_login.trading_day}")
                print(f"  登录时间: {rsp_user_login.login_time}")
                print(f"  经纪商: {rsp_user_login.broker_id}")
                print(f"  用户代码: {rsp_user_login.user_id}")
                print(f"  FrontID: {rsp_user_login.front_id}")
                print(f"  SessionID: {rsp_user_login.session_id}")
                print(f"  最大报文引用: {rsp_user_login.max_order_ref}")
        else:
            print(f"  ✗ 登录失败")

    def on_rsp_user_logout(
            self,
            user_logout: UserLogout,
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        """
            登出请求，对应响应on_rsp_user_logout。
        """
        print(f"\n[登出响应] RequestID={request_id}, IsLast={is_last}")
        if rsp_info is not None:
            error_id = rsp_info.error_id
            error_msg = rsp_info.error_msg
            if error_id == 0:
                print(f"  ✓ 登出成功")
            else:
                print(f"  ✗ 登出失败: ErrorID={error_id}, ErrorMsg={error_msg}")

    def on_rsp_error(
            self,
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        if rsp_info is not None:
            error_id = rsp_info.error_id
            error_msg = rsp_info.error_msg
            print(f"\n[错误应答] RequestID={request_id}, ErrorID={error_id}, ErrorMsg={error_msg}")

    def on_rsp_sub_market_data(
            self,
            instrument_id: str,
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        if instrument_id is not None and instrument_id != 'None':
            print(f"[订阅应答] 合约: {instrument_id}")
        if rsp_info is not None and rsp_info.error_id != 0:
            print(f"  ErrorID={rsp_info.error_id}, ErrorMsg={rsp_info.error_msg}")

    def on_rsp_un_sub_market_data(
            self,
            instrument_id: str,
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        if instrument_id is not None and instrument_id != 'None':
            print(f"[取消订阅应答] 合约: {instrument_id}")

    def on_rsp_sub_for_quote_rsp(
            self,
            instrument_id: str,
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        if instrument_id is not None and instrument_id != 'None':
            print(f"[订阅询价应答] 合约: {instrument_id}")

    def on_rsp_un_sub_for_quote_rsp(
            self,
            instrument_id: str,
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        if instrument_id is not None and instrument_id != 'None':
            print(f"[取消订阅询价应答] 合约: {instrument_id}")

    def on_rtn_depth_market_data(
            self,
            depth_market_data: DepthMarketData
    ) -> None:
        if depth_market_data is not None:
            instrument_id = depth_market_data.instrument_id
            last_price = depth_market_data.last_price
            volume = depth_market_data.volume
            update_time = depth_market_data.update_time
            bid_price1 = depth_market_data.bid_price1
            ask_price1 = depth_market_data.ask_price1
            print(f"[深度行情] {instrument_id} | {update_time} | "
                  f"最新价={last_price:.2f} | 成交量={volume} | "
                  f"买一价={bid_price1:.2f} | 卖一价={ask_price1:.2f}")

    def on_rtn_for_quote_rsp(
            self,
            for_quote_rsp: ForQuoteRsp
    ) -> None:
        print(f"[询价通知] {for_quote_rsp.to_dict()}")

    def on_rsp_qry_multicast_instrument(
            self,
            multicast_instrument: Optional[MulticastInstrument],
            rsp_info: RspInfo,
            request_id: int,
            is_last: bool
    ) -> None:
        print(f"[查询组播合约] RequestID={request_id}, IsLast={is_last}")


def main():
    """主测试函数"""
    print("=" * 70)
    print("PcCTP Capsule 版本测试")
    print("=" * 70)

    # 创建 API 实例
    print("\n[步骤1] 创建 MdApi 实例")
    api = MdApi.create_ftdc_md_api('./flows/')
    print(f"  API 版本: {MdApi.get_api_version()}")
    print("  ✓ MdApi 创建成功")

    # 创建 Python SPI 对象
    print("\n[步骤2] 创建 SPI 回调对象")
    spi = MdSpi()
    print("  ✓ SPI 对象创建成功")

    # 注册 SPI
    print("\n[步骤3] 注册 SPI 回调")
    api.register_spi(spi)
    print("  ✓ SPI 已注册")

    # 注册前置地址
    print("\n[步骤4] 注册前置地址")
    api.register_front("tcp://182.254.243.31:40011")
    print("  ✓ 前置地址已注册: tcp://182.254.243.31:40011")

    # 初始化（启动连接）
    print("\n[步骤5] 初始化 API（启动连接）")
    api.init()
    print("  正在连接服务器...")

    # 等待连接
    time.sleep(2)

    # 发起登录请求（Capsule 版本：创建对象并设置属性）
    print("\n[步骤6] 发起登录请求")
    req = ReqUserLogin()  # 创建空对象
    req.broker_id = "9999"
    req.user_id = "251795"
    req.password = "wjq!15074971011"
    req.user_product_info = ""
    result = api.req_user_login(req, 1)
    print(f"  登录请求已发送，返回值: {result}")

    # 等待登录响应
    print("  等待登录响应...")
    time.sleep(3)

    # 订阅行情 (支持 numpy array 或 Python list)
    print("\n[步骤7] 订阅行情")
    instrument_ids = np.array(['au2602', 'jm2605', 'ag2612'], dtype='S31')
    result = api.subscribe_market_data(instrument_ids)
    print(f"  订阅请求已发送，返回值: {result}")

    # 保持运行，接收行情
    print("\n[步骤8] 接收行情数据...")
    print("  " + "-" * 60)
    print("  接收中... (按 Ctrl+C 退出)")
    print("  " + "-" * 60)

    try:
        # 接收 10 秒行情数据
        time.sleep(10)
        api.un_subscribe_market_data(['ag2612'])
        time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n[中断] 用户请求退出")

    # ⚠️ 最终方案：不进行任何清理，让进程直接退出
    # CTP 资源由操作系统在进程退出时清理
    # 避免所有手动清理导致的竞态条件
    print("\n[完成] 测试结束，进程即将退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        os._exit(1)  # 异常时也使用 os._exit()
