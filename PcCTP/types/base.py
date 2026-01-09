"""
PcCTP Capsule 版本数据类型模块 - 基础类

核心特性：
- 零拷贝：C++ 端只传递指针
- 懒加载：首次访问时才 decode
- 自动缓存：后续访问从缓存读取（~10 ns）
- snake_case：属性使用 snake_case 命名
- 自动编码转换：GBK 字段自动转换为 UTF-8
- 统一基类：所有结构体类继承 CapsuleStruct
"""

import ctypes
from typing import Any, Dict, Type, Optional
# 获取操作系统类型
from PcCTP.types.utils import env

# 通过系统类型导入对应的接口
if env == 'win64':  # windows64位
    from PcCTP.win64 import pycapsule_new, pycapsule_check_exact, pycapsule_get_pointer
elif env == 'win32':  # windows32位
    from PcCTP.win32 import pycapsule_new, pycapsule_check_exact, pycapsule_get_pointer
elif env == 'linux':  # linux
    from PcCTP.linux import pycapsule_new, pycapsule_check_exact, pycapsule_get_pointer
else:  # 其他系统 暂不支持,如：macOS
    raise EnvironmentError('本CTP版本与当前系统不匹配')



# =============================================================================
# Capsule 辅助函数
# =============================================================================

def _capsule_to_ptr(capsule: Any, name: str) -> int:
    """
    从 Capsule 获取 C 指针

    Args:
        capsule: Python Capsule 对象
        name: Capsule 名称（如 "DepthMarketData"）

    Returns:
        C 结构体指针地址（整数）
    """
    if not pycapsule_check_exact(capsule):
        raise TypeError(f"Expected Capsule, got {type(capsule)}")

        # 失败时自动抛出异常，无需手动检查
    return pycapsule_get_pointer(capsule, name)


# =============================================================================
# CapsuleStruct - 统一基类
# =============================================================================

class CapsuleStruct:
    """
    Capsule 结构体基类

    所有 CTP 结构体类的统一基类，支持两种用法：
    1. 请求类：创建空对象，通过 setter 设置值
    2. 响应类：C++ 传入 capsule，直接读取

    使用示例：
        # 请求类用法
        order = InputOrder()
        order.instrument_id = "au2506"
        order.volume = 1

        # 响应类用法
        def on_rsp(self, rsp: RspInfo):
            print(rsp.error_msg)  # 直接读取
    """

    # 子类必须定义的类属性
    _Struct: Optional[Type[ctypes.Structure]] = None  # C 结构体类型
    _capsule_name: Optional[str] = None  # Capsule 名称（如 "DepthMarketData"）
    _field_mappings: Dict[str, str] = {}  # snake_case -> PascalCase 字段映射

    def __init__(self, capsule: Any = None):
        """
        创建对象

        Args:
            capsule: 可选，C++ 传入的 Capsule 对象
        """
        if self._Struct is None:
            raise NotImplementedError(f"{self.__class__.__name__}._Struct 未定义")
        if self._capsule_name is None:
            raise NotImplementedError(f"{self.__class__.__name__}._capsule_name 未定义")

        self._cache: Dict[str, Any] = {}

        if capsule is None:
            # 创建空对象（请求类）
            self._struct = self._Struct()
            self._capsule = None
        else:
            # 从 Capsule 创建（响应类）- 零拷贝：.contents 只是代理对象，不复制结构体
            self._capsule = capsule
            ptr = _capsule_to_ptr(capsule, self._capsule_name)
            self._struct = ctypes.cast(ptr, ctypes.POINTER(self._Struct)).contents

    def clear_cache(self) -> None:
        """清除懒加载缓存"""
        self._cache.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（用于调试和兼容）

        注意：
        - 会访问所有字段，触发所有懒加载
        - 性能较低，不推荐在高频场景使用
        """
        result = {}
        # 遍历所有属性（通过 property 定义的字段）
        for attr_name in dir(self):
            # 跳过私有属性和方法
            if attr_name.startswith('_'):
                continue
            # 跳过非 property 的方法
            attr = getattr(self.__class__, attr_name, None)
            if not isinstance(attr, property):
                continue
            # 获取属性值
            try:
                result[attr_name] = getattr(self, attr_name)
            except (AttributeError, TypeError, ValueError, KeyError):
                # 忽略无法访问的属性
                pass
        return result

    @classmethod
    def from_capsule(cls, capsule: Any) -> "CapsuleStruct":
        """
        从 Capsule 创建对象

        Args:
            capsule: C++ 传入的 Capsule 对象

        Returns:
            CapsuleStruct 对象

        注意：
            - 这是从 C++ 回调接收数据时的标准方法
            - 转换失败时会抛出异常，调用方应捕获并处理
        """
        return cls(capsule)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapsuleStruct":
        """
        从字典创建对象（用于测试或兼容性）

        Args:
            data: 字典数据

        Returns:
            CapsuleStruct 对象

        注意：
        - 此方法会创建新的 C 结构体实例
        - 需要 C 扩展支持才能创建 Capsule
        - 主要用于测试，不推荐在生产环境使用
        """
        # 检查 _Struct 是否已定义
        if cls._Struct is None:
            raise NotImplementedError(f"{cls.__name__}._Struct 未定义")

        # 创建结构体实例
        struct = cls._Struct()

        # 定义字段类型映射
        field_types = {}
        for field_name, field_type in cls._Struct._fields_:
            field_types[field_name] = field_type

        # 填充结构体
        for snake_key, value in data.items():
            # 查找对应的 C 结构体字段名
            pascal_key = cls._field_mappings.get(snake_key)
            if pascal_key is None:
                # 尝试直接使用 snake_key
                pascal_key = snake_key

            if pascal_key not in field_types:
                continue  # 字段不存在，跳过

            field_type = field_types[pascal_key]

            # 根据字段类型设置值
            if isinstance(field_type, type) and issubclass(field_type, ctypes.Array):
                # 字符数组字段
                if hasattr(value, 'encode'):
                    # 字符串 -> 字节（from_dict 是 Python 端创建，使用 UTF-8）
                    byte_value = value.encode('utf-8')
                    setattr(struct, pascal_key, byte_value[:ctypes.sizeof(field_type)].ljust(ctypes.sizeof(field_type), b'\x00'))
                else:
                    setattr(struct, pascal_key, value)
            elif field_type == ctypes.c_double:
                setattr(struct, pascal_key, float(value))
            elif field_type == ctypes.c_int:
                setattr(struct, pascal_key, int(value))
            elif field_type == ctypes.c_char:
                if isinstance(value, str):
                    setattr(struct, pascal_key, value.encode('ascii')[0])
                else:
                    setattr(struct, pascal_key, value)
            else:
                setattr(struct, pascal_key, value)

        # 创建 Capsule（需要 C 扩展支持）
        # 使用 addressof 获取结构体的内存地址，传递给 C 端
        capsule = pycapsule_new(ctypes.addressof(struct), cls._capsule_name, None)
        return cls(capsule)

    def to_capsule(self) -> Any:
        """
        转换为 Capsule（用于 C API 调用）

        零拷贝实现：复用 _struct 的内存地址，无需复制数据

        Returns:
            Python Capsule 对象
        """
        if self._capsule is None:
            # 延迟创建 Capsule，直接使用 _struct 的内存地址（零拷贝）
            # 这样 C++ 端可以直接访问 Python 内存中的数据
            self._capsule = pycapsule_new(
                ctypes.addressof(self._struct),  # _struct 的内存地址
                self._capsule_name,              # Capsule 名称
                None                              # 无析构函数
            )
        return self._capsule

    def __repr__(self) -> str:
        """字符串表示"""
        fields = []
        for k, v in self.to_dict().items():
            fields.append(f"{k}={v!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"
