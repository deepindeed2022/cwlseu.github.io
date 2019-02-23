---
layout: post
title: 奇怪的问题
categories: [blog ]
tags: [C++, android]
description: 工程
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>


* 关于字符串赋值导致crash的问题
{:toc}

## 问题描述
在**android**系统中，测试一个C++动态库时，出现segment fault或者Invalid Address free
等问题，最终定位crash的位置时在一个**字符串赋值**的位置。例如：
```cpp
class O {
public:
	void allfunction() = 0;
};

class AO : public O {
public:
	void otherFunc() = 0;
private:
	void setName(const std::string& name){
		LOG("Before setName");
		this->_name = name;
		LOG("Finished setName");
	}
	string _name;
};

class BAO: public AO {
public:
	void otherFunc() override;
private:
	type _value;
};

class CBAO: public BAO {
public:
	void otherFunc() override;
	void otherFunc2();
private:
	type2 other_value;
}

AO* cbao = new CBAO;
cbao->setName("a name");

```

其中输出的信息中含有"Before setName"的信息，但是没有"Finished setName"。
然后如果将`this->_name = name`这行代码注释，那么这个问题就可以不出现。此外，采用如下的时候
```cpp
AO* bao = new BAO;
bao->setName("a name");
```
竟然没有问题。好苦恼呀~~

## 探索1：
	
>是不是因为`std::string`采用引用的方式，导致内存释放的时候出现多次释放。
	
这是不应该的，因为我们采用的是标准库，而且string会自己管理内存的，不应该导致这种问题的。而且我们通过对`setName`中`_name`赋值常量字符串，该问题仍然存在。

## 探索2：
	
> 如果不采用set函数方式进行赋值，而是将`_name`变为`public`类型，从而直接对对象成员进行赋值。

这样操作并没有什么实质性的变化，仍然存在存在这种现象，也就是说与对象的存储方式是与成员的可见性是没有关系的。

## 探索3：
> 采用`char _name[256]`代替`std::string _name` ，我们自己负责内存的管理，而不是由C++管理内存分配。

虽然这个问题绕过去了，但是后面仍然出现了类似的问题，因为我们的对象中还有其他string类型的成员变量。如果全部改用C类型的数组进行替换，代价太大。

## 探索4：
> 采用静态库进行测试，问题没有问题

为什么动态库就有问题，静态库没有问题呢？ ~~因为静态库中是直接将所有需要的文件都包含到静态库中。而动态库中拥有的是所有**需要库的链接**~~我们的动态库中是将所有需要的符号，库等等都编译进行去了的。而我们的软件库依赖其他软件库。


## 未完待续





##附录： Android 对C++库的支持

| 名称|	说明 | 	功能  |
|:----------------:|:-------------------:|:---------------:|
| libstdc++（默认）	|默认最小系统 C++ 运行时库。| 不适用 |
| gabi++_static		| GAbi++ 运行时（静态）。	| C++ 异常和 RTTI|
| gabi++_shared		| GAbi++ 运行时（共享）。	| C++ 异常和 RTTI|
| stlport_static	| STLport 运行时（静态）。	| C++ 异常和 RTTI；标准库|
| stlport_shared	| STLport 运行时（共享）。	| C++ 异常和 RTTI；标准库|
| gnustl_static		|GNU STL（静态）。		| C++ 异常和 RTTI；标准库|
| gnustl_shared		|GNU STL（共享）。		| C++ 异常和 RTTI；标准库|
| c++_static 	| LLVM libc++ 运行时（静态）。	| C++ 异常和 RTTI；标准库|
| c++_shared 	| LLVM libc++ 运行时（共享）。	| C++ 异常和 RTTI；标准库|

https://developer.android.google.cn/ndk/guides/cpp-support


### 兼容性
NDK 的 libc++ 不稳定。并非所有测试都能通过，而且测试套件并不全面。一些已知的问题包括：

如果在 ARM 上使用 c++_shared，引发异常时可能会崩溃。
对`wchar_t`和语言区域 API 的支持受到限制。

### C++ 异常

在高于 NDKr5 的所有 NDK 版本中，NDK 工具链可让您使用支持异常处理的 C++ 运行时。 但为确保与早期版本兼容，默认情况下它会编译所有支持 -fno-exceptions 的 C++ 来源。 您可以为整个应用或个别模块启用 C++ 异常。

要为整个应用启用异常处理支持，请将以下行添加到 Application.mk 文件中。要为个别模块启用异常处理支持，请将以下行添加到其各自的 Android.mk 文件中。

`APP_CPPFLAGS += -fexceptions`

### RTTI
在高于 NDKr5 的所有 NDK 版本中，NDK 工具链可让您使用支持 RTTI 的 C++ 运行时。 但为确保与早期版本兼容，默认情况下它会编译所有支持 `-fno-rtti` 的 C++ 来源。

要为整个应用启用 RTTI 支持，请将以下行添加到 `Application.mk`文件中：
`APP_CPPFLAGS += -frtti`
要为个别模块启用 RTTI 支持，请将以下行添加到其各自的 Android.mk 文件中：
`LOCAL_CPP_FEATURES += rtti`
或者，您也可以使用：
`LOCAL_CPPFLAGS += -frtti`
### 静态运行时
将 C++ 运行时的静态库变体添加到多个二进制文件可能导致意外行为。 例如，您可能会遇到：

内存在一个库中分配，在另一个库中释放，从而导致内存泄漏或堆损坏。
libfoo.so 中引发的异常在 libbar.so 中未被捕获，从而导致您的应用崩溃。
`std::cout` 的缓冲未正常运行
此外，如果您将两个共享库 – 或者一个共享库和一个可执行文件 – 链接到同一个静态运行时，每个共享库的最终二进制映像包含运行时代码的副本。 运行时代码有多个实例是表明有问题，因为运行时内部使用或提供的某些全局变量会重复。

此问题不适用于只包含一个共享库的项目。例如，您可以链接 stlport_static，并预期您的应用正确运行。 如果您的项目需要多个共享库模块，建议使用 C++ 运行时的共享库变体。

### 共享运行时
如果您的应用针对早于 Android 4.3（Android API 级别 18）的 Android 版本，并且您使用指定 C++ 运行时的共享库变体，则必须先加载共享库，再加载依赖它的任何其他库。

例如，应用可能具有以下模块:
- libfoo.so
- libfoo.so 使用的 libbar.so
- libfoo 和 libbar 使用的 libstlport_shared.so
必须以相反的依赖顺序加载库：
	
	static {
      System.loadLibrary("stlport_shared");
      System.loadLibrary("bar");
      System.loadLibrary("foo");
    }

注：调用`System.loadLibrary()`时不要使用 lib 前缀。