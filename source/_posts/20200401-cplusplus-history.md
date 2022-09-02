---
layout: post
title: The history of C++
categories: [blog]
tags: [C++]
description: C++的发展历史
date: 2020-04-01 21:05:18
---

## 前言

在阅读C++相关的技术书籍或博客时，常常会提到一些日常开发中不常接触的名词，如cfront 2.0或者TR1等，这些名词在C++的历史发展中属于里程碑式的的名词。从C++不同时期的发展中可以看出对于程序员的开发需求逐渐满足，伴随着C++的标准的变化，编译器对语言的支持也逐渐完善。

## C++ 历史大事件

![wg21-timeline-2019-07](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030335629.png)

## 关键事件总结

| date | feature  | details | sample |
|:---:|---|----|----|
|  1979 | 首次实现引入类的C|C with Classes first implemented<br>1. 新特性：**类、成员函数、继承类**、独立编译、**公共和私有访问控制、友元、函数参数类型检查、默认参数、内联函数、赋值符号重载、构造函数、析构函数**、f()相当于f(void)、调用函数和返回函数（同步机制，不是在C++中）</br> 2. 库：并发任务程序库（不是在C++中）|
|  1985 | 编译器cfront 1.0 | 1. 新特性：**虚函数、函数和操作符重载**、**引用**、**new和delete操作符**、**const关键词**、范围解析运算符::<br>2. 新加入的库：复数（complex）、字符串（string）、输入输出流（iostream） |
|  1985 | 《C++编程语言第一版》 |The C++ Programming Language, 1st edition | 
|  1989 | 编译器cfront 2.0 | 1.新特性：多重继承、成员指针、保护访问控制、类型安全联接、抽象类、静态和常量成员函数、特定类的new和delete <br> 2. 新库：I/O 操作器 <br>|
| 1990 | ANSI C++委员会成立（ANSI C++ Committee founded） ||
| 1990 | 《C++参考手册注解》|The Annotated C++ Reference Manual was released. |
| 1991 | ISO C++委员会成立（ISO C++ Committee founded）||
| 1998 | C++98 | 1. 新特性：**运行时类型信息［RTTI（dynamic_cast, typeid）］**、协变返回类型（covariant return types）、cast 操作符、可变型、布尔型、声明情况、模板例示、成员模板、导出 <br> 2. 新库：容器、算法、迭代器、函数对象（STL中）、区域设置、位集合、值向量、自动指针（auto_ptr）、模块化字符串、输入输出流和复数<br> the C++ standards committee published the first international standard for C++ ISO/IEC 14882:1998, which would be informally known as C++98. The Annotated C++ Reference Manual was said to be a large influence in the development of the standard. **The Standard Template Library**, which began its conceptual development in 1979, was also included.|\*\*\*\*\*||
|1999|Boost由委员会成员成立，旨在开发新的高质量库以作为标准库的候选库|Boost founded by the committee members to produce new high-quality candidate libraries for the standard|
|2003|C++03 (ISO/IEC 14882:2003)|The committee responded to multiple problems that were reported with their 1998 standard, and revised it accordingly. The changed language was dubbed **C++03**. 这是一个次要修订版本，修正了一些错误。|
|2006| Performance TR (ISO/IEC TR 18015:2006) (ISO Store ) (2006 draft )|性能技术报告|
|2007 | 2007 Library extension TR1 (ISO/IEC TR 19768:2007) (ISO store ) (2005 draft ) |1. 源自Boost：**引用包装器（Reference wrapper）**、**智能指针（Smart pointers）**、成员函数（Member function）、Result of 、绑定（Binding）、函数（Function）、类型特征（type traits）、随机（Random）、数学特殊函数（Mathematical Special Functions）、元组（Tuple）、数组（Array）、无序容器［Unordered Containers包括哈希（Hash）］还有**正则表达式（Regular Expressions）** <br> 2. 源自C99：math.h中同时也是新加入C99的数学函数、空白字符类、浮点环境（Floating-point environment）、十六进制浮点I/O操作符（hexfloat I/O Manipulator）、固定大小整数类型（fixed-size integral types）、长整型（the long long type）、va_copy、snprintf() 和vscanf()函数族，还有C99 的printf()与scanf()函数族的指定转换。 TR1除了一些特殊函数，大部分都被囊括进C++11。|\*\*\*\*\*|
|2010| 数学特殊函数技术报告［2010 Mathematical special functions TR (ISO/IEC 29124:2010)(ISO Store)］|此TR是一个C++标准库扩展，加入了TR1中的部分特殊函数，但那些函数之前没有被包括进C++11：椭圆积分、指数积分、拉盖尔多项式（Laguerre polynomials）、勒让徳多项式（Legendre polynomials）、艾尔米特多项式（Hermite polynomials）、贝塞尔（Bessel）函数、纽曼（Newmann）函数、$\beta$函数和黎曼（Riemann）$\zeta$函数 |
|2011 | C++11 (ISO/IEC 14882:2011) (ISO Store) (ANSI Store ) |1. 新语言特性：**自动（auto）和类型获取（decltype）**、默认和已删除函数（defaulted and deleted functions）、**不可更改（final）和重载（override）**、**拖尾返回类型（trailing return type）**、**右值引用（rvalue references）**、**移动构造函数（move constructors）/移动赋值（move assignment）**、作用域枚举（scoped enums）、常量表达式（constexpr）和文字类型（literal types）、**列表初始化（list initialization）**、授权（delegating）和**继承构造器（inherited constructors）**、大括号或等号（brace-or-equal）初始化器、**空指针（nullptr）**、长整型（long long）、char16_t和char32_t、类型别名（type aliases）、**可变参数模板（variadic templates）**、广义联合体（generalized unions）、广义POD、Unicode字符串文字（Unicode string literals）、自定义文字（user-defined literals）、属性（attributes）、**$\lambda$表达式（lambda expressions）**、无异常（noexcept）、对齐查询（alignof）和对齐指定（alignas）、**多线程内存模型（multithreaded memory model）、线程本地存储（thread-local storage）**、**GC接口（GC interface）**、range for(based on a Boost library)、静态断言［static assertions（based on a Boost library）］<br> 2.新库特性：原子操作库（atomic operations library）、**emplace()**和贯穿整个现有库的右值引用的使用、std::initializer_list、状态性的和作用域内的分配器（stateful and scoped allocators）、前向列表（forward_list）、**计时库（chrono library）**、分数库（ratio library）、新算法（new algorithms）、Unicode conversion facets <br>3.源自TR1：除了特殊的函数，TR1中全部都被囊括进来 <br>4.源自Boost：线程库（The thread library）、异常指针（exception_ptr）、错误码（error_code）和错误情况（error_condition）、迭代器改进［iterator improvements（std::begin, std::end, std::next, std::prev）］<br>5.源自C：C风格的Unicode转换函数<br>6.搜集错误报告修复：363个错误在2008草案中被解决，另外有322个错误接着被修复。其中的错误包括530号，它使得std::basic_string对象相连。|\*\*\*\*\*|
|2011 | 十进制浮点技术报告［Decimal floating-point TR (ISO/IEC TR 24733:2011) (ISO Store ) (2009 draft )］| 这个TR根据IEEE 754-2008浮点算数标准（Floating Point Arithmetic）：std::decimal::decimal32、std::decimal::decimal64、std::decimal::decimal128 |
|2012 | 标准C++基金会成立|The Standard C++ Foundation founded|
|2013 | 《C++编程语言第四版》 | The C++ Programming Language, 4th edition||
| 2014 | C++14 (2014 final draft ) | 1. 新语言特性：变量模板（variable templates）、多态lambda（polymorphic lambdas）、λ动捕获（move capture for lambdas）、**new/delete elision**、常量表达式函数放宽限制（relax restrictions on constexpr functions）、二值文本（binary literals）、数字分隔符（digit separators）、函数返回类型推演（return type deduction for functions）、用大括号或等号初始符集合初始化类<br> 2. 新库特性：std::make_unique、std::shared_mutex和std::shared_lock、std::index_sequence、std::exchange、std::quoted，还有许多针对现有库的小改进，比如一些算法的双距离重载（two-range overloads for some algorithms）、类型特征的类型别名版本（type alias versions of type traits）、用户定义字符串（user-defined string）、持续期（duration）和复杂数字文本（complex number literals）等等<br> 3.搜集错误报告修复：149号库（149 library issues） 基础库技术规范（Library fundamentals TS）, 文件系统技术规范（Filesystem TS）和其他专业技术规范（ experimental technical specifications）|

## 不同编译器对c++标准的支持
cfront x.x就是Bjarne Stroustrup的第一个C++编译器，将C++转换成C语言。在1993年，cfront 4.0因为尝试支持异常机制失败而被取消。我们开发者最长打交道的工具就是编译器了。我们只要通过编写程序语言，编译器会翻译成具体的更底层命令来控制计算机去实现我们的需要的功能。但C++语言标准是一个庞大的特性集合，而不同编译器厂商在根据这个统一标准做编译器的过程中，由于各种原因，不可能支持全部的标准中列举出来的特性。
例如，C++11已经流行多年，很多特性是随着编译器版本release才逐渐支持的，如下图：

![@](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030335369.jpg)


* [关于不同编译器对C++不同时期的语言特性的支持程度](https://en.cppreference.com/w/cpp/compiler_support)

* [gnu gcc对C++语言特定的支持情况以及最低支持版本等信息](https://gcc.gnu.org/projects/cxx-status.html)

## 参考资料


* [gnu gcc常见问题](https://gcc.gnu.org/onlinedocs/libstdc++/faq.html)
* [C++官方的history页面](http://www.cplusplus.com/info/history/)
* [中文博客C++的历史与现状](https://www.cnblogs.com/fickleness/p/3154937.html)
* [Feature-Test Macros and Policies](https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations)
* [各编译器下载地址，包括vs2017社区版](https://isocpp.org/get-started)