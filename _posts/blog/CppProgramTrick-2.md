---
layout: post
title: "C++ Programming Tricks--模板"
categories: [blog ]
tags: [C++, 开发]
description: 
---

* content
{:toc}

## 引言

一个方法实现过程中，业务逻辑很多都是相似的，但是与具体的特化类型的不同有一定的差异。
这个时候可以采用特化模板的方式实现，不同的类型使用不同的特化实现。但是这种情况造成一定的业务逻辑的冗余。而trait技术可以将特化类型通过封装，以一个统一的调用方式实现相同的业务逻辑。

## Type_traits技术 

type_traits可以翻译为类型提取器或者类型萃取器，很直白的说就是通过这个机制可以获取被操作数据类型的一些特征。这个机制在编写模板代码的时候特别有用，可以在编译期间就根据数据类型的特征分派给不同的代码进行处理。

清单1: STL中关于copy的代码
```c
// This header file provides a framework for allowing compile time dispatch
// based on type attributes. This is useful when writing template code.
// For example, when making a copy of an array of an unknown type, it helps
// to know if the type has a trivial copy constructor or not, to help decide
// if a memcpy can be used.
struct __true_type {
};
struct __false_type {
};
template <class _Tp>
struct __type_traits { 
   typedef __true_type     this_dummy_member_must_be_first;
                   /* Do not remove this member. It informs a compiler which
                      automatically specializes __type_traits that this
                      __type_traits template is special. It just makes sure that
                      things work if an implementation is using a template
                      called __type_traits for something unrelated. */
   /* The following restrictions should be observed for the sake of
      compilers which automatically produce type specific specializations 
      of this class:
          - You may reorder the members below if you wish
          - You may remove any of the members below if you wish
          - You must not rename members without making the corresponding
            name change in the compiler
          - Members you add will be treated like regular members unless
            you add the appropriate support in the compiler. */
 
   typedef __false_type    has_trivial_default_constructor;
   typedef __false_type    has_trivial_copy_constructor;
   typedef __false_type    has_trivial_assignment_operator;
   typedef __false_type    has_trivial_destructor;
   typedef __false_type    is_POD_type;
};
// The class template __type_traits provides a series of typedefs each of
// which is either __true_type or __false_type. The argument to
// __type_traits can be any type. The typedefs within this template will
// attain their correct values by one of these means:
//     1. The general instantiation contain conservative values which work
//        for all types.
//     2. Specializations may be declared to make distinctions between types.
//     3. Some compilers (such as the Silicon Graphics N32 and N64 compilers)
//        will automatically provide the appropriate specializations for all
//        types.
// EXAMPLE:
//Copy an array of elements which have non-trivial copy constructors
template <class T> void copy(T* source, T* destination, int n, __false_type);
//Copy an array of elements which have trivial copy constructors. Use memcpy.
template <class T> void copy(T* source, T* destination, int n, __true_type);
//Copy an array of any type by using the most efficient copy mechanism
template <class T> inline void copy(T* source,T* destination,int n) {
   copy(source, destination, n,
        typename __type_traits<T>::has_trivial_copy_constructor());
}
```
POD意思是Plain Old Data,也就是标量性别或者传统的C struct型别。POD性别必然拥有trivial ctor/doct/copy/assignment 函数,因此我们就可以对POD型别采用最为有效的复制方法，而对non-POD型别采用最保险安全的方法
```cpp
// uninitialized_copy
// Valid if copy construction is equivalent to assignment, and if the
//  destructor is trivial.
template <class _InputIter, class _ForwardIter>
inline _ForwardIter 
__uninitialized_copy_aux(_InputIter __first, _InputIter __last,
                         _ForwardIter __result,
                         __true_type)
{
  return copy(__first, __last, __result);
}
template <class _InputIter, class _ForwardIter>
_ForwardIter 
__uninitialized_copy_aux(_InputIter __first, _InputIter __last,
                         _ForwardIter __result,
                         __false_type)
{
  _ForwardIter __cur = __result;
  __STL_TRY {
    for ( ; __first != __last; ++__first, ++__cur)
      _Construct(&*__cur, *__first);
    return __cur;
  }
  __STL_UNWIND(_Destroy(__result, __cur));
}
template <class _InputIter, class _ForwardIter, class _Tp>
inline _ForwardIter
__uninitialized_copy(_InputIter __first, _InputIter __last,
                     _ForwardIter __result, _Tp*)
{
  typedef typename __type_traits<_Tp>::is_POD_type _Is_POD;
  return __uninitialized_copy_aux(__first, __last, __result, _Is_POD());
}
```


## trait技术和template 元编程的例子

```cpp
template<template<int> class LOGICAL, class SEQUENCE>
struct sequence_any;

template<template<int> class LOGICAL, int NUM, int...NUMS>
struct sequence_any<LOGICAL, sequence<NUM, NUMS...> >
{
	static const bool value = LOGICAL<NUM>::value || sequence_any<LOGICAL, sequence<NUMS...>>::value;
};

template<template<int> class LOGICAL>
struct sequence_any<LOGICAL, sequence<> >
{
	static const bool value = false;
};
template<int A>
struct static_is_zero
{
	static const bool value = false;
};
template<>
struct static_is_zero<0>
{
	static const bool value = true;
};
 const bool SINGLEROWOPT = 
sequence_any<static_is_zero, sequence<SPECIALIZATIONS...>>::value;
```

## 函数的调用过程
如果一个程序中很多多个同名的函数，那编译器是如何找应该调用哪一个函数呢？
编译器会通过如下顺序进行查找。
1. 函数直接匹配
2. 模板函数
3. 通过一定的隐形转换数据类型可以调用

```cpp
#include <iostream>
void func(float a) {
  std::cout << "float func:" << a << std::endl;
}
void func(int a) {
  std::cout << "int func:" << a << std::endl;
}
template <class T>
void func(T a) {
  std::cout << "template func:" << a << std::endl;
}
int main(int argc, char const *argv[])
{
  int ia = 1;
  func(ia);
  func<int>(ia);
  float fb = 2;
  func(fb);
  func<float>(fb);
  double db = 3;
  func(db);
  func<double>(db);
  return 0;
}
```
>结果输出
int func:1
template func:1
float func:2
template func:2
template func:3
template func:3

## 模板函数的声明与定义一般有两种方式

1. 声明定义在header文件中。这种情况往往是模板针对不同的类型处理方式是一样的，这样可以直接放到头文件中。当实际调用过程中实现template的调用
2. 声明+特化在头文件中，实际定义在cpp文件中。这种情况往往特化几种就是几种。

### 模板invoke模板函数
两个模板函数, 如果**被调用的模板函数的只有声明在头文件中,定义与特化**. 而模板的实际定义在cpp文件中，就会出现undefined的问题.

这是由于在头文件中进行调用模板函数过程中，找不到特化的被调用函数.
在头文件中显示特化声明被调用的函数, 这种情况比较适合针对不同的类型的特化有不同的处理方案.
或者直接将模板函数定义放到头文件中,这种比较适合所有的函数都适用一种情况.



## 纯虚类

#### 定义1

> 含有一个纯虚函数的类，叫做纯虚类。纯虚类不可以定义对象。

我个人觉得这个说法应该就是把纯虚类的主要特点说明了：

> 只要有一个纯虚函数。就称为纯虚类。所以如果子类没有实现纯虚函数，相当子类也有纯虚函数，所以子类也是纯虚类。

其他类的定义与使用方式都与一般的类差不多。大致有如下地方：
* 纯虚类可以有成员变量 （可以)
* 纯虚类可以有普通的成员函数（可以）
* 纯虚类可不可以有其他虚函数（可以）
* 纯虚类可不可以又带有参数的构造函数？ (可以)
* 可不可以在纯虚类的派生类的构造函数中显式调用纯虚类的带参数构造函数(可以)

> 使用方式上：**不可以定义一个对象。**

#### 定义2

> 纯虚类也称为抽象类
**带有纯虚函数的类称为抽象类。**抽象类是一种特殊的类，它是为了抽象和设计的目的而建立的，它处于继承层次结构的较上层（而不是绝对的上层，也有可能是中层，甚至底层？）。抽象类是不能定义对象的，在实际中为了强调一个类是抽象类，可将该类的构造函数（设置为protected)说明为保护的访问控制权限。

抽象类的主要作用是将有关的组织在一个继承层次结构中，由它来为它们提供一个公共的根(其实不一定是根)，相关的子类是从这个根派生出来的。

抽象类刻画了一组子类的操作接口的通用语义，这些语义也传给子类。一般而言，抽象类只描述这组子类共同的操作接口，而完整的实现留给子类。

**抽象类只能作为基类来使用(大多数情况是其他类的基类，但是抽象类本身也有可能是子类），其纯虚函数的实现由派生类给出。**如果派生类没有重新定义纯虚函数，而派生类只是继承基类的纯虚函数，则这个派生类仍然还是一个抽象类。如果派生类中给出了基类纯虚函数的实现，则该派生类就不再是抽象类了，它是一个可以建立对象的具体类了。

## 模板继承接口类

### 安全类型转换

```cpp
template <typename T, typename TFrom>
T safe_cast(TFrom &input) {
	FW_ASSERT(input.type() == blob_elem_trait<typename T::value_type>::type_enum);
	return static_cast<T>(input);
}
```

### `cv::Mat` 与`cv::Mat_<T>`就是典型的案例
1. 采用enum类型或者整数类型进行区分类型
2. 采用`trait`技术将自定义类型与系统类型映射
3. 采用模板继承接口，实现接口的统一调用

## 附录：
- [Trait技术实现迭代器](http://cwlseu.github.io/images/codes/iterator.cpp)
