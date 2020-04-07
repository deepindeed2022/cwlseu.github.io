---
layout: post
title: "C++ Object Model"
categories: [blog]
tags: [C++]
description: "如果不了解编译器对我们的C++代码做了什么手脚，很多困惑永远都解不开。"
---

## 前言
面向对象的三大特征是抽象、继承、多态。《深度探索C++对象模型》一书中从数据的排布，C++对象函数的调用设计等等。
我尝试以一个编译器的设计者的角度去理解C++对象，该书中也提到了多种编译器，有的时候也会涉及一些不同编译器厂商在设计过程中的不同，虽然没有深入探究不同的原因以及优劣对比，
但对于我这个新手来说已经开了很大的窗户。

整个书籍通过横向切割方式，分别从构造、数据成员、成员函数、运行时C++对象的特点来介绍，从缔造者的视角来理解C++对象的设计，有利于我们写出更加高效、简洁的程序。

## 关于对象
### C++对象比C struct对象在空间与时间的有额外负担吗？
 封装的后布局成本与C struct是一样的。member functions虽然旱灾class的声明之内，却不出现在的object中。每一个non-inline memberfunction 只会诞生一个函数实例。
   C++在布局以及存取时间上的主要的额外负担是由virtual引起的，包括：
* virtual function机制，引入vptr以及vtbl，支持一个有效率的"执行期绑定"
* virtual base class，用以实现"多次出现在继承体系中的base class，有一个单一而被共享的实例"
* 多重继承下，派生类跟第二个以及后续基类之间的转换

### C++对象模型

在C++中，有两种数据成员（class data members）：static 和nonstatic,以及三种类成员函数（class member functions）:static、nonstatic和virtual:
```cpp
class Base {
public:
    Base(int i) :baseI(i){};    
    int getI(){ return baseI; } 
    static void countI(){};   //static
    virtual void print(void){ cout << "Base::print()"; } // virtual
    virtual ~Base(){}         // virtual
private:
    int baseI;  // no static 
    static int baseS;  // static
};
```

在此模型下，nonstatic 数据成员被置于每一个类对象中，而static数据成员被置于类对象之外。static与nonstatic函数也都放在类对象之外，而对于virtual 函数，则通过虚函数表+虚指针来支持，具体如下：
- 每个类生成一个表格，称为虚表（virtual table，简称vtbl）。虚表中存放着一堆指针，这些指针指向该类每一个虚函数。虚表中的函数地址将按声明时的顺序排列，不过当子类有多个重载函数时例外，后面会讨论。
- 每个类对象都拥有一个虚表指针(vptr)，由编译器为其生成。虚表指针的设定与重置皆由类的复制控制（也即是构造函数、析构函数、赋值操作符）来完成。vptr的位置为编译器决定，传统上它被放在所有显示声明的成员之后，不过现在许多编译器把vptr放在一个类对象的最前端。关于数据成员布局的内容，在后面会详细分析。
- 另外，虚函数表的前面设置了一个指向type_info的指针，用以支持RTTI（Run Time Type Identification，运行时类型识别）。RTTI是为多态而生成的信息，包括对象继承关系，对象本身的描述等，只有具有虚函数的对象在会生成。

![@vs2015下对象的内存结构](https://cwlseu.github.io/images/gcc/cppobjmodel_1.png)

这个模型的优点在于它的空间和存取时间的效率；缺点如下：如果应用程序本身未改变，但当所使用的类的non static数据成员添加删除或修改时，需要重新编译。

> Note: 针对析构函数，g++中的实现有一些令人疑惑的地方，~Base在虚表中出现了两次，我表示不能理解，网上也没有找到相关说明。
> 
    Vtable for Base
    Base::_ZTV4Base: 6u entries
    0     (int (*)(...))0
    4     (int (*)(...))(& _ZTI4Base)
    8     (int (*)(...))Base::print
    12    (int (*)(...))Base::~Base
    16    (int (*)(...))Base::~Base

我猜测可能是我们使用g++编译中合成根据我添加的~Base()合成了一个用于动态内存分配释放的析构函数和静态释放的析构函数。当然如果有大佬知道这个是为什么，请务必指导一番，不胜感激。

### 多重继承

```cpp
#include<iostream>
using namespace std;
class Base1
{
public:
	virtual ~Base1() {};
	virtual void speakClearly() {cout<<"Base1::speakClearly()"<<endl;}
	virtual Base1 *clone() const {cout<<"Base1::clone()"<<endl; return new Base1;}
protected:
	float data_Base1;
};
class Base2
{
public:
	virtual ~Base2() {};
	virtual void mumble() {cout<<"Base2::mumble()"<<endl;}
	virtual Base2 *clone() const {cout<<"Base2::clone()"<<endl; return new Base2;}
protected:
	float data_Base2;
};
class Derived : public Base1,public Base2
{
public:
	virtual ~Derived()  {cout<<"Derived::~Derived()"<<endl;}
	virtual Derived *clone() const {cout<<"Derived::clone()"<<endl; return new Derived;}
protected:
	float data_Derived;
}
```

![@逻辑上的图](https://cwlseu.github.io/images/gcc/multi-inherited.png)

类似问题在vs2010中也有，[主要是多重继承的时，将派生类赋值给第二个基类时](https://blog.csdn.net/Microsues/article/details/6452249?depth_1-utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1&utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1)

```
1>  Derived::$vftable@Base1@:
1>   | &Derived_meta
1>   |  0
1>   0 | &Derived::{dtor}
1>   1 | &Base1::speakClearly
1>   2 | &Derived::clone
1>  
1>  Derived::$vftable@Base2@:
1>   | -8
1>   0 | &thunk: this-=8; goto Derived::{dtor}
1>   1 | &Base2::mumble
1>   2 | &thunk: this-=8; goto Base2* Derived::clone
1>   3 | &thunk: this-=8; goto Derived* Derived::clone
```

* 派生类的虚函数表数目是它所有基类的虚函数数目之和，基类的虚函数表被复制到派生类的对应的虚函数表中。

* 派生类中重写基类的虚拟函数时，该被重写的函数在派生类的虚函数列表中得到更新，派生类的虚析构函数覆盖基类的虚析构函数。

* 派生类中新增加的虚函数被添加到与第一个基类相对应的虚函数表中。

* virtual table[1]中的clone分别为：`Base2* Derived::clone` 和 Derived* Derived::clone 。这里为什么会比table[0]多一个`Base2* Derived::clone`呢？
因为：如果将一个Derived对象地址指定给一个Base1指针或者Derived指针是，虚拟机制使用的是virtual table[0] ；如果将一个Derived对象地址指定给一个Base2指针时，虚拟机制使用的是virtual table[1]。 （<<C++对象模型>> P164)


<!-- 1. "指针的类型"会教导编译器如何解释某个特定地址中的内存内容以及其大小（void*指针只能够持有一个地址，而不能通过它操作所指向的object）
2. C++通过class的pointers和references来支持多态，付出的代价就是额外的间接性。它们之所以支持多态是因为它们并不引发内存中任何"与类型有关的内存委托操作(type-dependent commitment)"，会受到改变的，只有他们所指向的内存的"大小和内容的解释方式"而已。
 -->

## 构造函数

![](https://cwlseu.github.io/images/gcc/ctor.png)

## 参考链接

[MSVC应对多重继承中的thunk技术](https://docs.microsoft.com/zh-cn/archive/blogs/zhanli/c-tips-adjustor-thunk-what-is-it-why-and-how-it-works)
[C++对象模型详解](https://www.cnblogs.com/tgycoder/p/5426628.html)
[图说C++对象模型：对象内存布局详解](https://www.cnblogs.com/QG-whz/p/4909359.html)
[RTTI实现详解](https://blog.csdn.net/heyuhang112/article/details/41982929)