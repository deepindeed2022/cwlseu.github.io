---
layout: post
title: Effective C++
categories: [blog ]
tags: [C++, ]
description: C++开发建议50条
---

* content
{:toc}

## 引言
读《Effective C++》中的笔记

##  C++是一组语言的集合
C + 面向对象的C + 模板 + STL
Rules for effective C++ programming vary, depending on the part of C++ you are using.
 
## 条款1：`const` `enums` and `inlines` 比 `#defines`更好
* 预处理器可能将`#define`去掉，导致编译器不知道某些符号
* 宏是全部替换，可能因为多次copy而使用更多的空间
* const指针类型要注意，尤其是`const char*` 要注意：
`const char* const author_name = "charles";`

1. 为方便调试，最好使用常量
注意：常量定义一般放在头文件中，可将指针和指针所指的类型都定义成const，如const char * const authorName = Scott Meyers;
类中常量通常定义为静态成员， 而且需要先声明后定义可以在声明时或定义时赋值，也可使用借用enum的方法如enum{Num = 5};
2. `#define`语句造成的问题
如`#define max(a, b) ((a) > (b) ? (a) : (b))`
在下面情况下：
Int a= 5, b = 0;
max(++ a, b);
max(++ a, b + 10);
max内部发生些什么取决于它比较的是什么值解决方法是使用inline函数，可以使用template来产生一个函数集。
