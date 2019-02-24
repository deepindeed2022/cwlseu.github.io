---
layout: post
title: Effective C++
categories: [blog ]
tags: [C++, ]
description: C++开发建议50条
---
[TOC]

###  C++是一组语言的集合
C + 面向对象的C + 模板 + STL
Rules for effective C++ programming vary, depending on the part of C++ you are using.
 
### `const` `enums` and `inlines` 比 `#defines`更好
* 预处理器可能将`#define`去掉，导致编译器不知道某些符号
* 宏是全部替换，可能因为多次copy而使用更多的空间
* const指针类型要注意，尤其是`const char*` 要注意：
`const char* const author_name = "charles";`

