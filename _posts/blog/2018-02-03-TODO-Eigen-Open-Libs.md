---
layout: post
title: "开发笔记：Eigen开源库的应用"
categories: [blog]
tags: [tools]
description: Eigen开源库的入门
---

{:toc} 


## Eigen与C之间数据转换

### to Eigen
```cpp
float* raw_data = malloc(...);
Map<MatrixXd> M(raw_data, rows, cols);
// use M as a MatrixXd
M = M.inverse();
```

### from Eigen
```cpp
MatrixXd M;
float* raw_data = M.data();
int stride = M.outerStride();
raw_data[i+j*stride]
```
## 一些预备知识

### template programming

### 4 levels 并行

- cluster of PCs --MPI
- multi/many-cores -- OpenMP
- SIMD -- intrinsics for vector instructions
- pipelining -- needs non dependent instructions

### Peak Performance

> Example： Intel Core2 Quad CPU Q9400 @ 2.66GHz (x86_64)

    * pipelining → 1 mul + 1 add / cycle (ideal case)
    * SSE → x 4 single precision ops at once
    * frequency → x 2.66G
    * peak performance: 21,790 Mflops (for 1 core)

这就是我们优化的目标


## Fused operation: Expression Template
Expression Template是个好东西。通过编译融合嵌入的方式，减少了大量的读写和计算。

## Curiously Recurring Template Pattern


## [Eigen常见的坑](https://zhuanlan.zhihu.com/p/32226967)

### 编译的时候最好-DEIGEN_MPL2_ONLY(详见: Eigen)
这是因为Eigen虽然大部分是MPL2 licensed的，但是还有少部分代码是LGPL licensed的，如果修改了其代码，则必须开源。
这可能产生法律风险，而遭到法务部门的Complain

### 要注意alignment的问题
```
my_program: path/to/eigen/Eigen/src/Core/DenseStorage.h:44:
Eigen::internal::matrix_array<T, Size, MatrixOptions, Align>::internal::matrix_array()
[with T = double, int Size = 2, int MatrixOptions = 2, bool Align = true]:
Assertion `(reinterpret_cast<size_t>(array) & (sizemask)) == 0 && "this assertion
is explained here: http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
     READ THIS WEB PAGE !!! ****"' failed.
There are 4 known causes for this issue. Please read on to understand them and learn how to fix them.
```

例如下面的代码都是有问题的，可能导致整个程序Crash。

* 结构体含有Eigen类型的成员变量

```cpp
class Foo {
  //...
  Eigen::Vector2d v; // 这个类型只是一个例子，很多类型都有问题
  //...
};
//...
Foo *foo = new Foo;
```

* Eigen类型的变量被放到STL容器中

```cpp
// Eigen::Matrix2f这个类型只是一个例子，很多类型都有问题
std::vector<Eigen::Matrix2f> my_vector;
struct my_class { ... Eigen::Matrix2f m; ... }; 
std::map<int, my_class> my_map;
```

* Eigen类型的变量被按值传入一个函数

```cpp
// Eigen::Vector4d只是一个例子，很多类型都有这个问题
void func(Eigen::Vector4d v);
```

* 在栈上定义Eigen类型的变量(只有GCC4.4及以下版本 on Windows被发现有这个问题，例如MinGW or TDM-GCC)

```cpp
void foo() {
  Eigen::Quaternionf q; // Eigen::Quaternionf只是一个例子，很多类型都有这个问题
}
```

- [Explanation of the assertion on unaligned arrays](http://eigen.tuxfamily.org/dox/group__TopicUnalignedArrayAssert.html)