---
layout: post
title: "开发笔记：Eigen开源库的应用"
categories: [blog]
tags: [工具]
description: Eigen开源库的入门
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/> 


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