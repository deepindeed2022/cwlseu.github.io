---
layout: post
title: "混合精度训练"
categories: [blog ]
tags: [Algorithm]
description: "TF社区中相继出现相关的应用，为了更快的在Pytorch中加入对Volta GPU的支持，并实现针对混合精度训练的优化，NVIDIA发布了Apex开源工具库。cuda9中已经开始支持混合精度训练，tensorRT作为NVIDIA的inference引擎，同样支持混合精度的inference."
---

* content
{:toc}


## 引言

CUDA在推出7.5的时候提出了 可以计算16位浮点数据的新特性。定义了两种新的数据类型half和half2. cuda9中已经开始支持混合精度训练，tensorRT作为NVIDIA的inference引擎，同样支持混合精度的inference. 
之前在网上看到半精度memory copy与计算，发现copy的代价会减少一半，而计算的提升并不是很理想。后来看到了《[why cublasHgemm is slower more than cublasSgemm when I use?](https://devtalk.nvidia.com/default/topic/972337/gpu-accelerated-libraries/why-cublashgemm-is-slower-more-than-cublassgemm-when-i-use-/)》这个帖子，终于发现其中的一点规律。

问题的提出者问，为什么在GTX1070上运行 cublasHgemm（半精度计算） 比 cublasSgemm（单精度计算）计算的慢呢？nv官方的回答说，当前的Pascal架构的GPU只有的 P100 的FP16计算快于 FP32。并且给出了编程手册的吞吐量的表。

## Alibaba PAI: Auto-Mixed Precision Training Techniques
PAI-TAO是alibaba内部一个关于混合精度训练的一个研究项目。在整个AI模型的生命周期中的位置如下：
![@PAI-TAO](http://cwlseu.github.io/images/mixed-precision/PAI-TAO.png)

首先这项任务的是在CUDA9中支持TensorCore之后开展的。
TensorCore brought by Volta architecture
![@tensor core](http://cwlseu.github.io/images/mixed-precision/tensorcore.png)

### 为什么 AMP

#### Mixed-precision的优势

* 充分发挥Volta架构引入的TensorCore计算性能 (15->120TFLOPs, 8X)
* 减少了访存带宽

#### No free-lunch

* 用户模型改写的人力负担
* 精度调优问题
* 充分利用TensorCore的技术tricks
  - 数据尺寸对齐问题
  - Layout问题
* TensorCore将计算密集部分比例降低以后的进一步优化空间挖掘

### 如何AMP：Design Philosophy

* 精度问题
  - 模型以FP32进行保存
  - 不同算子的区别处理
    - 计算密集型算子（ MatMul/Conv）
      输入为FP16，FP32累加中间结果，输出为FP32，计算基于TensorCore
    - 访存密集型算法（ Add/Reduce/…)
      输入输出均为FP16，计算为FP16/FP32, 不使用TensorCore，访存量减少
  - Loss scaling策略解决gradient underflow问题
  - 表达精度问题： FP32->FP16
    * 尾数位减少: precision gap in sum (Solution: 模型以FP32进行保存)
    * 指数位减少: gradient underflow

![@scale在训练过程中的作用](http://cwlseu.github.io/images/mixed-precision/scaling.png)

* 速度及易用性问题
  - 通过图优化pass自动完成混合精度所需的图转换工作

### result

* No laborious FP32/FP16 casting work anymore
* Already supporting diversified internal workloads:
NLP/CNN/Bert/Graph Embedding…
* 1.3~3X time-to-accuracy speed-up


## 参考文献

[1]. [混合精度训练之APEX](https://cloud.tencent.com/developer/news/254121)

[2]. [一种具有混合精度的高度可扩展的深度学习训练系统](http://m.elecfans.com/article/721085.html)

[3]. [百度和NVIDIA联合出品：MIXED PRECISION TRAINING](https://arxiv.org/pdf/1710.03740.pdf)

[4]. [Code for testing the native float16 matrix multiplication performance on Tesla P100 and V100 GPU based on cublasHgemm](https://github.com/hma02/cublasHgemm-P100)

[5]. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions

PAI自动混合精度训练的实现与应用-阿里巴巴+高级算法工程师王梦娣

[英伟达发布全新AI芯片Jetson Xavier](http://m.elecfans.com/article/640489.html)
