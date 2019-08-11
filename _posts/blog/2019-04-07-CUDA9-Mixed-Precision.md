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
从中可以看出，自动混合精度主要是在训练过程中，为了加快计算节点之间的数据交换和层之间的数据交换与计算，采用FP16来替换FP32，这样在计算结果精度几乎不损失的情况下，带了数据交换和计算速度方面的性能提升，从而加快模型训练速度。

而这项任务的成功，与CUDA9中支持TensorCore的特性是息息相关的。下面对TensorCode进行简单介绍。 
![@tensor core](http://cwlseu.github.io/images/mixed-precision/tensorcore.png)
TensorCore是NVIDIA在Volta architecture下引入的，专门针对计算4x4矩阵的计算模块。
以前NVIDIA的GPU中只有FP32和FP64计算单元，在TensorCore中，特别针对FP16做了相应的补充，
来补充在半精度浮点方面的不足。TensorCore相比较直接进行FP32的计算，速度有了很大的提升。

### 为什么采用AMP（Auto mixed-precision）

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
    - 计算密集型算子（MatMul/Conv）
      输入为FP16，FP32累加中间结果，输出为FP32，计算基于TensorCore
    - 访存密集型算法（Add/Reduce/…)
      输入输出均为FP16，计算为FP16/FP32, 不使用TensorCore，访存量减少
  - Loss scaling策略解决gradient underflow问题
  - 表达精度问题： FP32->FP16
    * 尾数位减少: precision gap in sum (Solution: 模型以FP32进行保存)
    * 指数位减少: gradient underflow

![@scale在训练过程中的作用](http://cwlseu.github.io/images/mixed-precision/scaling.png)

* 速度及易用性问题
  - 通过图优化pass自动完成混合精度所需的图转换工作

### 结果

* No laborious FP32/FP16 casting work anymore
* Already supporting diversified internal workloads:
  NLP/CNN/Bert/Graph Embedding
* 1.3~3X time-to-accuracy speed-up
  与PAI-TAO	Compiler联合使用可以达到1+1>2的加速收益

## 题外思考

现在我们的训练应该是没有引入混合精度训练的，而且inference框架中没有混合精度的苗头。
我们的inference应该可以先支持起混合精度的，然后后面慢慢地在训练框架中添加相关功能。
然后重构节点之间的数据交换代码，加大对混合精度训练的时候并行度，进一步降低训练模型的成本。
尤其是弱计算能力的芯片上，通过添加混合计算功能，能够在加速的同时，追求更高的精度。
现在很多AI推理芯片如华为himix200中，支持int8和int16的计算，而且同一个模型可以混合int8和int16的精度类型。

## 参考文献

[1]. [混合精度训练之APEX](https://cloud.tencent.com/developer/news/254121)

[2]. [一种具有混合精度的高度可扩展的深度学习训练系统](http://m.elecfans.com/article/721085.html)

[3]. [百度和NVIDIA联合出品：MIXED PRECISION TRAINING](https://arxiv.org/pdf/1710.03740.pdf)

[4]. [Code for testing the native float16 matrix multiplication performance on Tesla P100 and V100 GPU based on cublasHgemm](https://github.com/hma02/cublasHgemm-P100)

[5]. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions

[6]. [Training-Mixed-Precision-User-Guide](https://docs.nvidia.com/deeplearning/sdk/pdf/Training-Mixed-Precision-User-Guide.pdf)

[7]. [英伟达发布全新AI芯片Jetson Xavier](http://m.elecfans.com/article/640489.html)
<!-- PAI自动混合精度训练的实现与应用-阿里巴巴+高级算法工程师王梦娣 -->

