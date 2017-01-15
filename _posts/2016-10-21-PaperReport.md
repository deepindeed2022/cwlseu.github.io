---
layout: post
title: Paper Report Weekly
categories: [blog ]
tags: [DeepLearning, ]
description: 深度神经网络，知识图谱等内容
--- 

## 知识图谱
![李孜]
[Hinton,Representations]{Hinton. Learning distributed representations of concepts.}
对正样本进行复制，生成错误的测试结合，组成pair训练对
*Pylearn2* 是在thrano基础上的封装，通过配置文件。
### 关系推断：
    随机游走模型，通过对关系矩阵的不断相乘，最后达到收敛

### Node2Vector
使用随机游走方法

## 
YOLO： You only live once

## 从TransE到高斯嵌入表示
将实体与关系映射到一个点向量，忽略了实体与关系的不确定性，如Trans全家桶系列
提出基于密度的嵌入方法，采用非对称的KL三度来给三元组打分，使得关系多种类型建模更加有效
- ImageNet表示成知识库，做成 **图片知识库**。图片detection与理解
- SceneGraph
### 写conclusion的要点
* 核心方法：这个是关键点，有的时候可能这部分可以分为两部分
* 度量方法:
* 测量结果：

[gaussian embedding word representation]{}
将高斯分布嵌入到神经网络中，

开题：3月初左右
开题之前论文阅读数量 > 50篇
* 核心论文要开题报告之后列满1页
* cite{\PaperName}

1. 问题概述
2. 关键技术研究进展
    2.1 网络压缩方面
    * 方法1
    * 方法2
    * 方法3
    2.2 网络训练目标方面
    * min Matrix Error
    * min reconstruct network object
    * using original network output as target
3. 面临的问题和挑战
 没有统一的评价标准和体系，当前主流的是以original network的output与compression network output 进行比较获取，但是要知道，原来的original network也是存在一定的错误标准的，因此更加正确的评价标准是讲original的错误考量进去，而不是简单的进行比较。
4. 研究趋势
   * 完善神经网络压缩的评价体系
   * 更加普适性的网络压缩方法
   * 网络压缩的benchmark是什么? e.g. 对网络简单进行dropout或者聚类之后的效果
   * 压缩比例和正确率之间没有函数关系？

## 2016-11-03 星期四
根据今天对dropout的讨论，发现模型训练过程中的dropout是可以对卷积层进行一定程度的压缩的，而且卷积层距离输入越远，压缩的程度越高。
但是有一个问题：
卷积层的压缩到底是减少了什么？下面是几个猜想:
* 每个核减少了权重数目，如3*3的可能变成了不是9个链接，而是6个权重起作用；
* 核在进行卷积的过程中，减少了对某些卷积核的计算过程。如对一张中间有条小狗的图片，可以减少对背景区域扫描的过程，重点对有小狗的部分进行卷积；
* 虚拟dropout，仅仅是对输出到下一层的时候，隐藏部分结果，但是具体权重还是存在的
是不是可以通过加入训练集合的统计概率，加速训练过程中的dropout


## 逆向思维
现在当前都是做减法
可以这么着，先做减法，大幅度地减法，然后考虑计算的时候采用线性插值的方法进行计算，获得更加精确的分类器。
从其中的线性插值分类器中获取的想法：
1. Hypercolumns for Object Segmentation and Fine-grained Localization




## Reactor模式
我想了一下Reactor模式，要想设置最优的size， 最好是研究一下排队论的问题。从排队论的角度思考或者验证初始size的大小，以及不同初始值对于后面性能的影响等等

### 排队论
这是随机过程中的一个问题。
