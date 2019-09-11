---
layout: post
title: 余额宝
tags: [计算机视觉, CV算法] 
categories: [blog ]
notebook: 视觉算法
---

* content
{:toc}


## ResNet & ResNeXt

- 论文：Aggregated Residual Transformations for Deep Neural Networks
- 论文链接：https://arxiv.org/abs/1611.05431
- PyTorch代码：https://github.com/miraclewkf/ResNeXt-PyTorch

作者的核心创新点就在于提出了 aggregrated transformations，用一种平行堆叠相同拓扑结构的blocks代替原来 ResNet 的三层卷积的block，在不明显增加参数量级的情况下提升了模型的准确率，同时由于拓扑结构相同，超参数也减少了，便于模型移植。

- https://blog.csdn.net/hejin_some/article/details/80743818
- https://www.cnblogs.com/bonelee/p/9031639.html