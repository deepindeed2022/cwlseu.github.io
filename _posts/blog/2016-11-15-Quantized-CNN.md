---
layout: post
title: Quantized Convolutional Neural Networks for Mobile Devices
categories: [blog ]
tags: [Math, ]
description: "CNN在很多计算机视觉任务中取得了显著的成果，然而高性能硬件对于CNN模型来说不可或缺。因为CNN模型具有计算复杂繁琐，使得其拓展成为困难。"
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## 论文作者信息

[Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng](National Laboratory of Patter Recognition Institute of Automation, Chinese Academy of Sciences)

## 简介
目标是减少模型大小，同时提高计算速度。通过对卷积层的filter kernal和全连接层的权重矩阵同时进行量化，最小化每一层的错误估值进行训练。

## 问题
之前的工作很少有能够同时实现整个网络的显著的加速和压缩。

## 主要贡献
1. 加速和压缩于一体的神经网络
2. 提出了有效的训练方式来减少训练过程中的累积残差
3. 实现4~6倍的加速，同时15~20倍的压缩比。