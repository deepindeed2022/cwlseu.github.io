---
layout: post
title: "高通芯片笔记"
categories: [blog ]
tags: [android]
description: 最近几年，随着移动互联，物联网的发展，高通赚得盆满钵满。本文主要总结日常工作汇总接触到与高通相关的一些东西。
comments: true
header-img: "images/bg1.jpg"
---
* content
{:toc}

## 引言

从2011年甚至更早开始，智能手机，智能终端，车载芯片等等智能终端中，高通芯片无处不在。相比较Intel，高通抓住了移动处理器中续航的问题，不断推出低功耗移动处理器，从而抓住移动处理器的市场。日常工作中接触到很多冠以高通之名的产品，记录以习之。

## 性能排行榜

收集了一下[2018年高通骁龙CPU处理器排行榜](https://www.xianshua.net/top/5534.html)和[各种手机品牌的处理器性能对比](http://mobile.zol.com.cn/soc/)，从中可以看出，骁龙系列
处理器也是分为高中低端处理器的，其中去年最常见的845系列，占据较大的市场份额。与之争锋麒麟
980虽然在HUWEI Meta 20 Pro的跑分汇总获得更高名次，性能前10中高通独占8席。

## 骁龙

为什么要选择骁龙处理器？
骁龙移动处理器是Qualcomm开发的完整片上系统解决方案系列产品，该系列适应用户需求而提供卓越的用户体验和更长的电池寿命。利用骁龙处理器先进的处理能力和并发性能，您可以同时运行多个高级应用，而对电池的消耗却可以降到最少。

骁龙处理器经过发展，早已不再仅仅支持先进的手机，还可在各种智能产品和互连设备上看到它的身影，包括可穿戴设备、智能家电、智能电话亭和数字标识系统等。我们的一系列软硬件解决方案专门提供您所需要的，以帮助您最大限度地利用采用骁龙处理器的设备。我们的SDK、Profiler分析器和调试器能帮助您分析和提升应用性能、打造创新特性和创造新的互连体验。我们甚至能帮助您开始按照您自身的设计打造设备（从原型设计到生产的全过程）。如果您要打造下一代设备，采用骁龙处理器的开发设备，您便已经可以将未来握在手中了。

[高通公司官网开发文档](https://developer.qualcomm.com/)

### CPU

有了高品质的处理内核，骁龙处理器中经优化的CPU是专为让您的应用运行得更快、更流畅而设计。我们所有CPU的目标是将世界级的移动体验带进生活，同时智能地控制电池寿命。但是如果没有能完全发挥其特性的软件，即使是最高性能的CPU也不能开发出自身的全部潜力。采用骁龙LLVM编译器编译的代码在骁龙处理器上会执行的更好，因为它具有独特的优化处理和漏洞修复。

### GPU

图形性能对于现代移动体验是一个重要部分，这就是为什么我们的Qualcomm骁龙处理器要内置开拓性的Adreno™图形处理器的原因。Adreno是目前最先进的移动图形处理背后的发电站，它能加速游戏、用户界面和网络浏览器中复杂几何体的渲染。快来下载Adreno SDK，优化您针对Adreno GPU的应用，该SDK含打造沉浸式手机游戏体验所需的工具、库、示例、文档和辅导资料。您还可利用Adreno Profiler分析器来分析和优化您应用的图形性能。该分析器具有的特性包括：基于硬件的性能监视器、渲染调用指标、Shader原型设计等。

### DSP

在最适合的处理引擎上运行适当的任务能为开发者带来性能优势。这就是为什么开发Hexagon DSP的原因，该产品专为优化调制解调器和多媒体应用而设计，具有的特性包括**硬件辅助多线程**。Hexagon SDK使您能最大化发挥DSP的性能，提供一个用于生成动态Hexagon DSP代码模块的环境，并且使您能访问Hexagon DSP上的内置计算资源。该SDK是专为帮助确保处理效率而设计，这意味着它具备更高的流动性、更低的延迟和卓越的应用性能。

## [CSDN中高通专栏](https://qualcomm.csdn.net/)


## [【中科创达-王庆民】关于Hexagon DSP功能介绍](https://blog.csdn.net/awangqm/article/details/49333385)
Qualcomm的晓龙芯片从创立之几乎一直内置Hexagon DSP芯片，它是移动异构计算必需的处理引擎。Hexagon架构设计的核心在于如何在低功耗的情况下能够高性能的处理各种各样的应用，它具有的特性包括多线程，特权级，VLIW，SIMD以及专门适应于信号处理的指令。该CPU可以在单个时间周期中依序快速的将四个指令（已打包好）处理为执行单元。硬件多线程则由 TMT（TemporalMultiThreading，时间多线程）来实现，在这种模式下，频率600MHz的物理核心可以被抽象成三个频率200MHz的核心。许多体验如声音和图像增强功能以及高级摄像头和传感器功能都包括信号处理任务，而DSP尤其擅长在低功耗下处理这些任务。起初，Hexagon DSP作为处理引擎，主要用于语音和简单的音频播放。现在，Hexagon DSP的作用已经扩展至多种用途，如图像增强、计算机视觉、扩增实境、视频处理和传感器处理。随着智能手机使用需求的不断加大，现在包括摄像头和传感器功能都包括信号处理任务都需要借助DSP来完成，相比强大的CPU，DSP尤其擅长在低功耗下处理这些任务。

![@Qualcomm最新发布的Hexagon 680 DSP版本新特性](https://cwlseu.github.io/images/dsp/820.png)


## [高清图像处理，低功耗——Qualcomm® Hexagon™ Vector eXtensions (HVX)](https://www.csdn.net/article/a/2015-09-15/15828177)

摘要：过去几年，开发人员一直在利用 Hexagon SDK，量身定制 DSP，处理音频、图像与计算 。在 HotChips 半导体会议上，我们揭开了即将上市的 Snapdragon 820 处理器中全新 Hexagon DSP 的部分面纱。这款 Hexagon 680 DSP ，集成宽幅向量处理 Hexagon 向量扩展（HVX）核心，充分利用新的DSP 应用实例。
英文原版[High-Res Image Processing, Low Power Consumption – Qualcomm® Hexagon™ Vector eXtensions (VX)](https://developer.qualcomm.com/blog/high-res-image-processing-low-power-consumption-qualcomm-hexagon-vector-extensions-vx)
关于HVX技术，可以参考如下介绍
https://www.hotchips.org/wp-content/uploads/hc_archives/hc27/HC27.24-Monday-Epub/HC27.24.20-Multimedia-Epub/HC27.24.211-Hexagon680-Codrescu-Qualcomm.pdf

高通向量拓展技术的概括
与NEON编程模型相类似，在计算机视觉应用领域
![Alt text](https://cwlseu.github.io/images/dsp/DSP-HVX.png)


指令和CPU的NEON指令相比，指令简单，更低功耗
![Alt text](https://cwlseu.github.io/images/dsp/DSP-Difference.png)

性能方面,CPU使用NEON优化虽然能够提升1~3的速度，但是单pixel功耗方面大约是DSP的4~18倍。
![@Benchmark](https://cwlseu.github.io/images/dsp/DSP-Benchmark.png)

## Snapdragon Neural Processing Engine (SNPE)
### Capabilities
The Snapdragon Neural Processing Engine (SNPE) is a Qualcomm Snapdragon software accelerated runtime for the execution of deep neural networks. With SNPE, users can:

* Execute an arbitrarily deep neural network
* Execute the network on the SnapdragonTM CPU, the AdrenoTM GPU or the HexagonTM DSP.
* Debug the network execution on x86 Ubuntu Linux
* Convert Caffe, Caffe2, ONNXTM and TensorFlowTM models to a SNPE Deep Learning Container (DLC) file
* Quantize DLC files to 8 bit fixed point for running on the Hexagon DSP
* Debug and analyze the performance of the network with SNPE tools
* Integrate a network into applications and other code via C++ or Java

### Workflow
Model training is performed on a popular deep learning framework (Caffe, Caffe2, ONNX and TensorFlow models are supported by SNPE.) After training is complete the trained model is converted into a DLC file that can be loaded into the SNPE runtime. This DLC file can then be used to perform forward inference passes using one of the Snapdragon accelerated compute cores.
The basic SNPE workflow consists of only a few steps:

![@SNPE运行模型的工作流](https://cwlseu.github.io/images/dsp/snpe.png)
* Convert the network model to a DLC file that can be loaded by SNPE.
* Optionally quantize the DLC file for running on the Hexagon DSP.
* Prepare input data for the model.
* Load and execute the model using SNPE runtime.

### 测试模型
`./snpe-net-run --container ./modelname.dlc --input_list list.one --use_dsp`


- [SNPE sdk download](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- [SNPE document](https://developer.qualcomm.com/docs/snpe/overview.html)
- [SNPE支持的网络层](https://developer.qualcomm.com/docs/snpe/network_layers.html)
- [SNPE用户自定义层JNI实现](https://blog.csdn.net/guvcolie/article/details/77937786)

## 参考链接

- [手机处理器性能排行榜](http://mobile.zol.com.cn/soc/)
- [手机CPU性能天梯图](http://www.mydrivers.com/zhuanti/tianti/01/)
- [2018年高通骁龙CPU处理器排行榜](https://www.xianshua.net/top/5534.html)
- [【德州仪器DSP技术应用工程师 冯华亮】影响高性能DSP功耗的因素及其优化方法](http://www.ti.com.cn/general/cn/docs/gencontent.tsp?contentId=61574)
- [移动端深度学习框架小结](https://blog.csdn.net/yuanlulu/article/details/80857211)