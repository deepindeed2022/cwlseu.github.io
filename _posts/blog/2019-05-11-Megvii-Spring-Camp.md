---
layout: post
title: "Megvii:计算机视觉之物体检测与深度神经网络模型设计专场"
categories: [blog ]
tags: [计算机视觉]
description: "参加计算机视觉物体检测的一些想法"
---

* content
{:toc}

## 概述

今天主要围绕计算机视觉领域的一些任务中，如何提高score的方面进行了各种方面的讨论。研究领域包括：基本的场景的物体检测，开放领域的物体检测，不规则文本检测。涉及的相关技术有
* 在backbone设计中总结的一些tips；
* 如何在物体检测领域，通过分析对比各种方法，提出自己的解决方案；
* 物体检测领域中多尺度问题的重要性；
* anchor是什么东西，anchor free和anchor base之间的 vs
* attention在计算机视觉中的应用SENet

## Deform ConvNet(DCN) v.s. ConvNet

这个专题主要是为了改善遮挡和分割算法而做的相关研究。在普通的卷积中，加入旋转，scale等特性，使得能够对一个非正方凸区域进行卷积。这就可以使得卷积提取特征不仅仅局限在的一个方正的空间区域，可以根据物体的形状等等学习到相关的角度变化等权重，从而实现对非规则空间区域的特征提取。这无疑将物体分割任务直接统一为物体检测任务，还是比较nice的构想。虽然有好处，但由于多个计算过程中引入了非规则的问题，导致计算速度方面可能受到一定的影响。
![](http://cwlseu.github.io/images/detection/megvii/deformableConv_1.jpg)

![](http://cwlseu.github.io/images/detection/megvii/deformableConv_2.jpg)

- [arxiv paper](https://arxiv.org/abs/1703.06211)

## Review Object Detection from anchor base to anchor-free

这个主要通过介绍anchor-base和anchor-free的发展路线，讨论两种方法的之间的优劣之处。
认为现在anchor-base的research空间已经有限，所以开始考虑向anchor-free开始进展。
anchor-base需要设置较多的超参数，落地的过程中存在较大的调参困难。而anchor-free相对来说，比较少。

![@detection blueprint](http://cwlseu.github.io/images/detection/megvii/detection.jpg)


## one-stage和two-stage的anchor-base detection

它们的主要区别
* one-stage网络速度要快很多
* one-stage网络的准确性要比two-stage网络要低

### 为什么one-stage网络速度要快很多？

首先来看第一点这个好理解，one-stage网络生成的anchor框只是一个逻辑结构，或者只是一个数据块，只需要对这个数据块进行分类和回归就可以，不会像two-stage网络那样，生成的 anchor框会映射到feature map的区域（rcnn除外），然后将该区域重新输入到全连接层进行分类和回归，每个anchor映射的区域都要进行这样的分类和回归，所以它非常耗时

### 为什么one-stage网络的准确性要比two-stage网络要低？

我们来看rcnn，它是首先在原图上生成若干个候选区域，这个候选区域表示可能会是目标的候选区域，注意，这样的候选区域肯定不会特别多，假如我一张图像是100x100的，它可能会生成2000个候选框，然后再把这些候选框送到分类和回归网络中进行分类和回归，fast-rcnn其实差不多，只不过它不是最开始将原图的这些候选区域送到网络中，而是在最后一个feature map将这个候选区域提出来，进行分类和回归，它可能最终进行分类和回归的候选区域也只有2000多个并不多。再来看faster-rcnn，虽然faster-rcnn它最终一个feature map它是每个像素点产生9个anchor，那么100x100假如到最终的feature map变成了26x26了，那么生成的anchor就是26x26x9 = 6084个，虽然看似很多，但是其实它在rpn网络结束后，它会不断的筛选留下2000多个，然后再从2000多个中筛选留下300多个，然后再将这300多个候选区域送到最终的分类和回归网络中进行训练，所以不管是rcnn还是fast-rcnn还是faster-rcnn，它们最终进行训练的anchor其实并不多，几百到几千，不会存在特别严重的正负样本不均衡问题，但是我们再来看yolo系列网络，就拿yolo3来说吧，它有三种尺度，13x13，26x26，52x52，每种尺度的每个像素点生成三种anchor，那么它最终生成的anchor数目就是(13x13+26x26+52x52)*3 = 10647个anchor，而真正负责预测的可能每种尺度的就那么几个，假如一张图片有3个目标，那么每种尺度有三个anchor负责预测，那么10647个anchor中总共也只有9个anchor负责预测，也就是正样本，其余的10638个anchor都是背景anchor，这存在一个严重的正负样本失衡问题，虽然位置损失，类别损失，这10638个anchor不需要参与，但是目标置信度损失，背景anchor参与了，因为

$$总的损失 = 位置损失 + 目标置信度损失 + 类别损失$$

所以背景anchor对总的损失有了很大的贡献，但是我们其实不希望这样的，我们更希望的是非背景的anchor对总的损失贡献大一些，这样不利于正常负责预测anchor的学习，而two-stage网络就不存在这样的问题，two-stage网络最终参与训练的或者计算损失的也只有2000个或者300个，它不会有多大的样本不均衡问题，不管是正样本还是负样本对损失的贡献几乎都差不多，所以网络会更有利于负责预测anchor的学习，所以它最终的准确性肯定要高些

总结下：
说了那么多，用一个句话总结，one-stage网络最终学习的anchor有很多，但是只有少数anchor对最终网络的学习是有利的，而大部分anchor对最终网络的学习都是不利的，这部分的anchor很大程度上影响了整个网络的学习，拉低了整体的准确率；而two-stage网络最终学习的anchor虽然不多，但是背景anchor也就是对网络学习不利的anchor也不会特别多，它虽然也能影响整体的准确率，但是肯定没有one-stage影响得那么严重，所以它的准确率比one-stage肯定要高

### 那么什么情况下背景anchor不会拉低这个准确率呢？

设置阀值，与真实GrundTruth IOU阀值设得小一点，只要大于这个阀值，就认为你是非背景anchor（注意这部分anchor只负责计算目标置信度损失，而位置、类别损失仍然还是那几个负责预测的anchor来负责）或者假如一个图片上有非常多的位置都是目标，这样很多anchor都不是背景anchor；总之保证背景anchor和非背景anchor比例差不多，那样可能就不会拉低这个准确率，但是只要它们比例相差比较大，那么就会拉低这个准确率，只是不同的比例，拉低的程度不同而已

### 解决one-stage网络背景anchor过多导致的不均衡问题方案

* 采用focal loss，将目标置信度这部分的损失换成focal loss

* 增大非背景anchor的数量

某个像素点生成的三个anchor，与真实GrundTruth重合最大那个负责预测，它负责计算位置损失、目标置信度损失、类别损失，这些不管，它还有另外两个anchor，虽然另外两个anchor不是与真实GrundTruth重合最大，但是只要重合大于某个阀值比如大于0.7，我就认为它是非背景anchor，但注意它只计算目标置信度损失，位置和类别损失不参与计算，而小于0.3的我直接不让它参与目标置信度损失的计算，实现也就是将它的权重置0，这个思想就类似two-stage网络那个筛选机制，从2000多个anchor中筛选300个参与训练或者计算目标置信度损失，相当于我把小于0.3的anchor我都筛选掉了，让它不参与损失计算

* 设置权重
在目标置信度损失计算时，将背景anchor的权重设置得很小，非背景anchor的权重设置得很大。

## 当前scene text detection中的一些问题和研究进展

这个话题我还比较感兴趣的。因为我后面要做一些自然场景文本检测的一个项目，所以最近在突击这方面的论文。而这个论坛中关于自然场景文本检测的分享让我受益颇丰。我才发现原来要找的相关文献很多都是白翔老师这个组的工作。

> 白翔，华中科技大学电信学院教授，先后于华中大获得学士、硕士、博士学位。他的主要研究领域为计算机视觉与模式识别、深度学习应用技术。尤其在形状的匹配与检索、相似性度量与融合、场景OCR取得了一系列重要研究成果，入选2014、2015、2016年Elsevier中国高被引学者。他的研究工作曾获微软学者，首届国家自然科学基金优秀青年基金的资助。他已在相关领域一流国际期刊或会议如PAMI、IJCV、CVPR、ICCV、ECCV、NIPS、ICML、AAAI、IJCAI上发表论文40余篇。任国际期刊Pattern Recognition, Pattern Recognition Letters, Neurocomputing, Frontier of Computer Science编委，VALSE指导委员，曾任VALSE在线委员会(VOOC)主席, VALSE 2016大会主席, 是VALSE在线活动（VALSE Webinar）主要发起人之一。

现在还是没有入门，师傅已经带我走马观花了，剩下的就是我自己细细品读各个方法的奥秘了。

### 暂时定的路线是：

1. SWT，MSER两个经典方法和衍生方法
2. 引入CNN之后如何演变到CNN+LSTM的方法
3. 分割技术如何提升不规则排布文本检测的

### 需要思考的内容：

1. 常用数据集，和应用场景的实际数据之间的差别
2. 各个方法之间的改进侧略的出发点是什么，有什么收货
3. 水平文本检测性能、速度比较推荐的算法是什么，简单应用场景呢？
4. 常规检测算法的与文本检测算法之间能不能直接迁移应用，有什么gap，克服这个gap需要做什么工作
5. 分割算法应用到文本检测中有什么结果，为什么会对非规则文本检测有好处

## 相关文献及blog

- [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751)
- [Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/81437413)
- https://blog.csdn.net/francislucien2017/article/details/88583219
