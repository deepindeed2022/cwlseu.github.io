---
layout: post
title: "DSSD:Deconvolution Single Shot Detector"
categories: [blog ]
tags: [CV算法, detection]
description: SSD的出现，引起了人们对YOLO和SSD的双重关注。SSD为什么会那么牛掰，还没有搞清楚SSD的数据增强，DSSD就来了
---

* content
{:toc}

## 论文来源
作者: http://www.cs.unc.edu/~wliu/

论文: <https://arxiv.org/abs/1701.06659>

效果展示: <http://www.cs.unc.edu/~cyfu/dssd_lalaland.mp4>

其中主要展示的是对大物体-- 人物的追踪比较多，其他物体比重不大。

## Abstract
采用Residual-101代替VGG网络
引入反卷积层

![@DSS和SSD架构图](https://cwlseu.github.io/images/ssd/DSSD.png)
对于小物体和多个物体的识别正确率提升，但是速度是下降的。如下图所示，每组图片中左边是SSD的结果，右边是DSSD的结果。
![@DSS和SSD结果对比](https://cwlseu.github.io/images/ssd/a.png)
![@DSS和SSD结果对比](https://cwlseu.github.io/images/ssd/b.png)
从速度方面，SSD的速度是DSSD是快的。原来SSD可以大搞46FPS,当前使用Residual-101之后大于为11.2FPS。速度是原来的1/4.
![@Speed and Accuracy on Pascal VOC2007](https://cwlseu.github.io/images/ssd/speed-accuracy-1.png)
也就是我们可以考虑在ResNet和VGG选择，权衡正确率和速度。

## SSD 回顾
这个可以看[gwyve的ssd]<http://gwyve.com/blog/2017/03/01/reading-note-SSD.html>
### Prediction Module
这个模块的设计是根据MS-CNN[^1]提出的"improving the sub-network of each task can improve accuracy."
![@Deconvolution Module介绍](https://cwlseu.github.io/images/ssd/predictionmodule.png)

## Deconvolutional SSD

### Deconvolution Module
![@Deconvolution Module介绍](https://cwlseu.github.io/images/ssd/deconv.png)

1. a batch normalization layer is added after each convolution
layer. 
2. learned deconvolution layer instead of bilinear upsampling. 
3. test different combination methods: element-wise sum and element-wise product. 

![@PM的效果](https://cwlseu.github.io/images/ssd/PM.png)

### Box radio
原理ssd中box radio选用的是2 and 3. 本文中采用YOLO9000中的kmeans聚类方法进行选取aspect radio.


### result

![@Speed and Accuracy on Pascal VOC2007](https://cwlseu.github.io/images/ssd/speed-accuracy-1.png)


### 参考文献

[^1]. Z. Cai, Q. Fan, R. S. Feris, and N. Vasconcelos. A unified multiscale deep convolutional neural network for fast object detection. In ECCV, 2016.