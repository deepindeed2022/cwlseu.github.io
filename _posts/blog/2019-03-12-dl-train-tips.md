---
layout: post
comments: true
title: "神经网络模型训练提升成果总结"
categories: [blog ]
tags: [detection]
description: 物体检测算法概览
---

* content
{:toc}

## 问题: foreground-background class imbalance

### OHEM：Training Region-based Object Detectors with Online Hard Example Mining
- intro: CVPR 2016 Oral. Online hard example mining (OHEM)
- arxiv: http://arxiv.org/abs/1604.03540
- paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf
- github（Official）: https://github.com/abhi2610/ohem
- author page: http://abhinav-shrivastava.info/
paper里直接说的最初的思想就是想把Bootstrapping移植到dl当中来，所以产生了ohem。由于bootstrapping本身是迭代算法，直接应用到dl当中，会使dl的训练变得非常慢。为此，作者借用了其核心思想--让hard example起更大的作用，提出了ohem。
具体的方法就是先让所有样本通过dl计算出loss,以loss大为hard example的标准，从而选出hard example来，然后再组成一组batch,进行训练。
![@OHEM设计架构](https://cwlseu.github.io/images/detection/OHEM.png)

### Focal Loss for Dense Object Detection

* 论文地址: https://arxiv.org/abs/1708.02002
* 主要观点：
提出Single stage detector不好的原因完全在于：
    - 极度不平衡的正负样本比例: anchor近似于sliding window的方式会使正负样本接近1000：1，而且绝大部分负样本都是easy example，这就导致下面一个问题：
    - gradient被easy example dominant的问题：往往这些easy example虽然loss很低，但由于数量众多，对于loss依旧有很大贡献，从而导致收敛到不够好的一个结果。所以作者的解决方案也很直接：直接按照loss decay掉那些easy example的权重，这样使训练更加bias到更有意义的样本中去。
![@Focal Loss v.s. Cross Enctroy Loss train performance](https://cwlseu.github.io/images/detection/FocalLoss.png)

### Gradient Harmonized Single-stage Detector

- 论文地址：https://arxiv.org/pdf/1811.05181.pdf
- 作者：Buyu Li, Yu Liu and Xiaogang Wang
- 针对的的痛点：Huge difference in quantity between positive and negative examples as well as between easy and hard examples.
- Abstract
GHM can be easily embedded into both classification loss function like cross-entropy (CE) and regression
loss function like smooth-L1 (SL1) loss.two novel loss functions called GHM-C and GHM-R are designed
to balancing the gradient flow for anchor classification and bounding box refinement

### A-Fast-RCNN: Hard positive generation via adversary for object detection
从更好的利用数据的角度出发，OHEM和S-OHEM都是发现困难样本，而A-Fast-RCNN的方法则是通过GAN的方式在特征空间产生具有部分遮挡和形变的困难样本

## 其他参考链接

- [熟睡的孩子关于GHM文章的解读，来自知乎](https://zhuanlan.zhihu.com/p/50217821)
- [为什么说ohem是bootstrapping在dl中的应用呢？](https://www.zhihu.com/question/56092850/answer/216461322)
- [OHEM论文解读](https://zhuanlan.zhihu.com/p/58162337)