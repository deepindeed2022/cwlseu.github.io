---
layout: post
title: "A Gift from Knowledge Distillation"
categories: [blog ]
tags: [深度学习]
description: 这是一篇关于知识迁移学习的文章
---
[TOC]

## 来源

[CVPR2017](https://cgv.kaist.ac.kr/sigmi/data/CVPR2017.pdf)

## 名词解释

**迁移学习**

迁移学习的效果与知识的表示是很相关的。

> Because a DNN uses many layers sequentially to map from the input space to the output space, the flow of solving a problem can be defined as the relationship between features from two layers.

[**Gramian Matrix格莱姆矩阵**](https://en.wikipedia.org/wiki/Gramian_matrix)

Gramian矩阵是通过计算一组特征向量的内积进行生成的，包含了两个向量之间的方向关系，可以用来表示文理特征。

**FSP matrix**

> The extracted feature maps from two layers are used to generate the flow of solution procedure (FSP) matrix. The student DNN is trained to make its FSP matrix similar to that of the teacher DNN

本文中的FSP矩阵的计算与Gramian矩阵的计算是类似的。我们计算GramianMatrix是跨层计算的。而传统Gramian Matrix是在一个层内的features之间进行计算的。如下图所示。

![本文中的迁移学习方法示意图](https://cwlseu.github.io/images/cvpr2017/kd/method.JPG)

**Distilled Knowledge**
如果将DNN的输入视为问题，将DNN的输出作为回答，那么我们可以将中间的生成的特征看作回答问题过程中的中间结果。
老师教学生的过程也是类似的，学生需要着重学习的是**某类**题目的解答方法，而不是学会解**某个**题目。因此，一般是对问题的解决过程进行建模。

## Model
对问题的解决过程往往被定义为两个中间结果的关系进行表示。本文中定义FSP矩阵来表示问题解决过程。
> 如何计算FSP矩阵

![FSP矩阵计算](https://cwlseu.github.io/images/cvpr2017/kd/FSP.JPG)

> 如何优化FSP矩阵

首先是Teacher Network中生成的FSP矩阵，Gt_i (i = 1, ..n). Student Network 中生成了n个FSP矩阵，Gs_i (i=1,.. n). 然后Teacher和Student Network的FSP矩阵组成对(Gt_i, Gs_i), i = 1,2,...n
![FSP矩阵计算](https://cwlseu.github.io/images/cvpr2017/kd/Loss.JPG)

> 方法架构图

> Stage 1: 学习FSP矩阵

    Weights of the student and teacher networks: Ws, Wt
    1: Ws = arg minWs LFSP(Wt, Ws) # 就是上面优化FSP矩阵中提到的损失函数的优化

> Stage 2: 针对原始任务进行训练

    1: Ws = arg min Ws Lori(Ws)  # 例如是分类任务的话， 我们可以使用softmax交叉熵损失作为任务的损失函数进行学习和优化

## 实验

### **Fast Optimization**
![DenseNet的组成结构](https://cwlseu.github.io/images/cvpr2017/kd/Fast.JPG)

从结果中可以看出student network比teacher network 收敛速度更快。 大概快了3倍，试验中Teacher Network和Student Network是相同的结构。
依次类推，1/3原来的迭代次数，我们在Student Network训练过程中使用相应的学习率。实验结果如下表：
![Recignition rates on CIFAR-10](https://cwlseu.github.io/images/cvpr2017/kd/Table1.JPG)

\* 表示每个网络训练了21000iteration, 原始网络迭代次数为63000.
两个+的符号（++）表示Teacher Network在前面64000次迭代基础上，又训练了21000次迭代。
宝剑符号(+-)表示stage 1中，student network学习的是randomly shuffled 的FSP矩阵。Student*+-表示Student network在stage 1训练了21000次迭代，stage 2训练了21000次迭代。

    As both teacher networks and student networks are of the same architecture, one can also transfer knowledge by directly copying weights. FSP is less restrictive than copying the weights and allows for better diversity and ensemble performance

###  **Network Minimization**
最近，很多研究结果都是使用更大更深的神经网络获得更好的性能表现。那么，我们想通过将学习深层网络的知识应用到提升小网络中来。就像下图所示：

![DenseNet的组成结构](https://cwlseu.github.io/images/cvpr2017/kd/arch.JPG)

    Because the student DNN and teacher DNN had the same number of channels, the sizes of the FSP matrices were the same. By minimizing the distance between the FSP matrices of the student network and teacher network, we found a good initial weight for the student network. Then,the student network was trained to solve the main task.

来看看使用这个想法，全部不使用数据增强，训练网络的结果：
![DenseNet的组成结构](https://cwlseu.github.io/images/cvpr2017/kd/Table4.JPG)
从中可以看出使用distill knowledge还是有效果的。而且使用本文中的方法效果比FitNet好很多(2.0%+)。

###  **Transfer Learning**
是跨任务知识使用。尤其是利用那些使用大数据集合训练的神经网络模型应用到小规模的数据集合方面的应用。本文中试验了ImageNet-> Caltech-UCSD Birds (CUB) 200-2011 dataset的迁移学习，神经网络是比Teacher DNN 浅1.7倍左右。但是这个实验做的还是不够充分，感觉是对前面网络最小化实验的另一种说法啊。

## 评测标准
>Recognition rates

由于本文中使用的数据集合是单一主要是单一物体，主要是物体分类的问题。

>收敛速度

## 小结
本文利用使用两层featuremap 之间的关系表示为FSP。利用这种学习过程知识表示方法，在DNN加速方面有很的表现。并且这个方法还可以应用到迁移学习中。当前的研究主要是在一些简单得数据集合上进行实验的，是不是可以考虑在更复杂的数据集合上进行实验。

## 参考文献

1. [MSRA:Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification](http://blog.csdn.net/shuzfan/article/details/51347572)
 MSRA初始化是一个均值为0方差为2/n的高斯分布
2. [A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta,and Y. Bengio. Fitnets: Hints for thin deep nets. In In Proceedings of ICLR, 2015.](https://arxiv.org/abs/1412.6550)