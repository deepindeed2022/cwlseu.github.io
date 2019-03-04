---
layout: post
title: "一起来看数据增强"
categories: [blog ]
tags: [detection, 深度学习]
description: 一起来看数据增强
comments: true
---

* content
{:toc}

## 引言

从AlexNet夺取ImageNet的冠军，到RCNN的出现，再到后来的SSD算法，数据增强仿佛像是一位功成名就的老者，虽然数据增强对于算法性能的提升起到重要的作用，但是他从来不居功，默默在背后付出"众里寻他千百度"，只为让你"蓦然回首，她在灯火阑珊处"。

## 数据增强(Data Augmentation)的目的与作用

卷积神经网络能够鲁棒地将物体分类，即便物体放置在不同的方向上，这也就是所说不变性的性质，即使卷积神经网络被放在不同方向上，它也能进行对象分类。更具体的说，卷积神经网络对平移、视角、尺寸或照度（或以上组合）保持不变性。
这就是数据增强的本质前提。在现实世界中，我们可能会有一组在有限的条件下拍摄的图像 。但是，我们的目标应用可能是在多变的环境中，例如，不同的方向、位置、比例、亮度等。我们通过使用经综合修改过的数据来训练神经网络，以应对这些情形。

> 数据少的负面影响：
1. 模型训练的时候可能无法收敛，少量训练数据难以提供足够的信息给模型学习
2. 过拟合，模型容易将训练数据的完全不相关信息学习进去，如噪声
3. 容易陷入局部最优值
4. 难以衡量模型好坏，除了训练数据，测试数据也非常少，少量数据整的与否可能对结果产生较大影响。

> 数据不平衡的负面影响：
最常见的就是模型的权值更新被数据多的一个方向带跑偏了。

> 数据增强的作用
1. 补充数据样本不足
2. 减少网络的过拟合现象，通过对训练图片进行变换可以得到泛化能力更强的网络，更好的适应应用场景。

## 基本方法
现在最常用的数据方案是
数据增强的基本方法无非就是图像的基本操作进行排列组合，生成千万种数据增强的可能性：
* 旋转/反射变换(Rotation/reflection): 随机旋转图像一定角度; 改变图像内容的朝向;
* 翻转变换(flip): 沿着水平或者垂直方向翻转图像;
* 缩放变换(zoom): 按照一定的比例放大或者缩小图像;
* 平移变换(shift): 在图像平面上对图像以一定方式进行平移;
* 可以采用随机或人为定义的方式指定平移范围和平移步长, 沿水平或竖直方向进行平移. 改变图像内容的位置;
* 尺度变换(scale): 对图像按照指定的尺度因子, 进行放大或缩小; 或者参照SIFT特征提取思想, 利用指定的尺度因子对图像滤波构造尺度空间. 改变图像内容的大小或模糊程度;
* 对比度变换(contrast): 在图像的HSV颜色空间，改变饱和度S和V亮度分量，保持色调H不变. 对每个像素的S和V分量进行指数运算(指数因子在0.25到4之间), 增加光照变化;
* 噪声扰动(noise): 对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声;
* 颜色变化：在图像通道上添加随机扰动。
* PCA Jittering
首先，按照 RGB 三个颜色通道计算均值和方差，规范网络输入数据；
然后，计算整个训练数据集的协方差矩阵，进行特征分解，得到特征向量和特征值，以作 PCA Jittering.

## caffe中的数据增强

`caffe/src/caffe/data_transformer.cpp`
只发现mirror、scale、crop三种。 其中Data_Transformer被调用的时候，会采用1/2的随机镜像，以及对应输入参数的scale和crop进行生成新的样本，输出到下一层网络中。因此，我们使用caffe训练的时候，只训练一个epoch就可以的情况是万万不能的。即使是同一个图片，同一套参数，也要进行多次采样才行。每个epoch进行shuffle一次，每次的batch中的分布就会发生变化，同样一张图片，虽然是同一套参数，也可能会出现不同的结果。在训练过程中的数据采样，随机性让样本不至于将噪声过度的学习。

## SSD中的数据增强
SSD中的数据采样，在caffe中数据采样的基础上，进行了充分扩充，增强方式包括resize，crop，distort，...
更重要的是引入BatchSampler, 以Batch中的数据基础，达到真正的增加不同overlap的数据的目的，使得检测能力极大增强。因此，我一度认为，SSD的成功不是One-Stage在Detection的突破，而是数据增强方法的提升。

```
// Sample a batch of bboxes with provided constraints.
message BatchSampler {
  // 是否使用原来的图片
  optional bool use_original_image = 1 [default = true];
  // sampler的参数
  optional Sampler sampler = 2;
  // 对于采样box的限制条件，决定一个采样数据positive or negative
  optional SampleConstraint sample_constraint = 3;
  // 当采样总数满足条件时，直接结束
  optional uint32 max_sample = 4;
  // 为了避免死循环，采样最大try的次数.
  optional uint32 max_trials = 5 [default = 100];
}
```

更多内容，参考博客：http://deepindeed.cn/2017/04/05/SSD-Data-Augmentation/

## 海康威视MSCOCO比赛中的数据增强

* 第一，对颜色的数据增强，包括色彩的饱和度、亮度和对比度等方面，主要从Facebook的代码里改过来的。
* 第二，PCA Jittering，最早是由Alex在他2012年赢得ImageNet竞赛的那篇NIPS中提出来的. 我们首先按照RGB三个颜色通道计算了均值和标准差，对网络的输入数据进行规范化，随后我们在整个训练集上计算了协方差矩阵，进行特征分解，得到特征向量和特征值，用来做PCA Jittering。
* 第三，在图像进行裁剪和缩放的时候，我们采用了随机的图像差值方式。
* 第四， Crop Sampling，就是怎么从原始图像中进行缩放裁剪获得网络的输入。比较常用的有2种方法：
  - 一是使用Scale Jittering，VGG和ResNet模型的训练都用了这种方法。
  - 二是尺度和长宽比增强变换，最早是Google提出来训练他们的Inception网络的。我们对其进行了改进，提出Supervised Data Augmentation方法。

## 启发点

* 并不是越多越好，要在多的基础上保持随机性，因为应用场景的不是固定的输入
* 结合新的方法，例如GAN进行生成图片等技术，进一步扩充训练集合
* 合理性采样降低样本不均衡的影响


## 小结
同样的算法，数据增强能够显著提升算法的性能

## 可参考链接
- [Discriminative Unsupervised Feature Learning
with Exemplar Convolutional Neural Networks](https://arxiv.org/pdf/1406.6909.pdf)
- [输入图像随机选择一块区域涂黑，《Random Erasing Data Augmentation》](https://arxiv.org/pdf/1511.05635.pdf)
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501v1)
- [海康威视研究院ImageNet2016竞赛经验分享](https://zhuanlan.zhihu.com/p/23249000)
- https://github.com/kevinlin311tw/caffe-augmentation 
- https://github.com/codebox/image_augmentor
- https://github.com/aleju/imgaug.git
- [The art of Data Augmentation](http://lib.stat.cmu.edu/~brian/905-2009/all-papers/01-jcgs-art.pdf)
- [Augmentation for small object detection](https://arxiv.org/abs/1902.07296)
- [使用深度学习(CNN)算法进行图像识别工作时，有哪些data augmentation 的奇技淫巧？](https://www.zhihu.com/question/35339639)

## 案例-医学图像分割的数据增广
[Data augmentation using learned transforms for one-shot medical image segmentation](http://arxiv.org/abs/1902.09383)

github: https://github.com/xamyzhao/brainstorm

    Biomedical image segmentation is an important task in many medical applications. Segmentation methods based on convolutional neural networks attain state-of-the-art accuracy; however, they typically rely on supervised training with large labeled datasets. Labeling datasets of medical images requires significant expertise and time, and is infeasible at large scales. To tackle the lack of labeled data, researchers use techniques such as hand-engineered preprocessing steps, hand-tuned architectures, and data augmentation. However, these techniques involve costly engineering efforts, and are typically dataset-specific. We present an automated data augmentation method for medical images. We demonstrate our method on the task of segmenting magnetic resonance imaging (MRI) brain scans, focusing on the one-shot segmentation scenario -- a practical challenge in many medical applications. Our method requires only a single segmented scan, and leverages other unlabeled scans in a semi-supervised approach. We learn a model of transforms from the images, and use the model along with the labeled example to synthesize additional labeled training examples for supervised segmentation. Each transform is comprised of a spatial deformation field and an intensity change, enabling the synthesis of complex effects such as variations in anatomy and image acquisition procedures. Augmenting the training of a supervised segmenter with these new examples provides significant improvements over state-of-the-art methods for one-shot biomedical image segmentation.

### 医学图像segment中U-Net
- paper: [U-net: Convolutional networks for biomedical image segmentation](https://lmb.informatik.uni-freiburg.de/Publications/2019/FMBCAMBBR19/paper-U-Net.pdf)
- 作者： Olaf Ronneberger, Philipp Fischer, and Thomas Brox 
- [project](https://lmb.informatik.uni-freiburg.de/resources/opensource/unet/)
其中使用的数据增强方案为论文 [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/pdf/1406.6909.pdf)中的方法：

> 
    we train the network to discriminate between a set of surrogate classes. 
    Each surrogate class is formed by applying a variety of transformations 
    to a randomly sampled ’seed’ image patch. In contrast to supervised network 
    training, the resulting feature representation is not class specific. 
    It rather provides robustness to the transformations that have been applied 
    during training. This generic feature representation allows for classification 
    results that outperform the state of the art for unsupervised learning on 
    several popular datasets (STL-10, CIFAR-10, Caltech-101, Caltech-256). 
    While such generic features cannot compete with class specific features 
    from supervised training on a classification task, we show that they are 
    advantageous on geometric matching problems, where they also outperform the 
    SIFT descriptor.