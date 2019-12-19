---
layout: post
comments: true
title: "Detection算法Overview"
categories: [blog ]
tags: [detection, CV算法]
description: 物体检测算法概览
---

* content
{:toc}

# 物体检测算法概述

深度学习让物体检测从实验室走到生活。基于深度学习的物体检测算法分类两大类。一类是像RCNN类似的两stage方法，将ROI的选择和对ROI的分类score过程。
另外一类是类似YOLO将ROI的选择和最终打分实现端到端一步完成。前者是先由算法生成一系列作为样本的候选框，再通过卷积神经网络进行样本分类；后者则不用产生候选框，直接将目标边框定位的问题转化为回归问题处理。正是由于两种方法的差异，在性能上也有不同，前者在检测准确率和定位精度上占优，后者在算法速度上占优。

![@物体检测算法概览图](https://cwlseu.github.io/images/detection/Detection-All.png)

[各种检测算法之间的性能对比，准确率，速度，以及一些可能加速的tips](https://www.jianshu.com/p/0586fdb412bf?utm_source=oschina-app)

## R-CNN的前世
- HOG
- DPM
- Selective Search
- [深度学习应用到物体检测以前](https://zhuanlan.zhihu.com/p/32564990)

# 基于region proposals的方法（Two-Stage方法）

- RCNN => Fast RCNN => Faster RCNN => FPN 
![@R-CNN、Fast R-CNN、Faster R-CNN三者关系](https://cwlseu.github.io/images/detection/RCNN-types2.png)

## RCNN
在早期深度学习技术发展进程中，主要都是围绕分类问题展开研究，这是因为神经网络特有的结构输出将概率统计和分类问题结合，提供一种直观易行的思路。国内外研究人员虽然也在致力于将其他如目标检测领域和深度学习结合，但都没有取得成效，这种情况直到R-CNN算法出现才得以解决。

- 论文链接：https://arxiv.org/pdf/1311.2524.pdf
- 作者：Ross Girshick Jeff Donahue Trevor Darrell Jitendra Malik
之前的视觉任务大多数考虑使用SIFT和HOG特征，而近年来CNN和ImageNet的出现使得图像分类问题取得重大突破，那么这方面的成功能否迁移到PASCAL VOC的目标检测任务上呢？基于这个问题，论文提出了R-CNN。
R-CNN (Region-based CNN features)
性能：RCNN在VOC2007上的mAP是58%左右。

### 主要工作流程

![@R-CNN要完成目标定位，其流程主要分为四步](https://cwlseu.github.io/images/detection/RCNN.png)
R-CNN要完成目标定位，其流程主要分为四步：
* 输入图像
* 利用选择性搜索(Selective Search)这样的区域生成算法提取Region Proposal 提案区域(2000个左右)
* 将每个Region Proposal分别resize(因为训练好的CNN输入是固定的)后(也即下图中的warped region，文章中是归一化为227×227)作为CNN网络的输入。
* CNN网络提取到经过resize的region proposal的特征送入每一类的SVM分类器，判断是否属于该类

### RCNN的缺点

* 对于提取的每个Region Proposal，多数都是互相重叠，重叠部分会被多次重复提取feature)，都要分别进行CNN前向传播一次(相当于进行了2000吃提特征和SVM分类的过程)，计算量较大。
* CNN的模型确定的情况下只能接受固定大小的输入(也即wraped region的大小固定)

### 优化思路

既然所有的Region Proposal都在输入图像中，与其提取后分别作为CNN的输入，为什么不考虑将带有Region Proposal的原图像直接作为CNN的输入呢？原图像在经过CNN的卷积层得到feature map，原图像中的Region Proposal经过特征映射(也即CNN的卷积下采样等操作)也与feature map中的一块儿区域相对应。

## SPP net

- 论文链接：]https://arxiv.org/abs/1406.4729
- 作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
简述：SPP net中Region Proposal仍然是在原始输入图像中选取的，不过是通过CNN映射到了feature map中的一片区域。

### SPP-NET的主要思想

![@SPPNet架构图](https://cwlseu.github.io/images/detection/SPPNet-arch.png)
* 对卷积层的feature map上的Region Proposal映射区域分别划分成1×1，2×2，4×4的窗口(window)，
* 在每个窗口内做max pooling，这样对于一个卷积核产生的feature map，就可以由SPP得到一个(1×1+2×2+4×4)维的特征向量。
* 论文中采用的网络结构最后一层卷积层共有256个卷积核，所以最后会得到一个固定维度的特征向量(1×1+2×2+4×4)×256维)，并用此特征向量作为全连接层的输入后做分类。

### 相对于R-CNN，SPP-net的优势

* 使用原始图像作为CNN网络的输入来计算feature map(R-CNN中是每个Region Proposal都要经历一次CNN计算)，大大减少了计算量。
* RCNN要resize，易于失真，而SPP-net不需要，原因是，SPP net中Region Proposal仍然是通过选择性搜索等算法在输入图像中生成的，通过映射的方式得到feature map中对应的区域，并对Region Proposal在feature map中对应区域做空间金字塔池化。通过空间金字塔池化操作，对于任意尺寸的候选区域，经过SPP后都会得到固定长度的特征向量。

<!-- ### SPP-net缺点
* 训练分多个阶段，步骤繁琐(微调网络+训练SVM+训练边框回归器)
* SPP net在微调网络的时候固定了卷积层，只对全连接层进行微调 -->

## Fast RCNN

- [`Fast R-CNN`](https://arxiv.org/abs/1504.08083)
- 作者：Ross Girshick
性能：在VOC2007上的mAP也提高到了68%

### 算法框架图

![](https://cwlseu.github.io/images/detection/FastRCNN-1.png)
![](https://cwlseu.github.io/images/detection/FastRCNN.png)

### 优点&贡献

* Fast R-CNN引入了RoI 池化层(相当于是一层SPP)，对于图像中的Region Poposal(也即RoI)，通过映射关系(图中的RoI projection)可以得到feature map中Region Proposal对应的区域。
* RoI Pooling层的操作是将feature map上的RoI区域划分为7×7的窗口，在每个窗口内进行max pooling，然后得到(7×7)×256的输出，最后连接到全连接层得到固定长度的RoI特征向量。
* 前面得到的RoI特征向量再通过全连接层作为Softmax和Regressor的输入,训练过程可以更新所有的网络层
* 训练过程是端到端的(Sigle-stage),并使用了一个多任务的损失函数(也即将边框回归直接加入到CNN网络中后,Fast R-CNN网络的损失函数包含了Softmax的损失和Regressor的损失)

### 小结

在前面三种目标检测框架中(R-CNN，SPP net，Fast R-CNN)，Region Proposal都是通过区域生成的算法(选择性搜索等)在原始输入图像中产生的，不过在SPP net及Fast R-CNN中都是输入图像中的Region Proposal通过映射关系映射到CNN中feature map上再操作的。Fast R-CNN中RoI池化的对象是输入图像中产生的proposal在feature map上的映射区域

## Faster RCNN

- 论文链接：https://arxiv.org/pdf/1506.01497.pdf
- 作者：Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun

### Faster RCNN算法框架

![@faster RCNN的算法框架](https://cwlseu.github.io/images/detection/FasterRCNN.png)
我们先整体的介绍下上图中各层主要的功能

* **卷积网络提取特征图**：

作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取input image的feature maps,该feature maps会用于后续的RPN层和全连接层。

* **RPN(Region Proposal Networks,区域提议网络)**:

RPN网络主要用于生成region proposals，
- 首先生成一堆Anchor box，对其进行裁剪过滤后通过softmax判断anchors属于前景(foreground)或者后景(background)，即是物体or不是物体，所以这是一个二分类；
- 另一分支bounding box regression修正anchor box，形成较精确的proposal（注：这里的较精确是相对于后面全连接层的再一次box regression而言）

Feature Map进入RPN后，先经过一次$3*3$的卷积，同样，特征图大小依然是$60*40$,数量512，这样做的目的应该是进一步集中特征信息，接着看到两个全卷积,即kernel_size=1*1,p=0,stride=1;
- cls layer 逐像素对其9个Anchor box进行二分类
- reg layer 逐像素得到其9个Anchor box四个坐标信息

特征图大小为60*40，所以会一共生成60*40*9=21600个Anchor box

![@FasterRCNN-RPN](https://cwlseu.github.io/images/detection/FasterCNN-RPN.png)

* **Roi Pooling**：

该层利用RPN生成的proposals和VGG16最后一层得到的feature map，得到固定大小的proposal feature map,进入到后面可利用全连接操作来进行目标识别和定位

* **Classifier**：

会将ROI Pooling层形成固定大小的feature map进行全连接操作，利用Softmax进行具体类别的分类，同时，利用SmoothL1Loss完成bounding box regression回归操作获得物体的精确位置。

![@FasterRCNN算法详细过程图](https://cwlseu.github.io/images/detection/FasterRCNN-Arch.png)
![@FasterRCNN proposal&RPN Netscope](https://cwlseu.github.io/images/detection/FasterRCNNNetwork.png)


### 参考链接

- [1]. https://www.cnblogs.com/wangyong/p/8513563.html
- [2]. https://www.jianshu.com/p/00a6a6efd83d
- [3]. https://www.cnblogs.com/liaohuiqiang/p/9740382.html
- [4]. https://blog.csdn.net/u011436429/article/details/80414615
- [5]. https://blog.csdn.net/xiaoye5606/article/details/71191429

![@RCNN系列对比总结表](https://cwlseu.github.io/images/detection/RCNN-types.png)

向[RGB大神](http://www.rossgirshick.info/),[He Kaiming](http://kaiminghe.com/)致敬！

## FPN(feature pyramid networks for object detection)

- 论文链接：https://arxiv.org/abs/1612.03144
- poster链接： https://vision.cornell.edu/se3/wp-content/uploads/2017/07/fpn-poster.pdf
- caffe实现: https://github.com/unsky/FPN
- 作者：Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie

### 图像金字塔

图像金字塔,在很多的经典算法里面都有它的身影，比如SIFT、HOG等算法。
我们常用的是高斯金字塔，所谓的高斯金字塔是通过高斯平滑和亚采样获得
一些下采样图像，也就是说第K层高斯金字塔通过平滑、亚采样操作就可以
获得K+1层高斯图像，高斯金字塔包含了一系列低通滤波器，其截止频率从
上一层到下一层是以因子2逐渐增加，所以高斯金字塔可以跨越很大的频率范围。
总之，我们输入一张图片，我们可以获得多张不同尺度的图像，我们将这些
不同尺度的图像的4个顶点连接起来，就可以构造出一个类似真实金字塔的一
个图像金字塔。通过这个操作，我们可以为2维图像增加一个尺度维度（或者说是深度），
这样我们可以从中获得更多的有用信息。整个过程类似于人眼看一个目标由远及近的
过程（近大远小原理）。

![@图像金字塔](https://cwlseu.github.io/images/detection/pyramidImage.jpg)

### 论文概述：

作者提出的多尺度的object detection算法：FPN（feature pyramid networks）。原来多数的object detection算法都是只采用顶层特征做预测，但我们知道低层的特征语义信息比较少，但是目标位置准确；高层的特征语义信息比较丰富，但是目标位置比较粗略。另外虽然也有些算法采用多尺度特征融合的方式，但是一般是采用融合后的特征做预测，而本文不一样的地方在于预测是在不同特征层独立进行的。

![@FPN架构图](https://cwlseu.github.io/images/detection/FPN.png)

前面已经提到了高斯金字塔，由于它可以在一定程度上面提高算法的性能，
因此很多经典的算法中都包含它。但是这些都是在传统的算法中使用，当然也可以将
这种方法直应用在深度神经网络上面，但是由于它需要大量的运算和大量的内存。
但是我们的特征金字塔可以在速度和准确率之间进行权衡，可以通过它获得更加鲁棒
的语义信息，这是其中的一个原因。

![@FPN不同层识别的目标不同](https://cwlseu.github.io/images/detection/FPN-multiScale.png)

如上图所示，我们可以看到我们的图像中存在不同尺寸的目标，而不同的目标具有不同的特征，
利用浅层的特征就可以将简单的目标的区分开来；
利用深层的特征可以将复杂的目标区分开来；这样我们就需要这样的一个特征金字塔来完成这件事。
图中我们在第1层（请看绿色标注）输出较大目标的实例分割结果，
在第2层输出次大目标的实例检测结果，在第3层输出较小目标的实例分割结果。
检测也是一样，我们会在第1层输出简单的目标，第2层输出较复杂的目标，第3层输出复杂的目标。

### 小结

作者提出的FPN（Feature Pyramid Network）算法同时利用低层特征高分辨率和高层特征的高语义信息，通过融合这些不同层的特征达到预测的效果。并且预测是在每个融合后的特征层上单独进行的，这和常规的特征融合方式不同。

## Mask-RCNN

- 论文地址：https://arxiv.org/abs/1703.06870
- 作者：Kaiming He，Georgia Gkioxari，Piotr Dollar，Ross Girshick
- FAIR Detectron：https://github.com/facebookresearch/Detectron
- tensorflow: https://github.com/matterport/Mask_RCNN

## Mask Scoring R-CNN
- 论文地址：https://arxiv.org/abs/1903.00241
- github: https://github.com/zjhuang22/maskscoring_rcnn

![@Mask Scoring RCNN的架构图](https://cwlseu.github.io/images/detection/MSRCNN.png)

# One-stage方法

以R-CNN算法为代表的two stage的方法由于RPN结构的存在，虽然检测精度越来越高，但是其速度却遇到瓶颈，比较难于满足部分场景实时性的需求。
因此出现一种基于回归方法的one stage的目标检测算法，不同于two stage的方法的分步训练共享检测结果，one stage的方法能实现完整单次
训练共享特征，且在保证一定准确率的前提下，速度得到极大提升。

### SSD原理与实现

https://blog.csdn.net/u010712012/article/details/86555814
https://github.com/amdegroot/ssd.pytorch
http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf

## CornerNet  人体姿态检测

- paper出处：https://arxiv.org/abs/1808.01244
- https://zhuanlan.zhihu.com/p/46505759

## RPN中的Anchor

Anchor是RPN网络的核心。需要确定每个滑窗中心对应感受野内存在目标与否。由于目标大小和长宽比例不一，需要多个尺度的窗。Anchor即给出一个基准窗大小，按照倍数和长宽比例得到不同大小的窗。有了Anchor之后，才能通过Select Search的方法\Slide Windows方法进行选取ROI的。

首先我们需要知道anchor的本质是什么，本质是SPP(spatial pyramid pooling)思想的逆向。而SPP本身是做什么的呢，就是将不同尺寸的输入resize成为相同尺寸的输出。所以SPP的逆向就是，将相同尺寸的输出，倒推得到不同尺寸的输入。

接下来是anchor的窗口尺寸，这个不难理解，三个面积尺寸（128^2，256^2，512^2），然后在每个面积尺寸下，取三种不同的长宽比例（1:1,1:2,2:1）.这样一来，我们得到了一共9种面积尺寸各异的anchor。示意图如下：
![@9个Anchor示意图](https://cwlseu.github.io/images/detection/Anchor.png)
至于这个anchor到底是怎么用的，这个是理解整个问题的关键。

* Faster RCNN
* SSD 
* YOLO
* Guided Anchor: https://arxiv.org/abs/1901.03278

## 目标检测算法研究问题小结

目标检测领域的深度学习算法，需要进行目标定位和物体识别，算法相对来说还是很复杂的。当前各种新算法也是层不出穷，但模型之间有很强的延续性，大部分模型算法都是借鉴了前人的思想，站在巨人的肩膀上。我们需要知道经典模型的特点，这些tricks是为了解决什么问题，以及为什么解决了这些问题。这样才能举一反三，万变不离其宗。综合下来，目标检测领域主要的难点如下:

* 检测速度：实时性要求高，故网络结构不能太复杂，参数不能太多，卷积层次也不能太多。
* **位置准确率**：`(x y w h)`参数必须准确，也就是检测框大小尺寸要匹配，且重合度IOU要高。SSD和faster RCNN通过多个bounding box来优化这个问题
* **漏检率**：必须尽量检测出所有目标物体，特别是靠的近的物体和尺寸小的物体。SSD和faster RCNN通过多个bounding box来优化这个问题
* **物体宽高比例不常见**：SSD通过不同尺寸feature map，yoloV2通过不同尺寸输入图片，来优化这个问题。
* 靠的近的物体准确率低
* 小尺寸物体准确率低：SSD取消全连接层，yoloV2增加pass through layer，采用高分辨率输入图片，来优化这个问题

# 目标检测特殊层

## ROIpooling

ROIs Pooling顾名思义，是Pooling层的一种，而且是针对RoIs的Pooling，他的特点是输入特征图尺寸不固定，但是输出特征图尺寸固定；

> ROI是Region of Interest的简写，指的是在“特征图上的框”; 
> * 在Fast RCNN中， RoI是指Selective Search完成后得到的“候选框”在特征图上的映射，如下图中的红色框所示；
> * 在Faster RCNN中，候选框是经过RPN产生的，然后再把各个“候选框”映射到特征图上，得到RoIs。

![@](https://cwlseu.github.io/images/detection/ROIPooling.png)

参考faster rcnn中的ROI Pool层，功能是将不同size的ROI区域映射到固定大小的feature map上。

### 缺点：由于两次量化带来的误差；
* 将候选框边界量化为整数点坐标值。
* 将量化后的边界区域平均分割成$k\times k$个单元(bin),对每一个单元的边界进行量化。

### 案例说明

下面我们用直观的例子具体分析一下上述区域不匹配问题。如 图1 所示，这是一个Faster-RCNN检测框架。输入一张$800\times 800$的图片，图片上有一个$665\times 665$的包围框(框着一只狗)。图片经过主干网络提取特征后，特征图缩放步长（stride）为32。因此，图像和包围框的边长都是输入时的1/32。800正好可以被32整除变为25。但665除以32以后得到20.78，带有小数，于是ROI Pooling 直接将它量化成20。接下来需要把框内的特征池化$7\times7$的大小，因此将上述包围框平均分割成$7\times7$个矩形区域。显然，每个矩形区域的边长为2.86，又含有小数。于是ROI Pooling 再次把它量化到2。经过这两次量化，候选区域已经出现了较明显的偏差（如图中绿色部分所示）。更重要的是，该层特征图上0.1个像素的偏差，缩放到原图就是3.2个像素。那么0.8的偏差，在原图上就是接近30个像素点的差别，这一差别不容小觑。

[`caffe中实现roi_pooling_layer.cpp`](https://github.com/ShaoqingRen/caffe/blob/062f2431162165c658a42d717baf8b74918aa18e/src/caffe/layers/roi_pooling_layer.cpp)

```cpp
template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //输入有两部分组成，data和rois
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // ROIs的个数
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    // 把原图的坐标映射到feature map上面
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    // 计算每个roi在feature map上面的大小
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    //pooling之后的feature map的一个值对应于pooling之前的feature map上的大小
    //注：由于roi的大小不一致，所以每次都需要计算一次
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);
    //找到对应的roi的feature map，如果input data的batch size为1
    //那么roi_batch_ind=0
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    //pooling的过程是针对每一个channel的，所以需要循环遍历
    for (int c = 0; c < channels_; ++c) {
      //计算output的每一个值，所以需要遍历一遍output，然后求出所有值
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          // 计算output上的一点对应于input上面区域的大小[hstart, wstart, hend, wend]
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));
          //将映射后的区域平动到对应的位置[hstart, wstart, hend, wend]
          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);
          //如果映射后的矩形框不符合
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          //pool_index指的是此时计算的output的值对应于output的位置
          const int pool_index = ph * pooled_width_ + pw;
          //如果矩形不符合，此处output的值设为0，此处的对应于输入区域的最大值为-1
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }
          //遍历output的值对应于input的区域块
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
             // 对应于input上的位置
              const int index = h * width_ + w;
              //计算区域块的最大值，保存在output对应的位置上
              //同时记录最大值的索引
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}
```

## ROI Align

![@ROIAlign模块使用示意图](https://cwlseu.github.io/images/detection/ROIAlign-1.png)

为了解决ROI Pooling的上述缺点，作者提出了ROI Align这一改进的方法。ROI Align的思路很简单：取消量化操作，使用双线性内插的方法获得坐标为浮点数的像素点上的图像数值,从而将整个特征聚集过程转化为一个连续的操作。值得注意的是，在具体的算法操作上，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后将这些坐标点进行池化，而是重新设计了一套比较优雅的流程，如下图所示：
![@浮点坐标计算过程](https://cwlseu.github.io/images/detection/ROIAlign-2.png)
* 遍历每一个候选区域，保持浮点数边界不做量化。
* 将候选区域分割成$k\times k$个单元，每个单元的边界也不做量化。
* 在每个单元中计算固定四个坐标位置，用双线性内插的方法计算出这四个位置的值，然后进行最大池化操作。

这里对上述步骤的第三点作一些说明：这个固定位置是指在每一个矩形单元（bin）中按照固定规则确定的位置。比如，如果采样点数是1，那么就是这个单元的中心点。如果采样点数是4，那么就是把这个单元平均分割成四个小方块以后它们分别的中心点。显然这些采样点的坐标通常是浮点数，所以需要使用插值的方法得到它的像素值。在相关实验中，作者发现将采样点设为4会获得最佳性能，甚至直接设为1在性能上也相差无几。
事实上，ROIAlign在遍历取样点的数量上没有ROIPooling那么多，但却可以获得更好的性能，这主要归功于解决了**misalignment**的问题。值得一提的是，我在实验时发现，ROIAlign在`VOC2007`数据集上的提升效果并不如在`COCO`上明显。经过分析，造成这种区别的原因是`COCO`上小目标的数量更多，而小目标受**misalignment**问题的影响更大（比如，同样是0.5个像素点的偏差，对于较大的目标而言显得微不足道，但是对于小目标，误差的影响就要高很多）。ROIAlign层要将feature map固定为2*2大小，那些蓝色的点即为采样点，然后每个bin中有4个采样点，则这四个采样点经过MAX得到ROI output；

> 通过双线性插值避免了量化操作，保存了原始ROI的空间分布，有效避免了误差的产生；小目标效果比较好

## NMS算法优化的必要性

### NMS算法的功能

非极大值抑制（NMS）非极大值抑制顾名思义就是抑制不是极大值的元素，搜索局部的极大值。例如在对象检测中，滑动窗口经提取特征，经分类器分类识别后，每个窗口都会得到一个分类及分数。但是滑动窗口会导致很多窗口与其他窗口存在包含或者大部分交叉的情况。这时就需要用到NMS来选取那些邻域里分数最高（是某类对象的概率最大），并且抑制那些分数低的窗口。印象最为深刻的就是Overfeat算法中的狗熊抓鱼图了。

### 从R-CNN到SPPNet

$RCNN$主要作用就是用于物体检测，就是首先通过$selective search$选择$2000$个候选区域，这些区域中有我们需要的所对应的物体的bounding-box，然后对于每一个region proposal都wrap到固定的大小的scale, $227\times227$(AlexNet Input),对于每一个处理之后的图片，把他都放到CNN上去进行特征提取，得到每个region proposal的feature map,这些特征用固定长度的特征集合feature vector来表示。
最后对于每一个类别，我们都会得到很多的feature vector，然后把这些特征向量直接放到SVM现行分类器去判断，当前region所对应的实物是background还是所对应的物体类别，每个region都会给出所对应的score，因为有些时候并不是说这些region中所包含的实物就一点都不存在，有些包含的多有些包含的少，包含的多少还需要合适的bounding box，所以我们才会对于每一region给出包含实物类别多少的分数，选出前几个对大数值，然后再用非极大值抑制canny来进行边缘检测，最后就会得到所对应的bounding box.

![Alt text](https://cwlseu.github.io/images/detection/SPPNet.png)
同样，SPPNet作者观察得，对selective search(ss)提供的2000多个候选区域都逐一进行卷积处理，势必会耗费大量的时间，
所以SPPNet中先对一整张图进行卷积得到特征图，然后再将ss算法提供的2000多个候选区域的位置记录下来，通过比例映射到整张图的feature map上提取出候选区域的特征图B,然后将B送入到金字塔池化层中，进行权重计算. 然后经过尝试，这种方法是可行的，于是在RCNN基础上，进行了这两个优化得到了这个新的网络SPPNet.

####  Faster RCNN

NMS算法，非极大值抑制算法，引入NMS算法的目的在于：根据事先提供的score向量，以及regions(由不同的bounding boxes，矩形窗口左上和右下点的坐标构成) 的坐标信息，从中筛选出置信度较高的bounding boxes。

![@FasterRCNN中的NMS的作用](https://cwlseu.github.io/images/detection/FasterRCNN_NMS.jpeg)

![@FasterRCNN中anchor推荐框的个数](https://cwlseu.github.io/images/detection/FasterRCNN_anchor.jpeg)
Faster RCNN中输入s=600时，采用了三个尺度的anchor进行推荐，分别时128,256和512，其中推荐的框的个数为$1106786$，需要将这$1100k$的推荐框合并为$2k$个。这个过程其实正是$RPN$神经网络模型。

### SSD

https://blog.csdn.net/wfei101/article/details/78176322
SSD算法中是分为default box(下图中(b)中为default box示意图)和prior box(实际推荐的框)
![@SSD算法中的anchor box和default box示意图](https://cwlseu.github.io/images/detection/SSD-1.png)

![@SSD算法架构图](https://cwlseu.github.io/images/detection/SSD-2.png)

![SSD算法推荐框的个数](https://cwlseu.github.io/images/detection/SSD-3.png)

### 注意

在图像处理领域，几点经验：
1. 输入的图像越大，结果越准确，但是计算量也更多
2. 推荐的框越多，定位准确的概率更高，但是计算量也是会增多
3. 推荐的框往往远大于最终的定位的个数

那么NMS存在什么问题呢，其中最主要的问题有这么几个：
- 物体重叠：如下面第一张图，会有一个最高分数的框，如果使用nms的话就会把其他置信度稍低，但是表示另一个物体的预测框删掉（由于和最高置信度的框overlap过大）
- 某些情况下，所有的bbox都预测不准，对于下面第二张图我们看到，不是所有的框都那么精准，有时甚至会出现某个物体周围的所有框都标出来了，但是都不准的情况
- 传统的NMS方法是基于分类分数的，只有最高分数的预测框能留下来，但是大多数情况下IoU和分类分数不是强相关，很多分类标签置信度高的框都位置都不是很准

### 参考文献

1. [NMS的解释](https://www.cnblogs.com/makefile/p/nms.html)
2. [附录中ROI的解释](http://www.cnblogs.com/rocbomb/p/4428946.html)
3. [SSD算法](https://blog.csdn.net/u013989576/article/details/73439202/)
4. [One-Stage Detector, With Focal Loss and RetinaNet Using ResNet+FPN, Surpass the Accuracy of Two-Stage Detectors, Faster R-CNN](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4)
5. [非极大值抑制算法的两个改进算法 & 传统NMS的问题](https://blog.csdn.net/lcczzu/article/details/86518615)
6. [非极大值抑制算法（NMS）与源码分析](http://blog.prince2015.club/2018/07/23/NMS/)

## one-stage和two-stage的anchor-base detection

它们的主要区别
* one-stage网络速度要快很多
* one-stage网络的准确性要比two-stage网络要低

### 为什么one-stage网络速度要快很多？

首先来看第一点这个好理解，one-stage网络生成的anchor框只是一个逻辑结构，或者只是一个数据块，只需要对这个数据块进行分类和回归就可以，不会像two-stage网络那样，生成的 anchor框会映射到feature map的区域（rcnn除外），然后将该区域重新输入到全连接层进行分类和回归，每个anchor映射的区域都要进行这样的分类和回归，所以它非常耗时

### 为什么one-stage网络的准确性要比two-stage网络要低？

我们来看RCNN，它是首先在原图上生成若干个候选区域，这个候选区域表示可能会是目标的候选区域，注意，这样的候选区域肯定不会特别多，假如我一张图像是$100\times100$的，它可能会生成`2000`个候选框，然后再把这些候选框送到分类和回归网络中进行分类和回归，Fast R-CNN其实差不多，只不过它不是最开始将原图的这些候选区域送到网络中，而是在最后一个feature map将这个候选区域提出来，进行分类和回归，它可能最终进行分类和回归的候选区域也只有`2000`多个并不多。再来看Faster R-CNN，虽然Faster R-CNN它最终一个feature map它是每个像素点产生9个anchor，那么$100\times100$假如到最终的feature map变成了$26\times26$了，那么生成的anchor就是$$26\times 26 \times 9 = 6084$$个，虽然看似很多，但是其实它在RPN网络结束后，它会不断的筛选留下`2000`多个，然后再从`2000`多个中筛选留下`300`多个，然后再将这`300`多个候选区域送到最终的分类和回归网络中进行训练，所以不管是R-CNN还是Fast-RCNN还是Faster-RCNN，它们最终进行训练的anchor其实并不多，几百到几千，不会存在特别严重的正负样本不均衡问题.
但是我们再来看yolo系列网络，就拿yolo3来说吧，它有三种尺度，$13\times 13$，$26\times 26$，$52\times 52$，每种尺度的每个像素点生成三种anchor，那么它最终生成的anchor数目就是
$$(13\times 13+26\times 26+52\times52)\times 3 = 10647$$个anchor，而真正负责预测的可能每种尺度的就那么几个，假如一张图片有3个目标，那么每种尺度有三个anchor负责预测，那么10647个anchor中总共也只有9个anchor负责预测，也就是正样本，其余的10638个anchor都是背景anchor，这存在一个严重的正负样本失衡问题，虽然位置损失，类别损失，这10638个anchor不需要参与，但是目标置信度损失，背景anchor参与了，因为

$$总的损失 = 位置损失 + 目标置信度损失 + 类别损失$$

所以背景anchor对总的损失有了很大的贡献，但是我们其实不希望这样的，我们更希望的是非背景的anchor对总的损失贡献大一些，这样不利于正常负责预测anchor的学习，而two-stage网络就不存在这样的问题，two-stage网络最终参与训练的或者计算损失的也只有`2000`个或者`300`个，它不会有多大的样本不均衡问题，不管是正样本还是负样本对损失的贡献几乎都差不多，所以网络会更有利于负责预测anchor的学习，所以它最终的准确性肯定要高些

> 总结

one-stage网络最终学习的anchor有很多，但是只有少数anchor对最终网络的学习是有利的，而大部分anchor对最终网络的学习都是不利的，这部分的anchor很大程度上影响了整个网络的学习，拉低了整体的准确率；而two-stage网络最终学习的anchor虽然不多，但是背景anchor也就是对网络学习不利的anchor也不会特别多，它虽然也能影响整体的准确率，但是肯定没有one-stage影响得那么严重，所以它的准确率比one-stage肯定要高。

### 那么什么情况下背景anchor不会拉低这个准确率呢？

设置阀值，与真实GrundTruth IOU阀值设得小一点，只要大于这个阀值，就认为你是非背景anchor（注意这部分anchor只负责计算目标置信度损失，而位置、类别损失仍然还是那几个负责预测的anchor来负责）或者假如一个图片上有非常多的位置都是目标，这样很多anchor都不是背景anchor；总之保证背景anchor和非背景anchor比例差不多，那样可能就不会拉低这个准确率，但是只要它们比例相差比较大，那么就会拉低这个准确率，只是不同的比例，拉低的程度不同而已

### 解决one-stage网络背景anchor过多导致的不均衡问题方案

* 采用focal loss，将目标置信度这部分的损失换成focal loss
* 增大非背景anchor的数量

某个像素点生成的三个anchor，与真实GrundTruth重合最大那个负责预测，它负责计算位置损失、目标置信度损失、类别损失，这些不管，它还有另外两个anchor，虽然另外两个anchor不是与真实GrundTruth重合最大，但是只要重合大于某个阀值比如大于`0.7`，我就认为它是非背景anchor，但注意它只计算目标置信度损失，位置和类别损失不参与计算，而小于`0.3`的我直接不让它参与目标置信度损失的计算，实现也就是将它的权重置0，这个思想就类似two-stage网络那个筛选机制，从`2000`多个anchor中筛选`300`个参与训练或者计算目标置信度损失，相当于我把小于`0.3`的anchor我都筛选掉了，让它不参与损失计算

* 设置权重
在目标置信度损失计算时，将背景anchor的权重设置得很小，非背景anchor的权重设置得很大。

### 四步交替训练Faster RCNN

[训练RPN网络](https://zhuanlan.zhihu.com/p/34327246)

Faster RCNN有两种训练方式，一种是四步交替训练法，一种是end-to-end训练法。主文件位于/tools/train_fast_rcnn_alt_opt.py。

第一步，训练RPN，该网络用ImageNet预训练的模型初始化，并端到端微调，用于生成region proposal;

第二步，由imageNet model初始化，利用第一步的RPN生成的region proposals作为输入数据，训练Fast R-CNN一个单独的检测网络，这时候两个网络还没有共享卷积层;

第三步，用第二步的fast-rcnn model初始化RPN再次进行训练，但固定共享的卷积层，并且只微调RPN独有的层，现在两个网络共享卷积层了;

第四步，由第三步的RPN model初始化fast-RCNN网络，输入数据为第三步生成的proposals。保持共享的卷积层固定，微调Fast R-CNN的fc层。这样，两个网络共享相同的卷积层，构成一个统一的网络。

## Faster-RCNN和YOLO的anchor有什么区别

![@FasterRCNN generator anchor](https://img-blog.csdnimg.cn/20190116235428577.jpg)

可以看到yolov3是直接对你的训练样本进行k-means聚类，由训练样本得来的先验框（anchor），也就是对样本聚类的结果。Kmeans因为初始点敏感，所以每次运行得到的anchor值不一样，但是对应的avg iou稳定。用于训练的话就需要统计多组anchor，针对固定的测试集比较了。

- https://blog.csdn.net/xiqi4145/article/details/86516511

- https://blog.csdn.net/cgt19910923/article/details/82154401