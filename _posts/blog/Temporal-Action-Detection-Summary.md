---
layout: post
title: 时序动作识别综述
tags: [视频, 动作识别] 
categories: [blog]
notebook: 视觉算法
---

# 时序动作检测

时序动作检测主要解决的是两个任务：`localization` + `recognization`
* where:什么时候发生动作，即开始和结束时间；
* what：每段动作是什么类别
一般把这个任务叫做Temporal Action Detection，有的直接叫Action Detection，还有叫Action Localization

# Metric
### average recall (AR):
Temporal Action Proposal任务不需要对活动分类，只需要找出proposals，所以判断找的temporal proposals全不全就可以测评方法好坏，常用average recall (AR) ，Average Recall vs. Average Number of Proposals per Video (AR-AN) 即曲线下的面积(ActivityNet Challenge 2017就用这个测评此项任务)。如下图：
这里写图片描述

### mean Average Precision (mAP) :

Temporal Action Detection(Localization)问题中最常用的评估指标。一般对tIOU=0.5的进行对比，`tIOU`是时间上的交并。

关于鲁棒性
      为了测试/表征算法的鲁棒性，仅仅有以上的两个简单参数怕是远远不够的。简单对OPE、TRE、SRE进行介绍。
OPE: one-pass evaluation;
TRE: temporal robustness evaluation (different start frame);
SRE:spatial robustness evaluation (four center shifts and four corner shifts).
      其中OPE即为普通方法，TRE、SRE为测试鲁棒性方法。另还有其它更加复杂的方法，如果感兴趣的话可以在参考文献中查找到。在这里就不做过多的描述了。


# 数据集


| dataset名称 | 大小 | State-of-Art | 描述 |
| :---------- | :--------  | :--------|:-------|
|THUMOS2014|||该数据集包括行为识别和时序行为检测两个任务，大多数论文都在此数据集评估。<br>训练集：UCF101数据集，101类动作，共13320段分割好的视频片段；<br>验证集：1010个未分割过的视频；其中200个视频有时序行为标注(3007个行为片 段，只有20类，可用于时序动作检测任务)<br>测试集：1574个未分割过的视频；其中213个视频有时序行为标注(3358个行为片段，只有20类，可用于时序动作检测任务)<br>|
|ActivityNet|||200类，每类100段未分割视频，平均每段视频发生1.54个行为，共648小时|
|MUTITHUMOS|||一个稠密、多类别、逐帧标注的视频数据集，包括30小时的400段视频，65个行为类别38,690个标注，平均每帧1.5个label，每个视频10.5个行为分类，算是加强版THUMOS|

# 经典算法总结

## 卡尔曼滤波
https://wenku.baidu.com/view/ace40c671ed9ad51f01df2d1.html

## KCF目标跟踪方法分析与总结

https://www.cnblogs.com/YiXiaoZhou/p/5925019.html

Structured SVM, Ridge Regression
## Structure SVM
https://www.ncbi.nlm.nih.gov/pubmed/28055842
https://blog.csdn.net/sgfmby1994/article/details/52370496

## MDNet
https://blog.csdn.net/sgfmby1994/article/details/79863613

如果刚开始看这方面，17工作直接看SSN（TAG找proposal）、R-C3D、CBR（TURN找proposal）就好了，找proposal方法简单看看TAG和TURN（网络其他部分不用看），github也有代码，对性能要求不高可以试试SSN（用到了光流），不然的话可以用一下R-C3D。
SSN代码：https://github.com/yjxiong/action-detection
CDC代码：https://github.com/ColumbiaDVMM/CDC
R-C3D代码：https://github.com/VisionLearningGroup/R-C3D
CBR代码：https://github.com/jiyanggao/CBR
Learning Latent Super-Events to Detect Multiple Activities in Videos
代码：https://github.com/piergiaj/super-events-cvpr18


| method | UCF101 | HMDB51 | Kinetics | Jester| ActivityNet|
| :----- | :---:  | :---:  | :---:    | :---: |:----:|
| TSM[^1] | 94.4/99.5|||||
| I3D pre-training|
|ARTNet with TSN|
| TRN |
| T3D |
| R(2+1)D(RGB)|
|S3D(RGB) |



[^1]: Temporal Shift Module for Efficient Video Understanding
    - paper: https://arxiv.org/abs/1811.08383
    - code: https://github.com/mit-han-lab/temporal-shift-module

# 参考文献

[^10]: https://blog.csdn.net/Miracle_520/article/details/84991358 "Temporal Action Detection (时序动作检测)综述"
https://blog.csdn.net/wzmsltw/article/details/70849132
https://blog.csdn.net/qq_41590635/article/details/101478277