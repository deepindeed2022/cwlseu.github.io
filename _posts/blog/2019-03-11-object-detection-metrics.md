---
layout: post
comments: true
title: "Object Detection Metrics"
categories: [blog ]
tags: [detection,CV算法]
description: 物体检测算法的评估方法
---

* content
{:toc}

## 物体检测效果评估相关的定义 

### Intersection Over Union (IOU)
Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box  $B_{gt}$and a predicted bounding box $B_p$ By applying the IOU we can tell if a detection is valid (True Positive) or not (False Positive).  
IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between them:  
$$IOC = \frac{area of overlap}{area of union} = \frac{area(B_p \cap B_{gt})}{area(B_p \cup B_{gt})}$$

The image below illustrates the IOU between a ground truth bounding box (in green) and a detected bounding box (in red).

<p align="center">
<img src="https://cwlseu.github.io/images/detection/iou.png" align="center"/></p>

### True Positive, False Positive, False Negative and True Negative  

Some basic concepts used by the metrics:  

* **True Positive (TP)**: A correct detection. Detection with `IOU ≥ _threshold_`
* **False Positive (FP)**: A wrong detection. Detection with `IOU < _threshold_`
* **False Negative (FN)**: A ground truth not detected
* **True Negative (TN)**: Does not apply. It would represent a corrected misdetection. In the object detection task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were corrrectly not detected (so many possible boxes within an image). That's why it is not used by the metrics.

`_threshold_`: depending on the metric, it is usually set to 50%, 75% or 95%.

### Precision

Precision is the ability of a model to identify **only** the relevant objects. It is the percentage of correct positive predictions and is given by:
$$Precision = \frac{TP}{TP + FP} = \frac{TP}{all-detections}$$

### Recall 

Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:
$$Recall = \frac{TP}{TP + FN} = \frac{TP}{all-groundtruths}$$
![@混淆矩阵](http://cwlseu.github.io/images/detection/confusion-metrics.png)

## 评估方法Metrics

* Receiver operating characteristics (ROC) curve
* Precision x Recall curve
* Average Precision
  * 11-point interpolation
  * Interpolating all points

## 物体检测中的损失函数


## 参考文献

1. [评估标准](https://github.com/cwlseu/Object-Detection-Metrics)
2. [机器学习之分类性能度量指标 : ROC曲线、AUC值、正确率、召回率](https://www.jianshu.com/p/c61ae11cc5f6)
3. [How and When to Use ROC Curves and Precision-Recall Curves for Classification in Python](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
4. [Precision-recall curves – what are they and how are they used](https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used)
5. [精确率、召回率、F1 值、ROC、AUC 各自的优缺点是什么？](https://www.zhihu.com/question/30643044)
