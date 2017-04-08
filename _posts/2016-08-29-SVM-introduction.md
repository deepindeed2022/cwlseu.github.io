---
layout: post
title: 支持向量机的学习与理解
categories: [blog ]
tags: [SVM]
description: 支持向量机，因其英文名为 Support Vector Machine，故一般简称 SVM，通俗来讲，它是一种二类分类模型，其基本模型定义为特征空间上的间隔最大的线 性分类器，其学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。
---
## SVM介绍
支持向量机，因其英文名为 Support Vector Machine，故一般简称 SVM，通俗来讲，它是一种二类分类模型，其基本模型定义为特征空间上的间隔最大的线 性分类器，其学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。

## SVM设计原理

>Train data:
$ (x_i, y_i), i = 1,...l,   where \quad x_i  \in  R^n  and \quad y \in \{1, -1\}^l $

> 优化目标函数
$$ min_{W,b,\epsilon} \frac{1}{2}W^tW + C \sum_{i = 1}^l{\epsilon_i} $$
$$s.t.  \quad y_i(W^T\phi(x_i) + b) \geq 1- \epsilon_i $$