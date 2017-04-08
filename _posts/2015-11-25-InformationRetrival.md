---
layout: post
title: IR的设计
categories: [blog ]
tags: [信息检索系统]
description: 信息检索系课程回顾与课程设计
---

## 课程概要

* 布尔查询
* 词项分割与去停用词
* 倒排索引的构建与词典
* 输入纠错与编辑距离
* 索引压缩
* 向量空间模型与tf-idf权重计算
* 检索系统的评价
* 检索模型的介绍
* Web采集与链接分析

## google的查询

## 课程设计

### 任务内容

#### Part 1:

基本要求：构建词典和倒排索引
* 实现 Single-pass In-memory Indexing
* 实现倒排索引的 Gamma 编码压缩 / 解压
* 实现词典的单一字符串形式压缩 / 解压，任意数据结构（如哈希表、 B 树等）
* 实现关键字的查找，命令行中 Print 给定关键字的倒排记录表
* 给出以下语料统计量：词项数量，文档数量，词条数量，文档平均长度（词条数量）
* 对停用词去除、词干还原等无要求，但应实现最基本的词条化功能 例如：将所有非字母和非数字字符转换为空格，不考虑纯数字词项

[Test Data](http://gucasir.org/ModernIR/shakespeare-merchant.trec.tgz)
解压命令： `tar zxvf shakespear-merchant.trec.tgz`

#### Part 2:

采用类似 TREC 竞赛的形式
* 以小组形式在给定数据上进行实验
* 鼓励创新思维
   – 评分：综合考虑实验结果和使用的新方法、提出的新思路

### 任务设计思路

第一部分是对我们课上学习内容的实现，通过实现任务一，对于加深对课程学习的立即是很有帮助的  
主要涉及:  
1. 词项归一化(选择哪种归一化方法？)与去停用词(哪些是停用词？)和词条化  
2. 字符串压缩与解码  
3. 倒排索引的构建  
4. 语料的读取与分词   
5. 统计分析词项数量与文档数量、文档长度(词条数目)，词条数量  

第二部分是research能力与开发能力并重的，可以使用开源软件提升检索能力，也可以自己实现检索器


### 实现
It's preferred to use Python and with python package unittest for unit test to imply the part one while C++ implement is optional.

### 总结

## 网络爬虫
### TODO表
使用Berkely DB实现TODO表
### 使用布隆过滤器构建Visited表
经常要判断一个元素是否在一个集合中。这个可以使用HashMap进行存储，速度匹配快。但是费存储空间，尤其是当集合巨大的时候，hash存储效率低的问题就会凸现出来，使得一般的服务器不能够承受巨大的存储。[布隆过滤器](http://www.cnblogs.com/KevinYang/archive/2009/02/01/1381803.html)是1970年由巴顿.布隆提出来的，实际上是一个二进制向量和一系列随机映射函数。