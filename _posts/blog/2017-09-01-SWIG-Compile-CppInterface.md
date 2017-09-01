---
layout: post
title: "SWIG编译C++接口笔记"
categories: [blog ]
tags: [cmake, C++接口, SeetaFace]
description: SeetaFace人脸识别引擎提供了人脸识别系统所需的三个核心模块。为了使用方便，决定使用swig编译python接口进行使用。
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>
                                        
## 来源
SeetaFaceEngine使用C++编译，而且使用OpenMP技术和向量化技术进行加速，已经基本可以满足业界对人脸识别功能的需求。在项目中用到人脸识别
功能，OpenCV自带的基于Haar特征的算法，效果不理想，仅仅能够识别正脸，人脸歪一定的角度都不能够识别。使用SeetaFaceEngine需要重新编译python接口，对于没有接触过的人来说还真不简单，在此新路记录。
[SeetaFaceEngine源代码](https://github.com/seetaface/SeetaFaceEngine) 

## SWIG

## numpy.ndarray和cv::Mat

## 连接库

## 结果


## 参考文献