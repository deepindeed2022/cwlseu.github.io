---
layout: post
title: Caffe小白学习记
categories: [blog ]
tags: [Caffe, ]
description: 学习Caffe的过程中的一些趣事
---

## Ubuntu 14.04 && CUDA 7.5
打开安装了CUDA的ubuntu14.04发现，开机的过程中一直停止在开机等待界面，无法进入。

通过选择recovery mode进行恢复之后，然后重启，重启之后才能正常进入。然而，这不是一劳永逸的。等下一次再次开机重新进入的时候，又遇到了同样的问题，让我不得其解。

后来经过调研和重新格式化系统进行安装之后发现，原来是CUDA7.5 的.deb对Ubuntu 14.04 的支持性不好，导致显示驱动程序有问题，从而无法正常进入系统。而且有人建议采用.run的toolkit进行安装。可是又有新的问题出现。