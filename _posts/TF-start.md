---
layout: post
title: "Tensorflow: 启动"
categories: [blog ]
tags: [Tensorflow, ]
description:  
---
声明：本博客欢迎转发，但请保留原作者信息!                                      
作者: [曹文龙]                                                                 
博客： <https://cwlseu.github.io/>                                             

[TOC]

## 安装与启动

### install prepare tools

`sudo apt-get install virtualenv`
`sudo apt-get install python-pip python-dev python-virtualenv`

### 启动tensorflow虚拟环境
1. 第一次使用，创建tensorflow
`virtualenv --system-site-packages ~/tensorflow`
2. 创建tensorflow virtualenv

```sh
source ~/tensorflow/bin/activate  # If using bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```
### 启动与关闭virtualenv
`source ~/tensorflow/bin/activate`
`deactivate`


## test using demo 
`python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'`

`python -m tensorflow.models.image.mnist.convolutional`

## 参考资料
[^Tensorflow install](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#overview)                                            
[^Tensorflow Models](https://github.com/tensorflow/models)                                                                   