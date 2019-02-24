---
layout: post
title: "深度学习应用开发"
categories: [blog ]
tags: [深度学习]
description:  
---

{:toc}                                             


## install vim
sudo apt-get install vim  git

## tensorflow virtualenv
https://www.tensorflow.org/install/install_linux#InstallingVirtualenv

```sh
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --system-site-packages tensorflow
source ~/tensorflow/bin/activate
(tensorflow)$ pip install --upgrade tensorflow
```

## ML package

```sh
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
chmod +x Anaconda2-4.4.0-Linux-x86_64.sh
./Anaconda2-4.4.0-Linux-x86_64.sh
## 根据提示填写内容

## 添加内容到.bashrc
export PATH=/opt/anaconda2/bin:$PATH
export PYTHONPATH=/opt/anaconda2/lib/python2.7/site-packages:$PYTHONPATH

```
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


## Comparative Study of Deep Learning Software Frameworks（caffe、Neon、TensorFlow、Theano、Torch 之比较）
<https://arxiv.org/abs/1511.06435>
五个开源框架：caffe、Neon、TensorFlow、Theano、Torch 
比较的三方面：可扩展性（extensibility）、hardware utilization（硬件利用率）以及大家最关心的：速度（speed）

评估测试都是部署在单机上，对于多线程 CPU、GPU（Nvidia Titan X）都进行测试； 速度评估标准包括了梯度计算时间（gradient computation time）、前向传播时间（forward time）； 对于卷积网络，作者还对这几个深度框架支持的不同的卷积算法以及相应的性能表现做了实验；最后通过实验，作者发现 Theano、Torch是最具扩展性的深度学习框架； 在 CPU 上的测试性能来看，Torch 最优，其次是 Theano； 在 GPU 上的性能表现，对于大规模卷积以及全连接网络，还是 Torch 最优，其次是 Neon； 但是 Theano 在部署和训练 LSTM 网络中夺得拔筹； caffe 是最容易测试评估性能的标准深度学习框架； 最后，TensorFlow 与 Theano 有些相似，是比较灵活的框架，但是其性能表现，目前还跟上面的几个框架比不起来。

### DL-benchmarks

This is the companion code for DL benchmarking study reported in the paper *Comparative Study of Deep Learning Software Frameworks* by *Soheil Bahrampour, Naveen Ramakrishnan, Lukas Schott, and Mohak Shah*. The paper can be found here http://arxiv.org/abs/1511.06435. The code allows the users to reproduce and extend the results reported in the study. The code provides timings of forward run and forward+backward (gradient computation) run of several deep learning architecture using Caffe, Neon, TensorFlow, Theano, and Torch. The deep learning architectures used includes LeNet, AlexNet, LSTM, and a stacked AutoEncoder. Please cite the above paper when reporting, reproducing or extending the results.

### Updated results
Here you can find a set of new timings obtained using **cuDNNv4** on a **single M40 GPU** on the same experiments performed in the paper. The result are reported using **Caffe-Nvidia 0.14.5**, **Neon 1.5.4**, **Tensoflow 0.9.0rc0**, **Theano 0.8.2**, and **Torch7**. Note that Neon does not use cuDNN.

1) **LeNet** using batch size of 64 (Extension of Table 3 in the paper)

|   Setting  | Gradient (ms) | Forward (ms) |
|:----------:|:-------------:|:------------:|
| Caffe |     2.4      |   0.8        |
| Neon |     2.7      |   1.3        |
| Tensorflow |      2.7      |      0.8     |
|   Theano   |      **1.6**      |      0.6     |
|    Torch   |      1.8      |      **0.5**    |

2) **Alexnet** using batch size of 256 (Extension of Table 4 in the paper)

|   Setting  | Gradient (ms) | Forward (ms) |
|:----------:|:-------------:|:------------:|
| Caffe |      279.3     |      88.3     |
| Neon |      **247.0**     |     **84.2**     |
| Tensorflow |      276.6      |      91.1     |
|    Torch   |     408.8      |      98.8     |

3) **LSTM** using batch size of 16 (Extension of Table 6 in the paper)

|   Setting  | Gradient (ms) | Forward (ms) |
|:----------:|:-------------:|:------------:|
| Tensorflow |      85.4      |      37.1     |
|    Theano   |     **17.3**      |      **4.6**     |
|    Torch   |     93.2      |      29.8     |

4) **Stacked auto-encoder** with encoder dimensions of 400, 200, 100 using batch size of 64 (Extension of Table 5 in the paper)

|   Setting  | Gradient (ms) AE1 | Gradient (ms) AE2 | Gradient (ms) AE3 | Gradient (ms) Total pre-training | Gradient (ms) SE | Forward (ms) SE |
|:----------:|:-----------------:|:-----------------:|:-----------------:|:--------------------------------:|:----------------:|:---------------:|
| Caffe |       0.8        |    0.9       |      0.9         |          2.6        |       1.1        |       0.6       |
| Neon |    1.2         |   1.5      | 1.9             |    4.6          |  2.0            |       0.9      |
| Tensorflow |        0.7        |        0.6        |        0.6        |                1.9               |        1.2       |       0.4       |
|   Theano   |        0.6        |        0.4        |        0.3        |                **1.3**              |        **0.4**       |       **0.3**       |
|    Torch   |        0.5        |        0.5        |        0.5        |                1.5               |        0.6       |       **0.3**       |

5)  **Stacked auto-encoder** with encoder dimensions of 800, 1000, 2000 using batch size of 64 (Extension of Table 7 in the paper)

|   Setting  | Gradient (ms) AE1 | Gradient (ms) AE2 | Gradient (ms) AE3 | Gradient (ms) Total pre-training | Gradient (ms) SE | Forward (ms) SE |
|:----------:|:-----------------:|:-----------------:|:-----------------:|:--------------------------------:|:----------------:|:---------------:|
| Caffe |         0.9     |      1.2     |        1.7      |          3.8       |     1.9        |       0.9       |
| Neon |    1.2         |   1.6      | 2.3             |    5.1         |  2.0            |       1.0   
  |
| Tensorflow |        0.9        |        1.1        |        1.6        |                3.6               |        2.1       |       0.7       |
|   Theano   |        0.7        |        1.0        |        1.8        |                3.5               |        **1.2**       |       **0.6**       |
|    Torch   |        0.7        |        0.9        |        1.4        |                **3.0**               |        1.4       |      **0.6**       |


## 参考资料
1. [Tensorflow install](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#overview)                                            
2. [Tensorflow Models](https://github.com/tensorflow/models)                            