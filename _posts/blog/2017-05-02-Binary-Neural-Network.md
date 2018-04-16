---
layout: post
title: 论文笔记：XNOR-Ne：
categories: [blog ]
tags: [论文 ]
description: "神经网络量化、二值化、设计package的神经网络模块"
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## 引言

面对神经网络over-parametrization和enormous parameters, 工作者在这者方面下了不少功夫。我们也想在这方面分的一杯羹, 当前主要的方向有：
- shallow network
- compressing pre-trained network
- designed compact layers：可以去看看[PVANet](http://cwlseu.github.io/PVANet)
- quantizing parameters
- network binarization

## 二值网络

## XNOR

## Binary-Weight-Network

## 相关文献
1. [Mohammad Rastegari Vicente Ordonez Joseph Redmon Ali Farhadi: XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks (2016)](https://arxiv.org/abs/1603.05279)
2. [Courbariaux, M., Bengio, Y., David, J.P.: Training deep neural networks with low precision multiplications. arXiv preprint arXiv:1412.7024 (2014) 4](https://arxiv.org/abs/1412.7024)
3. [Soudry, D., Hubara, I., Meir, R.: Expectation backpropagation: parameter-free training of multilayer neural networks with continuous or discrete weights. In: Advances in Neural Information Processing Systems. (2014) 963–971 4](http://papers.nips.cc/paper/5269-expectation-backpropagation-parameter-free-training-of-multilayer-neural-networks-with-continuous-or-discrete-weights.pdf)
4. [Esser, S.K., Appuswamy, R., Merolla, P., Arthur, J.V., Modha, D.S.: Backpropagation for energy-efficient neuromorphic computing. In: Advances in Neural Information Processing Systems. (2015) 1117–1125 4](https://papers.nips.cc/paper/5862-backpropagation-for-energy-efficient-neuromorphic-computing) 
5. [Courbariaux, M., Bengio, Y., David, J.P.: Binaryconnect: Training deep neural networks with binary weights during propagations. In: Advances in Neural Information Processing Systems. (2015) 3105–3113 4, 6, 10, 11](https://www.arxiv.org/abs/1511.00363) 
6. [Baldassi, C., Ingrosso, A., Lucibello, C., Saglietti, L., Zecchina, R.: Subdominant dense clusters allow for simple learning and high computational performance in neural networks with discrete synapses. Physical review letters 115(12) (2015) 128101 5](https://arxiv.org/abs/1509.05753v1) 
7. [Kim, M., Smaragdis, P.: Bitwise neural networks. arXiv preprint arXiv:1601.06071 (2016)](https://arxiv.org/abs/1601.06071)
