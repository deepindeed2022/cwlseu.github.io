---
layout: post
title: "Generative Adeversarial Networks: Overview"
categories: [blog ]
tags: [深度学习]
description: 去年和今年最火的技术GANs，到底是什么东西呢。
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## GANs
### 定义

GANs是半监督和无监督中的新兴技术。通过竞争学习一对神经网络模型来实现对数据的高纬度表示。在其中通过
骗子（Genrator G）与专家（Discriminator D）之间的博弈，骗子想绘制出真品的来，专家来鉴定两个作品哪个真品哪个是赝品，差距在哪里。两个网络都是在学习的，就像骗子模仿技术越来越厉害，但是专家也是越来越强的，“魔高一尺，道高一丈”.

$G: G(z) -> R^{\|x\|}$

$D: D(x) -> (0, 1)$

### GANs的组成
1. Fully Connected GANs： G和D都是使用全连接神经网络

2. Convolutional GANs
特别适合与Image data的生成，但是使用相同表达能力的CNNs作为生成器和判别器，是很难训练的。其中LAP-GAN(Laplacian pyramid of adversarial networks)使用多尺度思想，将G的generation 过程分解为生成一个laplacian pyramid的过程.如果卷积网络使用deep convolution的话，即使DCGANs，通过利用stride和fractionally-strided convolutions在空间域中下采样和上采样操作。

## TODO

## 参考链接
