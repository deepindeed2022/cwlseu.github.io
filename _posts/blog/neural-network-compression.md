---
layout: post
title: 神经网络模型压缩与加速
tags: [计算机视觉, 神经网络压缩] 
categories: [blog ]
notebook: 视觉算法
---

## 引言

当前人工智能在图像、语音等领域取得的成功很大程度上归功于大型多层的深度神经网络模型。为了达到更好的效果或解决更复杂的问题，这些模型还在日益变大、变复杂。然而，在人工智能的多数应用领域，例如机器翻译、语音识别、自动驾驶等等，用户对人工智能系统的响应速度都非常敏感，有些甚至关乎生命安全。因此深度神经网络的低延迟推理是人工智能应用落地场景中一个非常关键的问题。

为了解决计算需求不断增长的问题，学术界和工业界都进行了大量的努力。一个直接的方法是通过定制化专用计算硬件（例如GPU、FPGA、ASIC），通过对深度神经网络进行领域特定体系结构设计（domain specific architecture design）提升神经网络的计算效率和速度，例如GPU中的Tensor Cores和 TPU中基于脉动阵列的矩阵乘法单元。然而，这种方法仍然满足不了日益增长的需求，深度学习模型还在增大，同时对于很多部署在移动端的AI应用，因为受到功耗和电源供应的限制，移动端处理器计算力的增长更是相对缓慢。

## Network Slimming

- paper: https://arxiv.org/pdf/1709.00513.pdf
- github: https://github.com/mengrang/Slimming-pytorch

### BN原理回顾([^2],[^1],[^4])
机器学习领域有个很重要的假设：独立同分布假设，就是假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障。那BatchNorm的作用是什么呢？BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。
$$\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$$

要注意，这里$t$层某个神经元的$x^{(k)}$不是指原始输入，就是说不是$t-1$层每个神经元的输出，而是$t$层这个神经元的线性激活$x=WU+B$，这里的$U$才是$t-1$层神经元的输出。变换的意思是：某个神经元对应的原始的激活x通过减去mini-Batch内m个实例获得的m个激活x求得的均值$E(x)$并除以求得的方差$Var(x)$来进行转换。

经过这个变换后某个神经元的激活x形成了均值为0，方差为1的正态分布，目的是把值往后续要进行的非线性变换的线性区拉动，增大导数值，增强反向传播信息流动性，加快训练收敛速度。但是这样会导致网络表达能力下降，为了防止这一点，每个神经元增加两个调节参数（scale($\gamma^{(k)}$)和shift($\beta^{(k)}$)），这两个参数是通过训练来学习到的，用来对变换后的激活反变换，使得网络表达能力增强，即对变换后的激活进行如下的scale和shift操作，这其实是变换的反操作：

$$y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}$$

1. 利用BN中的缩放因子$\gamma$ 作为评价上层输出贡献大小（下层输入）的因子，即$\gamma$越小，所对应的神经元越不重要，就可以裁剪掉。

2. 利用L1正则化$\gamma$，使其稀疏化，这样就可以在训练过程中自动评价神经元贡献大小，为0的因子可以安全剪掉。

这个思路还是很巧妙。但是，这个方法工程上会有一个需要注意的地方。根据BN原理，需要训练的因子有两种，scale和shift。$\gamma$是相乘因子，之后还要加上一个shift，$\beta$.那么，$\gamma$很小的时候，beta是不是很大？

[^1]: https://cloud.tencent.com/developer/article/1157136 "深入理解Batch Normalization批标准化"
[^2]: https://www.cnblogs.com/eilearn/p/9780696.html "深度学习-BN的理解"
[^3]: https://cloud.tencent.com/developer/article/1157135 "数据降维的方法(PCA/LDAb/LLE)"
[^4]: https://arxiv.org/pdf/1502.03167.pdf

## SSS: 侧重于channel prune 乃至group/block等结构上稀疏化的方法[^10]

- paper: https://arxiv.org/abs/1707.01213
- github: https://github.com/huangzehao/sparse-structure-selection.git


[^10]: https://zhuanlan.zhihu.com/p/48269250 "模型压缩 | 结构性剪枝"

## 早先研究成果: 

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks[C].in Advances In Neural Information Processing Systems, 2012, pp. 1–9.


[2]	S. Ren, K. He, R. Girshick, and J. Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[C]. in Advances In Neural Information Processing Systems, 2015, pp. 1–10.


[3]	K. Chellapilla et al., High Performance Convolutional Neural Networks for Document Processing[C]. in Tenth International Workshop on Frontiers in Handwriting Recognition, 2006, pp. 1–7.

[4]	J. Pool and W. J. Dally, “Learning both Weights and Connections for Efficient Neural Networks[C]. NIPS, 2015, pp. 1–9.

[5]	D. C. Cireşan, U. Meier, J. Masci, L. M. Gambardella, and J. Schmidhuber, High-Performance Neural Networks for Visual Object Classification[C].  in Advances In Neural Information Processing Systems, 2011, p. 12.

[6]	G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, Improving neural networks by preventing co-adaptation of feature detectors[J]. Computer  Science, vol. 3, no. 4, pp. 212–223, 2012.

[7]	Y. Le Cun, J. S. Denker, S. A. Solla, and T. B. Laboratories, Optimal Brain Damage[C],in International Conference on Neural Information, vol. 2, no. 279, pp. 598–605, 1989.

[8]	M. Zeiler and R. Fergus, Regularization of Neural Networks using DropConnect[C].  in International Conference on Machine Learning (ICML), 2013, pp. 1058–1066.

[9]	Z. Li, B. Gong, and T. Yang, Improved Dropout for Shallow and Deep Learning[C].  in International Conference on Machine Learning (ICML) Workshop on Resource-Efficient Machine Learning, 2016, pp. 1–9.

[10]	V. Lebedev and V. Lempitsky, Fast ConvNets Using Group-wise Brain Damage[C]. in Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2554–2564.

[11]	H. Mao, S. Han, J. Pool, W. Li, and X. Liu,  Exploring the Regularity of Sparse Structure in Convolutional Neural Networks[C]. in the 31st Conference on Neural Information Processing Systems, 2017, pp. 1–10.

[12]	W. Chen, J. W. W. Edu, C. Cse, and W. Edu, Compressing Neural Networks with the Hashing Trick[J]. Computer Science, pp. 2285–2294, 2015.

[13]	X. Zhang, J. Zou, X. Ming, K. He, and J. Sun, Efficient and Accurate Approximations of Nonlinear Convolutional Networks[C]. in the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[14]	M. Denil, B. Shakibi, L. Dinh, M. Ranzato, and N. de Freitas, Predicting Parameters in Deep Learning[C]. in International Conference on Neural Information Processing Systems, 2013, pp. 2148–2156.

[15]	Y. Gong, L. Liu, M. Yang, and L. Bourdev, Compressing deep convolutional networks using vector quantization[J].  Computer Science, pp. 1–10, 2014.

[16]	J. Wu, C. Leng, Y. Wang, Q. Hu, and J. Cheng, Quantized Convolutional Neural Networks for Mobile Devices[C],  in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 4820–4828.

[17]	G. Hinton, O. Vinyals, and J. Dean, Distilling the Knowledge in a Neural Network[J], Comput. Sci., vol. 14, no. 7, pp. 38–39, 2015.

[18]	E. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus,  Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation[C]. in International Conference on Neural Information, 2014, pp. 1269–1277.

[19]	Z. Yang et al., Deep Fried Convnets[C]. in IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 1476–1483.

[20]	V. Lebedev, Y. Ganin, M. Rakhuba1, I. Oseledets et al, Speeding-Up Convolutional Neural Networks Using Fine-tuned CP-Decomposition[C]. in International Conference on Learning Representations, 2015, pp. 1–10.

[21]	M. Jaderberg, A. Vedaldi, and A. Zisserman, Speeding up Convolutional Neural Networks with Low Rank Expansions[J]. Computer Science, vol. 4, no. 4, pp. 1–7, 2014.

[22]	C. Szegedy et al., Going deeper with convolutions[C].in the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 1–9.

[23]	M. Lin, Q. Chen, and S. Yan, Network In Network[C]. in International Conference on Learning Representations, 2014, pp. 1–10.

[24]	K. He, X. Zhang, S. Ren, and J. Sun, Deep Residual Learning for Image Recognition[C].in Computer Vision and Pattern Recognition, 2015, vol. 7, no. 3, pp. 171–180.

[25]	K.-H. Kim, S. Hong, B. Roh, Y. Cheon, and M. Park, PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection[J/OL]. (2016-09-30)[2017-05-12]. https://arxiv.org/abs/1608.08021v3

[26]	S. Lin, R. Ji, X. Guo, and X. Li, Towards Convolutional Neural Networks Compression via Global Error Reconstruction[C].  Proc. 25th Int. Jt. Conf. Artif. Intell. (IJCAI 2016), pp. 1753–1759, 2016.

[27]	J. Redmon and A. Farhadi, YOLO9000: Better, Faster, Stronger[C].  in Computer Vision and Pattern Recognition (CVPR), 2016.

[28]	M. Courbariaux and Y. Bengio, BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1[OL], (2016-3-17)[2017-4-10]. https://arxiv.org/abs/1602.02830

[29]	M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks[J] Springer Int. Publ., pp. 525–542, 2016.

[30]	W. G. Zefan Li, Bingbing Ni, Wenjun Zhang et al, Performance Guaranteed Network Acceleration via High-Order Residual Quantization[C].  in International Conference on Computer Vision(ICCV), 2017, pp. 1–9.

[31]	K. Lin, H. Yang, J. Hsiao, and C. Chen,  Deep Learning of Binary Hash Codes for Fast Image Retrieval Large-scale Image Search Query[C]. in IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2015, pp. 27–35.

[32]	Y. Mu and Z. Liu, Deep Hashing: A Joint Approach for Image Signature Learning[J/OL].(2016-8-12)[2017-5-12]. https://arxiv.org/abs/1608.03658

[33]	S. Han, H. Mao, and W. J. Dally, Deep Compression - Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding[C].in International Conference on Learning Representations, 2016, pp. 1–13.

[34]	R. Girshick,  Fast R-CNN[C]. in the International Conference on Computer Vision, 2015, pp. 1–9.

[35]	H. Liu, R. Ji, Y. Wu, and G. Hua, Supervised Matrix Factorization for Cross-Modality Hashing[C]. in International Joint Conference on Artificail Intelligent(IJCAI), pp. 1767–1773, 2016.