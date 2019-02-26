---
layout: post
title: "DeepLearning资料总结"
categories: [blog ]
tags: [深度学习]
description: 学习深度学习中的一些有意思的东西
comments: true
---

* content
{:toc}

## 什么是卷积
* [A Comprehensive Introduction to Different Types of Convolutions in Deep Learning](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)
* [A Tutorial on Filter Groups (Grouped Convolution)](https://blog.yani.io/filter-group-tutorial/)
  * AlexNet
  * MobileNet
  * 就是mxNet中的
* [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

| Convolution Name | 参考文献 | 典型代表 | 附录 |
| :-------------- | :--------  | :------------:| :------------:|
| 1x1 Convolution|  |  GoogLeNet, Inception|                   |
| Group Convolution| [Deep Roots:Improving CNN Efficiency with Hierarchical Filter Groups](https://arxiv.org/pdf/1605.06489.pdf) |  [MobileNet](), [AlexNet]()  |


## 优化
* [梯度下降(gradient descent)](https://www.quora.com/What-is-the-purpose-for-the-use-of-gradient-descent-in-machine-learning?__filter__=&__nsrc__=2&__snid3__=2889908801&redirected_qid=31223828)
* [梯度下降优化算法](http://ruder.io/optimizing-gradient-descent/)

## 深度学习教程
[CS231n: Convolutional Neural Networks for Visual Recognition.](https://cs231n.github.io/)

## 计算平台

1. [arm平台](https://en.wikipedia.org/wiki/ARM_architecture)
2. [linux上编译arm交叉编译链](https://www.acmesystems.it/arm9_toolchain)
3. [How to Build a GCC Cross-Compiler](http://preshing.com/20141119/how-to-build-a-gcc-cross-compiler/)

## 数据集合

https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/
这里我们列出了一组高质量的数据集，研究这些数据集将使你成为一个更好的数据科学家。
我们可以使用这些数据集来学习各种深度学习技术，也可以使用它们来磨练您的技能，理解如何识别和构造每个问题，考虑独特的应用场景!

### 图像类

| dataset名称 | 大小 | State-of-Art | 描述 |
| :-------------- | :--------  | :--------------------------:| :------------: |
|[MNIST](http://yann.lecun.com/exdb/mnist/)|50MB|[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)| 手写数字识别，包含60000个训练数据及10000个测试数据，可分为10类|
|[MSCOCO](http://cocodataset.org/#home) |~25G |[Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf)|COCO is a large-scale and rich for object detection, segmentation and captioning dataset. 330K images, 1.5 million object instances, 80 object categories, 5 captions per image, 250,000 people with key points|
|[ImageNet](http://www.image-net.org/)|150GB|[ResNeXt](https://arxiv.org/pdf/1611.05431.pdf)|ImageNet is a dataset of images that are organized according to the WordNet hierarchy. WordNet contains approximately 100,000 phrases and ImageNet has provided around 1000 images on average to illustrate each phrase. Number of Records: Total number of images: ~1,500,000; each with multiple bounding boxes and respective class labels|
|[Open Image Dataset](https://github.com/openimages/dataset#download-the-data)|500GB|[ResNet]()|一个包含近900万个图像URL的数据集。 这些图像拥有数千个类别及边框进行了注释。 该数据集包含9,011219张图像的训练集，41,260张图像的验证集以及125,436张图像的测试集。|
|[VisualQA](http://www.visualqa.org/)| 25GB |[Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge](https://arxiv.org/abs/1708.02711) |图像的问答系统数据集 265,016 images, at least 3 questions per image, 10 ground truth answers per question|
|[The Street View House Numbers(SVHN)](http://ufldl.stanford.edu/housenumbers/)| 2.5GB | [Distributional Smoothing With Virtual Adversarial Training]() | 门牌号数据集，可用来做物体检测与识别 |
|[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)| 170MB |[ShakeDrop regularization](https://openreview.net/pdf?id=S1NHaMW0b)|图像识别数据集，包含 50000张训练数据，10000张测试数据，可分为10类|
|[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)|30MB|[Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)|包含60000训练样本和10000测试样本的用于服饰识别的数据集，可分为10类。|

### 自然语言处理类

| dataset名称 | 大小 | State-of-Art | 描述 | 
| :-------------- | :--------  | :--------------------------:| :------------|
|[IMDB 影评数据](http://ai.stanford.edu/~amaas/data/sentiment/)| 80MB |[Learning Structured Text Representations](https://arxiv.org/abs/1705.09207)|可以实现对情感的分类，除了训练集和测试集示例之外，还有更多未标记的数据。原始文本和预处理的数据也包括在内。25,000 highly polar movie reviews for training, and 25,000 for testing|
|[Twenty Newsgroups](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) | 20MB |[Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781) | 包含20类新闻的文章信息，内类包含1000条数据|
|[Sentiment140](http://help.sentiment140.com/for-students/)| 80MB |[Assessing State-of-the-Art Sentiment Models on State-of-the-Art Sentiment Datasets](http://www.aclweb.org/anthology/W17-5202)|1,60,000 tweets,用于情感分析的数据集|
|[WordNet](https://wordnet.princeton.edu/)|10MB| [Wordnets: State of the Art and Perspectives](https://aclanthology.info/pdf/R/R11/R11-1097.pdf)|117,000 synsets is linked to other synsets by means of a small number of “conceptual relations.|
|[Yelp点评数据集](https://www.yelp.com/dataset)|2.66GB JSON文件,2.9GB SQL文件,7.5GB图片数据| [Attentive Convolution](https://arxiv.org/pdf/1710.00519.pdf) | 包括470万条用户评价，15多万条商户信息，20万张图片，12个大都市。此外，还涵盖110万用户的100万条tips，超过120万条商家属性（如营业时间、是否有停车场、是否可预订和环境等信息），随着时间推移在每家商户签到的总用户数。|
|[维基百科语料库（英语）](http://nlp.cs.nyu.edu/wikipedia-data/) | 20MB | [Breaking The Softmax Bottelneck: A High-Rank RNN language Model](https://arxiv.org/pdf/1711.03953.pdf) | 包含4400000篇文章 及19亿单词，可用来做语言建模|
|[博客作者身份语料库](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm)| 300MB | [Character-level and Multi-channel Convolutional Neural Networks for Large-scale Authorship Attribution](https://arxiv.org/pdf/1609.06686.pdf) | 从blogger.com收集到的19,320名博主的博客，其中博主的信息包括博主的ID、性别、年龄、行业及星座|
|[各种语言的机器翻译数据集](http://statmt.org/wmt18/index.html)|15GB |[Attention Is All You Need](https://arxiv.org/abs/1706.03762)|包含英-汉、英-法、英-捷克、英语- 爱沙尼亚、英 - 芬兰、英-德、英 - 哈萨克、英 - 俄、英 - 土耳其之间互译的数据集|

### 语音类

| dataset名称 | 大小 | State-of-Art | 描述 | 
| :-------------- | :--------  | :--------------------------:| :------------|
|[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)|10MB |[Raw Waveform-based Audio Classification Using Sample-level CNN Architectures](https://arxiv.org/pdf/1712.00866)|数字语音识别数据集，包含3个人的声音，每个数字说50遍，共1500条数据|
|[Free Music Archive (FMA)](https://github.com/mdeff/fma)| 1000GB | [Learning to Recognize Musical Genre from Audio](https://arxiv.org/pdf/1803.05337.pdf) | 可以用于对音乐进行分析的数据集，数据集中包含歌曲名称、音乐类型、曲目计数等信息。|
|[Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) | 14GB | [A Multi-Model Approach To Beat Tracking Considering Heterogeneous Music Styles]() | 舞厅舞曲数据集，可对舞曲风格进行识别。|
|[Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)| 280GB |[Preliminary Study on a Recommender System for the Million Songs Dataset Challenge]()|Echo Nest提供的一百万首歌曲的特征数据.该数据集不包含任何音频，但是可以使用他们提供的代码下载音频|
|[LibriSpeech](ttp://www.openslr.org/12/) | 60GB |[Letter-Based Speech Recognition with Gated ConvNets]()|包含1000小时采样频率为16Hz的英语语音数据及所对应的文本，可用作语音识别|
|[VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)|150MB|VoxCeleb: a large-scale speaker identification dataset]()|大型的说话人识别数据集。 它包含约1,200名来自YouTube视频的约10万个话语。 数据在性别是平衡的（男性占55％）。说话人跨越不同的口音，职业和年龄。 可用来对说话者的身份进行识别。|

### Analytics Vidhya实践问题
* [Twitter情绪分析](https://datahack.analyticsvidhya.com/contest/practice-problem-age-detection/register)
  * 描述：识别是否包含种族歧视及性别歧视的推文。
  * 大小：3MB
  * 31,962 tweets
* [印度演员的年龄识别数据集](https://datahack.analyticsvidhya.com/contest/practice-problem-age-detection/)
  * 描述：根据人的面部属性，识别人的年龄的数据集。
  * 大小：48MB
  * 19,906 images in the training set and 6636 in the test set
* [城市声音分类数据集](https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/)
  * 描述：该数据集包含来自10个类的城市声音的8732个标记的声音片段，每个片段时间小于4秒。
  * 大小：训练数据集3GB，训练数据集2GB。
  * 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes
## more dataset
[](https://github.com/Prasanna1991/DHCD_Dataset)