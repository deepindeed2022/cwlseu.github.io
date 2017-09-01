---
layout: post
title: 模式识别入门算法
categories: [blog ]
tags: [模式识别]
description: 模式识别
---



声明：本博客欢迎转发，但请保留原作者信息! 
作者: [cwlseu]
博客： [https://cwlseu.github.io/](https://cwlseu.github.io/)


## 概述

最近Deep Learning 很火爆，此前的各种OCR技术几乎不能够对自然场景下的文字进行识别的，自从有了DL，
* [微软OCR](http://www.csdn.net/article/2015-03-30/2824348)有了新的进展。
* 说道OCR,不得不提Google的[tesseract](http://code.google.com/p/tesseract-ocr/)，[tesseract](http://www.cs.cmu.edu/~antz/sarma_icse2009.pdf)据个人亲测，能够实现对**自然场景**下的文字获取，当然图片中文字的清晰程度对输出结果的影响还是很大大的，这方面可以研究一下。
* 除了OCR之外，科大讯飞是被人熟知是源于其**语音识别**，还有现在汽车的**自动驾驶技术**，甚至是自动挡车在某种程度上也算是一种经过学习之后的成果吧。

## 概念
**增广样本**  
![@对于原训练数据添加全部为1的一项，用来表示偏置的权重，从而将偏置和原始权重统一到一个矩阵乘法之中，这个预处理数据的过程叫数据的增广](../images/classifiction-pattern/1.jpg)

## 方法论

长久以来，人们对一件事情发生或不发生，只有固定的0和1，即要么发生，要么不发生，从来不会去考虑某件事情发生的概率有多大，不发生的概率又是多大。而且事情发生或不发生的概率虽然未知，但最起码是一个确定的值。比如如果问那时的人们一个问题：“有一个袋子，里面装着若干个白球和黑球，请问从袋子中取得白球的概率是多少？”他们会立马告诉你，取出白球的概率就是1/2，要么取到白球，要么取不到白球，即θ只能有一个值，而且不论你取了多少次，取得白球的概率$\theta$始终都是1/2，即不随观察结果X 的变化而变化。

这种频率派的观点长期统治着人们的观念，直到后来一个名叫 **Thomas Bayes** 的人物出现。

>		Thomas Bayes
>		贝叶斯(约1701-1761) Thomas Bayes，英国数学家。约1701年出生于伦敦，做过神甫。
>		1742年成为英国皇家学会会员。1761年4月7日逝世。贝叶斯在数学方面主要研究概率论。
>		他首先将归纳推理法用于概率论基础理论，并创立了贝叶斯统计理论，对于统计决策函数、
>		统计推断、统计的估算等做出了贡献。
>		贝叶斯所采用的许多术语被沿用至今。贝叶斯思想和方法对概率统计的发展产生了深远的
>		影响。今天，贝叶斯思想和方法在许多领域都获得了广泛的应用。


频率派把需要推断的参数θ看做是固定的未知常数，即概率虽然是未知的，但最起码是确定的一个值，同时，样本X 是随机的，所以频率派重点研究样本空间，大部分的概率计算都是针对样本X 的分布；
**最大似然估计(MLE)** 和 **最大后验估计(MAP)** 都是把待估计的参数看作一个拥有固定值的变量，只是取值未知。通常估计的方法都是找使得相应的函数最大时的参数；由于MAP相比于MLE会考虑先验分布的影响，所以MAP也会有**超参数**，它的超参数代表的是一种信念(belief)，会影响推断(inference)的结果。比如说抛硬币，如果我先假设是公平的硬币，这也是一种归纳偏置(bias)，那么最终推断的结果会受我们预先假设的影响。

## 贝叶斯决策论
"贝爷是站在食物链顶端的男人",可是这也不妨碍贝叶斯成为模式识别中的校长。著名的贝叶斯概率和全概率成为模式识别入门的法宝与门槛。有了这工具，模式识别不是问题；不理解这理论，就痛苦地哀嚎吧。
贝叶斯派既然把看做是一个随机变量，所以要计算的分布，便得事先知道的无条件分布，即在有样本之前（或观察到X之前），有着怎样的分布呢？
比如往台球桌上扔一个球，这个球落会落在何处呢？如果是不偏不倚的把球抛出去，那么此球落在台球桌上的任一位置都有着相同的机会，即球落在台球桌上某一位置的概率服从均匀分布。这种在实验之前定下的属于基本前提性质的分布称为先验分布，或的无条件分布。
贝叶斯派认为待估计的参数是随机变量，服从一定的分布，而样本X是固定的，由于样本是固定的，所以他们重点研究的是参数的分布。
贝叶斯及贝叶斯派思考问题的固定模式 `$先验分布（\pi(\theta） + 样本信息X  =  后验分布(\pi(\theta|x))$` 
上述思考模式意味着，新观察到的样本信息将修正人们以前对事物的认知。换言之，在得到新的样本信息之前，人们对的认知是先验分布，在得到新的样本信息后，人们对的认知为。

### 贝叶斯参数估计

### 最大似然估计
最大似然估计，只是一种概率论在统计学的应用，它是参数估计的方法之一。说的是已知某个随机样本满足某种概率分布，但是其中具体的参数不清楚，参数估计就是通过若干次试验，观察其结果，利用结果推出参数的大概值。最大似然估计是建立在这样的思想上：已知某个参数能使这个样本出现的概率最大，我们当然不会再去选择其他小概率的样本，所以干脆就把这个参数作为估计的真实值。
求最大似然函数估计值的一般步骤： 
* 写出似然函数
* 对似然函数取对数，并整理
* 求导数
* 解似然方程

满足KKT条件的凸优化问题常常使用拉格朗日算子转化为似然函数的极值问题。通过求解似然函数的极值点，从而求得最优解。

## 感知器
感知器是由美国计算机科学家罗森布拉特（F.Roseblatt）于1957年提出的。感知器可谓是最早的人工神经网络。单层感知器是一个具有一层神经元、采用阈值激活函数的前向网络。通过对网络权值的训练，可以使感知器对一组输人矢量的响应达到元素为0或1的目标输出，从而实现对输人矢量分类的目的。
### 模型
(http://blog.163.com/zzz216@yeah/blog/static/16255468420107875552606/)
### 局限性
由于感知器神经网络在结构和学习规则上的限制，其应用也有一定的局限性。
首先，感知器的输出只能取0或1。
其次，单层感知器只能对线性可分的向量集合进行分类。

## 线性不可分问题

### MSE最小平方误差准侧
可以参考（http://blog.csdn.net/xiaowei_cqu/article/details/9004193）
在线性不可分的情况下，不等式组不可能同时满足。一种直观的想法就是，希望求一个a*使被错分的样本尽可能少。这种方法通过求解线性不等式组来最小化错分样本数目，通常采用搜索算法求解。

为了避免求解不等式组，通常转化为方程组：$a^Ty_i = b_i >0, i = 1,2,..，N$
矩阵的形式为Ya = b，方程组的误差为$e = Ya - b$, 
可以求解方程组的最小平方误差求解，即：$a*: minJ_s(a)$
$$J_s(a) = ||Ya - b||^2 = \sum_{i=1}^N(a^Ty_i - b_i)^2$$

最小误差求解方法有两种，一种是基于矩阵理论求解伪逆矩阵,然后求得最佳位置；另一种就是基于梯度下降的方法，类似感知机一样进行单步修正.其中k为迭代次数

```python

def LMS():
	init a, b, criteion theta, delta, k = 0
	while delta_k*(b_k - a_t*y_k)*y_k > theta:
		k += 1
		a = a + delta_k*(b_k - a[t]*y_k)*y_k
	return a 
```

### Ho-Kashyap 算法
这是一种修改的MSE。MSE只能使 $||Ya-b||$极小。
如果训练样本恰好线性可分，那么存在$a*，b*$，满足$Ya*=b*>0$
如果我们知道b*，就能用MSE求解到一个分类向量，但是我们无法预知b*
所以，$$J(a,b)=||Ya-b||^2, b>0$$
用梯度下降求解，得到：$a(k)=inv(Y) b(k)$,  $inv(Y)$表示$Y$的伪逆

下面是Ho-Kashyap及其修改算法的实现，可以在type中选择

```matlab
	function [a, b, k] = HoKashyap(train_features, train_targets, eta, b_min, kmax)
	% Classify using the using the Ho-Kashyap algorithm
	% Inputs:
	% 	train_features: Train features
	%	train_targets: Train targets
	%	eta	: learning rate
	%	b_min : break condition
	%   kmax :  the max interation time
	% Outputs
	%   a : Classifier weights
	%   b : Margin
	%   k : iteration time
	[c, n]		   = size(train_features);
	train_features  = [train_features ; ones(1,n)];
	train_zero      = find(train_targets == 1);

	%Preprocessing (Needed so that b>0 for all features)
	processed_features = train_features;
	processed_features(:,train_zero) = -processed_features(:,train_zero);

	b = ones(1,n);
	Y = processed_features;
	a = pinv(Y')*b';
	k = 0;
	e = 1e3;
	while  (sum(abs(e) > b_min)>0) & (k < kmax) % threshold b_min, kmax
	    %k <- (k+1) mod n
	    k = k+1;
	    %e <- Ya - b
	    e = (Y' * a)' - b;
	    %e_plus <- 1/2(e+abs(e))
	    e_plus  = 0.5*(e + abs(e));
	    %b <- b + 2*eta*e_plus
	    b = b + 2*eta*e_plus;
	    a = pinv(Y')*b' ;
	    if sum(find(e_plus < 0)) > 0
	        disp('The train data cannot seperate');
	    end
	end
	end
```

#总结
总的来说，误差最小是模式分类的原则。在保证正确率的情况下，提高响应速度。当然，现在还是处于正确率不断提高的阶段吧。

必看论文：[https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)
