---
layout: post
title: Pattern Classification
categories: [blog ]
tags: [matlab, PR, 神经网络，模式识别]
description: 模式识别与深度学习
---
# 概述
最近Deep Learning 很火爆，此前的各种OCR技术几乎不能够对自然场景下的文字进行识别的，自从有了DL，
* [微软OCR](http://www.csdn.net/article/2015-03-30/2824348)有了新的进展。
* 说道OCR,不得不提Google的[tesseract](http://code.google.com/p/tesseract-ocr/)，[tesseract](http://www.cs.cmu.edu/~antz/sarma_icse2009.pdf)据个人亲测，能够实现对**自然场景**下的文字获取，当然图片中文字的清晰程度对输出结果的影响还是很大大的，这方面可以研究一下。
* 除了OCR之外，科大讯飞是被人熟知是源于其**语音识别**，还有现在汽车的**自动驾驶技术**，甚至是自动挡车在某种程度上也算是一种经过学习之后的成果吧。

# 概念
**增广样本**  
	![](http://github.com/cwlseu/cwlseu.github.io/raw/master/img/blog/classifiction-pattern/1.jpg)

# 方法论
## 贝叶斯决策论
   "贝爷是站在食物链顶端的男人",可是这也不妨碍贝叶斯成为模式识别中的校长。著名的贝叶斯概率和全概率成为模式识别入门的法宝与门槛。有了这工具，模式识别不是问题；不理解这理论，就痛苦地哀嚎吧。

## 贝叶斯参数估计

## 最大似然估计

## 感知器

### MSE最小平方误差准侧

### Ho-Kashyap 算法
	
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



#总结
总的来说，误差最小是模式分类的原则。在保证正确率的情况下，提高响应速度。当然，现在还是处于正确率不断提高的阶段吧。