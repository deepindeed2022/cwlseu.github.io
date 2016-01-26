---
layout: post
title: Digital Image Process
categories: [blog ]
tags: [matlab, DIP]
description: 学习数字图像处理过程中遇到的问题
---

#Matlab 中的图像处理函数
1. 
	`g = imfilter(f, w, filtering mode,boundary options, size options)`用于线性滤波时，要将图像首先转化为高精度，否则会导致图像灰度值溢出

		f = imread('fig.tif');  
		f = im2double(f);  
		w = ones(31);  
		gf =imfilter(f,w);  
		imshow(gf,[]);  

	![filter效果图](http://github.com/cwlseu/cwlseu.github.io/raw/master/img/blog/digital-img-process/1.jpg)
		
		f = imread('fig.tif');  
		%f = im2double(f);  %delete im2double
		w = ones(31);  
		gf =imfilter(f,w);  
		imshow(gf,[]); 

	![filter灰度值溢出后效果图](http://github.com/cwlseu/cwlseu.github.io/raw/master/img/blog/digital-img-process/2.jpg)

	当然，这个函数的参数还有很多，更多信息，请看博客(http://blog.sina.com.cn/s/blog_5d14765801014fi7.html)。
	如果自己想动手实现以下，可以参考[implementing-imfilter-in-matlab](http://stackoverflow.com/questions/10672184/implementing-imfilter-in-matlab)
2. 
	`C = conv2(A,B)` 计算数组A和B的卷积。如果一个数组描述了一个二维FIR滤波器，则另一个数组被二维滤波。当A的大小为[ma,na],B的大小为[mb,nb]时，C的大小为[ma+mb-1,mb+nb-1]。

#图像增强
##空间域图像增强
锐化空间滤波--基于laplace算子的二阶微分的图像增强
	$g(x,y) = f(x,y) - \delta^2f(x,y)$
* 明白laplace算子与掩模之间的关系
* 边缘增强
* 反锐化掩蔽与提升滤波处理，可以用来去除噪声

##频率域图像增强

* 在原始图像左边乘以(-1)^(x+y)；
* 计算离散傅里叶变换(DFT); 
* 用滤波器函数H(u, v)乘以F(u, v); 
* 计算傅里叶反变换; 
* 结果的实部再乘以(-1)^(x+y)

###傅里叶变换及其性质
