---
layout: post
title: Digital Image Process
categories: [blog ]
tags: [matlab, DIP]
description: 学习数字图像处理过程中遇到的问题
---

##Matlab 中的图像处理函数
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

