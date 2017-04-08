---
layout: post
title: 图像处理基础
categories: [blog ]
tags: [数字图像处理]
description: 学习数字图像处理过程中遇到的问题
---
声明：本博客欢迎转发，但请保留原作者信息! 
作者: [Clython]
博客： [https://cwlseu.github.io/](https://cwlseu.github.io/)


# Matlab 中的图像处理函数
matlab中有toolbox是关于图像处理的，很容易上手。当然，要是opencv用的很熟悉的话，也可以使用opencv.

### 基本的图像处理
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
### 卷积的理解
	`C = conv2(A,B)` 计算数组A和B的卷积。如果一个数组描述了一个二维FIR滤波器，则另一个数组被二维滤波。当A的大小为[ma,na],B的大小为[mb,nb]时，C的大小为[ma+mb-1,mb+nb-1]。
但是要知道，在当前神经网络中的卷积和这里定义的卷积操作是不一样的。当前卷积神经网络中的卷积相当于掩码操作似的。数学上的卷积定义比这个复杂得多。
## 图像滤波
处理一幅图像使得结果比原图像更适合于某种特定的应用 ，是一种基于局部运算的图像处理方法
#### 方法：
- 空域滤波：直接对图像的像素进行的滤波运算
- 频域滤波：在傅立叶变换域的滤波运算
#### 目的
- 平滑：去噪
- 锐化：增强图像的边缘及灰度变化的部分（一般先去噪）

#### 噪声与图像的关系
- 加性噪声
- 乘性噪声
#### 常见噪声种类
- 高斯噪声
- 椒盐噪声

#图像增强

##空间域图像增强
锐化空间滤波--基于laplace算子的二阶微分的图像增强
	$g(x,y) = f(x,y) - \delta^2f(x,y)$
* 明白laplace算子与掩模之间的关系
* 边缘增强
* 反锐化掩蔽与提升滤波处理，可以用来去除噪声

##频率域图像增强

首先我们要弄清楚为什么要将图像从空间域变换到频率域。弄清楚这个问题，困扰了我好久。
首先，在频率域进行处理比较直观。因为通过图像的频率信息很容易获取边缘信息、图像边界等敏感信息。而且滤波器也比较好设计。因此我们常常通过对频率域进行滤波器设计，然后频率域转换到空间域进行滤波处理。此外，卷积定理可知，在空间域的卷积等于频率域的乘积。因此，可以通过进行傅里叶变换减少计算，提高计算效率。
我们在图像处理中，常常采取的一下步骤进行处理问题：
* 在原始图像左边乘以(-1)^(x+y)；
* 计算离散傅里叶变换(DFT); 
* 用滤波器函数H(u, v)乘以F(u, v); 
* 计算傅里叶反变换; 
* 结果的实部再乘以(-1)^(x+y)

## 傅里叶变换及其性质

## 滤波器的设计

## 小波变换
[定义参考wikipedia:https://en.wikipedia.org/wiki/Wavelet_transform](https://en.wikipedia.org/wiki/Wavelet_transform)
### Haar 小波

### Daubechies wavelets

### DCWT

# 形态学

形态学（Mathematical morphology） 是一门建立在格论和拓扑学基础之上的图像分析学科，是数学形态学图像处理的基本理论。其基本的运算包括：二值腐蚀和膨胀、二值开闭运算、骨架抽取、极限腐蚀、击中击不中变换、形态学梯度、Top-hat变换、颗粒分析、流域变换、灰值腐蚀和膨胀、灰值开闭运算、灰值形态学梯度等。

简单来讲，形态学操作就是基于形状的一系列图像处理操作。OpenCV为进行图像的形态学变换提供了快捷、方便的函数。最基本的形态学操作有二种，他们是：膨胀与腐蚀(Dilation与Erosion)。

膨胀与腐蚀能实现多种多样的功能，主要如下：
* 消除噪声
* 分割(isolate)出独立的图像元素，在图像中连接(join)相邻的元素。
* 寻找图像中的明显的极大值区域或极小值区域
* 求出图像的梯度


可以发现erode和dilate这两个函数内部就是调用了一下morphOp，只是他们调用morphOp时，第一个参数标识符不同，一个为MORPH_ERODE（腐蚀），一个为MORPH_DILATE（膨胀）。

morphOp函数的源码在…\opencv\sources\modules\imgproc\src\morph.cpp
```cpp
/* 
第一个参数，InputArray类型的src，输入图像，即源图像，填Mat类的对象即可。图像通道的数量可以是任意的，但图像深度应为CV_8U，CV_16U，CV_16S，CV_32F或 CV_64F其中之一。
第二个参数，OutputArray类型的dst，即目标图像，需要和源图片有一样的尺寸和类型。
第三个参数，InputArray类型的kernel，膨胀操作的核。若为NULL时，表示的是使用参考点位于中心3x3的核。
我们一般使用函数 getStructuringElement配合这个参数的使用。getStructuringElement函数会返回指定形状和尺寸的结构元素（内核矩阵）。
其中，getStructuringElement函数的第一个参数表示内核的形状，我们可以选择如下三种形状之一:
	矩形: MORPH_RECT
	交叉形: MORPH_CROSS
	椭圆形: MORPH_ELLIPSE
而getStructuringElement函数的第二和第三个参数分别是内核的尺寸以及锚点的位置。
第四个参数，Point类型的anchor，锚的位置，其有默认值（-1，-1），表示锚位于中心。
第五个参数，int类型的iterations，迭代使用erode（）函数的次数，默认值为1。
第六个参数，int类型的borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。
第七个参数，const Scalar&类型的borderValue，当边界为常数时的边界值，有默认值morphologyDefaultBorderValue()，一般我们不用去管他。需要用到它时，可以看官方文档中的createMorphologyFilter()函数得到更详细的解释。 
*/

void cv::dilate( InputArray src,OutputArray dst, InputArray kernel,
                 Point anchor, int iterations,
                 int borderType, constScalar& borderValue )
{
//调用morphOp函数，并设定标识符为MORPH_DILATE
   morphOp( MORPH_DILATE, src, dst, kernel, anchor, iterations, borderType,borderValue );
}
```

我们一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得getStructuringElement函数的返回值。对于锚点的位置，有默认值Point(-1,-1)，表示锚点位于中心。且需要注意，十字形的element形状唯一依赖于锚点的位置。而在其他情况下，锚点只是影响了形态学运算结果的偏移。

```cpp
#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
 
int main(  )
{
	//载入原图 
	Mat image = imread("1.jpg");
	//创建窗口 
	namedWindow("【原图】膨胀操作");
	namedWindow("【效果图】膨胀操作");
	//显示原图
	imshow("【原图】膨胀操作", image);
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	Mat out;
	//进行膨胀操作
	dilate(image,out, element);
	//显示效果图
	imshow("【效果图】膨胀操作", out);
	waitKey(0);
	return 0;
}
```

# 图像处理总结

图像处理是入门机器视觉的基础课程，虽然学起来好像很痛苦，但是后面学习各种机器视觉
的课程会有更深的体会，这门课程原来很基础的。

# 资料
1. [机器视觉库OpenCV:http://opencv.org/](http://opencv.org/)
2. [机器学习算法和图像处理算法库：http://dlib.net/](http://dlib.net/)
3. [微软机器视觉：ittp://cn.bing.com/academic/search?FORM=SAFOS&q=Machine%20vision](http://cn.bing.com/academic/search?FORM=SAFOS&q=Machine%20vision)
4. [特征检测算子：http://fourier.eng.hmc.edu/e161/lectures/gradient/gradient.html](http://fourier.eng.hmc.edu/e161/lectures/gradient/gradient.html)
