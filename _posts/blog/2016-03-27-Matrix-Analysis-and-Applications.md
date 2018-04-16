---
layout: post
title: 数学基础：Matrix analysis and application
categories: [blog ]
tags: [矩阵分析, ]
description: 矩阵分析、应用与实践；主讲教师：Baobin Li
---



- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [cwlseu]
- 博客： <https://cwlseu.github.io/>

## 引言

矩阵也许就是为了计算机而出现的一门学科吧。虽然以前大学学习矩阵，好像感觉没有什么用处，但是当接触到图像之后发现，数字图像全是矩阵，矩阵成为每天必须面对的话题。
不管是特征提取还是聚类分析，矩阵的特征值在矩阵的作用尤为突出，那么特征值到底是什么含义呢？此外，矩阵的trace反应了矩阵的什么性质？

## 基本概念

* 数域
若数集F含有数1并且对四则运算封闭，则称数域F
* 负矩阵
对称
-A 为A的负矩阵
* 零矩阵
元素全为零的矩阵，称为零矩阵，记为0

## 线性方程组
从方程组的求解过程中，聪明的中国祖先率先在《九章算术》使用了矩阵的思想来解答方程组。可是中国祖先没有能够将这种技术抽象化为理论，所以让人感觉矩阵运算是外国人发明的。

### 线性方程组

Three possiblities：
* Unique solution
* No solution
* Infinitely many solutions

#### 高斯消元法
将一个复杂的线性系统转化为一个简单的线性系统

>三种基本行变换

* interchange row i and j
* replace row i by a nonzero multiple of itself
* replacerow j by a combination of itself plus a multiple of row i

1. 行选主元素方法
    - 选择过程是一行一行的进行
    - 针对当前行，从所有列中选择同列位置中最大的一行

2. 列选主元素方法
    - 选择过程是一列列地进行
    - 针对当前列，从所有行中选择同行位置中*最大*的一列，并且进行列交换
    - 注意：记录交换后的位置对应这原来要求变量的位置

3. 全局最大值方法
    - 选择过程是一行一行的进行
    - 针对当前行，从所有列中选择同行位置中*最大*的一列
    - 针对当前列，从所有行中选择该列位置中最大的一行
    - 选定pivot位置之后，注意要记录变换前后行的位置

#### 高斯-约旦方法
将每个pivot元素变换为1，而pivot元素的上下位置均为0

>问题

由于计算机可表示的数字是离散可数的，而现实计算中为实数空间，导致
我们计算应用问题的时候，往往出现由于pivot选择不当，使得计算结果不是实际结果，例如
[Partial Pivoting](http://github.com/cwlseu/cwlseu.github.io/raw/master/img/blog/matrix-analysis/1.jpg)
由于我们选择pivot位置元素，相对于另外其他行来说，会将其他行的数值在进行
初等行变换之后作为可忽略部分被round操作去掉了，从而造成最终结果的不正确。

>解决方法：

合理地选择pivot 元素位置

#### III-Conditioned Systems
A system of linear equations is said to be ill-conditioned when some small
perturbation in the system can produce relatively large changes in the exact
solution. Otherwise, the system is said to be well-conditioned

一般情况下，病态系统是那些几乎平行的线，或者几乎平行的面。一个小的不确定参数，将意味着极大的不确定解。

那么我们该如何检测这个系统是不是病态的哪？往往我们采取对系数较小的改变量，观察结果的差值。如果结果较大没那么该系统为病态系统。因此，我们往往采用计算结果检查的方式来对病态系统结果进行求解。

### 行阶梯形式和矩阵的秩

#### 矩阵的秩
假设 $A_{m*n} 通过行变换为阶梯矩阵形式 E,那么A的秩被定义为如下：

    rank(A) = number of pivots  
            = number of nonzero rows in E  
            = number of basic columns in A 

A的基本列为包含主元位置的列的集合。


###线性系统的系统的相容性 (Consistency of Linear Systems)

#### 增广矩阵 [A|b] 是一致的  （要求AX = b的解)
* 增广矩阵的约简形式不会出现（0 0 .... 0 | a), where a != 0
* b 不是增广矩阵基本列
* rank([A|b]) = rank(A)
* b 可以由A基本列线性表示

#### 齐次线性方程组
0 是Homogeneous System的平凡解
假设 A_{m*n} 是系数矩阵，m个等式，n个未知数。同时假设rank(A) = r
*基本变量* 基本列对应位置的变量称为基本变量
*自由变量* 未知数中对应于非基本列称为自由变量
因此这个方程中存在着r个基本变量和 n-r个自由变量
通解为 $x = x_f_1h_1 +ｘ_f_2h_2+... + x_f_{n-r}h_{n-r}$

#### 非次线性方程组
非齐次线性方程组通解为 $x = p+ x_f_1h_1 +ｘ_f_2h_2+... + x_f_{n-r}h_{n-r}$

-------------------------------------------------------------------------------

## 矩阵代数
### 矩阵加法

$[A+B]_{ij} = A_{ij} + B_{ij}$
Additive inverse of A : (-A) = [-a_{ij}]
遵循交换律、结合律、加法恒等率

### 数乘

$[aA]_{ij} = a[A]_{ij}$

#### 数乘的性质

* 结合律： (ab)A = a(bA)
* Distributive property: 
    - a(A + B) = aA + aB
    - (a + b) A = aA + bA
* Identity property: 1A = A

#### 转置

$[A]_{ij} = [A^T]_{ji}$
(AB)^T = B^TA^T

#### 共轭转置 conjugate transpose

$A^*$ 是对当矩阵A中存在复数的时候，先对A的每个元素取共轭，然后取复数

#### 对称矩阵
A is a symmetrix matrix whenerver $A^T = A$
skew-symmetrix matrix whenever $A^T = -A$
一个矩阵可以表示成为一个对称矩阵和一个反对程矩阵的和
一个矩阵可以构造对称矩阵和反对称矩阵

### Linear Fucntion

线性系统满足
* f(x + y) = f(x) + f(y)
* f(ax) = af(x)
例如：过原点的直线f(x) = kx或者过原点的平面
或者简化为如下一个条件：
*f(ax + y) = af(x) + f(y)*

### 矩阵乘法

这个是重点讲解的地方，我们主要从三个方面对这个知识点进行解释：
$[AB]ij = \sum_{k=0}^{k=n}a_{ik}b_{kj}$
[AB]i* <=> A的第i行与B的乘积
[AB]*j <=> A与B的第j列的乘积

>注意:
    Matrix multiplication isn't commutative:*AB != BA*