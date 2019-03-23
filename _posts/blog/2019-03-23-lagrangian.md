---
layout: post
title: "约束优化方法：拉格朗日乘子法与KKT条件"
categories: [blog ]
tags: [Algorithm]
description: "拉格朗日乘子，有限制条件的优化问题与无限制的优化问题的桥梁"
---

* content
{:toc}

## 引言

最速下降法、拟牛顿法等都是求解准则函数（即无约束优化问题）的算法，这就需要有一个前提：怎样得到无约束准则函数？而拉格朗日乘子，将有限制条件的优化问题转化为无限制的优化问题，可见拉格朗日乘子搭建了一个桥梁：将有限制的准则函数，转化为无限制准则函数，进而借助最速下降法、拟牛顿法等求参算法进行求解，在这里汇总一下拉格朗日乘子法是有必要的，全文包括：
* 含有等式约束的拉格朗日乘子法；
* 拉格朗日对偶方法；


## 不同类型的约束下的优化问题

### 无约束优化

如果不带任何约束的优化问题，对于变量$x \in \mathbb{R}^N$,无约束的优化问题
$$\min_xf(x)$$
这个问题直接找到使目标函数的导数为0的点即可，$\nabla_xf(x) = 0$,如果没有解析解的话，可以使用
梯度下降法或者牛顿法等迭代手段来使$x$沿**负**梯度方向逐步逼近极小值点。

### 等式约束优化

当目标函数加上约束条件之后：

$$
\begin{aligned}  
    &\min_{x } \  f(x)  \\
    &s.t.  \ \ \ g_i(x) = 0 , i = 1,2,...,m \\
\end{aligned}
$$

约束条件会将解的范围限定在一个可行域，此时不一定能找到使得
$\nabla_xf(x)$为 0 的点，只需找到在可行域内使得$f(x)$最小的值即可，
常用的方法即为拉格朗日乘子法，该方法首先引入Lagrange Multiplier 
$\alpha \in \mathbb{R}^m$，构建 Lagrangian 如下：

$$L(x,\alpha) = f(x) + \sum_{i=1}^m \alpha_i g_i(x)$$

求解方法，首先对Lagrangian关于$\alpha$和$x$求导数，令导数为0：

$$
\left \{ \begin{aligned}  \nabla_x L(x,\alpha)= 0  \\ \nabla_{ \alpha } L(x,\alpha)= 0 \end{aligned} \right.
$$

求得$x, \alpha$的值以后，将$x$带入$f(x)$即为在约束条件$h_i(x)$下的可行解。

看一个示例，对于二维情况下的目标函数是$f(x,y)$，
在平面中画出$f(x,y)$的等高线，如下图的虚线所示，并只给出一个约束等式$g(x,y)=c$，
如下图的红线所示，
目标函数$f(x,y)$与约束函数$g(x,y)$只有三种情况，相交、相切或者没有交集，
没交集肯定不是解,只有相交或者相切可能是解，
但*相交得到的一定不是最优值，因为相交意味着肯定还存在其它的等高线在该条等高线的内部或者外部，
使得新的等高线与目标函数的交点的值更大或者更小，这就意味着**只有等高线与目标函数的曲线相切**的时候，
才可能得到可行解.*

![@来自Wikipedia的图片](https://cwlseu.github.io/images/optmethods/LagrangeMultipliers-01.png)

因此给出结论：拉格朗日乘子法取得极值的必要条件是目标函数与约束函数相切，这时两者的法向量是平行的，即
$\nabla _xf(x) – \alpha \nabla_x g(x) = 0$

<!-- ![@解释为什么需要两个导数为0的时候是最优解](https://cwlseu.github.io/images/optmethods/kkt-01.png) -->

### 不等式约束
当约束加上不等式之后，情况变得更加复杂，首先来看一个简单的情况，给定如下不等式约束问题

$$
\begin{aligned} 
    &\min_x \ f(x) \\
    & \ s.t. \ \  g(x) \le 0
\end{aligned}
$$

对应的 Lagrangian 与图形分别如下所示,这时的可行解必须落在约束区域$g(x)$之内，下图给出了目标函数的等高线与约束

$$
L(x, \lambda) = f(x) + \lambda g(x)
$$

![@拉格朗日不等式的情况](https://cwlseu.github.io/images/optmethods/LagrangeMultipliers-02.png)
由图可见可行解$x$只能在$g(x)\le 0$的区域里取得：
* 当可行解$x$落在$g(x)<0$的区域内，此时直接极小化$f(x)$即可；
* 当可行解$x$落在$g(x)=0$即边界上，此时等价于等式约束优化问题。

> 当约束区域包含目标函数原有的的可行解时，此时加上约束可行解仍落在约束区域内部，对应$g(x)<0$的情况，这时约束条件不起作用；
> 当约束区域不包含目标函数原有的可行解时，此时加上约束后可行解落在边界$g(x)=0$上。
> 下图分别描述了两种情况，右图表示加上约束可行解会落在约束区域的边界上。

![@约束区域内是否包含目标函数的原有的可行解](https://cwlseu.github.io/images/optmethods/LagrangeMultipliers-03.png)

以上两种情况就是说，要么可行解落在约束边界上即得$g(x)=0$, 
要么可行解落在约束区域内部，此时约束不起作用，令$\lambda=0$消去约束即可，
所以无论哪种情况都会得到：
$\lambda g(x)=0$

还有一个问题是$\lambda$的取值，在等式约束优化中，约束函数与目标函数的梯度
只要满足平行即可，而在不等式约束中则不然，若$\lambda \ne 0$，
这便说明可行解$x$是落在约束区域的边界上的，**这时可行解应尽量靠近无约束时的解，
所以在约束边界上，目标函数的负梯度方向应该远离约束区域朝向无约束时的解，
此时正好可得约束函数的梯度方向与目标函数的负梯度方向应相同**：

$$-\nabla_x f(x) = \lambda  \nabla_xg(x)$$

上式需要满足的要求是拉格朗日乘子$\lambda>0$ ，这个问题可以举一个形象的例子，假设你去爬山，目标是山顶，但有一个障碍挡住了通向山顶的路，所以只能沿着障碍爬到尽可能靠近山顶的位置，然后望着山顶叹叹气，这里山顶便是目标函数的可行解，障碍便是约束函数的边界，此时的梯度方向一定是指向山顶的，与障碍的梯度同向，下图描述了这种情况:

![](https://cwlseu.github.io/images/optmethods/LagrangeMultipliers-04.png)
左图中这个$-\nabla_x f(x)$局部最小值指向可行区域（$g(x) \le 0$），也就是还有使得更小的点存在。
而右图中的$-\nabla_x f(x)$局部最小值是远离可行区域的。

### KKT条件

    任何不等式约束条件的函数凸优化问题，都可以转化为约束方程小于0且语义不变的形式，以便于使用KKT条件.

可见对于不等式约束，只要满足一定的条件，依然可以使用拉格朗日乘子法解决，这里的条件便是KKT条件。接下来给出形式化的KKT条件 首先给出形式化的不等式约束优化问题：

$$
\begin{aligned}  
&\min_x \  f(x)  \\
&s.t.  \ \ \ h_i(x) = 0 , \  i = 1,2,...,m \ \\
& \ \ \ \ \ \ \ \ \ \   g_j(x) \le 0, \  j = 1,2,...,n
\end{aligned}
$$

列出 Lagrangian 得到无约束优化问题：

$$
L(x,\alpha,\beta) =f(x) + \sum_{i=1}^m \alpha_i h_i(x) + \sum_{j=1}^n\beta_ig_i(x)
$$

经过之前的分析，便得知加上不等式约束后可行解$x$需要满足的就是以下的 KKT 条件：

$$
\begin{aligned}
\nabla_x L(x,\alpha,\beta) &= 0    \qquad\qquad\qquad\qquad(1)\\
             \beta_jg_j(x) &= 0  , \ j=1,2,...,n \qquad(2) \\
                    h_i(x) &= 0 , \ i=1,2,...,m \qquad(3) \\
                    g_j(x) &\le 0  , \  j=1,2,...,n \qquad(4) \\
                   \beta_j &\ge  0 , \ j=1,2,...,n  \qquad(5) \\
\end{aligned}
$$


满足 KKT 条件后极小化 Lagrangian 即可得到在不等式约束条件下的可行解。 KKT 条件看起来很多，其实很好理解:

(1). 拉格朗日取得可行解的必要条件；

(2). 这就是以上分析的一个比较有意思的约束，称作松弛互补条件；

(3)-(4). 初始的约束条件；

(5). 不等式约束的 Lagrange Multiplier 需满足的条件。

主要的KKT条件便是 (3) 和 (5) ，只要满足这俩个条件便可直接用拉格朗日乘子法， SVM 中的支持向量便是来自于此。

## 拉格朗日对偶问题
https://www.cnblogs.com/ooon/p/5723725.html#4200596
未完，待续

## 参考链接
* [一个经典问题：最短路径问题](http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html)
* [曲面上一点求法向量](https://zhidao.baidu.com/question/362692467232981652.html)
* [拉格朗日乘子法与KKT条件](http://www.cnblogs.com/ooon/p/5721119.html)
* [lagrange算子](https://en.wikipedia.org/wiki/Lagrange_multiplier)
* [KKT](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)
* http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/KKT.pdf
* http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/Duality.pdf
* [如何理解拉格朗日算子](https://www.zhihu.com/question/38586401)
* [从SVM算法理解拉格朗日对偶问题](https://www.svm-tutorial.com/2016/09/duality-lagrange-multipliers/)