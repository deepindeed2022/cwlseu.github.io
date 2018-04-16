---
layout: post
title: "概率论基础"
categories: [blog ]
tags: [概率论]
description: 概率论中的基本概念，做机器学习相关的内容，不得不知的知识。 
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [cwlseu]
- 博客： <https://cwlseu.github.io/>


## 概率论基础

$ N(\mu, \sigma^2)$正态分分布，若n个相互独立的随机变量ξ₁、ξ₂、……、ξn ，**均服从标准正态分布**（也称**独立同分布于标准正态分布**），则这n个服从标准正态分布的随机变量的平方和构成一新的随机变量，其分布规律称为卡方分布（$\chi^2$ chi-square distribution）, 其中参数n称为自由度

### 卡方分布

> 可加性

$\chi^2_1\sim\chi^2(m)$ 和 $\chi^2_2\sim\chi^2(n)$且相互独立，则
$\chi^2_1 + \chi^2_2 \sim \chi^2(n+m)$

> 期望和方差 

若$\chi^2\sim\chi^2(n)$， 则$E\chi^2 = n, D\chi^2 = 2n$

### t-分布

t-分布是一个正态分布除以(一个卡方分布除以它的**自由度**然后开根号);
$X \sim N(0, 1)$ Y服从自由度为n的卡方分布$Y \sim\chi^2(n)$
我们称随机变量
$$T = \frac{X}{\sqrt{\frac{Y}{n}}}$$
所服从的分布为自由度为n的t-分布，记为$T \sim t(n)$

> 极限逼近正态

t-分布的极限分布为标准正态分布。

> $\alpha$-分位点

假设$T\sim t(n), 0 \lt \alpha \lt 1, t_{\alpha}(n)$满足
$$P(T \ge t_\alpha(n)) = \alpha$$

另外 $t_{1- \alpha}(n) = - t_\alpha(n)$
例如：$t_{0.95}(15) = - t_{0.05}(15)$

### F-分布

设随机变量$X \sim \chi^2(m)$, $Y \sim \chi^2(n)$， 且$X, Y$相互独立。则称：
$$F = \frac{X}{Y}\frac{n}{m}$$
所服从的分布为自由度为m，n的F-分布，记为$F\sim F(m,n)$

假设$F\sim F(m, n)$, $F_{\alpha}(m, n)$满足
$$P(F \ge F_\alpha(m, n)) = \alpha$$
则称$F_{\alpha}(m, n)$是自由度为m, n的F-分布的上侧$\alpha$-分位点。

### 小结

比如X是一个正态分布分布, $X \sim N(\mu, \sigma^2)  $
$$X \sim \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x - \mu)^2}{2\sigma^2})$$

$$Y(n) = \chi^2(n)=X_1^2+X_2^2+……+X_n^2$$,这里每个$X_n$都是一个标准正态分布分布，
$$t(n)=\frac{X}{\sqrt{\frac{Y}{n}}} $$
$$F(m, n)=\frac{\frac{Y_1}{m}}{\frac{Y_2}{n}}$$


## 正态总体中统计量的分布

### 单个正态总体中统计量的分布

总体分布$X \sim N(\mu, \sigma^2)$, $（X_1, ... X_n）是来自总体X$的容量为n的简单随机样本， 样本均值
$\bar X = \frac{1}{n}\sum_{i=1}^n{X_i}$
样本方差为：$S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar X)^2$
则：

* $U = \frac{\bar X -\mu}{\sigma}\sqrt{n} \sim N (0, 1)$
* $\bar X$与$S^2$相互独立
* $\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^{n}(\frac{X_i - \bar X}{\sigma})^2 \sim \chi^2(n-1)$ 
* $T = \frac{\bar{X} - \mu}{S}\sqrt{n} \sim t(n-1)$


### 两个正态总体中的统计量的分布

##  各个分布的应用

1. **方差已知**情况下求均值是Z-检验。
2. 方差**未知**求均值是t检验（样本标准差s代替总体标准差R，由样本平均数推断总体平均数）
3. 均值方差都未知求方差是$\chi^2$检验
4. 两个正态分布样本的均值方差都未知情况下求两个总体的方差比值是F检验。
5. $\chi^2(n)$分布拟合检验：总体的分布未知的情况下，根据样本来检验总体分布的建设。样本容量足够大时，统计量（公式略）近似服从$\chi^2(k-1)$分布，通过$\chi^2$来验证拟合。同时需要进行偏度、峰度检验，避免在验证总体正态性是犯第二类（取伪）错误。

## 错误函数
[https://en.wikipedia.org/wiki/Error_function](https://en.wikipedia.org/wiki/Error_function)

# 相关系数和判定系数 
    相关系数的平方等于判定系数。其中相关系数的符号与X的参数相同。相关系数是仅被用来描述两个变量之间的线性关系的，但判定系数的适用范围更广，可以用于描述非线性或者有两个及两个以上自变量的相关关系。

## 特征选择



## 参考文献
[1]. http://www.cnblogs.com/baiboy/p/tjx11.html