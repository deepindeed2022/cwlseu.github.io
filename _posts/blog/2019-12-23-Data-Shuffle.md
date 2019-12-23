---
layout: post
comments: true
title: "大规模训练数据的shuffle"
categories: [blog ]
tags: [机器学习]
description: 训练数据shuffle
---

## 训练数据shuffle

## 重要性[^1][^3]
以猫狗分类为例， 假如数据集是

> Dog，Dog，Dog，... ，Dog，Dog，Dog，Cat，Cat，Cat，Cat，... ，Cat，Cat

所有的狗都在猫前面，如果不shuffle，模型训练一段时间内只看到了Dog，必然会过拟合于Dog，一段时间内又只能看到Cat，必然又过拟合于Cat，这样的模型泛化能力必然很差。 那如果Dog和Cat一直交替，会不会就不过拟合了呢？

> Dog，Cat，Dog，Cat，Dog ，Cat，Dog，...

依然会过拟合，模型是会记住训练数据路线的，为什么呢？

当用随机梯度下降法训练神经网络时，通常的做法是洗牌数据。在纠结细节的情况下，让我们用一个极端的例子来解释为什么shuffle是有用的。假设你正在训练一个分类器来区分猫和狗，你的训练集是50,000只猫后面跟着50,000只狗。如果你不洗牌，你的训练成绩就会很差。
严格地说，这个问题是由梯度噪声中的序列相关性和参数更新的不可交换性引起的。首先我们需要明白固定的数据集顺序，意味着给定迭代步，对应此迭代步的训练数据是固定的。 假如目标函数是$J=f(w, b)$，使用梯度下降优化$J$。给定权重取值$w、b$和迭代步step的情况下，固定的数据集顺序意味着固定的训练样本，也就意味着权值更新的方向是固定的，而无顺序的数据集，意味着更新方向是随机的。所以固定的数据集顺序，严重限制了梯度优化方向的可选择性，导致收敛点选择空间严重变少，容易导致过拟合。所以模型是会记住数据路线的，所以shuffle很重要，一定shuffle。

[^1]: https://juejin.im/post/5c6b989bf265da2ddd4a5261 "数据集shuffle的重要性"

## 两次shuffle算法
两次shuffle算法过程中包括pile的id的shuffle和pile内部的shuffle. Fisher Yates算法和 twice pass shuffle算法如下。

```python
import os
import random

# https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
def fisher_yates_shuffle(dataset=list()):
	size = len(dataset)
	for i in range(size-1):
		j = random.randint(i, size-1)
		dataset[i], dataset[j] = dataset[j], dataset[i]

def twice_pass_shuffle(dataset, total_size, M):
	# first pass
	p = [[] for _ in range(M)]
	for i in range(total_size):
		j = random.randint(0, M-1)
		p[j].append(dataset[i])
	# second pass
	result = []
	for j in range(M):
		fisher_yates_shuffle(p[j])
		result.extend(p[j])
	return result

if __name__ == '__main__':
	l = [i for i in range(1,101)]
	print("befor shuffle:\n", l)
	result = twice_pass_shuffle(l, total_size=100, M=10)
	print("\nshuffle result:\n", result)
```

## twice pass shuffle的其他问题处理与性能分析[^4]

[^2]: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle "Fisher Yates shuffle"
[^3]: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/#whyshuffle "why shuffle"
[^4]: https://blog.janestreet.com/how-to-shuffle-a-big-dataset "how to shuffle a big dataset"