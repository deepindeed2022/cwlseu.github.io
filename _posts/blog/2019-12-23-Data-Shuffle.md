---
layout: post
comments: true
title: "大规模训练数据的shuffle"
categories: [blog ]
tags: [机器学习]
description: 训练数据shuffle
---

* content
{:toc}


# 大规模训练数据的shuffle

## 重要性[^1][^3]
以猫狗分类为例， 假如数据集是

> Dog，Dog，Dog，... ，Dog，Dog，Dog，Cat，Cat，Cat，Cat，... ，Cat，Cat

所有的狗都在猫前面，如果不shuffle，模型训练一段时间内只看到了Dog，必然会过拟合于Dog，一段时间内又只能看到Cat，必然又过拟合于Cat，这样的模型泛化能力必然很差。 那如果Dog和Cat一直交替，会不会就不过拟合了呢？

> Dog，Cat，Dog，Cat，Dog ，Cat，Dog，...

依然会过拟合，模型是会记住训练数据路线的，为什么呢？

当用随机梯度下降法训练神经网络时，通常的做法是洗牌数据。在纠结细节的情况下，让我们用一个极端的例子来解释为什么shuffle是有用的。假设你正在训练一个分类器来区分猫和狗，你的训练集是50,000只猫后面跟着50,000只狗。如果你不洗牌，你的训练成绩就会很差。
严格地说，这个问题是由梯度噪声中的序列相关性和参数更新的不可交换性引起的。首先我们需要明白固定的数据集顺序，意味着给定迭代步，对应此迭代步的训练数据是固定的。 假如目标函数是$J=f(w, b)$，使用梯度下降优化$J$。给定权重取值$w、b$和迭代步step的情况下，固定的数据集顺序意味着固定的训练样本，也就意味着权值更新的方向是固定的，而无顺序的数据集，意味着更新方向是随机的。所以固定的数据集顺序，严重限制了梯度优化方向的可选择性，导致收敛点选择空间严重变少，容易导致过拟合。所以模型是会记住数据路线的，所以shuffle很重要，一定shuffle。

[^1]: https://juejin.im/post/5c6b989bf265da2ddd4a5261 "数据集shuffle的重要性"

## 2-pass-shuffle算法
我们假设一个数据集$X^m$包含样本数目为$m$, 大小为$S_{X^m}$, 计算内存RAM大小为$S_{RAM}$.
当$S_X \lt S_{RAM}$的时，我们完全可以使用训练框架中的`Dataset shuffle`函数进行处理,如`Fisher Yates Shuffle`。但我们实际应用场景中，$S_X \ggg S_{RAM}$. 本节将针对这种业务场景进行讨论。

```python
# https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
def fisher_yates_shuffle(dataset=list()):
	size = len(dataset)
	for i in range(size-1):
		j = random.randint(i, size-1)
		dataset[i], dataset[j] = dataset[j], dataset[i]
```

分块是一种很普遍的想法，但是如何分块，以及分块后如何随机地写回到文件中才是最终目标。而且要注意的是，数据集$X^m$的每一次访问都存在大量的IO，将非常耗时。因此，设计随机算法的过程中，IO也要考虑在内。

`2-pass-shuffle`算法过程中包括块id的shuffle和块内部的shuffle. Fisher Yates算法和 twice pass shuffle算法如下。

需要自己设置一个超参数$M$, 直观上需要满足的条件：
$$M \ge \frac{S_{X^m}}{S_{RAM}}$$

python代码模拟实现如下：

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

当我面对这个问题的时候，第一次并没有给出这个答案，第二次才给出接近这个算法的答案。
之前的算法分M块，然后M块之间两两洗牌，进行$\frac{M(M-1)}{2}$次。这个方法看上去好像可以，但是存在以下问题：
* IO次数太多，性能应该不好
* 两块之间怎么洗牌的算法决定这shuffle的结果是否随机，而且当时我并没有给出比较好的洗牌策略。

## 2-pass-shuffle的其他问题处理
还有可能遇到的问题就是第一次pass过程中，每个分块的数据并不是相等的，很有可能有那么一两块的大小比$S_{RAM}$大，导致后面不能进行内存内shuffle. 这个问题在*how to shuffle a big dataset*[^4]这篇文章中有一个解决方案。其实还有简单粗暴的方案就是针对这个特殊的分块进行单独处理，再进行一次类似的`2-pass-shuffle`就是了。

## 如何训练过程中，随机从一个超大数据集合中获取训练数据[^5]

The `Dataset.shuffle()` implementation is designed for data that could be shuffled in memory; we're considering whether to add support for external-memory shuffles, but this is in the early stages. In case it works for you, here's the usual approach we use when the data are too large to fit in memory:

1. Randomly shuffle the entire data once using a MapReduce/Spark/Beam/etc. job to create a set of roughly equal-sized files ("shards").
2. In each epoch:
  * Randomly shuffle the list of shard filenames, using `Dataset.list_files(...).shuffle(num_shards)`.
  * Use `dataset.interleave(lambda filename: tf.data.TextLineDataset(filename), cycle_length=N)` to mix together records from `N` different shards.
  * Use `dataset.shuffle(B)` to shuffle the resulting dataset. Setting `B` might require some experimentation, but you will probably want to set it to some value larger than the number of records in a single shard.

## 参考地址

[^2]: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle "Fisher Yates shuffle"
[^3]: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/#whyshuffle "why shuffle"
[^4]: https://blog.janestreet.com/how-to-shuffle-a-big-dataset "how to shuffle a big dataset"
[^5]: https://github.com/tensorflow/tensorflow/issues/14857