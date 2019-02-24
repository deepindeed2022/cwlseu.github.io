---
layout: post
title: "Google News Personalization：Scalable Online Collaborative Filtering"
categories: [blog ]
tags: [机器学习]
description:  这篇论文介绍了google news推荐系统的实现。在用户量很大的前提下，原有的推荐算法适用性较差，需要对其进行改进，例如使用mapreduce，bigtable等技术提高运算速度，综合考虑多种推荐算法等等。
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/> 

## abstract
这篇论文介绍了google news推荐系统的实现。在用户量很大的前提下，原有的推荐算法适用性较差，需要对其进行改进，例如使用mapreduce，bigtable等技术提高运算速度，综合考虑多种推荐算法等等。

## google news的特点 
> 处理google news的一些难点

- scalability:google news访客较多，新闻数据也较多
- item churn:物品(新闻)会动态变化，模型需要不断重建，这是一个非常费时的任务，对于google news来说，每分钟都会产生很多新闻，模型超过一段时间后效果会变差

> google news的一些假设

这里假设用户点击某一条新闻即表示对该新闻感兴趣，之所以可以这样假设，因为google news前端页面已经显示了文章的摘要，用户能够了解这篇文章的大概信息，如果点击该新闻即可证明对此新闻感兴趣。

## 问题表述
对于N个用户，u={u1,u2,...uN}和M个物品(文章)S={s1,s2...sm}，给定一个用户的浏览记录集合Cu，{s1,s2...si|cu|}，推荐K个用户感兴趣的文章。
同时要求服务器的**响应时间要短**，服务器要做的事情如下：对新闻聚类；针对HTTP请求返回HTML内容；推荐系统产生推荐列表

## 相关工作
### 基于内存的算法
根据用户过去的评价进行推荐，为此需要计算用户之间的相关性，`w(ui,uj)`矩阵衡量的就是任意两个用户的联系性，对于一个用户`u_a`，他对于文章`s_k`的评分如下：
$r_{u_a,s_k} = \sum_{i\neqa} I_{(u_i, s_k)}w(u_a, u_i)$

`I(ui,sk)`为1表明用户`u_i`点击过`s_k`。当计算得到的评分超过一定阈值后就可以假定用户`u_a`喜欢文章`s_k`。但是这个方法最大的问题就是可拓展性较差，由于所有数据要放入内存，当数据量较大值无法使用此方法。


### 基于模型的算法

此方法依据用户之前的评分为该用户建立模型，使用模型预测那些未被评价过的物品（协同过滤就是这种方法）。这种方法的缺陷是把每一个用户划分到一个指定的类别中，实际情况是用户对于不同的主题可能有不同的喜好，无法划分到一个指定的类别。

## 模型
本文中涉及的模型是一个混合模型，将上述两种模型进行线性加权结合。
![@整体模型公式](https://cwlseu.github.io/images/lsh/model_eq.PNG)
`I(u_i,s_k)` is 1 if the user `u_i` clicked on the story `s_k` and 0 otherwise.话说将来可以使用SVM算法学习不同algorithm之间的比重。
### MinHash

## PLSI
PLSI was introduced in [3], where Hofmann developed
probabilistic latent semantic models for performing collaborative
filtering.The relationship between
users and items is learned by modeling the joint distribution
of users and items as a mixture distribution.

![@plsi 公式](https://cwlseu.github.io/images/lsh/plsi_eq.PNG)

这个模型的关键在于隐变量Z的引入，使得users和items变得条件独立。这个模型也可以看做是一个生成模型。用户u通过p(z|u)随机选择状态z，然后再基于选择的状态z和CPD p(s|z)选择item.

> 但是有一个缺点：有新user或者新item需要**重新训练模型**，即使我们通过并行化技术可以加速训练过程，但是还是需要离线训练，不能实时更新模型。


### 分布式EM(Expectation Maximization)算法
#### 经典EM算法介绍

> 目标函数：最大化条件似然函数

![@plsi 公式](https://cwlseu.github.io/images/lsh/EM_obj.PNG)

> 训练过程：
* 初始化：

ˆp values stand for the parameter estimates from the previous iteration of the EM algorithm. For the first iteration, we set ˆp to appropriately normalized random values that form a probability distribution.
* 两步走：

![@plsi 公式](https://cwlseu.github.io/images/lsh/EM_2Step.PNG)

> 算法分析：
对于数据规模比较大的时候，内存空间消耗巨大。例如M=N=10M, L=1000,
需要空间(M+N)xLx4 ~ 80GB

### 并行EM算法训练PLSI模型

> 并行EM算法

* Step E

![@EM_Step_E](https://cwlseu.github.io/images/lsh/EM_distribution_E.PNG)

* Step M

![@EM_Step_E](https://cwlseu.github.io/images/lsh/EM_distribution_M.PNG)

考虑在一个RxK的集群上进行训练，如下图:
![@分布式EM架构图](https://cwlseu.github.io/images/lsh/mr_em.PNG)

用户和items分别被分为R组和K组。对于一个点击记录(u, s)对，我们通过一定的算法，如Hash，

* i = Hash(user_id) % R
* j = Hash(item_id) % K

分配到该用户和item所在的机器(i, j)上进行训练。这样，一台机器就只需要计算原来的1/R的用户和1/K的物品的CPD.

> Reducer如何归并

从前面的公式可以看出，剩下Reduce操作很简单，只需要计算p(u|z),操作就是加起来。

## [相关知识LSH](https://my.oschina.net/sulliy/blog/78777)
经常使用的哈希函数，冲突总是不招人喜欢。LSH却依赖于冲突，在解决NNS(Nearest neighbor search )时，我们期望：
- 离得越近的对象，发生冲突的概率越高
- 离得越远的对象，发生冲突的概率越低

由于是依靠概率来区分，总会有错判的问题（false positives/negatives）。由于LSH排除了不可能的对象集合，减少了需要处理的数据量，在NNS领域有很多成功的应用。

我们来看一个寻找相似文档问题，给你一个文档库，寻找其中相似的文档（仅仅是内容上相似，语义层次的可以考虑隐式语义（LSA））。寻找相似文档可以分为三个步骤：
1. Shingling：将文档转化为一些集合。类似于中文分词。
2. Minhashing：在保留文档之间相似度的同时，将上一步得到庞大数据，转化为短些的签名（signatures）。将对集合的比较转化为签名的比较。
3. Locality-sensitive hashing：得到的签名集还是很大，通过LSH进一步缩减处理的数据量。只是比较存在很高概率相似的一些签名。

![@相似文档查找过程](https://cwlseu.github.io/images/lsh/minhash.png)

> Shingling

按一个指定长度k，将文档分割为字符串集。我们将其称为`k -shingle`或者`k -gram`。举个例子`k=2，doc = abcab`，那么`2-shingles = {ab, bc, ca}`。字符串集也可以称为词袋。在这个例子里面，`ab`出现了2次，在这里我们不考虑权重，并不考虑一个字符串出现的次数。**k的选取比较重要，如果太小，数据量很大，过大的话，匹配效果不好**。

到此，第一步完成，我们得到了每个文档的字符串集合。要判断文档相似，就等价于判断集合的相似度。集合相似的一个重要办法就是`Jaccard相似度`。

$Sim(C1, C2) = \frac{C1 与 C2}{C1 或 C2}$

我们将集合转化为一个矩阵来进行处理：矩阵的行是每个字符串，矩阵的列是每一个文档。所有的矩阵行就构成了所有文档字符串集合的一个并集，是集合空间。如果文档`x`里面有字符串`y`，那么`(x,y) = 1`，否则`(x,y) = 0`。现在得到了一个布尔矩阵。

    C1  C2
    0   1
    1   0
    1   1
    0   0
    1   1
    0   1

如上面所示，`Sim (C1, C2) = 2/5 = 0.4`。需要注意的是：我们完全可以不采用`0/1`，而是字符串出现的次数来定义矩阵，另一个是**这个矩阵是稀疏的，并且非常庞大**。

> MinHashing

现在进行第二步：签名。给每个列，即每个文档个一个签名。这个签名应该满足2个性质：
- 足够小，才能减少存储空间，减少计算代价。
- 保持相似度，也就是签名的相似度与现在列之间的相似度是一样的。
  `Sim(C1, C2) = Sim(Sig(C1), Sig(C2))`

Minhashing首先假设，行是随机排列的，然后定义一个哈希函数h(C)，哈希函数的值是列C在前面定义的随机排列中，第一个为值1行号。然后使用一定数量的相互独立的哈希函数来构成签名。

![@相似文档查找过程](https://cwlseu.github.io/images/lsh/minhash_step2.png)

图中间是得到的布尔矩阵，左边是*三个随机的行的排列*。其中的值是行的编号。我们用行的排列数，在矩阵中找最先出现1的行。对于排列数`1376254`，
- 在矩阵第一列中为1的有1354，然后*取最小的1*；
- 在矩阵第二列中为1的有762，取最小的2；
- 在矩阵第三列中为1的有154，取最小的1；
- 在矩阵第四列中为1的有3762，取最小的2。
这就得到了签名矩阵的第三行。同样的方法可以得到签名矩阵的第一行和第二行。

    在布尔矩阵中计算第一列和第二列的Jaccard相似度：
        Sim (C1, C2) = 0/6 = 0，
    在签名矩阵中计算第一列和第二列的Jaccard相似度：
        Sim (C1, C2) = 0/3 = 0。

    在布尔矩阵中计算第二列和第四列的Jaccard相似度：
        Sim (C1, C2) = 3/4 = 0.75，
    在签名矩阵中计算第二列和第四列的Jaccard相似度：
        Sim (C1, C2) = 3/3 = 1。差别不是很大。

事实上，如果将行的所有排列都拿来计算签名，那么h (C1) = h (C2) 的概率等于Sim (C1, C2)。签名的相似度等于列的相似度。
如果将所有的行的排列数都拿来做签名，那不是又回到原来的规模了么？还有计算排列数的开销？的确，但是可以发现随着排列数的增加，签名相似度与行的相似度之间的误差越来越小，选择合适的排列数目就可以满足应用需求，通常我们取**100**个排列数。
计算排列数也是个不小的开销，我们可以用哈希函数近似处理。下面是一个参考算法：

```perl
for each row r
    for each column c
        if c has 1 in row r
            for each hash function hi do
                if hi (r) is a smaller value than M (i, c) then
                    M(i, c) := hi (r);
```

> locality sensitive hashing

得到的签名数量还是很大，在在线实时应用中，不可能去一一比较。LSH筛选出有极高相似概率的**候选对**，进而减少比较的次数。LSH要寻找一个哈希函数f(x,y)，它能够返回x，y是否是一个候选对。由于我们得到了签名矩阵，LSH就是将候选的列放到相同的Bucket中。

LSH选择一个相似度阀值s，如果两个列的签名相符度至少是s，那么就将它们放到同一个Bucket中。原理就这样简单，但是不是太粗糙了呢？

我们将矩阵的行分成b个Band（r行为一个Band）：
![@相似文档查找过程](https://cwlseu.github.io/images/lsh/lsh_bank.png)

然后对每个Band做哈希，将她分区的列的一部分哈希到k个Bucket中。现在的**候选对**就是至少在1个Band中被哈希到同一个Bucket的列。分割之后的工作就是，调制参数b和r使尽可能可多的相似的对放到同一个Bucket，并且尽量减少不相似的对放到同一个Bucket中。

![@相似文档查找过程](https://cwlseu.github.io/images/lsh/lsh_update.png)
分割Band的目的是要得到很好的区分度效果。

![@相似文档查找过程](https://cwlseu.github.io/images/lsh/lsh_why.png)
右边的函数感觉666的，几乎相当于分段函数了。




## 参考文献
1. [“Mining of Massive Datasets” : Anand Rajaraman, Jure Leskovec, and Jeffrey D. Ullman](http://infolab.stanford.edu/~ullman/mmds/book.pdf)
2. [Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem)
3. T. Hofmann Latent Semantic Models for Collaborative Filtering In ACM Transactions on Information Systems, 2004, Vol 22(1), pp. 89–115.