---
layout: post
title: 模式识别算法课程笔记
categories: [blog ]
tags: [机器学习]
description: 模式识别经典算法介绍。主要是参加模式识别课程中一些基本概念的记录与总结。
---

* content
{:toc}

## 概述

    模式识别它本质上讲是要找到一个影射的过程。尽管模式识别过程首先是要获取信号，比如你要识别苹果，要把苹果树拍下来，找到苹果在
哪个地方，把这些数据进行了预处理，然后再把它进行特征描述，当然你可以进行特征抽取，显性的特征或者隐性的特征，然后进行识别。机器的
模式识别能力反映了机器智能的类人程度。人工智能到现在已经有61年的时间，人工智能是模拟人的智能，模式识别世界上是模拟人的识别能力，所以应该说模式识别是人工智能一个非常重要的研究方向和研究领域，模式识别是模拟人的识别能力。
    在理论创新不断取得突破的同时，应用不断地拓展。这些年深度学习的热潮，人工智能的热潮，很多方面都是因为在模式识别得益于深度学习
的发展，很多方面都是模式识别方面的突破，比如说大家非常熟悉的早些年的VOC，图象识别、语音识别，都是模式识别典型的问题。谭铁牛老师在《模式识别研究的回顾与展望》中介绍模式识别发展的现状：
* 面向特定任务的模式识别已经取得突破性的进展，有的性能可以与人媲美，甚至超过人。
* 统计与基于神经网络的模式识别目前占主导地位，深度学习开创了新局面。
* 通用模式识别系统依然任重道远，关键问题是我们需不需要通用模式识别系统。如果需要，这样的系统还任重道远。
* 鲁棒性、自适应性和可泛化性是进一步发展的三大瓶颈。
这也讲是我们做这方面需要思考的问题。

## 研究方向与问题
### 生物启发的模式识别
历史上模式识别与计算机视觉的发展，很多方面受益于生物机制的启发。比如说我博士时期做了纹理分析，当时我用得最多的就是Gabor函数，通过这个函数发现人的感受和Gabor函数非常相似，所以我当时做博士论文的时候用了这个函数，我发现效果非常好，
从哪些方面可以借鉴人类大脑或者生物系统有哪些方面值得我们学习，分了四个层次，从微观到宏观都有一些值得我们借鉴的机理。

* 当然最微观的层面，我们的大脑有1000亿个神经元，它的链接就更多了，神经元有很多不同的类型，它有兴奋性、有抑制性的，在这里面如何把神经元得到体现，神经突触有功能可塑性、结构可塑性等等。
在神经元层次，神经元有不同的类型，有的是兴奋型，有的是抑制型的，引用不同类型的神经元，它的效果有不同的提升，同时神经元的类型可以自动学习获得。所以深度神经网络不是单一的类型，它有多种类型。Hinton教授大家很熟悉，他模拟神经元的噪声特性，在渲染过程中有的隐藏节点不考虑，暂时简化了网络结构，提升了网络的效率，从一定程度上解决了小样本的问题，解决了神经元的机制问题。神经元的放电效应也不一样，特别是神经元可塑性机制，Bengio借鉴这个机制发现确实能提高兴奋。至于怎么借鉴，怎么建模，需要参考相关的文章。

* 在神经回路这个层次，同样有很多值得我们借鉴的东西，比如说深度学习、深度神经网络，目前绝大多数都是前向链接，实际上在人的大脑上上还有后向和侧向的。
回路这个层次，有前向链接、反向链接和侧向链接。这是大家都非常熟悉的前向链接，现在大多数的深度学习都是前向链接的，包括AlexNet和VGG都是前向链接的。这里我要重点说的是侧向链接，就是在同一层的侧向链接，这是我引用清华大学的教授发表的文章，它确实可以提升兴奋。反向链接是我们自己的工作，是我的一个博士生做的，试图通过反向链接把高层的信息往低层再传递，发现效果也非常好。

* 还有更宏观的功能区域，可以有多脑区，不同脑功能区的协同等等。
另外一个层次是功能区域，就是更宏观的区域，在不同的脑区有不同的功能，中间怎么协同，或者不同的脑区协同完成一件任务，也有很多值得我们借鉴的。这是我的一个学生做的研究，把不同的脑区的功能协同机制借鉴到多任务训练学习方面，也取得不错的效果。再一个是多通道协同，这篇文章也许各位知道，这是牛津大学的教授做的，大家知道视觉通路有一个背侧通路，还有一个腹侧通路，他们借鉴这个机理提出双同路的卷积网络，一路负责挖掘表观信息，一路负责获取运动信息，这个效果非常不错。注意和记忆机制的研究比较多，这个比较好理解，记忆和选择性机制、注意机制，计算机视觉里面用得比较多。

* 最宏观的就是在行为层次的学习机制，我们人是怎么学习的，在学习机制方面，学习的过程我们可以借鉴，学习的方法我们可以借鉴，还有学习的效果也可以借鉴，所以在这几个层面有很多东西值得我们借鉴。

- 宏观层面就是行为层次，人的行为，特别是学习过程的行为有什么机理值得我们学习？机理方面学习的借鉴、过程方面的借鉴、方法方面的借鉴。学习的过程有发育学习、强化学习，方法有迁移学习、知识学习，学习的效果有生成学习、概念学习。
- 模仿生物从简单到复杂的学习过程，在积累的过程中拓展学习范围，人的学习就是这样的机理。这个研究是试图借鉴人从小到大学习过程的机理。
- 强化学习这一点大家都非常熟悉，这里我特别要说的是跟环境的交互，我们在成长的过程中跟环境的交互，对我们获取外部世界的信息，获取知识至关重要。我经常讲如果我站在这个地方，那个地方看不清楚，我会主动的动一动，通过跟环境交互来学习，从而增强对环境的自适应性。
- 迁移学习，这一点也是我们人具备的，如果你的乒乓球、羽毛球打得好，说不定网球打起来也会学得更快一点。
- 还有一个是知识学习。人有这个本领，在识别一个东西的时候我们会用大量的先验知识，再结合现场观测到的信息，也就是先验知识和数据的结合，来有效识别你所看到的物体，这是我们人都能做到的，基于这样的机制，计算机也能做得很好。
- 还有生成学习，现在这一点很火，它是通过产生更多的原始数据样本分布一致的大量的深层数据，一方面可以解决小样本的问题、训练数据的问题，同时可以提高算法的垄断性和泛化能力、自适应性，现在大家非常关注这方面的工作。
- 概念学习，就像一个小孩看到一个苹果，后来再看到更多的苹果他都能识别，这是小样本学习，它是典型的通过统计的方法，学习了规则，规则是结构模式识别所需要的，所以从统计方法获得规则，然后用这些规则来进行识别，所以它是一个从统计到结构，然后从结构到统计相结合的模式识别方法，这是一个很有前途的方法。

### 鲁棒性

如果鲁棒性的问题解决了，计算机视觉的问题也就都解决了，。而现实生活中，鲁棒性是我们经常碰到的，如果你的算法不鲁棒，基本上就没什么用，所以解决鲁棒问题很重要，当然解决鲁棒问题另外一个出路就是找到**鲁棒的特征**。比如说人脸图像，你的光照变一点、姿势变一点，人脸的识别也跟随变化，在跨媒体、多源异质的视觉大数据中找到具有较好泛化星和不变形的表达，这就是鲁棒要解决的问题。

### 结构和统计相结合的模式识别新理论

这是一个值得关注的发展趋势，目前的研究还不是太多，因为统计方法和结构的方法各有自己的优缺点。结构方法的原理很清晰，描述很紧凑，样本要求也少，但是它没有充分利用所有的数据。统计模式识别应用范围光，但是它对数据质量要求高，而且原理不清晰，有的时候不可解释。

### 数据和知识相结合

现在大家都强调数据的重要性，数据当然很重要，但是数据不是一切，前面讲到借鉴神经回路链接的过程中，提到了反向链接，也就是说从上一层的信息传递到下一层，把知识传递到下一层，数据和知识相结合也是一个非常重要的发展方向，所以数据和知识相结合非常重要。

### 以互联网为中心的模式识别。

互联网上有太多的数据，有大数据、知识、交互、众包等等，所以可以说是人类智能+机器智能的混合载体，怎么样把互联网的海量数据充分应用起来，对于推动模式识别的研究和发展非常重要，同时整个模式识别系统流程完全基于互联网信息，同时互联网上很多的任务需要模式识别完成，互联网上这么多的信息，反映什么样的态势，它是需要数据的挖掘、模式识别和分析。

## 基本概念
线性分类问题中，为了方便将权重与偏置统一到一个矩阵运算中进行计算，常常通过对权重矩阵添加全部为1的向量，从而将问题简化。
$$g(x) = w^Tx + w_0 = a^Ty$$
其中$a^T$为$x^T$的增广矩阵。对于原训练数据添加全部为1的一项，用来表示偏置的权重，从而将偏置和原始权重统一到一个矩阵乘法之中，这个预处理数据的过程叫数据的增广。

**增广样本**  
![@对于原训练数据添加全部为1的一项，用来表示偏置的权重，从而将偏置和原始权重统一到一个矩阵乘法之中，这个预处理数据的过程叫数据的增广](https://cwlseu.github.io/images/classifiction-pattern/1.jpg)

## 方法论-贝叶斯

长久以来，人们对一件事情发生或不发生，只有固定的0和1，即要么发生，要么不发生，从来不会去考虑某件事情发生的概率有多大，不发生的概率又是多大。而且事情发生或不发生的概率虽然未知，但最起码是一个确定的值。比如如果问那时的人们一个问题：“有一个袋子，里面装着若干个白球和黑球，请问从袋子中取得白球的概率是多少？”他们会立马告诉你，取出白球的概率就是1/2，要么取到白球，要么取不到白球，即$\theta$只能有一个值，而且不论你取了多少次，取得白球的概率$\theta$始终都是1/2，即不随观察结果X 的变化而变化。

这种频率派的观点长期统治着人们的观念，直到后来一个名叫 **Thomas Bayes** 的人物出现。

> Thomas Bayes
贝叶斯(约1701-1761) Thomas Bayes，英国数学家。约1701年出生于伦敦，做过神甫。1742年成为英国皇家学会会员。1761年4月7日逝世。贝叶斯在数学方面主要研究概率论。他首先将归纳推理法用于概率论基础理论，并创立了贝叶斯统计理论，对于统计决策函数、统计推断、统计的估算等做出了贡献。贝叶斯所采用的许多术语被沿用至今。贝叶斯思想和方法对概率统计的发展产生了深远的影响。今天，贝叶斯思想和方法在许多领域都获得了广泛的应用。


频率派把需要推断的参数$\theta$看做是固定的未知常数，即概率虽然是未知的，但最起码是确定的一个值，同时，样本X 是随机的，所以频率派重点研究样本空间，大部分的概率计算都是针对样本X 的分布；
**最大似然估计(MLE)** 和 **最大后验估计(MAP)** 都是把待估计的参数看作一个拥有固定值的变量，只是取值未知。通常估计的方法都是找使得相应的函数最大时的参数；由于MAP相比于MLE会考虑先验分布的影响，所以MAP也会有**超参数**，它的超参数代表的是一种信念(belief)，会影响推断(inference)的结果。比如说抛硬币，如果我先假设是公平的硬币，这也是一种归纳偏置(bias)，那么最终推断的结果会受我们预先假设的影响。

## 贝叶斯决策论

"贝爷是站在食物链顶端的男人",可是这也不妨碍贝叶斯成为模式识别中的校长。著名的贝叶斯概率和全概率成为模式识别入门的法宝与门槛。有了这工具，模式识别不是问题；不理解这理论，就痛苦地哀嚎吧。
贝叶斯派既然把看做是一个随机变量，所以要计算的分布，便得事先知道的无条件分布，即在有样本之前（或观察到X之前），有着怎样的分布呢？
比如往台球桌上扔一个球，这个球落会落在何处呢？如果是不偏不倚的把球抛出去，那么此球落在台球桌上的任一位置都有着相同的机会，即球落在台球桌上某一位置的概率服从均匀分布。这种在实验之前定下的属于基本前提性质的分布称为先验分布，或的无条件分布。
贝叶斯派认为待估计的参数是随机变量，服从一定的分布，而样本X是固定的，由于样本是固定的，所以他们重点研究的是参数的分布。
贝叶斯及贝叶斯派思考问题的固定模式 `$先验分布（\pi(\theta） + 样本信息X  =  后验分布(\pi(\theta|x))$` 
上述思考模式意味着，新观察到的样本信息将修正人们以前对事物的认知。换言之，在得到新的样本信息之前，人们对的认知是先验分布，在得到新的样本信息后，人们对的认知为。

### 最大似然估计

最大似然估计，只是一种概率论在统计学的应用，它是参数估计的方法之一。说的是**已知某个随机样本满足某种概率分布**，但是其中具体的参数不清楚，参数估计就是通过若干次试验，观察其结果，利用结果推出参数的大概值。最大似然估计是建立在这样的思想上：已知某个参数能使这个样本出现的概率最大，我们当然不会再去选择其他小概率的样本，所以干脆就把这个参数作为估计的真实值。
求最大似然函数估计值的一般步骤： 
* 写出似然函数
* 对似然函数取对数，并整理
* 求导数
* 解似然方程

满足**KKT条件**的凸优化问题常常使用拉格朗日算子转化为似然函数的极值问题。通过求解似然函数的极值点，从而求得最优解。

### 案例：EM算法

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

### 贝叶斯参数估计
## 感知器

感知器是由美国计算机科学家罗森布拉特（F.Roseblatt）于1957年提出的。感知器可谓是最早的人工神经网络。单层感知器是一个具有一层神经元、采用阈值激活函数的前向网络。通过对网络权值的训练，可以使感知器对一组输人矢量的响应达到元素为0或1的目标输出，从而实现对输人矢量分类的目的。

### 模型

### 局限性

由于感知器神经网络在结构和学习规则上的限制，其应用也有一定的局限性。
首先，感知器的输出只能取0或1。
其次，单层感知器只能对线性可分的向量集合进行分类。

## 线性不可分问题

### MSE最小平方误差准侧

可以参考（http://blog.csdn.net/xiaowei_cqu/article/details/9004193）
在线性不可分的情况下，不等式组不可能同时满足。一种直观的想法就是，希望求一个a*使被错分的样本尽可能少。这种方法通过求解线性不等式组来最小化错分样本数目，通常采用搜索算法求解。

为了避免求解不等式组，通常转化为方程组：$a^Ty_i = b_i >0, i = 1,2,..，N$
矩阵的形式为Ya = b，方程组的误差为$e = Ya - b$, 
可以求解方程组的最小平方误差求解，即：$a*: minJ_s(a)$
$$J_s(a) = ||Ya - b||^2 = \sum_{i=1}^N(a^Ty_i - b_i)^2$$

最小误差求解方法有两种，一种是基于矩阵理论求解伪逆矩阵,然后求得最佳位置；另一种就是基于梯度下降的方法，类似感知机一样进行单步修正.其中k为迭代次数

```python

def LMS():
	init a, b, criteion theta, delta, k = 0
	while delta_k*(b_k - a_t*y_k)*y_k > theta:
		k += 1
		a = a + delta_k*(b_k - a[t]*y_k)*y_k
	return a 
```

### Ho-Kashyap 算法

这是一种修改的MSE。MSE只能使 $||Ya-b||$极小。
如果训练样本恰好线性可分，那么存在$a*，b*$，满足$Ya*=b*>0$
如果我们知道b*，就能用MSE求解到一个分类向量，但是我们无法预知b*
所以，$$J(a,b)=||Ya-b||^2, b>0$$
用梯度下降求解，得到：$a(k)=inv(Y) b(k)$,  $inv(Y)$表示$Y$的伪逆

下面是Ho-Kashyap及其修改算法的实现，可以在type中选择

```matlab
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
```

## 总结

总的来说，误差最小是模式分类的原则。在保证正确率的情况下，提高响应速度。当然，现在还是处于正确率不断提高的阶段吧。

必看论文：[https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)

## 模式识别的一些应用

* [微软OCR](http://www.csdn.net/article/2015-03-30/2824348)有了新的进展。
  * 说道OCR,不得不提Google的[tesseract](http://code.google.com/p/tesseract-ocr/)，[tesseract](http://www.cs.cmu.edu/~antz/sarma_icse2009.pdf)据个人亲测，能够实现对**自然场景**下的文字获取，当然图片中文字的清晰程度对输出结果的影响还是很大大的，这方面可以研究一下。
* 除了OCR之外，科大讯飞是被人熟知是源于其**语音识别**，还有现在汽车的**自动驾驶技术**，甚至是自动挡车在某种程度上也算是一种经过学习之后的成果吧。
* 图像分类任务
* 虹膜识别

## 参考链接

1. [The Science of Pattern Recognition. Achievements and Perspectives](https://link.springer.com/chapter/10.1007/978-3-540-71984-7_10)

	Automatic pattern recognition is usually considered as an engineering area which focusses on the development and evaluation of systems that imitate or assist humans in their ability of recognizing patterns. It may, however, also be considered as a science that studies the faculty of human beings (and possibly other biological systems) to discover, distinguish, characterize patterns in their environment and accordingly identify new observations. The engineering approach to pattern recognition is in this view an attempt to build systems that simulate this phenomenon. By doing that, scientific understanding is gained of what is needed in order to recognize patterns, in general.

2. [谭铁牛：模式识别研究的回顾与展望](http://finance.jrj.com.cn/tech/2017/07/07120722713184.shtml)
	模式识别是人类最重要的智能行为，是智能化时代的关键使能技术；
　　鲁棒性、自适应性和可泛化性是模式识别面临的三大瓶颈；
　　向生物系统学习、结构与统计相结合，数据与知识相结合，并充分利用海量的互联网数据，是特别值得关注的研究方向。
3. [类脑智能研究的回顾与展望](http://cjc.ict.ac.cn/online/onlinepaper/zy-20151228131518.pdf)
4. [贝叶斯参数估计](https://blog.csdn.net/jinping_shi/article/details/53444100)