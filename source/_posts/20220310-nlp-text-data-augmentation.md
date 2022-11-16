---
layout: post
title: "NLP 文本场景的数据优化"
categories: [paper]
tags: [自然语言处理]
description: NLP 文本场景的数据优化
date: 2022-03-10 21:05:18
---

## 序言
数据增强（Data Augmentation，简称DA），是指根据现有数据，合成新数据的一类方法。毕竟数据才是真正的效果天花板，有了更多数据后可以提升效果、增强模型泛化能力、提高鲁棒性等。数据增强主要在CV应用中比较常见，然而由于NLP任务天生的难度，类似CV的裁剪方法可能会改变语义，既要保证数据质量又要保证多样性，所以大家在做数据增强时要十分谨慎。

### 数据增强的目的
- 在很多机器学习场景下，没有足够的数据（数据稀缺场景）来训练高质量的模型。
- 提高训练数据的多样性，从而得到在真实场景下（很多没有见过的数据）更好的泛化效果。
- 样本不均衡
- 为了模型安全，应对模型的对抗攻击。


### NLP数据增强研究基本现状[^1]
- 在CV上很成功，逐渐在NLP任务上发现有效
- 在文本分类[^2]领域数据增强方法也比较多，其他任务例如NER，多标签分类等就相对少一些;
- 语言输入是离散，而且一定的文本改变容易引起文本分布的巨大改变，无法做到像图片那样不可见的抖动;
- 一般算法都可以从输入文本空间和文本编码空间进行数据增强。
- 对抗攻击: 相比较CV的对抗，文本的对抗存在很大差异。文本输入为离散的

问题：
- 数据增广在当前迁移学习大背景下的大规模预训练模型上有用吗？

***
## Data Augmentation in NLP

Paraphrasing：对句子中的词、短语、句子结构做一些更改，保留原始的语义
Noising：在保证label不变的同时，增加一些离散或连续的噪声，对语义的影响不大
Sampling：旨在根据目前的数据分布选取新的样本，会生成更多样的数据

> Data Augmentation Approaches in Natural LanguageProcessing: A Survey[^5]


### Paraphrasing

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030348551.png" alt="Alt text|center|600x350" style="zoom:67%;" />

小结: 在尽可能保留句子整体语义的情况下，增加文本丰富度，包括让每个词拥有更加丰富的上下文context，让相似的语义表达有更多样的语法构成，词汇构成等等

### Noiseing
作者给出了以下5种增加噪声的方法：
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030348404.png" alt="Alt text|center|600x600" style="zoom:67%;" />

- **Swapping**：除了交换词之外，在分类任务中也可以交换instance或者sentence
- **Deletion**：可以根据tf-idf等词的重要程度进行删除
- **Insertion**：可以把同义词随机插入句子中
- **Substitution**：把一些词随机替换成其他词（非同义），模拟misspelling的场景。为了避免改变label，可以使用label-independent的词，或者利用训练数据中的其他句子
- **Mixup**：这个方法最近两年比较火，把句子表示和标签分别以一定权重融合，引入连续噪声，可以生成不同label之间的数据，但可解释性较差
总的来说，引入噪声的DA方法使用简单，但会对句子结构和语义造成影响，多样性有限，主要还是提升鲁棒性。
ConSERT时用到的方法：
- 对抗样本
- **Dropout**：也是SimCSE用到的，还有R-drop，都是通过dropout来加入连续噪声
- **Feature Cut-off**：比如BERT的向量都是768维，可以随机把一些维度置为0，这个效果也不错

小结： 增加模型稳健性，在不过多影响training error的前提下，降低模型的复杂度从而降低generalization error, 类比dropout，l2，random noise injection

### Sampling
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030348885.png" alt="Alt text|center|750x500" style="zoom:67%;" />
Sampling是指从数据分布中采样出新的样本，不同于较通用的paraphrasing，**采样更依赖任务，需要在保证数据可靠性的同时增加更多多样性**，比前两个数据增强方法更难。作者整理了4种方法：

- Rules：用规则定义新的样本和label，比如把句子中的主谓进行变换
- Seq2Seq Models：根据输入和label生成新的句子，比如在NLI任务中，有研究者先为每个label（entailment，contradiction，neutral）训一个生成模型，再给定新的句子，生成对应label的。对比之下，paraphrasing主要是根据当前训练样本进行复述
- Language Models：给定label，利用语言模型生成样本，有点像前阵子看的谷歌UDG。有些研究会加个判别模型过滤
- Self-training：先有监督训练一个模型，再给无监督数据打一些标签，有点蒸馏的感觉

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030349468.png" alt="Alt text|center|600x250" style="zoom:67%;" />



### 增强方法选择依据
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030349203.png" alt="三种类别的数据增强方法特点总结" style="zoom:50%;" />

Method Stacking
实际应用时可以应用多种方法、或者一种方法的不同粒度。

作者推荐了两款工具eda[^7]和uda[^8], eda_chinese[^10], nlpaug[^9]

第一，在使用增强的数据时，如果数据质量不高，可以先让模型在增强后的数据上pre-train，之后再用有标注数据训练。如果要一起训练，在增强数据量过大的情况下，可以对原始训练数据过采样

第二，在进行数据增强时注意这些超参数的调整：
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030349147.png" alt="各种方法的超参数" style="zoom:50%;" />
第三，其实增强很多简单数据的提升有限，可以注重困难样本的生成。比如有研究加入对抗训练、强化学习、在loss上下文章等。如果用生成方法做数据增强，也可以在生成模型上做功夫，提升数据多样性。

第四，如果生成错数据可能引入更多噪声，可以增加其他模型对准确性进行过滤。

***
## 分类任务
1、Mixup: Mixup-Transformer: Dynamic Data Augmentation for NLP Tasks

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030349722.png" alt="Alt text" style="zoom:50%;" />
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030349283.png" alt="Alt text|center|500x60" style="zoom: 50%;" />

在数据不足的情况下，只用40%的数据就可以比不应用增强方案的全量数据好。应用Mixup增强方法可以提升2.46%

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030349677.png" alt="Alt text" style="zoom:50%;" />

2、On Data Augmentation for Extreme Multi-label Classification

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030350873.png" alt="Alt text|center|700x300" style="zoom:67%;" />

3、分类算法中的数据增强方法：综述
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030350998.png" alt="Alt text|center|600x400" style="zoom:50%;" />
 <img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030350887.png" alt="Alt text" style="zoom: 67%;" />


这些在线blog或者paper[^1][^2][^3]中提到了很多增强方法，主要有如下特点
- 多分类任务，为英文任务
- 有针对不同应用场景进行分析的增强方法。虽然现在都用预训练模型，但是在数据增强方法中，通过额外的静态词embedding进行数据增强也是常见的方法。

4、EDA
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030350625.png" alt="Alt text" style="zoom:67%;" />

- paper:EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
- github: http://github.com/jasonwei20/eda_nlp
- <img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030350431-20221117002001491.png" alt="Alt text" style="zoom: 67%;" />             <img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030350097.png" alt="Alt text" style="zoom: 67%;" />

EDA主要采用表一中的同义词替换，随机插入，随机交换，随机删除，从可视化结果中来看，增强样本与原始样本分布基本是一致的。
作者给出了在实际使用EDA方法的建议，表格的左边是数据的规模$N_{train}$, 右边$\alpha$是概率、比率
比如同义词替换中，替换的单词数$n=\alpha * l$ , $l$是句子长度。随机插入、随机替换类似.
$p=\alpha *  n_{aug}$ 代表使用EDA方法从每一个句子拓展出的句子数量。

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030350995.png" alt="@作者的一些建议|center|400x250" style="zoom:67%;" />



之后，又有新的AEDA
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030350389.png" alt="Alt text" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030350145.png" alt="Alt text" style="zoom:67%;" />           <img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030351859.png" alt="Alt text" style="zoom:67%;" />


### Text Smoothing
<img src="../../images/nlp/NLP文本场景的数据优化/1646230538691.png" alt="Alt text" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030351711.png" alt="Alt text" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030351853.png" alt="Alt text" style="zoom:67%;" />

```python
	sentence = "My favorite fruit is pear ."
	lambd = 0.1 # interpolation hyperparameter
	mlm.train() # enable dropout, dynamically mask
	tensor_input = tokenizer(sentence, return_tensors="pt")
	onehot_repr = convert_to_onehot(**tensor_input)
	smoothed_repr = softmax(mlm(**tensor_input).logits[0])
	interpolated_repr = lambd * onehot_repr + (1 - lambd) * smoothed_repr
```

-code: https://github.com/1024er/cbert_aug

### PromDA
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030351964.png" alt="Alt text" style="zoom:67%;" />

- paper:https://arxiv.org/pdf/2202.12230.pdf
- 论文目的: low-resource Natural Language Understanding (NLU) tasks

少数据的场景，可能使用PLM不是最优的方案
我们期望构造的数据$\mathcal{T}_{LM}$与已有的数据集$\mathcal{T}$不同，能够从中学习到一些新的信息。
冻结PLMs参数可能有助于在训练过程中进行泛化。然而，寻找合适的离散任务引入并不容易以端到端方式进行优化，而且需要额外的人力。

引入**$soft Prompt$**
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030351292.png" alt="Alt text" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030351728.png" alt="Alt text" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/1646293978880.png" alt="Alt text" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/1646293992580.png" alt="Alt text" style="zoom:67%;" />


### DualCL
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030352998.png" alt="Alt text" style="zoom:67%;" />

- paper: Dual Contrastive Learning: Text Classification via Label-Aware Data Augmentation
- github: https://github.com/hiyouga/Dual-Contrastive-Learning
- 设计主要思想: 将类别与文本表征map到同一个空间

传统自监督对比学习损失函数定义如下左侧公式，但是没有利用标注信息。将标注信息考虑进去，

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030352067.png" alt="Alt text" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030352672.png" alt="Alt text" style="zoom:67%;" />



到目前为止发展起来的监督对比学习似乎是对分类问题的无监督对比学习的一种简单朴素的适配。

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030352328.png" alt="Alt text" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030352671.png" alt="Alt text" style="zoom:67%;" />

- K+1+ 其他文本

- 学习到多个表征，其中1个原来的[CLS],另外K个是用来判断分类的结果的。$$ \hat{y}_i = \arg\max_k(\theta_i^k \cdot z_i)$$

  <img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030352326-20221117002647446.png" alt="Alt text" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030352085.png" alt="Alt text" style="zoom:67%;" />

算法对比结果，少样本与全样本的对比：
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030352508.png" alt="Alt text" style="zoom:67%;" />

### Sample Efficiency of Data Augmentation Consistency Regularization
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353087.png" alt="Alt text" style="zoom:67%;" />

DA-ERM（data augmentation empirical risk minimization）: DAC可以使用未标记的样本，因为可以在不知道真实标签的情况下增加训练样本并执行一致的预测。这绕过了传统算法只能增加标记样本并将其添加到训练集的限制
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353158.png" alt="Alt text" style="zoom:67%;" />

少量数据+data augmentation
少量数据+unlabel data

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353466.png" alt="Alt text" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353005.png" alt="Alt text|center" style="zoom:67%;" />
我们可以看到对标注样本$\phi(x_i)$和增强产生的样本$\phi(x_{i,j})$之间的差异作为惩罚项。

我们从经验和理论上论证了DAC与DA-ERM(用增强样本扩展训练集)相比的优点。理论上，线性回归和逻辑回归的泛化误差更小，两层神经网络的泛化上界更紧。另一个好处是，DAC可以更好地处理由强扩充数据引起的模型错误规范。在经验上，我们提供了关于增广ERM和一致性正则化的比较。这些共同证明了一致性规则化优于DA-ERM的有效性

### ALP: Data Augmentation using Lexicalized PCFGs for Few-Shot Text Classification
- 标题：ALP：基于词汇化PCFGS的Few-Shot文本分类数据增强
- 链接：https://arxiv.org/abs/2112.11916
- 作者：Hazel Kim,Daecheol Woo,Seong Joon Oh,Jeong-Won Cha,Yo-Sub Han
- 机构： Yonsei University, Seoul, Republic of Korea, NAVER AI Lab, Changwon National University, Changwon, Republic of Korea
- 备注：Accepted to AAAI2022

这个是基于文法分析树的方式进行数据增强的

##  NER[^prompt base在NER中的应用]

该任务中需要生成句子和token级别的标签。且序列标注为细粒度的文本任务。
现有的生成模型智能生成没有标签的序列；
启发式的数据增强方法不可行，直接对标签替换或者上下文替换，被注入错误的可能性比较大，相比较分类任务更容易破坏序列上下文关系。

### An Analysis of Simple Data Augmentation for Named Entity Recognition
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353604.png" alt="Alt text" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353756.png" alt="Alt text" style="zoom:67%;" />

- **Label-wise token replacement (LwTR) **：即同标签token替换，对于每一token通过二项分布来选择是否被替换；如果被替换，则从训练集中选择相同的token进行替换。
- **Synonym replacement (SR) **：即同义词替换，利用WordNet查询同义词，然后根据二项分布随机替换。如果替换的同义词大于1个token，那就依次延展BIO标签。
- **Mention replacement (MR) **：即实体提及替换，与同义词方法类似，利用训练集中的相同实体类型进行替换，如果替换的mention大于1个token，那就依次延展BIO标签，如上图：「headache」替换为「neuropathic pain syndrome」，依次延展BIO标签。
- **Shuffle within segments (SiS)** ：按照mention来切分句子，然后再对每个切分后的片段进行shuffle。如上图，共分为5个片段： [She did not complain of], [headache], [or], [any other neurological symptoms], [.]. 。也是通过二项分布判断是否被shuffle（mention片段不会被shuffle），如果shuffle，则打乱片段中的token顺序。

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/1646362972193.png" alt="Alt text" style="zoom:67%;" />

由上图可以看出：
- 各种数据增强方法都超过不使用任何增强时的baseline效果。
-  对于RNN网络，实体提及替换优于其他方法；对于Transformer网络，同义词替换最优。
-  总体上看，所有增强方法一起使用（ALL）会优于单独的增强方法。
-  低资源条件下，数据增强效果增益更加明显；充分数据条件下，数据增强可能会带来噪声，甚至导致指标下降；


### DAGA: Data Augmentatino with a Generation Approach for Low-resource Tagging Tasks
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353938.png" alt="Alt text" style="zoom:67%;" />

DAGA的思想简单来讲就是标签线性化：即将原始的**「序列标注标签」与「句子token」进行混合，也就是变成「Tag-Word」**的形式，如下图：将「B-PER」放置在「Jose」之前，将「E-PER」放置在「Valentin」之前；对于标签「O」则不与句子混合。标签线性化后就可以生成一个句子了，文章基于此句子就可以进行「语言模型生成」了。
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353943.png" alt="Alt text" style="zoom:67%;" />



### SeqMix
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353921.png" alt="Alt text" style="zoom:67%;" />

- 标题: SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup
- 链接: https://rongzhizhang.org/pdf/emnlp20_SeqMix.pdf
- 开源代码: https://github.com/rz-zhang/SeqMix
- 备注: EMNLP 2020
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353427.png" alt="Alt text" style="zoom:67%;" />

### Boundary Smoothing for Named Entity Recognition
<img src="../../images/nlp/NLP文本场景的数据优化/1646817503789.png" alt="Alt text" style="zoom:67%;" />

- 标题: 针对命名实体识别的span类的算法的边界平滑
- code: https://github.com/syuoni/eznlp

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030353619-20221117002747265.png" alt="Alt text" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/202209030354558.png" alt="Alt text" style="zoom:50%;" />

An example of hard and smoothed boundaries. The example sentence has ten tokens and two entities of spans (1, 2) and (3, 7), colored in red and blue, respectively. The first subfigure presents the entity recognition targets of hard boundaries. The second subfigure presents the corresponding targets of smoothed boundaries, where the span (1, 2) is smoothed by a size of 1, and the span (3, 7) is smoothed by a size of 2. 其中周边区域有$\epsilon$的概率会被赋值，此时原标注位置值为$1 - \epsilon$，周边区域$D$赋值$\epsilon / D$,

对NER标签位置的平滑处理，提升模型的泛化性。边界平滑可以防止模型对预测实体过于自信，从而获得更好的定标效果。D一般不用太大，1或者2即可， $\epsilon$一般取[0.1, 0.2, 0.3]
<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030353317.png" alt="Alt text" style="zoom:67%;" />



## 参考文献

[^1]: Steven Y. Feng, Varun Gangal, Jason Wei, Sarath Chandar, Soroush Vosoughi, Teruko Mitamura, & Eduard Hovy (2021). A Survey of Data Augmentation Approaches for NLP Meeting of the Association for Computational Linguistics.

[^2]: Markus Bayer, Marc-André Kaufhold, & Christian Reuter (2021). A Survey on Data Augmentation for Text Classification.. arXiv: Computation and Language.

[^3]: Amit Chaudhary(2020). A Visual Survey of Data Augmentation in NLP. https://amitness.com/2020/05/data-augmentation-for-nlp

[^4]: Liu, P. ,  Yuan, W. ,  Fu, J. ,  Jiang, Z. ,  Hayashi, H. , &  Neubig, G. . (2021). Pre-train, prompt, and predict: a systematic survey of prompting methods in natural language processing.

[^5]: Li, B. , Hou, Y. , & Che, W. . (2021). Data augmentation approaches in natural language processing: a survey.

[^6]: https://zhuanlan.zhihu.com/p/91269728

[^7]:  https://github.com/jasonwei20/eda_nlp

[^8]: https://github.com/google-research/uda

[^9]: https://github.com/makcedward/nlpaug

[^10]: https://github.com/zhanlaoban/eda_nlp_for_Chinese

[^11]: EDA/ Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks

[^prompt base在NER中的应用]:https://zhuanlan.zhihu.com/p/462332297

[^中文词向量]: https://github.com/Embedding/Chinese-Word-Vectors