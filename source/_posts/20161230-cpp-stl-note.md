---
layout: post
title: STL容器底层实现
categories: [blog]
tags: [tools]
date: 2016-12-30 09:25:24
description: "STL作为C++开发中常用的库，包括容器、迭代器以及常用算法等，这里单独将常用容器拿出来看看"
---

## 引言
STL全程是Standard Template Library，也就是这个库通过C++ Template的方式实现的标准库。这里面包括容器、迭代器、仿函数、常用的基本算法等。
几乎所有的代码都采用了模板类或者模板函数，这相比传统的由函数和类组成的库来说提供了更好的代码重用机会。本文主要对日常开发中常用的容器进行总结。

## **vector**

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030848232.png)

vector 的数据安排以及操作方式，与 array 非常相似。两者的唯一区别在于空间的运用的灵活性。array 是静态空间，一旦配置了就不能改变，vector 是动态数组。在**堆上分配空间**。vector 是动态空间，随着元素的加入，它的内部机制会自行扩充空间以容纳新元素（有保留内存，如果减少大小后内存也不会释放。如果新值>当前大小时才会再分配内存，这大大影响了 vector 的效率）。因此，vector 的运用对于内存的合理利用与运用的灵活性有很大的帮助，我们再也不必因为害怕空间不足而一开始要求一个大块的 array。

vector 动态增加大小，并不是在原空间之后持续新空间（因为无法保证原空间之后尚有可供配置的空间），而是以原大小的两倍**另外配置**一块较大的空间，然后将原内容拷贝过来，然后才开始在原内容之后构造新元素，并释放原空间。因此，对 vector 的任何操作，一旦引起空间重新配置，同时指向原vector 的所有迭代器就都失效了。

对最后元素操作最快（在后面添加删除最快），此时一般不需要移动内存。对中间和开始处进行添加删除元素操作需要移动内存。**如果你的元素是结构或是类,那么移动的同时还会进行构造和析构操作，所以性能不高**（最好将结构或类的指针放入 vector 中，而不是结构或类本身，这样可以避免移动时的构造与析构）。访问方面，对任何元素的访问都是 O(1)，也就是常数时间的。 

> **总结：**

vector 常用来保存需要**经常进行随机访问**的内容，并且不需要经常对中间元素进行添加删除操作。

STL源码剖析：

https://blog.csdn.net/weixin_40673608/article/details/87103742

## **list**

相对于 vector 的连续空间，list 就显得复杂许多，它的好处是每次插入或删除一个元素，就配置或释放一个元素空间，元素也是在堆中。因此，list 对于空间的运用有绝对的精准，一点也不浪费。而且，对于**任何位置的元素插入或元素移除，永远是常数时间**。STL 中的list 底层是一个双向链表，而且是一个环状双向链表。这个特点使得它的随即存取变的非常没有效率，因此它没有提供 [] 操作符的重载。

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209040049339.png)

> **总结：**

如果你喜欢经常添加删除大对象的话，那么请使用 list；

要保存的对象不大，构造与析构操作不复杂，那么可以使用 vector 代替。

list<指针> 完全是性能最低的做法，这种情况下还是使用 vector<指针> 好，因为指针没有构造与析构，也不占用很大内存

## **deque**

deque 是一种双向开口的连续线性空间，元素也是在堆中。所谓双向开口，意思是可以在队尾两端分别做元素的插入和删除操作。deque 和 vector 的最大差异，一在于 deque 允许于常数时间内对起头端进行元素的插入或移除操作，二在于deque没有所谓容量观念，因为**它是动态地以分段连续空间组合而成，随时可以增加一段新的空间并链接在一起**。换句话说，像 vector 那样“因旧空间不足而重新配置一块更大空间，然后复制元素，再释放旧空间”这样的事情在 deque 是不会发生的。它的保存形式如下:

> [堆1] --> [堆2] -->[堆3] --> ...

deque 是由一段一段的定量连续空间构成。一旦有必要在 deque 的前端或尾端增加新空间，便配置一段定量连续空间，串接在整个 deque 的头端或尾端。deque 的最大任务，便是在这些分段的定量连续空间上，维护其整体连续的假象，并提供随机存取的接口。避开了“重新配置，复制，释放”的轮回，代价则是复杂的迭代器架构。因为有分段连续线性空间，就必须有中央控制器，而为了维持整体连续的假象，数据结构的设计及迭代器前进后退等操作都颇为繁琐。

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030848812.png)

**deque 采用一块所谓的 map 作为主控。这里的 map 是一小块连续空间，其中每个元素都是指针，指向另一段连续线性空间，称为缓冲区。缓冲区才是 deque 的存储空间主体**。（ 底层数据结构为一个中央控制器和多个缓冲区）SGI STL 允许我们指定缓冲区大小，默认值 0 表示将使用 512 bytes 缓冲区。

支持[]操作符，也就是支持随即存取，可以在前面快速地添加删除元素，或是在后面快速地添加删除元素，然后还可以有比较高的随机访问速度和vector 的效率相差无几。deque 支持在两端的操作：`push_back`,`push_front`,`pop_back`,`pop_front`等，并且在两端操作上与 list 的效率也差不多。

在标准库中 vector 和 deque 提供几乎相同的接口，在结构上区别主要在于在组织内存上不一样，**deque 是按页或块来分配存储器的，每页包含固定数目的元素；相反 vector 分配一段连续的内存，vector 只是在序列的尾段插入元素时才有效率**，而 deque 的分页组织方式即使在容器的前端也可以提供常数时间的 insert 和 erase 操作，而且在体积增长方面也比 vector 更具有效率。

详细实现信息可以参考: http://c.biancheng.net/view/6908.html

**如果deque中的map满了怎么办？**

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030848240.png)**总结：**

- vector 是可以快速地在最后添加删除元素，并可以快速地访问任意元素；
- list 是可以快速地在所有地方添加删除元素，但是只能快速地访问最开始与最后的元素；
- deque 在开始和最后添加元素都一样快，并提供了随机访问方法，像vector一样使用 [] 访问任意元素，但是随机访问速度比不上vector快，因为它要内部处理堆跳转。deque 也有保留空间。另外，由于 deque 不要求连续空间，所以可以保存的元素比 vector 更大。还有就是在前面和后面添加元素时都不需要移动其它块的元素，所以性能也很高。

因此在实际使用时，如何选择这三个容器中哪一个，一般应遵循下面的原则：

- 如果你需要高效的随即存取，而不在乎插入和删除的效率，使用 vector；
- 如果你需要大量的插入和删除，而不关心随即存取，则应使用 list；
- 如果你需要随即存取，而且关心两端数据的插入和删除，则应使用deque。

## **stack**

stack 是一种先进后出（First In Last Out , FILO）的数据结构。它只有一个出口，stack 允许新增元素，移除元素，取得最顶端元素。但除了最顶端外，没有任何其它方法可以存取stack的其它元素，stack不允许遍历行为。

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030848810.png)

以某种容器（ **一般用 list 或 deque 实现，封闭头部即可**，不用 vector 的原因应该是容量大小有限制，扩容耗时）作为底部结构，将其接口改变，使之符合“先进后出”的特性，形成一个 stack，是很容易做到的。deque 是双向开口的数据结构，若以 deque 为底部结构并封闭其头端开口，便轻而易举地形成了一个stack。因此，SGI STL 便以 deque 作为缺省情况下的 stack 底部结构，由于 stack 系以底部容器完成其所有工作，而具有这种“修改某物接口，形成另一种风貌”之性质者，称为 adapter（配接器），因此，STL stack 往往不被归类为 container(容器)，而被归类为 container adapter。

## **queue**

queue 是一种先进先出（First In First Out,FIFO）的数据结构。它有两个出口，queue 允许新增元素，移除元素，从最底端加入元素，取得最顶端元素。但除了最底端可以加入，最顶端可以取出外，没有任何其它方法可以存取 queue 的其它元素。

以某种容器 （ **一般用 list 或 deque 实现，封闭头部即可** ，不用 vector 的原因应该是容量大小有限制，扩容耗时 ）作为底部结构，将其接口改变，使之符合“先进先出”的特性，形成一个 queue，是很容易做到的。deque 是双向开口的数据结构，若以 deque 为底部结构并封闭其底部的出口和前端的入口，便轻而易举地形成了一个 queue。因此，SGI STL 便以 deque 作为缺省情况下的 queue 底部结构，由于 queue 系以底部容器完成其所有工作，而具有这种“修改某物接口，形成另一种风貌”之性质者，称为 adapter（配接器），因此，STL queue 往往不被归类为container(容器)，而被归类为 container adapter。

stack 和 queue 其实是适配器，而不叫容器，因为是对容器的再封装。

## **heap**

heap 并不归属于 STL 容器组件，它是个幕后英雄，扮演 priority queue（优先队列）的助手。priority queue 允许用户以任何次序将任何元素推入容器中，但取出时一定按从优先权最高的元素开始取。按照元素的排列方式，heap 可分为 max-heap 和 min-heap 两种，前者每个节点的键值(key)都大于或等于其子节点键值，后者的每个节点键值(key)都小于或等于其子节点键值。因此， max-heap 的最大值在根节点，并总是位于底层array或vector的起头处；min-heap 的最小值在根节点，亦总是位于底层array或vector起头处。STL 供应的是 max-heap，用 C++ 实现。

## **priority_queue**

priority_queue 是一个拥有权值观念的 queue，它允许加入新元素，移除旧元素，审视元素值等功能。由于这是一个 queue，所以只允许在底端加入元素，并从顶端取出元素，除此之外别无其它存取元素的途径。priority_queue 带有权值观念，其内的元素并非依照被推入的次序排列，而是自动依照元素的权值排列（通常权值以实值表示）。权值最高者，排在最前面。缺省情况下 priority_queue 系利用一个 max-heap 完成，后者是一个以vector 表现的 complete binary tree.max-heap 可以满足 priority_queue 所需要的“依权值高低自动递减排序”的特性。

priority_queue 完全**以底部容器（一般为vector为底层容器）作为根据**，再加上 heap 处理规则，所以其实现非常简单。缺省情况下是以 vector 为底部容器。queue 以底部容器完成其所有工作。具有这种“修改某物接口，形成另一种风貌“”之性质者，称为 adapter(配接器)，因此，STL priority_queue 往往不被归类为 container(容器)，而被归类为 container adapter。

## **set 和 multiset 容器**

set 的特性是，所有元素都会根据元素的键值自动被排序。set 的元素不像 map 那样可以同时拥有实值(value)和键值(key)，set 元素的键值就是实值，实值就是键值，set不允许两个元素有相同的值。set 底层是通过红黑树（RB-tree）来实现的，由于红黑树是一种**平衡二叉搜索树**，自动排序的效果很不错，所以标准的 STL 的 set 即以 RB-Tree 为底层机制。又由于 set 所开放的各种操作接口，RB-tree 也都提供了，所以几乎所有的 set 操作行为，都只有转调用 RB-tree 的操作行为而已。

**multiset**的特性以及用法和 set 完全相同，唯一的差别在于它**允许键值重复**，因此它的插入操作采用的是底层机制是 RB-tree 的 insert_equal() 而非 insert_unique()。

## **map 和 multimap 容器**

map的特性是，所有元素都会根据元素的键值自动被排序。map 的所有元素都是 pair，同时拥有实值（value）和键值（key）。pair 的第一元素被视为键值，第二元素被视为实值。map不允许两个元素拥有相同的键值。由于 RB-tree 是一种平衡二叉搜索树，自动排序的效果很不错，所以标准的STL map 即以 RB-tree 为底层机制。又由于 map 所开放的各种操作接口，RB-tree 也都提供了，所以几乎所有的 map 操作行为，都只是转调 RB-tree 的操作行为。

multimap 的特性以及用法与 map 完全相同，唯一的差别在于它允许键值重复，因此它的插入操作采用的是底层机制 RB-tree 的 insert_equal() 而非 insert_unique。 

## **unordered_set、unordered_map**

hashtable作为unordered_set和unordered_map的底层数据结构，是隐藏起来的，正常不会直接用到它，但要理解好unordered_set和unordered_map，需要先理解好hashtable

hashtable的底层数据结构是vector，vector中的元素是链表

vector代表篮子，初始化大小为53（GNU的做法），存的是结点指针

元素放进来的时候，会经过一个hash函数，找到对应的篮子，然后连在篮子的后面

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030848924.png)

当放进去的元素的数量超过vector的长度的时候，就会执行rehashing：

vector的大小会先变大为2倍，但不一定是2倍，会变成2倍左右的一个素数

vector变大后，原先的每一个元素都需要重新用hash函数计算并放到对应的篮子里

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030848225.png)

如果还需要rehashing的话，vector的大小会按照以下增长，这些数字是已经先算好的，vector在扩充的时候，不会再花时间去计算，而是直接在下面抓取对应的大小

![img](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/img/202209030848474.png)
