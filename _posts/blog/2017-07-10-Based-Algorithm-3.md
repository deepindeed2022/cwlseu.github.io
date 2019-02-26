---
layout: post
title: "算法的乐趣：递归和指针"
categories: [blog ]
tags: [Algorithm, ]
description: 算法是编程的基础框架，就像是建房子的砖头，生产的原料，爸妈做饭的柴米油盐。没有良好的算法基础，哪里做得出好菜，生产出优质的产品，建造出结实的房子。
---
* content
{:toc}


## Node的定义
```cpp
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};

struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};

```
## 一些比较trick的问题

### 删除List中的一个节点
```cpp
bool delete_node(ListNode* curr)
{
	if(curr->next)
	{
		curr->val = curr->next->val;
		curr->next = curr->next->next;
		return true;
	}
	else
	{
		return false;
	}
}
```

## 将二叉树转化为双向链表
![@算法示意图](https://cwlseu.github.io/images/algorithm/bstconvertlist.jpg)

```cpp
/***************************************************************************
 *
 * 将一颗二叉树转化为双向链表的操作
 * 思路：
 * 采用中序遍历的想法的，对二叉树进行中序遍历，遍历过程就是最终的结果过程，然后将
 * 前后进行指针的重新设置即可。
 * 本题目主要考察递归和指针的操作
 *
 **************************************************************************/
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree == NULL) return pRootOfTree;
        pRootOfTree = ConvertNode(pRootOfTree);
        // 获取头节点的地址，最小的值对应的指针地址
        while(pRootOfTree->left) pRootOfTree = pRootOfTree->left;
        return pRootOfTree;
    }
    // 进行中序遍历
    TreeNode* ConvertNode(TreeNode* root)
    {
        if(root == NULL) return root;
		// 中序遍历左子树
        if(root->left)
        {
            TreeNode *left = ConvertNode(root->left);
            while(left->right) left = left->right;
            left->right = root;
            root->left = left;
        }
        // 中序遍历右子树
        if(root->right)
        {
            TreeNode *right = ConvertNode(root->right);
            while(right->left) right = right->left;
            right->left = root;
            root->right = right;
        }
        return root;
    }
};
```

## 从链表中查找倒数Kth个

本题的思路就是通过两个指针，一个快一个慢，中间相差k个，当其中快指针指向结尾的时候，慢指针指向的位置就是所求。

```cpp
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {  
    	ListNode* fast = pListHead;
    	int i = k;
    	while(fast && i > 0)
    	{
    		fast = fast->next;
    		i--;
    	}
    	if(i != 0) return NULL;

    	ListNode* slow = pListHead;
    	while(fast && slow)
    	{
    		fast  = fast->next;
    		slow = slow->next;
    	}
    	return slow;
    }
};
```

## 红黑树
https://blog.csdn.net/weewqrer/article/details/51866488

### 用途

红黑树和AVL树一样都对插入时间、删除时间和查找时间提供了最好可能的最坏情况担保。对于查找、插入、删除、最大、最小等动态操作的时间复杂度为O(lgn).常见的用途有以下几种：

STL（标准模板库）中在set map是基于红黑树实现的。
Java中在TreeMap使用的也是红黑树。
epoll在内核中的实现，用红黑树管理事件块。
linux进程调度Completely Fair Scheduler,用红黑树管理进程控制块

### 红黑树 VS AVL树

常见的平衡树有红黑树和AVL平衡树，为什么STL和linux都使用红黑树作为平衡树的实现？大概有以下几个原因：

从实现细节上来讲，如果插入一个结点引起了树的不平衡，AVL树和红黑树都最多需要2次旋转操作，即两者都是O(1)；但是在删除node引起树的不平衡时，最坏情况下，AVL需要维护从被删node到root这条路径上所有node的平衡性，因此需要旋转的量级O(logN)，而RB-Tree最多只需3次旋转，只需要O(1)的复杂度

从两种平衡树对平衡的要求来讲，AVL的结构相较RB-Tree来说更为平衡，在插入和删除node更容易引起Tree的unbalance，因此在大量数据需要插入或者删除时，AVL需要rebalance的频率会更高。因此，RB-Tree在需要大量插入和删除node的场景下，效率更高。自然，由于AVL高度平衡，因此AVL的search效率更高。

总体来说，RB-tree的统计性能是高于AVL的


## 参考链接
[1].[关于红黑树的介绍](https://blog.csdn.net/weewqrer/article/details/51866488)