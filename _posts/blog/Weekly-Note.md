---
layout: post
title: 余额宝
tags: [计算机视觉, CV算法] 
categories: [blog ]
notebook: 视觉算法
---

* content
{:toc}


## ResNet & ResNeXt

- 论文：Aggregated Residual Transformations for Deep Neural Networks
- 论文链接：https://arxiv.org/abs/1611.05431
- PyTorch代码：https://github.com/miraclewkf/ResNeXt-PyTorch

作者的核心创新点就在于提出了 aggregrated transformations，用一种平行堆叠相同拓扑结构的blocks代替原来 ResNet 的三层卷积的block，在不明显增加参数量级的情况下提升了模型的准确率，同时由于拓扑结构相同，超参数也减少了，便于模型移植。

- https://blog.csdn.net/hejin_some/article/details/80743818
- https://www.cnblogs.com/bonelee/p/9031639.html


## Boost Pool程序库

## 缺省情况下C++在global作用域内

```cpp
void* operator new(std::size_t) throw(std::bad_alloc);
void* operator new(std::size_t, void*) throw();
void* operator new(std::size_t, const std::nothrow_t&) throw();
```

## 接口与实现分离
关键在于声明的依存性替换定义的依存性。
如果使用object reference或者obj pointers可以完成任务，就不要使用objects。你可以只依靠一个类型声明式
就定义出指向该类型的references和pointers

## 1. 从前序与中序遍历序列构造二叉树
根据一棵树的前序遍历与中序遍历构造二叉树。
> 注意:你可以假设树中没有重复的元素。
### 例如，
给出
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7

### 分析

前序遍历顺序是遍历根节点，左子树，右子树，而中序遍历则是左子树，根节点，右子树，因此这类题目的解题思路是根据前序遍历的第一个元素确定根节点，然后在中顺遍历中找到根节点的位置。在中序遍历的左侧是左子树，右侧是右子树。
如上面的例子，首先我们根据前序的第一个节点确定3是根节点，那么在中序遍历结果中找到3，那么中序遍历结果中左侧的序列【9】则是3为根节点的左子树的中序结果，而右侧的序列【15，20，7】则是右子树的中序结果。
    确定了左右子树，继续在左子树的中序遍历结果中找到出现在先序遍历结果的元素，因为在先序遍历结果首先出现的一定是子树的根节点。如本题，左子树的中序遍历结果为【9】，只有一个元素，那么一定是9先出现在先序的结果中，因此左子树根节点为9。右子树的中序遍历结果为【15，20，7】，那么首先出现在先序遍历结果【3，9，20，15，7】的元素是20，那么20是右子树的根节点。
    因为左子树根节点9在其子树对应的中序结果【9】中没有左侧和右侧的序列，那么9则是一个叶子节点。而右子树根节点20在其对应子树的中序结果【15，20，7】中存在左侧序列【15】和右侧序列【7】，那么【15】对应的则是以20为根节点的左子树的中序结果，而【7】则是以20为根节点的右子树的中序结果。循环递归上面的过程构造子树。
反应到程序中需要解决两个重要的问题：
1. 先序遍历结果的第一个元素（根节点）在中序遍历结果中的位置如何确定？
2. 左子树中序遍历子序列的根节点，即左子树的根节点如何确定？同样，右子树中序遍历子序列的根节点，即右子树的根节点如何确定？

代码
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.size()==0) return NULL;    //空树
        TreeNode* root = new TreeNode(preorder[0]);
        if(preorder.size()==1) return root;    //只有一个节点

        vector<int> leftIn,leftPre,rightIn,rightPre;
        int location = 0;
        while(inorder[location]!=root->val){
            leftIn.push_back(inorder[location]);
            location++;
        }
        for(int i=1;i<=location;i++) leftPre.push_back(preorder[i]);
        for(int i=location+1;i<preorder.size();i++){
            rightPre.push_back(preorder[i]);
            rightIn.push_back(inorder[i]);
        }
        root->left = buildTree(leftPre, leftIn);
        root->right = buildTree(rightPre, rightIn);
        return root;
    }
}
```

## 2. 二叉树的层次遍历

思路
二叉树或一般树的水平层次遍历，可以使用BFS（广度搜素）算法，使用队列 Queue标记每一层的结点元素；
Queue：先进先出， 后进后出。可以保证每一层遍历时的结点顺序；
BFS：类似于电影中的病毒传染，先感染靠近自己的，再由易感染层感染更外层；
该题二叉树中，先把根结点压入队列，当队列不为空时，移除队首结点，并判断该结点的左右子树中有无非空结点，若存在，则再次入队对应的左右子树结点……同一层的每个结点循环以上操作，直至队列为空，循环结束。

```cpp 
#include <vector>
#include <queue>
#include <iostream>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> result;
        queue<TreeNode*> que;
        if (root == NULL) return result;
        que.push(root);

        while (!que.empty()){
            int size = que.size();
            vector<int> temp;
            for (int i = 0; i < size; i++)
            {
                TreeNode* node = que.front();
                que.pop();
                temp.push_back(node->val);
                if (node->left != NULL) que.push(node->left);
                if (node->right != NULL) que.push(node->right);
            }
            result.push_back(temp);
        }
        return result;
    }
};

int main(int argc, char* argv[])
{
    string a = "3,9,20,null,null,15,7";
    auto tree = stringToTreeNode(a);
    auto res = Solution().levelOrder(tree);
    return 0;
}
```

## 3. 二叉树的中序遍历

### 迭代法

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode> s;
        vector<int> ans;
        TreeNode* t = root;
        while(t || !s.empty()){
            while(t){  //遍历到最左边的叶结点
                s.push(*t);
                t = t->left;
            }
            if(!s.empty()){
                ans.push_back(s.top().val);
                t = s.top().right;
                s.pop();
            }
        }
        return ans;
    }
};
```
#### 复杂度
时间复杂度：O(n)
空间复杂度：O(n)

### 递归法

```cpp
class Solution {
    vector<int> ans;
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(root) {
            inorderTraversal(root->left);
            ans.push_back(root->val);
            inorderTraversal(root->right);
        }
        return ans;
    }
};

```
#### 复杂度分析
- 时间复杂度：$O(n)$, 递归函数 $T(n) = 2 \cdot T(n/2)+1$
- 空间复杂度：最坏情况下需要空间$O(n)$，平均情况为$O(\log n)$


## 4. 一棵BTree如下，我们从右边看会看到{1,3,5},输出这个`vector<int>`
```cpp
        1
       / \
      2   3
    /   \
   4     5
```
这个问题可以通过广度有限搜索的方式实现，关键是要找到每一层最右边的那个节点。

```cpp
#include <vector>
#include <deque>

typedef struct btreenode {
    btreenode* left;
    btreenode* right;
    int value;
} btreenode;

std::vector<int> get_right_slice_btree(btreenode* root) {
    if(root == nullptr) return {};
    std::vector<int> result;
    std::deque<btreenode*> dp;
    dp.push_back(root);
    int prelevel_child_cnt = 1;

    int curlevel_child_cnt = 0;
    while(!dp.empty()) {
        btreenode* node = dp.pop_front();
        if(node->left) {
            dp.push_back(node->left);
            curlevel_child_cnt++;
        }
        if(node->right) {
            dp.push_back(node->right);
            curlevel_child_cnt++;
        }
        prelevel_child_cnt --;
        if(prelevel_child_cnt == 0) {
            prelevel_child_cnt = curlevel_child_cnt;
            curlevel_child_cnt = 0;
            result.push_back(node->value);
        }
    }
    return result;
}
```

btree如果一共有$n$个节点，该算法的时间复杂度是$O(n)$, 因为我们是遍历了一遍所有的节点。
空间复杂度：
$$x(1 + \frac{1}{2} + \frac{1}{4} + ..) = x(\frac{1}{1- \frac{1}{2}}) = n  $$
$$x = \frac{n}{2}$$
所以最长的队列长度是$\frac{n}{2}$,那么空间复杂度为$O(n)$

首先最简单的实现方法是使用两个deque来实现，这样每次deque保存当前的层，然后最后出队的就是尾部节点。但是这个是采用两个队列，空间上有一些浪费资源，需要二外的$k/2$的空间资源。


## 5. Find All Anagrams in a String 找出字符串中所有的变位词

Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.

Example 1:

Input:
s: "cbaebabacd" p: "abc"

Output:
[0, 6]

Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
 

Example 2:

Input:
s: "abab" p: "ab"

Output:
[0, 1, 2]

Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".

### 采用hash表法

用两个哈希表，分别记录p的字符个数，和s中前p字符串长度的字符个数，然后比较，如果两者相同，则将0加入结果res中，然后开始遍历s中剩余的字符，每次右边加入一个新的字符，然后去掉左边的一个旧的字符，每次再比较两个哈希表是否相同即可，参见代码如下：
```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        if (s.empty()) return {};
        vector<int> res, m1(256, 0), m2(256, 0);
        for (int i = 0; i < p.size(); ++i) {
            ++m1[s[i]]; ++m2[p[i]];
        }
        if (m1 == m2) res.push_back(0);
        for (int i = p.size(); i < s.size(); ++i) {
            ++m1[s[i]]; 
            --m1[s[i - p.size()]];
            if (m1 == m2) res.push_back(i - p.size() + 1);
        }
        return res;
    }
};
```

### 滑动窗口Sliding Window的方法

首先统计字符串p的字符个数，然后用两个变量left和right表示滑动窗口的左右边界，用变量cnt表示字符串p中需要匹配的字符个数，然后开始循环，
- Step1: 如果右边界的字符已经在哈希表中了，说明该字符在p中有出现，则cnt自减1，然后哈希表中该字符个数自减1，右边界自加1，
- Step2:如果此时cnt减为0了，说明p中的字符都匹配上了，那么将此时左边界加入结果res中。
- Step3:如果此时right和left的差为p的长度，说明此时应该去掉最左边的一个字符，我们看如果该字符在哈希表中的个数大于等于0，说明该字符是p中的字符，因为上面Step1我们有让每个字符自减1，如果不是p中的字符，那么在哈希表中个数应该为0，自减1后就为-1，所以这样就知道该字符是否属于p，如果我们去掉了属于p的一个字符，cnt自增1.
参见代码如下：

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        if (s.empty()) return {};
        vector<int> res, m(256, 0);
        int left = 0, right = 0;
        int cnt = p.size(), n = s.size();
        for (char c : p) ++m[c];
        while (right < n) {
            if (m[s[right++]]-- >= 1) --cnt;
            if (cnt == 0) res.push_back(left);
            if (right - left == p.size() && m[s[left++]]++ >= 0) ++cnt;
        }
        return res;
    }
};
```
https://www.cnblogs.com/grandyang/p/6014408.html

## 6. 最长公共子序列问题(LCS问题)

给定两个字符串A和B，长度分别为m和n，要求找出它们最长的公共子序列，并返回其长度。例如：
A = "HelloWorld"
B = "loop"

则A与B的最长公共子序列为 "loo",返回的长度为3。此处只给出动态规划的解法：定义子问题dp[i][j]为字符串A的第一个字符到第 i 个字符串和字符串B的第一个字符到第 j 个字符的最长公共子序列，如A为“app”,B为“apple”，dp[2][3]表示 “ap” 和 “app” 的最长公共字串。注意到代码中 dp 的大小为 (n + 1) x (m + 1) ，这多出来的一行和一列是第 0 行和第 0 列，初始化为 0，表示空字符串和另一字符串的子串的最长公共子序列，例如dp[0][3]表示  "" 和 “app” 的最长公共子串。

当我们要求dp[i][j]，我们要先判断A的第i个元素B的第j个元素是否相同即判断A[i - 1]和 B[j -1]是否相同，如果相同它就是dp[i-1][j-1]+ 1，相当于在两个字符串都去掉一个字符时的最长公共子序列再加 1；否则最长公共子序列取dp[i][j - 1] 和dp[i - 1][j]中大者。所以整个问题的

- 初始状态为：
$$dp[i][0]=0, dp[0][j]=0$$
- 相应的状态转移方程为：
$$
dp[i][j] = \begin{cases} \max\{dp[i - 1][j], dp[i][j - 1]\} ,& {A[i - 1]  != B[j - 1]} 
\\ dp[i - 1][j - 1] + 1 , & {A[i - 1]  == B[j - 1]} \end{cases}
$$
- 代码的实现如下：
```cpp
int findLCS(string A, int n, string B, int m)
{
    if(n == 0 || m == 0) return 0;
    std::vector<std::vector<int>> dp(n + 1,std::vector<int>(m + 1, 0)); //定义状态数组
    for(int i = 0 ; i <= n; i++) dp[i][0] = 0;
    for(int i = 0; i <= m; i++) dp[0][i] = 0;

    for(int i = 1; i <= n; i++)
        for(int j = 1; j<= m; j++)
        {
            if(A[i - 1] == B[j - 1])//判断A的第i个字符和B的第j个字符是否相同
                dp[i][j] = dp[i -1][j - 1] + 1;
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
        return dp[n][m];//最终的返回结果就是dp[n][m]
}
```
该算法的时间复杂度为$O(n*m)$，空间复杂度为$O(n*m)$。此外，由于遍历时是从下标1开始的，因为下标为0表示空字符串；所以第A的第i个字符实际上为A[i -1]，B的第j个字符为B[j-1]。

## 7. 最长公共子串问题

给定两个字符串A和B，长度分别为m和n，要求找出它们最长的公共子串，并返回其长度。例如：
A = "HelloWorld"
B = "loop"
则A与B的最长公共子串为 "lo",返回的长度为2。
我们可以看到子序列和子串的区别：子序列和子串都是字符集合的子集，但是子序列不一定连续，但是子串一定是连续的。

这里只给出动态规划的解法：定义dp[i][j]表示以A中第i个字符结尾的子串和B中第j个字符结尾的子串的的最大公共子串(公共子串实际上指的是这两个子串的所有部分)的长度(要注意这里和LCS的不同，LCS中的dp[i+1][j+1]一定是大于等于dp[i][j]的；但最长公共子串问题就不一定了，它的dp[i][j]表示的子串不一定是以A[0]开头B[0]开头的，但是一定是以A[i-1]、B[j-1]结尾的)，同样地， dp 的大小也为 (n + 1) x (m + 1) ，这多出来的一行和一列是第 0 行和第 0 列，初始化为 0，表示空字符串和另一字符串的子串的最长公共子串。

当我们要求dp[i][j]，我们要先判断A的第i个元素B的第j个元素是否相同即判断A[i - 1]和 B[j -1]是否相同，如果相同它就是dp[i - 1][j- 1] + 1，相当于在两个字符串都去掉一个字符时的最长公共子串再加 1；否则最长公共子串取0。

- 整个问题的初始状态为：
dp[i][0]=0, dp[0][j]=0
- 相应的状态转移方程为：
$$dp[i][j] = \begin{cases} 0 ,& {A[i - 1]  != B[j - 1]} \\ dp[i - 1][j - 1] + 1 , & {A[i - 1]  == B[j - 1]} \end{cases}$$

- 代码的实现如下：

```cpp
class LongestSubstring {
public:
    int findLongest(std::string A, int n, std::string B, int m) {
         if(n == 0 || m == 0)
            return 0;
        int rs = 0;
        std::vector<std::vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        //初始状态
        for(int i = 0 ; i <= n; i++) dp[i][0] = 0;
        for(int i = 0; i <= m; i++) dp[0][i] = 0;

        for(int i = 1; i <= n; i++)
            for(int j = 1; j<= m; j++)
            {
                if(A[i - 1] == B[j - 1]) {
                    dp[i][j] = dp[i -1][j - 1] + 1;
                    rs = max(rs, dp[i][j]);//每次更新记录最大值
                } else {//不相等的情况
                    dp[i][j] = 0;
                }
            }
            return rs;//返回的结果为rs
    }
};
```
该算法的时间复杂度为O(n*m)，空间复杂度为O(n*m)。同样地，遍历下标也是从1开始的。不过关于最长公共子串问题，有几点需要注意下：

1.由于dp[i][j]不像LCS是个递增的数组，所以它在每次更新时需要同时更新最大值rs，且最后返回的结果是rs。而LCS中返回的直接就是dp[n][m]。
2.从代码上来看，两者的结构其实差不多，只不过状态转移方程有些小许的不同，分析过程也类似。

## 8. [leetcode 673] Number of Longest Increasing Subsequence 最长递增序列的个数
Given an unsorted array of integers, find the number of longest increasing subsequence.

Example 1:

    Input: [1,3,5,4,7]
    Output: 2
    Explanation: The two longest increasing subsequence are [1, 3, 4, 7] and [1, 3, 5, 7].

Example 2:

    Input: [2,2,2,2,2]
    Output: 5
    Explanation: The length of longest continuous increasing subsequence is 1, and there are 5 subsequences' length is 1, so output 5.


https://www.cnblogs.com/grandyang/p/7603903.html

## 9. 给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

示例 1:
- 输入: [1,2,3,4,5,6,7] 和 k = 3 
- 输出: [5,6,7,1,2,3,4] 
- 解释: 
  - 向右旋转 1 步: [7,1,2,3,4,5,6] 
  - 向右旋转 2 步: [6,7,1,2,3,4,5] 
  - 向右旋转 3 步: [5,6,7,1,2,3,4]

示例 2:
- 输入: [-1,-100,3,99] 和 k = 2 
- 输出: [3,99,-1,-100] 
- 解释: 
  - 向右旋转 1 步: [99,-1,-100,3] 
  - 向右旋转 2 步: [3,99,-1,-100]

说明:
尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
要求使用空间复杂度为 O(1) 的原地算法。
```cpp

class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k %= nums.size();
        vector<int> temp(k);
        for(int i = 0; i < k; i++) temp[i] = nums[nums.size() - 1 - i];
        for(int i = nums.size() - 1; i >= 0; i--) {
            nums[i] = nums[i - k];
        }
        for(int i = 0; i < k; i++) {
            nums[i] = temp[k - i - 1];
        }
        return;
    }
};

```

类似翻转字符的方法，先把前n-k个数字翻转一下，再把后k个数字翻转一下，最后再把整个数组翻转一下
```cpp
class Solution {
    public void rotate(vector<int> &nums, int k) {
        int count = k % nums.size();
        int bounder= nums.size() - count;
        reverse(nums,0,bounder-1);
        reverse(nums,bounder, nums.size() - 1);
        reverse(nums,0,nums.size() - 1);
    }
    private void reverse(vector<int> &arr,int st,int end){
        while(st < end){
            int temp = arr[st];
            arr[st] = arr[end];
            arr[end] = temp;
            st++;
            end--;
        }
    }
};
```

## 10. [LeetCode] Sort Characters By Frequency 根据字符出现频率排序

Given a string, sort it in decreasing order based on the frequency of characters.

Example 1:

    Input: "tree"
    Output: "eert"

Explanation:
'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

Example 2:

    Input: "cccaaa"
    Output:"cccaaa"

Explanation:
Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
Note that "cacaca" is incorrect, as the same characters must be together.
 

Example 3:

    Input: "Aabb"
    Output: "bbAa"

Explanation:
"bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.
 
这道题让我们给一个字符串按照字符出现的频率来排序，那么毫无疑问肯定要先统计出每个字符出现的个数，那么之后怎么做呢？我们可以利用优先队列的自动排序的特点，把个数和字符组成pair放到优先队列里排好序后，再取出来组成结果res即可，参见代码如下：

### 解法一：

```cpp
class Solution {
public:
    string frequencySort(string s) {
        string res = "";
        priority_queue<pair<int, char>> q;
        unordered_map<char, int> m;
        for (char c : s) ++m[c];
        for (auto a : m) q.push({a.second, a.first});
        while (!q.empty()) {
            auto t = q.top(); q.pop();
            // 向str中添加t.sencond个t.first的char
            res.append(t.first, t.second);
        }
        return res;
    }
};
```

我们也可以使用STL自带的sort来做，关键就在于重写comparator，由于需要使用外部变量，记得中括号中放入＆，然后我们将频率大的返回，注意一定还要处理频率相等的情况，要不然两个频率相等的字符可能穿插着出现在结果res中，这样是不对的。参见代码如下：

### 解法二：

```cpp
class Solution {
public:
    string frequencySort(string s) {
        unordered_map<char, int> m;
        for (char c : s) ++m[c];
        sort(s.begin(), s.end(), [&](char& a, char& b){
            return m[a] > m[b] || (m[a] == m[b] && a < b);
        });
        return s;
    }
};
```

我们也可以不使用优先队列，而是建立一个字符串数组，因为某个字符的出现次数不可能超过s的长度，所以我们将每个字符根据其出现次数放入数组中的对应位置，那么最后我们只要从后往前遍历数组所有位置，将不为空的位置的字符串加入结果res中即可，参见代码如下：

### 解法三：

```cpp
class Solution {
public:
    string frequencySort(string s) {
        string res;
        vector<string> v(s.size() + 1);
        unordered_map<char, int> m;
        for (char c : s) ++m[c];
        for (auto &a : m) {
            v[a.second].append(a.second, a.first);
        }
        for (int i = s.size(); i > 0; --i) {
            if (!v[i].empty()) res.append(v[i]);
        }
        return res;
    }
};
```
## 11. Search a 2D Matrix（搜索二维矩阵）
编写一个高效的算法来判断$m \times n$矩阵中，是否存在一个目标值。该矩阵具有如下特性：
- 每行中的整数从左到右按升序排列。
- 每行的第一个整数大于前一行的最后一个整数。
- 示例输入:

matrix = [
[1,   3,  5,  7],
[10, 11, 16, 20],
[23, 30, 34, 50]]
target = 3

输出: true

### 算法思路1
1.找规律，首先此二维数组是有序的，我们可以从右上角开始查找，每次只需要左移或下移即可，也就是row++或col--；
2.初始化右上角数字下标的指针常量，如果target等于当前数则return true，如果大于右上角的数字，那么target肯定不在当前行，row++，省去了一行的比较，如果target小于右上角的数字，则target肯定不在当前列，那么col++即可。
3.完结。

### 算法思路2
根据二维数组数值特点，将其想象成为我们熟悉的一维数组求解。而这里二维转成一维的关键是一维数组的下标mid和二维数组下标[i][j]的换算关系：[i][j]=[mid/列数][mid%列数]。直接上代码，比较简介应该很容易看懂，就不再赘述了。

```cpp
class Solution {
  public:
  bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    if (m == 0) return false;
    int n = matrix[0].size();

    // 二分查找
    int left = 0, right = m * n - 1;
    int pivotIdx, pivotElement;
    while (left <= right) {
      pivotIdx = (left + right) / 2;
      pivotElement = matrix[pivotIdx / n][pivotIdx % n];
      if (target == pivotElement) return true;
      else {
        if (target < pivotElement) right = pivotIdx - 1;
        else left = pivotIdx + 1;
      }
    }
    return false;
  }
};
```

## 写一个申请二维数组的实现

```c
char** mymalloc(const int m, const int n) {
    if(m <=0 || n <=0) return nullptr;
    char** ret = (char**) malloc(sizeof(char*)* m);
    for(int i = 0; i < m; i++) {
        ret[i] = (char*)malloc(sizeof(char)* n);
    }
    return ret;
}

void myfree(char** a, const int m, const int n) {
    // param n is useless
    for(int i = 0; i < m; i++) {
        free(a[i]);
    }
    free(a);
}
```
