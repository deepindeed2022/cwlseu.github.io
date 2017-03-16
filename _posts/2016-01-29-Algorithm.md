---
layout: post
title: DongBo Bu Algorithm
categories: [blog ]
tags: [Algorithm, ]
description: In me the tiger sniffs the rose.(心有猛虎，细嗅蔷薇)
---

# 课程介绍
卜老师上课热情，兴奋，通过实际案例抛出问题解决方案。通过介绍算法，讲算法应用的实际应用中的授课方式，让我们在今后的教学、整理论文框架等都有知道作用的。
*实际问题*
如 0-1 背包问题
*思考问题*
- 基于问题的结构观察的算法设计
- 从最简单的例子做起
- 试图把大的问题分解为小的问题
- 试图从**粗糙的解开始逐步改进**
- 试图枚举所有的解，但是要做到smart
*解决方案*
贪心
动态规划
...

# 主要内容
## 算法复杂度
复杂度包括时间复杂度和空间复杂度，往往针对不同的机器性能有不同的具体时间。因此在工程学中，我们常常使用相对时间复杂度。就像化学中的物质的量一般，我们采用一个统一的单位制，叫运行一步运算的时间为一个单位，关键在于考察算法要运行多少步。

### 递归树(主定理)

### 均摊分析
主要针对要消耗大量计算资源的操作出现次数很少，而频繁步骤操作消耗为常数的情况。如Hash表的连续插入和删除。
1.*归纳枚举法* 与递归树类似

2.*Account Bank* 

3.*势函数方式* 

### 平滑复杂度
平滑复杂度分析是针对更加特殊，时间消耗类似噪声毛刺一般的算法时间复杂度分析。平滑分析主要是针对LP过程中进行讨论的。一般认为ILP规划是线性的，因为在实际的应用中效率实在是太高了，我们一直认为是多项式时间的。但是后来[smoothedcomplexity](http://bioinfo.ict.ac.cn/~dbu/AlgorithmCourses/Lectures/Lec8-smoothedcomplexity.pdf)来解释了ILP是NP问题。
## 分治D&C
1.一个问题的解可以分成多个部分求解，每个部分之间是相互独立的
2.通过子问题的解通过*combine*就可以组成整个解
###例题：
1. 归并排序
归并排序时间复杂度为O(nlogn),空间复杂度为O(n)

```python
def merge_sort(A):
    def merge(A, B):
        n = len(A)
        m = len(B)
        i = 0
        j = 0
        k = 0
        C = [0 for _ in xrange(n+m)]
        while i < n and j < m:
            if A[i] <= B[j]:
                C[k] = A[i]
                k += 1
                i += 1
            else:
                C[k] = B[j]
                k += 1
                j += 1
        if i < n:
            C[k:] = A[i:]
        else:
            C[k:] = B[j:]
        return C

    n = len(A)
    if n == 1: return A
    if n == 2:
        if A[0] > A[1]:
            A[0], A[1] = A[1], A[0]
        return A
    else:
        B = merge_sort(A[:n/2])
        C = merge_sort(A[n/2:])
    return merge(B, C)
```
2. 逆序对计数最近点对

>   算法描述
>   如果采用暴力搜索的方法的话需要O(n^2)的时间复杂度，但是使用分治算法时间复杂度为O(nlogn)
    * Sort points according to their x-coordinates.
    * Split the set of points into two equal-sized subsets by a vertical line x=xmid.
    * Solve the problem recursively in the left and right subsets. This yields the left-side and right-side minimum distances dLmin and dRmin, respectively.
    * Find the minimal distance dLRmin among the set of pairs of points in which one point lies on the left of the dividing vertical and the other point lies to the right.
    * The final answer is the minimum among dLmin, dRmin, and dLRmin.

[More information in wikipedia](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem)

```python

def get_euclidean_distance2(p1, p2):
    return abs(p1[1] - p2[1])**2 + abs(p1[0] - p2[0])**2

def get_dmrange_points(s, l, dm, index = 0):
    '''
    argv:
        s : a list of points
        l : median point's x coordinate
        dm : the mininum distance currently
    return:
        the points list which distance to median is smaller than dm
    '''
    return [i for i in s if abs(i[index] - l) <= dm]
            
def Cpair2(points):
    # divide to 2 set
    if len(points) < 2:
        return 0x7FFFFFFFFFFFFFFFL
    elif len(points) == 2:
        return get_euclidean_distance2(points[0], points[1])
    elif len(points) == 3:

        return min(get_euclidean_distance2(points[0], points[1]),\
                    get_euclidean_distance2(points[0], points[2]),\
                    get_euclidean_distance2(points[1], points[2]))
    elif len(points) > 3:
        # get the median number of points_x_index
        m = points[len(points)/2][0]

        # divide the points set to 2 part by coordinate x 
        S1, S2 = filter(lambda x: x[0] <= m, points), filter(lambda x: x[0] > m, points)
    
        cd_1 = cd_2 = 0x7FFFFFFFFFFFFFFFL
        if len(S1) != len(points) and len(S2) != len(points):
            cd_1, cd_2 = Cpair2(S1), Cpair2(S2)
        dm = min(cd_1, cd_2)

        # calculate the 
        P1,P2 =  sorted(get_dmrange_points(S1, m, dm), key = lambda x: x[1]) , \
                sorted(get_dmrange_points(S2, m, dm), key = lambda x: x[1])

        ## There are some tip to promote performance by update dm immediately if find some smaller dm
        euclidist = []
        for point in P1:
            #Y = get_dmrange_points(P2, l = point[1], dm = dm, index = 1)
            for point2 in P2:
                euclidist.append(get_euclidean_distance2(point,point2))
        if euclidist:
            return min(dm, min(euclidist))
        else:
            return dm


def sort_by_index(pointslist, index = 0):
    '''
    index = 0 sorted by x
    index = 1 sorted by y
    '''
    return sorted(pointslist, key = lambda x: x[index])

def main():
    result = sort_by_index(pointslist = getdata_from_file(filename = opts.input),index = 1)
    mindistance = Cpair2(result)**0.5
    print "min distance in pairs is: %s" % mindistance
```
3. 矩阵乘法
4. 快速傅里叶变换
5. 生成全排列


>   Heap's algorithm
    Heap's algorithm generates all possible permutations of n objects. 
    It was first proposed by B. R. Heap in 1963.[1] 
    The algorithm minimizes movement: 
    it generates each permutation from the previous one by interchanging 
    a single pair of elements; the other n−2 elements are not disturbed. 
    In a 1977 review of permutation-generating algorithms, Robert Sedgewick 
    concluded that it was at that time the most effective algorithm for 
    generating permutations by computer.[2]


[reference](https://en.wikipedia.org/wiki/Heap%27s_algorithm)

```python
# -*-encoding:utf-8 -*-
import time
'''
'''
def generate(l, n):
    if n == 1:
        pass#print l
    else:
        for i in xrange(n - 1):
            generate(l, n - 1)
            if n % 2 == 0:
                # swap i and n-1
                l[i], l[n-1] = l[n-1], l[i]
            else:
                # swap 0 and n-1
                l[0], l[n-1] = l[n-1], l[0]
        generate(l, n-1)


def generate_norecursive(l, n):
    c = [0 for _ in xrange(n)]
    print l 
    i = 0
    # i 就是当前队列[0, i]的最后一个元素的坐标
    # c[i]表示当前
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                l[0], l[i] = l[i], l[0]
            else:
                l[i], l[c[i]] = l[c[i]], l[i]
            print l 
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
'''
假设generate(N-1)能产生所有排列，那么通过和之前N-1个元素交换最后的第N个元素（或者不交换）可以产生新的排列，所产生的个数正好是N×generate(N-1)，满足排列定义，所以是N！个。算法中的后一个swap是为了保证把最后一个元素换回原来位子，还原整个序列，注意所有递归操作施加在了同一个静态数组上。
'''
def generate_backtrack(l, n):
    if n == 1:
        pass #print l
    else:
        for i in xrange(0, n):
            l[i], l[n-1] = l[n-1], l[i]
            generate_backtrack(l, n-1)
            l[i], l[n-1] = l[n-1], l[i]

if __name__ == '__main__':
    looptime = 1
    n = 3
    s = range(1, n+1)
    start = time.time()
    for _ in xrange(looptime): generate(s, n)
    end = time.time()
    print "time:", end - start
    start = time.time()
    for _ in xrange(looptime): generate_backtrack(s, n)
    end = time.time()
    print "time:", end - start
    start = time.time()
    for _ in xrange(looptime): generate_norecursive(s, n)
    end = time.time()
    print "time:", end - start
```


## 动态规划
动态规划算法通常用于求解具有某种最优性质的问题。在这类问题中，可能会有许多可行解。每一个解都对应于一个值，我们希望找到具有最优值的解。动态规划算法分解得到子问题往往*不是互相独立*的。若用分治法来解这类问题，则分解得到的子问题数目太多，有些子问题被重复计算了很多次。如果我们能够保存已解决的子问题的答案，而在需要时再找出已求得的答案，这样就可以避免大量的重复计算，节省时间。我们可以用一个表来记录所有已解的子问题的答案。不管该子问题以后是否被用到，只要它被计算过，就将其结果填入表中。这就是动态规划法的基本思路。具体的动态规划算法多种多样，但它们具有相同的填表格式。
1.找出最优解的性质，并刻画其结构特征；
2.递归地定义最优值（写出动态规划方程）；
3.以*自底向上*的方式计算出最优值；
4.根据计算最优值时得到的信息，构造一个最优解。
## 贪心

## LP

## NL

## NP

## APP


