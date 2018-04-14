---
layout: post
title: "完美的图算法"
categories: [blog ]
tags: [Algorithm, ]
description: 算法是编程的基础框架，就像是建房子的砖头，生产的原料，爸妈做饭的柴米油盐。没有良好的算法基础，哪里做得出好菜，生产出优质的产品，建造出结实的房子。
---
- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## 图的表示方法

## 图的邻接矩阵存储方法

二维数组中第i行第j列表示顶点i到顶点j是否有边。1表示有边，-1或者无穷表示无边，激励我们将自己到自己设为0.如果表示的为无向图，则矩阵为对称矩阵。
graph_adjacent_1.txt中先输入有V个顶点有E条边
然后接下来E行为边的两个顶点。

  5 5
  1 2
  1 3
  1 5
  2 4
  3 5

## 图的邻接表的存储方法
邻接表是图的一种链式存储结构。对图的每个顶点建立一个单链表（n个顶点建立n个单链表），第i个单链表中的结点包含顶点Vi的所有邻接顶点。又称链接表。适用于**稀疏图**的存储。

```cpp
int u[MAX_LEN];
int v[MAX_LEN];
int w[MAX_LEN];
int first[MAX_LEN]; // 存储第i个顶点的第一条边的编号总长度为V
int next[MAX_LEN];  // 存储编号为i的边的下一条的编号总长度为E
// 读入边的格式为u v w
// weight(u, v) = w
void read_graph(FILE *f, int E)
{
   memset(first, -1, sizeof(first));
   for (int i = 1; i <= E; i++)
   {
      fscanf(f, "%d %d %d", &u[i], &v[i], &w[i]);
      //关键
      next[i]     = first[u[i]];
      first[u[i]] = i;
   }
}

void access_graph(const int V)
{
   for (int i = 1; i <= V; i++)
   {
      int k = first[i];
      while (k != -1)
      {
         printf("w(%d,%d) = %d\n", u[k], v[k], w[k]);
         k = next[k];
      }
   }
}
```
其中Dijstra算法使用堆进行选择最小距离，基于连接表的存储方式的时间复杂度为$O((V+E)logV)$ ,如果[基于邻接矩阵的表示方法](https://github.com/cwlseu/Algorithm/blob/master/aha/ch6/dijkstra.cpp), 时间复杂度为$O(V^2)$. 当图比较稀疏的时候，E << V^2, 这个时候$O((V+E)logV)$比$O(V^2)$小得多。

## 遍历方法

**深度优先遍历**的主要思想就是首先以一个未被访问过的顶点作为起始出发点，沿着当前顶点的边走到未被访问过的顶点：当没有未被访问过的顶点的是偶，则回到上一个顶点，继续试探访问别的顶点，知道所有的顶点都被访问过。显然，深度优先遍历是沿着图的某一条分支遍历直到末端，然后回溯，再沿着另一条进行同样的遍历，直到所有的顶点都被访问过为止。

**广度优先遍历**更加适用于所有边的权值相同的情况。

## 最小生成树的构造
### 定义
最小生成树是一副连通加权无向图中一棵权值最小的生成树。在一给定的无向图 G = (V, E) 中，(u, v) 代表连接顶点 u 与顶点 v 的边（即 $(u,v)\in E$），而 $w(u, v)$ 代表此边的权重，若存在 T 为 E 的子集（即 $T\subseteq E$）且为无循环图，使得
$$ w(T)=\sum _{(u,v)\in T} w(u,v)$$
的 w(T) 最小，则此 T 为 G 的最小生成树。最小生成树其实是最小权重生成树的简称。
[最小生成树](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E7%94%9F%E6%88%90%E6%A0%91) 

### 普里姆算法（Prim算法）
从单一顶点开始，普里姆算法按照以下步骤逐步扩大树中所含顶点的数目，直到遍及连通图的所有顶点。

- 输入：一个加权连通图，其中顶点集合为V，边集合为E；
- 初始化：Vnew = {x}，其中x为集合V中的任一节点（起始点），Enew = {}；
    重复下列操作，直到Vnew = V：
        1. 在集合E中选取**权值最小的边（u, v）**，其中 *u为集合Vnew中的元素，而v则是V中没有加入Vnew的顶点*（如果存在有多条满足前述条件即具有相同权值的边，则可任意选取其中之一）；
        2. 将v加入集合Vnew中，将（u, v）加入集合Enew中；
- 输出：使用集合Vnew和Enew来描述所得到的最小生成树。
* 从任意一个顶点开发构造生成树，假设从1号顶点开始。首先将顶点1加入生成树中，用一个一维数组book来标记那些顶点已经加入了生成树
* 用数组dist记录生成树到各个顶点的距离。最初生成树中只有1号顶点。有直连边的时候，dist存储的就是一号顶点到该点的权值，没有直连边的时候为Infinity
* 从数组dist中选出离生成树最近的点（假设该点为j），加入到生成树中。再以j为中间点，更新生成树到每一个非树顶点的距离。即`dist[k] > e[j][k]` 更新 `dist[k] = e[j][k]`
* 重复第三步，直到所有的节点被加入为止。

#### 采用邻接矩阵表示的实现
时间复杂度为O(V^2)

```cpp
// 初始化距离矩阵
   for(int i = 1; i <= V; ++i) dist[i] =  graph[1][i];
   book[1] = true;
   int sum = 0;
   int count = 1;
   while(count < V)
   {
      // 查找距离当前树最小的点
      // 时间复杂度为O(V), 如果采用堆进行优化的话可以降到O(logV)
      int j = 0;
      int min = MAX_VAL;
      for(int i = 1; i <= V; ++i)
      {
         if(!book[i] && dist[i] < min)
         {
            min = dist[i]; j = i;
         }
      }
      book[j] = true;
      count++;
      sum += dist[j];
      // 更新各点到树的距离
      for(int k = 1; k <= V; ++k)
      {
         if(!book[k] && dist[k] > graph[j][k])
            dist[k] = graph[j][k];
      }
   }
```
<https://github.com/cwlseu/Algorithm/blob/master/aha/ch8/prim_arr_minimal_spanning_tree.cpp>

#### 采用邻接表表示的方法O(ElogV)

其中推荐最小距离点的时候采用堆实现推荐，其中获取顶点元素时间为O(1)，调整堆时间为O(logV)。
调整代码实现如下:

```cpp
// 从i节点向下调整堆
void minheap_shiftdown(int i)
{
   int t, flag = 0;
   while (i * 2 <= size && flag == 0)
   {
      if (dist[h[i]] > dist[h[i << 1]])
         t = i << 1;
      else
         t = i;
      if (i * 2 + 1 <= size)
      {
         if (dist[h[t]] > dist[h[i * 2 + 1]])
         {
            t = i * 2 + 1;
         }
      }
      if (t != i)
      {
         swap(t, i);
         i = t;
      }
      else
         flag = 1;
   }
}
```
Prim算法关键部分如下：

```cpp
//弹出堆顶元素
   minheap_pop();
   while (count < V)
   {
      // 查找距离当前树最小的点
      int j = minheap_pop();

      book[j] = true;
      count++;
      sum += dist[j];

      // 更新各点到树的距离
      k = first[j]; // 获取顶点i的所有相连边的头
      while(k != -1)
      {
         if(!book[v[k]] && dist[v[k]] > w[k])
         {
            dist[v[k]] = w[k];
            minheap_shiftup(pos[v[k]]);
         }
         k = next[k];
      }
   }
```
<https://github.com/cwlseu/Algorithm/blob/master/aha/ch8/prim_heap_minimal_spanning_tree.cpp>

#### 时间复杂度

| 最小边、权的数据结构 |时间复杂度（总计|
|:-------:|:----------:| 
| 邻接矩阵、搜索 | O(V2)|
| 二叉堆（后文伪代码中使用的数据结构）、邻接表 | O((V + E) log(V)) = O(E log(V))|
| 斐波那契堆、邻接表   |O(E + V log(V))|

通过邻接矩阵图表示的简易实现中，找到所有最小权边共需O（V2）的运行时间。使用简单的二叉堆与邻接表来表示的话，普里姆算法的运行时间则可缩减为O(E log V)，其中E为连通图的边数，V为顶点数。如果使用较为**复杂的斐波那契堆**，则可将运行时间进一步缩短为O(E + V log V)，这在**连通图足够密集**时（当E满足Ω（V log V）条件时），可较显著地提高运行速度。

### Kruskal算法
#### 算法描述
- 新建图G，G中拥有原图中相同的节点，但没有边
- 将原图中所有的边按权值从小到大排序
- 从权值最小的边开始，如果这条边连接的两个节点于图G中不在同一个连通分量中，则添加这条边到图G中
- 重复3，直至图G中所有的节点都在同一个连通分量中

#### 实现

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
using std::vector;

typedef struct tEdge
{
   int u;
   int v;
   int val;
   tEdge()
      : u(-1)
      , v(-1)
      , val(0)
   {
   }
   tEdge(int _u, int _v, int _val)
      : u(_u)
      , v(_v)
      , val(_val)
   {
   }
} tEdge;
static const int MAX_LEN = 101;

bool comp_edge(tEdge e1, tEdge e2) { return e1.val < e2.val; }

tEdge edges[MAX_LEN];
int find[MAX_LEN];

inline void init(int* f, const int n) 
{
   for(int i = 1; i <= n; ++i) f[i] = i;
}
int getfather(int v)
{
   // 采用递归方式实现
   // 每次在函数返回的时候，将
   if(find[v] == v)
      return v;
   find[v] = getfather(find[v]);
   return find[v];
}
bool merge(int v, int u)
{
   if(v > u) return merge(u, v);
   int t1, t2;
   t1 = getfather(v);
   t2 = getfather(u);
   if(t1 != t2)
   {
      find[t2] = t1;// 靠左原则，左边的变成右边的boss
      return true;
   }
   return false;
}
int main(int argc, char const *argv[])
{

   FILE *f;
   f = fopen("minimal_spanning_tree.txt", "r");
   if (f == NULL) perror ("Error opening file");

   int V, E;
   int u, v, val;
   int sum = 0; // 最小生成树的花费
   int count_v = 0;

   fscanf(f, "%d %d", &V, &E);
   for (int i = 1; i <= E; ++i)
   {
      fscanf(f, "%d %d %d", &u, &v, &val);
      edges[i] = tEdge(u, v, val);
   }
   std::vector<tEdge> edgev(edges+1, edges+E+1);
   std::sort(edgev.begin(), edgev.end(), comp_edge);
   init(find, V);
   // kruskal 算法
   // 从小到大枚举每一条边
   for(int i = 0; i < E; ++i)
   {
      // 判断两个顶点是否已经联通，是否在一个集合里
      if(merge(edgev[i].u, edgev[i].v))
      {
         count_v++;
         sum += edgev[i].val;
      }
      // 选择v - 1条边即可
      if(count_v == V - 1) break;
   }
   // for(int i = 1 ; i <= V; ++i) printf("%d ", find[i]);
   printf("\n");
   printf("总共要花费银票是： %d\n", sum);
   return 0;
}
```
#### 算法时间复杂度

O(Elog(E)) E为图中的边数

## 最短路径问题

### Floyd-warshall
Floyd-Warshall算法,中文亦称弗洛伊德算法,是解决**任意两点**间的最短路径的一种算法,可以正确处理有向图或负权（**但不可存在负权回路**)的最短路径问题,
同时也被用于计算有向图的传递闭包。Floyd-Warshall算法的时间复杂度为$O(N^3)$，空间复杂度为$O(N^{2})$。

#### Floyd-Warshall算法的原理是动态规划
设 $D_{i,j,k}$为从$i$到$j$的只以$(1..k)$集合中的节点为中间节点的最短路径的长度。
若最短路径经过点k,则 $D_{i,j,k}=D_{i,k,k-1}+D_{k,j,k-1}$；
若最短路径不经过点k,则 $D_{i,j,k}=D_{i,j,k-1}$。
因此，$D_{i,j,k}=min(D_{i,j,k-1},D_{i,k,k-1}+D_{k,j,k-1})$。
在实际算法中,为了节约空间,可以直接在原来空间上进行迭代,这样空间可降至二维。

#### 伪代码

```python
"""
letdistbea |V|x|V| arrayofminimumdistanceinitializedtoinfinity
其中dist[i][j]表示由点i到点j的代价,当其为 ∞ 表示两点之间没有任何连接
"""
# initthegraph
forvinvertex:
    dist[v][v] = 0
foredge(u, v) inedge:
    dist[u][v] = w(u, v)

# startthemainalgorithm
forkrange(1,|V|):
    forirange(1, |V|):
        forjrange(1, |V|):
            ifdist[i][j] > dist[i][k] + dist[k][j]: 
                dist[i][j] = dist[i][k] + dist[k][j]
            endif
        endfor
    endfor
endfor
```

> **为什么不能解决带有"负权回路"的图,因为带有负权回路的图没有最短路径。因为1->2->3->1->2->3->1->2->3,每次绕一次就减少1, 永远都找不到最短路径。**

### Dijkstra最短路径算法
![@Dijkstra最短路径算法示意图, ref:wikipedia](../../images/algorithm/Dijkstra_Animation.gif)

戴克斯特拉算法是由荷兰计算机科学家艾兹赫尔·戴克斯特拉提出。迪科斯彻算法使用了**广度优先**搜索解决赋权有向图的单源最短路径问题,算法最终得到一个最短路径树。该算法常用于路由算法或者作为其他图算法的一个子模块。
举例来说,如果图中的顶点表示城市,而边上的权重表示城市间开车行经的距离,该算法可以用来找到两个城市之间的最短路径。该算法的输入包含了一个有权重的有向图G,以及G中的一个来源顶点S。我们以V表示G中所有顶点的集合。每一个图中的边,都是两个顶点所形成的有序元素对。(u, v) 表示从顶点u到v有路径相连。我们以E表示G中所有边的集合,而边的权重则由权重函数w: E → [0, ∞] 定义。因此,w(u, v) 就是从顶点u到顶点v的**非负权重**（weight）。边的权重可以想像成两个顶点之间的距离。任两点间路径的权重,就是该路径上所有边的权重总和。已知有V中有顶点s及t,Dijkstra算法可以找到s到t的最低权重路径(例如,最短路径)。这个算法也可以在一个图中, 找到从**一个顶点s到任何其他顶点**的最短路径。


#### 伪代码

```python
def Dijkstra(Graph, source):
     dist[source] ← 0                 # Initialization
     create vertex set Q

     for each vertex v in Graph:           
         if v ≠ source
             dist[v] ← INFINITY       # Unknown distance from source to v
             prev[v] ← UNDEFINED      # Predecessor of v

         Q.add_with_priority(v, dist[v])


     while Q is not empty:            # The main loop
        u ← Q.extract_min()           # Remove and return best vertex
        for each neighbor v of u:     # only v that is still in Q
            alt ← dist[u] + length(u, v) 
            if alt < dist[v]
                 dist[v] ← alt
                 prev[v] ← u
                 Q.decrease_priority(v, alt)
     return dist[], prev[]
```

### Bellman Ford算法
对所有的E条**边**进行V-1次松弛操作。因为最短路径上最多有V-1条边. 第一次循环相当于经过一条边到达各个顶点的最短路径，经过k次循环相当于经过k条边到达各个顶点的最短路径。

```cpp
   for (int k = 1; k <= V - 1; k++)
   {
      for(int i = 1; i <= E; ++i)
      {
         if(dist[v[i]] > dist[u[i]] + w[i])
            dist[v[i]] = dist[u[i]] + w[i]; 
      }
   }
```

除此之外，Bellman ford算法还可以用来检查是否有**负权回路**. 如果在进行V-1次松弛操作之后，仍然存在

```cpp
  if(dist[v[i]] > dist[u[i]] + w[i])
            dist[v[i]] = dist[u[i]] + w[i]; 
```

的情况的话，也就是说V-1轮松弛之后，仍然可以松弛，那么必存在负权回路。

```cpp
   for (int k = 1; k <= V - 1; k++)
   {
      for(int i = 1; i <= E; ++i)
      {
         if(dist[v[i]] > dist[u[i]] + w[i])
            dist[v[i]] = dist[u[i]] + w[i]; 
      }
   }
   // 检查是否有负权回路
   bool flag = false;
   for(int i = 1; i <= E; ++i)
   {
      if(dist[v[i]] > dist[u[i]] + w[i])
        flag = true; 
   }
   if (flag) printf("这图有负权回路\n");
```

### 复杂度总结

|         | Floyd     | Dijkstra    | Bellman Ford | Bellman Ford Proiority |
|:-------:|:---------:|:-----------:|:----------:|:------------:|
| 空间复杂度 |  O(V^2) |  O(E) |    O(E)   | O(E) |
| 时间复杂度 |  O(V^3) | O((V+E)lgV)| O(VE)| 最坏O(VE)|
| 适应情景  |稠密图和顶点关系密切|稠密图和顶点关系密切|稀疏图和边关系密切|稀疏图和边关系密切|
| 负权    |不可以|不能解决|可以解决|可以解决负权|

### 有向图的拓扑排序

## 参考
1.[aha!算法](http://www.ahalei.com/)

2.[wikipedia- Dijstra算法](https://zh.wikipedia.org/wiki/%E6%88%B4%E5%85%8B%E6%96%AF%E7%89%B9%E6%8B%89%E7%AE%97%E6%B3%95)