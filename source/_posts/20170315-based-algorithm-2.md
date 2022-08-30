---
layout: post
title: "算法的乐趣：再看深度优先搜索"
categories: [blog ]
tags: [Algorithm, ]
date: 2017-03-15 02:12:12
description: 算法是编程的基础框架，就像是建房子的砖头，生产的原料，爸妈做饭的柴米油盐。没有良好的算法基础，哪里做得出好菜，生产出优质的产品，建造出结实的房子。
---



## 适用场景

**输入数据**：如果是递归数据结构，如单链表，二叉树，集合，则百分之百可以用深搜；如果是
非递归数据结构，如一维数组，二维数组，字符串，图，则概率小一些。
**状态转换图**：树或者图。
**求解目标**：必须要走到最深（例如对于树，必须要走到叶子节点）才能得到一个解，这种情况
适合用深搜。

## 思考步骤

1. 是求路径条数，还是路径本身（或动作序列）？深搜最常见的三个问题，求可行解的总数，求一个可行解，求所有可行解。
- 如果是求路径本身，则要用一个数组 `path[]`  存储路径。跟宽搜不同，宽搜虽然最终求的也是一条路径，但是需要存储扩展过程中的所有路径，在没找到答案之前所有路径都不能放弃；而深搜，在搜索过程中始终只有一条路径，因此用一个数组就足够了。
- 如果是路径条数，则不需要存储路径。

2. 只要求一个解，还是要求所有解？
如果只要求一个解，那找到一个就可以返回；如果要求所有解，找到了一个后，还要继续扩展，直到遍历完。广搜一般只要求一个解，因而不需要考虑这个问题（广搜当然也可以求所有解，这时需要扩展到所有叶子节点，相当于在内存中存储整个状态转换图，非常占内存，因此广搜不适合解这类问题）。

3. 如何表示状态？
即一个状态需要存储哪些些必要的数据，才能够完整提供如何扩展到下一步状态的所有信息。跟广搜不同，深搜的惯用写法，不是把数据记录在状态 struct 里，而是添加函数参数（有时为了节省递归堆栈，用全局变量）， struct 里的字段与函数参数一一对应。

4. 如何扩展状态？
这一步跟上一步相关。状态里记录的数据不同，扩展方法就不同。对于固定不变的数据结构（一般题目直接给出，作为输入数据），如二叉树，图等，扩展方法很简单，直接往下一层走，对于隐式图，要先在第 1 步里想清楚状态所带的数据，想清楚了这点，那如何扩展就很简单了。

5. 关于判重
- 如果状态转换图是一棵树，则不需要判重，因为在遍历过程中不可能重复。
- 如果状态转换图是一个图，则需要判重，方法跟广搜相同，见第 §9.4 节。这里跟第 8 步
中的加缓存是相同的，如果有重叠子问题，则需要判重，此时加缓存自然也是有效果的。

6. 终止条件是什么？
终止条件是指到了不能扩展的末端节点。对于树，是叶子节点，对于图或隐式图，是出度为 0 的节点。

7. 收敛条件是什么？
收敛条件是指找到了一个合法解的时刻。
- 如果是正向深搜（父状态处理完了才进行递归，即父状态不依赖子状态，递归语句一定是在最后，尾递归），则是指是否达到目标状态；
- 如果是逆向深搜（处理父状态时需要先知道子状态的结果，此时递归语句不在最后），则是指是否到达初始状态。
- 由于很多时候终止条件和收敛条件是是合二为一的，因此很多人不区分这两种条件。仔细区分这两种条件，还是很有必要的。
为了判断是否到了收敛条件，要在函数接口里用一个参数记录当前的位置（或距离目标还有多远）。如果是求一个解，直接返回这个解；*如果是求所有解，要在这里收集解，即把第一步中表示路径的数组 path[] 复制到解集合里*。

8. 如何加速？
- 剪枝。深搜一定要好好考虑怎么剪枝，成本小收益大，加几行代码，就能大大加速。这里没有通用方法，只能具体问题具体分析，要充分观察，充分利用各种信息来剪枝，在中间节点提前返回。
- 缓存。如果子问题的解会被重复利用，可以考虑使用缓存。
    - 前提条件：子问题的解会被重复利用，即子问题之间的依赖关系是有向无环图(DAG)。如果依赖关系是树状的（例如树，单链表），没必要加缓存，因为子问题只会一层层往下，用一次就再也不会用到，加了缓存也没什么加速效果。
    - 具体实现：可以用数组或 HashMap。维度简单的，用数组；维度复杂的，用HashMap， C++ 有 map， C++ 11 以后有 unordered_map，比 map 快。

```cpp
/******************************************************************************
 * @description：
 *    走迷宫问题
 * 考虑要点：
 *  1.
 拓展状态+表示状态。当前步与下一步之间如何转换，本题目中采用一个book进行记录已经走过的步骤。
 *  当前步骤与下一步至今采用重置book值的方式实现切换
 *  2. 一次搜索终止条件是什么?
 *  3. 优化：
      是否可以进行剪枝操作？判断条件是什么？

 *
 ******************************************************************************/

#include "dfs_format_print.h"
#include <cassert>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
using namespace std;

//迷宫
static std::vector<std::vector<bool>> maze;
//迷宫的长宽
static int maze_h, maze_w;
//标记是否走过了
static std::vector<std::vector<bool>> book;
// 初始位置和目标位置
static int startx, starty;
static int p, q; //目标的位置
// 记录最小路径长度
static int min_step = INT_MAX;

static std::vector<std::pair<int, int>> min_path;

static int total = 0;
/**
 * 走迷宫
 * @param x, y 当前的位置坐标
 * @param step 当前走了多少步了
 * @return
 **/
void dfs(int x, int y, int step, std::vector<std::pair<int, int>> &path)
{
   if (x == p && q == y)
   {
      min_step = step < min_step ? step : min_step;
      min_path = path;
      return;
   }
   // 进行合理性剪枝, 当前步骤已经比最小步骤多了的话，这种情况我们可以不考虑了
   // 直接舍弃掉，可以通过输出total进行验证
   if (step >= min_step)
      return;
   int tx, ty;
   for (int i = 0; i < 4; ++i)
   {
      tx = x + next_[i][0];
      ty = y + next_[i][1];
      if (!inMaze(tx, ty, maze_w, maze_h))
         continue;

      if (maze[tx][ty] && !book[tx][ty])
      {
         path.push_back(std::make_pair(tx, ty));
         book[tx][ty] = true;
         dfs(tx, ty, step + 1, path);
         path.pop_back();
         book[tx][ty] = false;
         // total ++;
      }
   }
}



int main(int argc, char const *argv[])
{
   // 读取数据
   ifstream cin("dfs_find_maze.txt");
   // ofstream cout("result.txt");

   cin >> maze_h >> maze_w;
   maze.resize(maze_h, std::vector<bool>(maze_w, true));
   book.resize(maze_h, std::vector<bool>(maze_w, false));

   int c;
   for (int i = 0; i < maze_h; ++i)
      for (int j = 0; j < maze_w; ++j)
      {
         cin >> c;
         maze[i][j] = (c == 0);
      }
   cin >> startx >> starty >> p >> q;

   //检查输入
   assert(p < maze_h && p >= 0);
   assert(q < maze_w && q >= 0);

   cout << " startx:" << startx << " starty: " << starty << std::endl;
   cout << " endx:" << p << " endy: " << q << std::endl;
   cout << " maze_h:" << maze_h << " maze_w: " << maze_w << std::endl;
   std::vector<std::pair<int, int>> path;
   path.clear();

   path.push_back(std::make_pair(startx, starty));
   // 走迷宫
   dfs(startx, starty, 0, path);
   //结果输出
   cout <<" min_step:" << min_step << std::endl << " step list:\n  ";
   for(int  i = 0; i < min_path.size() -1; ++i)
        cout <<"(" <<min_path[i].first << " " << min_path[i].second<<") -> ";
   cout <<"(" <<min_path[min_path.size() -1].first << " " << min_path[min_path.size() -1].second<<")";
   cout << std::endl;
   //cout << total << std::endl;
   cout << " Format result:\n";
   format_path(min_path, maze_w, maze_h, cout);
   return 0;
}
```

[github示例代码](https://github.com/cwlseu/Algorithm/tree/master/aha/ch4)