---
layout: post
title: "测试：Performance Test中时间测试"
categories: [blog ]
tags: [linux开发]
description: "在项目中或者研究中，经常需要对系统性能进行测试，今天将对我在项目中或者研究中用到的性能测试的有关东西进行总结"
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## 引言
在项目中或者研究中，经常需要对系统性能进行测试，今天将对我在研究生期间项目中用到的性能测试的有关东西进行简单总结。

## 程序运行时间

### 使用linux 命令`time`进行运行整个程序

`time main_exe`

### 使用在测试程序中打印系统时间的方法

1. 获取最佳性能

```cpp
#include <time.h>
#include <sys/time.h>
//
static long get_current_time()
{
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec * 1000000 + tv.tv_usec;
}
long perf_test()
{
    long start = get_current_time();
    // execute the program

    long end = get_current_time();
    return end - start;
}
void test()
{
    long totle = 0;
    perf_test();
    for(size_t i = 0; i < 10 ;++i)
        totle += per_test();
    printf("The execute time: %lfus", (double)totle/10.0);
}
```
值得注意的是，我们首先运行了一边被测函数，测试后面执行的效果。这样测试的性能比平均好一点。

2. 调用次数很频繁的功能时间测试可以使用上述方法，但是有时候我们的函数调用次数很少，我们可以在测试程序之间申请很大的空间，将cache都填写满，然后再释放掉的方式进行测试。

```cpp
void clean_cache()
{
    const int size = 128*1024*1024; //128M
    char* p;
    p = (char*)malloc(size);
    if(p != NULL)
        free(p);
}
```

## 系统性能测试

1. cpu指标：cpu利用率
    
cpu瓶颈原因
    - 动态web生成频繁
    - process再不同kernel之间迁移
    - 频繁中断与上下文切换
uptime可以查看cpu负载量
upstat展示多核运行中，user，sys，iowait，irq，soft，idle等信息
free显示linux系统空闲已用和swap的内存
pmap分析ssh进城分析
top展示cpu与内存信息，看到各个进程的CPU使用率和内存使用率

2. 内存指标
    - 空闲内存
    - swap利用率
    - 缓冲和缓存
    
2.1 内存瓶颈原因
* 吃内存的程序mongodb
* 缓存太多
* 内存泄露
    - 现在很多编译器都自带内存泄漏检查功能，有的甚至帮你释放内存，如icc -O3 编译选项
*　numa架构导致内存利用率低

3. io设备
    io等待
    平均队列长度
    平均服务:等待＋服务
磁盘瓶颈
糟糕的编程方式，如直接写磁盘，而不是写缓存，频繁执行将导致瓶颈
* iostat查看磁盘信息

4. 网络指标
    接受发送包
    接收发送字节

`vmstat`命令是最常见的Linux/Unix监控工具，可以展现给定时间间隔的服务器的状态值,包括服务器的CPU使用率，内存使用，虚拟内存交换情况,IO读写情况。这个命令是我查看Linux/Unix最喜爱的命令，一个是Linux/Unix都支持，二是相比top，我可以看到整个机器的CPU,内存,IO的使用情况，而不是单单看到各个进程的CPU使用率和内存使用率(使用场景不一样)。

## 综合系统性能测评工具nmon

使用命令nmon -f -s 2 -c 1800启动nmon
在当前目录中生成：计算机名_日期_时间.nmon文件

使用命令C://Program\ Files/Photoshop7.0/Photoshop.exe运行Photoshop
Photoshop软件正常启动，出现Photoshop软件窗口

选择“文件”菜单，选择“打开”
弹出下拉菜单，有“打开”；点击打开出现文件选择窗口
选择一个几MB~十几MB的图片

使用命令ps | grep nmon查看nmon进程并杀死该进程
nmon停止运行

## 其他工具

[vmstat](http://www.cnblogs.com/ggjucheng/archive/2012/01/05/2312625.html)
sar工具

另外一波福利：
1. [Linux工具快速教程](http://linuxtools-rst.readthedocs.io/zh_CN/latest/)
2. [工具进阶](http://linuxtools-rst.readthedocs.io/zh_CN/latest/)
3. [工具参考篇章](http://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/index.html)

## oprofile
opreport可以定义到具体的函数

## 冒烟测试 & 回归测试

**冒烟测试**是自由测试的一种。**冒烟测试**(smoketest)在测试中发现问题，找到了一个Bug，然后开发人员会来修复这个Bug。这时想知道这次修复是否真的解决了程序的Bug，或者是否会对其它模块造成影响，就需要针对此问题进行专门测试，这个过程就被称为SmokeTest。在很多情况下，做SmokeTest是开发人员在试图解决一个问题的时候，造成了其它功能模块一系列的连锁反应，原因可能是只集中考虑了一开始的那个问题，而忽略其它的问题，这就可能引起了新的Bug。SmokeTest优点是节省测试时间，防止build失败。缺点是覆盖率还是比较低。

**回归测试**是指修改了旧代码后，重新进行测试以确认修改没有引入新的错误或导致其他代码产生错误。自动回归测试将大幅降低系统测试、维护升级等阶段的成本。回归测试作为软件生命周期的一个组成部分，在整个软件测试过程中占有很大的工作量比重，软件开发的各个阶段都会进行多次回归测试。在渐进和快速迭代开发中，新版本的连续发布使回归测试进行的更加频繁，而在极端编程方法中，更是要求每天都进行若干次回归测试。因此，通过选择正确的回归测试策略来改进回归测试的效率和有效性是非常有意义的。

## 总结
实际项目中借助自研和第三方项目，实现快速迭代开发。相应的单元测试和集成测试，性能测试工具都比较完善，有的公司甚至能够实现自动话测试。