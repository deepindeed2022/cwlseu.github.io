---
layout: post
title: 测试：Performance Test中时间测试
categories: [blog ]
tags: [Perfmance Test, ]
description: "在项目中或者研究中，经常需要对系统性能进行测试，今天将对我在项目中或者研究中用到的性能测试的有关东西进行总结"
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

在项目中或者研究中，经常需要对系统性能进行测试，今天将对我在项目中或者研究中用到的性能测试的有关东西进行总结

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