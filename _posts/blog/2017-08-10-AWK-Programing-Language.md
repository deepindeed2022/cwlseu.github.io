---
layout: post
title: "A AWK Programming Language"
categories: [blog ]
tags: [Linux, awk ]
description: 这是关于在Linux开发中常用小工具--awk的故事
---
声明：本博客欢迎转发，但请保留原作者信息!                            
作者: [曹文龙]                                                                 
博客： <https://cwlseu.github.io/>

# 序言

Computer users spend a lot of time doing simple, mechanical data manipulation - changing the format of data, checking its validity, finding items with some property, adding up numbers, printing reports, and the like. All of these jobs ought to be mechanized, but it's a real nuisance to have to write a specialpurpose
program in a standard language like C or Pascal each time such a task comes up.

记住，
# Chapter 1 — Chapter2
通过案例程序的方式介绍基本的语法，读完第一章就可以开始写awk程序了。然后第二章将系统详细的介绍整个awk语言

### 操作文件
employee record: emp.data, every line is a record
   
    Beth 4.00 0
    Dan 3.75 0
    Kathy 4.00 10
    Mark 5.00 20
    Mary 5.50 22
    Susie 4.25 18

### 获取每个人干活了的人应该付多少钱
   `awk '$3 > 0 { print $1, $2*$3 }' emp.data`

    Kathy 40
    Mark 100
    Mary 121
    Susie 76.5

**pattern { action }**

### 运行awk程序
awk 'program' input files
如果省略输入文件，那么就会在terminal中等待输入，知道输入文件停止符为止。

如果命令很长怎么办呢？
我们可以将命令输入到一个文件中，例如`progfile`文件。
`awk -f progfile optional list of input files`

### 特殊符号

NF： the number of Fields
NR： 正在打印的行号

{ print "total pay for", $1, "is", $2*$3 }


























# Chapter 3
