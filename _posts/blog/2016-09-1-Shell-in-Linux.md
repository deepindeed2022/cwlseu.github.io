---
layout: post
title: 开发：Shell 学习之路
categories: [blog ]
tags: [linux开发]
description: linux下的shell脚本的学习与使用
--- 
{:toc}



##  简单介绍

你能不能用shell判断一个文件中的数字是否有序啊？想想这不挺简单的吗，就开始动手写了，然后就有了这个版本。

```bash
#!bin/sh

filename=$1
before=-1;
flag=1;
for line in `cat data.txt`;
do
    after=${line};
    echo $line
    if [ "$before">"$after" ];then
        echo "FAILED";
        flag=0;
        break;
    else
        before=${line};
    fi
done

if [ $flag = 1 ];then
    echo "SUCCESS"
fi
```

哎呀，shell里不是有自带的`sort`命令吗，怎么不懂得试试那个`sort`呢。于是我就查阅了相关博客
* [Linux下Sort命令的一些使用技巧](http://www.hustyx.com/ubuntu/72/)
* [LINUX SHELL脚本攻略笔记[速查]](http://www.wklken.me/posts/2013/07/04/note-of-linux-shell-scripting-cookbook.html)

## #返回值

原来shell脚本的返回值不是直接返回啊，而是通过
linux中shell变量`$#`,`$@`,`$0`,`$1`,`$2`的含义解释: 
    变量说明: 
    `$$` 
    Shell本身的PID（ProcessID） 
    `$!` 
    Shell最后运行的后台Process的PID 
    `$?` 
    最后运行的命令的结束代码（返回值） 
    `$-` 
    使用Set命令设定的Flag一览 
    `$*` 
    所有参数列表。如"$*"用「"」括起来的情况、以`$1 $2 … $n`的形式输出所有参数。 
    `$@` 
    所有参数列表。如"`$@`"用「"」括起来的情况、以"`$1`" "`$2`" … "`$n`" 的形式输出所有参数。 
    `$#` 
    添加到Shell的参数个数 
    `$0` 
    Shell本身的文件名 
    `$1～$n` 
    添加到Shell的各参数值。`$1`是第1参数、`$2`是第2参数…。 

因此判断一个文件是否是有序的sort返回结果需要通过`$?`的值进行判断。可是sort是按照行来判断是否有序的，而不是判断所有的是否有序的。比如说datafile 中内容如下：

```
1 2 3 4 5 6 8 9
2 3 4 5 6 7 5 32
3 4 5 6 6 34 3 
```
按照默认sort 情况下，上述文件是有序的。但是实际上总体来说，我们需要返回该文件为无序的，因此，sort的方案只好作罢。

## #数字比较
后来看看之前那个实现的逻辑，应该是没有什么大问题的呀。

#### 数字的比较

-eq 相等（equal）
-ne 不等（not equal）
-gt 大于（greater than）
-lt 小于（less than）
-ge 大于等于 （greater than or equal）
-le 小于等于 （less than or equal）

#### 字符串的比较

`[ $str1 = $str2 ]` #等于
`[ $str1 != $str2 ]` #不等于
`[ -z $str ]`   #空字符串返回true
`[ -n $str ]`或者`[ $str ]` #非空字符串返回true


OMG， 原来shell里的`>`不是大于号啊，而是表示输入输出，下面就查找了一下关于linux标准文件描述符：
| 文件描述符| 缩写| 描述|
|----:|------:|------:|
|0    | STDIN |标准输入 |
|1    |STDOUT | 标准输出|
|2    |STDERR | 标准错误|  
标准输入和标准输出指的就是键盘和显示器。
当文件描述符（0,1,2）与重定向符号`<`组合之后，就可以重新定向输入，输出，及错误。
* `command    2>file1`
   命令执行的错误信息保存到了file1文件中。显示屏只是显示正确的信息。
* `command    1>file1  2>file2` 
   命令执行后，没有显示。因为正确输出到file1，错误定向到file2
* `command    &>file1`
命令执行后，输出和错误都定向到file1中
在shell脚本中，可以定义“错误”输出到STDERR指定的文件.需要在重定向符和文件描述符之间加一个and符`&` 

经过这番折腾，终于在shell下将这个简单的问题搞定了！！！

```bash
#!bin/sh
function is_sorted()
{
    before=-1;
    flag=1;
    for line in $(<$1); do
        if [ $before -gt $line ];then
            echo "Failed at $before, $line"
            flag=0
            break
        fi
        before=$line;
    done

    if [ $flag = 1 ];then
        echo "SUCCESS"
    fi
}

is_sorted $1
``` 


调用函数的方法为 `is_sorted datafilename` 
或者调用bash脚本 `bashfilename.sh datafilename`

## LeetCode Shell Test
### #[Word Frequency](https://leetcode.com/problems/word-frequency/)
Write a bash script to calculate the frequency of each word in a text file words.txt.

For simplicity sake, you may assume:

words.txt contains only lowercase characters and space ' ' characters.
Each word must consist of lowercase characters only.
Words are separated by one or more whitespace characters.

`cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{ print $2, $1 }'`

* `tr -s`: truncate the string with target string, but only remaining one instance (e.g. multiple whitespaces)
* `sort`: To make the same string successive so that uniq could count the same string fully and correctly.
* `uniq -c`: uniq is used to filter out the repeated lines which are successive, -c means counting
* `sort -r`: -r means sorting in descending order
* `awk '{ print $2, $1 }'`: To format the output, see here.

[Linux 中使用awk](https://linux.cn/article-3945-1.html)
[awk_1line](http://www.pement.org/awk/awk1line.txt)
[shell基础知识](http://github.com/cwlseu/cwlseu.github.io/raw/master/pdf/#Shell programming.pdf)