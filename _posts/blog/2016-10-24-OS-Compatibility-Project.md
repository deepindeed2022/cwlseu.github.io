---
layout: post
title: Wine项目
categories: [blog ]
tags: [开发]
description: 每周都要开会，开会的内容有时有趣，有时乏味
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>
 

## 问题概述

IE发展落后于浏览器技术的发展，但是由于历史原因，网银等软件被IE技术挟持，导致我们现在如果要继续发展，兼容性问题的价值很突出。
当前兼容采用双核的解决方案，仅仅在windows

## 历史

2001 IE6支持CSS1和DOM1标准，2002年市场占有率90%，安全隐患逐渐被关注。
IE7采用另外一个进程进行ActiveX的处理，使得安全性得到比较好的使用。
IE11，由于对一些过时插件的屏蔽和更改标记名称，使得很多应用不能在IE上运行。

## 标准差异

getElementbyId
IE支持嵌入VB类型脚本，该脚本由vbscript.dll负责执行
特有的属性和方法
ActiveX控件_ActiveXObject
1. 可以实现对office组件进行交互
2. 嵌入式ActiveX控件<object>
    * type
    * codebase
    * clsid
3. 嵌入式silverlight控件
    * 作为移动weboc跨平台的解决方案
    * 但是现在已经关闭

Pipelight:
特点：
针对ActiveX控件的解决方案
不依赖IE内核

基于Trident双引擎

## 布局显示兼容

应对不容浏览器内容内核对现实元素的默认CSS属性差异

### wine 运行IE

注册表根本就不存在，那么就是安装就失败了
注册表里有，但是调用API过程中出现了各种问题

国内由于很多视屏网站采用Flash进行支持，导致国内兼容性问题还是很重要的
但是国外对htmlvideo的支持很普遍
