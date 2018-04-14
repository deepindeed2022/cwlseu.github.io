---
layout: post
title: 开发：Windows开发中的问题
categories: [blog ]
tags: [C++开发]
description: windows下开发常见问题
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>


## LINK : fatal error LNK1123: 转换到 COFF 期间失败: 文件无效或损坏

### 问题说明：

当安装VS2012之后，原来的.NET 4.0会被替换为.NET 4.5。卸载VS2012时，不会恢复.NET 4.0。当VS2012安装后，VS2010的cvtres.exe就无法使用了。如果你的PATH环境变量中VS2010的工具路径第一个出现，而且链接器需要将.res文件转换为COFF 对象格式，就会导致LNK1123错误。
当VS生成PE文件头时，使用的cvtres.exe版本错误，不能与当前的.NET平台兼容。

### 解决方案：

[link-fatal-error](http://stackoverflow.com/questions/10888391/link-fatal-error-lnk1123-failure-during-conversion-to-coff-file-invalid-or-c)
因为是cvtres.exe版本错误导致的结果，所以凡是能使VS链接器找到正确的cvtres.exe版本的方法都可以解决该问题。或者使VS链接器不生成COFF的方法都可以。

### 方案一

当前系统中存在两个cvtres.exe文件，版本不同。让VS2010使用.NET 4.5的cvtres.exe程序。
>重命名或删除：（vs2010安装的位置）C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\cvtres.exe
>这样C:\Windows\Microsoft.NET\Framework\v4.0.30319 (.NET 4.5)中的cvtres.exe文件就可以被VS2010使用。

### 方案二

>项目\属性\配置属性\清单工具\输入和输出\嵌入清单：原来是“是”，改成“否”。
>说明：这种方法每个工程均需要修改配置。

### 方案三

>安装：VS2010 SP1. 该版本应该是能使用.NET 4.5的，并配有正确的cvtres.exe版本。
>注意：安装VS 2010 SP1 时会移除64-bit 编译器. 通过安装 VS 2010 SP1 compiler pack 能够重新获得。
 
 我刚开始使用了方案二，可是后来当关闭vs之后，再想打开这个项目的时候，会出现问题。所以最终选择方案三彻底解决问题。


## 安装Boost库
### 下载源代码

[Boost Source](http://sourceforge.net/projects/boost/files/boost/1.59.0/)

### Build Source Code

1. 将源码解压都某个目录下面，如E:\boost_1_59_0,解压过程还是比较缓慢的。
2. 查找bat文件boostrap.bat,运行bat脚本

```shell
	@ECHO OFF
	REM Copyright (C) 2009 Vladimir Prus
	REM
	REM Distributed under the Boost Software License, Version 1.0.
	REM (See accompanying file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)

	ECHO Building Boost.Build engine
	if exist ".\tools\build\src\engine\bin.ntx86\b2.exe" del tools\build\src\engine\bin.ntx86\b2.exe
	if exist ".\tools\build\src\engine\bin.ntx86\bjam.exe" del tools\build\src\engine\bin.ntx86\bjam.exe
	if exist ".\tools\build\src\engine\bin.ntx86_64\b2.exe" del tools\build\src\engine\bin.ntx86_64\b2.exe
	if exist ".\tools\build\src\engine\bin.ntx86_64\bjam.exe" del tools\build\src\engine\bin.ntx86_64\bjam.exe
	pushd tools\build\src\engine

	call .\build.bat %* > ..\..\..\..\bootstrap.log
	@ECHO OFF

	popd

	if exist ".\tools\build\src\engine\bin.ntx86\bjam.exe" (
	   copy .\tools\build\src\engine\bin.ntx86\b2.exe . > nul
	   copy .\tools\build\src\engine\bin.ntx86\bjam.exe . > nul
	   goto :bjam_built)

	if exist ".\tools\build\src\engine\bin.ntx86_64\bjam.exe" (
	   copy .\tools\build\src\engine\bin.ntx86_64\b2.exe . > nul
	   copy .\tools\build\src\engine\bin.ntx86_64\bjam.exe . > nul
	   goto :bjam_built)

	goto :bjam_failure

	:bjam_built

	REM Ideally, we should obtain the toolset that build.bat has
	REM guessed. However, it uses setlocal at the start and does not
	REM export BOOST_JAM_TOOLSET, and I don't know how to do that
	REM properly. Default to msvc for now.
	set toolset=msvc

	ECHO import option ; > project-config.jam
	ECHO. >> project-config.jam
	ECHO using %toolset% ; >> project-config.jam
	ECHO. >> project-config.jam
	ECHO option.set keep-going : false ; >> project-config.jam
	ECHO. >> project-config.jam

	ECHO.
	ECHO Bootstrapping is done. To build, run:
	ECHO.
	ECHO     .\b2
	ECHO.    
	ECHO To adjust configuration, edit 'project-config.jam'.
	ECHO Further information:
	ECHO.
	ECHO     - Command line help:
	ECHO     .\b2 --help
	ECHO.     
	ECHO     - Getting started guide: 
	ECHO     http://boost.org/more/getting_started/windows.html
	ECHO.     
	ECHO     - Boost.Build documentation:
	ECHO     http://www.boost.org/build/doc/html/index.html

	goto :end

	:bjam_failure

	ECHO.
	ECHO Failed to build Boost.Build engine.
	ECHO Please consult bootstrap.log for further diagnostics.
	ECHO.
	ECHO You can try to obtain a prebuilt binary from
	ECHO.
	ECHO    http://sf.net/project/showfiles.php?group_id=7586^&package_id=72941
	ECHO.
	ECHO Also, you can file an issue at http://svn.boost.org 
	ECHO Please attach bootstrap.log in that case.

	goto :end

	:end
```

## bat 的基本语法

```cpp
	@                      	//关闭单行回显   
	echo off               	//从下一行开始关闭回显   
	@echo off              	//从本行开始关闭回显。一般批处理第一行都是这个   
	echo on                	//从下一行开始打开回显   
	echo                   	//显示当前是 echo off 状态还是 echo on 状态   
	echo.                  	//输出一个”回车换行”，空白行 (同echo, echo; echo+ echo[ echo] echo/ echo")   
	echo %errorlevel% 	   	//每个命令运行结束，可以用这个命令行格式查看返回码,默认值为0，一般命令执行出错会设 errorlevel 为1  
```


## 在window上使用caffe深度学习框架，安装路程艰辛，不过也是很有乐趣的。

### NuGet

[install](http://docs.nuget.org/consume/installing-nuget)
第一次接触到NuGet工具，很是帅气，简单一句话，就像是python里的pip，ubuntu里的sudo apt-get 命令，NuGet有一个server管理着大量的package，我们通过一个简单的*Install-Package* 的命令就可以实现对响应的依赖库的安装，很是方便。

### 在VS2012上安装

vs2012由于对于C++11的支持还是不够全面的，在caffe中用了很多C++11的特性，导致错误。[MSVC对 C++11 Core Language Feature的支持性](https://msdn.microsoft.com/en-us/library/hh567368(v=vs.110).aspx)

'''
    D:\Document\Repo\VS2012\caffe\include\caffe/common.hpp(84): error : namespace "std" has no member "isnan"
    D:\Document\Repo\VS2012\caffe\include\caffe/common.hpp(85): error : namespace "std" has no member "isinf"
'''

### 在VS2013安装

VS2013对于C++11的支持性就好多了

"pyconfig.h"或者"patchlevel.h"文件找不到的问题：
将python的安装路径下的include路径添加到项目include项目中。
如：C:\Develop\Python27\include


错误提示：error C2220: 警告被视为错误 - 没有生成“object”文件
错误原因：原因是该文件的代码页为英文，而我们系统中的代码页为中文。

解决方法：
1，将源码转化为正确的编码方式
    用vs2013打开对应的文档，文件->打开->选择该cpp，然后保存。
    如果不起作用的话，修改其中一部分，或者 选择替换，选中正则表达式，将\n替换为\n。
   也可以用文本编辑器如Notepad，更改代码文件的编码方式，改为ANSI。

2，设置项目属性，取消警告视为错误
    VS2013菜单 - 项目 - 属性 - 通用配置 - C/C++ - 常规 - 将警告视为错误 修改为 否，重新编译即可。


http://zhidao.baidu.com/link?url=rikLum87Ilmdo15lb2DWIDkM0P0d6UbE6BcZq1oJfXnA7B5C2EhptRUkuJgLmw_YJSByUizGU-xQe5nniFYaY_