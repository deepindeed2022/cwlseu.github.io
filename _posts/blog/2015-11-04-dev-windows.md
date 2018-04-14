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