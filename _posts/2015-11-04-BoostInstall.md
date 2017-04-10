---
layout: post
title: Windows下安装boost库
categories: [blog ]
tags: [C++开发]
description: 安装boost库
---



声明：本博客欢迎转发，但请保留原作者信息! 
作者: [cwlseu]
博客： [https://cwlseu.github.io/](https://cwlseu.github.io/)


## 下载源代码
[Boost Source](http://sourceforge.net/projects/boost/files/boost/1.59.0/)

## Build Source Code
1. 将源码解压都某个目录下面，如E:\boost_1_59_0,解压过程还是比较缓慢的。
2. 查找bat文件boostrap.bat,运行bat脚本
 
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

### bat 的基本语法

	@                      	//关闭单行回显   
	echo off               	//从下一行开始关闭回显   
	@echo off              	//从本行开始关闭回显。一般批处理第一行都是这个   
	echo on                	//从下一行开始打开回显   
	echo                   	//显示当前是 echo off 状态还是 echo on 状态   
	echo.                  	//输出一个”回车换行”，空白行 (同echo, echo; echo+ echo[ echo] echo/ echo")   
	echo %errorlevel% 	   	//每个命令运行结束，可以用这个命令行格式查看返回码,默认值为0，一般命令执行出错会设 errorlevel 为1   