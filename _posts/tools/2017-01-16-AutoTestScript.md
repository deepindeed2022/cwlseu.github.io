---
layout: post
title: "测试：自动化测试工具AutoIt"
categories: [blog ]
tags: [linux开发]
description: Autoit是一个使用脚本语言的免费软件,它设计用于Windows GUI(图形用户界面)中进行自动化操作。它利用脚本模拟键盘按键，鼠标移动和窗口/控件的组合来实现自动化任务。
---
{:toc}

## 简介
Autoit是一个使用脚本语言的免费软件,它设计用于Windows GUI(图形用户界面)中进行自动化操作。它利用脚本模拟键盘按键，鼠标移动和窗口/控件的组合来实现自动化任务。

## 脚本入门
数据类型只有一种Variant,使用过程中决定是numeric还是string.该脚本语言属于弱语言类型。

```AutoIt
10*20       equals the number 200
10*"20"     equals the number 200
10 & 20     equals to string 1020 & is used to join strings
```

## Tip


1. 判断某个exe是否在运行

```au3
If ProcessExists($SetupFile) Then ProcessClose($SetupFile)
If Run($SetupFile) = 0 Then ShellExecute($SetupFile)
```

2. 鼠标点击某个控件操作 | 激活某个控件

```au3
ControlClick($Wintitle_1, $Wintxt_1, "Button2");点击按钮控件: 安装
ControlEnable($hWnd, "",'ComboBox1');使能开关
```

3. Windows窗口操作

```au3
WinWaitActive($Wintitle_1, $Wintxt_1);
WinKill($handleWindows)
WinClose($Wintitle_1)
```

4. 自定义日志操作

```au3
Func ControlClickWithLog( $winTitle, $contrlTxt, $contrlId, $logFileName, $succMsg, $failMsg)
    $clicked=ControlClick($winTitle,$contrlTxt,$contrlId)
    if( $clicked ) Then
        $msg = $succMsg
    Else
        $msg = $failMsg
    EndIf
    _FileWriteLog( $logFileName, $msg)  
    Return $clicked
EndFunc
```

5. 预定义宏

```au3
@HomePath
@WindowsDir
@DesktopDir
@TempDir
```

6. Send

```
! ALT
+ SHIFT
^ CTRL
# WIN
Send ( "keys" [, flag = 0] )

```

## 参考

[1]. [中文文档]<http://www.jb51.net/shouce/autoit/>
[2]. [AutoIt Help]<https://zh.scribd.com/document/211815582/Autoit-Help-Manual>