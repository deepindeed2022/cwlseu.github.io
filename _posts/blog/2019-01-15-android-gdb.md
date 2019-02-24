---
layout: post
title: android手机上gdb调试
categories: [blog ]
tags: [android]
description: android 开发
---

{:toc}


## 前言

在android开发C++过程，除了使用logcat进行追踪程序之外，更加便捷的方式便是像在linux上开发使用gdb程序进行调试一样调试android程序，只不过程序不是运行在linux机器上，而是在某个android手机上。
常见的android手机上程序调试有： 
1. 通过打印log，这种方式对于比较庞大的项目有利于迅速定位问题的模块，但是，如果引用第三方库，某些程序调试起来会很无力。而且打印log的程序往往比较大，不利于部署发布，因此release版本往往都没有log
2. 通过ssh连接，使用gdb进行调试，这种便于团队开发。
3. 直接将手机usb连接在电脑上进行调试，这种情况适合个人开发

## SSH连接进行调试

1. 将相关运行程序和依赖库拷贝到手机某个目录下，例如我将sdk_xxx拷贝到`root@10.1.42.15:/data/data/berserker.android.apps.sshdroid/home/cwl/sdk_xxx`
2. 从本地机器中交叉编译链中查找gdb
例如我的机器上是在`$NDK_DIR/prebuilt/android-arm64/gdbserver`
到手机某个位置下，例如我放在跟项目相同目录下
3. andorid手机上运行 `gdbserver [port] [exe file]`
运行该程序之前，先确认该[exe file]是否需要动态链接其他库，如果需要添加动态库路径到`LD_LIBRARY_PATH`
执行`export LD_LIBRARY_PATH=./target/android-aarch64/test`
然后再执行如下程序
`./gdbserver :5039 target/android-aarch64/test_sdk_xxx`
将test_sdk_xxx与端口5039绑定

4. 在本地查找对应的gdb程序进行调试例如
`$NDK_DIR/android-aarch64/bin/aarch64-linux-android-gdb ./target/android-aarch64/test/test_sdk_xxx`
如果ndk14b找不该程序，可以使用ndk10里面的gdb进行调试
```
(gdb) help target
target core -- Use a core file as a target
target exec -- Use an executable file as a target
target extended-remote -- Use a remote computer via a serial line
target record -- Log program while executing and replay execution from log
target record-btrace -- Collect control-flow trace and provide the execution history
target record-core -- Log program while executing and replay execution from log
target record-full -- Log program while executing and replay execution from log
target remote -- Use a remote computer via a serial line
target tfile -- Use a trace file as a target

Type "help target" followed by target subcommand name for full documentation.
Type "apropos word" to search for commands related to "word".
Command name abbreviations are allowed if unambiguous.
```

```
(gdb) target remote 10.1.42.15:5039
(gdb) continue
(gdb) bt
```

## 连接真机gdb

### 查看手机usb

	cwl@ubuntu:~$ lsusb 
	Bus 002 Device 004: ID 18c3:6255  
	Bus 002 Device 002: ID 8087:0020 Intel Corp. Integrated Rate Matching Hub
	Bus 002 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
	Bus 001 Device 005: ID 22b8:41db Motorola PCS Motorola Droid (USB Debug)
	Bus 001 Device 004: ID 04d9:a06b Holtek Semiconductor, Inc. 

### 添加udev规则

udev就是一个动态硬件管理服务 

	cwl@ubuntu:/etc/udev/rules.d$ cd /etc/udev/rules.d/
	cwl@ubuntu:/etc/udev/rules.d$ sudo vi 50-android-usb.rules
	 
编辑规则文件并保存 
 `SUBSYSTEM=="usb", SYSFS("Motorola PCS Motorola Droid (USB Debug)")=="22b8",MODE="0666"`
 其中，`sysfs`括号内是自己android手机的实际描述信息，==后面的是id号，`mode`是读取模式，`0666`是所有人可以访问，以上的信息都是`lsusb`查处来的。
 
### 设置规则文件权限并重启`udev`

	cwl@ubuntu:/etc/udev/rules.d$ sudo chmod a+rx /etc/udev/rules.d/50-android-usb.rules 
	cwl@ubuntu:/etc/udev/rules.d$ sudo /etc/init.d/udev restart 

 会看到udev相关的提示信息
`adb push target/android-aarch64/test/test_sdk_xxx /data/local/tmp/sdk_xxx`


## 其他可能问题
在手机里面执行文件的时候提示`can't execute: Permission denied`
一开始以为是没有root权限，自己傻逼了，错误意思是，不能执行，权限定义，

### 解决办法
`chmod +x filename`给文件可执行就可以。
一般把文件放到 `/data/local/tmp/`目录下
然后 `chmod +x file`

## 小结

交叉编译和开源项目，使得交叉编译、跨平台开发等问题越来容易，才能够有这种在android 调试程序和在linux上调试程序毫无间隙地切换。

## 参考链接
[https://www.cnblogs.com/hangxin1940/archive/2011/07/10/2101552.html](https://www.cnblogs.com/hangxin1940/archive/2011/07/10/2101552.html)