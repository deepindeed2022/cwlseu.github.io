---
layout: post
title: Google Fuchsia
categories: [blog ]
tags: [操作系统, ]
description: Google Fuchsia OS
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## Google Fuchsia

### Fuchsia简介

该操作系统是为具有高速处理器和大容量内存的硬件而开发。它的Github页只是简要地将Google的这个新项目描述为为“Pink+Purple==Fuchsia（新的操作系统）”。
Chris McKillop是Google的产品和研发负责人，按照他的解释Purple指的是一个“具有高性能图像显示、输入延迟低、用户交互界面美观的系统”，Pink指的是“面向开发者和用户的模块化系统”。
Fuchsia目前运行于Intel Broadwell和Skylake处理器上，不过它也能够在增强后运行于比较老的Intel甚至AMD处理器上。不久之后，它也将增加对ARM和树莓派3的支持。该操作系统的是为“具有高速处理器和大容量内存的现代手机及个人电脑”而构建的，令人推测将会在未来代替安卓。Fuchsia的用户界面似乎是使用了Flutter控件框架，并用Escher渲染。Escher是一个基于物理的渲染器，支持复杂的特效，例如体阴影、色溢、光扩散等。高超的图像处理能力以及目标硬件平台需要具有高速处理器和大容量内存，表明该操作系统可能是针对虚拟现实的头戴设备。

### 团队介绍

#### 项目负责人：Chris McKillop 
Low level generalist with much experience in operating system development and hardware interfacing. Much experience in smartphone OS development. Experience in game engine development on modern multi-threaded game consoles. Interested in OS kernel design and implementation, game engine development, optimization and modern GPU based graphics development.
Specialties: Operating System design and implementation. Low level OS bringup on new hardware through high level design and architecture. Much experience in low level assembly, and getting the most out of compilers and optimizations. Experience in modern 3D accelerators at the hardware level, including shader development.
综上，就是项目负责人对操作系统开发设计与实现都有很强的功底， 此外，在游戏引擎方面有造诣，游戏引擎需要大量的图像处理开发技能，所以这个系统可能与VR设备有关。

#### 其他成员

团队包括Brian Swetland和Travis Geiselbrecht，他们过去曾从事Android、BeOS、ChromeOS、DangerOS、iOS、MacOS、QNX、webOS和其他操作系统的开发工作。
-  Brian Swetland 
kernel development, drivers, bootloaders, embedded systems, tools, nifty products
- Travis Geiselbrecht ： 
在多家公司造过轮子，如Jawbone, Palm, Secret Level, Apple, Danger Inc., Be Inc. 主要工作是操作系统内核的开发，下面是其部分业余工作：
NewOS - A OS project I started years ago and didn't know when to stop. It's reasonably well mature, and even scored a slahdotting and offer to write an article for Dr Dobbs magazine. It has been forked years ago and now forms the kernel for the Haiku project.
lk embedded kernel - little embedded style kernel for some ARM devices. source at git://github.com/travisg/lk.git. Web browseable here.
Arm Emulator - Hacked a fun little arm emulator. Still needs some work, but it currently emulates a "generic" ARM based system. Good for prototyping a kernel. SVN depot at git://github.com/travisg/armemu.git. Web browseable here.
Apple ][ Emulator - A cheezy little Apple ][ emulator I wrote a long time ago for fun. Source here. Here and here are a couple of screenshots.
两位核心开发成员都是有15年以上的内核开发经验的程序员，参与若干内核相关的工作。

### 技术架构

#### 内核：Magenta

有可以在特定设备运行的早期版本，框架设计、核心功能、相关工具链较完善。如果需要支持新的（外围）设备，还有大量驱动需要移植。

#### UI层：Flutter

组件库已经基本完善，写一些简单程序没有障碍，暂时还无法在 Fuchsia 运行。官方自称属于早期阶段的版本，实际也是。另外目前没有OpenGL ES 方面 3D相关的支持，官方承诺未来会公开自己的优化版 3D编程接口。（所以可能也是暂时没有炫酷亮闪未来界面的原因）

#### 渲染引擎：Escher
资料较少不了解，不过应该在渲染速度和效果上有非常突出的优势。

#### 主力编程语言：Dart
面向对象的跨平台语言，相关的库已经比较完善且是 Flutter 的开发语言。以后为 Fuchsia 开发App主要使用 Dart语言。

从搭建的框架可以看出，其特点基本上是其所选取组件的优点的组合，比较有代表性的是：

**适合嵌入式设备和高性能设备**：magenta内核的基础lk就是一个嵌入式系统的内核，它的代码非常的简洁，适合移植到不同的设备上。可以想象到的目标设备是：物联网、移动手持设备、可穿戴设备等。

**低延迟、高效率**：在 Magenta、Escher、Flutter 的项目介绍中都可以看到“实时”、"高效”、“低延迟”这几个关键字，那么可以预见 Fuchsia 的目标也是实现一个实时性非常高的操作系统。低延迟有什么用呢，想象一下VR眼镜上看虚拟现实的时候，画面没有延迟的惊艳爽快感。所以低延迟对这个操作系统一个非常重要的考虑指标，也是核心优点之一。顺便提一下，我们普通人用到的 Windows，Linux，Android 都不是实时操作系统。OSX,  iOS的延迟都比较低，但是一般也没有把他们划为实时操作系统。

**高级编程语言**：Dart 的目标是设计一个随处部署、接口稳定、基础库完善的开发语言。用在 Fuchsia 可以看出来要摆脱掉 Java 语言的意图比较明显，另外它也是一种需要VM的语言。不过Flutter 的官方资料显示：经过对比测试，Dart 的在执行性能、开发效率、面向对象、快速内存分配（回收）上的得分都非常高，因此才会被 Flutter 选为开发语言。

**统一的UI体验**：由于采用了 Matrial Design 设计语言，所以在 Fuchsia 上运行的程序理论上具有统一的UI体验。
没有历史包袱：完全从头设计的系统，不会有为了兼容考虑的历史包袱——比如 Java 虚拟机的慢速，又比如 Android 较慢的渲染速度。因此也可以把体积做的很小，塞到存储容量很小的设备中。

###源代码情况

![@图 1: 各个模块贡献人员的统计图](https://cwlseu.github.io/images/Fuchsia/sourcecode.jpg)

可以看到参与人数多代码提交比较活跃的，是内核 Magenta 和图形界面层 Flutter 两个项目。Flutter 项目比较特殊，它其实很早就启动了，一直致力于为 Android/iOS 移动设备提供编码统一的开发环境。除了 Flutter 之外，其他所有项目大概都是最近两个月内被启动的。其中目前最活跃的是操作系统内核部分，22 名贡献者中可以看到不少 Google, Chromium 官方成员在提交代码。内核开发极度需要技术和经验，也不是劳动力密集型工种，几名核心工程师已经足够，由此可见目前谷歌对内核项目的干劲还是比较足的，只是其他配套项目关注的人数就稍微少了点。

[Magenta内核代码地址](https://github.com/fuchsia-mirror/magenta)
[Google source 也有主页](https://fuchsia.googlesource.com/)

### 参与其中

1.  **参与内核开发**
Fuchsia是基于Magenta，这一部分是当前开发最为活跃的部分。Swetland(上面提到的主要开发人员之一)将Magenta描述为一个迷你内核：“97%的驱动和服务位于用户空间，但是系统调用面提供了更为广泛的基本指令，而不仅仅是核心微内核设计所采用的send/recv/exit。虽然继承于C语言写的LK，但是Manenta内核新的表面部分是用受限的C++写的。”[1]
Swetland称：“Magenta的驱动和服务大部分是用C语言写的，不过其中的一部分将会随着时间的推移用C++重写。”当然，任何人都能够添加使用其他语言编写的组件，只要它们是通过现有的RPC协议和内核通信。

2.  **构建应用程序**
Fuchsia使用Mojo来帮助构建应用程序，Mojo是"一个进程间通信技术和协议的集合，同时也是一个用于创建可组合、低耦合应用程序和服务的运行时。Pauli Olavi Ojala称："Mojo已经可以绑定Dart、Go、Java、JavaScript、Python和Rust等语言"。我们只要选择自己熟悉的语言，通过绑定的Mojo的方式，就可以进行应用程序。

3. **Magenta** 
Magenta，是一个微内核和一系列用户空间的服务、驱动的组合。目前它已经能够在虚拟机、某几款NUC小电脑和某款笔记本上启动运行。在虚拟机里面运行后就是一个字符终端，执行一个叫 mxsh 的 shell，另外还有少量的基本工具和测试程序集 。只有 Magenate 内核的 Fuchsia 系统，在虚拟机运行起来是这个样子：
  
![@图 2 内核安装运行图1](https://cwlseu.github.io/images/Fuchsia/1.png)
![图 2 内核安装运行图2](https://cwlseu.github.io/images/Fuchsia/2.png)
为了方便安装，编写了相应的安装脚本INSTALL_MAGENTA.sh安装测试是在ubuntu16.04LTS 上进行的，其他环境安装没有进行实验。
其中支持的命令：
 
![@图 3 内核支持的系统命令](https://cwlseu.github.io/images/Fuchsia/3.png)
当前该kernal内置40个测试用例，测试报告：magenta.txt
Magenta作为Fuchsia的内核，基本框架已经实现，同时还有API文档，详见
https://github.com/fuchsia-mirror/magenta/tree/master/docs
文档中说明了系统补丁应该如何贡献，内容如下：
***Contributing Patches to Magenta***
At this point in time, Magenta is under heavy, active development, and we're
not seeking major changes or new features from new contributors, but small
bugfixes are welcome.
Here are some general guidelines for patches to Magenta.  This list is
incomplete and will be expanded over time:
* GitHub pull requests are not accepted.  Patches are handled via
  Gerrit Code Review at: https://fuchsia-review.googlesource.com/#/q/project:magenta
* Indentation is with spaces, four spaces per indent.  Never tabs.
Do not leave trailing whitespace on lines.  Gerrit will flag bad
whitespace usage with a red background in diffs.
* Match the style of surrounding code.
* Avoid whitespace or style changes.  Especially do not mix style changes
with patches that do other things as the style changes are a distraction.
* Avoid changes that touch multiple modules at once if possible.  Most
changes should be to a single library, driver, app, etc.
* Include [tags] in the commit subject flagging which module, library,
app, etc, is affected by the change.  The style here is somewhat informal.
Look at past changes to get a feel for how these are used.
* Magenta should be buildable for all major targets (x86-64, arm64, arm32)
at every change.  ./scripts/build-all-magenta can help with this.
* Avoid breaking the unit tests.  Boot Magenta and run "runtests" to
verify that they're all passing.
* The #fuchsia channel on the freenode irc network is a good place to ask
questions.
这说明开源爱好者有了比较方便的入门教程。此外，我已经与项目主要贡献者通邮件(geist@foobox.com)，但至今还没有回信。
Flutter
Flutter 是可以运行在 Android 和 iOS 上的用户界面开发库，从它的源代码提交和bug跟踪日志中的信息看，目前它的引擎还不能运行在 Fuchsia 上，不过已经很接近可以工作。Flutter 官网声称自己还是一个早期阶段的开源项目，“未来” 操作系统上的程序可能会是什么样子，在[2]中进行了测试（请忽略Android自带黑边和某运营商标志）：
  
![图 4跑在 Android 手机上的 Flutter Gallery 演示程序1](https://cwlseu.github.io/images/Fuchsia/4.png)
![@图 4跑在 Android 手机上的 Flutter Gallery 演示程序2](https://cwlseu.github.io/images/Fuchsia/5.png)
Flutter采用 Materal Design 设计语言（规范），该规范定义了用户界面上的元素的用途、外观、展现形式以及形态变化的规范。

从用户可见的角度来看，未来 Fuchsia 操作系统内运行的程序，其中的按钮，对话框，图片框等等界面组件，基本就应该跟上面图片中差不多——当然未来也可能会改变——而那些科幻电影中炫酷亮眼的3D特效、隔空指点、虚拟（增强）现实画面，暂时还不能从演示程序中看到。

### 参考：
[1]. http://www.infoq.com/cn/news/2016/08/fuchsia?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io
[2]. http://news.zol.com.cn/600/6001486.html
[3]. http://news.mydrivers.com/1/495/495530.htm
[4]. https://segmentfault.com/a/1190000006758011#articleHeader3
[5].快速了解相关Fuchsia信息
https://www.cnet.com/news/google-fuchsia-challenger-to-windows-android/
[6].google source git:https://fuchsia.googlesource.com/
[7].Fuchsia 安装成功案例：http://www.jianshu.com/p/12f68e2e5753


### 附件
```sh
#!/bin/bash

# Test the fuchsia kernal magenta in Linux ubuntu 4.4.0-34-generic x86_64 x86_64 x86_64 GNU/Linux for Ubuntu16.04LTS
# BY: wenlong@nfs.iscas.ac.cn
# Date: 2016-9-27

# Prepare dependencies package in ubuntu
sudo apt-get -y update
sudo apt-get install -y texinfo libglib2.0-dev autoconf libtool libsdl-dev build-essential
sudo apt-get install -y bison flex


export SRC=$(pwd)
# clone kernal source 
git clone https://github.com/fuchsia-mirror/magenta.git

cd  $SRC/magenta
# prebuild the toolchain for gcc compiler
./scripts/download-toolchain

# If the prebuilt toolchain binaries do not work for you, there are a set of scripts which 
# will download and build suitable gcc toolchains for building Magenta for ARM32, ARM64, 
# and x86-64 architectures:
# cd $SRC
# git clone https://fuchsia.googlesource.com/third_party/gcc_none_toolchains toolchains
# cd toolchains
# ./doit -a 'arm aarch64 x86_64' -f -j32



# Build Qemu for testing in virtual machine, if you're only testing on actual
# hardware, skit this.
# If you don't want to install in /usr/local (the default), which will require 
# you to be root, add --prefix=/path/to/install (perhaps $HOME/qemu) and then you'll need to add /path/to/install/bin to your PATH.
cd $SRC
git clone https://fuchsia.googlesource.com/third_party/qemu
cd qemu
./configure --target-list=arm-softmmu, aarch64-softmmu, x86_64-softmmu
make -j8
sudo make install

cd $SRC/magenta

export PATH=$PATH:$SRC/toolchains/aarch64-elf-5.3.0-Linux-x86_64/bin
export PATH=$PATH:$SRC/toolchains/x86_64-elf-5.3.0-Linux-x86_64/bin
#build for x86-64
make -j8 magenta-pc-x86-64
# for aarch64
# make -j32 magenta-qemu-arm64
echo "BUILD magenta Success, the mirror file in $(pwd)/build-magenta-pc-x86-64"
cd $SRC

##run magenta
# cd $SRC/magenta 
# ./scripts/run-magenta-x86-64
```
