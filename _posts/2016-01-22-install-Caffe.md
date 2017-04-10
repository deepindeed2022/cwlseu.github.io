---
layout: post
title: Caffe framework 
categories: [blog ]
tags: [Caffe, ]
description: Caffe install in Ubuntu
---



声明：本博客欢迎转发，但请保留原作者信息!
作者: [cwlseu]
博客： <https://cwlseu.github.io/>

建议直接在机器上安装linux进行下面操作，要是在虚拟机里整，几乎没有什么戏，而且会把你给整疯了了的。

### 查看机器参数是否满足CUDA计算的最低要求
lspci | grep -i nvidia
01:00.0 3D controller: NVIDIA Corporation GF117M [GeForce 610M/710M/820M / GT 620M/625M/630M/720M] (rev a1)
参照nvidia [往年发布的gpus](http://developer.nvidia.com/cuda-gpus)
我的机器为Compute Capability 2.1，是可以使用CUDA加速的。：）

### CUDA 
我当前尝试安装[version-7.5 CUDA](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb)

`sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb`
`sudo apt-get update`
`sudo apt-get install cuda`


### The basic dependencies
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler 

sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev 


### 安装BLAS
BLAS 可以通过mkl atlas openblas等实现，[性能比较](http://www.wdong.org/wordpress/blog/2013/08/30/mkl-vs-atlas-vs-openblas/)
发现这个mkl是不错的，但是要[收费](https://software.intel.com/en-us/intel-mkl/)
最后选择默认的[Atlas](http://sourceforge.net/settings/mirror_choices?projectname=math-atlas&filename=Stable/3.10.2/atlas3.10.2.tar.bz2)

********** Important Install Information: CPU THROTTLING ***********
    Architecture configured as  Corei2 (27)
    /tmp/ccp8Kkgo.o: In function `ATL_tmpnam':
    /home/charles/Repo/ATLAS//CONFIG/include/atlas_sys.h:224: warning: the use of `tmpnam' is dangerous, better use `mkstemp'

    Clock rate configured as 800Mhz

    Maximum number of threads configured as  4
    probe_pmake.o: In function `ATL_tmpnam':
    /home/charles/Repo/ATLAS//CONFIG/include/atlas_sys.h:224: warning: the use of `tmpnam' is dangerous, better use `mkstemp'
    Parallel make command configured as '$(MAKE) -j 4'
    CPU Throttling apparently enabled!
    It appears you have cpu throttling enabled, which makes timings
    unreliable and an ATLAS install nonsensical.  Aborting.
    See ATLAS/INSTALL.txt for further information
    xconfig exited with 1
******************************* Solution ***************************
use ubuntu main software source 
switch to root admin

apt-get install gnome-applets
cpufreq-selector -g performance -c 0

sudo apt-get install libatlas-base-dev 

Unpacking libatlas-base-dev (3.10.1-4) ...
Setting up libgfortran3:amd64 (4.8.4-2ubuntu1~14.04) ...
Setting up libatlas3-base (3.10.1-4) ...
Setting up libblas3 (1.2.20110419-7) ...
Setting up libblas-dev (1.2.20110419-7) ...
Setting up libatlas-dev (3.10.1-4) ...
Setting up libatlas-base-dev (3.10.1-4) ...

### 安装Boost
* preinstall boost should install following software
* compile the source code 
下载源代码，当前最新版本为version 1.60
wget http://downloads.sourceforge.net/project/boost/boost/1.60.0/boost_1_60_0.tar.gz
unpacking boost 1.60.tar.gz
source boot
./b2
./b2 install --prefix=/usr/local

```
    #include <boost/lexical_cast.hpp>
    #include <iostream>
    int main()
    {
        using boost::lexical_cast;
        int a = lexical_cast<int>("123");
        double b = lexical_cast<double>("123.12");
        std::cout<<a<<std::endl;
        std::cout<<b<<std::endl;
    return 0;
    }
```
### 安装 Caffe

Optional dependencies:

    OpenCV >= 2.4 including 3.0
    IO libraries: lmdb, leveldb (note: leveldb requires snappy)
    cuDNN for GPU acceleration (v3)


### 安装 OpenCV
wget https://github.com/Itseez/opencv/archive/3.1.0.zip

### install cuDNN
PREREQUISITES
    CUDA 7.0 and a GPU of compute capability 3.0 or higher are required.
Extract the cuDNN archive to a directory of your choice, referred to below as <installpath>.Then follow the platform-specific instructions as follows.

LINUX

    cd <installpath>
    export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH

    Add <installpath> to your build and link process by adding -I<installpath> to your compile
    line and -L<installpath> -lcudnn to your link line.

WINDOWS

    Add <installpath> to the PATH environment variable.

    In your Visual Studio project properties, add <installpath> to the Include Directories 
    and Library Directories lists and add cudnn.lib to Linker->Input->Additional Dependencies.


### In ubuntu 14.04, we can use the following commend to sample our install steps.
sudo apt-get install nvidia-cuda-toolkit

###安装Boost
* preinstall boost should install following software

* compile the source code 
下载源代码，当前最新版本为version 1.60
wget http://downloads.sourceforge.net/project/boost/boost/1.60.0/boost_1_60_0.tar.gz
boost 1.60


### Ubuntu 14.04 && CUDA 7.5
打开安装了CUDA的ubuntu14.04发现，开机的过程中一直停止在开机等待界面，无法进入。

通过选择recovery mode进行恢复之后，然后重启，重启之后才能正常进入。然而，这不是一劳永逸的。等下一次再次开机重新进入的时候，又遇到了同样的问题，让我不得其解。

后来经过调研和重新格式化系统进行安装之后发现，原来是CUDA7.5 的.deb对Ubuntu 14.04 的支持性不好，导致显示驱动程序有问题，从而无法正常进入系统。而且有人建议采用.run的toolkit进行安装。可是又有新的问题出现。

### 不是所有Nvida显卡都支持Cudnn的
折腾了很久的cudnn安装，后来才发现是自己的显卡太low了，不支持Cudnn，因为Compute Capability 才2.1，要支持Cudnn， Capability >= 3.0，[查看自己显卡的计算能力](https://developer.nvidia.com/cuda-gpus)

### 从7.5之后安装的方法简单得多
`sudo apt-get --purge remove nvidia-*`
到https://developer.nvidia.com/cuda-downloads下载对应的deb文件
到deb的下载目录下

```sh
    sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb 
    sudo apt-get update 
    sudo apt-get install cuda
    sudo reboot
```
完成，cuda和显卡驱动就都装好了；其他的什么都不用动
而网上大部分中文和英文的参考教程都是过时的，折腾几个小时不说还容易装不成。

## 综合一个安装Caffe的教程
直接从<https://gwyve.github.io/>博客中拷贝过来的。

## 引言      

使用NVIDIA GPU进行dnn目前已经成为了主流，年前就打算自行安装一遍，拖了这么长时间，到今天才弄得差不多了。本来觉得这个不打算写个东西的，后来怕忘了，还是写下来吧

## 设备介绍

主机: ThinkStation-P300       
CPU: Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz              
GPU: Tesla K20c

## 所需软件

以下软件，在国内只有tensorflow需要翻墙，NVIDIA的都可以直接下载，速度还可以的

### Ubuntu

[Ubuntu 16.04.2 LTS](https://www.ubuntu.com/download/alternative-downloads)                   
[file_torrent](http://releases.ubuntu.com/16.04/ubuntu-16.04.2-desktop-amd64.iso.torrent?_ga=1.169319585.1810803403.1486517128)

### 驱动

[Nvidia-375.39](http://www.nvidia.cn/download/driverResults.aspx/115286/cn)                               
[file](http://cn.download.nvidia.com/XFree86/Linux-x86_64/375.39/NVIDIA-Linux-x86_64-375.39.run)

### cuda

[cuda-8.0](https://developer.nvidia.com/cuda-downloads)                          
[file](https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run)      

### cuDNN

[cuDNN_5.1](https://developer.nvidia.com/rdp/cudnn-download)                         
[file](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/cudnn-8.0-linux-x64-v5.1-tgz)


## 安装步骤

### 系统安装

安装Ubuntu的过程自行搜索吧，教程很多，在此不说了。

### 安装NVIDIA Driver

起初，我是选择使用直接安装cuda的，cuda中有driver，但是，cuda安装完了之后，重复出现登录界面，无法进入系统，所以，我选择单独安装driver，然后，再安装cuda。解决重复登录界面参考[参考1](http://blog.csdn.net/u012759136/article/details/53355781)

1.[选择](http://www.nvidia.cn/Download/index.aspx?lang=cn)机子所需要的驱动，并下载。                  
2.卸载原有驱动：

```bash
$ sudo apt-get remove –purge nvidia*
```     

3.关闭nouveau                                   
创建 /etc/modprobe.d/blacklist-nouveau.conf并包含

```bash
blacklist nouveau
options nouveau modeset=0
```
在terminal输入

`sudo update-initramfs -u`    

4.进入命令行模式                            
Ctrl+Alt+F1              
5.关闭lightdm服务 

`sudo service lightdm stop`

6.运行驱动文件
改变权限       

`sudo chmod a+x NVIDIA-Linux-x86_64-375.39.run`

运行  **注意参数**

`sudo ./NVIDIA-Linux-x86_64-375.39.run –no-x-check –no-nouveau-check –no-opengl-files`  

- no-x-check 安装驱动时关闭X服务
- no-nouveau-check 安装驱动时禁用nouveau
- no-opengl-files 只安装驱动文件，不安装OpenGL文件

7.重启，不出现循环登录问题


### 安装cuda

本来是按照deb安装的，后来各种问题，就改成选择runfile的方式了。

这里主要参考[参考2](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#runfile-nouveau-ubuntu)，全是英文的，要是不想看英文的话，我觉得，那还是放弃做dnn吧，目前这个前沿领域中文文献比较少～

1.在运行.run文件之后，在选择是否安装驱动的位置选择no，剩下的都是yes。                       
2.添加环境变量                      
打开 ～/.bashrc在最后添加

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
```

在terminal输入  
`source ~/.bashrc`
3.测试cuda安装是否成功
`nvcc -V`
输入cuda的版本信息                       
4.测试samples                      
这部分参考[参考3](http://blog.csdn.net/u012235003/article/details/54575758)
           
进入 NVIDIA_CUDA-8.0-Samples/
```bash
$ make
```
运行NVIDIA_CUDA-8.0-Samples/bin/x86_64/linux/release/deviceQuery


显示最后出现“Resalt=PASS”，代表成功

### 安装cuDNN
安装之前一定要确认你的GPU是支持cuDNN的。
这个都不应该叫做安装，就是一个创建链接的过程。这个主要参考[参考4](http://blog.csdn.net/jk123vip/article/details/50361951)

1.下载tar文件，解压                         
解压出一个叫做cuda的文件夹，以下操作都是在该文件夹下进行                  
2.复制文件  

```bash
sudo cp include/cudnn.h /usr/local/include
sudo cp lib64/libcudnn.so* /usr/local/lib
```
3.创建链接

```bash
$ sudo ln -sf /usr/local/lib/libcudnn.so.5.1.10 /usr/local/lib/libcudnn.so.5
$ sudo ln -sf /usr/local/lib/libcudnn.so.5 /usr/local/lib/libcudnn.so
$ sudo ldconfig -v
```


## 参考

1.[【解决】Ubuntu安装NVIDIA驱动后桌面循环登录问题](http://blog.csdn.net/u012759136/article/details/53355781)                                
2.[NVIDIA CUDA Installation Guide for Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#runfile-nouveau-ubuntu)         
3.[二、CUDA安装和测试](http://blog.csdn.net/u012235003/article/details/54575758)                
4.[import TensorFlow提示Unable to load cuDNN DSO](http://blog.csdn.net/jk123vip/article/details/50361951)                        
5.[Installing TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux)