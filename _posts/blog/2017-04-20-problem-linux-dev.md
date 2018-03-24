---
layout: post
title: "Linux开发中的问题录"
categories: [blog ]
tags: [Linux开发]
description:  开发中的问题记录，当前主要为安装问题
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## fatal error: metis.h: No such file or directory
### NOT sudo user
I am trying to install Metis. Since I am working on a public server, I couldn't install it as a root user. So I have installed metis in my account /home/jd/metis.

When I try to execute something, I get

> fatal error: metis.h: No such file or directory

I guess the system looks for metis.h under /usr/local/include but couldnt find it there. How do I make linux look for metis.h under /home/jd/metis/include directory?
I added this path to the $PATH variable. But still the same error. Please advise.

Work with cmake. Adding `include_directories("/home/xxx/metis/include")`

### sudo user
参看[stack-overflow](http://stackoverflow.com/questions/36046189/how-to-install-metis-on-ubuntu/41336362#41336362)

### 查看机器cpu信息

1. 查看物理CPU的个数
`cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l`
 
2. 查看逻辑CPU的个数
`cat /proc/cpuinfo | grep "processor" | wc -l`
 
3. 查看CPU是几核
`cat /proc/cpuinfo | grep "cores" | uniq`
 
4. 查看CPU的主频
`cat /proc/cpuinfo | grep MHz | uniq`

## caffe installation : opencv libpng16.so.16 linkage issues

```sh
usr/bin/ld: warning: libpng16.so.16, needed by /home/andrei/anaconda/lib/libopencv_highgui.so, not found (try using -rpath or -rpath-link)
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_create_read_struct@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_interlace_handling@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_IHDR@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_get_io_ptr@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_longjmp_fn@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_gray_to_rgb@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_compression_level@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_bgr@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_filter@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_rgb_to_gray@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_init_io@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_destroy_read_struct@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_swap@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_get_IHDR@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_palette_to_rgb@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_compression_strategy@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_get_tRNS@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_write_info@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_packing@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_set_read_fn@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_create_info_struct@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_read_end@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_read_update_info@PNG16_0'
/home/andrei/anaconda/lib/libopencv_highgui.so: undefined reference to 'png_write_image'
```

[caffe-installation-opencv-libpng16-so-16-linkage-issues](http://stackoverflow.com/questions/32405035/caffe-installation-opencv-libpng16-so-16-linkage-issues)

在编译Makefile或者CMake文件中添加`opencv_highgui`的链接信息，确认opencv 的lib路径是否已经添加到LD_LIBRARY_PATH中

## 误删/var/lib/dpkg/

首先创建一些文件
`sudo mkdir -p /var/lib/dpkg/{alternatives,info,parts,triggers,updates} `
从备份数据中恢复
`sudo cp /var/backups/dpkg.status.0 /var/lib/dpkg/status `

Now, lets see if your dpkg is working (start praying):
`apt-get download dpkg`
`sudo dpkg -i dpkg*.deb` 

修复base files
`apt-get download base-files`
`sudo dpkg -i base-files*.deb `

要选Y，要选Y要选Y，否则就会出现
`Setting up grub-pc (2.02~beta2-36ubuntu3.17) ...
Setting up unattended-upgrades (0.90ubuntu0.9) ...
`
这两关过不了的情况。

试试可以更新了不
`sudo apt-get update`
`sudo apt-get check`