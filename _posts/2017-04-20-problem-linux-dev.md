---
layout: post
title: "Linux开发中的问题录"
categories: [blog ]
tags: [Linux开发]
description:  开发中的问题记录，当前主要为安装问题
---


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

http://stackoverflow.com/questions/32405035/caffe-installation-opencv-libpng16-so-16-linkage-issues



## 
![@build opencv using cuda](../images/linux/opencv-cuda.png)

```sh
1>------ Build started: Project: opencv_cudalegacy, Configuration: Debug x64 ------
1>  graphcuts.cpp
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(120): error C2061: syntax error: identifier 'NppiGraphcutState'
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(135): error C2833: 'operator NppiGraphcutState' is not a recognized operator or type
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(135): error C2059: syntax error: 'newline'
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(136): error C2334: unexpected token(s) preceding '{'; skipping apparent function body
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(141): error C2143: syntax error: missing ';' before '*'
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(141): error C4430: missing type specifier - int assumed. Note: C++ does not support default-int
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(141): error C2238: unexpected token(s) preceding ';'
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(127): error C2065: 'pState': undeclared identifier
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(132): error C2065: 'pState': undeclared identifier
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(132): error C3861: 'nppiGraphcutFree': identifier not found
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(174): error C3861: 'nppiGraphcutGetSize': identifier not found
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(182): error C2065: 'nppiGraphcutInitAlloc': undeclared identifier
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(190): error C3861: 'nppiGraphcut_32s8u': identifier not found
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(195): error C3861: 'nppiGraphcut_32f8u': identifier not found
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(246): error C3861: 'nppiGraphcut8GetSize': identifier not found
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(254): error C2065: 'nppiGraphcut8InitAlloc': undeclared identifier
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(264): error C3861: 'nppiGraphcut8_32s8u': identifier not found
1>V:\Opencv31\opencv\sources\modules\cudalegacy\src\graphcuts.cpp(271): error C3861: 'nppiGraphcut8_32f8u': identifier not found
2>------ Build started: Project: opencv_cudaoptflow, Configuration: Debug x64 ------
2>LINK : warning LNK4044: unrecognized option '/LV:/NVIDIA GPU Computing Toolkit/Cuda8/lib/x64'; ignored
2>LINK : fatal error LNK1104: cannot open file '..\..\lib\Debug\opencv_cudalegacy310d.lib'
```

[cudalegacy-not-compile-nppigraphcut-missing](http://answers.opencv.org/question/95148/cudalegacy-not-compile-nppigraphcut-missing/)
这个问题在opencv中已经fix掉了。