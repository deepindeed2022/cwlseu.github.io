---
layout: post
title: "Linux开发中的问题录"
categories: [blog ]
tags: [Linux开发]
description: 
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