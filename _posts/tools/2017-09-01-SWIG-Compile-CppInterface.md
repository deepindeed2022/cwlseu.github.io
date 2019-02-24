---
layout: post
title: "Python：SWIG编译C++接口"
categories: [blog ]
tags: [工具, Python]
description: SeetaFace人脸识别引擎提供了人脸识别系统所需的三个核心模块。为了使用方便，决定使用swig编译python接口进行使用。
---

{:toc}


## 来源

SeetaFaceEngine使用C++编译，而且使用OpenMP技术和向量化技术进行加速，已经基本可以满足业界对人脸识别功能的需求。在项目中用到人脸识别
功能，OpenCV自带的基于Haar特征的算法，效果不理想，仅仅能够识别正脸，人脸歪一定的角度都不能够识别。使用SeetaFaceEngine需要重新编译python接口，对于没有接触过的人来说还真不简单，在此新路记录。
[SeetaFaceEngine源代码](https://github.com/seetaface/SeetaFaceEngine) 

## SWIG

[SWIG（Simplified Wrapper and Interface Generator）](http://swig.org)是一个为C/C++库提供脚本调用支持的工具，支持Lua, Perl, Python, Go等多种脚本语言。[1]中详细介绍了如何使用SWIG编写C/C++接口，很是详细。

### 安装

Ubuntu下使用`sudo apt-get install swig`就可以安装swig, 当然也可以下载[源码](http://swig.org/)进行安装
安装`sudo apt-get install boost-dev`，这是为了支持numpy.ndarray和cv::Mat的转化
安装OpenCV-2.4，最好从源代码进行编译安装，这个网上教程很多，就不说了

### 编写.i

```c
%module pyfacedetect # 要生成的python模块的名称
%{
#define SWIG_FILE_WITH_INIT
#include <vector>
#include "crop_img.h"
%}
%include "crop_img.h"
```

## numpy.ndarray和cv::Mat
首先需要明确的是opencv在C++和python中表示一张图片是不同的。直接进行测试就可以知道：

```python
import cv2
img_file = "../img/1.jpg"
img = cv2.imread(img_file)
print type(img)
```
输出结果为：
    <type 'numpy.ndarray'>
而在C++中通过`cv::Mat cv::imread(string filename, cv::IMREAD_GRAYSCALE);`是直接返回`cv::Mat`类型的。因此，要想实现python调用原来的数据，需要实现一层数据结构的适配工作。网上的解决方案有很多，大部分人都没有说明opencv的版本，尤其是现在OpenCV 3.*版本已经日益普及，但是OpenCV2.4仍然不断发展的今天，我们在进行实际使用过程中，遇到OpenCV相关的问题，需要明确我们的版本是什么，网上解决方案对应的版本是什么。
1. (convert the numpy.ndarray to a cv::Mat using Python/C API)[https://stackoverflow.com/questions/22667093/how-to-convert-the-numpy-ndarray-to-a-cvmat-using-python-c-api]
2. (create Python/Cython wrapper for C++ library that uses cv::Mat class from OpenCV. )[https://stackoverflow.com/questions/22736593/what-is-the-easiest-way-to-convert-ndarray-into-cvmat/25382316#25382316]
上述两种方案都是在opencv 2中的方案，对于OpenCV 3就不适应了。

[github 代码](https://github.com/spillai/numpy-opencv-converter.git)中添加了对opencv 3的支持，虽然我在opencv 3.1上没有编译成功，但是从中
可以看出opencv 在v2 -> v3的过程中发生的部分变化。

```cpp
#if OPENCV_3
    m.addref();
    Py_INCREF(o);
#else
    m.refcount = refcountFromPyObject(o);
    m.addref();     // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
#endif
```

OpenCV3 中的新数据结构`UMatData`和`cv::AutoBuffer`

## 一些问题小结
1. printf的实现方法
```cpp
static int printf(const char *fmt, ...)
{
    char str[1000];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);
    return 0;
}
```

2. swig对C的API支持性比较好，往往可以采用将函数返回`void*`的方式, 例如：
`void* detectface(cv::Mat& img, const char* modelpath);`
然后再实际返回的类型重新封装，其中要包含以`void*`为输入的构造函数，例如`Rects`

```cpp
Rects(void* _rects)
{
    Rects* rects = (Rects*)_rects;
    num = rects->num;
    data = (Rect*) malloc(sizeof(Rect)*num);
    memcpy((char*)data, (char*)rects->data, sizeof(Rect)*num);
}
```
这样就可以实现在python中调用`rects = Rects(detectface(mat, model_path))`就可以获得结果。

## 生成代码 & 测试

### 依赖库
使用的swig生成的_pyxxxxx.so文件是依赖原来的C++ lib的，因此，这个lib放置的位置需要让_pyxxxxx.so能够找到。
否则会出现连接lib找不到的问题。因此，最方便的方法就是直接进行编译C++库的时候，将库安装到系统路径下。

### swig生成代码

```python
from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_pyfacedetect', [dirname(__file__)])
        except ImportError:
            import _pyfacedetect
            return _pyfacedetect
        if fp is not None:
            try:
                _mod = imp.load_module('_pyfacedetect', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _pyfacedetect = swig_import_helper()
    del swig_import_helper
else:
    import _pyfacedetect
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)

# 打印swig生成的对象的信息
def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0

class Rect(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Rect, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Rect, name)
    __repr__ = _swig_repr
    __swig_setmethods__["x"] = _pyfacedetect.Rect_x_set
    __swig_getmethods__["x"] = _pyfacedetect.Rect_x_get
    if _newclass:
        x = _swig_property(_pyfacedetect.Rect_x_get, _pyfacedetect.Rect_x_set)
    __swig_setmethods__["y"] = _pyfacedetect.Rect_y_set
    __swig_getmethods__["y"] = _pyfacedetect.Rect_y_get
    if _newclass:
        y = _swig_property(_pyfacedetect.Rect_y_get, _pyfacedetect.Rect_y_set)
    __swig_setmethods__["width"] = _pyfacedetect.Rect_width_set
    __swig_getmethods__["width"] = _pyfacedetect.Rect_width_get
    if _newclass:
        width = _swig_property(_pyfacedetect.Rect_width_get, _pyfacedetect.Rect_width_set)
    __swig_setmethods__["height"] = _pyfacedetect.Rect_height_set
    __swig_getmethods__["height"] = _pyfacedetect.Rect_height_get
    if _newclass:
        height = _swig_property(_pyfacedetect.Rect_height_get, _pyfacedetect.Rect_height_set)

    def __init__(self, *args):
        this = _pyfacedetect.new_Rect(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _pyfacedetect.delete_Rect
    __del__ = lambda self: None
Rect_swigregister = _pyfacedetect.Rect_swigregister
Rect_swigregister(Rect)

class Rects(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Rects, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Rects, name)
    __repr__ = _swig_repr
    __swig_setmethods__["num"] = _pyfacedetect.Rects_num_set
    __swig_getmethods__["num"] = _pyfacedetect.Rects_num_get
    if _newclass:
        num = _swig_property(_pyfacedetect.Rects_num_get, _pyfacedetect.Rects_num_set)
    __swig_setmethods__["data"] = _pyfacedetect.Rects_data_set
    __swig_getmethods__["data"] = _pyfacedetect.Rects_data_get
    if _newclass:
        data = _swig_property(_pyfacedetect.Rects_data_get, _pyfacedetect.Rects_data_set)

    def __init__(self, *args):
        this = _pyfacedetect.new_Rects(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def getRect(self, idx):
        return _pyfacedetect.Rects_getRect(self, idx)
    __swig_destroy__ = _pyfacedetect.delete_Rects
    __del__ = lambda self: None
Rects_swigregister = _pyfacedetect.Rects_swigregister
Rects_swigregister(Rects)

def detectface(*args):
    return _pyfacedetect.detectface(*args)
detectface = _pyfacedetect.detectface
```

## 测试脚本：

```python
from pyfacedetect import detectface, Rects, Rect
import cv2


model_path = "../model/seeta_fd_frontal_v1.0.bin"
def face_detect(img_file):
	try:
		rects = Rects(detectface(img_file, model_path))
		n = rects.num
		res = []
		for i in xrange(n):
			res.append(Rect(rects.getRect(i)))
		return res
	except IOError:
	 	raise IOError("Image Not Found")

def face_detect_mat(mat):
	try:
		print "in face detect mat"
		rects = Rects(detectface(mat, model_path))
		print "detect face finished"
		n = rects.num
		res = []
		for i in xrange(n):
			res.append(Rect(rects.getRect(i)))
		return res
	except IOError:
	 	raise IOError("Image Not Found")


def test_face_detect():
	img_file = "../img/5.jpg"
	result = detectface(img_file, "../../model/seeta_fd_frontal_v1.0.bin")
	rects = Rects(result)
	print rects.num
	for i in xrange(rects.num):
		rect = Rect(rects.getRect(i))
		print rect.x, rect.y, rect.width, rect.width
if __name__ == '__main__':
	test_face_detect()
```

## 结果

![@facedet](https://cwlseu.github.io/images/python/facedet.jpg)

![@visual detection](https://cwlseu.github.io/images/python/facedetresult.jpg)


## 参考文献
[1]. Interfacing C/C++ and Python with SWIG <http://www.swig.org/papers/PyTutorial98/PyTutorial98.pdf>

[2]. numpy opencv converter source code for opencv 2 <https://github.com/yati-sagade/opencv-ndarray-conversion.git>

[3]. numpy opencv converter source code for opencv 2 and 3 <https://github.com/spillai/numpy-opencv-converter.git>