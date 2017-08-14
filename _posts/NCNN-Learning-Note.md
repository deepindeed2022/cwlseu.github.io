## 从C++ 到android
在ncnn中是用C++写的，没玩过android很是愧疚。幸好项目中有android依赖链的cmake文件。

	Android CMake toolchain file, for use with the Android NDK r5-r10d 
	Requires cmake 2.6.3 or newer (2.8.9 or newer is recommended).
	See home page: https://github.com/taka-no-me/android-cmake

	## Usage Linux:
	$ export ANDROID_NDK=/absolute/path/to/the/android-ndk
	$ mkdir build && cd build
	$ cmake -DCMAKE_TOOLCHAIN_FILE=path/to/the/android.toolchain.cmake ..
	$ make -j8

	## Usage Windows:
	You need native port of make to build your project.
	Android NDK r7 (and newer) already has make.exe on board.
	For older NDK you have to install it separately.
	For example, this one: http://gnuwin32.sourceforge.net/packages/make.htm

	$ SET ANDROID_NDK=C:\absolute\path\to\the\android-ndk
	$ mkdir build && cd build
	$ cmake.exe -G"MinGW Makefiles"
	-DCMAKE_TOOLCHAIN_FILE=path\to\the\android.toolchain.cmake
	-DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%\prebuilt\windows\bin\make.exe" ..
	$ cmake.exe --build .

内心很是欣喜，再次凸显注释的重要性。沿着提示信息，我就学习了一下android-cmake

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=android.toolchain.cmake \
      -DANDROID_NDK=<ndk_path>                       \
      -DCMAKE_BUILD_TYPE=Release                     \
      -DANDROID_ABI="armeabi-v7a with NEON"          \
      <source_path>
cmake --build .
```

这个跟直接在x86平台最大的区别就是需要额外指定cmake tool chain的配置文件`android.toolchain.cmake`和tool chain的依赖路径`-DANDROID_NDK=<ndk_path>`，其他跟直接使用CMakeLists.txt相似。让我再次感慨，CMake很是666的。另外`-DANDROID_ABI="armeabi-v7a with NEON" `是用来制定编译target的一些属性和优化方案等等。这个具体可以到`android.toolchain.cmake`文件中看个究竟，而且在注释中有相应的说明。

## 实战C++编译android lib

## 宏观剖析NCNN
### 涉及技术
1. OpenMP

```cpp
    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
      ...
	}
```
2. SIMD
3.  loop unrolling
4. CNN的基本概念

主要实现CPU上的forward和backward的加速方案。其中卷积过程主要采用直接进行卷积计算的方式，没有采用img2col和FFT的方式，就是对直接卷积过程进行了实现和加速。

### 卷积计算方法分析


	--src
		|--blob.cpp   
		|--blob.h
		|--cpu.cpp
		|--cpu.h
		|--layer
		|--layer.cpp
		|--layer.h
		|--mat.cpp
		|--mat.h
		|--mat_pixel.cpp
		|--net.cpp
		|--net.h
		|--opencv.cpp
		|--opencv.h
		|--platform.h.in


cpu.h: 主要获取cpu的个数，频率，设置省电模式，以及动态调整的函数。

#### Mat
> mat.h：主要数据结构。
在其中扮演着参数传递和图像存储的作用。如果想对内存管理进行了解，可以好好看看mat.cpp的相关实现。理解这个数据结构，对后面理解优化过程很有帮助。
```cpp
// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
static inline void NCNN_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif
```
是编译器的一些特性函数，此处主要用来实现原子加法操作了。毕竟将来是多线程运行的，引用计数错误可是会导致内存泄露的。

> 几个有用的内存管理对齐方案

```c
// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}
// 原来不止python可以使用负下标，C也是可以的喽
static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}
```

> mat_pixel.cpp： 

主要用来实现Mat与图像的流的转化，纯手撸，没有依赖opencv，没有依赖opencv，没有依赖opencv. **但是图片是怎么输入的呢**

> 图片如何输入的

在opencv.h和opencv.cpp中实现了一个缩小版本的的图片输入输出文件，默认是不使用的。这个我得试试
`option(NCNN_OPENCV "minimal opencv structure emulation" OFF)`
因为我看example中使用的不是自己实现的这个文件输入输出，而是直接调用OpenCV的图片读取接口，然后调用的`ncnn::Mat::from_pixels_resize(...)`进行的数据转化，加上那么多格式的图片，所以这个重新进行自己运行确认一下比较好。

> blob.h 
主要用来记录层与层之间关系的，采用了生产者和消费者模型实现的。其使用的地方在`Net`中的一个参数`blobs`

```cpp
class Blob
{
public:
    // empty
    Blob();

public:
#if NCNN_STRING
    // blob name
    std::string name;
#endif // NCNN_STRING
    // layer index which produce this blob as output
    int producer;
    // layer index which need this blob as input
    std::vector<int> consumers;
};
```

> net.h 这个整个神经网络模型的框架实现。重要的有
Net如何加载模型和参数的

## 涨姿势

> 网络结构
```cpp
class Net
{
protected:
    std::vector<Blob> blobs;
    std::vector<Layer*> layers;
    std::vector<layer_registry_entry> custom_layer_registry;
};
```
> 注册函数

```cpp
// layer factory function
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
    // layer type name
    const char* name;
    // layer factory entry
    layer_creator_func creator;
};

// get layer type from type name
int layer_to_index(const char* type);
// create layer from layer type
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name) \
    Layer* name##_layer_creator() { return new name; }

} // namespace ncnn


```
> 添加自定义网络层次

```cpp
int Net::register_custom_layer(const char* type, layer_creator_func creator)
{
    int typeindex = layer_to_index(type);
    if (typeindex != 0)
    {
        fprintf(stderr, "can not register build-in layer type %s\n", type);
        return -1;
    }

    int custom_index = custom_layer_to_index(type);
    if (custom_index == -1)
    {
        struct layer_registry_entry entry = { type, creator };
        custom_layer_registry.push_back(entry);
    }
    else
    {
        fprintf(stderr, "overwrite existing custom layer type %s\n", type);
        custom_layer_registry[custom_index].name = type;
        custom_layer_registry[custom_index].creator = creator;
    }

    return 0;
}
```
## reference
1. [android cmake入门指导](https://github.com/taka-no-me/android-cmake)
2. [SMP Symmetric Multi-Processor](https://www.ibm.com/developerworks/cn/linux/l-linux-smp/index.html)  
