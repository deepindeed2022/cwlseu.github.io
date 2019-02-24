---
layout: post
title: CUDA并行编程学习笔记
categories: [blog ]
tags: [GPU编程]
description: CUDA并行编程指南
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## 名词
* SIMD： 单指令多数据，是基于一个处理器核的，128位
* MMX：多媒体拓展
* AVX 高级适量拓展， 256位

## 计算机架构

### 冯诺依曼计算机架构
* 内存受限型
* QPI (quick path interconnect) 快速通道互联

### 连接机

采用4096个16核的CPU组装到一台机器上，也就是说64K个处理器来完成一个任务。连接机采用SIMD型并行处理，但是处理器之间的同步和通讯是很大的问题

### Cell处理器(众核)
用一个常规处理器作为监管处理器(PowerPC)，该处理器与大量高速流处理(SPE)相连。
* 每个流处理单元SPE调用执行一个程序
* 通过共享的网络，SPE之间和SPE与PowerPC之间进行相互通讯
* 
![国产申威 26010 处理器架构图](https://cwlseu.github.io/images/cuda/cell_arch.png)

### 多点计算
集群，当前最流行的莫过于Hadoop和spark了，一个是分布式文件系统，一个是分布式计算框架，这两个工具使得多点计算的方法充分发挥。

### GPU架构
![](https://cwlseu.github.io/images/cuda/2.png)

![](https://cwlseu.github.io/images/cuda/1.png)

## CUDA编程基础知识
学习CUDA C，可以在异构计算平台中实现高性能的应用。CUD的编译原则--基于虚拟指令集的运行时编译。

### 函数的类型

`__host__ float HostFunc()` 默认情况下，被host函数调用在CPU上执行

`__devide__ float DeviceFunc()` 被GPU设备执行调用

`__global__ void Kernelfunc()` 被host函数调用，在设备上执行

	Note：
	* __global__函数返回值必须为void
	* 在设备上执行的函数不能是递归，函数参数是固定的，不能再函数内部使用static变量

### 变量类型

`__shared__ A[4]`；//在share memory，块内线程共享。
设备上的函数，声明的变量都是存在register上的，存不下的放到local memory；
`cudaMalloc()`的空间是在设备的global memory上的。

### CUDA几个头文件

```cpp
#include<cuda_runtime.h>  // cuda程序运行必须的头文件

```

### CUDA routine
1. `cudaError_t err = cudaSuccess;`
   `cudaError_t`类型，表示错误类型。`cudaSuccess`表示成功。一般cuda routine的返回值都是`cudaError_t`类型，表示函数是否执行成功。  
    
2. `printf("%s\n", cudaGetErrorString(cudaGetLastError()));`
   输出错误时，使用以上函数转化为string。

3. `err = cudaMalloc((void **)&d_A, size);`
   动态内存申请函数，在设备的global memory上申请size个字节空间。
    
4. `err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);`or
    `err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);`
    //内存拷贝函数：从cpu上的内存h_A上拷贝size个字节数据到gpu上的内存d_A。反之，一样。

5. `int threadsPerBlock = 256;`
    `int blocksPerGrid =(nElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, nElements);`
    //前2句，表示Grid，block都是1维时，设置网格内的块数，每块内的线程数。
    //最后一句，启动kernel（运行在gpu端的函数）函数。
    //注意前2句可以改成。dim3 threadsPerBlock(256);这种形式。

6. `err = cudaGetLastError();`
//启动kernel函数时，并没有返回值，通过这个调用这个函数，查看kernel函数是否启动成功。

7. `err = cudaFree(d_A);`
//释放使用cudaMalloc申请的空间。
    
8. `err = cudaMemset(d_a, 0, size)`
//类似于memset函数。将d_A的size个字节置0.

```cpp
/**
 * CUDA device properties
 */
struct __device_builtin__ cudaDeviceProp
{
    char   name[256];                  /**< ASCII string identifying device */
    size_t totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int    regsPerBlock;               /**< 32-bit registers available per block */
    int    warpSize;                   /**< Warp size in threads */
    size_t memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int    maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int    maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int    maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int    clockRate;                  /**< Clock frequency in kilohertz */
    size_t totalConstMem;              /**< Constant memory available on device in bytes */
    int    major;                      /**< Major compute capability */
    int    minor;                      /**< Minor compute capability */
    size_t textureAlignment;           /**< Alignment requirement for textures */
    size_t texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int    deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int    multiProcessorCount;        /**< Number of multiprocessors on device */
    int    kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
    int    integrated;                 /**< Device is integrated as opposed to discrete */
    int    canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int    computeMode;                /**< Compute mode (See ::cudaComputeMode) */
    int    maxTexture1D;               /**< Maximum 1D texture size */
    int    maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int    maxTexture1DLinear;         /**< Maximum size for 1D textures bound to linear memory */
    int    maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int    maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int    maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int    maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int    maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int    maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int    maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int    maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int    maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int    maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int    maxSurface1D;               /**< Maximum 1D surface size */
    int    maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int    maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int    maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int    maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int    maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int    maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t surfaceAlignment;           /**< Alignment requirements for surfaces */
    int    concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int    ECCEnabled;                 /**< Device has ECC support enabled */
    int    pciBusID;                   /**< PCI bus ID of the device */
    int    pciDeviceID;                /**< PCI device ID of the device */
    int    pciDomainID;                /**< PCI domain ID of the device */
    int    tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int    asyncEngineCount;           /**< Number of asynchronous engines */
    int    unifiedAddressing;          /**< Device shares a unified address space with the host */
    int    memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
    int    memoryBusWidth;             /**< Global memory bus width in bits */
    int    l2CacheSize;                /**< Size of L2 cache in bytes */
    int    maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int    streamPrioritiesSupported;  /**< Device supports stream priorities */
    int    globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int    localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int    regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int    managedMemory;              /**< Device supports allocating managed memory on this system */
    int    isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int    multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
};
```