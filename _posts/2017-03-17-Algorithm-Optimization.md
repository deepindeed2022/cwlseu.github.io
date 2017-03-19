---
layout: post
title: "算法优化的一些技巧"
categories: [blog ]
tags: [优化, ]
description: 向量化和编译器优化
---

对于这方面的姿势，也是属于意外。在使用Caffe的过程中，需要依赖一个关于矩阵计算的库，可是
使用atlas或者openblas, 当然有资金支持的话可以使用更快地MKL, 而一个穷小白就只能从开源免费的计算库中选了，就选了OpenBlas。 后来因为缘分，认识了OpenBlas的创始人，从他们公司的工作中了解到还有机器学习算法优化加速的这么个工作。其中涉及到如OpenMP, SIMD, 当然编译器优化也是不容忽视的。在此，总结一下工作中用到的一些函数，免得下次见到不认识了。

## Intrinsic function[^1]
我对这个的理解就是在汇编的基础上进行向量化的封装的一些宏或者函数, 同时可以操作很多个数据，如使用SSE可以操作128位的数据，可以使4个int类型，也可以使用8个short类型也可以是16个char类型的数据。

从intrinsic guide[^2]中就可以看出Intel关于SIMD方面的发展历程。MMX(主要是16位指令)到后面的SSE2 SSE4.2(32位指令)等等。 查阅文档的时候后可以按照计算的类别：

### 计算类型
计算类型根据操作数据的类型分别封装了加减乘除,文档中对接口函数说明得很是清楚，还包括生成目标指令是什么。如：

     Synopsis
        __m128i _mm_add_epi16 (__m128i a, __m128i b)
        #include "emmintrin.h"
        Instruction: paddw xmm, xmm
        CPUID Flags: SSE2
    Description
        Add packed 16-bit integers in a and b, and store the results in dst.
    Operation
        FOR j := 0 to 7
            i := j*16
            dst[i+15:i] := a[i+15:i] + b[i+15:i]
        ENDFOR

从外，函数命名很有规律 `_mm_操作_操作的数据类型`， 数据类型`__m128i` 表示integer类型的向量数组，`__m128`表示当精度类型的向量数组,`__128d`表示双精度类型的向量数组。

```cpp
__m128i _mm_add_epi16 (__m128i a, __m128i b); //Add packed 16-bit integers in a and b
__m128d _mm_div_pd (__m128d a, __m128d b); // Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b
__m128d _mm_mul_sd (__m128d a, __m128d b); // Multiply the lower double-precision (64-bit) floating-point element in a and b, store the result in the lower element of dst, and copy the upper element from a to the upper element of dst.
__m128i _mm_subs_epi16 (__m128i a, __m128i b)
__m128i _mm_subs_epu16 (__m128i a, __m128i b)
```

## 设置\初始化向量数组
```cpp
__m128i _mm_set_epi16 (short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
__m128i _mm_set_epi32 (int e3, int e2, int e1, int e0);

__m128i _mm_set_epi8 (char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0);

__m128d _mm_set_pd (double e1, double e0);

__m128d _mm_set_pd1 (double a); // Broadcast double-precision (64-bit) floating-point value a to all elements of dst.

__m128 _mm_set_ps (float e3, float e2, float e1, float e0);
__m128 _mm_set_ps1 (float a);
__m128d _mm_set_sd (double a);
__m128 _mm_set_ss (float a);
__m128i _mm_set1_epi16 (short a);

__m128i _mm_set1_epi32 (int a);

__m128i _mm_set1_epi64 (__m64 a);

__m128i _mm_set1_epi64x (__int64 a);
__m128i _mm_set1_epi8 (char a);
__m128d _mm_set1_pd (double a);
__m128 _mm_set1_ps (float a);
__m128i _mm_setr_epi16 (short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
__m128i _mm_setr_epi32 (int e3, int e2, int e1, int e0);
__m128i _mm_setr_epi64 (__m64 e1, __m64 e0);
__m128i _mm_setr_epi8 (char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0);
__m128d _mm_setr_pd (double e1, double e0);
__m128 _mm_setr_ps (float e3, float e2, float e1, float e0);
__m128d _mm_setzero_pd (void);
__m128 _mm_setzero_ps (void);
__m128i _mm_setzero_si128 ();
```

## 从内存中加载数据
从内存中加载数据，根据数据的类型(整数，单精度浮点数，双精度浮点数，向量数组等类型)，数据存储地址是否对齐等属性有不同的函数接口封装。常常数据不对齐(SSE2函数名称中常常带一个u表示不要求地址对齐， SSE函数中常用1表示不要求地址对齐)的接口要比对齐的接口效率低很多。地址对齐常常是以16bit对齐。

```cpp
/*Load 128-bits of integer data from unaligned memory into dst. This intrinsic may perform better than _mm_loadu_si128 when the data crosses a cache line boundary.*/
__m128i _mm_lddqu_si128 (__m128i const* mem_addr)
__m128 _mm_load_ps (float const* mem_addr)
__m128 _mm_load_ps1 (float const* mem_addr)

__m128i _mm_load_si128 (__m128i const* mem_addr)

/*Load 128-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.*/
__m128i _mm_loadu_si128 (__m128i const* mem_addr)

/*Load 64-bit integer from memory into the first element of dst.*/
__m128i _mm_loadl_epi64 (__m128i const* mem_addr)

__m128d _mm_loadl_pd (__m128d a, double const* mem_addr)

__m128 _mm_loadl_pi (__m128 a, __m64 const* mem_addr)

/*
 Load 2 double-precision (64-bit) floating-point elements from memory into dst in reverse order. mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated. 
*/
__m128d _mm_loadr_pd (double const* mem_addr)


/*Load 128-bits (composed of 2 packed double-precision (64-bit) floating-point elements) from memory into dst. mem_addr does not need to be aligned on any particular boundary.*/
__m128d _mm_loadu_pd (double const* mem_addr)

__m128 _mm_loadu_ps (float const* mem_addr)

```

## 图像中进行亮点查找的关键函数
`int _mm_movemask_epi8 (__m128i a)`
Create mask from the most significant bit of each 8-bit element in a, and store the result in dst.
创建从签名的最高有效位的 16 位掩码 16 或在 a 和零的无符号 8 位整数扩展上面的位。


## 编译器buidin函数

1. `void __builtin___clear_cache (char *begin, char *end)` 
This function is used to flush the processor’s instruction cache for the region of memory between begin inclusive and end exclusive. Some targets require that the instruction cache be flushed, after modifying memory containing code, in order to obtain deterministic behavior.
有的时候为了获取确定性的性能测试结果，使用该函数对处理器的指令和数据进行清空操作。

If the target does not require instruction cache flushes, `__builtin___clear_cache` has no effect. Otherwise either instructions are emitted in-line to clear the instruction cache or a call to the `__clear_cache function `in libgcc is made.
如何设置begin和end的信息，请参见[^5]

2. `void __builtin_prefetch (const void *addr, ...)`
This function is used to minimize cache-miss latency by moving data into a cache before it is accessed. You can insert calls to` __builtin_prefetch` into code for which you know addresses of data in memory that is likely to be accessed soon. If the target supports them, data prefetch instructions are generated. If the prefetch is done early enough before the access then the data will be in the cache by the time it is accessed.

The value of addr is the address of the memory to prefetch. There are two optional arguments, rw and locality. The value of rw is a compile-time constant one or zero; one means that the prefetch is preparing for a write to the memory address and zero, the default, means that the prefetch is preparing for a read. The value locality must be a compile-time constant integer between zero and three. A value of zero means that the data has no temporal locality, so it need not be left in the cache after the access. A value of three means that the data has a high degree of temporal locality and should be left in all levels of cache possible. Values of one and two mean, respectively, a low or moderate degree of temporal locality. The default is three.
`__builtin_prefetch (const void *addr, ...)`它通过对数据手工预取的方法，在使用地址addr的值之前就将其放到cache中，减少了读取延迟，从而提高了性能，但该函数也需要 CPU 的支持。该函数可接受三个参数，第一个参数addr是要预取的数据的地址，第二个参数可设置为0或1（1表示我对地址addr要进行写操作，0表示要进行读操作），第三个参数可取0-3（0表示不用关心时间局部性，取完addr的值之后便不用留在cache中，而1、2、3表示时间局部性逐渐增强）

```cpp
for (i = 0; i < n; i++)
  {
    a[i] = a[i] + b[i];
    __builtin_prefetch (&a[i+j], 1, 1);
    __builtin_prefetch (&b[i+j], 0, 1);
    /* … */
  }
```

Data prefetch does not generate faults if addr is invalid, but the address expression itself must be valid. For example, a prefetch of p->next does not fault if p->next is not a valid address, but evaluation faults if p is not a valid address.

If the target does not support data prefetch, the address expression is evaluated if it includes side effects but no other code is generated and GCC does not issue a warning.


3. ` int __builtin_clz (unsigned int x)`
Returns the number of leading 0-bits in x, starting at the most significant bit position. If x is 0, the result is undefined.
返回从左边起第一个1之前的0个个数

4. `int __builtin_ctz (unsigned int x)`
Returns the number of trailing 0-bits in x, starting at the least significant bit position. If x is 0, the result is undefined.
返回从右边其第一个1之后的0个个数

5. `int __builtin_clz (unsigned int x)`

Returns the number of leading 0-bits in x, starting at the most significant bit position. If x is 0, the result is undefined.
返回左起第一个‘1’之前0的个数。

6. `int __builtin_ffs (unsigned int x)`

Returns one plus the index of the least significant 1-bit of x, or if x is zero, returns zero.
返回右起第一个‘1’的位置。

7. `int __builtin_popcount (unsigned int x)`
Returns the number of 1-bits in x.
返回‘1’的个数。


8. `int __builtin_parity (unsigned int x)`
Returns the parity of x, i.e. the number of 1-bits in x modulo 2.
返回‘1’的个数的奇偶性。

### 例子
```cpp
#include <stdio.h>
int main(int argc, char const *argv[])
{
    for(auto i = 0; i < 10; ++i)
    {
#if defined(__GNUC__) || defined(__GNUG__)
    //printf("the number of trailing 0-bits in %d is %d \n", i,  __builtin_ctz(i));
    //printf("%d have %d 1-bits\n", i, __builtin_popcount(i));
    printf("%d parity value: %d\n", i, __builtin_parity(i));
    printf("%d swap32 %d\n", i, __builtin_bswap32(i));
#endif
    }
#if defined(__GNUC__) || defined(__GNUG__)
    printf("test __builtin___clear_cache\n");
    char* a;
    for(int i = 0; i < 10; ++i)
    {
        __builtin___clear_cache(a, a + 4096);
        a = new char[4096];
        delete[] a;
    }
#endif
    return 0;
}


```

### Result:

    0 have 0 1-bits
    0 parity value: 0
    0 swap32 0
    the number of trailing 0-bits in 1 is 0 
    1 have 1 1-bits
    1 parity value: 1
    1 swap32 16777216
    the number of trailing 0-bits in 2 is 1 
    2 have 1 1-bits
    2 parity value: 1
    2 swap32 33554432
    the number of trailing 0-bits in 3 is 0 
    3 have 2 1-bits
    3 parity value: 0
    3 swap32 50331648
    the number of trailing 0-bits in 4 is 2 
    4 have 1 1-bits
    4 parity value: 1
    4 swap32 67108864
    the number of trailing 0-bits in 5 is 0 
    5 have 2 1-bits
    5 parity value: 0
    5 swap32 83886080
    the number of trailing 0-bits in 6 is 1 
    6 have 2 1-bits
    6 parity value: 0
    6 swap32 100663296
    the number of trailing 0-bits in 7 is 0 
    7 have 3 1-bits
    7 parity value: 1
    7 swap32 117440512
    the number of trailing 0-bits in 8 is 3 
    8 have 1 1-bits
    8 parity value: 1
    8 swap32 134217728
    the number of trailing 0-bits in 9 is 0 
    9 have 2 1-bits
    9 parity value: 0
    9 swap32 150994944
    test __builtin___clear_cache

## Example:寻找数组中第一个非0元素的位置的intrinsic 函数

```cpp
int findStartContourPoint(const uchar *src_data,int width, int j) 
{
#if  PERFCV_SSE_4_2
        __m128i v_zero = _mm_setzero_si128();
        int v_size = width - 32;

        for (; j <= v_size; j += 32) {
            __m128i v_p1 = _mm_loadu_si128((const __m128i*)(src_data + j));
            __m128i v_p2 = _mm_loadu_si128((const __m128i*)(src_data + j + 16));

            __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_zero);
            __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_zero);

            unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
            unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

            mask1 ^= 0x0000ffff;
            mask2 ^= 0x0000ffff;

            if (mask1) {
                j += trailingZeros(mask1);
                return j;
            }
            if (mask2) {
                j += trailingZeros(mask2 << 16);
                return j;
            }
        }

        if (j <= width - 16) {
            __m128i v_p = _mm_loadu_si128((const __m128i*)(src_data + j));

            unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_zero)) ^ 0x0000ffff;

            if (mask) {
                j += trailingZeros(mask);
                return j;
            }
            j += 16;
        }
#endif
    for (; j < width && !src_data[j]; ++j)
        ;
    return j;
}
```

其中 trailingZeros就是调用编译器的内置函数实现的。

```cpp
#if PERFCV_SSE_4_2
inline unsigned int trailingZeros(unsigned int value) {
    assert(value != 0); // undefined for zero input (https://en.wikipedia.org/wiki/Find_first_set)
if defined(__GNUC__) || defined(__GNUG__)
    return __builtin_ctz(value);
#elif defined(__ICC) || defined(__INTEL_COMPILER)
    return _bit_scan_forward(value);
#elif defined(__clang__)
    return llvm.cttz.i32(value, true);
#else
    static const int MultiplyDeBruijnBitPosition[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9 };
    return MultiplyDeBruijnBitPosition[((uint32_t)((value & -value) * 0x077CB531U)) >> 27];
#endif
}
#endif

```

## Reference

[^1Wikipedia名词解释]:https://en.wikipedia.org/wiki/Intrinsic_function
[^2Intel官网文档]:https://software.intel.com/sites/landingpage/IntrinsicsGuide
[^3微软提供的文档]:https://msdn.microsoft.com/zh-cn/library/0d4dtzhb(v=vs.110).aspx
[^4 GCC buildin]:https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
[^5 How does __builtin___clear_cache work?]:http://stackoverflow.com/questions/35741814/how-does-builtin-clear-cache-work



