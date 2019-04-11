---
layout: post
title: "《高质量的C++代码笔记》"
categories: [blog ]
tags: [C++, 开发]
description: 软件质量是被大多数程序员挂在嘴上而不是放在心上的东西！有多少软件开发人员对正确性、健壮性、可靠性、效率、易用性、可读性（可理解性）、可扩展性、可复用性、兼容性、可移植性等质量属性了如指掌？并且能在实践中运用自如？。
--- 

* content
{:toc}

## 引言

软件质量是被大多数程序员挂在嘴上而不是放在心上的东西！
除了完全外行和真正的编程高手外，初读本书，你最先的感受将是惊慌：“哇！我
以前捏造的 C++/C 程序怎么会有那么多的毛病？”有多少软件开发人员对正确性、健壮性、可靠性、效率、易用性、可读性（可理解性）、可扩展性、可复用性、兼容性、可移植性等质量属性了如指掌？并且能在实践中运用自如？。至少我现在还不是，我只能将平时遇到的一些值得记录的东西记录下来，以供下次翻阅。

## 从小程序看问题

`strcpy`的实现可以看出一个人的
* 编程风格
* 出错处理
* 算法复杂度分析

```cpp
char* strcpy(char* dest, const char* source)
{
    char * destcopy = dest;
    if((dest == NULL) || (source == NULL))
        throw "Invalid Arguments";
    while((*dest++=*source++)!= '\0');
    return destcopy;
}
```

## 文件结构
1. 声明在头文件.h，定义在源代码文件.cpp或者.c .cc
2. 为了防止头文件被重复引用，应当用 ifndef/define/endif 结构产生预
处理块。
3. 用 #include <filename.h> 格式来引用标准库的头文件（编译器将
从标准库目录开始搜索）。用 #include “filename.h” 格式来引用非标准库的头文件（编译器将从用户的工作目录开始搜索）。
4. 头文件中只存放“声明”而不存放“定义”
5. 不提倡使用全局变量， 尽量不要在头文件中出现象 extern int value 这
类声明。

```cpp
    /*
    * Copyright (c) 2001,上海贝尔有限公司网络应用事业部
    * All rights reserved.
    *
    * 文件名称： filename.h
    
    * 文件标识： 见配置管理计划书
    * 摘 要： 简要描述本文件的内容
    *
    * 当前版本： 1.1
    * 作 者： 输入作者（或修改者）名字
    * 完成日期： 2001年7月20日
    *
    * 取代版本： 1.0
    * 原作者 ： 输入原作者（或修改者）名字
    * 完成日期： 2001年5月10日
    */
```

### 为什么要声明和定义分离：

1. 通过头文件来调用库功能。在很多场合，源代码不便（或不准）向用户公布，只
要向用户提供头文件和二进制的库即可。用户只需要按照头文件中的接口声明来调用库
功能，而不必关心接口怎么实现的。编译器会从库中提取相应的代码。
2. 头文件能加强类型安全检查。如果某个接口被实现或被使用时，其方式与头文件
中的声明不一致，编译器就会指出错误，这一简单的规则能大大减轻程序员调试、改错
的负担。
3. 便于管理。如果代码文件比较多，可以将头文件放到include目录下，源文件放到source目录下，方便分别管理

## 程序

1. 在每个类声明之后、每个函数定义结束之后都要加空行
2. 在一个函数体内，逻揖上密切相关的语句之间不加空行，其它地方应
加空行分隔。
3. 一行代码只做一件事情，如只定义一个变量，或只写一条语句。
4. if、 for、 while、 do 等语句自占一行，执行语句不得紧跟其后。不论
执行语句有多少都要加{}。这样可以防止书写失误。
5. 尽可能在定义变量的同时初始化该变量。如果变量的引用处和其定义处相隔比较远，变量的初始化很容易被忘记。如果引用了未被初始化的变量，可能会导致程序错误。本建议可以减少隐患。

### 指针声明

修饰符 * 和 ＆ 应该靠近数据类型还是该靠近变量名，是个有争议的活题。
若将修饰符 * 靠近数据类型，例如： int* x; 从语义上讲此写法比较直观，即 x
是 int 类型的指针。
上述写法的弊端是容易引起误解，例如： int* x, y; 此处 y 容易被误解为指针变
量。虽然将 x 和 y 分行定义可以避免误解，但并不是人人都愿意这样做。

## 命名规则

unix系统中常常采用小写字母+_ 的方式
g_:全局变量
k_:static 变量
m_：class成员变量


## 类的构造函数、析构函数和赋值函数

每个类只有一个析构函数和一个赋值函数，但可以有多个构造函数（包含一个拷贝
构造函数，其它的称为普通构造函数）。对于任意一个类 A，如果不想编写上述函数，
C++编译器将自动为 A 产生四个缺省的函数，如
`A(void);`              // 缺省的无参数构造函数
`A(const A &a); `       // 缺省的拷贝构造函数
`~A(void);`             // 缺省的析构函数
`A & operate =(const A &a); `   // 缺省的赋值函数


## 经验

不少难以察觉的程序错误是由于变量没有被正确初始化或清除造成的，而初始化和清除工作很容易被人遗忘。

## 调试經典
```c
#define stub  fprintf(stderr, "error param in %s:%s:%d\n",  __FUNCTION__, __FILE__, __LINE__);
```

## mutable关键字用来解决常函数中不能修改对象的数据成员的问题

## 内存对齐
这是因为结构体内存分配有自己的对齐规则，结构体内存对齐默认的规则如下：
* 分配内存的顺序是按照声明的顺序。
* 每个变量相对于起始位置的偏移量必须是该变量类型大小的整数倍，不是整数倍空出内存，直到偏移量是整数倍为止。
* 最后整个结构体的大小必须是里面变量类型最大值的整数倍。

内存对齐<https://www.cnblogs.com/suntp/p/MemAlignment.html>

OpenCV中16b对齐的内存申请和释放
```cpp
#define CV_MALLOC_ALIGN 16
/*!
  Aligns pointer by the certain number of bytes
  This small inline function aligns the pointer by the certian number of bytes by shifting
  it forward by 0 or a positive offset.
*/
template <typename _Tp> 
static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

/*!
  Aligns buffer size by the certain number of bytes

  This small inline function aligns a buffer size by the certian number of bytes by enlarging it.
*/
static inline size_t alignSize(size_t sz, int n)
{
    assert((n & (n - 1)) == 0); // n is a power of 2
    return (sz + n-1) & -n;
}

void* fastMalloc( size_t size )
{
    uchar* udata = (uchar*)malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
    if(!udata)
        return OutOfMemoryError(size);
    uchar** adata = alignPtr((uchar**)udata + 1, CV_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void fastFree(void* ptr)
{
    if(ptr) {
        uchar* udata = ((uchar**)ptr)[-1];
        CV_DbgAssert(udata < (uchar*)ptr &&
               ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*)+CV_MALLOC_ALIGN));
        free(udata);
    }
}

```

[NCNN中使用该code进行内存对齐](https://www.jianshu.com/p/9c58dd414e5f)

## 内存的作用域，不要在函数中创建临时对象返回

```cpp
#include <cstdio>
#include <cstring>
///@brief 模型配置结构体
///@warning 该结构体在与did_model_set_config一起使用时，一定要先全部填充为0，再设置所需要的field。
/// set_model_config/get_model_config 是对该结构体的浅拷贝
typedef struct net_config_t {
	int  engine;

	///@brief each engine specific data can be passed to here, such as snpe_runtime, tensorrt_batch_size, ocl_context and so on.
	///@note each engine implementation will cast it to the corresponding runtime type, such as snpe_context_t, ppl_context_t.
	/// The lifetime of this pointer should span until create_handle finished, and the memory is managed by users.
	void* engine_context;
} net_config_t;

void set_net_config_t(net_config_t* config, int engine_type) {
    memset(config, 0, sizeof(net_config_t));
    int otherc[] = {1, 0, 5};
    config->engine = engine_type;
    config->engine_context = (void*)&otherc;
}

int main(int argc, char* argv[]) {
    net_config_t config;
    // 设置模型加载配置项
    set_net_config_t(&config, 3);
    fprintf(stderr, "config.engine %d\n", config.engine);
    int* context = (int*)config.engine_context;
    fprintf(stderr, "config.engine_context[0]=%d\n", context[0]);
    fprintf(stderr, "config.engine_context[1]=%d\n", context[1]);
    fprintf(stderr, "config.engine_context[2]=%d\n", context[2]);
    return 0;
}
```

> 第一次运行

  config.engine 3
  config.engine_context[0]=1286489600
  config.engine_context[1]=32624
  config.engine_context[2]=1288667592

> 第二次运行

  config.engine 3
  config.engine_context[0]=-204200448
  config.engine_context[1]=32695
  config.engine_context[2]=-202022456

从结果中可以看出engine_context中的内存是一块未初始化的内存空间。这是因为返回的局部数组被释放导致的结果。
这情况可能导致你的程序有不期望的执行结果。尤其是如果采用context[1]作为分支判断条件，本来应该为0或者false，
现在可能是正数，也可能是负数，为0的概率非常小。因此我们要避免这种返回局部变量的情况。

## Disable COPY和ASSIGN操作的方法， 将赋值函数和拷贝构造函数显示作为private下

> 方案 1

```cpp
// A macro to disallow copy constructor and operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type)\
  type(type const &);\
  void operator=(type const &)

class TestFactoryBase
{
private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestFactoryBase);
}

```

> 方案2

```cpp
class P {
public:
    P(const P &) = delete;
    P &operator =（const P &p) = delete;
};
```
以上两个delete声明禁止复制
能够通过明确的方式显式限定这些特殊方法有助于增强代码的可读性和可维护性


## ++i和i++的重载代码实现

```cpp
ClassName& operator++()
{
    ++cur;
    if(cur == last)
    {
      set_node(node + 1);
      cur = first;
    }
    return *this;
}

ClassName operator(int)
{
   ClassName tmp = *this;
   ++*this;
   return tmp;
}
``` 
## unsigned类型的默认转化造成的苦恼

u32Width是unsigned int类型的，在进行计算过程中如果`u32Width=2`，执行到`for (; j <= u32Width - 4; j += 4)`的时候，会出现问题：
由于j是size_t类型的， `u32Width-4`会被转化为`unsigned int`类型，从而造成该判断可通过，从直观上看来就发生了 `j <= -2`(实际是`j <= 4294967294`)是为`true`的事情了。


```cpp
	const unsigned int blob_len = u32Num * u32Chn * u32Height;
	for (size_t i = 0; i < blob_len; ++i) {
		size_t j = 0;
		for (; j <= u32Width - 4; j += 4) {
			dataDstBlob[j] = (dst_type)(ps32Ptr[j] * NNIE_DATA_SCALE_INV);
			dataDstBlob[j + 1] = (dst_type)(ps32Ptr[j + 1] * NNIE_DATA_SCALE_INV);
			dataDstBlob[j + 2] = (dst_type)(ps32Ptr[j + 2] * NNIE_DATA_SCALE_INV);
			dataDstBlob[j + 3] = (dst_type)(ps32Ptr[j + 3] * NNIE_DATA_SCALE_INV);
		}
		for (; j < u32Width; ++j) {
			dataDstBlob[j] = (dst_type)(ps32Ptr[j] * NNIE_DATA_SCALE_INV);
		}
		dataDstBlob += u32Width;
		ps32Ptr += blob->u32Stride / getElemSize(blob->enType);
	}
```


## C++中容易忽略的库bitset
bitset是处理*进制转换*，*基于bit的算法*中简单算法，虽然也可以使用raw的char array替代，但是很多bitset自带的方法，可以让程序飞起来。

```cpp
#include <iostream>
#include <bitset>
using namespace std;
void increment(std::bitset<5>& bset)
{
    for (int i = 0; i < 5; ++i)
    {
        if(bset[i] == 1)
            bset[i] = 0;
        else
        {
            bset[i] = 1;
            break;
        }
    }

}
void method_1()
{
    for (int i = 0; i < 32; ++i)
    {
        std::bitset<5> bset(i);
        std::cout << bset << std::endl;
    }
}
int main(int argc, char const *argv[])
{
    std::bitset<5> bset(0);
    for (int i = 0; i < 32; ++i)
    {
        std::cout << bset << std::endl;
        increment(bset);
    }
    
    return 0;
}
```

## 仿函数

仿函数(functor)，就是使一个类的使用看上去象一个函数。其实现就是类中实现一个
operator()，这个类就有了类似函数的行为，就是一个仿函数类了。C语言使用函数指针和回调函数来实现仿函数，例如一个用来排序的函数可以这样使用仿函数.在C++里，我们通过在一个类中重载括号运算符的方法使用一个函数对象而不是一个普通函数。

```cpp
template <typename T>
struct xxx
{
  returnType operator()(const T& x)
  {
    return returnType;
  }
}


template<typename T>  
class display  
{  
public:  
    void operator()(const T &x)  
    {  
        cout<<x<<" ";   
    }   
};   
```

```c
#include <stdio.h>  
#include <stdlib.h>  
//int sort_function( const void *a, const void *b);  
int sort_function( const void *a, const void *b)  
{     
    return *(int*)a-*(int*)b;  
}  
  
int main()  
{  
     
   int list[5] = { 54, 21, 11, 67, 22 };  
   qsort((void *)list, 5, sizeof(list[0]), sort_function);//起始地址，个数，元素大小，回调函数   
   int  x;  
   for (x = 0; x < 5; x++)  
          printf("%i\n", list[x]);  
                    
   return 0;  
}  
```

### 仿函数在STL中的定义

要使用STL内建的仿函数，必须包含<functional>头文件。而头文件中包含的仿函数分类包括

1. 算术类仿函数
  加：plus<T>
  减：minus<T>
  乘：multiplies<T>
  除：divides<T>
  模取：modulus<T>
  否定：negate<T>

```cpp
#include <iostream>  
#include <numeric>  
#include <vector>   
#include <functional>   
using namespace std;  
  
int main()  
{  
    int ia[]={1,2,3,4,5};  
    vector<int> iv(ia,ia+5);  
    cout<<accumulate(iv.begin(),iv.end(),1,multiplies<int>())<<endl;   
      
    cout<<multiplies<int>()(3,5)<<endl;  
      
    modulus<int>  modulusObj;  
    cout<<modulusObj(3,5)<<endl; // 3   
    return 0;   
}   
```
2. 关系运算类仿函数

  等于：equal_to<T>
  不等于：not_equal_to<T>
  大于：greater<T>
  大于等于：greater_equal<T>
  小于：less<T>
  小于等于：less_equal<T>

从大到小排序：

```cpp
#include <iostream>  
#include <algorithm>  
#include <vector>   
  
using namespace std;  
  
template <class T>   
class display  
{  
public:  
    void operator()(const T &x)  
    {  
        cout<<x<<" ";   
    }   
};  
  
int main()  
{  
    int ia[]={1,5,4,3,2};  
    vector<int> iv(ia,ia+5);  
    sort(iv.begin(),iv.end(),greater<int>());  
    for_each(iv.begin(),iv.end(),display<int>());   
    return 0;   
}   
```

3. 逻辑运算仿函数

  逻辑与：logical_and<T>
  逻辑或：logical_or<T>
  逻辑否：logical_no<T>

## google test的一些疑问：TEST_F与TEST的区别

TEST_F与TEST的区别是，TEST_F提供了一个初始化函数（SetUp）和一个清理函数(TearDown)，在TEST_F中使用的变量可以在初始化函数SetUp中初始化，在TearDown中销毁，并且所有的TEST_F是互相独立的，都是在初始化以后的状态开始运行，一个TEST_F不会影响另一个TEST_F所使用的数据。

```cpp
//A.h
#ifndef A_H
#define A_H
class A
{
private:
　　int _a;
public:
　　A( int a );
　　~A( );
　　void add( int a );
　　int getA( );
};
#endif
A.cpp
#include "A.h"
A::A( int a ){
　　this->_a = a;
}
A::~A( ){
}
void A::add( int a ){
　　this->_a += a;
}
int A::getA( ){
　　return this->_a;
}
```

- A_test.cpp

```cpp
// 　A_test.cpp
#include "A.h"
#include <gtest/gtest.h>
class A_test : public testing::Test {
protected:
　　A* _p_a;
　　virtual void SetUp( ){　　　//初始化函数
　　　　this->_p_a = new A( 1 );
　　}
　　virtual void TearDown( ){　 //清理函数
　　　　delete this->_p_a;
　　}
};
//第一个测试，参数A_test是上面的那个类，第二个参数FirstAdd是测试名称
TEST_F( A_test,FirstAdd ){　　　　
　　EXPECT_EQ( 1,_p_a->getA( ) );
　　_p_a->add( 3 );
　　EXPECT_EQ( 4,_p_a->getA( ) );
}

//第二个测试
TEST_F( A_test,SecondAdd ){
　　EXPECT_EQ( 1,_p_a->getA( ) );
　　_p_a->add( 5 );
　　EXPECT_EQ( 6,_p_a->getA( ) );
}

/*
上面的两个测试都是在SetUp函数执行后的状态下执行，也就是说在执行任意一个TEST_F时 _p_a->_a 的值都是初始值1
*/
```
- main.cpp

```cpp
#include <gtest/gtest.h>

int main(int argc, char * argv[])
{
　　testing::InitGoogleTest(&argc, argv);
　　return RUN_ALL_TESTS();
}
```


```cpp
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#define __TIC__()                                    \
	struct timeval __timing_start, __timing_end; \
	gettimeofday(&__timing_start, NULL);

#define __TOC__()                                                        \
	do {                                                             \
		gettimeofday(&__timing_end, NULL);                       \
		double __timing_gap = (__timing_end.tv_sec -     \
					       __timing_start.tv_sec) *  \
					      1000.0 +                     \
				      (__timing_end.tv_usec -    \
					       __timing_start.tv_usec) / \
					      1000.0;                    \
		fprintf(stdout, "TIME(ms): %lf\n", __timing_gap);        \
	} while (0)

```

## 看看gtest的工作流程

- 入口

```cpp
//第一个测试，参数A_test是上面的那个类，第二个参数FirstAdd是测试名称
TEST(A_test, FirstAdd){　　　　
　　EXPECT_EQ( 1,_p_a->getA( ) );
　　_p_a->add( 3 );
　　EXPECT_EQ( 4,_p_a->getA( ) );
}

// Define this macro to 1 to omit the definition of TEST(), which
// is a generic name and clashes with some other libraries.
#if !GTEST_DONT_DEFINE_TEST
# define TEST(test_case_name, test_name) GTEST_TEST(test_case_name, test_name)
#endif

#define GTEST_TEST(test_case_name, test_name)\
  GTEST_TEST_(test_case_name, test_name, \
              ::testing::Test, ::testing::internal::GetTestTypeId())
```

- 首先看看函数中调用的一个宏的实现

```cpp
// Expands to the name of the class that implements the given test.
#define GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
  test_case_name##_##test_name##_Test

// Helper macro for defining tests.
// 这个宏声明了一个继承自parent_class ::testing::Test的类，然后对这个类的属性test_info_进行赋值

#define GTEST_TEST_(test_case_name, test_name, parent_class, parent_id)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
 private:\
  virtual void TestBody();\
  static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
};\
/*这个宏声明了一个继承自parent_class ::testing::Test的类，然后对这个类的属性test_info_进行赋值*/\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ =\
    ::testing::internal::MakeAndRegisterTestInfo(\
        #test_case_name, #test_name, NULL, NULL, \
        (parent_id), \
        parent_class::SetUpTestCase, \
        parent_class::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl<\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
// 实现我们的这个TestBody\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()
```
- 看一下MakeAndRegisterTestInfo函数

```cpp
TestInfo* MakeAndRegisterTestInfo(
    const char* test_case_name,
    const char* name,
    const char* type_param,
    const char* value_param,
    TypeId fixture_class_id,
    SetUpTestCaseFunc set_up_tc,
    TearDownTestCaseFunc tear_down_tc,
    TestFactoryBase* factory) {
  TestInfo* const test_info =
      new TestInfo(test_case_name, name, type_param, value_param,
                   fixture_class_id, factory);
  // 添加测试用例信息到UnitTestImpl的testcase_中
  GetUnitTestImpl()->AddTestInfo(set_up_tc, tear_down_tc, test_info);
  return test_info;
}
```
- AddTestInfo试图通过测试用例名等信息获取测试用例，然后调用测试用例对象去新增一个测试特例——test_info。
这样我们在此就将测试用例和测试特例的关系在代码中找到了关联。

```cpp
// Finds and returns a TestCase with the given name.  If one doesn't
// exist, creates one and returns it.  It's the CALLER'S
// RESPONSIBILITY to ensure that this function is only called WHEN THE
// TESTS ARE NOT SHUFFLED.
//
// Arguments:
//
//   test_case_name: name of the test case
//   type_param:     the name of the test case's type parameter, or NULL if
//                   this is not a typed or a type-parameterized test case.
//   set_up_tc:      pointer to the function that sets up the test case
//   tear_down_tc:   pointer to the function that tears down the test case
TestCase* UnitTestImpl::GetTestCase(const char* test_case_name,
                                    const char* type_param,
                                    Test::SetUpTestCaseFunc set_up_tc,
                                    Test::TearDownTestCaseFunc tear_down_tc) {
  // Can we find a TestCase with the given name?
  const std::vector<TestCase*>::const_iterator test_case =
      std::find_if(test_cases_.begin(), test_cases_.end(),
                   TestCaseNameIs(test_case_name));

  if (test_case != test_cases_.end())
    return *test_case;

  // No.  Let's create one.
  TestCase* const new_test_case =
      new TestCase(test_case_name, type_param, set_up_tc, tear_down_tc);

  // Is this a death test case?
  if (internal::UnitTestOptions::MatchesFilter(test_case_name,
                                               kDeathTestCaseFilter)) {
    // Yes.  Inserts the test case after the last death test case
    // defined so far.  This only works when the test cases haven't
    // been shuffled.  Otherwise we may end up running a death test
    // after a non-death test.
    ++last_death_test_case_;
    test_cases_.insert(test_cases_.begin() + last_death_test_case_,
                       new_test_case);
  } else {
    // No.  Appends to the end of the list.
    test_cases_.push_back(new_test_case);
  }

  test_case_indices_.push_back(static_cast<int>(test_case_indices_.size()));
  return new_test_case;
}
```


# Reference
- [1]. gtest测试相关: http://blog.csdn.net/breaksoftware/article/details/50948239
- [2]. [Floating-Point Arithmetic 浮点数结构](https://wenku.baidu.com/view/11d8f46527d3240c8447efa8.html)
