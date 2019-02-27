---
layout: post
title: "《高质量的C++代码笔记》"
categories: [blog ]
tags: [C++, 开发]
description: 软件质量是被大多数程序员挂在嘴上而不是放在心上的东西！除了完全外行和真正的编程高手外，初读本书，你最先的感受将是惊慌：“哇！我以前捏造的 C++/C 程序怎么会有那么多的毛病？”有多少软件开发人员对正确性、健壮性、可靠性、效率、易用性、可读性（可理解性）、可扩展性、可复用性、兼容性、可移植性等质量属性了如指掌？并且能在实践中运用自如？。
--- 

* content
{:toc}

## 引言

软件质量是被大多数程序员挂在嘴上而不是放在心上的东西！
除了完全外行和真正的编程高手外，初读本书，你最先的感受将是惊慌：“哇！我
以前捏造的 C++/C 程序怎么会有那么多的毛病？”有多少软件开发人员对正确性、健壮性、可靠性、效率、易用性、可读性（可理解性）、可扩展性、可复用性、兼容性、可移植性等质量属性了如指掌？并且能在实践中运用自如？。

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
[1]. gtest测试相关: http://blog.csdn.net/breaksoftware/article/details/50948239