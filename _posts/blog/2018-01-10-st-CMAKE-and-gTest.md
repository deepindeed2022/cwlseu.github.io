---
layout: post
title: Linux开发--CMake & gtest
categories: [blog ]
tags: [linux开发]
description: 
---
[TOC] 

## 一个关于CMake的例子的解读

```python
PROJECT(sdk_common_samples)
cmake_minimum_required(VERSION 3.0)

# 查找已经安装的包
FIND_PACKAGE(OpenCV 2)

# SET 指令的语法是:
# SET(VAR [VALUE] [CACHE TYPE DOCSTRING [FORCE]])

SET(
	SDK_COMMON_INCLUDE_DIR
	${CMAKE_CURRENT_SOURCE_DIR}/../../include
	CACHE PATH
	"SDK_COMMON HEADER FILE PATH"
)

# MESSAGE 指令的语法是:
# MESSAGE([SEND_ERROR | STATUS | FATAL_ERROR] "message to display" ...)
# 这个指令用于向终端输出用户定义的信息,包含了三种类型:
# SEND_ERROR,产生错误,生成过程被跳过。
# SATUS ,输出前缀为 — 的信息。
# FATAL_ERROR,立即终止所有 cmake 过程.

MESSAGE("Find libs in ${SDK_COMMON_LIB_DIR}")

# INCLUDE_DIRECTORIES,其完整语法为:
# INCLUDE_DIRECTORIES([AFTER|BEFORE] [SYSTEM] dir1 dir2 ...)
# 这条指令可以用来向工程添加多个特定的头文件搜索路径,路径之间用空格分割,如果路径
# 中包含了空格,可以使用双引号将它括起来,默认的行为是追加到当前的头文件搜索路径的
# 后面,你可以通过两种方式来进行控制搜索路径添加的方式:
# 1,CMAKE_INCLUDE_DIRECTORIES_BEFORE,通过 SET 这个 cmake 变量为 on,可以
# 将添加的头文件搜索路径放在已有路径的前面。
# 2,通过 AFTER 或者 BEFORE 参数,也可以控制是追加还是置前。
INCLUDE_DIRECTORIES(
	${PROJECT_SOURCE_DIR}
	${SDK_COMMON_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
)

# 添加链接库的文件夹路径
LINK_DIRECTORIES(${SDK_COMMON_LIB_DIR})

# set最长用的方法，就像shell中export一个变量一样
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O2 -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2 -std=gnu++0x")

# 查找在相对路径下与*.cpp所匹配的模式的所有文件，保存到变量samples中
FILE(GLOB samples ${PROJECT_SOURCE_DIR}/*.cpp)

# 针对samples中的所有元素进行操作
FOREACH (sample ${samples})
	STRING(REGEX MATCH "[^/]+$" sample_file ${sample})
	STRING(REPLACE ".cpp" "" sample_basename ${sample_file})
	ADD_EXECUTABLE(test_${sample_basename} ${sample})
	# 添加执行时的需要链接的lib： common OpenCV_Libs
	TARGET_LINK_LIBRARIES(test_${sample_basename}
	sdk_common ${OpenCV_LIBS})
	# 另外，如果不是再window下的话需要添加线程库 -lpthread
	if (NOT WIN32)
		TARGET_LINK_LIBRARIES(test_${sample_basename} pthread)
	endif()
	
	INSTALL(TARGETS test_${sample_basename} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
ENDFOREACH() # foreach 结束

```

## google test的一些疑问

### TEST_F
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

### 看看gtest的工作流程

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

### Disable COPY和ASSIGN操作的方法， 将赋值函数和拷贝构造函数显示作为private下

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

## Reference
[1]gtest测试相关: http://blog.csdn.net/breaksoftware/article/details/50948239