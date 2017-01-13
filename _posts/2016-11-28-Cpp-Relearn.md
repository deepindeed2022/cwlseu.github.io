---
layout: post
title: 在源码分析中的C++鲜为人知的故事
categories: [blog ]
tags: [C++, ]
description:如果不是使用C++开发过大型系统项目，可能一些编程语言的feature将继续雪藏，让我们一起来挖掘这些秘密吧。
---
# C++ Programming Tricks

## C++/C 宏定义（define）中\# \#\# 的含义
define 中的# ## 一般是用来拼接字符串的，但是实际使用过程中，有哪些细微的差别呢，我们通过几个例子来看看。

\#是字符串化的意思，出现在宏定义中的#是把跟在后面的参数转成一个字符串；

```cpp
// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

\#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
  g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
  return g_brew_map[name];
  } else {
  LOG(ERROR) << "Available caffe actions:";
  for (BrewMap::iterator it = g_brew_map.begin();
  it != g_brew_map.end(); ++it) {
  LOG(ERROR) << "\t" << it->first;
  }
  LOG(FATAL) << "Unknown action: " << name;
  return NULL; // not reachable, just to suppress old compiler warnings.
  }
}
```
上面这段代码是Caffe源码tools/caffe.cpp中的一段程序，主要完成了caffe不同工作阶段的注册工作。如caffe可以在 `train, test` 等不同环境下工作。每个环境对应着响应的处理函数。这些函数是如何通过main函数统一管理的。就是通过这个`GetBrewFunction`函数统一调用的。那么这个函数如何获取具体的调用函数，就得知道函数指针和宏替换的相关知识了。具体参考[caffe.cpp](https://github.com/BVLC/caffe/blob/master/tools/caffe.cpp)


## GNU C中不为人知的特色：`__attribute__`机制

偶然碰到了`__attribute__`，虽然之前在看Linux内核代码时见过很多次，但还是对它熟视无睹，罪过啊，下面的文章是从源码网上转载的，原文在这里:http://www.yuanma.org/data/2006/0625/article_948.htm，此处只是做简单阐述，共同进步。

1. GNU C的一大特色（却不被初学者所知）就是`__attribute__ `机制。`__attribute__ `可以设置函数属性（Function Attribute）、变量属性（Variable Attribute）和类型属性（Type Attribute）。它的书写特征是：`__attribute__ `前后都有两个下划线，并切后面会紧跟一对原括弧，括弧里面是相应的`__attribute__ `参数，语法格式如下： 
`__attribute__ ((attribute-list))`

2. 另外，它必须放于声明的尾部“；”之前。

函数属性可以帮助开发者把一些特性添加到函数声明中，从而可以使编译器在错误检查方面的功能更强大。`__attribute__`机制也很容易同非GNU应用程序做到兼容之功效。

**GNU CC需要使用 –Wall编译器来击活该功能**，这是控制警告信息的一个很好的方式。下面介绍几个常见的属性参数。

`__attribute__ format`。该`__attribute__`属性可以给被声明的函数加上类似`printf`或者`scanf`的特征，它可以使编译器检查函数声明和函数实际调用参数之间的格式化字符串是否匹配。该功能十分有用，尤其是处理一些很难发现的bug。`format`的语法格式为：

`format (archetype, string-index, first-to-check)`

format属性告诉编译器，按照printf, scanf, strftime或strfmon的参数表格式规则对该函数的参数进行检查。“archetype”指定是哪种风格；“string-index”指定传入函数的第几个参数是格式化字符串；“first-to-check”指定从函数的第几个参数开始按上述规则进行检查。

3. 具体使用格式如下：
`__attribute__((format(printf,m,n)))`
`__attribute__((format(scanf,m,n)))`

其中参数m与n的含义为：
* m：第几个参数为格式化字符串（format string）；
* n：参数集合中的第一个，即参数“…”里的第一个参数在函数参数总数排在第几，注意，有时函数参数里还有“隐身”的呢，后面会提到；

在使用上，`__attribute__((format(printf,m,n)))`是常用的，而另一种却很少见到。下面举例说明，其中myprint为自己定义的一个带有可变参数的函数，其功能类似于printf：

```cpp
//m=1；n=2
extern void myprint(const char *format,...) __attribute__((format(printf,1,2)));

//m=2；n=3
extern void myprint(int l，const char *format,...) __attribute__((format(printf,2,3)));
```
需要特别注意的是，如果myprint是一个函数的成员函数，那么m和n的值可有点“悬乎”了，例如：

```cpp
//m=3；n=4
extern void myprint(int l，const char *format,...) __attribute__((format(printf,3,4)));
```
其原因是，类成员函数的第一个参数实际上一个“隐身”的“this”指针。（有点C++基础的都知道点this指针，不知道你在这里还知道吗？）

这里给出测试用例：attribute.c，代码如下：

```cpp
extern myprint(const *format,...) attribute__((format(printf,1,2)));
void test()
{
  myprint("i=%d\n", 1);
  myprint("i=%s\n", 2);
  myprint("i=%s\n","abc");
  myprint("%s,%d,%d\n",1,2);
}
extern void myprint(const char *format,...) __attribute__((format(printf,1,2)));
void test()
{
  myprint("i=%d\n",6);
  myprint("i=%s\n",6);
  myprint("i=%s\n","abc");
  myprint("%s,%d,%d\n",1,2);
}
```
gcc编译后会提示`format argument is not a pointer`的警告。若去掉`__attribute__((format(printf,1,2)))`，则会正常编译。需要注意的是，编译器只能识别类似printf的标准输出库函数。

还有一个`__attribute__ noreturn`，该属性通知编译器函数从不返回值，当遇到类似函数需要返回值而却不可能运行到返回值处就已经退出来的情况，该属性可以避免出现错误信息。C库函数中的`abort()`和`exit()`的声明格式就采用了这种格式，如下所示：

```cpp
extern void exit(int) __attribute__((noreturn));
extern void abort(void) __attribute__((noreturn));
```
为了方便理解，大家可以参考如下的例子：

```cpp
//name: noreturn.c ；测试__attribute__((noreturn))
  extern void myexit();
  int test(int n)
  {
    if ( n > 0 )
    {
      myexit();
      /* 程序不可能到达这里*/
    }
    else
      return 0;
  }
```

//name: noreturn.c ；测试__attribute__((noreturn))

```cpp
extern void myexit();
int test(int n)
{
  if ( n > 0 )
  {
    myexit();
    /* 程序不可能到达这里*/
  }
  else
    return 0;
}
```
编译后的输出结果如下：

`$gcc –Wall –c noreturn.c`

noreturn.c: In function `test':

noreturn.c:12: warning: control reaches end of non-void function

很显然，这是因为一个被定义为有返回值的函数却没有返回值。加上\__attribute\__((noreturn))则可以解决此问题的出现。

后面还有`__attribute__const`、`-finstrument-functions`、`no_instrument_function`等的属性描述，就不多转了，感兴趣的可以看原文。

### 变量属性(Variable Attribute)

关键字`__attribute__ `也可以对变量或结构体成员进行属性设置。这里给出几个常用的参数的解释，更多的参数可参考原文给出的连接。

在使用`__attribute__ `参数时，你也可以在参数的前后都加上“\__”（两个下划线），例如，使用`__attribute__ `而不是aligned，这样，你就可以在相应的头文件里使用它而不用关心头文件里是否有重名的宏定义。

#### aligned (alignment)

该属性规定变量或结构体成员的最小的对齐格式，以字节为单位。例如：

`int x __attribute__ ((aligned (16))) = 0;`

编译器将以16字节（注意是字节byte不是位bit）对齐的方式分配一个变量。也可以对结构体成员变量设置该属性，例如，创建一个双字对齐的int对，可以这么写：

`struct foo { int x[2] __attribute__ ((aligned (8))); };`

如上所述，你可以手动指定对齐的格式，同样，你也可以使用默认的对齐方式。如果aligned后面不紧跟一个指定的数字值，那么编译器将依据你的目标机器情况使用最大最有益的对齐方式。例如：

`short array[3] __attribute__ ((aligned));`

1. 选择针对目标机器最大的对齐方式，可以提高拷贝操作的效率。aligned属性使被设置的对象占用更多的空间，相反的，使用packed可以减小对象占用的空间。

2. 需要注意的是，attribute属性的效力与你的连接器也有关，如果你的连接器最大只支持16字节对齐，那么你此时定义32字节对齐也是无济于事的。

3. 使用该属性可以使得变量或者结构体成员使用最小的对齐方式，即对变量是一字节对齐，对域（field）是位对齐。

下面的例子中，x成员变量使用了该属性，则其值将紧放置在a的后面：

```cpp
struct test
{
    char a;
    int x[2] __attribute__ ((packed));
};
```
其它可选的属性值还可以是：`cleanup，common，nocommon，deprecated，mode，section，shared，tls_model，transparent_union，unused，vector_size，weak，dllimport，dlexport`等。

### 类型属性（Type Attribute）

关键字`__attribute__`也可以对结构体（struct）或共用体（union）进行属性设置。大致有六个参数值可以被设定，即：`aligned, packed, transparent_union, unused, deprecated `和 `may_alias`。

在使用`__attribute__`参数时，你也可以在参数的前后都加上“\__”（两个下划线），例如，使用`__aligned__`而不是`aligned`，这样，你就可以在相应的头文件里使用它而不用关心头文件里是否有重名的宏定义。

#### aligned (alignment)

该属性设定一个指定大小的对齐格式（以字节为单位），例如：

`struct S { short f[3]; } __attribute__ ((aligned (8)));`

`typedef int more_aligned_int __attribute__ ((aligned (8)));`

    该声明将强制编译器确保（尽它所能）变量类型为struct S或者more-aligned-int的变量在分配空间时采用8字节对齐方式。

如上所述，你可以手动指定对齐的格式，同样，你也可以使用默认的对齐方式。如果aligned后面不紧跟一个指定的数字值，那么编译器将依据你的目标机器情况使用最大最有益的对齐方式。例如：

`struct S { short f[3]; } __attribute__ ((aligned));`

这里，如果sizeof（short）的大小为2（byte），那么，S的大小就为6。取一个2的次方值，使得该值大于等于6，则该值为8，所以编译器将设置S类型的对齐方式为8字节。

1. aligned属性使被设置的对象占用更多的空间，相反的，使用packed可以减小对象占用的空间。

2. 需要注意的是，attribute属性的效力与你的连接器也有关，如果你的连接器最大只支持16字节对齐，那么你此时定义32字节对齐也是无济于事的。

3. 使用该属性对struct或者union类型进行定义，设定其类型的每一个变量的内存约束。当用在enum类型定义时，暗示了应该使用最小完整的类型（it indicates that the smallest integral type should be used）。

下面的例子中，my-packed-struct类型的变量数组中的值将会紧紧的靠在一起，但内部的成员变量s不会被“pack”，如果希望内部的成员变量也被packed的话，my-unpacked-struct也需要使用packed进行相应的约束。

```cpp
struct my_unpacked_struct{
char c;
  int i;
};

struct my_packed_struct{
  char c;
  int i;
  struct my_unpacked_struct s;
}__attribute__ ((__packed__));
```

### 变量属性与类型属性举例

下面的例子中使用`__attribute__`属性定义了一些结构体及其变量，并给出了输出结果和对结果的分析。

程序代码为：

```cpp
//  程序代码为：
  struct p
  {
    int a;
    char b;
    char c;
  }__attribute__((aligned(4))) pp;
  struct q
  {
    int a;
    char b;
    struct n qn;
    char c;
  }__attribute__((aligned(8))) qq;
  
  int main()
  {
    printf("sizeof(int)=%d,sizeof(short)=%d.sizeof(char)=%d\n",sizeof(int),sizeof(short),sizeof(char));
    printf("pp=%d,qq=%d \n", sizeof(pp),sizeof(qq));
    return 0;
  }

```

* 输出结果：
  sizeof(int)=4,sizeof(short)=2.sizeof(char)=1
  pp=8,qq=24

* 结果分析：
sizeof(int)=4,sizeof(short)=2.sizeof(char)=1
pp=8,qq=24
sizeof(pp):
sizeof(a)+ sizeof(b)+ sizeof(c)=4+1+1=6<23=8= sizeof(pp)
sizeof(qq):
sizeof(a)+ sizeof(b)=4+1=5
sizeof(qn)=8;即qn是采用8字节对齐的，所以要在a，b后面添3个空余字节，然后才能存储qn，
4+1+（3）+8+1=17
因为qq采用的对齐是8字节对齐，所以qq的大小必定是8的整数倍，即qq的大小是一个比17大又是8的倍数的一个最小值，由此得到
17<24+8=24= sizeof(qq)

## `__declspec`

| Compiler |Simple deprecation| Deprecation with message|
|:---------|:------------------|:-----------------------|
|gcc and clang| `__attribute__((deprecated)) int a;`| `__attribute__((deprecated("message"))) int a;`|
|Visual Studio| `__declspec(deprecated) int a;`  |`__declspec(deprecated("message")) int a;` |
|Embarcadero(1)| `int a [[deprecated]];`  |`int a [[deprecated("message")]];`|

[table from](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html) 
[`__declspec` blog](http://www.cnblogs.com/ylhome/archive/2010/07/10/1774770.html)

## C++中容易忽略的库
1. bitset
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

2. type_traits

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

##  gcc `__attribute__`关键字举例之visibility
看opencv的源代码的时候，发现`CV_EXPORT`的宏定义是

```cpp
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined CVAPI_EXPORTS
# define CV_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
# define CV_EXPORTS __attribute__ ((visibility ("default")))
#else
# define CV_EXPORTS
#endif
```
我就发现了新大陆似的开始找这个属性的特点。
### 定义
visibility用于设置动态链接库中函数的可见性，将变量或函数设置为hidden，则该符号仅在本so中可见，在其他库中则不可见。

g++在编译时，可用参数-fvisibility指定所有符号的可见性(不加此参数时默认外部可见，参考man g++中-fvisibility部分)；若需要对特定函数的可见性进行设置，需在代码中使用`__attribute__`设置visibility属性。

编写大型程序时，可用`-fvisibility=hidden`设置符号默认隐藏，针对特定变量和函数，在代码中使用`__attribute__ ((visibility("default")))`另该符号外部可见，这种方法可用有效避免so之间的符号冲突。

下面是visibility的实例，这里extern “C”可以省略（另外两篇文章 gcc `__attribute__`关键字举例之alias 和 C++覆盖系统函数的方法 中extern "C"不可用省略）。

值得注意的是，visibility2.cc中可以调用fun1，原因是visibility1.o和visibility2.o同属于一个so文件。

    visibility1.cc：

```cpp
#include <stdio.h>
extern "C" void fun1()
{
  printf("in %s\n",__FUNCTION__);
}

__attribute__ ((visibility("hidden"))) void fun1();//
```
若编译此文件时使用了参数`-fvisibility=hidden`，则此行可以省略

    visibility2.cc：

```cpp
#include <stdio.h>
extern "C" void fun1();
extern "C" void fun2()
{
  fun1();
  printf("in %s\n",__FUNCTION__);
}
__attribute__ ((visibility("default"))) void fun2();//若编译此文件时没有使用参数-fvisibility或设置参数-fvisibility=default，则此行可以省略
```

    main.cpp

```cpp
extern "C" void fun1();
extern "C" void fun2();
int main()
{
  fun1();
  fun2();
  return 0;
}
```

    Makefile：

```Makefile
all:test
test:main.o libvisibility.so
        g++ -o test main.o -lvisibility -L .
main.o::main.cc
        g++ -c main.cc
libvisibility.so:visibility1.o visibility2.o
        g++ -shared -o libvisibility.so visibility1.o visibility2.o
visibility1.o:visibility1.cc
        g++ -fvisibility=hidden -fPIC -c visibility1.cc
visibility2.o:visibility2.cc
        g++ -fvisibility=hidden -fPIC -c visibility2.cc
clean:
        rm -f *.o *.so test
```
编译和输出：

    $ make
    g++ -c main.cc
    g++ -fvisibility=hidden -fPIC -c visibility1.cc
    g++ -fvisibility=hidden -fPIC -c visibility2.cc
    g++ -shared -o libvisibility.so visibility1.o visibility2.o
    g++ -o test main.o -lvisibility -L .
    main.o: In function `main':
    main.cc:(.text+0x5): undefined reference to `fun1'
    collect2: ld returned 1 exit status
    make: *** [test] Error 1

可以看到，`main()`中可以不可用调用`fun1`,可以调用`fun2`，因为`fun1`已经设置为外部不可见，`fun2`设置为外部可见。

使用readelf对各个.o文件分析可以看到，fun1的Vis属性为HIDDEN，fun2的Vis属性为DEFAULT：

$ readelf -s visibility1.o|grep fun
6: 0000000000000007    5 OBJECT  LOCAL  DEFAULT    6 _ZZ4fun1E12__FUNCTION__
12: 0000000000000000    30 FUNC    GLOBAL HIDDEN    2 fun1

$ readelf -s visibility2.o|grep fun
6: 0000000000000007    5 OBJECT  LOCAL  DEFAULT    6 _ZZ4fun2E12__FUNCTION__
12: 0000000000000000    35 FUNC    GLOBAL DEFAULT    2 fun2
15: 0000000000000000    0 NOTYPE  GLOBAL DEFAULT  UND fun1

$ readelf -s libvisibility.so|grep fun
9: 00000000000006ac    35 FUNC    GLOBAL DEFAULT  12 fun2
41: 000000000000071d    5 OBJECT  LOCAL  DEFAULT  14 _ZZ4fun1E12__FUNCTION__
43: 0000000000000729    5 OBJECT  LOCAL  DEFAULT  14 _ZZ4fun2E12__FUNCTION__
48: 000000000000068c    30 FUNC    LOCAL  HIDDEN  12 fun1
54: 00000000000006ac    35 FUNC    GLOBAL DEFAULT  12 fun2

-- 参考：
[Function Attributes](https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html#Function-Attributes)
[Visibility Pragmas](https://gcc.gnu.org/onlinedocs/gcc/Visibility-Pragmas.html#Visibility-Pragmas)
[GCC扩展 __attribute__ ((visibility("hidden")))](http://liulixiaoyao.blog.51cto.com/1361095/814329)
