---
layout: post
title: "C++ Programming Tricks"
categories: [blog ]
tags: [C++, 开发]
description: "如果不是使用C++开发过大型系统项目，可能一些编程语言的feature将继续雪藏，让我们一起来挖掘这些秘密吧。"
---

* content
{:toc}

## `pragma warning`

[关于warning的一个问题： warning C4200: nonstandard extension used : zero-sized array in struct/union Cannot generate copy-ctor or copy-assignment operator when UDT contains a zero-sized array](
https://stackoverflow.com/questions/3350852/how-to-correctly-fix-zero-sized-array-in-struct-union-warning-c4200-without)


#### 常用去警告：

* `#pragma warning(disable:4035)` //no return value
* `#pragma warning(disable:4068)` //unknown pragma
* `#pragma warning(disable:4201)` //nonstandard extension used : nameless struct/union
* `#pragma warning(disable:4267)`
* `#pragma warning(disable:4018)` //signed/unsigned mismatch
* `#pragma warning(disable:4127)` //conditional expression is constant
* `#pragma warning(disable:4146)`
* `#pragma warning(disable:4244)` //conversion from 'LONG_PTR' to 'LONG', possible loss of data
* `#pragma warning(disable:4311)` //'type cast' : pointer truncation from 'BYTE *' to 'ULONG'
* `#pragma warning(disable:4312)` //'type cast' : conversion from 'LONG' to 'WNDPROC' of greater size
* `#pragma warning(disable:4346)` //_It::iterator_category' : dependent name is not a type
* `#pragma warning(disable:4786)`
* `#pragma warning(disable:4541)` //'dynamic_cast' used on polymorphic type
* `#pragma warning(disable:4996)` //declared deprecated ?
* `#pragma warning(disable:4200)` //zero-sized array in struct/union
* `#pragma warning(disable:4800)` //forcing value to bool 'true' or 'false' (performance warning)

#### 常用用法:

```cpp
#pragma   warning(push) 
#pragma   warning(disable:XXXX)    // 需要消除警告的代码
#pragma   warning(pop)
```
or:
```cpp
#pragma   warning(disable:XXXX) // 需要消除警告的代码
#pragma   warning(enable:XXXX)  // 如果出现：'enable'not valid specifier 用 
                                // #pragma   warning(default:XXXX)  代替试试
```

#### `#pragma` 支持 
开发人员可以使用 `#pragma` 指令将警告作为错误处理；还可以启用或禁用警告，如下面的示例所示：
```cpp
#pragma warning (error: 6260) 
#pragma warning (disable: 6011) 
#pragma warning (enable: 6056)
```

> `Q: #pragma warning (disable : 4996)和#pragma warning (default : 4996) 是干啥用的呢？`

1. `#pragma warning(disable: n)`
将某个警报置为失效 
2. `#pragma warning(default: n)`
将报警置为默认 
使用VS2005,编译提示"xxxxxx被声明为否决的 
这是MS新的C库提供的带有检查的函数,有内存溢出检测。可以防止一部分程序bug, 抵制缓冲区溢出攻击(buffer overflow attack). 但是应该速度上有牺牲。 

> 解决办法 
- 所以在你确信安全的情况下,可以用#pragma warning(disable: 4996)消除这个警告 
- 建议使用_s的缓冲区安全的版本，而不是简单的屏蔽警告。 

### 关于#pragma warning

1. `#pragma warning`只对当前文件有效（对于.h，对包含它的cpp也是有效的），
而不是是对整个工程的所有文件有效。当该文件编译结束，设置也就失去作用。

2. `#pragma warning(push)` 存储当前报警设置。
`#pragma warning(push, n)` 存储当前报警设置，并设置报警级别为n。n为从1到4的自然数。
3. `#pragma warning(pop)`
恢复之前压入堆栈的报警设置。在一对push和pop之间作的任何报警相关设置都将失效。
4. `#pragma warning(disable: n)`  将某个警报置为失效
5. `#pragma warning(default: n)`  将报警置为默认
6. 某些警告如C4309是从上到下生效的。即文件内`#pragma warning`从上到下遍历，依次生效。
 
      例如：
      ```cpp
      void func()
      {
            #pragma warning(disable: 4189)
            char s;
            s = 128;
            #pragma warning(default: 4189)
            char c;
            c = 128;
      }
      ```
      则s = 128不会产生C4309报警，而C4309会产生报警。

7. 某些警告例如C4189是以函数中最后出现的#pragma warning设置为准的，其余针对该报警的设置都是无效的。
      例如：

      ```cpp
      void func()
      {
            #pragma warning(disable: 4189)
            int x = 1;
            #pragma warning(default: 4189)
      }
      ``` 
      则C4189仍然会出现，因为default指令是函数的最后一条。在该文件内的其他函数中，如果没有重新设置，C4189也是以`#pragma warning(default: 4189)`为准。如果重新设置，同样是按照其函数中的最后一个`#pragma warning`为准。

8. 某些警告（MSDN认为是大于等于C4700的警告）是在函数结束后才能生效。
      例如：

      ```cpp
      #pragma warning(disable:4700)
      void Func() {
            int x;
            int y = x;
            #pragma warning(default:4700)
            int z= x;
      }
      ```

      则y = x和z = x都不会产生C4700报警。只有在函数结束后的后的另外一个函数中，`#pragma warning(default:4700)`才能生效。



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

## 变量属性(Variable Attribute)

关键字`__attribute__ `也可以对变量或结构体成员进行属性设置。这里给出几个常用的参数的解释，更多的参数可参考原文给出的连接。

在使用`__attribute__ `参数时，你也可以在参数的前后都加上“\__”（两个下划线），例如，使用`__attribute__ `而不是aligned，这样，你就可以在相应的头文件里使用它而不用关心头文件里是否有重名的宏定义。

### aligned (alignment)

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

## 类型属性（Type Attribute）

关键字`__attribute__`也可以对结构体（struct）或共用体（union）进行属性设置。大致有六个参数值可以被设定，即：`aligned, packed, transparent_union, unused, deprecated `和 `may_alias`。

在使用`__attribute__`参数时，你也可以在参数的前后都加上“\__”（两个下划线），例如，使用`__aligned__`而不是`aligned`，这样，你就可以在相应的头文件里使用它而不用关心头文件里是否有重名的宏定义。

### aligned (alignment)

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

## 变量属性与类型属性举例

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


##  gcc `__attribute__`关键字举例之`visibility`

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
我就发现了新大陆似的开始找这个属性的特点。这个在工程中尤其重要，我们实现的函数要想被其他用户调用，就必须使用`visibility`让
用户可见，否则我们的实现的功能函数对用户隐藏，出现"undefined reference".

> visibility用于设置动态链接库中函数的可见性，将变量或函数设置为hidden，则该符号仅在本so中可见，在其他库中则不可见。

g++在编译时，可用参数`-fvisibility`指定所有符号的可见性(不加此参数时默认外部可见，参考man g++中`-fvisibility`部分)；若需要对特定函数的可见性进行设置，需在代码中使用`__attribute__`设置visibility属性。

编写大型程序时，可用`-fvisibility=hidden`设置符号默认隐藏，针对特定变量和函数，在代码中使用`__attribute__ ((visibility("default")))`另该符号外部可见，这种方法可用有效避免so之间的符号冲突。

下面是visibility的实例，这里extern “C”可以省略（另外两篇文章 gcc `__attribute__`关键字举例之alias 和 C++覆盖系统函数的方法 中extern "C"不可用省略）。

值得注意的是，visibility2.cc中可以调用fun1，原因是visibility1.o和visibility2.o同属于一个so文件。

> visibility1.cc：

```cpp
#include <stdio.h>
extern "C" void fun1()
{
  printf("in %s\n",__FUNCTION__);
}

__attribute__ ((visibility("hidden"))) void fun1();//
```
若编译此文件时使用了参数`-fvisibility=hidden`，则此行可以省略

> visibility2.cc：

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

> main.cpp

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

> Makefile：

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
> 编译和输出：
```sh
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
```
可以看到，`main()`中可以不可用调用`fun1`,可以调用`fun2`，因为`fun1`已经设置为外部不可见，`fun2`设置为外部可见。

使用readelf对各个.o文件分析可以看到，fun1的Vis属性为HIDDEN，fun2的Vis属性为DEFAULT：

```sh
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
```

# Linux 内核中的 GCC 特性

- 功能性 扩展提供新功能。
- 优化 扩展帮助生成更高效的代码。

## 功能性扩展

### 类型发现

GCC 允许通过变量的引用识别类型。这种操作支持泛型编程。在 C++、Ada 和 Java™ 语言等许多现代编程语言中都可以找到相似的功能。Linux 使用 typeof 构建 min 和 max 等依赖于类型的操作。清单 1 演示如何使用 typeof 构建一个泛型宏（见 ./linux/include/linux/kernel.h）。

清单 1. 使用 typeof 构建一个泛型宏
```
#define min(x, y) ({                \
    typeof(x) _min1 = (x);          \
    typeof(y) _min2 = (y);          \
    (void) (&_min1 == &_min2);      \
    _min1 < _min2 ? _min1 : _min2; })
```

### 范围扩展

GCC 支持范围，在 C 语言的许多方面都可以使用范围。其中之一是 switch/case 块中的 case 语句。在复杂的条件结构中，通常依靠嵌套的 if 语句实现与清单 2（见 ./linux/drivers/scsi/sd.c）相同的结果，但是清单 2 更简洁。使用 switch/case 也可以通过使用跳转表实现进行编译器优化。

清单 2. 在 case 语句中使用范围
```c
static int sd_major(int major_idx)
{
    switch (major_idx) {
    case 0:
        return SCSI_DISK0_MAJOR;
    case 1 ... 7:
        return SCSI_DISK1_MAJOR + major_idx - 1;
    case 8 ... 15:
        return SCSI_DISK8_MAJOR + major_idx - 8;
    default:
        BUG();
        return 0;   /* shut up gcc */
    }
}
```
还可以使用范围进行初始化，如下所示（见` ./linux/arch/cris/arch-v32/kernel/smp.c`）。在这个示例中，`spinlock_t` 创建一个大小为` LOCK_COUNT` 的数组。数组的每个元素初始化为` SPIN_LOCK_UNLOCKED` 值。
```c
/* Vector of locks used for various atomic operations */
spinlock_t cris_atomic_locks[] = { [0 ... LOCK_COUNT - 1] = SPIN_LOCK_UNLOCKED};
```
范围还支持更复杂的初始化。例如，以下代码指定数组中几个子范围的初始值。
`int widths[] = { [0 ... 9] = 1, [10 ... 99] = 2, [100] = 3 };`

### 零长度的数组

在 C 标准中，必须定义至少一个数组元素。这个需求往往会使代码设计复杂化。但是，GCC 支持零长度数组的概念，这对于结构定义尤其有用。这个概念与 ISO C99 中灵活的数组成员相似，但是使用不同的语法。

下面的示例在结构的末尾声明一个没有成员的数组（见 `./linux/drivers/ieee1394/raw1394-private.h`）。这允许结构中的元素引用结构实例后面紧接着的内存。在需要数量可变的数组成员时，这个特性很有用。
```
struct iso_block_store {
        atomic_t refcount;
        size_t data_size;
        quadlet_t data[0];
};
```

### 判断调用地址

在许多情况下，需要判断给定函数的调用者。GCC 提供用于此用途的内置函数 `__builtin_return_address`。这个函数通常用于调试，但是它在内核中还有许多其他用途。

如下面的代码所示，`__builtin_return_address` 接收一个称为 level 的参数。这个参数定义希望获取返回地址的调用堆栈级别。例如，如果指定 level 为 0，那么就是请求当前函数的返回地址。如果指定 level 为 1，那么就是请求进行调用的函数的返回地址，依此类推。
`void * __builtin_return_address( unsigned int level );`
在下面的示例中（见 ./linux/kernel/softirq.c），`local_bh_disable` 函数在本地处理器上禁用软中断，从而禁止在当前处理器上运行 `softirqs`、`tasklets `和 `bottom halves`。使用` __builtin_return_address` 捕捉返回地址，以便在以后进行跟踪时使用这个地址。
```
void local_bh_disable(void){
        __local_bh_disable((unsigned long)__builtin_return_address(0));
}
```

### 常量检测

在编译时，可以使用 GCC 提供的一个内置函数判断一个值是否是常量。这种信息非常有价值，因为可以构造出能够通过常量叠算（constant folding）优化的表达式。`__builtin_constant_p` 函数用来检测常量。

`__builtin_constant_p` 的原型如下所示。注意，`__builtin_constant_p` 并不能检测出所有常量，因为 GCC 不容易证明某些值是否是常量。
`int __builtin_constant_p( exp )`
Linux 相当频繁地使用常量检测。在清单 3 所示的示例中（见 ./linux/include/linux/log2.h），使用常量检测优化 `roundup_pow_of_two` 宏。如果发现表达式是常量，那么就使用可以优化的常量表达式。如果表达式不是常量，就调用另一个宏函数把值向上取整到 2 的幂。

清单 3. 使用常量检测优化宏函数
```
#define roundup_pow_of_two(n)           \
(                       \
    __builtin_constant_p(n) ? (     \
        (n == 1) ? 1 :          \
        (1UL << (ilog2((n) - 1) + 1)) \
                   ) :      \
    __roundup_pow_of_two(n)         \
)
```

### 函数属性

GCC 提供许多函数级属性，可以通过它们向编译器提供更多数据，帮助编译器执行优化。本节描述与功能相关联的一些属性。下一节描述 影响优化的属性。

如清单 4 所示，属性通过其他符号定义指定了别名。可以以此帮助阅读源代码参考，了解属性的使用方法（见 ./linux/include/linux/compiler-gcc3.h）。

清单 4. 函数属性定义
```
# define __inline__  __inline__  __attribute__((always_inline))
# define __deprecated           __attribute__((deprecated))
# define __attribute_used__     __attribute__((__used__))
# define __attribute_const__     __attribute__((__const__))
# define __must_check            __attribute__((warn_unused_result))
```
清单 4 所示的定义是 GCC 中可用的一些函数属性。它们也是在 Linux 内核中最有用的函数属性。下面解释如何使用这些属性：
- `always_inline` 让 GCC 以内联方式处理指定的函数，无论是否启用了优化。
- `deprecated` 指出函数已经被废弃，不应该再使用。如果试图使用已经废弃的函数，就会收到警告。还可以对类型和变量应用这个属性，促使开发人员尽可能少使用它们。
- `__used__` 告诉编译器无论 GCC 是否发现这个函数的调用实例，都要使用这个函数。这对于从汇编代码中调用 C 函数有帮助。
- `__const__` 告诉编译器某个函数是无状态的（也就是说，它使用传递给它的参数生成要返回的结果）。
- `warn_unused_result` 让编译器检查所有调用者是否都检查函数的结果。这确保调用者适当地检验函数结果，从而能够适当地处理错误。

下面是在 Linux 内核中使用这些属性的示例。deprecated 示例来自与体系结构无关的内核（./linux/kernel/resource.c），const 示例来自 IA64 内核源代码（./linux/arch/ia64/kernel/unwind.c）。

```cpp
int __deprecated __check_region(struct resource 
    *parent, unsigned long start, unsigned long n)
 
static enum unw_register_index __attribute_const__ 
    decode_abreg(unsigned char abreg, int memory)
```

## 优化扩展
现在，讨论有助于生成更好的机器码的一些 GCC 特性。

### 分支预测提示

在 Linux 内核中最常用的优化技术之一是` __builtin_expect`。在开发人员使用有条件代码时，常常知道最可能执行哪个分支，而哪个分支很少执行。如果编译器知道这种预测信息，就可以围绕最可能执行的分支生成最优的代码。

如下所示，`__builtin_expect` 的使用方法基于两个宏 likely 和 unlikely（见 ./linux/include/linux/compiler.h）。
```
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
```
通过使用 `__builtin_expect`，编译器可以做出符合提供的预测信息的指令选择决策。这使执行的代码尽可能接近实际情况。它还可以改进缓存和指令流水线。

例如，如果一个条件标上了 “likely”，那么编译器可以把代码的 True 部分直接放在分支指令后面（这样就不需要执行分支指令）。通过分支指令访问条件结构的 False 部分，这不是最优的方式，但是访问它的可能性不大。按照这种方式，代码对于最可能出现的情况是最优的。

清单 5 给出一个使用 likely 和 unlikely 宏的函数（见 ./linux/net/core/datagram.c）。这个函数预测 sum 变量将是零（数据包的 checksum 是有效的），而且 ip_summed 变量不等于 CHECKSUM_HW。

清单 5. likely 和 unlikely 宏的使用示例
```cpp
unsigned int __skb_checksum_complete(struct sk_buff *skb)
{
        unsigned int sum;
 
        sum = (u16)csum_fold(skb_checksum(skb, 0, skb->len, skb->csum));
        if (likely(!sum)) {
                if (unlikely(skb->ip_summed == CHECKSUM_HW))
                        netdev_rx_csum_fault(skb->dev);
                skb->ip_summed = CHECKSUM_UNNECESSARY;
        }
        return sum;
}
```

### 预抓取

另一种重要的性能改进方法是把必需的数据缓存在接近处理器的地方。缓存可以显著减少访问数据花费的时间。大多数现代处理器都有三类内存：
* 一级缓存通常支持单周期访问
* 二级缓存支持两周期访问
* 系统内存支持更长的访问时间

为了尽可能减少访问延时并由此提高性能，最好把数据放在最近的内存中。手工执行这个任务称为预抓取。GCC 通过内置函数 `__builtin_prefetch` 支持数据的手工预抓取。在需要数据之前，使用这个函数把数据放到缓存中。如下所示，`__builtin_prefetch` 函数接收三个参数：

- 数据的地址
- rw 参数，使用它指明预抓取数据是为了执行读操作，还是执行写操作
- locality 参数，使用它指定在使用数据之后数据应该留在缓存中，还是应该清除
`void __builtin_prefetch( const void *addr, int rw, int locality );`

Linux 内核经常使用预抓取。通常是通过宏和包装器函数使用预抓取。清单 6 是一个辅助函数示例，它使用内置函数的包装器（见 ./linux/include/linux/prefetch.h）。这个函数为流操作实现预抓取机制。使用这个函数通常可以减少缓存缺失和停顿，从而提高性能。

清单 6. 范围预抓取的包装器函数
```cpp
#ifndef ARCH_HAS_PREFETCH
#define prefetch(x) __builtin_prefetch(x)
#endif
 
static inline void prefetch_range(void *addr, size_t len)
{
#ifdef ARCH_HAS_PREFETCH
    char *cp;
    char *end = addr + len;
 
    for (cp = addr; cp < end; cp += PREFETCH_STRIDE)
        prefetch(cp);
#endif
}
```

### 变量属性

除了本文前面讨论的函数属性之外，GCC 还为变量和类型定义提供了属性。最重要的属性之一是 `aligned` 属性，它用于在内存中实现对象对齐。除了对于性能很重要之外，某些设备或硬件配置也需要对象对齐。`aligned` 属性有一个参数，它指定所需的对齐类型。

下面的示例用于软件暂停（见 ./linux/arch/i386/mm/init.c）。在需要页面对齐时，定义 `PAGE_SIZE` 对象。
```
char __nosavedata swsusp_pg_dir[PAGE_SIZE]
    __attribute__ ((aligned (PAGE_SIZE)));
```
清单 7 中的示例说明关于优化的两点：

`packed` 属性打包一个结构的元素，从而尽可能减少它们占用的空间。这意味着，如果定义一个 char 变量，它占用的空间不会超过一字节（8 位）。位字段压缩为一位，而不会占用更多存储空间。
这段源代码使用一个` __attribute__` 声明进行优化，它用逗号分隔的列表定义多个属性。
清单 7. 结构打包和设置多个属性
```cpp
static struct swsusp_header {
        char reserved[PAGE_SIZE - 20 - sizeof(swp_entry_t)];
        swp_entry_t image;
        char    orig_sig[10];
        char    sig[10];
} __attribute__((packed, aligned(PAGE_SIZE))) swsusp_header;
```



## 参考链接

1. [Function Attributes](https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html#Function-Attributes)

2. [Visibility Pragmas](https://gcc.gnu.org/onlinedocs/gcc/Visibility-Pragmas.html#Visibility-Pragmas)

3. [GCC扩展 __attribute__ ((visibility("hidden")))](http://liulixiaoyao.blog.51cto.com/1361095/814329)

4. [【IBM】Linux 内核中的 GCC 特性](https://www.ibm.com/developerworks/cn/linux/l-gcc-hacks/)
