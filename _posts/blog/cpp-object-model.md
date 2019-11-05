---
layout: post
title: "C++ Object Model"
categories: [blog]
tags: [C++]
description: "如果不了解编译器对我们的C++代码做了什么手脚，很多困惑永远都解不开。"
---

## warning C4200: nonstandard extension used : zero-sized array in struct/union Cannot generate copy-ctor or copy-assignment operator when UDT contains a zero-sized array

https://stackoverflow.com/questions/3350852/how-to-correctly-fix-zero-sized-array-in-struct-union-warning-c4200-without

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

```
#pragma   warning(push) 
#pragma   warning(disable:XXXX)    //

需要消除警告的代码

#pragma   warning(pop)
```
or:
```
#pragma   warning(disable:XXXX) 

需要消除警告的代码
#pragma   warning(enable:XXXX)//如果出现：'enable'not valid specifier 用 #pragma   warning(default:XXXX)  代替试试
```

#### `#pragma` 支持 
开发人员可以使用 `#pragma` 指令将警告作为错误处理；还可以启用或禁用警告，如下面的示例所示：
#pragma warning (error: 6260) 
#pragma warning (disable: 6011) 
#pragma warning (enable: 6056)
 
`Q: #pragma warning (disable : 4996)和#pragma warning (default : 4996) 是干啥用的呢？`

> 
    1. #pragma warning(disable: n) 
    将某个警报置为失效 
    1. #pragma warning(default: n) 
    将报警置为默认 
    使用VS2005,编译提示"xxxxxx被声明为否决的 
    这是MS新的C库提供的带有检查的函数,有内存溢出检测。可以防止一部分程序bug, 抵制缓冲区溢出攻击(buffer overflow attack). 但是应该速度上有牺牲。 

    解决办法 
    - 所以在你确信安全的情况下,可以用#pragma warning(disable: 4996)消除这个警告 
    - 建议使用_s的缓冲区安全的版本，而不是简单的屏蔽警告。 
    #pragma warning (disable: 4996) // 太多警告看着厌烦无视之 

### 关于#pragma warning

1. #pragma warning只对当前文件有效（对于.h，对包含它的cpp也是有效的），
而不是是对整个工程的所有文件有效。当该文件编译结束，设置也就失去作用。

2. #pragma warning(push) 存储当前报警设置。
#pragma warning(push, n) 存储当前报警设置，并设置报警级别为n。n为从1到4的自然数。
3. #pragma warning(pop)
恢复之前压入堆栈的报警设置。在一对push和pop之间作的任何报警相关设置都将失效。
4. #pragma warning(disable: n)  将某个警报置为失效
5. #pragma warning(default: n)  将报警置为默认
6. 某些警告如C4309是从上到下生效的。即文件内#pragma warning从上到下遍历，依次生效。
 
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
      则C4189仍然会出现，因为default指令是函数的最后一条。在该文件内的其他函数中，如果没有重新设置，C4189也是以#pragma warning(default: 4189)为准。如果重新设置，同样是按照其函数中的最后一个#pragma warning为准。

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