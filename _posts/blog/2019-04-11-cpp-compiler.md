---
layout: post
title: "笔记：编译器中的一些options"
categories: [blog ]
tags: [C++, 开发]
description: "编译器中一些不太常用，但是有的时候很有用的options"
---

* content
{:toc}

## 引言
记录开发工作中与编译器相关的一些控制项目，方便日后查阅。


## gcc包含静态库中所有符号的option

`-Wl,--whole-archive xxxxx_lib -Wl,--no-whole-archive`

```cmake
target_link_libraries(xxxx_export 
            PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:xxxxx_lib>
                    "-Wl,--no-whole-archive -Wl,--exclude-libs,ALL")
```

## `--exclude-libs` affected by the `--whole-archive` option.

### Problem

`--exclude-libs` does not work for static libraries affected by the `--whole-archive` option.

### Description

`--exclude-libs` creates a list of static library paths and does library lookups in this list.
`--whole-archive` splits the static libraries that follow it into separate objects. As a result, lld no longer sees static libraries among linked files and does no `--exclude-libs` lookups.

### Solution

The proposed solution is to make `--exclude-libs` consider object files too. When lld finds an object file it checks whether this file originates from an archive and, if so, looks the archive up in the `--exclude-libs` list.

**Reference**: https://reviews.llvm.org/D39353


## windows下export所有的符号

cmake使用`cmake -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE`

    Enable this boolean property to automatically create a module definition (.def) file with all global symbols found in the input .obj files for a SHARED library on Windows. The module definition file will be passed to the linker causing all symbols to be exported from the .dll. For global data symbols, __declspec(dllimport) must still be used when compiling against the code in the .dll. All other function symbols will be automatically exported and imported by callers. This simplifies porting projects to Windows by reducing the need for explicit dllexport markup, even in C++ classes.

    This property is initialized by the value of the CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS variable if it is set when a target is created.

**Reference**: [`WINDOWS_EXPORT_ALL_SYMBOLS`](https://cmake.org/cmake/help/v3.4/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html)


## gcc/g++的`--as-needed`

gcc/g++提供了`-Wl,--as-needed`和 `-Wl,--no-as-needed`两个选项，这两个选项一个是开启特性，一个是取消该特性。

在生成可执行文件的时候，通过 -lxxx 选项指定需要链接的库文件。以动态库为例，如果我们指定了一个需要链接的库，则连接器会在可执行文件的文件头中会记录下该库的信息。而后，在可执行文件运行的时候，动态加载器会读取文件头信息，并加载所有的链接库。在这个过程中，如果用户指定链接了一个毫不相关的库，则这个库在最终的可执行程序运行时也会被加载，如果类似这样的不相关库很多，会明显拖慢程序启动过程。

这时，通过指定`-Wl,--as-needed`选项，链接过程中，链接器会检查所有的依赖库，没有实际被引用的库，不再写入可执行文件头。最终生成的可执行文件头中包含的都是必要的链接库信息。`-Wl,--no-as-needed`选项不会做这样的检查，会把用户指定的链接库完全写入可执行文件中。

**Reference**: [GCC/G++选项 -Wl,--as-needed](https://my.oschina.net/yepanl/blog/2222870)


## -rdynamic

    Pass the flag `-export-dynamic` to the ELF linker, on targets that support
    it. This instructs the linker to add all symbols, not only used ones, to the dynamic symbol table. This option is needed for some uses of `dlopen` or to allow obtaining backtraces from within a program.

关键的不同是：`-Wl,--export-dynamic -pthread`
`-Wl`:指示后面的选项是给链接器的
`-pthread`: 链接程序的时包含libpthread.so
`--export-dynamic`：就是这个选项让主程序内定义的全局函数对库函数可见。

**Reference**: [gcc链接选项--export-dynamic的一次问题记录](https://blog.csdn.net/u011644231/article/details/88880362)

## `_GLIBCXX_USE_CXX11_ABI`
在GCC 5.1版本中，libstdc++引入了一个新的ABI，其中包括std::string和std::list的新实现。为了符合2011年c++标准，这些更改是必要的，该标准禁止复制即写字符串，并要求列表跟踪字符串的大小。
为了保持与libstdc++链接的现有代码的向后兼容性，库的soname没有更改，并且仍然支持与新实现并行的旧实现。这是通过在内联命名空间中定义新的实现来实现的，因此它们具有不同的用于链接目的的名称，例如，`std::list`的新版本实际上定义为`std:: _cxx11::list`。因为新实现的符号有不同的名称，所以两个版本的定义可以出现在同一个库中。
`_GLIBCXX_USE_CXX11_ABI`宏控制库头中的声明是使用旧ABI还是新ABI。因此，可以为正在编译的每个源文件分别决定使用哪个ABI。使用GCC的默认配置选项，宏的默认值为1，这将导致新的ABI处于活动状态，因此要使用旧的ABI，必须在包含任何库头之前显式地将宏定义为0。(**注意，一些GNU/Linux发行版对GCC 5的配置不同，因此宏的默认值是0，用户必须将它定义为1才能启用新的ABI**)。

```cmake
IF(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.1")
	ADD_DEFINITIONS(-D_GLIBCXX_USE_CXX11_ABI=0)
ENDIF()
```

## -Wl,--allow-shlib-undefined

`SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--allow-shlib-undefined")`

### Linking errors with “-Wl,--no-undefined -Wl,--no-allow-shlib-undefined”
第二个参数的默认值是`--allow-shlib-undefined`。如果您选择该选项，代码可能会生成。
第二个参数处理构建时检查，启用它意味着检查您所链接的库是否在构建时连接了其依赖项。

第一个参数确保您没有忘记声明对运行时库的依赖项(也可能是运行时库对另一个运行时库的依赖项)。
例如，如果您调用的函数的实现位于示例运行时库“libfunc”中。然后这个库会调用另一个运行时库中的函数libext。然后通过声明对libfunc的“func”和“ext”的依赖关系。因此，将在内部生成一个对libext的依赖引用。
如果您省略`--no undefined`并忘记添加依赖项声明，那么构建仍然会成功，因为您相信运行时链接器将在运行时解析依赖项。
由于构建成功了，您可能会相信一切都会好起来，而不知道构建已经将责任推迟到运行时链接器。
但大多数情况下，运行时链接器的设计目的不是搜索未解析的引用，而是希望找到运行时库中声明的此类依赖项。如果没有这样的引用，您将得到一个运行时错误。
运行时错误通常比解决编译时错误要昂贵得多。

## 更多C++内容
- http://deepindeed.cn/2018/11/28/gnu-cpp-Relearn/

- http://deepindeed.cn/2019/03/18/cpp-program-trick/

- libstdc++关于dual ABI文档: https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html


# GCC不同版本中一些东西

## GCC4.9.4
> The `-Wdate-time` option has been added for the C, C++ and Fortran compilers, which warns when the `__DATE__`, `__TIME__` or `__TIMESTAMP__` macros are used. Those macros might prevent bit-wise-identical reproducible compilations.

> With the new `#pragma GCC ivdep`, the user can assert that there are no loop-carried dependencies which would prevent concurrent execution of consecutive iterations using SIMD (single instruction multiple data) instructions.

### Inter-procedural optimization improvements:
* New type inheritance analysis module improving devirtualization. Devirtualization now takes into account anonymous name-spaces and the C++11 final keyword.
* New speculative devirtualization pass (controlled by `-fdevirtualize-speculatively`.
* Calls that were speculatively made direct are turned back to indirect where direct call is not cheaper.
* Local aliases are introduced for symbols that are known to be semantically equivalent across shared libraries improving dynamic linking times.

### Feedback directed optimization improvements:

* Profiling of programs using C++ inline functions is now more reliable.
* New time profiling determines typical order in which functions are executed.
* A new function reordering pass (controlled by -freorder-functions) significantly reduces startup time of large applications. Until binutils support is completed, it is effective only with link-time optimization.
* Feedback driven indirect call removal and devirtualization now handle cross-module calls when link-time optimization is enabled.

https://gcc.gnu.org/gcc-4.9/porting_to.html

## GCC 5.4 
未完,待续