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


## 更多C++内容
- http://deepindeed.cn/2018/11/28/gnu-cpp-Relearn/

- http://deepindeed.cn/2019/03/18/cpp-program-trick/