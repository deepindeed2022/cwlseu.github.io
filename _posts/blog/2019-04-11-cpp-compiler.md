---
layout: post
title: "gcc编译器不可不知的options"
categories: [blog ]
tags: [C++, 开发]
description: "编译器中一些不太常用，但是有的时候很有用的options"
---

* content
{:toc}

## 引言

编译器是我们开发人员与机器指令之间的翻译,现在编译器越来越优化,而且基于一些开源的编译器项目(gcc,clang)等,相继出现不同platform下的编译器。
此外，各种芯片、开发板层出不穷，各个商业公司都针对自己出产的开发板定制特定的编译链条。例如华为hisi系列的himix100中提供的编译链中，包括编译器，链接器，打包器之外，还提供了nm，gdb，gcov，gprof等等开发工具。
这篇文章将主要将开发工作中与编译器（这篇文章中不作特殊说明，指的是gnu gcc编译器）相关的一些options和配置参数进行总结,方便在后面的项目遇到相似的问题进行查阅与借鉴。

## 包含静态库中所有符号的option

编译器编译动态库或者运行程序的时候，会对依赖的静态库中进行基于`.o`的选择，但是有的时候我们希望我们编译的动态库能够包含所有的函数实现给用户使用。gcc中的链接控制选项`-Wl,--whole-archive xxxxx_lib -Wl,--no-whole-archive`就可以实现类似功能。

```cmake
target_link_libraries(xxxx_export 
            PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:xxxxx_lib>
                    "-Wl,--no-whole-archive -Wl,--exclude-libs,ALL")
```

#### 其他可能问题

`--exclude-libs` does not work for static libraries affected by the `--whole-archive` option.

* `--exclude-libs` creates a list of static library paths and does library lookups in this list.
* `--whole-archive` splits the static libraries that follow it into separate objects. As a result, lld no longer sees static libraries among linked files and does no `--exclude-libs` lookups.

#### Solution

The proposed solution is to make `--exclude-libs` consider object files too. When lld finds an object file it checks whether this file originates from an archive and, if so, looks the archive up in the `--exclude-libs` list.

**Reference**: https://reviews.llvm.org/D39353

### windows

在windows常用的编译器是VS里面的cl编译器。我们要实现上述
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
在交叉编译程序过程中，往往会有这样的情况，依赖的target系统上的动态库（例如android上的OpenCL.so）又依赖其他的许多动态库，这个时候，我们希望链接target系统上的这个动态库的时候，我们可以不要去找OpenCL相关的依赖符号。

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

### `TARGET_LINK_LIBRARY` & `LINK_LIBRARY`
target_link_libraries 会将需要链接的库作为属性挂在目标库上，
后面用户用到这个库的时候可以通过`get_target_property(interface_link_libs ${} TARGET_LINK_LIBRARIES)`进行获取相应的值。

## 编译运行查找头文件和库的顺序

### 头文件

gcc 在编译时如何去寻找所需要的头文件：
* 所以header file的搜寻会从-I开始
* 然后找gcc的环境变量 `C_INCLUDE_PATH`，`CPLUS_INCLUDE_PATH`，`OBJC_INCLUDE_PATH`
* 再找内定目录
  * `/usr/include`
  * `/usr/local/include`

gcc的一系列自带目录
`CPLUS_INCLUDE_PATH=/usr/lib/gcc/x86_64-linux-gnu/4.9.4/include:/usr/include/c++/4.9.4`

### 库文件

编译的时候：
* gcc会去找-L
* 再找gcc的环境变量LIBRARY_PATH
* 再找内定目录
  * `/lib`和`/lib64`
  * `/usr/lib` 和`/usr/lib64`
  * `/usr/local/lib`和`/usr/local/lib64`

这是当初compile gcc时写在程序内的

### 运行时动态库的搜索路径

动态库的搜索路径搜索的先后顺序是：
1. 编译目标代码时指定的动态库搜索路径；
2. 环境变量`LD_LIBRARY_PATH`指定的动态库搜索路径；
3. 配置文件`/etc/ld.so.conf`中指定的动态库搜索路径；
4. 默认的动态库搜索路径`/lib`；
5. 默认的动态库搜索路径`/usr/lib`。

### 动态库中的static变量

> In all cases, static global variables (or functions) are never visible from outside a module (dll/so or executable). The C++ standard requires that these have internal linkage, meaning that they are not visible outside the translation unit (which becomes an object file) in which they are defined.

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

* The default mode for C is now -std=gnu11 instead of -std=gnu89.
* The C++ runtime library (libstdc++) uses a new ABI by default (see below).
* The non-standard C++0x type traits `has_trivial_default_constructor`, `has_trivial_copy_constructor` and `has_trivial_copy_assign` have been deprecated and will be removed in a future version. The standard C++11 traits `is_trivially_default_constructible`, `is_trivially_copy_constructible` and `is_trivially_copy_assignable` should be used instead.

* 添加`-fipa-icf`的配置项目
> An Identical Code Folding (ICF) pass (controlled via -fipa-icf) has been added. Compared to the identical code folding performed by the Gold linker this pass does not require function sections. It also performs merging before inlining, so inter-procedural optimizations are aware of the code re-use. On the other hand not all unifications performed by a linker are doable by GCC which must honor aliasing information.

* The devirtualization pass was significantly improved by adding better support for speculative devirtualization and dynamic type detection.

* 虚表进行了优化以减少动态链接时间
Virtual tables are now optimized. Local aliases are used to reduce dynamic linking time of C++ virtual tables on ELF targets and data alignment has been reduced to limit data segment bloat.

* 添加针对不允许插入导出符号的shared库，添加了控制项目以提高代码质量
> A new -fno-semantic-interposition option can be used to improve code quality of shared libraries where interposition of exported symbols is not allowed.

* 内联可以控制
> With profile feedback the function inliner can now bypass --param inline-insns-auto and --param inline-insns-single limits for hot calls.

* 常量的过程间传播现在也传播指针参数的对齐。
> The interprocedural propagation of constants now also propagates alignments of pointer parameters. This for example means that the vectorizer often does not need to generate loop prologues and epilogues to make up for potential misalignments.

* 内存使用上一些优化
> Memory usage and link times were improved. Tree merging was sped up, memory usage of GIMPLE declarations and types was reduced, and, support for on-demand streaming of variable constructors was added.

### libstd++上的优化
* Dual ABI
* A new implementation of std::string is enabled by default, using the small string optimization(SSO) instead of copy-on-write(COW) reference counting.
* A new implementation of std::list is enabled by default, with an O(1) size() function;


## GCC dump preprocessor defines

- 最常用的输出编译器预定义的宏

`gcc -dM -E - < /dev/null`

`g++ -dM -E -x c++ - < /dev/null`

- How do I dump preprocessor macros coming from a particular header file?

`echo "#include <sys/socket.h>" | gcc -E -dM -`

- 添加某些options之后的

`gcc -dM -E -msse4 - < /dev/null | grep SSE[34]`
> #define __SSE3__ 1 \
> #define __SSE4_1__ 1 \
> #define __SSE4_2__ 1 \
> #define __SSSE3__ 1

## TODO

* 常用的交叉编译的选项
* -O3和-O2之间的差别
* 不同平台之间之间的差别
* 如何给不同版本的gcc打补丁

在文章[Algorithm-Optimization][^4]中介绍了一些有利于优化性能的函数，感兴趣可以结合不同平台的优化指令一起学习使用。



# GCC different platform的配置项

[Using static and shared libraries across platforms][^3]


![@](https://cwlseu.github.io/images/gcc/compilerflag_1.png)
![@](https://cwlseu.github.io/images/gcc/compilerflag_2.png)

## 更多C++内容
- http://deepindeed.cn/2018/11/28/gnu-cpp-Relearn/
- http://deepindeed.cn/2019/03/18/cpp-program-trick/
- libstdc++关于dual ABI文档: https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html

## 其他
- [gcc与g++的区别][^1]
- [ARM？华为？][^2]
- himix100的交叉编译链
```
➜  arm-himix100-linux tree -L 2 ./host_bin 
./host_bin
├── arm-linux-androideabi-addr2line
├── arm-linux-androideabi-ar
├── arm-linux-androideabi-as
├── arm-linux-androideabi-c++
├── arm-linux-androideabi-c++filt
├── arm-linux-androideabi-cpp
├── arm-linux-androideabi-elfedit
├── arm-linux-androideabi-g++
├── arm-linux-androideabi-gcc
├── arm-linux-androideabi-gcc-6.3.0
├── arm-linux-androideabi-gcc-ar
├── arm-linux-androideabi-gcc-nm
├── arm-linux-androideabi-gcc-ranlib
├── arm-linux-androideabi-gcov
├── arm-linux-androideabi-gcov-tool
├── arm-linux-androideabi-gdb
├── arm-linux-androideabi-gprof
├── arm-linux-androideabi-ld
├── arm-linux-androideabi-ld.bfd
├── arm-linux-androideabi-nm
├── arm-linux-androideabi-objcopy
├── arm-linux-androideabi-objdump
├── arm-linux-androideabi-ranlib
├── arm-linux-androideabi-readelf
├── arm-linux-androideabi-run
├── arm-linux-androideabi-size
├── arm-linux-androideabi-strings
├── arm-linux-androideabi-strip
```

[^1]: https://www.cnblogs.com/liushui-sky/p/7729838.html
[^2]: https://news.mydrivers.com/1/628/628308.htm
[^3]: http://www.fortran-2000.com/ArnaudRecipes/sharedlib.html
[^4]:http://deepindeed.cn/2017/03/17/Algorithm-Optimization/