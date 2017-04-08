---
layout: post
title: The Annotated STL sources
categories: [blog ]
tags: [C++ 开发, ]
description: STL源码剖析，在源码阅读中学习C++中的一些tricks
---

声明：本博客欢迎转发，但请保留原作者信息! 
作者: [Clython]
博客： [https://cwlseu.github.io/](https://cwlseu.github.io/)


# type_traits 
type_traits可以翻译为类型提取器或者类型萃取器，很直白的说就是通过这个机制可以获取被操作数据类型的一些特征。这个机制在编写模板代码的时候特别有用，可以在编译期间就根据数据类型的特征分派给不同的代码进行处理

```cpp

// This header file provides a framework for allowing compile time dispatch
// based on type attributes. This is useful when writing template code.
// For example, when making a copy of an array of an unknown type, it helps
// to know if the type has a trivial copy constructor or not, to help decide
// if a memcpy can be used.

struct __true_type {
};

struct __false_type {
};

template <class _Tp>
struct __type_traits { 
   typedef __true_type     this_dummy_member_must_be_first;
                   /* Do not remove this member. It informs a compiler which
                      automatically specializes __type_traits that this
                      __type_traits template is special. It just makes sure that
                      things work if an implementation is using a template
                      called __type_traits for something unrelated. */

   /* The following restrictions should be observed for the sake of
      compilers which automatically produce type specific specializations 
      of this class:
          - You may reorder the members below if you wish
          - You may remove any of the members below if you wish
          - You must not rename members without making the corresponding
            name change in the compiler
          - Members you add will be treated like regular members unless
            you add the appropriate support in the compiler. */
 
   typedef __false_type    has_trivial_default_constructor;
   typedef __false_type    has_trivial_copy_constructor;
   typedef __false_type    has_trivial_assignment_operator;
   typedef __false_type    has_trivial_destructor;
   typedef __false_type    is_POD_type;
};
// The class template __type_traits provides a series of typedefs each of
// which is either __true_type or __false_type. The argument to
// __type_traits can be any type. The typedefs within this template will
// attain their correct values by one of these means:
//     1. The general instantiation contain conservative values which work
//        for all types.
//     2. Specializations may be declared to make distinctions between types.
//     3. Some compilers (such as the Silicon Graphics N32 and N64 compilers)
//        will automatically provide the appropriate specializations for all
//        types.

EXAMPLE:

//Copy an array of elements which have non-trivial copy constructors
template <class T> void copy(T* source, T* destination, int n, __false_type);
//Copy an array of elements which have trivial copy constructors. Use memcpy.
template <class T> void copy(T* source, T* destination, int n, __true_type);

//Copy an array of any type by using the most efficient copy mechanism
template <class T> inline void copy(T* source,T* destination,int n) {
   copy(source, destination, n,
        typename __type_traits<T>::has_trivial_copy_constructor());
}
```

POD意思是Plain Old Data,也就是标量性别或者传统的C struct型别。POD性别必然拥有trivial ctor/doct/copy/assignment 函数,因此我们就可以对POD型别采用最为有效的复制方法，而对non-POD
型别采用最保险安全的方法

```cpp

// uninitialized_copy

// Valid if copy construction is equivalent to assignment, and if the
//  destructor is trivial.
template <class _InputIter, class _ForwardIter>
inline _ForwardIter 
__uninitialized_copy_aux(_InputIter __first, _InputIter __last,
                         _ForwardIter __result,
                         __true_type)
{
  return copy(__first, __last, __result);
}

template <class _InputIter, class _ForwardIter>
_ForwardIter 
__uninitialized_copy_aux(_InputIter __first, _InputIter __last,
                         _ForwardIter __result,
                         __false_type)
{
  _ForwardIter __cur = __result;
  __STL_TRY {
    for ( ; __first != __last; ++__first, ++__cur)
      _Construct(&*__cur, *__first);
    return __cur;
  }
  __STL_UNWIND(_Destroy(__result, __cur));
}


template <class _InputIter, class _ForwardIter, class _Tp>
inline _ForwardIter
__uninitialized_copy(_InputIter __first, _InputIter __last,
                     _ForwardIter __result, _Tp*)
{
  typedef typename __type_traits<_Tp>::is_POD_type _Is_POD;
  return __uninitialized_copy_aux(__first, __last, __result, _Is_POD());
}
```

