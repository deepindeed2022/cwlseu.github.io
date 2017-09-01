---
layout: post
title: malloc如何实现
categories: [blog ]
tags: [Linux开发, C++开发]
description: 基础原理的实现
---




声明：本博客欢迎转发，但请保留原作者信息!

作者: 曹文龙

博客： <https://cwlseu.github.io/>


## 简介
对于刚刚开始学习C 和Unix环境的人来说可能是知道这个名字的，但是*malloc*到底是什么呢，是一个语言的关键词还是仅仅是一个系统调用。其实这个名字的背后隐藏着很多计算机哲学，其中不乏计算机实际的的资源管理。
今天我将通过学习实现一个基本的malloc函数，作为C语言的一个练习，同时学习理解在堆的管理。

首先要了解一`malloc`的功能与约束。
`void* malloc(size_t size);`
* `malloc` 至少分配请求的数目的bytes
* 返回一个指向分配空间位置的指针
* 其他调用`malloc`不会分配未被释放的空间
* `malloc`必须是可快速执行，及时返回结果的，而不是NP-Hard问题
* `malloc` 应该可以重置大小和释放

## 堆和几个系统调用
要想实现`malloc`函数，需要首先对内存管理进行一些了解。
### 进程内存管理
每一个人用户进程又有一个虚拟地址空间，MMU是Memory Management Unit的缩写，中文名是内存管理单元，它是中央处理器（CPU）中用来管理虚拟存储器、物理存储器的控制线路，同时也负责虚拟地址映射为物理地址，以及提供硬件机制的内存访问授权。
![@图1: 内存组织结构示意图](./1472546009371.png)
上图中就是一个Heap空间组织图，从中可以看到有三个分界线。
* 一个开始位置
* 一个最大位置(`getrlimit` 和 `setrlimit` 是两个与rlimit有关系的系统调用) 
* 和映射空间结束位置，就是图中*break*的位置， 蓝色区域表示已经有与物理空间映射了的，而红色区域表示还没有真实的物理空间与这个堆区域映射。

### brk 和sbrk系统调用
为了实现`malloc`函数，我们需要获取堆中地开始位置和break的位置，那么我们就需要了解两个系统调用`brk`和`sbrk`了。
`int brk(const void *addr);`
将break放到`addr`的位置，返回0 表示成功，否则返回 -1;
`void* sbrk(intptr_t incr);` 向前移动incr个byte，返回是新的还是以前的break位置取决于具体的系统实现。

### mmap
虽然这里不使用`mmap`去实现 `malloc`，但是这个函数的一个匿名模式可以分配比较大的内存空间（一般大于一个页），可以实现直接对内存中的文件进行映射。相比`sbrk`更加高效。

##实现malloc
```cpp
#include <sys/types.h>
#include <unisted.h>
void* malloc(size_t size)
{
	void* p;
	p = sbrk(0);
	if(sbrk(size) == (void*)-1)
		return NULL;
	return p;
}
```
### block信息表示
![@堆中数据块的数据结构](./1472555738795.png)

```cpp
typedef stuct s_block *t_block;
struct s_block
{
	 size_t size;
	 t_block next;
	 int free;
}
```
当然上面声明的数据结构不是最终block的样子，我们后续会渐渐增加的。
当我们还没有创建`malloc`函数的时候，那么我们该如何创建`struct s_block`呢。其实，在内存中struct只是连续存在的几个数据罢了。比如上面我们声明的这个struct， 其实就是12 bytes, 因此我们常常在c程序中直接给struct赋值的情景，更有直接给struct对应的内存区域进行字符串拷贝的情形。

![@Block之间的链接关系](./1472563745651.png)

### 内存对齐
一般情况下，我们常常要考虑内存对齐对于访存速度的影响。一般我们的block原始数据会处理为对齐状态，那么我们申请的空间往往也需要对齐，而不是简单的申请。
为了内存对齐，我们需要运用数学计算的一点小把戏让对齐变得更加方便。假设要申请空间大小为x, 可以将x表示为$x = 4 * p + q  $,同时 $0 \leq q \leq 3$，通过对x是4的倍数和不是4的倍数的讨论，我们得出：
`(x - 1)/4 * 4 + 4`就是要找最近的相等或者大于x的数。
在C中实现就可以使用`<<`和`>>`操作

```cpp
#define align4(x) (((((x)-1)>>2)<<2)+4)
```
### 找一块空闲块
朝朝一个空闲而且大小可以得存储空间是比较简单的，从头开始找就是了:）.但是万一就是点背，从头找到尾就是没有找到合适的怎么办呢？所以后面我们还得考虑堆空间的拓展问题。后面再说，我们先把这个找空闲块的实现一下。

```cpp
t_block find_block(t_block* last, size_t size)
{
	t_block b = base;
	while(b && !(b->free && b->size >= size))
	{
		*last = b;
		b = b->next;
	}
	return b;
}
```
### 拓展堆空间
现在我们处理找不到合适空间分配时的情况。我们需要拓展堆空间，通过移动break的位置，重新申请一块大空间挂到最后一个块的next位置上就是了。

```cpp
t_block extend_heap(t_block last, size_t s)
{
	t_block b;
	//save the previous break, because we not sure the sbrk return the previoous break
	b = sbrk(0);
	if(sbrk(BLOCK_SIZE + s) == (void*)-1)
	{
		return NULL;
	}
	b->size = s;
	b->next = NULL;
	if(last)
		last->next = b;
	b->free = 0; //means block b have been used by process
	return b;
}
```
### heap分块
我们通过顺序查找，找到一个块满足条件，但是你就想申请4 bytes，但是找到的块是512 bytes, 你说我们是该全部分给申请者还是割下一块进行分配呢？显然，为了节省空间，尤其是在计算机起步初期，内存是很宝贵的，当然现在虽然内存条价格便宜了不少，但是如果`malloc`这么奢侈的话，很快就会败光所有空间的，因此我们得懂得节约，够用就好。
注意分配的时候我们不是仅仅分配请求的大小，还有有一块存储块元数据的空间，因此我们需要申请`BLOCK_SIZE + 4`

```cpp
void split_block(t_block b, size_t s)
{
	t_block new;
	new = b->data + s;
	new->size = b->size - s - BLOCK_SIZE;
	new->next = b->next;
	new->free = 1;
	b->size = s;
	b->next = new;
}
```
### 碎片问题
我们在申请新的空间的过程中，并不是都能够申请到适当空间的。因为我们在不断申请与释放过程中，将原本的空间变得支离破碎，虽然剩余的总空间可能是很大的，但是由于逻辑上连续的空间由于释放的次序不同，使得两块空间是分离的，我们可以考虑将逻辑上连续的free空间进行合并。
那么我们该如何查找空闲的chunk呢？
* 顺序查找，从头到尾查找，慢
* keep一个search的指针，保存上次查找到什么地方了
* 将单向链表变为双向链表

```cpp
typedef struct s_block* t_block;
struct s_block
{
	size_t size;
	t_block next;
	t_block prev;
	int free;
	char data[1];
}

t_block fusion(t_block b)
{
	if(b->next && b->next->free)
	{
		b->size += BLOCK_SIZE + b->next->size;
		b->next = b->next->next;
		if(b->next)
		{
			b->next->prev = b;
		}
		return b;
	}
}

```
实现很简单，就是如果当前block的下一个块是空闲的，那么就把当前块的size加上下一块和块的metadata的大小。然后处理块之间的链接关系就可以了。
## Free
### 找到正确释放的位置
* 指针是否正确
* 找到该块的元数据的位置
* 如何确定一个指针是否为`malloc`的?
最通俗的想法就是，只要指针不再heap的范围之内，就不是一个有效的指针。我们如何确定一个指针是malloc的呢？我们可以在block结构中放置一个*magic number*进行标记，当然更好的方式是我们刻意考虑用指针自身。如果`b->ptr == b->data`，那么b就是很有可能是一个合法的块。所以再次拓展block的数据结构如下：

```cpp
typedef struct s_block* t_block;
struct s_block
{
	size_t size;
	t_block next;
	t_block prev;
	int free;
	void * ptr; //a pointer to the allocated block
	char data[1];
}

t_block get_block(void* p)
{
	char* tmp;
	tmp = p;
	return (p= tmp -= BLOCK_SIZE);
}
int valid_addr(void* p)
{
	if( base)
	{
		if (p > base && p < sbrk(0))
		{
			return (p == (get_block(p))->ptr);
		}
	}
	return 0;
}
```
### free 函数
free函数要做的就是校验指针的合理性，释放相应的位置。当然，如果可能的话，可以消除一下内存碎片也是极好的。

* If the pointer is valid:
	* we get the block address
	* we mark it free
	*  if the previous exists and is free, we step backward in the block list and fusion the
two blocks.
	* we also try fusion with then next block
	*  if we’re the last block we release memory.
	*   if there’s no more block, we go back the original state (set base to NULL.)
* If the pointer is not valid, we silently do nothing.

```cpp
void free(void *p)
{
	t_block b;
	if (valid_addr(p))
	{
		b = get_block(p);
		b->free = 1;
		/* fusion with previous if possible */
		if(b->prev && b->prev ->free)
			b = fusion(b->prev);
		/* then fusion with next */
		if (b->next)
			fusion(b);
		else
		{
			/* free the end of the heap */
			if (b->prev)
			b->prev ->next = NULL;
			else
			/* No more block !*/
			base = NULL;
			brk(b);
		}

```

## 综合

```
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

typedef struct s_block* t_block;
struct s_block
{
	size_t size;
	t_block next;
	t_block prev;
	int free;
	void * ptr;
	char data[1];
}
// use for bit align
#define align4(x) (((((x) -1) >> 2) << 2) + 4)


#define BLOCK_SIZE 12
//Because we 
//#define BLOCK_SIZE sizeof(struct s_block)

t_block base = NULL; // the starting point of our process heap
t_block find_block(t_block* last, size_t size);
t_block extend_heap(t_block last, size_t s);
void split_block(t_block b, size_t s);
void* malloc(size_t size);
void free(void *p);

int main()
{
	int count, *array;
	if((array = (int*)malloc(10*sizeof(int))) == NULL)
	{
		printf("Cannot alloct memory space!");
		exit(1);
	}
	for( count = 0; count < 10; count ++)
	{
		array[count] = count;
	}
	for( count = 0; count < 10; count ++)
		printf("%2d",array[count]);
        printf("\n");	
	malloc(array);
	count = 0;	
	while(++count < 1000)
	{
		printf("%d time allocation\n", count);
		if((array = (int*)malloc(10*1024*1024)) == NULL)
		{
			printf("Alloca memory failed");
		}
		free(array);
	}
	return 1;
}


void *malloc(size_t size)
{
	t_block b,last;
	size_t s;
	s = align4(size);
	if (base) 
	{
		/* First find a block */
		last = base;
		b = find_block(&last ,s);
		if (b) 
		{
			/* can we split */
			if ((b->size - s) >= (BLOCK_SIZE + 4))
			split_block(b,s);
			b->free=0;
		} 
		else 
		{
			/* No fitting block , extend the heap */
			b = extend_heap(last ,s);
			if (!b)
			return(NULL);
		}
	} else {
		/* first time */
		b = extend_heap(NULL ,s);
		if (!b)
		return(NULL);
		base = b;
	}
	return(b->data);
}

t_block find_block(t_block* last, size_t size)
{
	t_block b = base;
	while(b && !(b->free && b->size >= size))
	{
		*last = b;
		b = b->next;
	}
	return b;
}
t_block extend_heap(t_block last, size_t s)
{
	t_block b;
	b = sbrk(0);
	if(sbrk(BLOCK_SIZE + s) == (void*)-1)
	{
		return NULL;
	}
	b->size = s;
	b->next = NULL;
	if(last)
		last->next = b;
	b->free = 0; //means block b have been used by process
	return b;
}
void split_block(t_block b, size_t s)
{
	t_block new;
	new = b->data + s;
	new->size = b->size - s - BLOCK_SIZE;
	new->next = b->next;
	new->free = 1;
	b->size = s;
	b->next = new;
}


///////////////////////////////////////////////////////////////////////
t_block fusion(t_block b)
{
	if(b->next && b->next->free)
	{
		b->size += BLOCK_SIZE + b->next->size;
		b->next = b->next->next;
		if(b->next)
		{
			b->next->prev = b;
		}
		return b;
	}
}
t_block get_block(void* p)
{
	char* tmp;
	tmp = p;
	return (p= tmp -= BLOCK_SIZE);
}
int valid_addr(void* p)
{
	if( base)
	{
		if (p > base && p < sbrk(0))
		{
			return (p == (get_block(p))->ptr);
		}
	}
	return 0;
}
/* The free */
/* See free(3) */
void free(void *p)
{
	t_block b;
	if (valid_addr(p))
	{
		b = get_block(p);
		b->free = 1;
		/* fusion with previous if possible */
		if(b->prev && b->prev ->free)
			b = fusion(b->prev);
		/* then fusion with next */
		if (b->next)
			fusion(b);
		else
		{
			/* free the end of the heap */
			if (b->prev)
			b->prev ->next = NULL;
			else
			/* No more block !*/
			base = NULL;
			brk(b);
		}
	}
}

```

此外，还有一个简单的实现方案，参见[my_malloc](https://github.com/cwlseu/Algorithm/blob/master/cppbasic/malloc/my_malloc.cpp)