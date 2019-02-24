---
layout: post
title: 探究Linux的奇妙之旅
categories: [blog ]
tags: [linux, 操作系统]
description: 探究linux 0.11的一些神奇的东西，课程教师杨立祥，一个成功的实现财富自由的男人。
---

{:toc}

## linux设计的作者

[Linus](https://en.wikipedia.org/wiki/Linus_Torvalds)
[Linux官网](http://www.linux.org/) 


## 理解schedule函数的调用

process 0 第一次调度和后面调度，schedule运行的机制是不同的。

```cpp
	void main(void)		/* This really IS void, no error here. */
	{			
	......
	/*
	 *   NOTE!!   For any other task 'pause()' would mean we have to get a
	 * signal to awaken, but task0 is the sole exception (see 'schedule()')
	 * as task 0 gets activated at every idle moment (when no other tasks
	 * can run). For task0 'pause()' just means we go check if some other
	 * task can run, and if not we return here.
	 */
		for(;;) pause();
	}
```

当进程0到达main函数的最后的循环之后，将通过系统调用，调用系统函数_sys_pause，其中调用schedule，

```cpp	
	int sys_pause(void)
	{
		current->state = TASK_INTERRUPTIBLE;
		schedule();
		return 0;
	}
```

查找到刚刚创建的进程1处于就绪状态，在schedule最后switch_to(1). 在switch中发生了什么呢？

```cpp	
	/*
	 *  'schedule()' is the scheduler function. This is GOOD CODE! There
	 * probably won't be any reason to change this, as it should work well
	 * in all circumstances (ie gives IO-bound processes good response etc).
	 * The one thing you might take a look at is the signal-handler code here.
	 *
	 *   NOTE!!  Task 0 is the 'idle' task, which gets called when no other
	 * tasks can run. It can not be killed, and it cannot sleep. The 'state'
	 * information in task[0] is never used.
	 */
	void schedule(void)
	{
		int i,next,c;
		struct task_struct ** p;

	/* check alarm, wake up any interruptible tasks that have got a signal */

		for(p = &LAST_TASK ; p > &FIRST_TASK ; --p)
			if (*p) {
				if ((*p)->alarm && (*p)->alarm < jiffies) {
						(*p)->signal |= (1<<(SIGALRM-1));
						(*p)->alarm = 0;
					}
				if (((*p)->signal & ~(_BLOCKABLE & (*p)->blocked)) &&
				(*p)->state==TASK_INTERRUPTIBLE)
					(*p)->state=TASK_RUNNING;
			}

	/* this is the scheduler proper: */

		while (1) {
			c = -1;
			next = 0;
			i = NR_TASKS;
			p = &task[NR_TASKS];
			while (--i) {
				if (!*--p)
					continue;
				if ((*p)->state == TASK_RUNNING && (*p)->counter > c)
					c = (*p)->counter, next = i;
			}
			if (c) break;
			for(p = &LAST_TASK ; p > &FIRST_TASK ; --p)
				if (*p)
					(*p)->counter = ((*p)->counter >> 1) +
							(*p)->priority;
		}
		switch_to(next);
	}
```

`ljmp 0 \n\t`这个得查看IA-32手册，就会发现，IA-32架构中允许同种特权级之间相互切换，但是不同特权级之间切换是不允许的。但是0特权级的进程0是如何切换到3特权级的进程1呢？就要引入'门'的概念了。具体参见[手册](http://www.intel.cn/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3a-part-1-manual.pdf).门是允许点到点的，不同特权级之间的切换。

那么当进程1运行过之后，再次由进程0切换到进程1时，由于进程切换都是0特权级，此时，此时就不需要经过什么门了，IA32是允许同特权级之间进行进程切换的。

## switch_to

```
	#define switch_to(n) {\
	struct {long a,b;} __tmp; \
	__asm__("cmpl %%ecx,_current\n\t" \
		"je 1f\n\t" \
		"movw %%dx,%1\n\t" \
		"xchgl %%ecx,_current\n\t" \
		"ljmp %0\n\t" \
		"cmpl %%ecx,_last_task_used_math\n\t" \
		"jne 1f\n\t" \
		"clts\n" \
		"1:" \
		::"m" (*&__tmp.a),"m" (*&__tmp.b), \
		"d" (_TSS(n)),"c" ((long) task[n])); \
	}
```

这里涉及到TSS的定义和GDT LDT管理结构

	#define _TSS(n) ((((unsigned long) n)<<4)+(FIRST_TSS_ENTRY<<3))
	#define FIRST_TSS_ENTRY 4

每个描述符占8个字节，第一个状态段是第四个，所以 <<3，得到第一个任务描述符在GDT中的位置，
而每个任务使用一个tss和ldt，占16字节，所以<<4，两者相加得到任务n的tss在GDT中的位置,写入EDX寄存器。
另外ECX指向要切换过去的新任务task[n]。


现在开始理解代码，首先声明了一个_tmp的结构，这个结构里面包括两个long型，32位机里面long占32位，声明这个结构主要与ljmp这个长跳指令有关，这个指令有两个参数，一个参数是段选择符，另一个是偏移地址，所以这个_tmp就是保存这两个参数。再比较任务n是不是当前任务，如果不是则跳转到标号1，否则交互ecx和current的内容，交换后的结果为ecx指向当前进程，current指向要切换过去的新进程，在执行长跳，%0代表输出输入寄存器列表中使用的第一个寄存器，即"m"(*&__tmp.a)，这个寄存器保存了*&__tmp.a，而_tmp.a存放的是32位偏移(对应EIP)，_tmp.b存放的是新任务的tss段选择符(对应CS)，长跳到段选择符会造成任务切换，这个是x86的硬件原理。"d" (_TSS(n)),"c" ((long) task[n])); 

## 缓冲区

Linux0.11 中没有实现预读和预写功能,而是将预读功能转化为普通读取。
其中缓冲区的理解要找到如下数据结构：

1. buffer head

```cpp
	struct buffer_head {
		char * b_data;			/* pointer to data block (1024 bytes) */
		unsigned long b_blocknr;	/* block number */
		unsigned short b_dev;		/* device (0 = free) */
		unsigned char b_uptodate;
		unsigned char b_dirt;		/* 0-clean,1-dirty */
		unsigned char b_count;		/* users using this block */
		unsigned char b_lock;		/* 0 - ok, 1 -locked */
		struct task_struct * b_wait;
		struct buffer_head * b_prev;
		struct buffer_head * b_next;
		struct buffer_head * b_prev_free;
		struct buffer_head * b_next_free;
	};
```

2. hash_table

```cpp
	struct buffer_head * hash_table[NR_HASH];
	#define _hashfn(dev,block) (((unsigned)(dev^block))%NR_HASH)
	#define hash(dev,block) hash_table[_hashfn(dev,block)]
```

3. request

```cpp
	/*
	 * Ok, this is an expanded form so that we can use the same
	 * request for paging requests when that is implemented. In
	 * paging, 'bh' is NULL, and 'waiting' is used to wait for
	 * read/write completion.
	 */
	struct request {
		int dev;		/* -1 if no request */
		int cmd;		/* READ or WRITE */
		int errors;
		unsigned long sector;
		unsigned long nr_sectors;
		char * buffer;
		struct task_struct * waiting;
		struct buffer_head * bh;
		struct request * next;
	};
```

### READA and WRITEA
```cpp
	static void make_request(int major,int rw, struct buffer_head * bh)
	{
		......
		/* WRITEA/READA is special case - it is not really needed, so if the */
		/* buffer is locked, we just forget about it, else it's a normal read */
		if (rw_ahead = (rw == READA || rw == WRITEA)) {
			if (bh->b_lock)
				return;
			if (rw == READA)
				rw = READ;
			else
				rw = WRITE;
		}
		if (rw!=READ && rw!=WRITE)
			panic("Bad block dev command, must be R/W/RA/WA");
		lock_buffer(bh);
		if ((rw == WRITE && !bh->b_dirt) || (rw == READ && bh->b_uptodate)) {
			unlock_buffer(bh);
			return;
		}
		......
	}
```

那么什么是预读预写呢？参考[Linux内核的文件预读](http://github.com/cwlseu/cwlseu.github.io/raw/master/pdf/linux_read_ahead.pdf)

## 系统调用

```cpp
	fn_ptr sys_call_table[] = { sys_setup, sys_exit, sys_fork, sys_read,
	sys_write, sys_open, sys_close, sys_waitpid, sys_creat, sys_link,
	sys_unlink, sys_execve, sys_chdir, sys_time, sys_mknod, sys_chmod,
	sys_chown, sys_break, sys_stat, sys_lseek, sys_getpid, sys_mount,
	sys_umount, sys_setuid, sys_getuid, sys_stime, sys_ptrace, sys_alarm,
	sys_fstat, sys_pause, sys_utime, sys_stty, sys_gtty, sys_access,
	sys_nice, sys_ftime, sys_sync, sys_kill, sys_rename, sys_mkdir,
	sys_rmdir, sys_dup, sys_pipe, sys_times, sys_prof, sys_brk, sys_setgid,
	sys_getgid, sys_signal, sys_geteuid, sys_getegid, sys_acct, sys_phys,
	sys_lock, sys_ioctl, sys_fcntl, sys_mpx, sys_setpgid, sys_ulimit,
	sys_uname, sys_umask, sys_chroot, sys_ustat, sys_dup2, sys_getppid,
	sys_getpgrp, sys_setsid, sys_sigaction, sys_sgetmask, sys_ssetmask,
	sys_setreuid,sys_setregid };
```

## 重新编译替换内核
先看一下当前linux的版本号`uname -a`
下载linux某个版本的linux内核源代码，如3.19.8,将源代码解压到/usr/src/目录下。

```sh
cd /usr/src/linux-3.19.8
# compile
make 
# compile moudles
make modules
# modules install, all the modules will be set /boot/ directories
make modules_install
# install kernel, and move the kernel to /boot.. update grub file
make install
```

其中要设置一番，设置过程可以参考[博客](http://blog.sina.com.cn/s/blog_4b14d8190100muj3.html)或者默认就好了。
最后安装完毕重启，重新执行`uname -a`查看是否内核版本已经更新为你编译安装的版本。

## 参考链接
1. 新设计团队. Linux内核设计的艺术[M]. 机械工业出版社, 2011.