---
layout: post
title: 探究Linux的奇妙之旅
categories: [blog ]
tags: [Linux, 操作系统, OS]
description: 探究linux 0.11的一些神奇的东西
---


##理解schedule函数的调用
process 0 第一次调度和后面调度，schedule运行的机制是不同的。

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

当进程0到达main函数的最后的循环之后，将通过系统调用，调用系统函数_sys_pause，其中调用schedule，
	
	int sys_pause(void)
	{
		current->state = TASK_INTERRUPTIBLE;
		schedule();
		return 0;
	}

查找到刚刚创建的进程1处于就绪状态，在schedule最后switch_to(1). 在switch中发生了什么呢？
	
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


`ljmp 0 \n\t`这个得查看IA-32手册，就会发现，IA-32架构中允许同种特权级之间相互切换，但是不同特权级之间切换是不允许的。但是0特权级的进程0是如何切换到3特权级的进程1呢？就要引入'门'的概念了。具体参见[手册](http://www.intel.cn/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3a-part-1-manual.pdf).门是允许点到点的，不同特权级之间的切换。

那么当进程1运行过之后，再次由进程0切换到进程1时，由于进程切换都是0特权级，此时，此时就不需要经过什么门了，IA32是允许同特权级之间进行进程切换的。