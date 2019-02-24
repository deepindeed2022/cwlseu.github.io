---
layout: post
title: "计算机基础-SQL操作"
categories: [blog ]
tags: [工具]
description: "SQL是数据库操作的基础语言，使用SQL可以说是和数据库交流沟通的工具"
---

{:toc}


 

## SQL查询语句

这部分的来源是因为什么呢，有一次笔试题目中突然出现久违的SQL语句，哎呀，当时大学的时候学得还可以，但是两三年没有碰这个玩意都快忘干净了。赶紧随着题目渐渐地回忆起来相关的一些操作。
SQL语句是操作数据库的语言，迄今为止，很多工作都是在数据库上进行操作分析的。虽然现在SQL的功能一部分被NoSQL的数据库所替代，但是大部分用户信息，客户信息，客户资料等等还是使用关系型数据库进行存储的。
前提：声明几张表

#### emp

|empmo| ename|age| mgr|deptno| job | year_sal|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1| charles | 18 | dept2 |2 | 测试 | 30万|
|1| ben | 28 | dept2 |2 | 开发 | 32万|

#### dept

|deptno | dname|
|:---:|:---:|
|1 | MSS |

#### salgrade 

|grade |losal |hisal|
|:---:|:---:|:---:|
|1 | 300 万| 400 万 |
|2 | 100 万| 300 万 |

### 基础查询

1. 查看表结构  `SQL>DESC emp;` 
2. 查询所有列
    `SQL>SELECT * FROM emp;`
3. 查询指定列

```sql
    SELECT empmo, ename, mgr FROM emp;
    SELECT DISTINCT mgr FROM emp; # 只显示结果不同的
```
4. 查询指定行

```sql
    SELECT * FROM emp WHERE job='CLERK';
```
5. 使用算术表达式

```sql
    SELECT ename, sal*13+nvl(comm,0)  FROM emp; 
```

`nvl(comm,0)`的意思是，如果comm中有值，则`nvl(comm, 0)=comm;`  `comm`中无值，则`nvl(comm, 0)=0`。

`SELECT ename, sal*13+nvl(comm,0) year_sal FROM emp;`（year_sal为别名，可按别名排序）

`SELECT * FROM emp WHERE hiredate>'01-1月-82';` 

6. 使用like操作符（%，_）

`%`表示一个或多个字符，_表示一个字符，[charlist]表示字符列中的任何单一字符，[^charlist]或者[!charlist]不在字符列中的任何单一字符。

```sql
    SELECT * FROM emp WHERE ename like 'S__T%';
```

7. 在where条件中使用In

```sql
    SELECT * FROM emp WHERE job IN ('CLERK','ANALYST');
```

8. 查询字段内容为空/非空的语句

```sql
SELECT * FROM emp WHERE mgr IS/IS NOT NULL; 
```
9. 使用逻辑操作符号

```sql
SELECT * FROM emp WHERE (sal>500 or job='MANAGE') and ename like 'J%';
```
10. 将查询结果按字段的值进行排序

`SELECT * FROM emp ORDER BY deptno, sal DESC; `(按部门升序，并按薪酬降序)

### 复杂查询

1. 使用统计函数进行数据分组查询操作(max,min,avg,sum,count)

```sql
SELECT MAX(sal),MIN(age),AVG(sal),SUM(sal) from emp;

SELECT * FROM emp where sal=(SELECT MAX(sal) from emp));

SELEC COUNT(*) FROM emp;
```

2. group by（用于对查询结果的分组统计） 和 having子句（用于限制分组显示结果）

```sql
SELECT deptno,MAX(sal),AVG(sal) FROM emp GROUP BY deptno;

SELECT deptno, job, AVG(sal),MIN(sal) FROM emp group by deptno,job having AVG(sal)<2000;
```

>   对于数据分组的总结：
    - 分组函数只能出现在选择列表、having、order by子句中（不能出现在where中）
    - 如果select语句中同时包含有group by, having, order by，那么它们的顺序是group by, having, order by。
    - 在选择列中如果有列、表达式和分组函数，那么这些列和表达式必须出现在group by子句中，否则就是会出错。

**NOTE**: 使用group by不是使用having的前提条件。

3. 多表查询

```sql
SELECT e.name,e.sal,d.dname FROM emp e, dept d WHERE e.deptno=d.deptno order by d.deptno;

SELECT e.ename,e.sal,s.grade FROM emp e, salgrade s WHER e.sal BETWEEN s.losal AND s.hisal;
```

4. **自连接**（指同一张表的连接查询）

```sql
SELECT er.ename, ee.ename mgr_name from emp er, emp ee where er.mgr=ee.empno;
```

5. 子查询（嵌入到其他sql语句中的select语句，也叫嵌套查询）
5.1 单行子查询

` SELECT ename FROM emp WHERE deptno=(SELECT deptno FROM emp where ename='SMITH');`
查询表中与smith同部门的人员名字。因为返回结果只有一行，所以用“=”连接子查询语句

5.2 多行子查询

```sql
SELECT ename,job,sal,deptno from emp WHERE job IN (SELECT DISTINCT job FROM emp WHERE deptno=10);
```

查询表中与部门号为10的工作相同的员工的姓名、工作、薪水、部门号。因为返回结果有多行，所以用“IN”连接子查询语句。

    in与exists的区别： exists() 后面的子查询被称做相关子查询，它是不返回列表的值的。只是返回一个ture或false的结果，其运行方式是先运行主查询一次，再去子查询里查询与其对 应的结果。如果是ture则输出，反之则不输出。再根据主查询中的每一行去子查询里去查询。in()后面的子查询，是返回结果集的，换句话说执行次序和 exists()不一样。子查询先产生结果集，然后主查询再去结果集里去找符合要求的字段列表去。符合要求的输出，反之则不输出。

5.3 使用ALL

```sql
SELECT ename,sal,deptno FROM emp WHERE sal> ALL (SELECT sal FROM emp WHERE deptno=30);
```

或 

```sql
SELECT ename,sal,deptno FROM emp WHERE sal> (SELECT MAX(sal) FROM emp WHERE deptno=30);
```

查询工资比部门号为30号的所有员工工资都高的员工的姓名、薪水和部门号。以上两个语句在功能上是一样的，但执行效率上，函数会高得多。

5.4 使用ANY

`SELECT ename,sal,deptno FROM emp WHERE sal> ANY(SELECT sal FROM emp WHERE deptno=30);` 
或 
`SELECT ename,sal,deptno FROM emp WHERE sal> (SELECT MIN(sal) FROM emp WHERE deptno=30);`
查询工资比部门号为30号的任意一个员工工资高（只要比某一员工工资高即可）的员工的姓名、薪水和部门号。以上两个语句在功能上是 一样的，但执行效率上，函数会高得多。

5.5 多列子查询

```sql
SELECT * FROM emp WHERE (job, deptno)=(SELECT job, deptno FROM emp WHERE ename='SMITH');
```

5.6 在from子句中使用子查询

```sql
SELECT emp.deptno,emp.ename,emp.sal,t_avgsal.avgsal FROM emp,(SELECT emp.deptno,avg(emp.sal) avgsal FROM emp GROUP BY emp.deptno) t_avgsal where emp.deptno=t_avgsal.deptno AND emp.sal>t_avgsal.avgsal ORDER BY emp.deptno;
```
5.7 分页查询
数据库的每行数据都有一个对应的行号，称为rownum.

```sql
SELECT a2.* FROM (SELECT a1.*, ROWNUM rn FROM (SELECT * FROM emp ORDER BY sal) a1 WHERE ROWNUM<=10) a2 WHERE rn>=6;
```

指定查询列、查询结果排序等，都只需要修改最里层的子查询即可。

5.8 用查询结果创建新表

```sql
CREATE TABLE mytable (id,name,sal,job,deptno) AS SELECT empno,ename,sal,job,deptno FROM emp;
```

5.9 合并查询（union 并集, intersect 交集, union all 并集+交集, minus差集)

```sql
SELECT ename, sal, job FROM emp WHERE sal>2500 UNION(INTERSECT/UNION ALL/MINUS) SELECT ename, sal, job FROM emp WHERE job='MANAGER';
```

**NOTE**:合并查询的执行效率远高于and,or等逻辑查询。

5.10 使用子查询插入数据

```sql
CREATE TABLE myEmp(empID number(4), name varchar2(20), sal number(6), job varchar2(10), dept number(2)); ##先建一张空表；

INSERT INTO myEmp(empID, name, sal, job, dept) SELECT empno, ename, sal, job, deptno FROM emp WHERE deptno=10;
# 再将emp表中部门号为10的数据插入到新表myEmp中，实现数据的批量查询。
```
5.11 使用了查询更新表中的数据

```sql
UPDATE emp SET(job, sal, comm)=(SELECT job, sal, comm FROM emp where ename='SMITH') WHERE ename='SCOTT';
```
