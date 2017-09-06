---
layout: post
title: 论文编辑-Latex
categories: [blog ]
tags: [工具 ]
description: 最近在写论文，用到latex进行编写，latex真是编辑神器，科研工作者 必备啊，在这向唐纳德先生致敬。
---

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## 引言

最近在写论文，用到latex进行编写，latex真是编辑神器，科研工作者 必备啊，在这向唐纳德*克努斯先生致敬。当年先生写《计算机程序设计的艺术》，感觉这排版工具用得太不爽了，就手撸一把,Tex就被撸出来了，而且版本号为数字PI的截断表示，听着更牛掰的是悬赏bug, 指数级悬赏bug，还带签名哦。

## 环境搭建
安装采用[CTEX]<http://www.ctex.org/CTeXDownload> 或者其他版本的都行，编辑采用Sublime Text 3， 同时安装Markdown Preview、MarkdownEditing、MarkdownHightlighting等插件，让我编辑Markdown的过程中感觉很舒适。

## Latex简介

```latex
\documentclass[12pt]{article} %声明要使用的类型为article，或者book
\usepackage{xxxx}
\begin{chinesetitle}
\title{Latex入门}
\author{cwlseu}
\date{}
\end{chinesetitle}
\begin{document}
	\maketitle
	\begin{abstract}
This is abstract
\end{abstract}
\section{Introduction}
	\subsection{Research In China}
	\subsection{Research In American}
\section{Method}
	\subsection{Assumption}
	\subsection{Deductive}
	\subsection{Objective}
\section{Experience}
	\subsection{DataSet}
	\subsection{TrainResult}
	\subsection{xxxx}
\section{Discussion}
	\subsection{No}
\begin{acknowledgement}

\end{acknowledgement}
\begin{Reference}
\end{Reference}
这是Latex册是测试，虽然当前不支持Chinese Charater
\end{document}
```

### 注释
1.  % 注释，如果想显示%，则 \%
2.  Document Class 文件类型
	Predefined Formats(article, report, book)
3. \footnode{下角标注释}
4. \newpage
5. 表格 

```latex
\documentclass{article}

\usepackage{booktabs} % Allows the use of \toprule, \midrule and \bottomrule in tables for horizontal lines

\begin{document}

\begin{table}
    \centering
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{rl|ccccccccc}
    \hline
    \hline
    \multicolumn{2}{c|}{Component Name} &\multicolumn{9}{c}{StairsNet321} \\
    \hline
    &Deconvolution& & \ding{51} & & & \ding{51} & \ding{51} & \ding{51} & \ding{51} &\ding{51}\\
    Down & Inception\_v4& & & & & \ding{51}           &           &           & \ding{51} &\ding{51}\\
    \multirow{2}{*}{Top}&Inception\_v4 & & & \ding{51} & &           & \ding{51} &  & \ding{51}&\\
    & Inception\_Res\_v2 & & & & \ding{51} &           &           & \ding{51} &           &\ding{51}\\
    \hline
    \multicolumn{2}{c|}{VOC2007 \texttt{test} mAP} & & ?? & ??  & ?? & ??  & ??  &    ??  &   ?? &\textbf{??}\\
    \hline
    \end{tabular}
    \caption{\textbf{Effects of various design choices and components on StairsNet performance.}}
    \label{tab:modelanalysis}
\end{table}

A reference to Table \ref{tab:template}.

\end{document}
```
![@table](../images/latex/table2.png)

6. 编写公式

```latex
    \begin{align}
    & \hat{g}^{cx}_j =\frac{(g_j^{cx} - d_i^{cx})}{d_i^w} & \hat{g}^{cy}_j =\frac{(g_j^{cy} - d_i^{cy})}{d_i^h} & \\
    & \h pre
    at{g}^{w}_j =\log(\frac{g_j^w}{d_i^w}) & \hat{g}^{h}_j =\log(\frac{g_j^h}{d_i^h}) & 
    \end{align}
 ```
公式显示结果
![@公式显示结果](../images/latex/math.png)

更多参考信息[Mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

## 一些经验总结

### 双栏带编号的公式
```latex
\begin{align}
& \hat{g}^{cx}_j =\frac{(g_j^{cx} - d_i^{cx})}{d_i^w} & \hat{g}^{cy}_j =\frac{(g_j^{cy} - d_i^{cy})}{d_i^h} & \\
& \hat{g}^{w}_j =\log(\frac{g_j^w}{d_i^w}) & \hat{g}^{h}_j =\log(\frac{g_j^h}{d_i^h}) & 
\end{align}
```
显示出来是这样子的![@](../images/latex/align.png)

### label让交叉引用更方便
```latex
\subsection{Training}
\label{sec:training}
或者
\begin{figure}[htb]        
   \center{\includegraphics[width=\textwidth]{images/b.png}}      
   \caption{}
   \label{fig:b}
\end{figure}
```
引用的时候`\ref{fig:b}`或者`\ref{sec:training}`就可以了。

### 符号资料
[Algorithms](https://en.wikibooks.org/wiki/LaTeX/Algorithms)
[Color](https://en.wikibooks.org/wiki/LaTeX/Colors)
[Source Code List](https://en.wikibooks.org/wiki/LaTeX/Source_Code_Listings)

## 下载网址
1. [CTEX](http://www.ctex.org/CTeXDownload)
2. [中科院毕业论文模板](http://www.ctex.org/PackageCASthesis)
3. [IEEE 会议论文模板](http://www.ieee.org/conferences_events/conferences/publishing/templates.html)
4. [Latex 网上book](https://en.wikibooks.org/wiki/LaTeX)
5. [数学符号](https://en.wikibooks.org/wiki/LaTeX/Mathematics#List_of_Mathematical_Symbols)