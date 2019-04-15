---
layout: post
title: 工具：Latex入门教程
categories: [blog ]
tags: [工具]
description: 最近在写论文，用到latex进行编写，latex真是编辑神器，科研工作者 必备啊，在这向唐纳德先生致敬。
---
* content
{:toc}

## 引言

最近在写论文，用到latex进行编写，latex真是编辑神器，科研工作者 必备啊，在这向唐纳德*克努斯先生致敬。当年先生写《计算机程序设计的艺术》，感觉这排版工具用得太不爽了，就手撸一把,Tex就被撸出来了，而且版本号为数字PI的截断表示，听着更牛掰的是悬赏bug, 指数级悬赏bug，还带签名哦。

## 环境搭建

安装采用[CTEX]<http://www.ctex.org/CTeXDownload> 或者其他版本的都行，编辑采用Sublime Text 3， 同时安装Markdown Preview、MarkdownEditing、MarkdownHightlighting等插件，让我编辑Markdown的过程中感觉很舒适。

## Latex简介

```tex
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

```tex
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
![@table](https://cwlseu.github.io/images/latex/table2.png)

6. 编写公式

```tex
    \begin{align}
    & \hat{g}^{cx}_j =\frac{(g_j^{cx} - d_i^{cx})}{d_i^w} & \hat{g}^{cy}_j =\frac{(g_j^{cy} - d_i^{cy})}{d_i^h} & \\
    & \h pre
    at{g}^{w}_j =\log(\frac{g_j^w}{d_i^w}) & \hat{g}^{h}_j =\log(\frac{g_j^h}{d_i^h}) & 
    \end{align}
```
 
公式显示结果
![@公式显示结果](https://cwlseu.github.io/images/latex/align.png)

更多参考信息[Mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

## 一些经验总结

### label让交叉引用更方便

```tex
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


### 算法模块

```tex
\usepackage[ruled]{algorithm2e}         
\usepackage[ruled,vlined]{algorithm2e} 
\usepackage[linesnumbered,boxed]{algorithm2e}
```

```tex
\renewcommand{\algorithmcfname}{算法}
\begin{algorithm}[H]
%\SetAlgoNoLine
\SetKwInOut{KIN}{输入}
\SetKwInOut{KOUT}{输出}
%\BlankLine  %空一行
    \caption{标准DE算法 }
    \label{DE_algo} %
    \KIN{Population: $ M$; Dimension: $ D $; Genetation: $ T $ }
    \KOUT{The best vector (solution)  $ \varDelta $ }
    $ t \leftarrow 1 (initialization) $\;
    \For{$i=1$ to $ M $    }
    {\For{$j=1$ to $ D $}
        {
            $  {x}_{i,t}^j=x_{min}^j + rand(0,1)\cdotp (x_{max}^j-x_{min}^j) $\;
        }
    }        
    %--------------------------------------------------    
\While{$(|f(\varDelta)| \geq\varepsilon )$      or     $(t \leq T )$}
    {
        \For{$ i=1$  to $M$}
        {
\emph{$ \blacktriangleright $ (Mutation and Crossover)}\\            
%\textit{ $ \blacktriangleright $ (Mutation and Crossover) }\\
            \For{$j=1$ to $ D $}
            {    
                $ v_{i,t}^j =Mutation(x_{i,t}^j)$\;    
                $ u_{i,t}^j =Crossover(x_{i,t}^j,v_{i,t}^j)$\;
            }
\emph{$ \blacktriangleright $ (Greedy Selection)}\\
            %\textit{ $ \blacktriangleright $ (Greedy Selection) }\\
            \eIf{$ f(\textbf{u}_{i,t}) <  f(\textbf{x}_{i,t}) $}
            {
                $  \textbf{x}_{i,t} \leftarrow\textbf{u}_{i,t}$\;    
                \If{$  f(\textbf{x}_{i,t}) < f(\varDelta)$}
                {
                    $ \varDelta \leftarrow \textbf{x}_{i,t}$ \;
                }
            }
            {
                $  \textbf{x}_{i,t} \leftarrow \textbf{x}_{i,t} $\;
            }
        }
        $ t \leftarrow t+1 $\;    
    }    %While        
    \Return the best vector  $\varDelta$\;
\end{algorithm}
```

```tex
\documentclass[8pt,a4paper,twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage{algorithm2e}
% \usepackage{algorithm}  
\usepackage{algorithmic} 

\begin{document}

\begin{algorithm}
	\caption{The algorithm of GetAllColumnIndices.}%算法标题
	\begin{algorithmic}[1]%一行一个标行号
		\REQUIRE{$\mathrm{thread\_start}$, $tid$}
		\ENSURE{$\mathrm{col\_idxs}$}
		\STATE {$ start\leftarrow \mathrm{thread\_start}[tid] + blockColStart$,
		$curPos\leftarrow tid*\mathrm{NNZ\_PER\_THREAD} + bid*\mathrm{NNZ\_PER\_BLOCK}$
		$left\leftarrow \mathrm{NNZ\_PER\_THREAD},len\leftarrow\mathrm{col\_len}[start + 1] - curPos$}
		\STATE $i \leftarrow 0$
		\WHILE{ $i < \mathrm{NNZ\_PER\_THREAD}$ }
		\IF{$len \ge left$}
			\FOR{$j=0$ to $left$}
				\STATE  $\mathrm{col\_idxs}[i$$++$$] \leftarrow start$
			\ENDFOR
		\ELSE
			\FOR{$j=0$ to $len$}
				\STATE  $\mathrm{col\_idxs}[i$$++$$] \leftarrow start$
			\ENDFOR
			\STATE $ start\leftarrow start - 1$, $ left \leftarrow left - len$
  			\STATE $ len \leftarrow \mathrm{col\_len}[start + 1] - curPos$
		\ENDIF
		\ENDWHILE
	\end{algorithmic}
\end{algorithm}

\end{document}



```
### 内部reference出现`???`的问题

```tex
    \begin{figure*}
        \centering
        \includegraphics[width=0.48\linewidth]{../Img/research/AlexNetLayerCompression.jpg}
        \includegraphics[width=0.48\linewidth]{../Img/research/AlexNetLayerCompressionFlops.jpg}
        \caption{AlexNet各网络层参数裁剪前后的参数数量}
        \label{AlexNetCpression}
    \end{figure*}
```

```tex
    \begin{figure*}
        \label{AlexNetCpression}
        \centering
        \includegraphics[width=0.48\linewidth]{../Img/research/AlexNetLayerCompression.jpg}
        \includegraphics[width=0.48\linewidth]{../Img/research/AlexNetLayerCompressionFlops.jpg}
        \caption{AlexNet各网络层参数裁剪前后的参数数量}
    \end{figure*}
```
上面两段代码的区别在于label的位置，这两个代码看上去应该没有什么问题，但是当你`\cite{AlexNetCpression}`的
时候你就会发现，下面那种容易导致出现`cite`的位置是`??`，根本不知道去哪里找原因。真是一个自己挖的好坑啊。

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
6. [LaTeX 算法代码排版 --latex2e范例总结](http://www.cnblogs.com/tsingke/p/5833221.html)

# tikz绘制论文的插图

The error of ' ! I can't find file `tikzlibraryarrows.meta.code.tex'.'
下载新的pgf库 https://ctan.org/pkg/pgf

http://www.texample.net/tikz/examples/

http://Altermundus.com
An impressive collection of various TikZ-related packages and examples.

[Graph Theory in LaTeX](http://graphtheoryinlatex.blogspot.com/)
A gallery of (combinatorial) graphs produced by using LaTeX

用tikz绘制流程图

https://blog.csdn.net/xiahn1a/article/details/46547981

Ubuntu下安装缺失的package

https://yq.aliyun.com/articles/523588
https://tex.stackexchange.com/questions/307933/the-error-of-i-cant-find-file-tikzlibraryarrows-meta-code-tex
