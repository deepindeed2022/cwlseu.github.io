---
layout: post
title: "笔记：OpenCV中的算法--透视和仿射变换"
categories: [blog ]
tags: [图像处理]
description: 图像处理的基本概念与算法
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/> 


# 图像处理

## warpPerspective 和 affineTransform的转换矩阵的区别

1. affineTransform保持平行性，而warpPerspective不能保证
2. warpPerspective至少4个点对，而 affineTransform至少三个点对
下面是opencv中关于这两个变换矩阵的求解过程。

```cpp

/* Calculates coefficients of perspective transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0 1 0 0 0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1 1 0 0 0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2 1 0 0 0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3 1 0 0 0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * | 0 0 0 x0 y0 1 -x0*v0 -y0*v0 | |c11| |v0|
 * | 0 0 0 x1 y1 1 -x1*v1 -y1*v1 | |c12| |v1|
 * | 0 0 0 x2 y2 1 -x2*v2 -y2*v2 | |c20| |v2|
 * \ 0 0 0 x3 y3 1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 * cij - matrix coefficients, c22 = 1
 */
cv::Mat cv::getPerspectiveTransform( const Point2f src[], const Point2f dst[] )
{
  Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.data);
  double a[8][8], b[8];
  Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

  for( int i = 0; i < 4; ++i )
  {
  a[i][0] = a[i+4][3] = src[i].x;
  a[i][1] = a[i+4][4] = src[i].y;
  a[i][2] = a[i+4][5] = 1;
  a[i][3] = a[i][4] = a[i][5] =
  a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
  a[i][6] = -src[i].x*dst[i].x;
  a[i][7] = -src[i].y*dst[i].x;
  a[i+4][6] = -src[i].x*dst[i].y;
  a[i+4][7] = -src[i].y*dst[i].y;
  b[i] = dst[i].x;
  b[i+4] = dst[i].y;
  }

  solve( A, B, X, DECOMP_SVD );
  ((double*)M.data)[8] = 1.;

  return M;
}


/* Calculates coefficients of affine transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0 1 0 0 0 \ /c00\ /u0\
 * | x1 y1 1 0 0 0 | |c01| |u1|
 * | x2 y2 1 0 0 0 | |c02| |u2|
 * | 0 0 0 x0 y0 1 | |c10| |v0|
 * | 0 0 0 x1 y1 1 | |c11| |v1|
 * \ 0 0 0 x2 y2 1 / |c12| |v2|
 *
 * where:
 * cij - matrix coefficients
 */

cv::Mat cv::getAffineTransform( const Point2f src[], const Point2f dst[] )
{
  Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.data);
  double a[6*6], b[6];
  Mat A(6, 6, CV_64F, a), B(6, 1, CV_64F, b);

  for( int i = 0; i < 3; i++ )
  {
  int j = i*12;
  int k = i*12+6;
  a[j] = a[k+3] = src[i].x;
  a[j+1] = a[k+4] = src[i].y;
  a[j+2] = a[k+5] = 1;

  a[j+3] = a[j+4] = a[j+5] = 0;
  a[k] = a[k+1] = a[k+2] = 0;

  b[i*2] = dst[i].x;
  b[i*2+1] = dst[i].y;
  }

  solve( A, B, X );
  return M;
}
```

如果我们要自己实现这个函数，其实关键就是在于如何求解AX=B的问题。当然，我们可以直接调用库函数，如`eigen`.

### 问题：这个函数如果要自己实现，如何测试正确性？

* 方案1：
采用引入opencv作为第三方库，然后相同的输入结果与opencv中进行对比。这种方法简单，但是需要引入庞大的第三方库opencv

* 方案2：
采用两次变换，例如测试`warp_perspective`其中第一次将`src_img`经过`warp_perspective`变换为`dst_img`，其中转换矩阵为M;
然后将`dst_img`经过`warp_perspective`变换为`dst_warp`，其中转换矩阵为`M‘`为`M`的逆矩阵;
最后比较`dst_warp`和`src`中进行逐个像素对照，统计diff的像素个数count， return count <= thresh_value.
这种方法的缺点就是需要设置thresh_value，同时需要求M的逆矩阵

* 方案3：
如果不能将opencv作为第三方库引入，那么我们可以这样，将opencv的输入参数和结果作为hard code的方式，进行测试。这种方法尤其是
再嵌入式开发中很常见。


## 更多信息可以参考

[1] https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html

[2] gTest的原理： http://cwlseu.github.io/st-CMAKE-and-gTest/搜索
