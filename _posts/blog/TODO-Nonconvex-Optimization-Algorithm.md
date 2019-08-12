## PG
- paper: https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

近端梯度法（Proximal Gradient Method ，PG）是一种特殊的梯度下降方法，主要用于求解目标函数不可微的最优化问题。如果目标函数在某些点是不可微的，那么该点的梯度无法求解，传统的梯度下降法也就无法使用。PG算法的思想是，使用临近算子作为近似梯度，进行梯度下降。

[^1]: https://blog.csdn.net/Chaolei3/article/details/81320940
