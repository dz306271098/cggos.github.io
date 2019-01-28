---
layout: post
title:  "矩阵分解与线性方程组"
date:   2019-01-28
categories: Math
tags: [Matrix, Least-squares Minimization]
---

[TOC]

# Matrix

## 矩阵类型

* 正规矩阵 $A^* A = AA^* {}$
* 酉矩阵 $U^* U = U U^* = I_n$
* 正交矩阵
* 对角阵
* 三角阵
* 正定阵

## 正定与半正定矩阵

在线性代数里，正定矩阵(positive definite matrix)有时简称为 **正定阵**。

对任意的非零向量 $\boldsymbol{x}$ 恒有 **二次型**  

$$
f = \boldsymbol{x}^T \boldsymbol{A} \boldsymbol{x} > 0
$$

则称 $f$ 为 **正定二次型**，对应的矩阵为 **正定矩阵**；若 $f \ge 0$，则 对应的矩阵为 **半正定矩阵**。


### 直观理解

令 $Y=MX$，则 $X^T Y > 0$，所以  

$$
cos(\theta) = \frac{X^T Y}{\|X\|\|Y\|} > 0
$$

因此，从另一个角度，正定矩阵代表一个向量经过它的变化后的向量与其本身的夹角 **小于90度**

### 判别对称矩阵A的正定性

* 求出A的所有特征值。若A的特征值均为正数，则A是正定的；若A的特征值均为负数，则A为负定的。
* 计算A的各阶顺序主子式。若A的各阶顺序主子式均大于零，则A是正定的；若A的各阶顺序主子式中，奇数阶主子式为负，偶数阶为正，则A为负定的。

# Matrix Decomposition

## EVD (Eigen Decomposition)

$$
A = VDV^{-1}
$$

* $A$ 是 **方阵**；$D$ 是 **对角阵**，其 **特征值从大到小排列**；$V$ 的列向量为 **特征向量**
* 若 $A$ 为 **对称阵**，则 $V$ 为 **正交矩阵**，其列向量为 $A$ 的 **单位正交特征向量**

## SVD (Singular Value Decomposition)

<div align=center>
  <img src="../images/maths/svd.jpg">
</div>

$$
A = UDV^T
$$

* $A$ 为 $m \times n$ 的矩阵；$D$ 是 **非负对角阵**，其 **奇异值从大到小排列**；$U$、$V$ 均为 **正交矩阵**

SVD分解十分强大且适用，因为任意一个矩阵都可以实现SVD分解，而特征值分解只能应用于方阵。

### 奇异值与特征值

$$
AV = UD \Rightarrow Av_i = \sigma_{i} u_i \Rightarrow
\sigma_{i} = \frac{Av_i}{u_i} \\[4ex]
A^T A = (V D^T U^T) (U D V^T) = V D^2 V^T \\[2ex]
A A^T = (U D V^T) (V D^T U^T) = U D^2 U^T
$$

* $A^T A$ 的 **特征向量** 组成的是SVD中的 $V$ 矩阵
* $A^T A$ 的 **特征值** $\lambda_i$ 与 $A$ 的 **奇异值** $\sigma_i$ 满足 $\sigma_i = \sqrt{\lambda_i}$

### PCA

**奇异值减少得特别快**，在很多情况下，**前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上**，可以用 **最大的 $k$ 个的奇异值和对应的左右奇异向量** 来近似描述矩阵  

$$
A_{m \times n} = U_{m \times m} D_{m \times n} V_{n \times n}^T
\approx U_{m \times k} D_{k \times k} V_{k \times n}^T
$$


## LU Decomposition

$$
A = LU
$$

* $A$ 是 **方阵**；$L$ 是 **下三角矩阵**；$U$ 是 **上三角矩阵**

### PLU 分解

$$
A = PLU
$$

事实上，PLU 分解有很高的数值稳定性，因此实用上是很好用的工具。

### LDU 分解

$$
A = LDU
$$

## Cholesky Decomposition

$$
A = LDL^T
$$

* $A$ 是 **方阵**，**正定矩阵**；$L$ 是 **下三角矩阵**

classic:

$$
A = LL^T \\[2ex]
A^{-1} = (L^T)^{-1} L^{-1} = (L^{-1})^T L^{-1}
$$

## QR Decomposition

$$
A = QR
$$

* $A$ 为 $m \times n$ 的矩阵；$Q$ 为 **酉矩阵**；$R$ 是 **上三角矩阵**


# 线性方程组

## 非齐次线性方程组

$$
A_{m \times n} x = b_{m \times 1}
$$

在非齐次方程组中，A到底有没有解析解，可以由增广矩阵来判断：

* r(A) > r(A | b) 不可能，因为增广矩阵的秩大于等于系数矩阵的秩
* r(A) < r(A | b) 方程组无解；
* r(A) = r(A | b) = n，方程组有唯一解；
* r(A) = r(A | b) < n，方程组无穷解；

### 非齐次线性方程组的最小二乘问题

$$
\min{\|Ax - b\|}_2^2
$$

m个方程求解n个未知数，有三种情况：

* m=n，且A为非奇异，则有唯一解 $x=A^{-1}b$
* m>n，**超定问题（overdetermined）**
* m<n，**负定/欠定问题（underdetermined）**

通常我们遇到的都是 **超定问题**，此时 $Ax=b$ 的解是不存在的，从而转向 **解最小二乘问题 $J(x)=\|Ax-b\|$**；$J(x)$ 为 凸函数，一阶导数为0，得到 $A^{T}Ax-A^{T}b=0$，称之为 **正规方程**，得到解 $x=(A^{T}A)^{-1}A^{T}b$

## 齐次线性方程组

$$
A_{m \times n} x = 0
$$

**齐次线性方程 解空间维数=n-r(A)**

* r(A) = n
  - A 是方阵，该方程组有唯一的零解
  - A 不是方阵(m>n)，解空间只含有零向量
* r(A) < n
  - 该齐次线性方程组有非零解，而且不唯一（自由度为n-r(A))
* r(A) > n
  - 需要求解 **最小二乘解**，在 **$\|x\|=1$** 的约束下，其最小二乘解为 **矩阵 $A^TA$ 最小特征值所对应的特征向量**

### 齐次线性方程组的最小二乘问题

$$
\min{\|Ax\|}_2^2 \quad s.t. \quad \|x\| = 1
$$

* 最小二乘解为 **矩阵 $A^TA$ 最小特征值所对应的特征向量**
* $eig(A^{T}A)=[V,D]$，找最小特征值对应的V中的特征向量
* $svd(A)=[U,S,V]$，找S中最小奇异值对应的V的右奇异向量
