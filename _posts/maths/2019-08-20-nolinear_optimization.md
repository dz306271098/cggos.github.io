---
layout: post
title:  "非线性优化"
date:   2019-08-20
categories: Math
tags: []
---

[TOC]

# 最小二乘

损失函数

$$
F(\mathbf{x})=\frac{1}{2} \sum_{i=1}^{m}\left(f_{i}(\mathbf{x})\right)^{2}
=\frac{1}{2} \mathbf{f}^{\top}(\mathbf{x}) \mathbf{f}(\mathbf{x})
$$

其中，$f_i$ 是残差函数。

二阶泰勒展开

$$
F(\mathrm{x}+\Delta \mathrm{x})=F(\mathrm{x})+\mathrm{J} \Delta \mathrm{x}+\frac{1}{2} \Delta \mathrm{x}^{\top} \mathrm{H} \Delta \mathrm{x}+O\left(\|\Delta \mathrm{x}\|^{3}\right)
$$

忽略泰勒展开的高阶项，损失函数变成了二次函数，可以轻易得到如下性质：
* 如果在点 $x_s$ 处有导数为0，则称这个点为稳定点
* 在点 $x_s$ 处对应的 Hessian 为 H
  - 如果是正定矩阵，即它的特征值都大于 0，则在 $x_s$ 处有 F (x) 为 局部最小值；
  - 如果是负定矩阵，即它的特征值都小于 0，则在 $x_s$ 处有 F (x) 为 局部最大值；
  - 如果是不定矩阵，即它的特征值大于 0 也有小于 0 的，则 $x_s$ 处 为鞍点

# 非线性最小二乘

残差函数 $f(x)$ 为非线性函数，对其一阶泰勒近似有

$$
\mathbf{f}(\mathbf{x}+\Delta \mathbf{x}) \approx \ell(\Delta \mathbf{x}) \equiv \mathbf{f}(\mathbf{x})+\mathbf{J} \Delta \mathbf{x}
$$

代入损失函数

$$
\begin{aligned}
F(\mathrm{x}+\Delta \mathrm{x}) \approx L(\Delta \mathrm{x}) & \equiv \frac{1}{2} \ell(\Delta \mathrm{x})^{\top} \ell(\Delta \mathrm{x}) \\
&=\frac{1}{2} \mathrm{f}^{\top} \mathrm{f}+\Delta \mathrm{x}^{\top} \mathrm{J}^{\top} \mathrm{f}+\frac{1}{2} \Delta \mathrm{x}^{\top} \mathrm{J}^{\top} \mathrm{J} \Delta \mathrm{x} \\
&=F(\mathrm{x})+\Delta \mathrm{x}^{\top} \mathrm{J}^{\top} \mathrm{f}+\frac{1}{2} \Delta \mathrm{x}^{\top} \mathrm{J}^{\top} \mathrm{J} \Delta \mathrm{x}
\end{aligned}
$$

这样损失函数就近似成了一个二次函数，并且如果雅克比是满秩的，则 $\mathbf{J}^{\top} \mathbf{J}$ 正定，损失函数有最小值

易得

$$
F^{\prime}(\mathbf{x})=\left(\mathbf{J}^{\top} \mathbf{f}\right)^{\top}
$$

$$
F^{\prime \prime}(\mathbf{x}) \approx \mathbf{J}^{\top} \mathbf{J}
$$

# 一阶梯度法（最速下降法）

梯度的负方向为最速下降方向

$$
\mathbf{d}=\frac{-\mathbf{J}^{\top}}{\|\mathbf{J}\|}
$$

* 适用于迭代的开始阶段
* 缺点：最优值附近震荡，收敛慢

# 二阶梯度法（牛顿法）

$$
\Delta \mathbf{x}=-\mathbf{H}^{-1} \mathbf{J}^{\top}
$$

* 适用于最优值附近
* 缺点：二阶导矩阵计算复杂

# 高斯-牛顿法

$$
\left(\mathbf{J}^{\top} \mathbf{J}\right) \Delta \mathbf{x}_{\mathrm{gn}}=-\mathbf{J}^{\top} \mathbf{f}
$$

# 列文伯格-马夸尔特法

$$
\left(\mathbf{J}^{\top} \mathbf{J}+\mu \mathbf{I}\right) \Delta \mathbf{x}_{\operatorname{lm}}=-\mathbf{J}^{\top} \mathbf{f} \quad \text { with } \mu \geq 0
$$

# DogLeg

# 鲁棒核函数
