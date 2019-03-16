---
layout: post
title: "VINS-Mono 论文公式推导与代码解析"
date: 2019-03-16
categories: SLAM
tags: [SLAM]
---

[TOC]

# 概述

# 1. 测量预处理

## 1.1 视觉处理前端

## 1.2 IMU 预积分

### IMU 测量方程

$$
\begin{aligned}
\hat{a}_t &= a_t + b_{a_t} + R_w^t g^w + n_a \\
\hat{\omega} &= \omega_t + b_{\omega}^t + n_{\omega}
\end{aligned}
$$

### 误差状态方程

IMU误差状态向量  

$$
\delta X =
[\delta P \quad \delta v \quad \delta \theta
 \quad \delta b_a \quad \delta b_g]^T
\in \mathbb{R}^{15 \times 1}
$$

根据参考文献[2]中 ***5.3.3 The error-state kinematics*** 小节公式  

<div align=center>
  <img src="../images/vins_mono/eskf_533.png">
</div>

对于 **中值法积分** 下的误差状态方程为  

$$
\dot{\delta X_k} =
\begin{cases}
\dot{\delta \theta_{k}} =&
-[\frac{w_{k+1}+w_{k}}{2}-b_{g_{k}}]_{\times} \delta \theta_{k}-\delta b_{g_{k}}+\frac{n_{w0}+n_{w1}}{2} \\
\dot{\delta\beta_{k}} =&
-\frac{1}{2}q_{k}[a_{k}-b_{a_{k}}]_{\times}\delta \theta \\
&-\frac{1}{2}q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}((I-[\frac{w_{k+1}+w_{k}}{2}-b_{g_{k}}]_{\times }\delta t) \delta \theta_{k} -\delta b_{g_{k}}\delta t+\frac{n_{w0}+n_{w1}}{2}\delta t) \\
&-\frac{1}{2}q_{k}\delta b_{a_{k}}-\frac{1}{2}q_{k+1}\delta b_{a_{k}}-\frac{1}{2}q_{k}n_{a0}-\frac{1}{2}q_{k}n_{a1} \\
\dot{\delta\alpha_{k}} =&
-\frac{1}{4}q_{k}[a_{k}-b_{a_{k}}]_{\times}\delta \theta\delta t \\
&-\frac{1}{4}q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}((I-[\frac{w_{k+1}+w_{k}}{2}-b_{g_{k}}]_{\times }\delta t) \delta \theta _{k} -\delta b_{g_{k}}\delta t+\frac{n_{w0}+n_{w1}}{2}\delta t)\delta t \\
&-\frac{1}{4}q_{k}\delta b_{a_{k}}\delta t-\frac{1}{4}q_{k+1}\delta b_{a_{k}}\delta t-\frac{1}{4}q_{k}n_{a0}\delta t-\frac{1}{4}q_{k}n_{a1}\delta t \\
\dot{\delta b_{a_k}} =&  n_{b_a} \\
\dot{\delta b_{g_k}} =&  n_{b_g}
\end{cases}
$$

简写为

$$
\dot{\delta X_k} = F \delta X_k + Gn
$$

所以

$$
\begin{aligned}
\delta X_{k+1}
&= \delta X_k + \dot{\delta X_k} \delta t \\
&= \delta X_k + (F \delta X_k + Gn) \delta t \\
&= (I + F \delta t) \delta X_k + (G \delta t) n
\end{aligned}
$$

展开得

$$
\begin{aligned}
\begin{bmatrix}
\delta \alpha_{k+1}\\
\delta \theta_{k+1}\\
\delta \beta_{k+1} \\
\delta b_{a{}{k+1}} \\
\delta b_{g{}{k+1}}
\end{bmatrix}&=\begin{bmatrix}
I & f_{01} &\delta t  & -\frac{1}{4}(q_{k}+q_{k+1})\delta t^{2} & f_{04}\\
0 & I-[\frac{w_{k+1}+w_{k}}{2}-b_{wk}]_{\times } \delta t & 0 &  0 & -\delta t \\
0 &  f_{21}&I  &  -\frac{1}{2}(q_{k}+q_{k+1})\delta t & f_{24}\\
0 &  0&  0&I  &0 \\
 0& 0 & 0 & 0 & I
\end{bmatrix}
\begin{bmatrix}
\delta \alpha_{k}\\
\delta \theta_{k}\\
\delta \beta_{k} \\
\delta b_{a{}{k}} \\
\delta b_{g{}{k}}
\end{bmatrix} \\
&+
\begin{bmatrix}
 \frac{1}{4}q_{k}\delta t^{2}&  v_{01}& \frac{1}{4}q_{k+1}\delta t^{2} & v_{03} & 0 & 0\\
 0& \frac{1}{2}\delta t & 0 & \frac{1}{2}\delta t &0  & 0\\
 \frac{1}{2}q_{k}\delta t&  v_{21}& \frac{1}{2}q_{k+1}\delta t & v_{23} & 0 & 0 \\
0 & 0 & 0 & 0 &\delta t  &0 \\
 0& 0 &0  & 0 &0  & \delta t
\end{bmatrix}
\begin{bmatrix}
n_{a0}\\
n_{w0}\\
n_{a1}\\
n_{w1}\\
n_{ba}\\
n_{bg}
\end{bmatrix}
\end{aligned}
$$

其中

$$
\begin{aligned}
f_{01}&=-\frac{1}{4}q_{k}[a_{k}-b_{a_{k}}]_{\times}\delta t^{2}-\frac{1}{4}q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}(I-[\frac{w_{k+1}+w_{k}}{2}-b_{g_{k}}]_{\times }\delta t)\delta t^{2} \\
f_{21}&=-\frac{1}{2}q_{k}[a_{k}-b_{a_{k}}]_{\times}\delta t-\frac{1}{2}q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}(I-[\frac{w_{k+1}+w_{k}}{2}-b_{g_{k}}]_{\times }\delta t)\delta t \\
f_{04}&=\frac{1}{4}(-q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}\delta t^{2})(-\delta t) \\
f_{24}&=\frac{1}{2}(-q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}\delta t)(-\delta t) \\
v_{01}&=\frac{1}{4}(-q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}\delta t^{2})\frac{1}{2}\delta t \\
v_{03}&=\frac{1}{4}(-q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}\delta t^{2})\frac{1}{2}\delta t \\
v_{21}&=\frac{1}{2}(-q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}\delta t^{2})\frac{1}{2}\delta t \\
v_{23}&=\frac{1}{2}(-q_{k+1}[a_{k+1}-b_{a_{k}}]_{\times}\delta t^{2})\frac{1}{2}\delta t
\end{aligned}
$$

令

$$
\begin{aligned}
F' &= I + F \delta t \\
V  &= G \delta t
\end{aligned}
$$

则简写为

$$
\delta X_{k+1} = F' \delta X_k + V n
$$

最后得到系统的 **雅克比矩阵** $J_{k+1}$ 和 **协方差矩阵** $P_{k+1}$

$$
\begin{aligned}
J_{k+1} &= F' J_k \\
P_{k+1} &= F' P_k F'^T + V Q V^T,
\quad Q = \text{diag}(
  \sigma_{a_0}^2 \quad \sigma_{\omega_0}^2 \quad
  \sigma_{a_1}^2 \quad \sigma_{\omega_1}^2 \quad
  \sigma_{b_a}^2 \quad \sigma_{b_g}^2) \in \mathbb{R}^{18 \times 18}
\end{aligned}
$$

初始状态下的雅克比矩阵和协方差矩阵为 **单位阵** 和 **零矩阵**

# 2. 初始化(松耦合)

## 相机与IMU之间的相对旋转

## 相机初始化

## 视觉与IMU对齐


# 3. 后端优化(紧耦合)

**滑动窗口** 中的 **全状态量**

$$
\begin{aligned}
X &= [x_{0},x_{1},\cdots ,x_{n},x^{b}_{c},{\lambda}_{0},{\lambda}_{1}, \cdots ,{\lambda}_{m}]  \\
x_{k} &= [p^{w}_{b_{k}},v^{w}_{b_{k}},q^{w}_{b_{k}},b_{a},b_{g}],\quad k\in[0,n] \\
x^{b}_{c} &= [p^{b}_{c},q^{b}_{c}]
\end{aligned}
$$

## IMU 测量残差

## 视觉 测量残差

## 边缘化


# 4. 重定位


# 5. 全局位姿图优化


# 参考文献
[1] VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator
[2] Quaternion kinematics for the error-state Kalman filter
