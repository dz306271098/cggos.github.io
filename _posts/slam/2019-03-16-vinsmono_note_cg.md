---
layout: post
title: "VINS-Mono 论文公式推导与代码解析"
date: 2019-03-16
categories: SLAM
tags: [SLAM]
---

[TOC]

# 概述

<div align=center>
  <img src="../images/vins_mono/vins_mono_framework.png">
</div>

# 1. 测量预处理

## 1.1 视觉处理前端

* 自适应直方图均衡化（ `cv::CLAHE` ）
* 掩模处理，特征点均匀分布（`setMask`）
* 提取图像Harris角点（`cv::goodFeaturesToTrack`）
* 金字塔光流跟踪（`cv::calcOpticalFlowPyrLK`）
* 本质矩阵(RANSAC)去除异常点（`rejectWithF`）
* 发布feature_points(id_of_point, un_pts, cur_pts, pts_velocity)

## 1.2 IMU 预积分

<div align=center>
  <img src="../images/vins_mono/imu_integration.png">
</div>

### IMU 测量方程

忽略地球旋转，IMU 测量方程为

$$
\begin{aligned}
\hat{a}_t &= a_t + b_{a_t} + R_w^t g^w + n_a \\
\hat{\omega} &= \omega_t + b_{\omega}^t + n_{\omega}
\end{aligned}
$$

从世界坐标系转为本体坐标系

<div align=center>
  <img src="../images/vins_mono/formular_w_b.png">
</div>

则 IMU测量模型（**观测值**）为

$$
\begin{bmatrix}
\hat{\alpha}^{b_{k}}_{b_{k+1}}\\
\hat{\gamma}^{b_{k}}_{b_{k+1}}\\
\hat{\beta }^{b_{k}}_{b_{k+1}}\\
0\\
0
\end{bmatrix}
=\begin{bmatrix}
R^{b_{k}}_{w}(p^{w}_{b_{k+1}}-p_{b_{k}}^{w}+\frac{1}{2}g^{w}\triangle t^{2}-v_{b_{k}}^{w}\triangle t)\\
p_{b_{k}}^{w^{-1}}\otimes q^{w}_{b_{k+1}}\\
R^{b_{k}}_{w}(v^{w}_{b_{k+1}}+g^{w}\triangle t-v_{b_{k}}^{w})\\
b_{ab_{k+1}}-b_{ab_{k}}\\
b_{wb_{k+1}}-b_{wb_{k}}
\end{bmatrix}
$$

### 预积分方程

离散状态下采用 **中值法积分** 的预积分方程为

$$
\begin{aligned}
\delta q_{i+1} &= \delta q_{i} \otimes
\begin{bmatrix}
1
\\
0.5w_{i}'
\end{bmatrix} \\
\delta\alpha_{i+1} &= \delta\alpha_{i}+\delta\beta_{i}t+0.5a_{i}'\delta t^{2} \\
\delta\beta_{i+1}&=\delta\beta_{i}+a_{i}' \delta t \\
{b_a}_{i+1}&= {b_a}_i \\
{b_g}_{i+1}&= {b_g}_i
\end{aligned}
$$

其中

$$
\begin{aligned}
w_{i}' &= \frac{w_{i+1}+w_{i}}{2}-b_{i} \\
a_{i}' &= \frac{
  \delta q_{i}(a_{i}+n_{a0}-b_{a_{i}})+
  \delta q_{i+1}(a_{i+1}++n_{a1}-b_{a_{i}})}{2}
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
  <img src="../images/vins_mono/formular_eskf_533.png">
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
F' &= I + F \delta t & \in \mathbb{R}^{15 \times 15} \\
V  &= G \delta t     & \in \mathbb{R}^{15 \times 18}
\end{aligned}
$$

则简写为

$$
\delta X_{k+1} = F' \delta X_k + V n
$$

最后得到系统的 **雅克比矩阵** $J_{k+1}$ 和 **协方差矩阵** $P_{k+1}$，初始状态下的雅克比矩阵和协方差矩阵为 **单位阵** 和 **零矩阵**

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

当bias估计轻微改变时，我们可以使用如下的一阶近似 **对中值积分得到的预积分项进行矫正**，而不重传播，从而得到 **预积分估计值**

$$
\begin{aligned}
{\alpha}^{b_{k}}_{b_{k+1}} &\approx
\hat{\alpha}^{b_{k}}_{b_{k+1}} +
J^{\alpha}_{b_a} \delta {b_a}_k +
J^{\alpha}_{b_g} \delta {b_g}_k \\
{\beta}^{b_{k}}_{b_{k+1}} &\approx
\hat{\beta}^{b_{k}}_{b_{k+1}} +
J^{\beta}_{b_a} \delta {b_a}_k +
J^{\beta}_{b_g} \delta {b_g}_k \\
{\gamma}^{b_{k}}_{b_{k+1}} &\approx
\hat{\gamma}^{b_{k}}_{b_{k+1}} \otimes
\begin{bmatrix}
1
\\
\frac{1}{2} J^{\gamma}_{b_g} \delta {b_g}_k
\end{bmatrix}
\end{aligned}
$$


# 2. 初始化(松耦合)

在提取的图像的Features和做完IMU的预积分之后，进入了系统的初始化环节，那么系统为什么要进行初始化，主要的目的有以下两个：      

- 系统使用单目相机，如果没有一个良好的尺度估计，就无法对两个传感器做进一步的融合。这个时候需要恢复出尺度；
- 要对IMU进行初始化，IMU会受到bias的影响，所以要得到IMU的bias。

所以我们要从初始化中恢复出尺度、重力、速度以及IMU的bias，因为视觉(SFM)在初始化的过程中有着较好的表现，所以在初始化的过程中主要以SFM为主，然后将IMU的预积分结果与其对齐，即可得到较好的初始化结果。

## 2.1 相机与IMU之间的相对旋转

相机与IMU之间的旋转标定非常重要，**偏差1-2°系统的精度就会变的极低**。

设相机利用对极关系得到的旋转矩阵为 $R^{c_{k}}_{c_{k+1}}$，IMU经过预积分得到的旋转矩阵为$R^{b_{k}}_{b_{k+1}}$，相机与IMU之间的相对旋转为 $R^{b}_{c}$，则对于任一帧满足，

$$
R^{b_{k}}_{b_{k+1}}R^{b}_{c}=R^{b}_{c}R^{c_{k}}_{c_{k+1}}
$$

将旋转矩阵写为四元数，则上式可以写为

$$
q^{b_{k}}_{b_{k+1}} \otimes q^{b}_{c}=q^{b}_{c}\otimes q^{c_{k}}_{c_{k+1}}
$$

将其写为左乘和右乘的形式

$$
({[q^{b_{k}}_{b_{k+1}}]}_L - {[q^{c_{k}}_{c_{k+1}}]}_R) q^b_c
= Q^k_{k+1} q^b_c = 0
$$

$[q]_L$ 与 $[q]_R$ 分别表示 **四元数左乘矩阵** 和 **四元数右乘矩阵**，其定义为（四元数实部在后）

$$
\begin{aligned}
[q]_L &=
\begin{bmatrix}
q_{w}I_{3}+[q_{xyz }]_{\times} & q_{xyz}\\
-q_{xyz} & q_{w}
\end{bmatrix} \\
[q]_R &=
\begin{bmatrix}
q_{w}I_{3}-[q_{xyz }]_{\times} & q_{xyz}\\
-q_{xyz} & q_{w}
\end{bmatrix}
\end{aligned}
$$

那么对于 $n$对测量值，则有

$$
\begin{bmatrix}
w^{0}_{1}Q^{0}_{1}\\
w^{1}_{2}Q^{1}_{2}\\
\vdots \\
w^{N-1}_{N}Q^{N-1}_{N}
\end{bmatrix}q^{b}_{c}=Q_{N}q^{b}_{c}=0
$$

其中 $w^{N-1}_{N}$ 为外点剔除权重，其与相对旋转求得的角度残差有关，$N$为计算相对旋转需要的测量对数，其由最终的终止条件决定。角度残差可以写为，

$$
{\theta}^{k}_{k+1}=
arccos\bigg(
  \frac{tr(\hat{R}^{b^{-1}}_{c}R^{b_{k}^{-1}}_{b_{k+1}}\hat{R}^{b}_{c}R^{c_{k}}_{c_{k+1}} )-1}{2}\bigg)
$$

从而权重为

$$
w^{k}_{k+1}=
\left\{\begin{matrix}
1, & {\theta}^{k}_{k+1}<threshold  (\text{一般5°}) \\
\frac{threshold}{\theta^{k}_{k+1}}, & otherwise
\end{matrix}\right.
$$

至此，就可以通过求解方程 $Q_{N}q^{b}_{c}=0$ 得到相对旋转，解为 $Q_{N}$ 的左奇异向量中最小奇异值对应的特征向量。

但是，在这里还要注意 __求解的终止条件(校准完成的终止条件)__ 。在足够多的旋转运动中，我们可以很好的估计出相对旋转 $R^{b}_{c}$，这时 $Q_{N}$ 对应一个准确解，且其零空间的秩为1。但是在校准的过程中，某些轴向上可能存在退化运动(如匀速运动)，这时 $Q_{N}$ 的零空间的秩会大于1。判断条件就是 $Q_{N}$ 的第二小的奇异值是否大于某个阈值，若大于则其零空间的秩为1，反之秩大于1，相对旋转$R^{b}_{c}$ 的精度不够，校准不成功。  

## 2.2 相机初始化

* 求取本质矩阵求解位姿
* 三角化特征点
* PnP求解位姿

不断重复的过程，直到恢复出滑窗内的Features和相机位姿

## 2.3 视觉与IMU对齐

* Gyroscope Bias Calibration
* Velocity, Gravity Vector and Metric Scale Initialization
* Gravity Refinement
* Completing Initialization

<div align=center>
  <img src="../images/vins_mono/visual_inertial_alignment.png">
</div>

### 陀螺仪Bias标定

标定陀螺仪Bias使用如下代价函数

$$
\underset{\delta b_{w}}{min}\sum_{k\in B}^{ }\left \| q^{c_{0}^{-1}}_{b_{k+1}}\otimes q^{c_{0}}_{b_{k}}\otimes\gamma _{b_{k+1}}^{b_{k}} \right \|^{2}
$$

因为四元数最小值为单位四元数 $[1,0_{v}]^{T}$，所以令

$$
q^{c_{0}^{-1}}_{b_{k+1}}\otimes q^{c_{0}}_{b_{k}}\otimes\gamma _{b_{k+1}}^{b_{k}} =
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

其中

$$
\gamma _{b_{k+1}}^{b_{k}}\approx \hat{\gamma}_{b_{k+1}}^{b_{k}}\otimes \begin{bmatrix}
1\\
\frac{1}{2}J^{\gamma }_{b_{w}}\delta b_{w}
\end{bmatrix}
$$

所以

$$
\hat{\gamma}_{b_{k+1}}^{b_{k}} \otimes
\begin{bmatrix}
1\\
\frac{1}{2}J^{\gamma }_{b_{w}}\delta b_{w}
\end{bmatrix}
= q^{c_{0}^{-1}}_{b_{k}}\otimes q^{c_{0}}_{b_{k+1}}
$$

$$
\begin{bmatrix}
1\\
\frac{1}{2}J^{\gamma }_{b_{w}}\delta b_{w}
\end{bmatrix}=\hat{\gamma}_{b_{k+1}}^{b_{k}^{-1}}\otimes q^{c_{0}^{-1}}_{b_{k}}\otimes q^{c_{0}}_{b_{k+1}}
$$


只取上式虚部，再进行最小二乘求解

$$
J^{\gamma^{T}}_{b_{w}}J^{\gamma }_{b_{w}}\delta b_{w}=
J^{\gamma^{T}}_{b_{w}}(\hat{\gamma}_{b_{k+1}}^{b_{k}^{-1}}\otimes q^{c_{0}^{-1}}_{b_{k}}\otimes q^{c_{0}}_{b_{k+1}})_{vec}
$$

求解上式的最小二乘解，即可得到 $\delta b_{w}$，注意这个地方得到的只是Bias的变化量，需要在滑窗内累加得到Bias的准确值。   

### 初始化速度、重力向量和尺度因子

要估计的状态量

$$
X_{I}=
[v^{b_{0}}_{b_{0}}, v^{b_{0}}_{b_{1}}, \cdots, v^{b_{n}}_{b_{n}}, g^{c_{0}}, s]
\in \mathbb{R}^{3(n+1)+3+1}
$$

其中，$g^{c_{0}}$ 为在第 0 帧 Camera 相机坐标系下的重力向量。

根据IMU测量模型可知

$$
\begin{aligned}
\alpha^{b_{k}}_{b_{k+1}} &= R^{b_{k}}_{c_{0}}(s(\bar{p}^{c_{0}}_{b_{k+1}}-\bar{p}^{c_{0}}_{b_{k}}) +
\frac{1}{2}g^{c_{0}}\Delta t_{k}^{2} -
R^{c_0}_{b_k} v^{b_k}_{b_{k}} \Delta t_{k}) \\
\beta ^{b_{k}}_{b_{k+1}} &=
R^{b_{k}}_{c_{0}}(R^{c_0}_{b_{k+1}} v^{b_{k+1}}_{b_{k+1}} +
g^{c_{0}}\Delta t_{k} -
R^{c_0}_{b_k} v^{b_k}_{b_{k}})
\end{aligned}
$$

我们已经得到了IMU相对于相机的旋转 $q_{b}^{c}$，假设IMU到相机的平移量$p_{b}^{c}$，那么可以很容易地将相机坐标系下的位姿转换到IMU坐标系下

$$
\begin{aligned}
q_{b_{k}}^{c_{0}} &=
q^{c_{0}}_{c_{k}}\otimes (q_{c}^{b})^{-1}  \\
s\bar{p}^{c_{0}}_{b_{k}} &=
s\bar{p}^{c_{0}}_{c_{k}} - R^{c_{0}}_{b_{k}}p_{c}^{b}
\end{aligned}
$$

所以，定义相邻两帧之间的IMU预积分出的增量 （$\hat{\alpha}^{b_{k}}_{b_{k+1}}$，$\hat{\beta}^{b_{k}}_{b_{k+1}}$）与预测值之间的残差，即

$$
\begin{aligned}
r(\hat{z}^{b_{k}}_{b_{k+1}}, X_I) &=
\begin{bmatrix}
\delta \alpha^{b_{k}}_{b_{k+1}} \\
\delta \beta ^{b_{k}}_{b_{k+1}}
\end{bmatrix} \\ &=
\begin{bmatrix}
\hat{\alpha}^{b_{k}}_{b_{k+1}} -& R^{b_{k}}_{c_{0}}(s(\bar{p}^{c_{0}}_{b_{k+1}}-\bar{p}^{c_{0}}_{b_{k}}) +
\frac{1}{2}g^{c_{0}}\Delta t_{k}^{2} -
R^{c_0}_{b_k} v^{b_k}_{b_{k}} \Delta t_{k})\\
\hat{\beta}^{b_{k}}_{b_{k+1}} -&
R^{b_{k}}_{c_{0}}(R^{c_0}_{b_{k+1}} v^{b_{k+1}}_{b_{k+1}} +
g^{c_{0}}\Delta t_{k} -
R^{c_0}_{b_k} v^{b_k}_{b_{k}})
\end{bmatrix}
\end{aligned}
$$

令 $r(\hat{z}^{b_{k}}_{b_{k+1}}, X_I) = \mathbf{0}$，转换成 $Hx=b$ 的形式

$$
\begin{bmatrix}
-I\Delta t_{k} & 0 & \frac{1}{2}R^{b_{k}}_{c_{0}} \Delta t_{k}^{2} &
R^{b_{k}}_{c_{0}}(\bar{p}^{c_{0}}_{c_{k+1}}-\bar{p}^{c_{0}}_{c_{k}}) \\
-I & R^{b_{k}}_{c_{0}} R^{c_0}_{b_{k+1}} & R^{b_{k}}_{c_{0}}\Delta t_{k} & 0
\end{bmatrix}
\begin{bmatrix}
v^{b_{k}}_{b_{k}}\\
v^{b_{k+1}}_{b_{k+1}}\\
g^{c_{0}}\\
s
\end{bmatrix} =
\begin{bmatrix}
\alpha^{b_{k}}_{b_{k+1}} - p_c^b + R^{b_{k}}_{c_{0}} R^{c_0}_{b_{k+1}} p_c^b \\
\beta ^{b_{k}}_{b_{k+1}}
\end{bmatrix}
$$

通过Cholosky分解求解 $X_I$

$$
H^T H X_I = H^T b
$$

### 优化重力

<div align=center>
  <img src="../images/vins_mono/gravity_tangent_space.png">
</div>

重力矢量的模长固定（9.8），其为2个自由度，在切空间上对其参数化

$$
\begin{aligned}
\hat{g} &=
\|g\| \cdot \bar{\hat{g}} + \omega_1 \vec{b_1} + \omega_2 \vec{b_2} \\ &=
\|g\| \cdot \bar{\hat{g}} + B \vec{\omega}
\end{aligned} , \quad
B \in \mathbb{R}^{3 \times 2}, \vec{\omega} \in \mathbb{R}^{2 \times 1}
$$

令 $\hat{g} = g^{c_{0}}$，将其代入上一小节公式得

$$
\begin{bmatrix}
-I\Delta t_{k} & 0 & \frac{1}{2}R^{b_{k}}_{c_{0}} \Delta t_{k}^{2} B &
R^{b_{k}}_{c_{0}}(\bar{p}^{c_{0}}_{c_{k+1}}-\bar{p}^{c_{0}}_{c_{k}}) \\
-I & R^{b_{k}}_{c_{0}} R^{c_0}_{b_{k+1}} & R^{b_{k}}_{c_{0}}\Delta t_{k} B & 0
\end{bmatrix}
\begin{bmatrix}
v^{b_{k}}_{b_{k}}\\
v^{b_{k+1}}_{b_{k+1}}\\
\vec{\omega}\\
s
\end{bmatrix} \\ =
\begin{bmatrix}
\alpha^{b_{k}}_{b_{k+1}} - p_c^b + R^{b_{k}}_{c_{0}} R^{c_0}_{b_{k+1}} p_c^b -
\frac{1}{2}R^{b_{k}}_{c_{0}} \Delta t_{k}^{2} \|g\| \cdot \bar{\hat{g}}\\
\beta ^{b_{k}}_{b_{k+1}} -
R^{b_{k}}_{c_{0}}\Delta t_{k} \|g\| \cdot \bar{\hat{g}}
\end{bmatrix}
$$

同样，通过Cholosky分解求得 $g^{c_{0}}$，即相机 $C_0$ 系下的重力向量。

最后，通过将 $g^{c_{0}}$ 旋转至惯性坐标系中的 z 轴方向，可以计算相机系到惯性系的旋转矩阵 $q_{c_0}^w$，这样就可以将所有变量调整至惯性世界系中。

# 3. 后端优化(紧耦合)

<div align=center>
  <img src="../images/vins_mono/sliding_window_vio.png">
</div>

**滑动窗口** 中的 **全状态量**

$$
\begin{aligned}
X &= [x_{0},x_{1},\cdots ,x_{n},x^{b}_{c},{\lambda}_{0},{\lambda}_{1}, \cdots ,{\lambda}_{m}]  \\
x_{k} &= [p^{w}_{b_{k}},v^{w}_{b_{k}},q^{w}_{b_{k}},b_{a},b_{g}],\quad k\in[0,n] \\
x^{b}_{c} &= [p^{b}_{c},q^{b}_{c}]
\end{aligned}
$$

优化过程中的 **误差状态量**

$$
\begin{aligned}
\delta X&=[\delta x_{0},\delta x_{1},\cdots ,\delta x_{n},\delta x^{b}_{c},\lambda_{0},\delta \lambda _{1}, \cdots , \delta \lambda_{m}]  \\
\delta x_{k}&=[\delta p^{w}_{b_{k}},\delta v^{w}_{b_{k}},\delta \theta ^{w}_{b_{k}},\delta b_{a},\delta b_{g}],\quad k\in[0,n] \\
\delta x^{b}_{c}&= [\delta p^{b}_{c},\delta q^{b}_{c}]
\end{aligned}
$$

进而得到系统优化的代价函数

$$
\underset{X}{min}
\begin{Bmatrix}
\left \|
r_{p}-H_{p}X
\right \|^{2} +
\sum_{k\in B}^{ } \left \|
r_{B}(\hat{z}^{b_{k}}_{b_{k+1}},X)
\right \|^{2}_{P^{b_{k}}_{b{k+1}}} +
\sum_{(i,j)\in C}^{ } \left \|
r_{C}(\hat{z}^{c_{j}}_{l},X)
\right \|^{2}_{P^{c_{j}}_{l}}
\end{Bmatrix}
\tag{4.2}
$$

其中三个残差项依次是

* 边缘化的先验信息
* IMU测量残差
* 视觉的观测残差

三种残差都是用 **马氏距离**（与量纲无关） 来表示的。

## 3.1 IMU 测量残差

上面的IMU预积分（测量值 - 估计值），得到IMU测量残差

$$
\begin{aligned}
r_{B}(\hat{z}^{b_{k}}_{b_{k+1}},X)=
\begin{bmatrix}
\delta \alpha ^{b_{k}}_{b_{k+1}}\\
\delta \theta   ^{b_{k}}_{b_{k+1}}\\
\delta \beta ^{b_{k}}_{b_{k+1}}\\
0\\
0
\end{bmatrix}
&=\begin{bmatrix}
q^{b_{k}}_{w}(p^{w}_{b_{k+1}}-p_{b_{k}}^{w}+\frac{1}{2}g^{w}\triangle t^{2}-v_{b_{k}}^{w}\triangle t)-\hat{\alpha }^{b_{k}}_{b_{k+1}}\\
[q_{b_{k+1}}^{w^{-1}}\otimes q^{w}_{b_{k}}\otimes \hat{\gamma  }^{b_{k}}_{b_{k+1}}]_{xyz}\\
q^{b_{k}}_{w}(v^{w}_{b_{k+1}}+g^{w}\triangle t-v_{b_{k}}^{w})-\hat{\beta }^{b_{k}}_{b_{k+1}}\\
b_{ab_{k+1}}-b_{ab_{k}}\\
b_{gb_{k+1}}-b_{gb_{k}}
\end{bmatrix}
\end{aligned}
$$

其中 $[\hat{\alpha }^{b_{k}}_{b_{k+1}},\hat{\gamma  }^{b_{k}}_{b_{k+1}},\hat{\beta }^{b_{k}}_{b_{k+1}}]$ 来自于 **IMU矫正预积分** 部分。

高斯迭代优化过程中会用到IMU测量残差对状态量的雅克比矩阵，但此处我们是 **对误差状态量求偏导**，下面对四部分误差状态量求取雅克比矩阵。

对$[\delta p^{w}_{b_{k}},\delta \theta ^{w}_{b_{k}}]$ 求偏导得

$$
J[0]=\begin{bmatrix}
-q^{b_{k}}_{w} & R^{b_{k}}_{w}[(p^{w}_{b_{k+1}}-p_{b_{k}}^{w}+\frac{1}{2}g^{w}\triangle t^{2}-v_{b_{k}}^{w}\triangle t)]_{\times }\\
0 & [q_{b_{k+1}}^{w^{-1}}q^{w}_{b_{k}}]_{L}[\hat{\gamma  }^{b_{k}}_{b_{k+1}}]_{R}J^{\gamma}_{b_{w}}\\
0 & R^{b_{k}}_{w}[(v^{w}_{b_{k+1}}+g^{w}\triangle t-v_{b_{k}}^{w})]_{\times } \\
0 & 0
\end{bmatrix}
\in \mathbb{R}^{15 \times 7}
$$

对 $[\delta v^{w}_{b_{k}},\delta b_{ab_{k}},\delta b_{wb_{k}}]$ 求偏导得

$$J[1]=
\begin{bmatrix}
-q^{b_{k}}_{w}\triangle t & -J^{\alpha }_{b_{a}} & -J^{\alpha }_{b_{a}}\\
0 & 0 & -[q_{b_{k+1}}^{w^{-1}}\otimes q^{w}_{b_{k}}\otimes \hat{\gamma  }^{b_{k}}_{b_{k+1}}]_{L}J^{\gamma}_{b_{w}}\\
-q^{b_{k}}_{w} & -J^{\beta }_{b_{a}} & -J^{\beta }_{b_{a}}\\
0& -I &0 \\
0 &0  &-I
\end{bmatrix}
\in \mathbb{R}^{15 \times 9}
$$

对 $[\delta p^{w}_{b_{k+1}},\delta \theta ^{w}_{b_{k+1}}]$ 求偏导得

$$
J[2]=
\begin{bmatrix}
-q^{b_{k}}_{w} &0\\
0 &  [\hat{\gamma  }^{b_{k}^{-1}}_{b_{k+1}}\otimes q_{w}^{b_{k}}\otimes q_{b_{k+1}}^{w}]_{L} \\
0 & 0 \\
0 & 0  \\
0 & 0   
\end{bmatrix}
\in \mathbb{R}^{15 \times 7}
$$

对 $[\delta v^{w}_{b_{k}},\delta b_{ab_{k}},\delta b_{wb_{k}}]$ 求偏导得

$$J[3]=
\begin{bmatrix}
-q^{b_{k}}_{w} &0 & 0\\
0 & 0 &0 \\
q^{b_{k}}_{w} & 0 & 0\\
 0& I &0 \\
0 &0  &I
\end{bmatrix}
\in \mathbb{R}^{15 \times 9}
$$


## 3.2 视觉 测量残差

<div align=center>
  <img src="../images/vins_mono/visual_residual_sphere.png">
</div>

视觉测量残差 即 **特征点的重投影误差**

$$
r_{C}=(\hat{z}_{l}^{c_{j}},X)=[b_{1},b_{2}]^{T}\cdot (\bar{P}_{l}^{c_{j}}-\frac{P_{l}^{c_{j}}}{\left \| P_{l}^{c_{j}} \right \|})
$$

其中，

$$
P_{l}^{c_{j}}=q_{b}^{c}(q_{w}^{b_{j}}(q_{b_{i}}^{w}(q_{c}^{b} \frac{\bar{P}_{l}^{c_{i}}}{\lambda _{l}}+p_{c}^{b})+p_{b_{i}}^{w}-p_{b_{j}}^{w})-p_{c}^{b})
$$

下面关于误差状态量对相机测量残差求偏导，得到高斯迭代优化过程中的雅克比矩阵。

对 $[\delta p^{w}_{b_{i}},\delta \theta ^{w}_{b_{i}}]$ 求偏导

$$
J[0]=\begin{bmatrix}
q_{b}^{c}q_{w}^{b_{j}} & -q_{b}^{c}q_{w}^{b_{j}}q_{b_{i}}^{w}[q_{c}^{b} \frac{\bar{P}_{l}^{c_{i}}}{\lambda_{l}}+p_{c}^{b}]_{\times }
\end{bmatrix}
\in \mathbb{R}^{3 \times 6}
$$

对 $[\delta p^{w}_{b_{j}},\delta \theta ^{w}_{b_{j}}]$ 求偏导

$$
J[1]=\begin{bmatrix}
-q_{b}^{c}q_{w}^{b_{j}} & q_{b}^{c}q_{w}^{b_{j}}[q_{b_{i}}^{w}(q_{c}^{b} \frac{\bar{P}_{l}^{c_{i}}}{\lambda _{l}}+p_{c}^{b})+p_{b_{i}}^{w}-p_{b_{j}}^{w}]_{\times }
\end{bmatrix}
\in \mathbb{R}^{3 \times 6}
$$

对 $[\delta p^{b}_{c},\delta \theta ^{b}_{c}]$ 求偏导

$$
J[2]=
\begin{bmatrix}
q_{b}^{c}(q_{w}^{b_{j}}q_{bi}^{w}-I_{3*3}) & -q_{b}^{c}q_{w}^{b_{j}}q_{b_{i}}^{w}q_{c}^{b}[\frac{\bar{P}_{l}^{c_{i}}}{\lambda_{l}}]_{\times }+[q_{b}^{c}(q_{w}^{b_{j}}(q_{b_{i}}^{w}p_{c}^{b}+p_{b_{i}}^{w}-p_{b_{j}}^{w})-p_{c}^{b})]
\end{bmatrix}
\in \mathbb{R}^{3 \times 6}
$$

对 $\delta \lambda_{l}$ 求偏导

$$
J[3]=-q_{b}^{c}q_{w}^{b_{j}}q_{b_{i}}^{w}q_{c}^{b} \frac{\bar{P}_{l}^{c_{i}}}{\lambda_{l}^{2}}
\in \mathbb{R}^{3 \times 1}
$$

## 3.3 边缘化


# 4. 重定位


# 5. 全局位姿图优化


# 参考文献

* [1] VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator  
* [2] Quaternion kinematics for the error-state Kalman filter
* [3] Xiaobuyi, [VINS-Mono代码分析总结](https://www.zybuluo.com/Xiaobuyi/note/866099)
