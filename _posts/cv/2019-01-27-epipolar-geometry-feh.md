---
layout: post
title: "对极几何之FEH"
date: 2019-01-27
categories: ComputerVision
tags: [Stereo Vision, 3D Reconstruction]
---

[TOC]

## Overview

<div align=center>
  <img src="../images/epipolar_geometry/epipolar_geometry.jpg">
</div>

* The gray region is the **epipolar plane**
* The orange line is the **baseline**
* the two blue lines are the **epipolar lines**

Basic Epipolar Geometry entities for **pinhole cameras** and **panoramic camera** sensors  

<div align=center>
  <img src="../images/epipolar_geometry/epipolar_geometry_pinhole.jpg"> <img src="../images/epipolar_geometry/epipolar_geometry_panoramic.jpg">
</div>

## Foundamental Matrix (基本矩阵)

$$
\boldsymbol{F} = \boldsymbol{K}'^{-T} \boldsymbol{E} \boldsymbol{K}^{-1}
$$

$$
\boldsymbol{p}'^T \cdot \boldsymbol{F} \cdot \boldsymbol{p} = 0
$$

其中，$\boldsymbol{p}, \boldsymbol{p}'$ 为两个匹配 **像素点坐标**

$\boldsymbol{F}$ 的性质：

* 对其乘以 **任意非零常数**，对极约束依然满足（尺度等价性，up to scale）
* 具有 **7个自由度**： 2x11-15=7 (why)
* 奇异性约束 $\text{rank}(\boldsymbol{F})=2$

$\boldsymbol{F}$ 与 极线和极点 的关系：

$$
\boldsymbol{l}' = \boldsymbol{F} \cdot \boldsymbol{p} \\[2ex]
\boldsymbol{l} = \boldsymbol{F}^T \cdot \boldsymbol{p}' \\[2ex]
\boldsymbol{F} \cdot \boldsymbol{e} = \boldsymbol{F}^T \cdot \boldsymbol{e}' = 0
$$

$\boldsymbol{F}$ 的计算：

* Compute from 7 image point correspondences
* 8点法（**Eight-Point Algorithm**）

### Foundamental Matrix Estimation

* 类似于下面 $\boldsymbol{E}$ 的估计


## Essential Matrix (本质矩阵)

* A 3×3 matrix is an essential matrix **if and only if two of its singular values are equal, and the third is zero**

对极约束：

$$
\boldsymbol{E} = \boldsymbol{t}^\wedge \boldsymbol{R} \\[2ex]
\boldsymbol{E} = \boldsymbol{K}'^{T} \boldsymbol{F} \boldsymbol{K} \\[2ex]
{\boldsymbol{p}'}^T \cdot \boldsymbol{E} \cdot \boldsymbol{p} = 0
$$

其中，$\boldsymbol{p}, \boldsymbol{p}'$ 为两个匹配像素点的 **归一化平面坐标**

$\boldsymbol{E}$ 的性质：

* 对其乘以 **任意非零常数**，对极约束依然满足（尺度等价性，up to scale）
* 根据 $\boldsymbol{E} = \boldsymbol{t}^\wedge \boldsymbol{R}$，$\boldsymbol{E}$ 的奇异值必定是 $[\sigma, \sigma, 0]^T$ 的形式
* 具有 **5个自由度**：平移旋转共6个自由度 + 尺度等价性
* 奇异性约束 $\text{rank}(\boldsymbol{E})=2$（因为 $\text{rank}(\boldsymbol{t^\wedge})=2$）

$\boldsymbol{E}$ 与 极线和极点 的关系：

$$
\boldsymbol{l}' = \boldsymbol{E} \cdot \boldsymbol{p} \\[2ex]
\boldsymbol{l} = \boldsymbol{E}^T \cdot \boldsymbol{p}' \\[2ex]
\boldsymbol{E} \cdot \boldsymbol{e} = \boldsymbol{E}^T \cdot \boldsymbol{e}' = 0
$$

$\boldsymbol{E}$ 的计算：

* 5点法（最少5对点求解）
* 8点法（**Eight-Point Algorithm**）

### Essential Matrix Estimation

矩阵形式：

$$
\begin{bmatrix} x' & y' & 1 \end{bmatrix}
\cdot
\begin{bmatrix}
e_{1} & e_{2} & e_{3} \\
e_{4} & e_{5} & e_{6} \\
e_{7} & e_{8} & e_{9}
\end{bmatrix}
\cdot
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = 0
$$

矩阵 $\boldsymbol{E}$ 展开，写成向量形式 $\boldsymbol{e}$，并把所有点（n对点，n>=8）放到一个方程中，**齐次线性方程组** 如下：

$$
\begin{bmatrix}
x'^1x^1 & x'^1y^1 & x'^1 &
y'^1x^1 & y'^1y^1 & y'^1 &
x^1     & y^1     & 1    \\
x'^2x^2 & x'^2y^2 & x'^2 &
y'^2x^2 & y'^2y^2 & y'^2 &
x^2     & y^2     & 1    \\
\vdots & \vdots & \vdots &
\vdots & \vdots & \vdots &
\vdots & \vdots & \vdots \\
x'^nx^n & x'^ny^n & x'^n &
y'^nx^n & y'^ny^n & y'^n &
x^n     & y^n     & 1    \\
\end{bmatrix}
\cdot
\begin{bmatrix}
e_{1} \\ e_{2} \\ e_{3} \\
e_{4} \\ e_{5} \\ e_{6} \\
e_{7} \\ e_{8} \\ e_{9}
\end{bmatrix} = 0
$$

即（the essential matrix lying in **the null space of this matrix A**）

$$
\boldsymbol{A} \cdot \boldsymbol{e} = \mathbf{0}
\quad s.t. \quad
\boldsymbol{A} \in \mathbb{R}^{n \times 9}, n \geq 8
$$

对上式 求解 **最小二乘解**（尺度等价性）

$$
\min_{\boldsymbol{e}} \|\boldsymbol{A} \cdot \boldsymbol{e}\|^2
\quad s.t. \quad
\|\boldsymbol{e}^T \boldsymbol{e}\| = 1
\quad \text{or} \quad
{\|\boldsymbol{E}\|}_F = 1
$$

SVD分解 $\boldsymbol{A}$（或者 特征值分解 $\boldsymbol{A}^T \boldsymbol{A}$）

$$
\text{SVD}(\boldsymbol{A}) = \boldsymbol{U} \boldsymbol{D} \boldsymbol{V}^T
$$

$\boldsymbol{e}$ 正比于 $\boldsymbol{V}$ 的最后一列，得到 $\boldsymbol{E}$  

根据 奇异性约束 $\text{rank}(\boldsymbol{E})=2$，再 SVD分解 $\boldsymbol{E}$  

$$
\text{SVD}(\boldsymbol{E}) =
\boldsymbol{U}_E \boldsymbol{D}_E \boldsymbol{V}_E^T
$$

求出的 $\boldsymbol{E}$ 可能不满足其内在性质（奇异值是 $[\sigma, \sigma, 0]^T$ 的形式），此时对 $\boldsymbol{D}_E$ 进行调整，假设 $\boldsymbol{D}_E = \text{diag}(\sigma_1, \sigma_2, \sigma_3)$ 且 $\sigma_1 \geq \sigma_2 \geq \sigma_3$，则令  

$$
\boldsymbol{D}_E' =
\text{diag}(\frac{\sigma_1+\sigma_2}{2}, \frac{\sigma_1+\sigma_2}{2}, 0)
$$

或者，更简单的（尺度等价性）

$$
\boldsymbol{D}_E' = \text{diag}(1, 1, 0)
$$

最后，$\boldsymbol{E}$ 矩阵的正确估计为

$$
\boldsymbol{E}' =
\boldsymbol{U}_E \boldsymbol{D}_E' \boldsymbol{V}_E^T
$$

### Rotation and translation from E

The four possible solutions for calibrated reconstruction from E  
* Between the left and right sides there is a baseline reversal
* Between the top and bottom rows camera B rotates 180 about the baseline
* only in (a) is the reconstructed point in front of both cameras

<div align=center>
  <img src="../images/epipolar_geometry/four_solutions_E.jpg">
</div>

Suppose that the SVD of E is $U \text{diag}(1, 1, 0) V^T$, there are (**ignoring signs**) two possible factorizations $E = t^\wedge R = SR$ as follows

$$
t^\wedge = S = UZU^T \\[2ex]
R_1 = UWV^T \quad or \quad R_2 = UW^TV^T
$$

where

$$
W =
\begin{bmatrix}
0 & -1 & 0 \\
1 &  0 & 0 \\
0 &  0 & 1
\end{bmatrix}
\quad \text{and} \quad
Z =
\begin{bmatrix}
 0 & 1 & 0 \\
-1 &  0 & 0 \\
 0 &  0 & 0
\end{bmatrix}
$$

and

$$
\begin{aligned}
 W &\approx R_z(\frac{\pi}{2}) \\[2ex]
 Z &= W^T \cdot \text{diag}(1, 1, 0) \\[2ex]
-Z &= W   \cdot \text{diag}(1, 1, 0)
\end{aligned}
$$

**DecomposeE** in ORB-SLAM2  

```c++
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t) {
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}
```

**[CheckRT](https://github.com/cggos/orbslam2_cg/blob/master/src/Initializer.cc)** in ORB-SLAM2

## Homography Matrix (单应性矩阵)

* For planar surfaces, 3D to 2D perspective projection reduces to a 2D to 2D transformation

**单应性矩阵** 通常描述处于 **共同平面** 上的一些点在 **两张图像之间的变换关系**。

$$
\boldsymbol{p'} = \boldsymbol{H} \cdot \boldsymbol{p}
$$

其中，$p, p'$ 为两个匹配像素点的 **归一化平面坐标** （也可为其他点，只要 **共面且3点不共线** 即可）

### Homography Estimation

矩阵形式：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\cdot
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

方程形式（两个约束条件）：

$$
x' =
\frac
{ h_{11}x + h_{12}y + h_{13} }
{ h_{31}x + h_{32}y + h_{33} } \\[2ex]
y' =
\frac
{ h_{21}x + h_{22}y + h_{23} }
{ h_{31}x + h_{32}y + h_{33} }
$$

因为上式使用的是齐次坐标，所以我们可以 **对 所有的 $h_{ij}$ 乘以 任意的非0因子** 而不会改变方程。

因此， $\boldsymbol{H}$ 具有 **8个自由度**，最少通过 **4对匹配点（不能出现3点共线）** 算出。

实际中，通过 **$h_{33}=1$** 或 **$\|\boldsymbol{H}\|_F=1$** 两种方法 使 $\boldsymbol{H}$ 具有 8自由度。

#### cont 1: H元素h33=1

<div align=center>
  <img src="../images/epipolar_geometry/homography_cont_01.jpg">
</div>

线性方程：

$$
\boldsymbol{A} \cdot \boldsymbol{h} = \boldsymbol{b}
$$

求解：

$$
\boldsymbol{A}^T \boldsymbol{A} \cdot \boldsymbol{h} =
\boldsymbol{A}^T \boldsymbol{b}
$$

所以

$$
\boldsymbol{h} =
(\boldsymbol{A}^T \boldsymbol{A})^{-1} \boldsymbol{A}^T \boldsymbol{b}
$$

#### cont 2: H的F范数|H|=1

<div align=center>
  <img src="../images/epipolar_geometry/homography_cont_02.jpg">
</div>

线性方程：

$$
\boldsymbol{A} \cdot \boldsymbol{h} = \mathbf{0}
$$

求解：

$$
\boldsymbol{A}^T \boldsymbol{A} \cdot \boldsymbol{h} = \mathbf{0}
$$

对上式 求解 **最小二乘解**（尺度等价性）

$$
\min_{\boldsymbol{h}} \|(\boldsymbol{A}^T \boldsymbol{A}) \cdot \boldsymbol{h}\|^2
\quad s.t. \quad
\|\boldsymbol{h}^T \boldsymbol{h}\| = 1
\quad \text{or} \quad
{\|\boldsymbol{H}\|}_F = 1
$$

SVD分解 或 特征值分解

$$
\text{SVD}(\boldsymbol{A}^T \boldsymbol{A}) =
\boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{U}^T
$$

最后 $\boldsymbol{h}$ 为 $\boldsymbol{\Sigma}$ 中 **最小特征值** 对应的 $\boldsymbol{U}$ 中的列向量（单位特征向量）；如果只用4对匹配点，那个特征值为0。



### H in PTAM

* [相关代码](https://github.com/cggos/ptam_cg/blob/master/src/HomographyInit.cc)  

#### 单应性矩阵的计算

```c++
Matrix<3> HomographyInit::HomographyFromMatches(vector<HomographyMatch> vMatches)
{
    unsigned int nPoints = vMatches.size();
    assert(nPoints >= 4);
    int nRows = 2*nPoints;
    if(nRows < 9)
        nRows = 9;
    Matrix<> m2Nx9(nRows, 9);
    for(unsigned int n=0; n<nPoints; n++)
    {
        double u = vMatches[n].v2CamPlaneSecond[0];
        double v = vMatches[n].v2CamPlaneSecond[1];

        double x = vMatches[n].v2CamPlaneFirst[0];
        double y = vMatches[n].v2CamPlaneFirst[1];

        // [u v]T = H [x y]T
        m2Nx9[n*2+0][0] = x;
        m2Nx9[n*2+0][1] = y;
        m2Nx9[n*2+0][2] = 1;
        m2Nx9[n*2+0][3] = 0;
        m2Nx9[n*2+0][4] = 0;
        m2Nx9[n*2+0][5] = 0;
        m2Nx9[n*2+0][6] = -x*u;
        m2Nx9[n*2+0][7] = -y*u;
        m2Nx9[n*2+0][8] = -u;

        m2Nx9[n*2+1][0] = 0;
        m2Nx9[n*2+1][1] = 0;
        m2Nx9[n*2+1][2] = 0;
        m2Nx9[n*2+1][3] = x;
        m2Nx9[n*2+1][4] = y;
        m2Nx9[n*2+1][5] = 1;
        m2Nx9[n*2+1][6] = -x*v;
        m2Nx9[n*2+1][7] = -y*v;
        m2Nx9[n*2+1][8] = -v;
    }

    if(nRows == 9)
        for(int i=0; i<9; i++)  // Zero the last row of the matrix,
            m2Nx9[8][i] = 0.0;  // TooN SVD leaves out the null-space otherwise

    // The right null-space of the matrix gives the homography...
    SVD<> svdHomography(m2Nx9);
    Vector<9> vH = svdHomography.get_VT()[8];
    Matrix<3> m3Homography;
    m3Homography[0] = vH.slice<0,3>();
    m3Homography[1] = vH.slice<3,3>();
    m3Homography[2] = vH.slice<6,3>();
    return m3Homography;
};
```

#### Rotation and translation from H

* *Motion and structure from motion in a piecewise planar environment*

#### 手写笔记

<div align=center>
  <img src="../images/epipolar_geometry/homography_matrix_ptam_note.jpg">
</div>


## 总结

当 特征点共面 或者 相机发生纯旋转 时，基础矩阵 $F$ 的自由度下降，就会出现所谓的 **退化(degenerate)**。  

为了能够避免退化现象的影响，通常会 **同时估计基础矩阵 $F$ 和 单应矩阵 $H$，选择重投影误差比较小的那个作为最终的运动估计矩阵**。

## Reference

* Epipolar Geometry and the Fundamental Matrix in MVG (Chapter 9)
* 《视觉SLAM十四讲》
