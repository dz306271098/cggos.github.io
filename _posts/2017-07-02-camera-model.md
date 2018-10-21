---
layout: post
title:  "相机投影模型以及畸变模型"
date:   2017-07-02
categories: ComputerVision
tags: [Camera Model]
---

[toc]

# 世界坐标系 到 像素坐标系
世界坐标系中三维点 $M=[X,Y,Z]^T$ 和 像素坐标系中二维点 $m=[u,v]^T$ 的关系为：
$$ s\tilde{m} = A [R \quad t] \tilde{M}$$
即（针孔相机模型）  

$$
\begin{aligned}
s\left[\begin{array}{c}u\\v\\1\end{array}\right] =  
\left[\begin{array}{ccc}
f_x&0&c_x\\0&f_y&c_y\\0&0&1
\end{array}\right]  
\left[\begin{array}{cccc}
r_{11}&r_{12}&r_{13}&t_1\\r_{21}&r_{22}&r_{23}&t_2\\r_{31}&r_{32}&r_{33}&t_3
\end{array}\right]  
\left[\begin{array}{c}X_w\\Y_w\\Z_w\\1\end{array}\right]
\end{aligned}
$$  

其中，$s$ 为缩放因子，$A$ 为相机的内参矩阵，$[R \quad t]$ 为相机的外参矩阵，$\tilde{m}$ 和 $\tilde{M}$ 分别为 $m$ 和 $M$ 对应的齐次坐标。

# 针孔相机模型 (pinhole)
相机将三维世界中的坐标点（单位：米）映射到二维图像平面（单位：像素）的过程能够用一个几何模型来描述，其中最简单的称为 **针孔相机模型 (pinhole camera model)** ，其框架如下图所示。
![pinhole_camera_model.png](../images/camera_model/pinhole_camera_model.png)

## 世界坐标系 到 相机坐标系

$$
\begin{aligned}
\left[\begin{array}{c}X_c\\Y_c\\Z_c\end{array}\right] =  
R \left[\begin{array}{c}X_w\\Y_w\\Z_w\end{array}\right] + t =
[R \quad t]
\left[\begin{array}{c}X_w\\Y_w\\Z_w\\1\end{array}\right]
\end{aligned}
$$

## 相机坐标系 到 像素坐标系
根据三角形相似关系，有  

$$
\frac{Z_c}{f} = \frac{X_c}{x} = \frac{Y_c}{y}
$$

整理，得

$$
\begin{cases}
x = f \cdot \frac{X_c}{Z_c} \\[2ex]
y = f \cdot \frac{Y_c}{Z_c}
\end{cases}
$$

**像素坐标系** 和 **成像平面坐标系** 之间，相差一个缩放和平移

$$
\begin{cases}
u = \alpha \cdot x + c_x \\[2ex]
v = \beta   \cdot y + c_y
\end{cases}
\text{或}
\begin{cases}
u =  \frac{x}{dx} + c_x \\[2ex]
v =  \frac{y}{dy} + c_y
\end{cases}
$$

整理，得

$$
\begin{cases}
u = \alpha f \frac{X_c}{Z_c} + c_x \\[2ex]
v = \beta   f \frac{Y_c }{Z_c} + c_y
\end{cases}
\text{或}
\begin{cases}
u = \frac{f}{dx} \frac{X_c}{Z_c} + c_x \\[2ex]
v = \frac{f}{dy} \frac{Y_c }{Z_c} + c_y
\end{cases}
$$

其中，

$$
\begin{cases}
f_x = \frac{f}{dx}\\[2ex]
f_y = \frac{f}{dy}
\end{cases}
\text{，}
\begin{cases}
dx = \frac{W_{sensor}}{W_{image}}\\[2ex]
dy = \frac{H_{sensor}}{H_{image}}
\end{cases}
\text{，}
\begin{cases}
f_{normal\_x} = \frac{f}{W_{sensor}}\\[2ex]
f_{normal\_y} = \frac{f}{H_{sensor}}
\end{cases}
$$

以$f_x$、$f_y$的方式表示为

$$
\begin{cases}
u = f_x \frac{X_c}{Z_c} + c_x \\[2ex]
v = f_y \frac{Y_c }{Z_c} + c_y
\end{cases}
$$
或 以$f_{normal\_x}$、$f_{normal\_y}$的方式表示为
$$
\begin{cases}
u = f_{normal\_x} W_{image} \frac{X_c}{Z_c} + c_x \\[2ex]
v = f_{normal\_y} H_{image} \frac{Y_c }{Z_c} + c_y
\end{cases}
$$

其中，  

*  $f$ 为镜头焦距，单位为米;
*  $\alpha$、$\beta$ 的单位为像素/米;
*  $dx$、$dy$ 为传感器x轴和y轴上单位像素的尺寸大小，单位为米/像素;
*  $f_x$、$f_y$ 为x、y方向的焦距，单位为像素;
* $f_{normal\_x}$、$f_{normal\_y}$ 为x、y方向的归一化焦距;
*  $(c_x,c_y)$ 为主点，图像的中心，单位为像素。  

最终，写成矩阵的形式为：

$$
\begin{aligned}
\left[\begin{array}{c}u\\v\\1\end{array}\right] =  
\frac{1}{Z_c}
\left[\begin{array}{ccc}
f_x&0&c_x\\0&f_y&c_y\\0&0&1
\end{array}\right]  
\left[\begin{array}{c}X_c\\Y_c\\Z_c\end{array}\right]
\end{aligned}
$$

按照传统的习惯将$Z_c$移到左侧

$$
\begin{aligned}
Z_c\left[\begin{array}{c}u\\v\\1\end{array}\right] =  
\left[\begin{array}{ccc}
f_x&0&c_x\\0&f_y&c_y\\0&0&1
\end{array}\right]  
\left[\begin{array}{c}X_c\\Y_c\\Z_c\end{array}\right]
\end{aligned}
$$



# 鱼眼模型 (fisheye)

Given a point $ X=[x_c \quad  y_c \quad  z_c] $ in camera coordinates, the fisheye projection is:

$$
r = \sqrt{x_c^2 + y_c^2} \\
\theta = atan2(r, |z|) \\
\theta_d = \theta (1 + k1 * \theta^2 + k2 * \theta^4 + k3 * \theta^6 + k4 * \theta^8) \\
x' = \frac{\theta_d * x_c} {r} \\
y' = \frac{\theta_d * y_c} {r} \\
u = f_x (x' + \alpha y') + c_x \\
v = f_y y' + c_y
$$

* [Fisheye camera model](https://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html)


# 畸变模型

## 多项式畸变模型 (radial-tangential)
透镜的畸变主要分为径向畸变和切向畸变。  
**径向畸变** 是由于透镜形状的制造工艺导致，且越向透镜边缘移动径向畸变越严重，实际情况中我们常用r=0处的泰勒级数展开的前几项来近似描述径向畸变，矫正径向畸变前后的坐标关系为：

$$
\begin{cases}
x_{distorted} = x (1+k_1r^2+k_2r^4+k_3r^6)\\[2ex]
y_{distorted} = y (1+k_1r^2+k_2r^4+k_3r^6)
\end{cases}
$$

**切向畸变** 是由于透镜和CMOS或者CCD的安装位置误差导致，切向畸变需要两个额外的畸变参数来描述，矫正前后的坐标关系为：

$$
\begin{cases}
x_{distorted} = x + 2p_1xy + p_2(r^2+2x^2)\\[2ex]
y_{distorted} = y + 2p_2xy + p_1(r^2+2y^2)
\end{cases}
$$

联合上式，整理得

$$
\begin{cases}
x_{distorted} = x (1+k_1r^2+k_2r^4+k_3r^6) + 2p_1xy + p_2(r^2+2x^2)\\[2ex]
y_{distorted} =  y (1+k_1r^2+k_2r^4+k_3r^6) + 2p_2xy + p_1(r^2+2y^2)
\end{cases}
$$

其中，$r^2 = x^2 + y^2$
综上，我们一共需要5个畸变参数 $(k_1, k_2, k_3, p_1, p_2)$ 来描述透镜畸变。

## FOV畸变模型

$$
r_d = \frac{1}{\omega}arctan(2r_u tan(\frac{\omega}{2}))
$$

其中，${r_u}^2 = x^2 + y^2$，$\omega$为FOV畸变参数。   

使用下式应用该畸变模型：

$$
\begin{cases}
x_{distorted} = \frac{r_u}{r_d}x\\[2ex]
y_{distorted} =  \frac{r_u}{r_d}y
\end{cases}
$$

## Distortion Models in ROS
* **plumb_bob**: a 5-parameter polynomial approximation of radial and tangential distortion
* **rational_polynomial**: an 8-parameter rational polynomial distortion model

## 畸变矫正

* [[图像]畸变校正详解](https://blog.csdn.net/humanking7/article/details/45037239)

参考：[Straight lines have to be straight: automatic calibration and removal of distortion from scenes of structured enviroments](https://hal.archives-ouvertes.fr/inria-00267247/document)    

# 其他

## ATAN model

## OmniCam model

## [Supported models in Kalibr](https://github.com/ethz-asl/kalibr/wiki/supported-models)
