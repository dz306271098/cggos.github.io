---
layout: post
title: "双目立体视觉三维重建"
date: 2018-07-22
categories: ComputerVision
tags: [Stereo Vision, 3D Reconstruction]
---

 [TOC]

# Overview

双目立体视觉的整体流程包括：  
* 图像采集
* 双目标定
* 双目矫正
* 立体匹配
* 三维重建

<div align=center>
  <img src="../images/stereo_vision/stereo_vision_system.png">
</div>


# 图像采集

双目相机采集 **左右目图像**

# 双目标定

通过 **双目标定工具** 对双目相机进行标定，得到如下结果参数：  

|内参|外参|
|:-:|:-:|
|相机矩阵 $K_1, K_2$|旋转矩阵 $R$|
|畸变系数 $D_1, D_2$|平移向量 $t$|  

《Learning OpenCV》中对于 Translation 和 Rotation 的图示是这样的:   

<div align=center>
  <img src="../images/stereo_vision/stereo_rt.jpg">
</div>

示例代码：  
```c++
cv::Matx33d K1, K2, R;
cv::Vec3d T;
cv::Vec4d D1, D2;

int flag = 0;
flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
flag |= cv::fisheye::CALIB_CHECK_COND;
flag |= cv::fisheye::CALIB_FIX_SKEW;

cv::fisheye::stereoCalibrate(
        obj_points_, img_points_l_, img_points_r_,
        K1, D1, K2, D2, img_size_, R, T,
        flag, cv::TermCriteria(3, 12, 0));
```

# 双目矫正

双目矫正 主要包括两方面：**畸变矫正** 和 **立体矫正** 。  

利用 OpenCV的函数，主要分为  

* stereoRectify
* initUndistortRectifyMap
* remap

（1）根据双目标定的结果 $K_1, K_2, D_1, D_2, R, t$，利用 OpenCV函数 **stereoRectify**，计算得到如下参数

* 左目 矫正矩阵(旋转矩阵) $R_1$ (3x3)
* 右目 矫正矩阵(旋转矩阵) $R_2$ (3x3)
* 左目 投影矩阵 $P_1$ (3x4)
* 右目 投影矩阵 $P_2$ (3x4)
* disparity-to-depth 映射矩阵 $Q$ (4x4)

其中，  

左右目投影矩阵（horizontal stereo, ${c_x}_1'={c_x}_2'$ if **CV_CALIB_ZERO_DISPARITY** is set）

$$
P_1 =
	\begin{bmatrix}
	f' & 0 & {c_x}_1' & 0 \\
	0 & f' & c_y' & 0 \\
	0 & 0 & 1   & 0
	\end{bmatrix}
$$

$$
P_2 =
	\begin{bmatrix}
	f' & 0 & {c_x}_2' & t_x' \cdot f' \\
	0 & f' & c_y' & 0 \\
	0 & 0 & 1   & 0
	\end{bmatrix}
$$

where

$$
t_x' = -B
$$

disparity-to-depth 映射矩阵

$$
Q =
	\begin{bmatrix}
	1 & 0 & 0 & -{c_x}_1' \\
	0 & 1 & 0 & -c_y'     \\
	0 & 0 & 0 & f'        \\
  0 & 0 & -\frac{1}{t_x'} & \frac{ {c_x}_1'-{c_x}_2'}{t_x'}
	\end{bmatrix}
$$

通过 $P_2$ 可计算出 **基线** 长度:

$$
\begin{aligned}
    baseline = B = - t_x' = - \frac{ {P_2}_{(03)} }{f'}
\end{aligned}
$$

示例代码：  
```c++
cv::Mat R1, R2, P1, P2, Q;
cv::fisheye::stereoRectify(
        K1, D1, K2, D2, img_size_, R, T,
        R1, R2, P1, P2, Q,
        CV_CALIB_ZERO_DISPARITY, img_size_, 0.0, 1.1);
```

（2）左右目 分别利用 OpenCV函数 **initUndistortRectifyMap** 计算 **the undistortion and rectification transformation map**，得到

* 左目map: $map^l_1, map^l_2$
* 右目map: $map^r_1, map^r_2$

示例代码：  
```c++
cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_16SC2, rect_map_[0][0], rect_map_[0][1]);
cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_16SC2, rect_map_[1][0], rect_map_[1][1]);
```

（3）左右目 分别利用 OpenCV函数 **remap** 并根据 **左右目map** 对左右目图像进行 **去畸变 和 立体矫正**，得到 **左右目矫正图像**

示例代码：  
```c++
cv::remap(img_l, img_rect_l, rect_map_[0][0], rect_map_[0][1], cv::INTER_LINEAR);
cv::remap(img_r, img_rect_r, rect_map_[1][0], rect_map_[1][1], cv::INTER_LINEAR);
```

# 立体匹配

根据双目矫正图像，通过 **BM或SGM等立体匹配算法** 对其进行立体匹配，计算 **视差图**

<div align=center>
  <img src="../images/stereo_vision/stereo_vision_model_01.png">
</div>

## 视差计算
通过 OpenCV函数 **stereoBM** (block matching algorithm)，生成 **视差图(Disparity Map)** (CV_16S or CV_32F)

> disparity map from stereoBM of OpenCV :
> It has the same size as the input images. When disptype == CV_16S, the map is a 16-bit signed single-channel image, containing disparity values scaled by 16. To get the true disparity values from such fixed-point representation, you will need to divide each disp element by 16. If disptype == CV_32F, the disparity map will already contain the real disparity values on output.

So if you've chosen **disptype = CV_16S** during computation, you can access a pixel at pixel-position (X,Y) by: `short pixVal = disparity.at<short>(Y,X);`, while the disparity value is `float disparity = pixVal / 16.0f;`; if you've chosen **disptype = CV_32F** during computation, you can access the disparity directly: `float disparity = disparity.at<float>(Y,X);`

* [Disparity Map](http://www.jayrambhia.com/blog/disparity-mpas)
* [Disparity map post-filtering](https://docs.opencv.org/3.1.0/d3/d14/tutorial_ximgproc_disparity_filtering.html)

# 三维重建

（1）算法1：根据视差图，利用 $f'$ 和 $B$ 通过几何关系计算 **深度值**，并利用相机内参计算 **三维坐标**

<div align=center>
  <img src="../images/stereo_vision/stereo_vision_model_02.png">
</div>

深度计算公式如下，通过遍历图像可生成 **深度图**

$$
\begin{aligned}
	Z = depth = \frac{f' \cdot B}{d_p} \\
  d_p = disp(u,v) + ({c_x}_2' - {c_x}_1')
\end{aligned}
$$

根据 **小孔成像模型**，已知 $Z$ 和 **相机内参** 可计算出 三维点坐标，从而可生成 **三维点云**

$$
\begin{aligned}
	\begin{cases}
	Z = depth = \frac{f' \cdot B}{d_p} \\
	X = \frac{u-{c_x}_1'}{f'} \cdot Z \\
	Y = \frac{v-{c_y}'}{f'} \cdot Z
	\end{cases}
\end{aligned}
\text{或}
\begin{aligned}
	\begin{cases}
  bd = \frac{B}{d_p}\\
	Z = depth = f' \cdot bd \\
	X = (u-{c_x}_1') \cdot bd \\
	Y = (u-{c_y}') \cdot bd
	\end{cases}
\end{aligned}
$$


其中，$disp(u,v)$ 代表 视差图 坐标值  

（2）算法2：根据视差图，利用 **$Q$ 矩阵** 计算 三维点坐标（**reprojectImageTo3D**）

$$
\begin{bmatrix} X' \\ Y' \\ Z' \\ W \end{bmatrix} =
Q \cdot
\begin{bmatrix} u \\ v \\ disp(u,v) \\ 1 \end{bmatrix}
$$

最终，三维点坐标为

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} =
\begin{bmatrix}
\frac{X'}{W} \\[2ex]
\frac{Y'}{W} \\[2ex]
\frac{Z'}{W}
\end{bmatrix}
$$

## 深度图 图像类型

* 单位meter --> 32FC1
* 单位millimeter --> 16UC1

# 总结

<div align=center>
  <img src="../images/stereo_vision/stereo_vision_note.jpg">
</div>
