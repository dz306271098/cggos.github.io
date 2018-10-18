---
layout: post
title: "Sensors in Robotics"
date: 2018-09-01
categories: Sensors
tags: [Robotics, Sensors]
---

[TOC]

# 1. Cameras

* [Sensors/Cameras (ROS Wiki)](http://wiki.ros.org/Sensors/Cameras)
* [Set up a Webcam with Linux](http://www.linuxintro.org/wiki/Set_up_a_Webcam_with_Linux)
* [Accessing the Video Device](https://www.tldp.org/HOWTO/Webcam-HOWTO/dev-intro.html)
* v4l2-ctl
* viewer: [GTK+ UVC Viewer](http://guvcview.sourceforge.net/index.html)
* Create Video Device ( /dev/video1 )
  ```
  sudo mknod /dev/video1 c 81 1
  sudo chmod 666 /dev/video1
  sudo chgrp video /dev/video1
  ```

## Camera Lenses
  * [Lensation](https://www.lensation.de/) provides free of charge consulting about lenses, illumination and optical components
  * [dxomark](https://www.dxomark.com/): source for image quality benchmarks for phones and cameras


## Camera Modules

* [In Search of a Better Serial Camera Module](http://sigalrm.blogspot.com/2013/07/in-search-of-better-serial-camera-module.html)

### [CMUcam](http://www.cmucam.org/)
**Open Source Programmable Embedded Color Vision Sensors**, The first CMUcam made its splash in 1999 as a CREATE Lab project.

* [The CMUcam1 Vision Sensor](https://www.cs.cmu.edu/~cmucam/qanda.html)  
![CMUcam1_B.JPG](../images/Sensors/CMUcam1_B.JPG)

#### [Pixy](https://pixycam.com/)

![pixy_cam.jpg](../images/Sensors/pixy_cam.jpg)

**Pixy** is **the fifth version of the CMUcam, or CMUcam5**, but “Pixy” is easier to say than CMUcam5, so the name more or less stuck.  Pixy got its start in 2013 as part of a successful Kickstarter campaign, and as a partnership between **Charmed Labs** and **CMU**.

**Pixy2** was announced recently as Pixy’s smaller, faster, and smarter younger sibling.  
![pixy2_cam.jpg](../images/Sensors/pixy2_cam.jpg)

### [OpenMV](https://openmv.io/)
The OpenMV(**Open-Source Machine Vision**) project aims at making machine vision more accessible to beginners by developing a user-friendly, open-source, low-cost **machine vision platform**.  

OpenMV cameras are programmable in **Python3** and come with an extensive set of **image processing functions** such as face detection, keypoints descriptors, color tracking, QR and Bar codes decoding, AprilTags, GIF and MJPEG recording and more.  

![openmv_cam.jpg](../images/Sensors/openmv_cam.jpg)

### [NXTCam-v4](http://www.mindsensors.com/ev3-and-nxt/14-vision-subsystem-camera-for-nxt-or-ev3-nxtcam-v4)
Vision Subsystem - Camera for NXT or EV3 (NXTCam-v4)  

![nxtcam_v4.jpg](../images/Sensors/nxtcam_v4.jpg)

## Typical Cameras

* [The IEEE1394/USB3 Digital Camera List](https://damien.douxchamps.net/ieee1394/cameras/)

### Kinect
* [Kinect for windows微软中国体感官方网站](http://www.k4w.cn/)
* [OpenKinect](https://openkinect.org/wiki/Main_Page) is an open community of people interested in making use of the amazing Xbox Kinect hardware with our PCs and other devices.
* [Kinect V1 and Kinect V2 fields of view compared](http://smeenk.com/kinect-field-of-view-comparison/)
* [Ubuntu + Kinect + OpenNI + PrimeSense](http://mitchtech.net/ubuntu-kinect-openni-primesense/)
* [【翻译】Kinect v1和Kinect v2的彻底比较](http://www.cnblogs.com/TracePlus/p/4136297.html)
* [code-iai/iai_kinect2](https://github.com/code-iai/iai_kinect2): Tools for using the Kinect One (Kinect v2) in ROS

### Realsense Camera
* [realsense_camera (ROS Wiki)](http://wiki.ros.org/realsense_camera)
* [Intel® RealSense­™ Camera ZR300](https://software.intel.com/en-us/realsense/zr300)

### Orbbec Astra Camera
* [astra_camera (ROS Wiki)](http://wiki.ros.org/astra_camera)
* [ROS wrapper for Astra camera](https://github.com/orbbec/ros_astra_camera)

### ASUS Xtion 2
* https://www.asus.com/3D-Sensor/

### ZED Stereo Camera
* [StereoLabs](https://www.stereolabs.com/)

### Event Camera
* [Event Camera动态视觉传感器，让无人机用相机低成本进行导航](https://www.leiphone.com/news/201709/LkfPqS60ZYgmXk8x.html)


## Camera Calibration

* [Tutorial Camera Calibration](http://boofcv.org/index.php?title=Tutorial_Camera_Calibration)

### [camera_calibration (ROS Wiki)](http://wiki.ros.org/camera_calibration)

### [Camera Calibration Toolbox for Matlab](http://www.vision.caltech.edu/bouguetj/calib_doc/)

### [CamOdoCal](https://github.com/hengli/camodocal)
Automatic Intrinsic and Extrinsic Calibration of a Rig with Multiple Generic Cameras and Odometry.This C++ library supports the following tasks:
* Intrinsic calibration of a generic camera.  
* Extrinsic self-calibration of a multi-camera rig for which odometry data is provided.  
* Extrinsic infrastructure-based calibration of a multi-camera rig for which a map generated from task  2 is provided.

### OpenCV
* [Camera calibration With OpenCV](http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html)
* [Interactive camera calibration application](http://docs.opencv.org/3.2.0/d7/d21/tutorial_interactive_calibration.html)
* [Camera calibration using C++ and OpenCV](http://sourishghosh.com/2016/camera-calibration-cpp-opencv/)
* [Stereo calibration using C++ and OpenCV](http://sourishghosh.com/2016/stereo-calibration-cpp-opencv/)
* [Stereo Calibration](http://www.jayrambhia.com/blog/stereo-calibration)
* [张氏法相机标定](https://zhuanlan.zhihu.com/p/24651968)
* [Camera Calibration in SVO](https://github.com/uzh-rpg/rpg_svo/wiki/Camera-Calibration)
* [Calibrate fisheye lens using OpenCV](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)
* [OpenCV C++ Stereo Fisheye Calibration](https://github.com/sourishg/fisheye-stereo-calibration)

Stereo Camera Calibration in ROS:
```
#!/usr/bin/env bash
rosrun camera_calibration cameracalibrator.py --approximate=0.1 --size 11x8 --square 30 \
right:=/stereo/right/image_raw left:=/stereo/left/image_raw
```

## Image Rectification
* [Stereo Fisheye Rectification](https://github.com/ShreyasSkandanS/stereo_fisheye_rectify)


-----

# 2. IMU

## IMU Tools
* **IMU Pose Calculation**
  - [ccny-ros-pkg/imu_tools](https://github.com/ccny-ros-pkg/imu_tools): ROS tools for IMU devices

* **IMU Performance Analysis**
  - [IMU-TK](https://bitbucket.org/alberto_pretto/imu_tk): Inertial Measurement Unit ToolKit
  - [gaowenliang/imu_utils](https://github.com/gaowenliang/imu_utils): A ROS package tool to analyze the IMU performance
  - [rpng/kalibr_allan](https://github.com/rpng/kalibr_allan): IMU Allan standard deviation charts for use with Kalibr and inertial kalman filters
  - [AllanTools](https://pypi.org/project/AllanTools/): A python library for calculating Allan deviation and related time & frequency statistics.


-----

# 3. LiDAR

![RPLIDAR A1](../images/Sensors/rplidar.png)

* [LiDARNews](https://lidarnews.com/)
* [rplidar (ROS Wiki)](http://wiki.ros.org/rplidar)
* [RPLIDAR A1 (slamtec)](http://www.slamtec.com/en/lidar/a1)
* [在自动驾驶中，单线激光雷达能干什么?](https://www.leiphone.com/news/201612/kEUZbebrEA2WJRVE.html)

## Application

* 运行 `roslaunch rplidar_ros view_rplidar.launch`，效果如下
![view_rplidar](../images/Sensors/view_rplidar.png)

* 通过 **hector_slam** 建图，运行 `roslaunch rplidar_ros view_slam.launch`，效果如下
![view_slam](../images/Sensors/view_slam.png)


-----

# 4. UltraSonic

![ultrasonic](../images/Sensors/ultrasonic.jpg)


-----

# 5. Sensor Calibration

## Kalibr

[Kalibr](https://github.com/ethz-asl/kalibr) is a toolbox that solves the following calibration problems:  

* Multiple camera calibration
* Camera-IMU calibration
* Rolling Shutter Camera calibration


-----

# 6. Time Synchronization
* [ROS CAMERA AND IMU SYNCHRONIZATION](http://grauonline.de/wordpress/?page_id=1951)


-----

# 7. Sensors Fusion
