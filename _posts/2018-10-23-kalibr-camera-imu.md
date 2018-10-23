---
layout: post
title:  "Kalibr 之 Camera-IMU 标定 (总结)"
date:   2018-10-23
categories: Robotics
tags: [Sensor Calibration]
---

[TOC]

# Overview
[ethz-asl/kalibr](https://github.com/ethz-asl/kalibr) is a toolbox that solves the following calibration problems:
* **Multiple camera calibration**: intrinsic and extrinsic calibration of a camera-systems with non-globally shared overlapping fields of view
* **Camera-IMU calibration**: spatial and temporal calibration of an IMU w.r.t a camera-system
* **Rolling Shutter Camera calibration**: full intrinsic calibration (projection, distortion and shutter parameters) of rolling shutter cameras

# 1. 标定 Camera

## kalibr_calibrate_cameras

### 采集 images

* 单目
  ```
  rosbag record /camera/fisheye/image_raw -O images.bag
  ```

* 双目
  ```
  rosbag record /stereo/left/image_raw /stereo/right/image_raw -O images.bag
  ```

### 标定 Camera

* 单目
  > kalibr_calibrate_cameras --target april_6x6_50x50cm.yaml --bag images.bag --models pinhole-fov --topics /camera/fisheye/image_raw

* 双目
  > kalibr_calibrate_cameras --target april_6x6_50x50cm.yaml --bag images.bag --models pinhole-radtan pinhole-radtan --topics /stereo/left/image_raw /stereo/right/image_raw

## Other Camera Calib Tools

## 输出 cam_chain.yaml

* 单目
  sample file output:
  ```yaml
  cam_overlaps: []
    camera_model: pinhole
    distortion_coeffs: [0.9183540411447179]
    distortion_model: fov
    intrinsics: [252.40344712951838, 253.29272771389083, 310.9288373770512, 227.37425906476517]
    resolution: [640, 480]
    rostopic: /camera/fisheye/image_raw
  ```

* 双目


# 2. 标定 IMU

## [imu_utils](https://github.com/gaowenliang/imu_utils)

### 采集 IMU 数据

* collect the data while the IMU is **Stationary**, with a **two hours** duration

```
rosbag record /camera/imu/data_raw -O imu.bag
```

### 标定 IMU

```
rosbag play -r 200 imu.bag
roslaunch imu_utils ZR300.launch
```

**ZR300.launch** 文件内容

```xml
<launch>
    <node pkg="imu_utils" type="imu_an" name="imu_an" output="screen">
        <param name="imu_topic" type="string" value= "/camera/imu/data_raw"/>
        <param name="imu_name" type="string" value= "ZR300"/>
        <param name="data_save_path" type="string" value= "$(find imu_utils)/data/"/>
        <param name="max_time_min" type="int" value= "80"/>
        <param name="max_cluster" type="int" value= "100"/>
    </node>
</launch>
```

输出 **ZR300_imu_param.yaml**，sample file output:

```yaml
%YAML:1.0
---
type: IMU
name: ZR300
Gyr:
   unit: " rad/s"
   avg-axis:
      gyr_n: 2.7878706973951564e-03
      gyr_w: 1.6503780396374297e-05
   x-axis:
      gyr_n: 3.2763884944799469e-03
      gyr_w: 1.8012497709865783e-05
   y-axis:
      gyr_n: 2.7204386280639753e-03
      gyr_w: 1.6637042617714669e-05
   z-axis:
      gyr_n: 2.3667849696415461e-03
      gyr_w: 1.4861800861542444e-05
Acc:
   unit: " m/s^2"
   avg-axis:
      acc_n: 2.5172832889483965e-02
      acc_w: 4.4150867224248972e-04
   x-axis:
      acc_n: 2.4450765767551903e-02
      acc_w: 4.0728821351916671e-04
   y-axis:
      acc_n: 2.1474226370935746e-02
      acc_w: 2.1468705215157706e-04
   z-axis:
      acc_n: 2.9593506529964245e-02
      acc_w: 7.0255075105672530e-04
```

## Other Camera Calib Tools

## 输出 imu.yaml

根据标定结果修改 **imu.yaml**，其文件内容为  

```yaml
#Accelerometers
accelerometer_noise_density: 2.52e-02   #Noise density (continuous-time)
accelerometer_random_walk:   4.41e-04   #Bias random walk

#Gyroscopes
gyroscope_noise_density:     2.78e-03   #Noise density (continuous-time)
gyroscope_random_walk:       1.65e-05   #Bias random walk

rostopic:                    /camera/imu/data_raw   #the IMU ROS topic
update_rate:                 200.0      #Hz (for discretization of the values above)
```


# 3. 标定 Camera-IMU

## 采集 images & imu 数据

* 单目 + IMU
  ```
  rosbag record /camera/imu/data_raw /camera/fisheye/image_raw -O images_imu.bag
  ```

* 双目 + IMU
  ```
  rosbag record /camera/imu/data_raw /stereo/left/image_raw /stereo/right/image_raw -O images_imu.bag
  ```

## 标定

> kalibr_calibrate_imu_camera --target april_6x6_50x50cm.yaml --bag images_imu.bag --bag-from-to 5 45 --cam camchain.yaml --imu imu.yaml --time-calibration --imu-models scale-misalignment --timeoffset-padding 0.1

* **--time-calibration**: the **temporal calibration** is turned off by default and can be enabled using the argument
* **--bag-from-to 5 45**: because there are shocks in the dataset (sensor pick-up/lay-down), only the data between 5 to 45 s is used

## 输出 camchain-imucam.yaml
