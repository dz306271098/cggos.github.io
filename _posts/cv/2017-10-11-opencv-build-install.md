---
layout: post
title:  "OpenCV编译安装配置总结"
date:   2017-10-11
categories: ComputerVision
---

﻿[TOC]

# Linux

## Compilation and Installation

### Dependencies

```sh
sudo apt install build-essential  
sudo apt install  libgtk2.0-dev libavcodec-dev libavformat-dev  libtiff4-dev  libswscale-dev libjasper-dev
sudo apt install cmake pkg-config
```

### CMake

```sh
#!/usr/bin/env bash
cmake \
  -D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_INSTALL_PREFIX=/usr/local/opencv_249 \
	-D WITH_VTK=OFF \
	-D WITH_MATLAB=OFF \
	-D WITH_TBB=ON \
	-D WITH_IPP=OFF \
	-D WITH_FFMPEG=OFF \
	-D WITH_V4L=ON \
	-D WITH_CUDA=OFF \
	-D CUDA_GENERATION=Kepler \
	-D ENABLE_PRECOMPILED_HEADERS=OFF \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	..
```

or by the CMake curses interface `ccmake`

### Make & Install

```sh
time make -j4
sudo make install
```

### Errors

* /usr/local/include/c++/6.2.0/cstdlib:75:25: fatal error: stdlib.h: No such file or directory
	```sh
	-D ENABLE_PRECOMPILED_HEADERS=OFF
	```

* nvcc fatal   : Unsupported gpu architecture 'compute_11'
CMake Error at cuda_compile_generated_matrix_operations.cu.o.cmake:206
	```sh
	-D CUDA_GENERATION=Kepler
	# When using cmake to do configurations,
	# set the option CUDA_GENERATION to specific your GPU architecture
	```

* opencv/modules/videoio/src/ffmpeg_codecs.hpp:111:7: error: ‘CODEC_ID_H263P’ was not declared in this scope
	```sh
	-D WITH_FFMPEG=OFF
	```

### Other Tutorials
* [opencv安装指南](http://www.cnblogs.com/zjutzz/p/6714490.html)
* [Install OpenCV 3 on Ubuntu 15.10](http://auronc.logdown.com/posts/336662-install-opencv-3-on-ubuntu-1510)
* [Ubuntu下编译安装OpenCV 2.4.7并读取摄像头](http://www.cnblogs.com/liu-jun/archive/2013/12/24/3489675.html)
* [UBUNTU 14.04: INSTALL OPENCV WITH CUDA](http://blog.aicry.com/ubuntu-14-04-install-opencv-with-cuda/)
* [Installing OpenCV on Debian Linux](https://indranilsinharoy.com/2012/11/01/installing-opencv-on-linux/)
* [Building / Cross Compiling OpenCV for Linux ARM](http://www.ridgesolutions.ie/index.php/2013/05/24/building-cross-compiling-opencv-for-linux-arm/)
* [Cross compiling Opencv 3 for ARM](http://magicsmoke.co.za/?p=375)

## Check Informations

```sh
# 查看opencv版本
pkg-config --modversion opencv

# 查看opencv包含目录
pkg-config --cflags opencv

# 查看opencv库目录
pkg-config --libs opencv
```

## Using

### Compilation

```sh
g++ `pkg-config opencv --cflags` test.cpp -o test `pkg-config opencv --libs`
```

## Multiple OpenCV

### Installation
[Ubuntu 15.04 Opencv 安装（多版本并存）](http://blog.csdn.net/cumt08113684/article/details/53006376)

### Using

#### Using CMake

在opencv编译好后，所在目录中一般会有一个叫OpenCVConfig.cmake的文件，这个文件指定了CMake要去哪里找OpenCV，设置OpenCV_DIR为包含OpenCVConfig.cmake的目录(可设置CMAKE_MODULE_PATH)，如在C++工程CMakeLists.txt中添加

```sh
set(OpenCV_DIR "/home/ubuntu/src/opencv-3.1.0/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
```

因此，使用哪个版本的Opencv，只要找到对应的OpenCVConfig.cmake文件，并且将其路径添加到工程的CMakeLists.txt中即可。

#### Using Makefile

将opencv-3.1.0.pc和opencv-2.4.12.pc拷贝到/usr/lib/pkgconfig目录(可设置PKG_CONFIG_PATH)下，
使用opencv-3.1.0时，Makefile中为：

```sh
COMMON  += -DOPENCV
CFLAGS  += -DOPENCV
LDFLAGS += `pkg-config --libs opencv-3.1.0`
COMMON  += `pkg-config --cflags opencv-3.1.0`
```

使用opencv-2.4.12时，Makefile中为：

```sh
COMMON  += -DOPENCV
CFLAGS  += -DOPENCV
LDFLAGS += `pkg-config --libs opencv-2.4.12`
COMMON  += `pkg-config --cflags opencv-2.4.12`
```

## Uninstall

```sh
sudo make uninstall
or
#install-mainfest.txt包含了安装文件的路径
sudo cat install-manifest.txt | sudo xargs rm
```

# Windows
