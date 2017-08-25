---
layout: post
title:  "Mathematical Libraries"
date:   2017-08-24 22:03:00 +0800
categories: Math
---

Mathematical libraries used in development and those usages with CMake.

### BLAS
The BLAS (Basic Linear Algebra Subprograms) are routines that provide standard building blocks for performing basic vector and matrix operations. The Level 1 BLAS perform scalar, vector and vector-vector operations, the Level 2 BLAS perform matrix-vector operations, and the Level 3 BLAS perform matrix-matrix operations. Because the BLAS are efficient, portable, and widely available, they are commonly used in the development of high quality linear algebra software, LAPACK for example.
### LAPACK
LAPACK is written in Fortran 90 and provides routines for solving systems of simultaneous linear equations, least-squares solutions of linear systems of equations, eigenvalue problems, and singular value problems.
```
# LAPACK
find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message(STATUS "LAPACK Libraries: ")
    foreach (lib ${LAPACK_LIBRARIES})
        message(STATUS "  " ${lib})
    endforeach()
    link_libraries( ${LAPACK_LIBRARIES} )
endif()
```
### TooN
TooN: Tom's Object-oriented numerics library.   
The TooN library is a set of C++ header files which provide basic linear algebra facilities.
```
# TooN
# Require linking the LAPACK library
```
### SuiteSparse
A Suite of Sparse matrix packages at http://www.suitesparse.com.
### Eigen
Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
```
# Eigen
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories( ${EIGEN3_INCLUDE_DIR} )
endif()
```
### Sophus
C++ implementation of Lie Groups using Eigen commonly used for 2d and 3d geometric problems (i.e. for Computer Vision or Robotics applications).
```
# Sophus
find_package( Sophus REQUIRED )
if(Sophus_FOUND)
    include_directories( ${Sophus_INCLUDE_DIRS} )
    link_libraries( ${Sophus_LIBRARIES} )
endif()
```
### Matrix Template Library (MTL2/MTL4)
The Matrix Template Library 4 (MTL4) is a development library for scientific computing that combines high productivity with high performance in the execution.
### Blitz++
Blitz++ is a C++ class library for scientific computing which provides performance on par with Fortran 77/90. It uses template techniques to achieve high performance. Blitz++ provides dense arrays and vectors, random number generators, and small vectors (useful for representing multicomponent or vector fields).
### FFTW
FFTW is a C subroutine library for computing the discrete Fourier transform (DFT) in one or more dimensions, of arbitrary input size, and of both real and complex data (as well as of even/odd data, i.e. the discrete cosine/sine transforms or DCT/DST).
### Ceres Solver
Ceres Solver [1] is an open source C++ library for modeling and solving large, complicated optimization problems. It can be used to solve Non-linear Least Squares problems with bounds constraints and general unconstrained optimization problems.
```
# Ceres
find_package( Ceres REQUIRED )
if(Ceres_FOUND)
    include_directories( ${CERES_INCLUDE_DIRS} )
    link_libraries( ${CERES_LIBRARIES} )
endif()
```
### G2O
g2o is an open-source C++ framework for optimizing graph-based nonlinear error functions. g2o has been designed to be easily extensible to a wide range of problems and a new problem typically can be specified in a few lines of code. The current implementation provides solutions to several variants of SLAM and BA.
```
# G2O
find_package( G2O REQUIRED )
if(G2O_FOUND)
    include_directories( ${G2O_INCLUDE_DIRS} )
    link_libraries( g2o_core g2o_stuff g2o_types_sba )
endif()
```
