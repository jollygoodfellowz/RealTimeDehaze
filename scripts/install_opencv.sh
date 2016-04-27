#!/bin/bash

cd ~
yum -y groupinstall "Development Tools" 
yum -y install gcc 
yum -y install cmake 
yum -y install git
yum -y install gtk2-devel
yum -y install pkgconfig 
yum -y install numpy 
yum -y install ffmpeg
yum -y install cmake.x86_64
yum -y install http://dl.fedoraproject.org/pub/epel/6/i386/epel-release-6-8.noarch.rpm
yum -y groupinstall "Development Tools"
yum -y install wget unzip opencv opencv-devel gtk2-devel cmake
mkdir /opt/working
cd /opt/working
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout tags/2.4.8.2
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j
make install

