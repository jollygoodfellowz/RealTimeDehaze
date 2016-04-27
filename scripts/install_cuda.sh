#!/bin/bash

mkdir downloads
cd downloads/
yum -y install pciutils
yum -y install nano
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-rhel7-7-5-local-7.5-18.x86_64.rpm
wget http://linux.dell.com/dkms/permalink/dkms-2.2.0.3-1.noarch.rpm
rpm -Uvh dkms-2.2.0.3-1.noarch.rpm 
rpm -Uvh cuda-repo-rhel7-7-5-local-7.5-18.x86_64.rpm
yum -y install epel-release
yum -y install cuda
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/352.55/NVIDIA-Linux-x86_64-352.55.run
chmod +x NVIDIA-Linux-x86_64-352.55.run 
./NVIDIA-Linux-x86_64-352.55.run 

echo "# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:\$LD_LIBRARY_PATH
PATH=\$PATH:$HOME/.local/bin:\$HOME/bin:/usr/local/cuda-7.5/bin

export PATH"> .bash_profile

source .bash_profile
