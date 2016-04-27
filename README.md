Be sure to have opencv installed and compiled with cuda before trying to compile

Below is the cmake command we used to compile opencv:

```bash
cmake -D WITH_TBB=ON -D WITH_EIGEN=ON -D WITH_OPENCL=ON -D WITH_CUDA=ON -D BUILD_opencv_gpu=ON -D BUILD_opencv_gpuarithm=ON -D BUILD_opencv_gpubgsegm=ON -D BUILD_opencv_gpucodec=ON -D BUILD_opencv_gpufeatures2d=ON -D BUILD_opencv_gpufilters=ON -D BUILD_opencv_gpuimgproc=ON -D BUILD_opencv_gpulegacy=ON -D BUILD_opencv_gpuoptflow=ON -D BUILD_opencv_gpustereo=ON -D BUILD_opencv_gpuwarping=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_FFMPEG=ON CUDA_GENERATION=Kepler -D WITH_GTK=ON  -D BUILD_DOCS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=ON WITH_VTK=ON WITH_OPENGL=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_QT=ON WITH_V4L=ON ..
```
