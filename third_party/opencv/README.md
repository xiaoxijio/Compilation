# opencv lib
## opencv version
- gitlib: https://gitlab.oritek.com.cn/npu/opensource/opencv/-/tree/4.7.0

## Environment linux
**build environment**:
- Ubuntu 20.04.4 LTS
- gcc 9.4.0

**build cmd**
```
cmake .. -DWITH_IPP=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_ZLIP=ON -DBUILD_TIFF=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_PNG=ON -DBUILD_OPENEXR=ON -DBUILD_SHARED_LIBS=ON -DWITH_PROTOBUF=OFF -DBUILD_LIST=core,imgcodecs,imgproc -DCMAKE_BUILD_TYPE=Release -DOPENCV_PYTHON_SKIP_DETECTION=1
```

## Environment aarch64
**build environment**:
- Ubuntu 20.04.4 LTS
- cross compile: aarch64-none-linux-gnu-gcc

**build cmd**
```
cmake .. -DWITH_IPP=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_ZLIP=ON -DBUILD_TIFF=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_PNG=ON -DBUILD_OPENEXR=ON -DBUILD_SHARED_LIBS=ON -DWITH_PROTOBUF=OFF -DBUILD_LIST=core,imgcodecs,imgproc,video,videoio -DCMAKE_BUILD_TYPE=Release -DOPENCV_PYTHON_SKIP_DETECTION=1
```