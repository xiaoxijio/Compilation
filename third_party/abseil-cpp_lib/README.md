# abseil-cpp_lib

## abseil-cpp version
- gitlib: https://gitlab.oritek.com.cn/npu/opensource/abseil-cpp.git
- commit: 63322a26c046ba31e6ea2da8e6b54bb989c8608d

## Environment linux
**build environment**:
- Ubuntu 20.04.4 LTS
- gcc 9.4.0

## Environment aarch64
**build environment**:
- Ubuntu 20.04.4 LTS
- cross compile: aarch64-none-linux-gnu-gcc

## build cmd
```
#x86_64
cmake -B build -DCMAKE_INSTALL_PREFIX=./x86_64 \
-DABSL_ENABLE_INSTALL=ON \
-DABSL_PROPAGATE_CXX_STD=ON \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_STANDARD_REQUIRED=ON 
cmake --build build -j32
cmake --install build

#aarch64
cmake -B build -DCMAKE_INSTALL_PREFIX=./aarch64 \
-DCMAKE_TOOLCHAIN_FILE=compile_aarch64_linux.cmake \
-DABSL_ENABLE_INSTALL=ON \
-DABSL_PROPAGATE_CXX_STD=ON \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_STANDARD_REQUIRED=ON 
cmake --build build -j32
cmake --install build
```