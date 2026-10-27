# re2_lib

## re2 version
- gitlib: https://gitlab.oritek.com.cn/npu/opensource/re2.git
- commit: 6dcd83d60f7944926bfd308cc13979fc53dd69ca

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
-DSPM_ENABLE_SHARED=OFF \
-DSPM_PROTOBUF_PROVIDER="internal" \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_STANDARD_REQUIRED=ON 
cmake --build build -j32
cmake --install build

#aarch64
cmake -B build -DCMAKE_INSTALL_PREFIX=./aarch64  \
-DCMAKE_TOOLCHAIN_FILE=compile_aarch64_linux.cmake \
-DSPM_ENABLE_SHARED=OFF \
-DSPM_PROTOBUF_PROVIDER="internal" \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_STANDARD_REQUIRED=ON 
cmake --build build -j32
cmake --install build
```