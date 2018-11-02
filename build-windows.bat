set rootdir=%cd%
::{ protobuf begin
set version=3.4.0
echo "protobuf v%version%"
set protobuf_file=v%version%.zip
if exist %protobuf_file% (
    echo Found %protobuf_file%
) else (
    echo download ...
    wget https://github.com/google/protobuf/archive/v%version%.zip
)

if not exist %protobuf_file% (
    echo "protobuf source file not found"
    echo "You can download manually from https://github.com/google/protobuf/archive/v%version%.zip"
    exit
)
echo "extract ..."
unzip -o -q %protobuf_file%

cd protobuf-%version%

mkdir install

set gitbin=C:\Program Files\Git\cmd

set cmakebin=C:\Program Files\CMake3.8\bin

set PATH=%gitbin%;%cmakebin%;%PATH%

cd cmake

mkdir build & cd build

mkdir release & cd release

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF  -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX=../../../../install ../..
 
nmake & nmake install

cd ..

:: vs solution
mkdir sln & cd sln
cmake -G "Visual Studio 14 2015 Win64"  -DCMAKE_INSTALL_PREFIX=../../../../install ../..
cd ..

echo "done protobuf"
:: protobuf end }

cd %rootdir%
::exit /b
:: compile 
echo "building tool"
set target="NMake Makefiles"
mkdir build-windows & cd build-windows
cmake -G%target% -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%rootdir%/install -DProtobuf_INCLUDE_DIR=%rootdir%/install/include -DProtobuf_LIBRARIES=%rootdir%/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=%rootdir%/install/bin/protoc.exe ..
nmake
nmake install
copy /y tools\*.exe         %rootdir%\install\bin
copy /y tools\caffe\*.exe   %rootdir%\install\bin
copy /y tools\mxnet\*.exe   %rootdir%\install\bin
copy /y tools\onnx\*.exe    %rootdir%\install\bin
cd ..

:: vs solution
set target="Visual Studio 14 2015 Win64"
mkdir sln & cd sln
cmake -G%target% -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%rootdir%/install -DProtobuf_INCLUDE_DIR=%rootdir%/install/include -DProtobuf_LIBRARIES=%rootdir%/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=%rootdir%/install/bin/protoc.exe ..
cd ..

cd %rootdir%
echo "done !"







