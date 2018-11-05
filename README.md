![](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png)

[ncnn](https://github.com/Tencent/ncnn) is a a high-performance neural network inference computing framework optimized for mobile platforms. More details can be found here [ncnn](https://github.com/Tencent/ncnn).

This project offers tools for building on **Windows x64**. And some C-style **wrappers** for using the library.



# Build for Windows

### Depends
- Visual Studio 2015 or 2017
- CMake (>3.2)

###  Environment Setup

Open the appropriate *Command Prompt* from the *Start* menu.

For example *VS2015 x64 Native Tools Command Prompt*:

```
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC>
```

Change to your working directory:

```
cd /to/project/dir
```

If *cmake* command is not available from *Command Prompt*, add it to system *PATH* variable:

```
set PATH=%PATH%;C:\Program Files (x86)\CMake\bin
```

###  Build
```
build-windows.bat
```

If succeed,  results will be in the **`install`** directory and Visual Studio Solution will be in the **`sln`** directory



# Build for Android

### Depends
[android-ndk]( http://developer.android.com/ndk/downloads/index.html)(<16)

### Build
```
set ANDROID_NDK=/your/ndk/path
build.bat
```
Build files will be here **`build-android\install`** 



# Wrapper

Instructions of using the original [ncnn](https://github.com/Tencent/ncnn) APIs can be found [here](https://github.com/Tencent/ncnn/wiki/how-to-use-ncnn-with-alexnet). Our C-style wrapper arms to make it easier to use. These wrappers were orignally developed for [caffe](https://github.com/BVLC/caffe) and we keep the heading `caffe_` just for compatible.

### Souce:  *src/wrapper*

### Usage:

| Method          | Code                                     | Note          |
| --------------- | ---------------------------------------- | ------------- |
| Load a net      | handle=caffe_net_load(const char* prototxt, const char* weights) |               |
| Set input       | caffe_net_setBlob(handle, "data", input_blob) | *             |
| Forward         | caffe_net_forward(handle)                |               |
| Get output      | out_blob = caffe_net_getBlob(handle, "feat") | *             |
| Get layer names | std::vector\<std::string\> names = caffe_net_getLayerNames(handle) |               |
| Clean up        | input_blob.destroy();<br />out_blob.destory() | destory blobs |
| Release handle  | caffe_net_release(handle)                |               |
