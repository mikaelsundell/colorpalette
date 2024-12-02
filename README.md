Colorpalette
==================

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg?style=flat-square)](https://github.com/mikaelsundell/logctool/blob/master/README.md)

Introduction
------------

Colorpalette is a tool to process and create palettes of unique colors from images

![Sample image or figure.](images/image.png 'colorpalette')

Building
--------

The colorpalette app can be built both from commandline or using optional Xcode `-GXcode`.

```shell
mkdir build
cd build
cmake .. -DCMAKE_MODULE_PATH=<path>/brawtool/modules -DCMAKE_PREFIX_PATH=<path> -GXcode
cmake --build . --config Release -j 8
```

**Example using 3rdparty on arm64 with Xcode**

```shell
mkdir build
cd build
cmake ..
cmake .. -DCMAKE_PREFIX_PATH=<path>/3rdparty/build/macosx/arm64.debug -GXcode
```

Usage
-----

Print colorpalette help message with flag ```--help```.

```shell
Colorpalette -- a tool to process and create palettes of unique colors from images

Usage: colorpalette [options] filename...

General flags:
    --help                           Print help message
    -v                               Verbose status messages
    --colors COLORS                  Number of unique colors
Input flags:
    --inputfilename INFILENAME       Input filename of image
Output flags:
    --outputfilename OUTFILENAME     Output filename of palette image
    --posterfilename POSTERFILENAME  Output filename of poster image
```


Generate a color palette from an image
--------

```shell
./colorpalette
-v
--inputfilename /Volumes/Build/github/image.jpg
--outputfilename /Volumes/Build/github/imagepalette.jpg
```

Download
---------

Colorpalette is included as part of pipeline tools. You can download it from the releases page:

* https://github.com/mikaelsundell/pipeline/releases

Dependencies
-------------

| Project     | Description |
| ----------- | ----------- |
| OpenImageIO | [OpenImageIO project @ Github](https://github.com/OpenImageIO/oiio)
| OpenCV      | [OpenCV project @ Github](https://github.com/opencv/opencv)
| 3rdparty    | [3rdparty project containing all dependencies @ Github](https://github.com/mikaelsundell/3rdparty)


Project
-------

* GitHub page   
https://github.com/mikaelsundell/colorpalette
* Issues   
https://github.com/mikaelsundell/colorpalette/issues


Contributors
---------

* ChatGPT :-)
