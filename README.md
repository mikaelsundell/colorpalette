Readme for colorpalette
==================

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg?style=flat-square)](https://github.com/mikaelsundell/logctool/blob/master/README.md)

Introduction
------------

Colorpalette is a tool to process and create palettes of unique colors from images

![Sample image or figure.](images/image.png 'logctool')

Building
--------

The colorpalette app can be built both from commandline or using optional Xcode `-GXcode`.

```shell
mkdir build
cd build
cmake .. -DCMAKE_MODULE_PATH=<path> -DCMAKE_INSTALL_PREFIX=<path> -DCMAKE_PREFIX_PATH=<path> -GXcode
cmake --build . --config Release -j 8
```

**Example using 3rdparty on arm64**

```shell
mkdir build
cd build
cmake ..
cmake .. -DCMAKE_MODULE_PATH=<path>/3rdparty/build/macosx/arm64.debug -DCMAKE_INSTALL_PREFIX=<path>/3rdparty/build/macosx/arm64.debug -DBUILD_SHARED_LIBS=TRUE -GXcode
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

Packaging
---------

The `macdeploy.sh` script will deploy mac bundle to dmg including dependencies.

```shell
./macdeploy.sh -e <path>/colorpalette -d <path> -p <path>
```

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
