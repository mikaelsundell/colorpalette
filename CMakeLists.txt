# Copyright 2022-present Contributors to the colorpalette project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/mikaelsundell/colorpalette

cmake_minimum_required(VERSION 3.27)
set( project_name "colorpalette" )

project( "${project_name} project" )

# dependencies
find_package( OpenImageIO CONFIG REQUIRED )
find_package( OpenCV CONFIG REQUIRED )

# package
add_executable( ${project_name} "colorpalette.cpp" )
set_property(TARGET ${project_name} PROPERTY CXX_STANDARD 14)

target_link_libraries(${project_name}
    PRIVATE 
        OpenImageIO::OpenImageIO
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)

install(TARGETS ${project_name}
    RUNTIME DESTINATION bin
)
