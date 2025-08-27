cmake_minimum_required(VERSION 3.23)
project(Project2)

set(CMAKE_CXX_STANDARD 17)
set(ENV{OPENCV_DIR} "C:\\tools\\opencv\\build")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories( include
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(Project2 src/main.cpp
        src/csv_util.cpp
        include/csv_util.h
        include/features.h
        src/features.cpp
        src/metrics.cpp
        include/metrics.h
        include/im2csv.h
        src/im2csv.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(Project2 ${OpenCV_LIBS})