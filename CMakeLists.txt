cmake_minimum_required(VERSION 3.26)
project(Reconocimiento-de-d-gitos-C-)
set(CMAKE_CXX_STANDARD 23)

set(EIGEN_DIR "C:/Users/RODRIGO/Reconocimiento-de-d-gitos-C-/cmake-build-debug/eigen-3.4.0")
set(ENV{OPENCV_DIR} "C:\\tools\\opencv\\build")

if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()

include_directories(${EIGEN_DIR})

FIND_PACKAGE(OpenCV REQUIRED)
set(OpenCV_FOUND TRUE)


add_executable(Reconocimiento-de-d-gitos-C- main.cpp
        NeuronalNetwork.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(Reconocimiento-de-d-gitos-C- ${OpenCV_LIBS})
