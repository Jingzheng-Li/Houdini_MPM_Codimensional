cmake_minimum_required(VERSION 3.15)

SET(CMAKE_C_COMPILER /usr/bin/clang)
SET(CMAKE_CXX_COMPILER /usr/bin/clang++)

project(MPM_Codimensional LANGUAGES CXX)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

# Set the build type
set(CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Add directory with macros
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(MPMCODCommon)

list(APPEND CMAKE_PREFIX_PATH "/opt/hfs19.5/toolkit/cmake")
find_package(Houdini REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include/)


# Add OpenMP flags
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(MPMCOD)
