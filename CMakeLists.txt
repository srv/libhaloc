cmake_minimum_required(VERSION 2.8.3)
project(libhaloc)

# C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    tf
    cv_bridge
    image_geometry
    std_msgs
    cmake_modules)

# Dependencies - Eigen:
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Dependencies - OpenCV:
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

catkin_package(INCLUDE_DIRS include LIBRARIES haloc)

# Add the Image Hashing library
add_library(haloc
            src/hash.cpp
            src/publisher.cpp)
target_link_libraries(haloc
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES})

# Include directories
include_directories(${catkin_INCLUDE_DIRS} include)

# Add examples
add_executable(lip6indoor_dataset
  examples/lip6indoor_dataset.cpp)
target_link_libraries(lip6indoor_dataset
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  haloc)

