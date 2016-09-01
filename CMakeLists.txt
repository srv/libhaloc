cmake_minimum_required(VERSION 2.8.3)
project(libhaloc)
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
            src/lc.cpp
            src/image.cpp
            src/hash.cpp)
target_link_libraries(haloc ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

# Include directories
include_directories(${catkin_INCLUDE_DIRS} include)