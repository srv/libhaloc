cmake_minimum_required(VERSION 2.8.3)
project(libhaloc)
find_package(catkin REQUIRED COMPONENTS 
    roscpp
    roslib
    tf
    cv_bridge
    image_geometry)

# Dependencies - Boost:
find_package(Boost REQUIRED COMPONENTS signals filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

# Dependencies - Eigen:
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

# Dependencies - OpenCV:
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

catkin_package()

# Add the Image Hashing library
add_library(haloc
			libhaloc/lc.cpp
			libhaloc/image.cpp
			libhaloc/hash.cpp)

# Include directories
include_directories(${catkin_INCLUDE_DIRS} libhaloc)

# Add the examples
add_executable(mono 
			   examples/mono.cpp)
target_link_libraries(mono haloc ${Boost_LIBRARIES} ${Eigen_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
