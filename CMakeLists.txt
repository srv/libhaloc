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

## Declare ROS messages and services
add_service_files(FILES RemoteExecution.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

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

catkin_package(INCLUDE_DIRS include LIBRARIES haloc)

# Add the Image Hashing library
add_library(haloc
           src/lc.cpp
	   src/image.cpp
	   src/hash.cpp)
target_link_libraries(haloc ${Boost_LIBRARIES} ${Eigen_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

# Include directories
include_directories(${catkin_INCLUDE_DIRS} include)

# Add the examples
add_executable(mono
			   examples/mono.cpp)
target_link_libraries(mono haloc ${Boost_LIBRARIES} ${Eigen_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
add_executable(remote_exec_srv
			   tools/remote_exec_srv.cpp)
target_link_libraries(remote_exec_srv ${catkin_LIBRARIES})
