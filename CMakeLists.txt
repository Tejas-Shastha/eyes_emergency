cmake_minimum_required(VERSION 2.8.3)
project(eyes_emergency)
add_compile_options(-std=c++11 -O3 )

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  rospy
  sensor_msgs
)

find_package(OpenCV 4 REQUIRED)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

