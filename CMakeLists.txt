cmake_minimum_required(VERSION 3.8)
project(hand_landmarker)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(hand_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# ---------------------------
# C++ Node
# ---------------------------
add_executable(hand_landmarks_visualizer
  src/hand_landmarks_visualizer_node.cpp
)

ament_target_dependencies(
  hand_landmarks_visualizer
  rclcpp
  sensor_msgs
  hand_msgs
  cv_bridge
)

target_link_libraries(
  hand_landmarks_visualizer
  ${OpenCV_LIBS}
)

install(TARGETS
  hand_landmarks_visualizer
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------
# Python Node
# ---------------------------
install(PROGRAMS
  src/hand_landmarker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------
# Install launch files
# ---------------------------
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()