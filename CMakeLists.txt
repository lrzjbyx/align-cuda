cmake_minimum_required(VERSION 3.8)
project(sealnet_align LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# 使用CUDA库
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find nlohmann json
find_package(nlohmann_json 3.6.0 REQUIRED)

# find pybind11
find_package(pybind11)
# 寻找Boost库
find_package(Boost REQUIRED COMPONENTS python)

# 设定Python版本，根据你的情况设定
set(Python_ADDITIONAL_VERSIONS 3.8)

# 包含头文件路径
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Link OpenCV libraries



# add_executable(SEALNET_ALIGN main.cpp align.cpp sealnet_align.cpp)
# add_library(sealnet_align MODULE main.cu)
add_library(sealnet_align MODULE main.cu)
# add_executable(sealnet_align align.cu main.cu)
# add_executable(sealnet_align main.cu)


# 链接 Pybind11
target_link_libraries(sealnet_align PRIVATE ${OpenCV_LIBS}  pybind11::module ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json  ${CUDA_LIBRARIES})

# target_include_directories(sealnet_align ${CUDA_INCLUDE_DIRS})

set_target_properties(sealnet_align PROPERTIES PREFIX "")



# 设置CUDA版本
set_property(TARGET sealnet_align 
             PROPERTY CUDA_ARCHITECTURES OFF)