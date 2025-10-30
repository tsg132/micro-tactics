cmake_minimum_required(VERSION 3.18)
project(MicroTactics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 75)
else()
  message(FATAL_ERROR "CUDA compiler not found!")
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/rl
  ${CUDAToolkit_INCLUDE_DIRS}
)

set(SOURCES
  src/env.cpp
  src/render.cpp
)

add_executable(train_mt
  src/main_train.cpp
  ${SOURCES}
  rl/policy_kernels.cu
)
target_link_libraries(train_mt PRIVATE ${CUDAToolkit_LIBRARIES} cublas)

add_executable(play_mt
  src/main_human.cpp
  ${SOURCES}
  rl/policy_kernels.cu
)
target_link_libraries(play_mt PRIVATE ${CUDAToolkit_LIBRARIES} cublas)