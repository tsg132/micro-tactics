# cmake_minimum_required(VERSION 3.18)
# project(MicroTactics LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# include(CheckLanguage)
# check_language(CUDA)
# if(CMAKE_CUDA_COMPILER)
#   enable_language(CUDA)
#   set(CMAKE_CUDA_ARCHITECTURES 75)
# else()
#   message(FATAL_ERROR "CUDA compiler not found!")
# endif()

# find_package(CUDAToolkit REQUIRED)

# include_directories(
#   ${CMAKE_SOURCE_DIR}/include
#   ${CMAKE_SOURCE_DIR}/rl
#   ${CUDAToolkit_INCLUDE_DIRS}
# )

# set(SOURCES
#   src/env.cpp
#   src/render.cpp
# )

# add_executable(train_mt
#   src/main_train.cpp
#   ${SOURCES}
#   rl/policy_kernels.cu
# )
# target_link_libraries(train_mt PRIVATE ${CUDAToolkit_LIBRARIES} cublas)

# add_executable(play_mt
#   src/main_human.cpp
#   ${SOURCES}
#   rl/policy_kernels.cu
# )
# target_link_libraries(play_mt PRIVATE ${CUDAToolkit_LIBRARIES} cublas)

cmake_minimum_required(VERSION 3.18)
project(MicroTactics LANGUAGES CXX CUDA)

# === Standards ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === CUDA Setup ===
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    # Colab → NVIDIA T4 / A100 → sm_75 or sm_80
    set(CMAKE_CUDA_ARCHITECTURES 75)
else()
    message(FATAL_ERROR "CUDA compiler not found! Building aborted.")
endif()

# === CUDA Toolkit ===
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# === Include directories ===
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/rl
    ${CMAKE_SOURCE_DIR}/src
)

# === Common sources ===
set(SOURCES
    src/env.cpp
    src/render.cpp
)

# === Train Executable ===
add_executable(train_mt
    src/main_train.cpp
    ${SOURCES}
    rl/policy_kernels.cu
)
set_target_properties(train_mt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(train_mt PRIVATE ${CUDAToolkit_LIBRARIES} cublas)

# === Play (Human) Executable ===
add_executable(play_mt
    src/main_human.cpp
    ${SOURCES}
    rl/policy_kernels.cu
)
set_target_properties(play_mt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(play_mt PRIVATE ${CUDAToolkit_LIBRARIES} cublas)

# === Output summary ===
message(STATUS "")
message(STATUS "=== MicroTactics Build Configuration ===")
message(STATUS "C++ Compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version : ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include : ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Enabled : TRUE")
message(STATUS "========================================")