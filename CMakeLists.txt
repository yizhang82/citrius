cmake_minimum_required(VERSION 3.20)

project(citrius
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CITRIUS_ENABLE_METAL "Build Citrius with Apple Metal support" OFF)
option(CITRIUS_ENABLE_CUDA "Build Citrius with NVIDIA CUDA support" OFF)
option(CITRIUS_CUDA_USE_CUBLAS "Use cuBLAS for top-level CUDA matmul by default" ON)

add_library(citrius
    src/devices/cpu/cpu_device.cpp
    src/devices/cpu/cpu_storage.cpp
    src/operations.cpp
    src/tensor.cpp
    src/tensor_factory.cpp
    src/types.cpp
)

if(CITRIUS_ENABLE_METAL)
    enable_language(OBJCXX)

    target_sources(citrius
        PRIVATE
            src/devices/metal/metal_device.mm
            src/devices/metal/metal_storage.mm
    )

    target_link_libraries(citrius
        PUBLIC
            "-framework Foundation"
            "-framework Metal"
    )
    target_compile_definitions(citrius PRIVATE CITRIUS_HAS_METAL=1)
endif()

if(CITRIUS_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CITRIUS_CUTLASS_ROOT "${PROJECT_SOURCE_DIR}/third_party/cutlass")
    if(NOT EXISTS "${CITRIUS_CUTLASS_ROOT}/include/cutlass/cutlass.h")
        message(FATAL_ERROR
            "CUTLASS submodule is missing. Run: git submodule update --init --recursive")
    endif()
    target_sources(citrius PRIVATE
        src/devices/cuda/cuda_device.cu
        src/devices/cuda/cuda_storage.cu
        src/devices/cuda/cublas_cuda_device.cu
        src/devices/cuda/cutlass_cuda_device.cu
    )
    target_include_directories(citrius PRIVATE ${CITRIUS_CUTLASS_ROOT}/include)
    if(MSVC)
        target_compile_options(citrius PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()
    target_link_libraries(citrius PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(citrius PRIVATE CITRIUS_HAS_CUDA=1)
    if(CITRIUS_CUDA_USE_CUBLAS)
        target_compile_definitions(citrius PRIVATE CITRIUS_CUDA_DEFAULT_CUBLAS=1)
    endif()
    set_target_properties(citrius PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

target_include_directories(citrius
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(citrius PUBLIC cxx_std_20)

add_executable(example_operations examples/operations.cpp)
target_link_libraries(example_operations PRIVATE citrius)

add_executable(matmul_benchmark benchmarks/matmul_benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE citrius)
if(CITRIUS_ENABLE_CUDA)
    target_compile_definitions(matmul_benchmark PRIVATE CITRIUS_HAS_CUDA=1)
endif()

enable_testing()

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)

    # Keep GoogleTest's MSVC runtime consistent with the parent project.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

add_executable(tensor_test
    tests/devices/cpu/cpu_device_test.cpp
    tests/operations_test.cpp
    tests/tensor_test.cpp
)

if(CITRIUS_ENABLE_METAL)
    target_sources(tensor_test
        PRIVATE
            tests/devices/metal/metal_device_test.cpp
    )
endif()

if(CITRIUS_ENABLE_CUDA)
    target_sources(tensor_test PRIVATE tests/devices/cuda/cuda_device_test.cpp)
endif()

target_link_libraries(tensor_test
    PRIVATE
        citrius
        GTest::gtest_main
)

get_target_property(GTEST_MAIN_LOCATION GTest::gtest_main IMPORTED_LOCATION)
if(NOT GTEST_MAIN_LOCATION)
    get_target_property(GTEST_MAIN_LOCATION GTest::gtest_main IMPORTED_LOCATION_RELEASE)
endif()

if(GTEST_MAIN_LOCATION)
    get_filename_component(GTEST_LIB_DIR "${GTEST_MAIN_LOCATION}" DIRECTORY)
    set_target_properties(tensor_test PROPERTIES BUILD_RPATH "${GTEST_LIB_DIR}")
endif()

gtest_discover_tests(tensor_test DISCOVERY_MODE PRE_TEST)
