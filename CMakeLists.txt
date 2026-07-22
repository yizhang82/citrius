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
    src/models/qwen3.cpp
    src/models/qwen3_tokenizer.cpp
    src/devices/cpu/cpu_device.cpp
    src/devices/cpu/cpu_storage.cpp
    src/devices/cpu/multi_thread_cpu_device.cpp
    src/nn/dropout.cpp
    src/nn/embedding.cpp
    src/nn/feed_forward.cpp
    src/nn/linear.cpp
    src/nn/layer_norm.cpp
    src/nn/functional.cpp
    src/nn/module.cpp
    src/nn/module_list.cpp
    src/nn/multi_head_attention.cpp
    src/nn/transformer_block.cpp
    src/indexing_operations.cpp
    src/operations.cpp
    src/reduction_operations.cpp
    src/safetensors.cpp
    src/shape_operations.cpp
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
            "-framework MetalPerformanceShaders"
    )
    target_compile_definitions(citrius PRIVATE CITRIUS_HAS_METAL=1)
    target_compile_options(citrius PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )
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
        src/devices/cuda/cuda_allocation.cu
        src/devices/cuda/cuda_context.cu
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

# Resolve clang-format during configuration so helper scripts do not need to
# know where Visual Studio, LLVM, or a package manager installed it.
find_program(CITRIUS_CLANG_FORMAT
    NAMES clang-format clang-format-20 clang-format-19 clang-format-18
    HINTS
        "${CMAKE_GENERATOR_INSTANCE}/VC/Tools/Llvm/x64/bin"
        "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/x64/bin"
        "$ENV{ProgramFiles}/LLVM/bin"
)

if(CITRIUS_CLANG_FORMAT)
    file(GLOB_RECURSE CITRIUS_FORMAT_FILES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/src/*.mm"
        "${PROJECT_SOURCE_DIR}/tests/*.cpp"
        "${PROJECT_SOURCE_DIR}/tests/*.cu"
        "${PROJECT_SOURCE_DIR}/examples/*.cpp"
        "${PROJECT_SOURCE_DIR}/examples/*.h"
        "${PROJECT_SOURCE_DIR}/benchmarks/*.cpp"
        "${PROJECT_SOURCE_DIR}/benchmarks/*.cu"
        "${PROJECT_SOURCE_DIR}/tools/*.cpp"
    )
    add_custom_target(format
        COMMAND "${CITRIUS_CLANG_FORMAT}" -i -style=file ${CITRIUS_FORMAT_FILES}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Formatting Citrius sources"
        VERBATIM
    )
else()
    add_custom_target(format
        COMMAND "${CMAKE_COMMAND}" -E echo
            "clang-format was not found. Install LLVM or the Visual Studio LLVM tools."
        COMMAND "${CMAKE_COMMAND}" -E false
        VERBATIM
    )
endif()

add_executable(example_operations examples/operations.cpp)
target_link_libraries(example_operations PRIVATE citrius)

add_executable(example_custom_module examples/custom_module.cpp)
target_link_libraries(example_custom_module PRIVATE citrius)

add_executable(operations_benchmark benchmarks/operations_benchmark.cpp)
target_link_libraries(operations_benchmark PRIVATE citrius)

add_executable(qwen3_decoding_benchmark benchmarks/qwen3_decoding_benchmark.cpp)
target_link_libraries(qwen3_decoding_benchmark PRIVATE citrius)

add_executable(decoder_chat tools/decoder_chat.cpp)
target_link_libraries(decoder_chat PRIVATE citrius)
if(CITRIUS_ENABLE_METAL)
    target_compile_definitions(operations_benchmark PRIVATE CITRIUS_HAS_METAL=1)
    target_compile_definitions(qwen3_decoding_benchmark PRIVATE CITRIUS_HAS_METAL=1)
endif()
if(CITRIUS_ENABLE_CUDA)
    target_compile_definitions(operations_benchmark PRIVATE CITRIUS_HAS_CUDA=1)
    target_compile_definitions(qwen3_decoding_benchmark PRIVATE CITRIUS_HAS_CUDA=1)

    add_executable(cuda_elementwise_benchmark
        benchmarks/cuda_elementwise_benchmark.cu
    )
    target_link_libraries(cuda_elementwise_benchmark PRIVATE CUDA::cudart)
    set_target_properties(cuda_elementwise_benchmark PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    add_executable(cuda_matmul_benchmark
        benchmarks/cuda_matmul_benchmark.cu
    )

    target_link_libraries(cuda_matmul_benchmark PRIVATE CUDA::cudart)
    set_target_properties(cuda_matmul_benchmark PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

enable_testing()

include(FetchContent)

# Build GoogleTest with the same toolchain and C++ runtime as Citrius. In particular,
# a shared GoogleTest installed by Conda on macOS can pull Conda's libc++ into the
# process alongside the system libc++, which crashes during test discovery.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(tensor_test
    tests/device_nn_conformance_test.cpp
    tests/device_operation_conformance_test.cpp
    tests/models/qwen3_test.cpp
    tests/models/qwen3_tokenizer_test.cpp
    tests/devices/cpu/cpu_device_test.cpp
    tests/nn/container_test.cpp
    tests/nn/embedding_test.cpp
    tests/nn/feed_forward_test.cpp
    tests/nn/linear_test.cpp
    tests/nn/layer_norm_test.cpp
    tests/nn/functional_test.cpp
    tests/nn/module_test.cpp
    tests/nn/multi_head_attention_test.cpp
    tests/nn/transformer_block_test.cpp
    tests/operations_test.cpp
    tests/reduction_operations_test.cpp
    tests/shape_operations_test.cpp
    tests/tensor_test.cpp
    tests/tensor_utils_test.cpp
)

if(CITRIUS_ENABLE_METAL)
    target_sources(tensor_test
        PRIVATE
            tests/devices/metal/metal_device_test.cpp
    )
    target_compile_definitions(tensor_test PRIVATE CITRIUS_TEST_HAS_METAL=1)
endif()

if(CITRIUS_ENABLE_CUDA)
    target_sources(tensor_test PRIVATE tests/devices/cuda/cuda_device_test.cpp)
    target_compile_definitions(tensor_test PRIVATE CITRIUS_TEST_HAS_CUDA=1)
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

gtest_discover_tests(tensor_test DISCOVERY_MODE PRE_TEST NO_PRETTY_VALUES)
