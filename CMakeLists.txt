cmake_minimum_required(VERSION 3.20)

project(citrius
    VERSION 0.1.0
    LANGUAGES CXX OBJCXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(citrius
    src/cpu_device.cpp
    src/cpu_storage.cpp
    src/metal_device.mm
    src/metal_storage.mm
    src/tensor.cpp
    src/types.cpp
)

target_include_directories(citrius
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(citrius PUBLIC cxx_std_20)

target_link_libraries(citrius
    PUBLIC
        "-framework Foundation"
        "-framework Metal"
)

enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(tensor_test
    tests/cpu_device_test.cpp
    tests/metal_device_test.cpp
    tests/tensor_test.cpp
)

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

gtest_discover_tests(tensor_test)
