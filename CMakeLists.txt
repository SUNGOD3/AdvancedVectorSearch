cmake_minimum_required(VERSION 3.14)
project(AdvancedVectorSearch)

# add_subdirectory(_deps/googletest-src)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.10.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Disable specific warnings for gtest
if (CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wno-error=maybe-uninitialized)
    set_target_properties(gtest PROPERTIES COMPILE_FLAGS "-Wno-error=maybe-uninitialized")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    include
    ${PYBIND11_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

add_library(search_lib STATIC 
    src/base_search.cpp
    src/linear_search.cpp
    src/knn_search.cpp
)

target_link_libraries(search_lib PRIVATE pybind11::pybind11)

enable_testing()
add_subdirectory(tests)