cmake_minimum_required(VERSION 3.10)

project(mcsparse LANGUAGES CXX C)

set(CMAKE_SKIP_BUILD_RPATH TRUE)

set(MACA_PLATFORM "maca")

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
)
add_compile_options(-std=c++17)

# Close -Wunused-result warning that trigger [[nodiscard]] in c++17
# add_compile_options(-Wno-unused-result)

include_directories(
    mcsparse/include
)
add_definitions(-w)

include(compile_maca)

# set cpack using param
set( LIBRARY_NAME mcsparse)
