include(load_maca)
message(STATUS "Build with MACA")

message("MACA_CLANG_PATH:" ${MACA_CLANG_PATH})

add_subdirectory(mcsparse)