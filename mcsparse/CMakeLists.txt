
file(GLOB KERNELS_SOURCE
     "src/kernels/level1/*.cpp"
     "src/kernels/level2/spmv/*.cpp"
     "src/kernels/level2/bsrmv/*.cpp"
     "src/kernels/level2/spsv/*.cpp"
     "src/kernels/level2/bsrsv/*.cpp"
     "src/kernels/level2/gemvi/*.cpp"
     "src/kernels/level3/spmm/*.cpp"
     "src/kernels/level3/bsrmm/*.cpp"
     "src/kernels/level3/gemmi/*.cpp"
     "src/kernels/level3/spsm/*.cpp"
     "src/kernels/level3/bsrsm/*.cpp"
     "src/kernels/extra/spgemm/*.cpp"
     "src/kernels/extra/csrgeam/*.cpp"
     "src/kernels/conversion/*.cpp"
     "src/kernels/precond/*.cpp"
     "src/kernels/reordering/*.cpp"
     "src/kernels/generic/*.cpp"
     "src/tbd/*.cpp")
file(GLOB COMMON_SOURCE "src/common/*.cpp" "src/utils/*.cpp" "src/wrapper/*.cpp")

if(BUILD_MACA_SPARSE)
    link_directories(${MACA_LIBRARY})
endif()

set(MCSP_PLATFORM "mcr")
set(MCSP_PLATFORM_RUNTIME_H "mc_runtime.h")
set(MCSP_PLATFORM_STREAM_T "mcStream_t")
set(MACA_COMMON_DEFINE "\#include \"common/__clang_maca_common_define.h\"")
set(MCSP_HALF_H "common/maca_fp16.h")
set(MCSP_BFLOAT16_H "common/maca_bfloat16.h")
set(MCSP_DATA_TYPE "common/mc_library_types.h")

set(MCSP_COMPLEX_H "internal/mcsp_complex.hpp")
configure_file(include/interface/mcsp_helper.h.in "${CMAKE_CURRENT_SOURCE_DIR}/include/interface/mcsp_helper.h")
configure_file(include/common/mcsp_types.h.in "${CMAKE_CURRENT_SOURCE_DIR}/include/common/mcsp_types.h")

set(MCSP_COMPLEX_H "common/mcComplex.h")
configure_file(include/common/mcsp_types.h.in "${CMAKE_CURRENT_SOURCE_DIR}/include/common/mcsp_types_export.h")

include_directories (
    src/include
    src/common
    src/mcprim
    src/kernels/level1
    src/kernels/level2
)


add_library(mcsparse_static STATIC ${KERNELS_SOURCE} ${COMMON_SOURCE})
add_library(mcsparse_shared SHARED ${KERNELS_SOURCE} ${COMMON_SOURCE})


set_target_properties(mcsparse_shared PROPERTIES OUTPUT_NAME "mcsparse")
set_target_properties(mcsparse_static PROPERTIES POSITION_INDEPENDENT_CODE OFF)
set_target_properties(mcsparse_shared PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../lib64:$ORIGIN/../lib64")

if(BUILD_MACA_SPARSE)
    install (TARGETS mcsparse_shared LIBRARY
        COMPONENT lib
        DESTINATION lib)

    install (TARGETS mcsparse_static ARCHIVE
        COMPONENT lib
        DESTINATION lib)

    install(DIRECTORY  ${PROJECT_SOURCE_DIR}/mcsparse/include/common ${PROJECT_SOURCE_DIR}/mcsparse/include/interface
        DESTINATION include/mcsparse/
        COMPONENT header
        PATTERN *.h.in EXCLUDE
        PATTERN internal EXCLUDE
        PATTERN mcsp_types.h EXCLUDE
        PATTERN mcsp_types_export.h EXCLUDE
    )

    install(FILES ${PROJECT_SOURCE_DIR}/mcsparse/include/common/mcsp_types_export.h
        COMPONENT header
        DESTINATION include/mcsparse/common/
        RENAME mcsp_types.h
    )
    install(FILES ${PROJECT_SOURCE_DIR}/mcsparse/include/mcsparse.h
        COMPONENT header
        DESTINATION include/mcsparse/
    )
endif()
