# Load the MACA environment
# MACA_PATH  The directory that maca installed
# MXCC mxcc


if(NOT DEFINED MACA_CLANG_PATH)
  message(FATAL_ERROR "\n The MACA_CLANG_PATH is not specified.")
endif()

if(NOT DEFINED ENV{MACA_PATH})
  message(FATAL_ERROR "No MACA_PATH defined")
else()
  set(MACA_PATH $ENV{MACA_PATH})
  set(MXCC ${MACA_CLANG_PATH}/mxcc)
  set(MACA_FOUND TRUE)
endif()

if(MACA_FOUND)
  set(MACA_INCLUDE_DIR ${MACA_PATH}/include/)
  set(MACA_LIBRARY ${MACA_PATH}/lib)

  set(CMAKE_C_COMPILER ${MXCC})
  set(CMAKE_CXX_COMPILER ${MXCC})

  add_compile_options(-x maca) # Specify the language for the input files is maca.
  add_compile_options(-std=c++17)

  # Set MXCC flags
  # maca-path The MACA installed folder
  # maca-device-lib-path The MACA device library installed folder
  # maca-device-lib The MACA device library
  set(MACA_CC_FLAGS "--maca-path=$ENV{MACA_PATH} --maca-device-lib-path=$ENV{MACA_PATH}/lib/ --maca-device-lib=maca_mathlib.bc --maca-device-lib=maca_kernellib.bc")
  set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} ${MACA_CC_FLAGS}")

  include_directories(
    ${MACA_INCLUDE_DIR}
    ${MACA_CLANG_PATH}/../lib
  )

  link_directories(
    ${MACA_LIBRARY}
  )
  set(MACA_LIBS mcruntime)

  message("MACA PATH:" ${MACA_PATH})
  message("MACA LIBRARY:" ${MACA_LIBRARY})
  message("MACA CC FLAGS:" ${MACA_CC_FLAGS})
  message("MACA_INLUCDE_DIR:" ${${MACA_PATH}/include/})
  message("MXCC:" ${MXCC})

endif()

