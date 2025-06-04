# mcSparse
mcSparse is Metax sparse basic linear algebra subprograms contains a set of basis linear algebra subroutines used for handling sparse matrices.

## Build
### 1. Prerequisites
Set the following environment variables before building.
```bash
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${MACA_PATH}/lib:$LD_LIBRARY_PATH
```

### 2. Build using cmake

CMakeLists and Makefile is provided, we can build mcSparse by following steps:

```bash
make
```

The include and lib will be installed on ${PROJECT_ROOT}/build/opt_maca

