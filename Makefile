
all:
	mkdir -p build/src_maca
	rsync -av ./ build/src_maca/ --exclude build/ --exclude .git/
	mkdir -p build/build_maca
	cd build/build_maca; \
	cmake ../src_maca \
	        -DBUILD_MCSPARSE_CPU_LIB=ON \
			-DCMAKE_BUILD_TYPE=Debug \
			-DBUILD_MACA_SPARSE=ON \
			-DMACA_CLANG_PATH=${MACA_CLANG_PATH} \
			-DCMAKE_INSTALL_PREFIX=${PWD}/build/opt_maca \
			-DPACKAGE_GENERATOR=${PACKAGE_GENERATOR}\
			-DCMAKE_EXPORT_COMPILE_COMMANDS=1 && \
			make -j32 && make install

clean:
	rm -rf build
