# Minimum Makefile for the extension package
PKG_CFLAGS = -std=c++14 -stdlib=libc++ -O2 -fPIC -Iinclude/ -Iexternal/\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I$(STONNE_ROOT)/include\
	-I$(STONNE_ROOT)/external\
	-I$(STONNE_ROOT)/stonne_linker_src



PKG_LDFLAGS =-L${TVM_ROOT}/build 
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	PKG_LDFLAGS += -undefined dynamic_lookup
endif

bifrost/stonne/stonne_lib/stonne_lib.so: $(wildcard src/*cpp) ${STONNE_ROOT}/stonne_linker_src/stonne_linker.cpp $(wildcard ${STONNE_ROOT}/src/*cpp)
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS)