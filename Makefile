CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

OPTION_PROFILING=-pg

all: svm-train svm-predict svm-scale

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -W1,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared "; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o /usr/lib64/libOpenCL.so gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLGeneral.o gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLKernel.o  gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLKernelPredict.o -lm -L/opt/AMDAPP/lib/x86_64/ -I/opt/AMDAPP/include -L/usr/lib64/OpenCL/vendors/intel -I~/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/ -pg -o  libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o OpenCLKernel.o
	$(CXX) $(CFLAGS) svm-predict.c  -pg -o svm-predict svm.o gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLGeneral.o gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLKernel.o  gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLKernelPredict.o -lm -L/opt/AMDAPP/lib/x86_64/ -L/usr/lib64/OpenCL/vendors/intel -I/opt/AMDAPP/include -I~/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/ /usr/lib64/libOpenCL.so

svm-train: svm-train.c svm.o OpenCLKernel.o
	$(CXX) $(CFLAGS) svm-train.c -pg -o  svm-train svm.o gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLGeneral.o gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLKernel.o  gpuImplementations/OpenCLLibsvmVersionMultiKernel/OpenCLKernelPredict.o -lm -L/opt/AMDAPP/lib/x86_64/ -L/usr/lib64/OpenCL/vendors/intel -I/opt/AMDAPP/include -I~/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/ /usr/lib64/libOpenCL.so 

svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -pg -o svm-scale

svm.o: svm.cpp
	$(CXX) $(CFLAGS) -I/opt/AMDAPP/include -I/veu/jcordero/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -c -g svm.cpp 

OpenCLKernel.o:
	make -C gpuImplementations/OpenCLLibsvmVersionMultiKernel/

clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale libsvm.so.$(SHVER)
	make -C gpuImplementations/OpenCLLibsvmVersionMultiKernel/ clean
