
all: OpenCLKernel.o OpenCLKernelPredict.o

OpenCLKernelPredict.o: OpenCLKernelPredict.cpp OpenCLKernelPredict.hpp OpenCLGeneral.o
	g++ -L/veu/jcordero/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib -I/veu/jcordero/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -L/opt/AMDAPP/lib/x86_64/ -I/opt/AMDAPP/include OpenCLKernelPredict.cpp -c -g -fPIC

OpenCLKernel.o: OpenCLKernel.cpp OpenCLKernel.hpp OpenCLGeneral.o
	g++ -L/veu/jcordero/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib -I/veu/jcordero/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -L/opt/AMDAPP/lib/x86_64/ -I/opt/AMDAPP/include OpenCLKernel.cpp -c -g -fPIC

OpenCLGeneral.o: ../OpenCLgeneral/OpenCLGeneral.cpp ../OpenCLgeneral/OpenCLGeneral.hpp
	g++ -L/veu/jcordero/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib -I/veu/jcordero/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -L/opt/AMDAPP/lib/x86_64/ -I/opt/AMDAPP/include ../OpenCLgeneral/OpenCLGeneral.cpp -c -g -fPIC

clean:
	rm *.o
