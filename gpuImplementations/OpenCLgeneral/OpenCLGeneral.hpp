
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class OpenCLGeneral{
    public:
	OpenCLGeneral(int number_mem_objects, char * cosa);
	~OpenCLGeneral();

    protected:
	int number_mem_objects;
	cl_mem *memObjects;
        cl_context context;
        cl_command_queue commandQueue;
        cl_program program;
        cl_device_id device;
        cl_kernel kernel;
        cl_int errNum;


	static cl_context CreateContext();
	static cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
	static cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
	void Cleanup(cl_context context, cl_command_queue commandQueue,
		     cl_program program, cl_kernel kernel, cl_mem memObjects[]);
	void init(char * cosa);
};


