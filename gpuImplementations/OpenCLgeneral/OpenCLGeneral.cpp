
#include "OpenCLGeneral.hpp"


OpenCLGeneral::OpenCLGeneral(int number_mem_objects, char * file_kernel_name) {
    this->number_mem_objects=number_mem_objects;
    memObjects = new cl_mem[number_mem_objects];
    for(int i = 0; i < number_mem_objects; ++i) {
    	memObjects[i]=0;
    }
    init(file_kernel_name);
}



OpenCLGeneral::~OpenCLGeneral() {
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);

}


void OpenCLGeneral::init(char *programName) {
    context = 0;
    commandQueue = 0;
    program = 0;
    device = 0;
    kernel = 0;
    errNum = 0;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
	std::cerr << "Failed to create OpenCL context." << std::endl;
	return;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)

    {
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, programName);
    if (program == NULL)
    {
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }

    // Create OpenCL kernel
    // The child class will create the program.
    /*
    kernel = clCreateKernel(program, "sparseDot", NULL);
    if (kernel == NULL)
    {
	std::cerr << "Failed to create kernel" << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }
    */
    return;
}



///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context OpenCLGeneral::CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
	std::cerr << "Failed to find any OpenCL platforms." << std::endl;
	return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
	CL_CONTEXT_PLATFORM,
	(cl_context_properties)firstPlatformId,
	0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
	                              NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
	std::cout << "Could not create GPU context, trying CPU..." << std::endl;
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
	                                  NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
	    std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
	    return NULL;
	}
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue OpenCLGeneral::CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
	std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
	return NULL;
    }

    if (deviceBufferSize <= 0)
    {
	std::cerr << "No devices available.";
	return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
	delete [] devices;
	std::cerr << "Failed to get device IDs";
	return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
	delete [] devices;
	std::cerr << "Failed to create commandQueue for device 0";
	return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program OpenCLGeneral::CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
	std::cout << fileName << std::endl;
	std::cerr << "Failed to open file for reading: " << fileName << std::endl;
	return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
	                                (const char**)&srcStr,
	                                NULL, NULL);
    if (program == NULL)
    {
	std::cerr << "Failed to create CL program from source." << std::endl;
	return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
	// Determine the reason for the error
	char buildLog[16384];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
	                      sizeof(buildLog), buildLog, NULL);

	std::cerr << "Error in kernel: " << std::endl;
	std::cerr << buildLog;
	clReleaseProgram(program);
	return NULL;
    }

    return program;
}

///
//  Cleanup any created OpenCL resources
//
void OpenCLGeneral::Cleanup(cl_context context, cl_command_queue commandQueue,
	     cl_program program, cl_kernel kernel, cl_mem memObjects[])
{
    for (int i = 0; i < number_mem_objects; i++)
    {
	if (memObjects[i] != 0)
	    clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
	clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
	clReleaseKernel(kernel);

    if (program != 0)
	clReleaseProgram(program);

    if (context != 0)
	clReleaseContext(context);

}


