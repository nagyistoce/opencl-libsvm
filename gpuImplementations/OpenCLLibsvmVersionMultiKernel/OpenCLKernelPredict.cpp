
//BIG TODO!!!!!!! Los vectores empiezan con indice 1!!!!!!!! Yo aquí hago una chapuza 

#include "OpenCLKernelPredict.hpp"
#include <time.h>
using namespace std;


void OpenCLKernelPredict::loadKernel() {
        required_comp_kernel = false;
	switch (kernel_type) {
	    case LINEAR:
		        required_comp_kernel = false;
   		        kernel = clCreateKernel(program, "sparseDot", NULL);
				break;
	    case POLY:

				params_poly.gamma = gamma;
                params_poly.coef0 = coef0;
                params_poly.degree = degree;
		        required_comp_kernel = true;
	        	kernel = clCreateKernel(program, "sparsePoly",NULL);
				break;
	    case RBF:

                params_rbf.gamma = gamma;
                required_comp_kernel = true;
                kernel = clCreateKernel(program, "sparseRBF",NULL);
				break;
	    case SIGMOID:

                params_sigmoid.gamma = gamma;
				params_sigmoid.coef0 = coef0;
                required_comp_kernel = true;
                kernel = clCreateKernel(program, "sparseSigmoid",NULL);
				break;
	}

        if (kernel == NULL)
        {
         		std::cerr << "Failed to create kernel" << std::endl;
                Cleanup(context, commandQueue, program, kernel, memObjects);
                return;
        }

}

OpenCLKernelPredict::~OpenCLKernelPredict() {
		std::cout << "Destroying GPU instance 1" << std::endl;
        delete[] features;
        std::cout << "Destroying GPU instance 2" << std::endl;
		delete[] my_nodes;
		std::cout << "Destroying GPU instance 3" << std::endl;
		delete[] i_vec;
		std::cout << "Destroying GPU instance 4" << std::endl;
		
		//delete[] y;
		//if(required_comp_kernel) clReleaseKernel(extra_kernel);
}

void OpenCLKernelPredict::InitialyzeInputs(svm_node * const *x) {
	
    features = new int[l*2];
	total_nodes = 0;
	max_features = 0; 
	int total_features = 0;



	for(int i = 0; i < l; ++i) {
	   features[i] = total_features;
	   for(int z = 0; x[i][z].index != -1; ++z) {
		total_nodes++;
		total_features++;
                if(max_features < x[i][z].index) max_features = x[i][z].index;
	   }
	   features[i+l] = total_features;
           
	}
	max_features = max_features + 1;// TODO!! Esto es para arreglar que los index empiezan en 1!!
    
    // PROBLEMON!!!!!!!!!!! Hay que decir el maximo de muestras DESDE FUERA!!!
    i_vec = new double[max_features];

    my_nodes = new svm_node[total_nodes];

    int cont = 0;
	for(int i = 0; i < l; ++i) {
	    int limit = features[i+l]-features[i];
	    for(int j = 0; j < limit; ++j) {
		    my_nodes[cont].index = x[i][j].index;
		    my_nodes[cont].value = x[i][j].value;
		    cont++;
	   }
	}

	//The matrix is, the 'total nodes' vector plus the 'features'! This two things must be allocated in memory
}


double OpenCLKernelPredict::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

OpenCLKernelPredict::OpenCLKernelPredict(const int l,svm_node * const *x, const svm_parameter& param):OpenCLGeneral(5, "./gpuImplementations/OpenCLLibsvmVersionMultiKernel/GPUKernelCalculationPredict.cl") {

    kernel_type = param.kernel_type;
    degree = param.degree;
    gamma = param.gamma;
    coef0 = param.coef0;

    this->l = l;
    InitialyzeInputs(x);

    double output[l];
    
    //output -- memObjects[0]
    if(!CreateInputOutputMemoryObject(memObjects[0], output, l, sizeof(output[0]))) { 
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    } 
    //whole matrix -- memObjects[1]
    if(!CreateInputMemoryObject(memObjects[1], my_nodes, total_nodes, sizeof(my_nodes[0]))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }
    //sample indexs -- memObjects[2]
    if(!CreateInputMemoryObject(memObjects[2], features, l*2, sizeof(features[0]))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }
    //vector -- memObjects[3]
    if(!CreateInputMemoryObject(memObjects[3], i_vec, max_features, sizeof(i_vec[0]))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }

    loadKernel();

    
   
	    if(kernel_type == RBF) {
			x_square = new double[l];
			for(int i=0;i<l;i++)
				x_square[i] = this->dot(x[i],x[i]);
	    	
		    if(!CreateInputMemoryObject(memObjects[4], x_square, l, sizeof(double))) {
			std::cerr << "Error creating memory objects in device." << std::endl;
		    	Cleanup(context, commandQueue, program, kernel, memObjects);
		   	return;
		    }
		    clSetKernelArg(kernel,6,sizeof(cl_mem), &memObjects[4]);
	    }
    
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[1]); // todos los nodos... la pseudo-matriz
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[2]);   // features
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[3]);  // input vector
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[0]);  // output
    clSetKernelArg(kernel,4,sizeof(int), &l);


    w_gr_size = 32;
    lower_mult = w_gr_size*(l/w_gr_size);
    if(l%w_gr_size!=0) lower_mult = lower_mult + w_gr_size;

}


bool OpenCLKernelPredict::CreateOutputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__) {
    memObject = clCreateBuffer(context, CL_MEM_READ_WRITE, size__ * quant_elems, NULL, NULL);	    
    if (memObject == NULL) {
	std::cerr << "Error creating vector memory object." << std::endl;
	return false;
    }
    return true;
}

bool OpenCLKernelPredict::SetVector(const svm_node *cosa){

 
    for(int j = 0; j < max_features; ++j) i_vec[j] = 0;
       
    for(int z = 0; cosa[z].index != -1; ++z) { 
        if(cosa[z].index < max_features) {
            i_vec[cosa[z].index] = cosa[z].value;
        }  
        
    }
    
    errNum = clEnqueueWriteBuffer(commandQueue, memObjects[3], CL_FALSE, 0, sizeof(double)*max_features, i_vec, 0, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
	std::cerr << "Error writing algo." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return false;
    }
    
    if(kernel_type == RBF) {
			params_rbf.common_x_square = dot(cosa, cosa);
	 }

}

bool OpenCLKernelPredict::CreateInputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__) {
    memObject = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size__ * quant_elems, a, NULL);
    if (memObject == NULL) {
	std::cerr << "Error creating input memory object." << std::endl;
	return false;
    }
    return true;
}

bool OpenCLKernelPredict::CreateInputOutputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__) {
    memObject = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size__ * quant_elems, a, NULL);
    if (memObject == NULL) {
	std::cerr << "Error creating input memory object." << std::endl;
	return false;
    }
    return true;
}

void OpenCLKernelPredict::runMatrixMult(double *result, int offset) {
    //*avoid_const = *avoid_const + 1;

        switch(kernel_type){
                case LINEAR:
                        clSetKernelArg(kernel,5,sizeof(params_dot), &params_dot);
                        break;
                case SIGMOID:
                        clSetKernelArg(kernel,5,sizeof(params_sigmoid), &params_sigmoid);
                        break;
                case RBF:
                        clSetKernelArg(kernel,5,sizeof(params_rbf), &params_rbf);
                        break;
                case POLY:
                        clSetKernelArg(kernel,5,sizeof(params_poly), &params_poly);
                        break;
           }



    if (errNum != CL_SUCCESS)
    {
	std::cerr << "Error setting kernel arguments." << std::endl;
        std::cout << errNum << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }

    size_t globalWorkSize[1] = { lower_mult };
    size_t localWorkSize[1] = { w_gr_size };
    size_t offset__[1] = {offset};

    clFlush(commandQueue);

    
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, offset__,globalWorkSize, localWorkSize, 0, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
	std::cerr << "Error queuing kernel for execution." << std::endl;
        std::cerr << "Error ID: " << errNum << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }

    clFlush(commandQueue);
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE, 0, l * sizeof(double), result, 0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
	std::cerr << "Error reading result buffer." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }
    return;
}
