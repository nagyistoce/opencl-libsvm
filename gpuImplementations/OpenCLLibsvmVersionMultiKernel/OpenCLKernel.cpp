
//BIG TODO!!!!!!! Los vectores empiezan con indice 1!!!!!!!! Yo aquí hago una chapuza 

#include "OpenCLKernel.hpp"
#include <time.h>
using namespace std;


void OpenCLKernel::runPrecalculation(int first,int last, double **results) {

    //clock_t t3 = clock();

    int l_min_mult = 1;
    while(l_min_mult <= l) l_min_mult = l_min_mult * 2;
    
    int l_min_mult2 = 1;
    while(l_min_mult2 <= last-first) l_min_mult2 = l_min_mult2 * 2;
   
    errNum = clSetKernelArg(kernel_init, 8, sizeof(int), &first);
    errNum |= clSetKernelArg(kernel_init, 9, sizeof(int), &last);

    if (errNum != CL_SUCCESS) { std::cerr << "Error passing arguments." << std::endl;std::cerr << "Error ID: " << errNum << std::endl;Cleanup(context, commandQueue, program, kernel, memObjects);return;}
    //Be carefull!!! The current implementation shares a vector for an entire block, so the first dimension must be just 1... or assure you load the vector in each block    
    //size_t globalWorkSize[2] = { (last-first), l_min_mult };
    //size_t localWorkSize[2] = { 1 , min(l_min_mult,32)};
    
    size_t globalWorkSize[2] = { l, l_min_mult };
    size_t localWorkSize[2] = { 1 , min(l_min_mult,32)};
    
    size_t offset__[2] = {0,0};

    //std::cout << "Dimension 2 :" << min(l_min_mult,32) << std::endl;

    clFlush(commandQueue);

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel_init, 2, offset__,globalWorkSize, localWorkSize, 0, NULL, NULL);              if (errNum != CL_SUCCESS){	 std::cerr << "Error queuing kernel for execution." << std::endl; std::cerr << "Error ID: " << errNum << std::endl; Cleanup(context, commandQueue, program, kernel, memObjects); return; }

    clFlush(commandQueue);

    //clock_t t1 = clock();
    for(int i = first; i < last; ++i) {
    	errNum = clEnqueueReadBuffer(commandQueue, super_sortida, CL_TRUE, (i-first)*l*sizeof(double), l*sizeof(double), results[i-first], 0, NULL, NULL);                if (errNum != CL_SUCCESS) { std::cerr << "Error reading result." << std::endl; std::cerr << "Error ID: " << errNum << std::endl; Cleanup(context, commandQueue, program, kernel, memObjects); return; }
    }
    //clock_t t2 = clock();
    //std::cout << float(t2)-float(t1) << "           "  << float(t2)-float(t3) << std::endl;
}  
   
   
void OpenCLKernel::startPrecalculation(int prec_step) {
    //Precalculation
   
    kernel_init = clCreateKernel(program, "sparseDotASaco",NULL);      if (kernel == NULL) { std::cerr << "unable to create the precomputation kernel" << std::endl;}
    super_sortida =  clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*l*prec_step, NULL, NULL); if(super_sortida == NULL) {std::cout << "problem al crear buffer de memoria" << std::endl; return;}
   
    errNum  = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &memObjects[1]); // todos los nodos.$
    errNum |= clSetKernelArg(kernel_init, 1, sizeof(cl_mem), &memObjects[2]);   // features
    errNum |= clSetKernelArg(kernel_init, 2, sizeof(double)*max_features, NULL);  // input vecto
    errNum |= clSetKernelArg(kernel_init, 3, sizeof(cl_mem), &super_sortida);  // output
    errNum |= clSetKernelArg(kernel_init, 4, sizeof(cl_mem), &memObjects[4]);  // labels
    errNum |= clSetKernelArg(kernel_init, 5, sizeof(param_struct_dot), &params_dot);
    errNum |= clSetKernelArg(kernel_init, 6, sizeof(int), &l);
    errNum |= clSetKernelArg(kernel_init, 7, sizeof(int), &max_features);
    if (errNum != CL_SUCCESS) { std::cerr << "Error passing arguments." << std::endl;std::cerr << "Error ID: " << errNum << std::endl;Cleanup(context, commandQueue, program, kernel, memObjects);return;}    

}  
   
void OpenCLKernel::finishedPrecalculation() {
    errNum = clReleaseKernel(kernel_init);         if(errNum != CL_SUCCESS) std::cout << "peto al liberar kernel!" << std::endl;
    errNum = clReleaseMemObject(super_sortida);         if(errNum != CL_SUCCESS) std::cout << "peto al liberar kernel!" << std::endl;
}  


void OpenCLKernel::loadKernel() {
	switch (kernel_type) {
	    case LINEAR:
                common_labels = &params_dot.common_label;
	    
   	        kernel = clCreateKernel(program, "sparseDot", NULL);
		break;
	    case POLY:
		common_labels = &params_poly.common_label;
		params_poly.gamma = gamma;
                params_poly.coef0 = coef0;
                params_poly.degree = degree;
	        kernel = clCreateKernel(program, "sparsePoly",NULL);
		break;
	    case RBF:
		common_labels = &params_rbf.common_label;
        params_rbf.gamma = gamma;
        
        kernel = clCreateKernel(program, "sparseRBF",NULL);
		
		break;
	    case SIGMOID:
		common_labels = &params_sigmoid.common_label;
                params_sigmoid.gamma = gamma;
		params_sigmoid.coef0 = coef0;
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

OpenCLKernel::~OpenCLKernel() {
        delete[] features;
	delete[] my_nodes;
	delete[] i_vec;
	delete[] y;
}

void OpenCLKernel::InitialyzeInputs(svm_node * const *x) {
	//I need the total size to pass to the OpenCL device plus the indexs where each row
	// starts and finish (for implement the swap parts!!!!).
	//I need to know the maximum number of features to reserve this space in the device 
	// and calcule faster in future steps of the process
        features = new int[l*2];
	total_nodes = 0;
	max_features = 0; 
	int total_features = 0;

	perro_debug = 0;


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


OpenCLKernel::OpenCLKernel(const int l,svm_node * const *x, const schar * y_, const svm_parameter& param, double *XX):OpenCLGeneral(6, "./gpuImplementations/OpenCLLibsvmVersionMultiKernel/GPUKernelCalculation.cl") {

    kernel_type = param.kernel_type;
    degree = param.degree;
    gamma = param.gamma;
    coef0 = param.coef0;

    this->l = l;
    y = new double[l];
    for(int i = 0; i < l; ++i) y[i] = (double)y_[i];
    x_square = XX;
    InitialyzeInputs(x);

    double output[l];
    if(!CreateInputOutputMemoryObject(memObjects[0], output, l, sizeof(output[0]))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    } 
    if(!CreateInputMemoryObject(memObjects[1], my_nodes, total_nodes, sizeof(my_nodes[0]))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }
    if(!CreateInputMemoryObject(memObjects[2], features, l*2, sizeof(features[0]))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }

    if(!CreateInputMemoryObject(memObjects[3], i_vec, max_features, sizeof(i_vec[0]))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }
    if(!CreateInputMemoryObject(memObjects[4], y, l, sizeof(double))) {
	std::cerr << "Error creating memory objects in device." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return;
    }



    loadKernel();

    
	    if(kernel_type == RBF) {
		    if(!CreateInputMemoryObject(memObjects[5], x_square, l, sizeof(double))) {
			std::cerr << "Error creating memory objects in device." << std::endl;
		    	Cleanup(context, commandQueue, program, kernel, memObjects);
		   	return;
		    }
		    clSetKernelArg(kernel,7,sizeof(cl_mem), &memObjects[5]);
	    }
    
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[1]); // todos los nodos... la pseudo-matriz
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[2]);   // features
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[3]);  // input vector
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[0]);  // output
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[4]);  // labels

    clSetKernelArg(kernel,6,sizeof(int), &l);


    w_gr_size = 32;
    lower_mult = w_gr_size*(l/w_gr_size);
    if(l%w_gr_size!=0) lower_mult = lower_mult + w_gr_size;
    //std::cout << "Vectors#: " << l << "     " << "  Max feats: " << max_features << "  " << lower_mult/w_gr_size << "	 " << w_gr_size << "	  " << lower_mult << std::endl;


}


bool OpenCLKernel::CreateOutputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__) {
    memObject = clCreateBuffer(context, CL_MEM_READ_WRITE, size__ * quant_elems, NULL, NULL);	    
    if (memObject == NULL) {
	std::cerr << "Error creating vector memory object." << std::endl;
	return false;
    }
    return true;
}

bool OpenCLKernel::SetVector(int i){
 
    for(int j = 0; j < max_features; ++j) i_vec[j] = 0;
   
    for(int j = features[i]; j < features[i+l]; ++j) {
	i_vec[my_nodes[j].index] = my_nodes[j].value;
    }

    errNum = clEnqueueWriteBuffer(commandQueue, memObjects[3], CL_FALSE, 0, sizeof(double)*max_features, i_vec, 0, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
		std::cerr << "Error writing algo." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		act_index = -1;
		return false;
    }
    act_index = i;

}

bool OpenCLKernel::Swap(int i, int j){
    int aux[2]; 
    aux[0] = features[i];        	aux[1] = features[i+l];
    features[i] = features[j]; 		features[i+l] = features[j+l];
    features[j] = aux[0];        	features[j+l] = aux[1];    
    double s = y[i];
    y[i] = y[j];    
    y[j] = s;
    
    clEnqueueWriteBuffer(commandQueue, memObjects[2], CL_FALSE, 0, sizeof(int)*2*l, features, 0, NULL, NULL);
    
    if(kernel_type == RBF) {
	    s = x_square[i];	
    	x_square[i] = x_square[j];
    	x_square[j] = s;
        clEnqueueWriteBuffer(commandQueue, memObjects[5], CL_FALSE, 0, sizeof(double)*l, x_square, 0,NULL, NULL);  
    }
}

bool OpenCLKernel::CreateInputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__) {
    memObject = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size__ * quant_elems, a, NULL);
    if (memObject == NULL) {
	std::cerr << "Error creating input memory object." << std::endl;
	return false;
    }
    return true;
}

bool OpenCLKernel::CreateInputOutputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__) {
    memObject = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size__ * quant_elems, a, NULL);
    if (memObject == NULL) {
	std::cerr << "Error creating input memory object." << std::endl;
	return false;
    }
    return true;
}

void OpenCLKernel::runMatrixMult(double *result, int offset) { 
        *avoid_const = *avoid_const + 1;
	    *common_labels = y[act_index];

        switch(kernel_type){
                case LINEAR:
                        clSetKernelArg(kernel,5,sizeof(params_dot), &params_dot);
                case SIGMOID:
                        clSetKernelArg(kernel,5,sizeof(params_sigmoid), &params_sigmoid);
                        break;
                case RBF:
                        params_rbf.common_x_square = x_square[act_index];
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
