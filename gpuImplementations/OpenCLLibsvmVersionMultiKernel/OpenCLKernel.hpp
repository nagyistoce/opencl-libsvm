
#include <iostream>
#include <fstream>
#include <sstream>


#ifndef OPENCL_GENERAL
#include "../OpenCLgeneral/OpenCLGeneral.hpp"
#define OPENCL_GENERAL
#endif


#ifndef TYPPES
#define TYPPES
#include "../../types.h"
#endif


/*
#ifndef SVM
#include "../../svm.h"
#define SVM
#endif
*/

#ifndef SCHAR
typedef signed char schar;
#define SCHAR
#endif

#ifndef TIPOS
struct param_struct_dot {
    double common_label;
};

struct param_struct_poly{
    double common_label;
    double gamma;
    double coef0;
    int degree;
};

struct param_struct_sigmoid{
    double common_label;
    double gamma;
    double coef0;
};

struct param_struct_rbf {
    double common_label;
    double gamma;
    double common_x_square;
};

#define TIPOS
#endif


class OpenCLKernel:OpenCLGeneral {

    public:
	OpenCLKernel(const int l,svm_node * const *x, const schar *y, const svm_parameter& param, double *x_squar);
	~OpenCLKernel();
	bool SetVector(int i);
	bool Swap(int i, int j);
	void runMatrixMult(double *result, int offset); //the offset is for calcule only what's required by cache!
	void runPrecalculation(int first,int last, double **results);
	int kernel_type;
        int perro_debug;

	void startPrecalculation(int prec_step);
	void finishedPrecalculation();

	int *avoid_const;
    protected:
        param_struct_dot params_dot;
	param_struct_poly params_poly;
        param_struct_sigmoid params_sigmoid;
	param_struct_rbf params_rbf;
        double *common_labels;

	int degree;
	double gamma;
	double coef0;

        int w_gr_size;
        int lower_mult;
        int feat_lower_mult;

	bool required_comp_kernel;
        cl_kernel extra_kernel;

	int *features;
	int max_features;
	int total_nodes;
	svm_node *my_nodes;
	double *i_vec;
	double *y;
	int act_index;
	double *x_square;

	int l;
	
	void loadKernel();
	void InitialyzeInputs(svm_node * const *x);
	bool CreateOutputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__);
	bool CreateInputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__);
	bool CreateInputOutputMemoryObject(cl_mem &memObject, void *a, int quant_elems, int size__);


        cl_kernel kernel_init; 
        cl_mem super_sortida;
};
