#pragma once
#include "clDevice.hpp"

struct gpu_stabilization_image {
	size_t globalWork[3];
	size_t localWork[3];
	cl_int kernel_image_stabilization;
	cl_uint indices[6];
	cl_int length_args[6];
};
struct gpu_gauss_image {
	size_t globalWork[3];
	cl_int kernel_image_gauss;
	cl_uint indices[7];
	cl_int length_args[7];
};
struct gpu_data {
	gpu_stabilization_image stabilization_image_data;
	gpu_gauss_image gauss_image_data;
	clDevice* _device;
	size_t width;
	size_t height;
	size_t norm_image_gpu_0;
	size_t norm_image_gpu_1;
	size_t memory_buffer;
};
class Image_Stabilization
{
	gpu_data* _gpu_data;
	void(Image_Stabilization::*ptr_gauss_function)(void* data, void* result);
	void(Image_Stabilization::*ptr_stabilization_function)(void* data, void* result);
public:
	Image_Stabilization(clDevice * device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint radius);

	void gpu_Calculate_Gauss_function(void * data, void * result);
	void cpu_sse2_Gauss_function(void * data, void * result);
	void cpu_MSE_SSE2_Stabilization_function(void * data, void * result);
	void gpu_Stabilization_function(void * data_next_image, void * result);
	void Calculate_Gauss_function(void * data, void * result);
	void Stabilization_function(void * data_next_image, void * result);
	void cpu_Stabilization_function(void * data, void * result);
	~Image_Stabilization();
};