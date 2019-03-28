#include "ImageStabilization.hpp"


Image_Stabilization::Image_Stabilization(clDevice* device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint radius)
{
	_gpu_data = NULL;
	_gpu_data = (gpu_data*)malloc(sizeof(gpu_data));
	_gpu_data->_device = device;
	_gpu_data->gauss_image_data.kernel_image_gauss = device->findKernel((const cl_char*)"make_gauss_vec1_image_uchar_rgba", sizeof("make_gauss_vec1_image_uchar_rgba"));
	_gpu_data->stabilization_image_data.kernel_image_stabilization = device->findKernel((const cl_char*)"version_1_MSE_stabilization_image_rgba", sizeof("version_1_MSE_stabilization_image_rgba"));
	size_t x = sqrt(float(_gpu_data->_device->kernelInfo[_gpu_data->stabilization_image_data.kernel_image_stabilization].max_work_group_size));
	size_t y = _gpu_data->_device->kernelInfo[_gpu_data->stabilization_image_data.kernel_image_stabilization].max_work_group_size / x;
	_gpu_data->height = height;
	_gpu_data->width = width;
	_gpu_data->gauss_image_data.globalWork[0] = width;
	_gpu_data->gauss_image_data.globalWork[1] = height;
	_gpu_data->gauss_image_data.globalWork[2] = 1;
	_gpu_data->stabilization_image_data.length_args[0] = sizeof(cl_uint);
	_gpu_data->stabilization_image_data.length_args[1] = sizeof(cl_uint);
	_gpu_data->stabilization_image_data.length_args[2] = sizeof(cl_uint);
	_gpu_data->stabilization_image_data.length_args[3] = sizeof(cl_uint);
	_gpu_data->stabilization_image_data.length_args[4] = sizeof(cl_uint);
	_gpu_data->stabilization_image_data.indices[0] = width;
	_gpu_data->stabilization_image_data.indices[1] = height;
	_gpu_data->stabilization_image_data.indices[2] = radius;
	_gpu_data->stabilization_image_data.indices[3] = block_x;
	_gpu_data->stabilization_image_data.indices[4] = block_y;
	_gpu_data->stabilization_image_data.localWork[0] = x;
	_gpu_data->stabilization_image_data.localWork[1] = y;
	_gpu_data->stabilization_image_data.localWork[2] = 1;
	_gpu_data->stabilization_image_data.globalWork[0] = width / block_x;
	_gpu_data->stabilization_image_data.globalWork[1] = height / block_y;
	_gpu_data->stabilization_image_data.globalWork[2] = 1;
	if (_gpu_data->stabilization_image_data.globalWork[0] % _gpu_data->stabilization_image_data.localWork[0])
		_gpu_data->stabilization_image_data.globalWork[0] += _gpu_data->stabilization_image_data.localWork[0] - _gpu_data->stabilization_image_data.globalWork[0] % _gpu_data->stabilization_image_data.localWork[0];
	if (_gpu_data->stabilization_image_data.globalWork[1] % _gpu_data->stabilization_image_data.localWork[1])
		_gpu_data->stabilization_image_data.globalWork[1] += _gpu_data->stabilization_image_data.localWork[1] - _gpu_data->stabilization_image_data.globalWork[1] % _gpu_data->stabilization_image_data.localWork[1];
	if (_gpu_data->stabilization_image_data.globalWork[2] % _gpu_data->stabilization_image_data.localWork[2])
		_gpu_data->stabilization_image_data.globalWork[2] += _gpu_data->stabilization_image_data.localWork[2] - _gpu_data->stabilization_image_data.globalWork[2] % _gpu_data->stabilization_image_data.localWork[2];

	x = x - (x %  block_x);
	y = y - (y %  block_y);
	_gpu_data->width = width;
	_gpu_data->height = height;
	_gpu_data->stabilization_image_data.localWork[0] = x, _gpu_data->stabilization_image_data.localWork[1] = y, _gpu_data->stabilization_image_data.localWork[2] = 1;
	_gpu_data->stabilization_image_data.globalWork[0] = width, _gpu_data->stabilization_image_data.globalWork[1] = height, _gpu_data->stabilization_image_data.globalWork[2] = 1;
	if (_gpu_data->stabilization_image_data.globalWork[0] % _gpu_data->stabilization_image_data.localWork[0])
		_gpu_data->stabilization_image_data.globalWork[0] += _gpu_data->stabilization_image_data.localWork[0] - _gpu_data->stabilization_image_data.globalWork[0] % _gpu_data->stabilization_image_data.localWork[0];
	if (_gpu_data->stabilization_image_data.globalWork[1] % _gpu_data->stabilization_image_data.localWork[1])
		_gpu_data->stabilization_image_data.globalWork[1] += _gpu_data->stabilization_image_data.localWork[1] - _gpu_data->stabilization_image_data.globalWork[1] % _gpu_data->stabilization_image_data.localWork[1];
	if (_gpu_data->stabilization_image_data.globalWork[2] % _gpu_data->stabilization_image_data.localWork[2])
		_gpu_data->stabilization_image_data.globalWork[2] += _gpu_data->stabilization_image_data.localWork[2] - _gpu_data->stabilization_image_data.globalWork[2] % _gpu_data->stabilization_image_data.localWork[2];
	int origin_width = width;
	int origin_height = height;
	size_t length_row_pitch_data = width * sizeof(cl_uchar4);
	_gpu_data->norm_image_gpu_0 = device->mallocImage2DMemory(NULL, height, width, length_row_pitch_data, CL_RGBA, CL_UNORM_INT8);
	_gpu_data->norm_image_gpu_1 = device->mallocImage2DMemory(NULL, height, width, length_row_pitch_data, CL_RGBA, CL_UNORM_INT8);
	_gpu_data->memory_buffer = device->mallocBufferMemory(NULL, width * height * sizeof(cl_uchar4));
	_gpu_data->gauss_image_data.indices[0] = origin_width, _gpu_data->gauss_image_data.indices[1] = origin_height, _gpu_data->gauss_image_data.indices[2] = block_x, _gpu_data->gauss_image_data.indices[3] = block_y;
	_gpu_data->stabilization_image_data.indices[0] = origin_width, _gpu_data->stabilization_image_data.indices[1] = origin_height, _gpu_data->stabilization_image_data.indices[2] = origin_width, _gpu_data->stabilization_image_data.indices[3] = origin_height, _gpu_data->stabilization_image_data.indices[4] = block_x, _gpu_data->stabilization_image_data.indices[5] = block_y, _gpu_data->stabilization_image_data.indices[6] = 0u;
	_gpu_data->gauss_image_data.length_args[0] = sizeof(cl_uint), _gpu_data->gauss_image_data.length_args[1] = sizeof(cl_uint), _gpu_data->gauss_image_data.length_args[2] = sizeof(cl_uint), _gpu_data->gauss_image_data.length_args[3] = sizeof(cl_uint);

	int size_local_memory = (x*y);
	_gpu_data->stabilization_image_data.length_args[0] = sizeof(cl_uint), _gpu_data->stabilization_image_data.length_args[1] = sizeof(cl_uint), _gpu_data->stabilization_image_data.length_args[2] = sizeof(cl_uint), _gpu_data->stabilization_image_data.length_args[3] = sizeof(cl_uint),
		_gpu_data->stabilization_image_data.length_args[4] = sizeof(cl_uint), _gpu_data->stabilization_image_data.length_args[5] = sizeof(cl_uint), _gpu_data->stabilization_image_data.length_args[6] = -(int)(size_local_memory) * sizeof(cl_float);
	ptr_gauss_function = &Image_Stabilization::gpu_Calculate_Gauss_function;
	ptr_stabilization_function = &Image_Stabilization::gpu_Stabilization_function;
}
void Image_Stabilization::gpu_Calculate_Gauss_function(void* data, void* result) {
	size_t length_row_pitch_data = _gpu_data->width * sizeof(cl_uchar4);
	_gpu_data->_device->write2DImage(data, _gpu_data->norm_image_gpu_0, _gpu_data->width, _gpu_data->height);
	_gpu_data->_device->callOpenclFunction(_gpu_data->gauss_image_data.kernel_image_gauss, &_gpu_data->memory_buffer, &_gpu_data->norm_image_gpu_0, (cl_char*)_gpu_data->gauss_image_data.indices, _gpu_data->gauss_image_data.length_args, 1, 1, 4, _gpu_data->gauss_image_data.globalWork);
	_gpu_data->_device->readBuffer(result, _gpu_data->memory_buffer, _gpu_data->height*_gpu_data->width*sizeof(cl_uchar4));
}

void Image_Stabilization::gpu_Stabilization_function(void* data_next_image, void* result) {
	size_t images[] = { _gpu_data->norm_image_gpu_0, _gpu_data->norm_image_gpu_1 };
	_gpu_data->_device->write2DImage(data_next_image, _gpu_data->norm_image_gpu_1, _gpu_data->stabilization_image_data.indices[0], _gpu_data->stabilization_image_data.indices[1]);
	_gpu_data->_device->callOpenclFunction(_gpu_data->stabilization_image_data.kernel_image_stabilization, &_gpu_data->memory_buffer, images, (cl_char*)_gpu_data->stabilization_image_data.indices, _gpu_data->stabilization_image_data.length_args, 1, 2, 5, _gpu_data->stabilization_image_data.globalWork, _gpu_data->stabilization_image_data.localWork);
	_gpu_data->_device->readBuffer(result, _gpu_data->memory_buffer, _gpu_data->stabilization_image_data.indices[0] * _gpu_data->stabilization_image_data.indices[1] * sizeof(cl_uchar4));
}
void Image_Stabilization::Calculate_Gauss_function(void* data, void* result) {
	(this->*ptr_gauss_function)(data, result);
}
void Image_Stabilization::Stabilization_function(void* data_next_image, void* result) {
	(this->*ptr_stabilization_function)(data_next_image, result);
}
Image_Stabilization::~Image_Stabilization()
{
	if (_gpu_data) {
		_gpu_data->_device->freeMemory(_gpu_data->norm_image_gpu_0);
		_gpu_data->_device->freeMemory(_gpu_data->norm_image_gpu_1);
		_gpu_data->_device->freeMemory(_gpu_data->memory_buffer);
		free(_gpu_data);
	}
}