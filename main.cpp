#include <stdio.h>
#include "clDevice.hpp"
#include "ImageStabilization.hpp"
const char* image_stabilization =
#include "image_stabilization.cl"
;
int main(){
    clPlatform platform;
    clDevice** cl_devices = (clDevice**)malloc(platform.getNumberDevices()*sizeof(clDevice*));
    for (size_t i = 0; i < platform.getNumberDevices(); i++) {
        cl_devices[i] = new clDevice(&platform, i);
        cl_devices[i]->clPushProgram((cl_char*)image_stabilization, sizeof(image_stabilization), NULL);
    }
    size_t i = 0;
	if (i < platform.getNumberDevices()) {
		Image_Stabilization image(cl_devices[i], 1280, 720, 30, 30, 10);
	
    }
    for (size_t i = 0; i < platform.getNumberDevices(); i++) {
        delete cl_devices[i];
    }
    printf("%s\n", "Hello World");
    return 0;
}