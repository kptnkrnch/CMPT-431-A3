Server compiling instructions:
	Steps:
	1. Set the path (for nvcc) using - export PATH=$PATH:/usr/local/cuda/bin
	2. For the Tiled implementation, compile using - nvcc -std=c++11 contrast-enhancement.cu histogram-equalization.cu kernel.cu
	   For the Naive implementation, compile using - nvcc -std=c++11 -D NAIVE contrast-enhancement.cu histogram-equalization.cu kernel.cu

Who did what:
	Joshua Campbell (jkcampbe):
		-ported the gpu_histogram function
		-ported the gpu_histogram_equalization wrapper/controller function for cuda kernel functions
		-created the gpu_histogram_equalization_lutcalc function to assist in calculations
		-created/ported the gpu_histogram_equalization_imgoutcalc function
		-ported the gpu_hsl2rgb function
		-ported the gpu_rgb2hsl function
		-ported the gpu_Hue_2_RGB function
		-ported the gpu_clip_rgb function