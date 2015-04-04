#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"

//#define NAIVE

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}

__global__ void gpu_histogram(int * hist_out, unsigned char * img_in, int * img_size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nbr_bin = 256;

#ifdef NAIVE
	if (id < *img_size) {
        atomicAdd(&hist_out[(int)img_in[id]], 1);
	}
#else
	__shared__ int hist_temp[256];
    //for ( i = 0; i < nbr_bin; i ++){
	if (threadIdx.x < nbr_bin) {
        hist_temp[threadIdx.x] = 0;
	}
    //}
	__syncthreads();
    //for ( i = 0; i < img_size; i ++){
	if (id < *img_size) {
        //atomicAdd(&hist_out[(int)img_in[id]], 1);
		atomicAdd(&hist_temp[(int)img_in[id]], 1);
	}
	__syncthreads();
	if (threadIdx.x < nbr_bin) {
		atomicAdd(&hist_out[threadIdx.x], hist_temp[threadIdx.x]);
	}
#endif
    //}
}

void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
	int i = 0;
	int * lut = (int*)malloc(sizeof(int) * nbr_bin);
    int *g_lut = 0;
	HANDLE_ERROR( cudaMalloc(&g_lut, sizeof(int)* nbr_bin) );
	int *cdf = (int *)malloc(sizeof(int)* nbr_bin);
	int *g_cdf = 0;
	HANDLE_ERROR( cudaMalloc(&g_cdf, sizeof(int)* nbr_bin) );
	
	int * g_hist_in = 0;
	HANDLE_ERROR( cudaMalloc(&g_hist_in, sizeof(int) * 256) );
	int * g_img_size = 0;
	HANDLE_ERROR( cudaMalloc(&g_img_size, sizeof(int)) );
	int * g_nbr_bin = 0;
	HANDLE_ERROR( cudaMalloc(&g_nbr_bin, sizeof(int)) );
	int min = 0;
	i = 0;
	while(min == 0){
		min = hist_in[i++];
	}
	int * g_min = 0;
	HANDLE_ERROR( cudaMalloc(&g_min, sizeof(int)) );
	HANDLE_ERROR( cudaMemcpy(g_min, &min, sizeof(int), cudaMemcpyHostToDevice) );

	unsigned char *g_img_in = 0;
	HANDLE_ERROR( cudaMalloc(&g_img_in, sizeof(unsigned char) * img_size) );
	unsigned char *g_img_out = 0;
	HANDLE_ERROR( cudaMalloc(&g_img_out, sizeof(unsigned char) * img_size) );

	HANDLE_ERROR( cudaMemcpy(g_img_in, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(g_img_out, img_out, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice) );


	HANDLE_ERROR( cudaMemcpy(g_hist_in, hist_in, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(g_img_size, &img_size, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(g_nbr_bin, &nbr_bin, sizeof(int), cudaMemcpyHostToDevice) );

	cdf[0] = hist_in[0];
	for(i = 1; i < nbr_bin; i++){
        cdf[i] = hist_in[i] + cdf[i - 1];
	}


	HANDLE_ERROR( cudaMemset(g_lut, 1, sizeof(int) * nbr_bin) );
	HANDLE_ERROR( cudaMemcpy(g_cdf, cdf, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice) );

	gpu_histogram_equalization_lutcalc<<< 1, MAXTHREADS >>>(g_cdf, g_hist_in, g_lut, g_img_size, g_nbr_bin, g_min);

	HANDLE_ERROR( cudaMemcpy(lut, g_lut, sizeof(int) * nbr_bin, cudaMemcpyDeviceToHost) );

	int block_count = (int)ceil((float)img_size / MAXTHREADS);
	gpu_histogram_equalization_imgoutcalc<<< block_count, MAXTHREADS >>>(g_img_out, g_img_in, g_lut, g_img_size);

	HANDLE_ERROR( cudaMemcpy(img_out, g_img_out, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost) );
	/*for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }*/
	free(cdf);
	free(lut);
	cudaFree(g_lut);
	cudaFree(g_cdf);
	cudaFree(g_img_out);
	cudaFree(g_img_in);
	cudaFree(g_hist_in);
	cudaFree(g_img_size);
	cudaFree(g_nbr_bin);
}

__global__ void gpu_histogram_equalization_lutcalc(int * cdf,
                            int * hist_in, int * lut, int * img_size, int * nbr_bin, int * min){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (id < *nbr_bin) {
		/*int mmin = *min;
		while(mmin == 0){
			mmin = hist_in[i++];
		}*/
		lut[id] = 0;
		int d = 0;
		d = (*img_size) - (*min);
    
        lut[id] = (int)(((float)cdf[id] - *min)*255/d + 0.5);
        if(lut[id] < 0){
            lut[id] = 0;
        }
		if (lut[id] > 255) {
			lut[id] = 255;
		}
	}
}

__global__ void gpu_histogram_equalization_imgoutcalc(unsigned char * img_out, unsigned char * img_in, 
                            int * lut, int * img_size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < *img_size) {
    
        img_out[i] = (unsigned char)lut[(int)img_in[i]];
        
    }
}