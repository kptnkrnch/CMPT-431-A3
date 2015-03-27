#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"


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
	printf("CPU:\n");
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        printf("[%d] - %d\n", i, cdf);
        
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

__global__ void gpu_histogram(int * hist_out, unsigned char * img_in, int * img_size, int * nbr_bin, int * debug){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
    //for ( i = 0; i < nbr_bin; i ++){
	if (id < *nbr_bin) {
        hist_out[id] = 0;
	}
    //}

    //for ( i = 0; i < img_size; i ++){
	if (id < *img_size) {
        hist_out[img_in[id]] ++;
	}
	if (id > 1024) {
		*debug = 777;
	} else {
		*debug = 666;
	}
    //}
}

void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
	int i = 0;
	int * lut = (int*)malloc(sizeof(int) * nbr_bin);
    int *g_lut = 0;
	cudaMalloc(&g_lut, sizeof(int)* nbr_bin);
	int *cdf = (int *)malloc(sizeof(int)* nbr_bin);
	int *g_cdf = 0;
	cudaMalloc(&g_cdf, sizeof(int)* nbr_bin);
	unsigned char *g_img_in = 0;
	cudaMalloc(&g_img_in, sizeof(unsigned char) * img_size);
	unsigned char *g_img_out = 0;
	cudaMalloc(&g_img_out, sizeof(unsigned char) * img_size);
	int * g_hist_in = 0;
	cudaMalloc(&g_hist_in, sizeof(int) * 256);
	int * g_img_size = 0;
	cudaMalloc(&g_img_size, sizeof(int));
	int * g_nbr_bin = 0;
	cudaMalloc(&g_img_size, sizeof(int));

	cudaMemcpy(g_img_in, &img_in, sizeof(int) * img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(g_img_out, &img_out, sizeof(int) * img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(g_hist_in, &hist_in, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice);
	cudaMemcpy(g_img_size, &img_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_nbr_bin, &nbr_bin, sizeof(int), cudaMemcpyHostToDevice);

	cdf[0] = hist_in[0];
	for(i = 1; i < nbr_bin; i++){
        cdf[i] = hist_in[i] + cdf[i - 1];
	}
	printf("GPU:\n");
	for (int i = 0; i < 256; i++) {
		printf("[%d] - %d\n", i, cdf[i]);
	}

	cudaMemcpy(g_cdf, cdf, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice);

	gpu_histogram_equalization_lutcalc<<< 1, nbr_bin >>>(g_cdf, g_hist_in, g_lut, g_img_size, g_nbr_bin);

	cudaMemcpy(lut, g_lut, sizeof(int) * nbr_bin, cudaMemcpyDeviceToHost);
	
	int block_count = (int)ceil((float)img_size / MAXTHREADS);
	gpu_histogram_equalization_imgoutcalc<<< block_count, MAXTHREADS >>>(g_img_out, g_img_in, g_lut, g_img_size);

	cudaMemcpy(img_out, g_img_out, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);

	free(cdf);
	free(lut);
	cudaFree(g_lut);
	cudaFree(g_cdf);
	cudaFree(g_img_size);
	cudaFree(g_nbr_bin);
}

__global__ void gpu_histogram_equalization_lutcalc(int * cdf,
                            int * hist_in, int * lut, int * img_size, int * nbr_bin){
    int i, min, d;
    // Construct the LUT by calculating the CDF 
    min = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    while(min == 0){
        min = hist_in[i++];
    }
    d = *img_size - min;
	if (i < *nbr_bin) {
    //for(i = 0; i < *nbr_bin; i ++){
        //cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf[i] - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }
}

__global__ void gpu_histogram_equalization_imgoutcalc(unsigned char * img_out, unsigned char * img_in, 
                            int * lut, int * img_size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *img_size) {
    //for(i = 0; i < *img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}