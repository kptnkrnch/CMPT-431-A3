#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"
#include <chrono>
//#include <Windows.h>
#include <iostream>

void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);
void run_gpu_gray_test(PGM_IMG img_in);

void GetElapsedTimeWindows(bool end);

#define MSEC 1000


int main(){
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;
    
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm("in.pgm");
    run_cpu_gray_test(img_ibuf_g);
    run_gpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g);
    
    printf("Running contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm("in.ppm");
    run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    return 0;
}

void run_gpu_color_test(PPM_IMG img_in)
{
    printf("Starting GPU HSL processing...\n");
    cudaEvent_t start, end;
    float elapsedTime;
    PPM_IMG img_obuf_hsl;
    PPM_IMG img_obuf_yuv;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&end) );
    HANDLE_ERROR( cudaEventRecord(start,0) );
    img_obuf_hsl = gpu_contrast_enhancement_c_hsl(img_in);
    HANDLE_ERROR( cudaEventRecord(end,0) );
    HANDLE_ERROR( cudaEventSynchronize(end) );
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, end) );
    printf("\tGPU Color HSL time: %fms\n", elapsedTime);
    write_ppm(img_obuf_hsl, "gpu_out_hsl.ppm");


    HANDLE_ERROR( cudaEventRecord(start,0) );
    img_obuf_yuv = gpu_contrast_enhancement_c_yuv(img_in);
    HANDLE_ERROR( cudaEventRecord(end,0) );
    HANDLE_ERROR( cudaEventSynchronize(end) );
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, end) );
    printf("\tGPU Color YUV time: %fms\n", elapsedTime);
    write_ppm(img_obuf_yuv, "gpu_out_yuv.ppm");

    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(end) );
}

void run_gpu_gray_test(PGM_IMG img_in)
{
    printf("Starting GPU processing...\n");

    PGM_IMG img_obuf;

    cudaEvent_t start, end;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,0);

    img_obuf = gpu_contrast_enhancement_g(img_in);

    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("\tGPU Gray time: %fms\n", elapsedTime);
    
    write_pgm(img_obuf, "gpu_out.pgm");
    free_pgm(img_obuf);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

void run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    
    printf("Starting CPU processing...\n");
    
    auto start_hsl = std::chrono::steady_clock::now();
	//GetElapsedTimeWindows(false);
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
	//GetElapsedTimeWindows(true);
    auto end_hsl = std::chrono::steady_clock::now();
    auto elapsed_hsl = std::chrono::duration_cast<std::chrono::milliseconds>(end_hsl - start_hsl);
    std::cout << "\tCPU Color HSL time: " << elapsed_hsl.count() << "ms" << std::endl;
    
    write_ppm(img_obuf_hsl, "cpu_out_hsl.ppm");

    auto start_yuv = std::chrono::steady_clock::now();
	//GetElapsedTimeWindows(false);
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
	//GetElapsedTimeWindows(true);
    auto end_yuv = std::chrono::steady_clock::now();
    auto elapsed_yuv = std::chrono::duration_cast<std::chrono::milliseconds>(end_yuv - start_yuv);
    std::cout << "\tCPU Color YUV time: " << elapsed_yuv.count() << "ms" << std::endl;
    
    write_ppm(img_obuf_yuv, "cpu_out_yuv.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}




void run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    
    printf("Starting CPU processing...\n");

    auto start = std::chrono::steady_clock::now();
	//GetElapsedTimeWindows(false);
    img_obuf = contrast_enhancement_g(img_in);
	//GetElapsedTimeWindows(true);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\tCPU Graytime: " << elapsed.count() << "ms" << std::endl;
    
    write_pgm(img_obuf, "cpu_out.pgm");
    free_pgm(img_obuf);
}



PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

//void GetElapsedTimeWindows(bool end) {
//	static LARGE_INTEGER frequency;        // ticks per second
//	static LARGE_INTEGER t1, t2;           // ticks
//	double elapsedTime;
//
//	// get ticks per second
//	QueryPerformanceFrequency(&frequency);
//
//	// start timer
//	if (!end) {
//		QueryPerformanceCounter(&t1);
//	} else {
//		// stop timer
//		QueryPerformanceCounter(&t2);
//
//		// compute and print the elapsed time in millisec
//		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
//		printf("Time: %f (ms)\n", elapsedTime);
//	}
//}