#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include "hist-equ.cuh"


PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];

    //TODO remove
    // cudaEvent_t start, end;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    // cudaEventRecord(start,0);
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\thistogram time: %fms\n", elapsedTime);
	
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    //TODO remove
    // cudaEventDestroy(start);
    // cudaEventDestroy(end);
    return result;
}


PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;

    //TODO remove
    // cudaEvent_t start, end;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    
    unsigned char * y_equ;
    int hist[256];
    
    // cudaEventRecord(start,0);
    yuv_med = rgb2yuv(img_in);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\trgb2yuv time: %fms\n", elapsedTime);

    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));
    
    histogram(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);
	
    histogram_equalization(y_equ,yuv_med.img_y,hist,yuv_med.h * yuv_med.w, 256);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    // cudaEventRecord(start,0);
    result = yuv2rgb(yuv_med);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\tyuv2rgb time: %fms\n", elapsedTime);

    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);

        //TODO remove
    // cudaEventDestroy(start);
    // cudaEventDestroy(end);
    
    return result;
}


PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;

        //TODO remove
    // cudaEvent_t start, end;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    
    unsigned char * l_equ;
    int hist[256];

    // cudaEventRecord(start,0);
    hsl_med = rgb2hsl(img_in);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\trgb2hsl time: %fms\n", elapsedTime);

    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));

    histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);
    histogram_equalization(l_equ, hsl_med.l,hist,hsl_med.width*hsl_med.height, 256);
    
    free(hsl_med.l);
    hsl_med.l = l_equ;

    // cudaEventRecord(start,0);
    result = hsl2rgb(hsl_med);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\thsl2rgb time: %fms\n", elapsedTime);

    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);

            //TODO remove
    // cudaEventDestroy(start);
    // cudaEventDestroy(end);
    return result;
}


//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG rgb2hsl(PPM_IMG img_in)
{
    int i;
    float H, S, L;
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    for(i = 0; i < img_in.w*img_in.h; i ++){
        
        float var_r = ( (float)img_in.img_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)img_in.img_g[i]/255 );
        float var_b = ( (float)img_in.img_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        img_out.h[i] = H;
        img_out.s[i] = S;
        img_out.l[i] = (unsigned char)(L*255);
    }
    
    return img_out;
}


float Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}


//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
PPM_IMG hsl2rgb(HSL_IMG img_in)
{
    int i;
    PPM_IMG result;
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    for(i = 0; i < img_in.width*img_in.height; i ++){
        float H = img_in.h[i];
        float S = img_in.s[i];
        float L = img_in.l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * Hue_2_RGB( var_1, var_2, H );
            b = 255 * Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        result.img_r[i] = r;
        result.img_g[i] = g;
        result.img_b[i] = b;
    }

    return result;
}


//Convert RGB to YUV, all components in [0, 255]
YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }
    
    return img_out;
}


unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}


//Convert YUV to RGB, all components in [0, 255]
PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int  rt,gt,bt;
    int y, cb, cr;
    
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }
    
    return img_out;
}


///////////////////////////////////////////////////////////////////////////////
//
//  GPU SECTION
//
///////////////////////////////////////////////////////////////////////////////


PGM_IMG gpu_contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
	int img_size = 0;
	int grey_count = 0;
    
	unsigned char * cuda_img_in = 0;
	unsigned char * cuda_img_out = 0;
	int * cuda_img_size = 0;
	int * cuda_hist = 0;
	

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
	cudaMalloc(&cuda_img_in, sizeof(unsigned char) * result.w * result.h);
	cudaMalloc(&cuda_img_out, sizeof(unsigned char) * result.w * result.h);
	cudaMalloc(&cuda_img_size, sizeof(int));
	cudaMalloc(&cuda_hist, sizeof(int) * 256);

	img_size = img_in.h * img_in.w;
	grey_count = 256;

	cudaMemcpy(cuda_img_in, img_in.img, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_img_size, &img_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(cuda_hist, 0, sizeof(int) * 256);
	int block_count = (int)ceil((float)img_size / MAXTHREADS);

	gpu_histogram<<< block_count, MAXTHREADS >>>(cuda_hist, cuda_img_in, cuda_img_size);
	cudaMemcpy(hist, cuda_hist, sizeof(int) * 256, cudaMemcpyDeviceToHost);
	
	gpu_histogram_equalization(result.img, img_in.img, hist, img_size, grey_count);

	cudaFree(cuda_img_in);
	cudaFree(cuda_img_out);
	cudaFree(cuda_img_size);
	cudaFree(cuda_hist);

	return result;
}


PPM_IMG gpu_contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    PPM_IMG result;
    
    unsigned char * l_equ;
	unsigned char * temp_hsl_l;
	float * temp;

    //TODO remove
    // cudaEvent_t start, end;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);

	int * t_hist = (int *)malloc(sizeof(int) * 256);

	int img_size = img_in.w * img_in.h;
	int nbr_size = 256;

	temp_hsl_l = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
	temp = (float *)malloc(sizeof(float) * img_size);

	int * gpu_hist = 0;
	int * gpu_img_size = 0;
	unsigned char * gpu_img_in_r = 0;
	unsigned char * gpu_img_in_g = 0;
	unsigned char * gpu_img_in_b = 0;
	unsigned char * gpu_img_out_r = 0;
	unsigned char * gpu_img_out_g = 0;
	unsigned char * gpu_img_out_b = 0;
	float * gpu_hsl_img_h = 0;
	float * gpu_hsl_img_s = 0;
	unsigned char * gpu_hsl_img_l = 0;
	cudaMalloc(&gpu_hist, sizeof(int) * 256);
	cudaMalloc(&gpu_img_size, sizeof(int));
	cudaMalloc(&gpu_img_in_r, sizeof(unsigned char) * img_size);
	cudaMalloc(&gpu_img_in_g, sizeof(unsigned char) * img_size);
	cudaMalloc(&gpu_img_in_b, sizeof(unsigned char) * img_size);
	cudaMalloc(&gpu_img_out_r, sizeof(unsigned char) * img_size);
	cudaMalloc(&gpu_img_out_g, sizeof(unsigned char) * img_size);
	cudaMalloc(&gpu_img_out_b, sizeof(unsigned char) * img_size);
	cudaMalloc(&gpu_hsl_img_h, sizeof(float) * img_size);
	cudaMalloc(&gpu_hsl_img_s, sizeof(float) * img_size);
	cudaMalloc(&gpu_hsl_img_l, sizeof(unsigned char) * img_size);
	
	cudaMemcpy(gpu_img_size, &img_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img_in_r, img_in.img_r, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img_in_g, img_in.img_g, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img_in_b, img_in.img_b, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

	int block_count = (int)ceil((float)img_size / MAXTHREADS);
    //hsl_med = rgb2hsl(img_in);

    //cudaEventRecord(start,0);
	gpu_rgb2hsl<<< block_count, MAXTHREADS >>>(gpu_img_in_r, gpu_img_in_g, gpu_img_in_b, gpu_hsl_img_h, gpu_hsl_img_s, gpu_hsl_img_l, img_size);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\tgpu_rgb2hsl time: %fms\n", elapsedTime);

    l_equ = (unsigned char *)malloc(img_size *sizeof(unsigned char));


	cudaMemcpy(temp_hsl_l, gpu_hsl_img_l, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);

	cudaMemset(gpu_hist, 0, sizeof(int) * 256);

    // cudaEventRecord(start,0);
	gpu_histogram<<< block_count, MAXTHREADS >>>(gpu_hist, gpu_hsl_img_l, gpu_img_size);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\tgpu_histogram time: %fms\n", elapsedTime);

	cudaMemcpy(t_hist, gpu_hist, sizeof(int) * nbr_size, cudaMemcpyDeviceToHost);

    gpu_histogram_equalization(l_equ, temp_hsl_l, t_hist, img_size, 256);
    
	cudaMemcpy(gpu_hsl_img_l, l_equ, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

    // cudaEventRecord(start,0);
	gpu_hsl2rgb<<< block_count, MAXTHREADS >>>(gpu_hsl_img_h, gpu_hsl_img_s, gpu_hsl_img_l, gpu_img_out_r, gpu_img_out_g, gpu_img_out_b, img_size);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\tgpu_hsl2rgb time: %fms\n", elapsedTime);

	result.img_r = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
	result.img_g = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
	result.img_b = (unsigned char *)malloc(sizeof(unsigned char) * img_size);

	result.w = img_in.w;
	result.h = img_in.h;

	cudaMemcpy(result.img_r, gpu_img_out_r, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(result.img_g, gpu_img_out_g, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(result.img_b, gpu_img_out_b, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);

	free(t_hist);
	free(temp_hsl_l);
	free(temp);
	free(l_equ);
	cudaFree(gpu_hist);
	cudaFree(gpu_img_size);
	cudaFree(gpu_img_in_r);
	cudaFree(gpu_img_in_g);
	cudaFree(gpu_img_in_b);
	cudaFree(gpu_img_out_r);
	cudaFree(gpu_img_out_g);
	cudaFree(gpu_img_out_b);
	cudaFree(gpu_hsl_img_h);
	cudaFree(gpu_hsl_img_s);
	cudaFree(gpu_hsl_img_l);

    //TODO remove
    // cudaEventDestroy(start);
    // cudaEventDestroy(end);

    return result;
}


PPM_IMG gpu_contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    //host vars
    PPM_IMG result;
    unsigned char * yuv_med_img_y;
    unsigned char * yuv_med_img_u;
    unsigned char * yuv_med_img_v;
    unsigned char * y_equ;
    int hist[256];
    int image_size = img_in.w * img_in.h;

    //TODO remove
    // cudaEvent_t start, end;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);

    //device vars
    unsigned char * gpu_result_img_r = 0;
    unsigned char * gpu_result_img_g = 0;
    unsigned char * gpu_result_img_b = 0;
    unsigned char * gpu_img_in_img_r = 0;
    unsigned char * gpu_img_in_img_g = 0;
    unsigned char * gpu_img_in_img_b = 0;
    unsigned char * gpu_yuv_med_img_y = 0;
    unsigned char * gpu_yuv_med_img_u = 0;
    unsigned char * gpu_yuv_med_img_v = 0;
    int * gpu_image_size;
    int * gpu_hist = 0;

    int block_count = (int)ceil((float)image_size / MAXTHREADS);

    //setup host vars
    yuv_med_img_y = (unsigned char *)malloc(sizeof(unsigned char)*image_size);
    yuv_med_img_u = (unsigned char *)malloc(sizeof(unsigned char)*image_size);
    yuv_med_img_v = (unsigned char *)malloc(sizeof(unsigned char)*image_size);

    //allocate variables for gpu_rgb2yuv
    //Pointers to device memory inside the structure still need to be allocated and freed individually.

    HANDLE_ERROR( cudaMalloc(&gpu_img_in_img_r, sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMemcpy(gpu_img_in_img_r, img_in.img_r, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice) );
    
    HANDLE_ERROR( cudaMalloc(&gpu_img_in_img_g, sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMemcpy(gpu_img_in_img_g, img_in.img_g, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMalloc(&gpu_img_in_img_b, sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMemcpy(gpu_img_in_img_b, img_in.img_b, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMalloc(&gpu_yuv_med_img_y, sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMalloc(&gpu_yuv_med_img_u, sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMalloc(&gpu_yuv_med_img_v, sizeof(unsigned char) * image_size) );


    HANDLE_ERROR( cudaMalloc(&gpu_image_size, sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(gpu_image_size, &(image_size), sizeof(int), cudaMemcpyHostToDevice) );


    //convert to yuv
    // cudaEventRecord(start,0);
    gpu_rgb2yuv<<< block_count, MAXTHREADS >>>(gpu_image_size, gpu_img_in_img_r, gpu_img_in_img_g, gpu_img_in_img_b,
                                                gpu_yuv_med_img_y, gpu_yuv_med_img_u, gpu_yuv_med_img_v);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\tgpu_rgb2yuv time: %fms\n", elapsedTime);

    //copy back to host
    HANDLE_ERROR( cudaMemcpy(yuv_med_img_u, gpu_yuv_med_img_u, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(yuv_med_img_v, gpu_yuv_med_img_v, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost) );
    //free used data
    HANDLE_ERROR( cudaFree(gpu_img_in_img_r) );
    HANDLE_ERROR( cudaFree(gpu_img_in_img_g) );
    HANDLE_ERROR( cudaFree(gpu_img_in_img_b) );

    y_equ = (unsigned char *)malloc(image_size*sizeof(unsigned char));

    //setup hist for gpu
    cudaMalloc(&gpu_hist, sizeof(int) * 256);
    cudaMemset(gpu_hist, 0, sizeof(int) * 256);
    
    gpu_histogram<<< block_count, MAXTHREADS >>>(gpu_hist, gpu_yuv_med_img_y, gpu_image_size);

    HANDLE_ERROR( cudaMemcpy(yuv_med_img_y, gpu_yuv_med_img_y, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(hist, gpu_hist, sizeof(int) * 256, cudaMemcpyDeviceToHost) );
    
    gpu_histogram_equalization(y_equ, yuv_med_img_y, hist, image_size, 256);

    free(yuv_med_img_y);
    yuv_med_img_y = (unsigned char *)malloc(sizeof(unsigned char)*image_size);
    yuv_med_img_y = y_equ;

    //start allocate for converting back to rgb
    HANDLE_ERROR( cudaMalloc(&(gpu_result_img_r), sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMalloc(&(gpu_result_img_g), sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMalloc(&(gpu_result_img_b), sizeof(unsigned char) * image_size) );
    HANDLE_ERROR( cudaMemcpy(gpu_yuv_med_img_y, yuv_med_img_y, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice) );
    
    //convert back to rgb
    // cudaEventRecord(start,0);
    gpu_yuv2rgb<<< block_count, MAXTHREADS >>>(gpu_image_size, gpu_yuv_med_img_y, gpu_yuv_med_img_u, gpu_yuv_med_img_v, 
                                                gpu_result_img_r, gpu_result_img_g, gpu_result_img_b);
    // cudaEventRecord(end,0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsedTime, start, end);
    // printf("\tgpu_yuv2rgb time: %fms\n", elapsedTime);

    result.img_r = (unsigned char *)malloc(sizeof(unsigned char)*image_size);
    result.img_g = (unsigned char *)malloc(sizeof(unsigned char)*image_size);
    result.img_b = (unsigned char *)malloc(sizeof(unsigned char)*image_size);

    //copy back to host
    HANDLE_ERROR( cudaMemcpy(result.img_r, gpu_result_img_r, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(result.img_g, gpu_result_img_g, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(result.img_b, gpu_result_img_b, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost) );

    result.w = img_in.w;
    result.h = img_in.h;

    free(yuv_med_img_y); //by freeing this you are freeing y_equ as well
    free(yuv_med_img_u);
    free(yuv_med_img_v);
    HANDLE_ERROR( cudaFree(gpu_result_img_r) );
    HANDLE_ERROR( cudaFree(gpu_result_img_g) );
    HANDLE_ERROR( cudaFree(gpu_result_img_b) );
    HANDLE_ERROR( cudaFree(gpu_yuv_med_img_y) );
    HANDLE_ERROR( cudaFree(gpu_yuv_med_img_u) );
    HANDLE_ERROR( cudaFree(gpu_yuv_med_img_v) );

    //TODO remove
    // cudaEventDestroy(start);
    // cudaEventDestroy(end);

    return result;
}


//Convert RGB to YUV, all components in [0, 255]
__global__ void gpu_rgb2yuv(int* image_size, unsigned char* img_in_r, unsigned char* img_in_g, unsigned char* img_in_b,
                            unsigned char* img_out_y, unsigned char* img_out_u, unsigned char* img_out_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char r, g, b;
    unsigned char y, cb, cr;

    if(i < *image_size) {
        r = img_in_r[i];
        g = img_in_g[i];
        b = img_in_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_out_y[i] = y;
        img_out_u[i] = cb;
        img_out_v[i] = cr;
    }
}


__device__ unsigned char gpu_clip_rgb(int x) {
	if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}


//Convert YUV to RGB, all components in [0, 255]
__global__ void gpu_yuv2rgb(int* image_size, unsigned char* img_in_y, unsigned char* img_in_u, unsigned char* img_in_v,
                            unsigned char* img_out_r, unsigned char* img_out_g, unsigned char* img_out_b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int  rt,gt,bt;
    int y, cb, cr;

    if(i < *image_size) {
        y  = (int)img_in_y[i];
        cb = (int)img_in_u[i] - 128;
        cr = (int)img_in_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

		img_out_r[i] = gpu_clip_rgb(rt);
		img_out_g[i] = gpu_clip_rgb(gt);
		img_out_b[i] = gpu_clip_rgb(bt);
    }
}


__device__ float gpu_Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}


__global__ void gpu_rgb2hsl(unsigned char * img_in_r, unsigned char * img_in_g, unsigned char * img_in_b,
							float * img_out_h, float * img_out_s, unsigned char * img_out_l, int img_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float H, S, L;

	if (i < img_size) {
        
        float var_r = ( (float)img_in_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)img_in_g[i]/255 );
        float var_b = ( (float)img_in_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        img_out_h[i] = H;
        img_out_s[i] = S;
        img_out_l[i] = (unsigned char)(L*255);
    }
    
}


__global__ void gpu_hsl2rgb(float * img_in_h, float * img_in_s, unsigned char * img_in_l, 
							unsigned char * img_out_r, unsigned char * img_out_g, unsigned char * img_out_b, int img_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < img_size) {
        float H = img_in_h[i];
        float S = img_in_s[i];
        float L = img_in_l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * gpu_Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * gpu_Hue_2_RGB( var_1, var_2, H );
            b = 255 * gpu_Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        img_out_r[i] = r;
        img_out_g[i] = g;
        img_out_b[i] = b;
    }
}