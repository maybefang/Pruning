// CUDA runtime 库 + CUBLAS 库 
#include "cuda_runtime.h"
#include "cublas_v2.h"
 
#include <time.h>
#include <iostream>
#include<torch/extension.h>
#include <thrust/device_vector.h>

#include <typeinfo>

#include "cuda.h" 
 

//void calculate_kernel(float* d_input, float* d_weight, float* d_output, float* bias, int64_t size, int64_t M, int64_t N, int64_t K) 
void calculate_dataingpu(float* d_input, float* d_weight, float* d_output, int64_t M, int64_t N, int64_t K) 
{   
    // 定义状态变量
    cublasStatus_t status;
    

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle=blas_handle();
 
    // 同步函数
    //cudaDeviceSynchronize();

    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    float a=1; float b=0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
    cublasSgemm (
        handle,    // blas 库对象 
        CUBLAS_OP_N,    // 矩阵 A 属性参数
        CUBLAS_OP_N,    // 矩阵 B 属性参数
        N,    // A, C 的行数 
        M,    // B, C 的列数
        K,    // A 的列数和 B 的行数
        &a,    // 运算式的 α 值
        d_weight,    // A 在显存中的地址
        N,    // lda，因为是列优先，所以此处传入每列多少元素
        d_input,    // B 在显存中的地址
        K,    // ldb，同lda
        &b,    // 运算式的 β 值
        d_output,    // C 在显存中的地址(结果矩阵)
        N    // ldc
    );
    // 同步函数
    //cudaDeviceSynchronize();
}

void calculate_dataincpu(float* h_input, float* h_weight, float* h_output, int64_t M, int64_t N, int64_t K) 
{   
    // 定义状态变量
    cublasStatus_t status;
    

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle=blas_handle();
 
    float *d_input, *d_weight;
    cudaMalloc (
        (void**)&d_input,   
        K*M * sizeof(float)  
     );
    cudaMalloc (
        (void**)&d_weight,
        N*K * sizeof(float)
    );


    float *d_output;
    //在 显存 中为将要存放运算结果的矩阵开辟空间
    cudaMalloc (
       (void**)&d_output,        // 指向开辟的空间的指针
       M*N * sizeof(float)     //　需要开辟空间的字节数
    );

    
    // 同步函数
    //cudaDeviceSynchronize();
 
    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    float a=1; float b=0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
    cublasSgemm (
        handle,    // blas 库对象 
        CUBLAS_OP_N,    // 矩阵 A 属性参数
        CUBLAS_OP_N,    // 矩阵 B 属性参数
        N,    // A, C 的行数 
        M,    // B, C 的列数
        K,    // A 的列数和 B 的行数
        &a,    // 运算式的 α 值
        d_weight,    // A 在显存中的地址
        N,    // lda，因为是列优先，所以此处传入每列多少元素
        d_input,    // B 在显存中的地址
        K,    // ldb，同lda
        &b,    // 运算式的 β 值
        d_output,    // C 在显存中的地址(结果矩阵)
        N    // ldc
    );
    
    // 同步函数
    //cudaDeviceSynchronize();

    
    // 从 显存 中取出运算结果至 内存中去
    
    cublasGetVector (
        M*N,    //  要取出元素的个数
        sizeof(float),    // 每个元素大小
        d_output,    // GPU 端起始地址
        1,    // 连续元素之间的存储间隔
        h_output,    // 主机端起始地址
        1    // 连续元素之间的存储间隔
    );
    
    cudaFree (d_input);
    cudaFree (d_weight);
    cudaFree (d_output);

}
