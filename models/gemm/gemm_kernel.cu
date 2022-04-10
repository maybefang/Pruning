// CUDA runtime 库 + CUBLAS 库 
#include "cuda_runtime.h"
#include "cublas_v2.h"
 
#include <time.h>
#include <iostream> 
 

//void calculate_kernel(float* d_input, float* d_weight, float* d_output, float* bias, int64_t size, int64_t M, int64_t N, int64_t K) 
void calculate_dataingpu(float* d_input, float* d_weight, float* d_output, int64_t size, int64_t M, int64_t N, int64_t K) 
{   
    // 定义状态变量
    cublasStatus_t status;
    
    /*
    ** GPU 计算矩阵相乘
    */
    //std::cout<<"m:"<<M<<std::endl;
    //for (int i=0;i<6;i++){std::cout<<*(h_A+i)<<std::endl;}
    //std::cout<<"n:"<<N<<std::endl;
    //for (int i=0;i<6;i++){std::cout<<*(h_B+i)<<std::endl;}
    //std::cout<<"k:"<<K<<std::endl;
    //for (int i=0;i<4;i++){std::cout<<*(bias+i)<<std::endl;}

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle;
    status = cublasCreate_v2(&handle);
    
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS 对象实例化出错" << std::endl;
        }
        getchar ();
        //return EXIT_FAILURE;
        
        return;
    }
 
    //float *d_output;
    // 在 显存 中为将要存放运算结果的矩阵开辟空间
    //cudaMalloc (
    //    (void**)&d_output,        // 指向开辟的空间的指针
    //    M*N * sizeof(float)     //　需要开辟空间的字节数
    //);

    //std::cout<<"在gpu中为abc申请空间"<<std::endl;

    //std::cout<<"向gpu中写入ab"<<std::endl;
    // 同步函数
    cudaDeviceSynchronize();
 
    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    float a=1; float b=0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
    cublasSgemm (
        handle,    // blas 库对象 
        CUBLAS_OP_T,    // 矩阵 A 属性参数
        CUBLAS_OP_T,    // 矩阵 B 属性参数
        M,    // A, C 的行数 
        N,    // B, C 的列数
        K,    // A 的列数和 B 的行数
        &a,    // 运算式的 α 值
        d_input,    // A 在显存中的地址
        K,    // lda，因为是列优先，所以此处传入每列多少元素
        d_weight,    // B 在显存中的地址
        N,    // ldb，同lda
        &b,    // 运算式的 β 值
        d_output,    // C 在显存中的地址(结果矩阵)
        M    // ldc
    );
    
    // 同步函数
    cudaDeviceSynchronize();

    //std::cout<<"计算c"<<std::endl;
    // 从 显存 中取出运算结果至 内存中去
    /*
    cublasGetVector (
        M*N,    //  要取出元素的个数
        sizeof(float),    // 每个元素大小
        d_output,    // GPU 端起始地址
        1,    // 连续元素之间的存储间隔
        h_output,    // 主机端起始地址
        1    // 连续元素之间的存储间隔
    );
    */
    //std::cout<<*h_C<<std::endl;
    
    //for(int i=0;i<4;i++){
    //    std::cout<<*(h_C+i)<<" ";
    //}
    //std::cout<<"从gpu中取出c到cpu"<<std::endl;
    // 清理掉使用过的内存
    //free (h_A);
    //free (h_B);
    //free (h_C);
    //cudaFree (d_A);
    //cudaFree (d_B);
    //cudaFree (d_output);

    //std::cout<<"在gpu、cpu中释放abc空间"<<std::endl;
    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
    //std::cout<<"最后释放库对象"<<std::endl;
}

void calculate_dataincpu(float* h_input, float* h_weight, float* h_output, int64_t size, int64_t M, int64_t N, int64_t K) 
{   
    // 定义状态变量
    cublasStatus_t status;
    
    /*
    ** GPU 计算矩阵相乘
    */
    //std::cout<<"m:"<<M<<std::endl;
    //for (int i=0;i<6;i++){std::cout<<*(h_A+i)<<std::endl;}
    //std::cout<<"n:"<<N<<std::endl;
    //for (int i=0;i<6;i++){std::cout<<*(h_B+i)<<std::endl;}
    //std::cout<<"k:"<<K<<std::endl;
    //for (int i=0;i<4;i++){std::cout<<*(bias+i)<<std::endl;}

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle;
    status = cublasCreate_v2(&handle);
    
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS 对象实例化出错" << std::endl;
        }
        getchar ();
        //return EXIT_FAILURE;
        
        return;
    }
 
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

    //std::cout<<"在gpu中为abc申请空间"<<std::endl;

    //std::cout<<"向gpu中写入ab"<<std::endl;
    // 同步函数
    cudaDeviceSynchronize();
 
    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    float a=1; float b=0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
    cublasSgemm (
        handle,    // blas 库对象 
        CUBLAS_OP_T,    // 矩阵 A 属性参数
        CUBLAS_OP_T,    // 矩阵 B 属性参数
        M,    // A, C 的行数 
        N,    // B, C 的列数
        K,    // A 的列数和 B 的行数
        &a,    // 运算式的 α 值
        d_input,    // A 在显存中的地址
        K,    // lda，因为是列优先，所以此处传入每列多少元素
        d_weight,    // B 在显存中的地址
        N,    // ldb，同lda
        &b,    // 运算式的 β 值
        d_output,    // C 在显存中的地址(结果矩阵)
        M    // ldc
    );
    
    // 同步函数
    cudaDeviceSynchronize();

    //std::cout<<"计算c"<<std::endl;
    // 从 显存 中取出运算结果至 内存中去
    
    cublasGetVector (
        M*N,    //  要取出元素的个数
        sizeof(float),    // 每个元素大小
        d_output,    // GPU 端起始地址
        1,    // 连续元素之间的存储间隔
        h_output,    // 主机端起始地址
        1    // 连续元素之间的存储间隔
    );
    
    //std::cout<<*h_C<<std::endl;
    
    //for(int i=0;i<4;i++){
    //    std::cout<<*(h_C+i)<<" ";
    //}
    //std::cout<<"从gpu中取出c到cpu"<<std::endl;
    // 清理掉使用过的内存
    //free (h_A);
    //free (h_B);
    //free (h_C);
    cudaFree (d_input);
    cudaFree (d_weight);
    cudaFree (d_output);

    //std::cout<<"在gpu、cpu中释放abc空间"<<std::endl;
    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
    //std::cout<<"最后释放库对象"<<std::endl;
}
