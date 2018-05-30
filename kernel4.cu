#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <pthread.h>
#include <unistd.h>

#include "common.h"

const int num_submatrix = 2;
const int numStreams = 2;
const int num_threads = numStreams;


struct thread_args{
    int threadId;
    unsigned long long overflowA;
    unsigned long long numSubMatrixA;
    unsigned long long subRows;
    unsigned long long subCols;
    unsigned long long m;
    unsigned long long n;
    unsigned long long k;
    unsigned long long y;
    unsigned long long i;
    float *C;
    float *A;

};

volatile int running_threads = 0;
pthread_mutex_t running_mutex = PTHREAD_MUTEX_INITIALIZER;


cudaStream_t streams[numStreams];
float* b[num_threads];
float* a[num_threads];
float* c[num_threads];
float* a_h[num_threads];
float* c_h[num_threads];
struct thread_args targs[num_threads];
pthread_t threads[num_threads];
char threads_active[num_threads];
cublasHandle_t handles[num_threads];


void msplitm(char transa, char transb, unsigned long long m, unsigned long long n, unsigned long long k, float alpha, float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    
    printf("entering msplitm \n");
    float* A_d;
    float* B_d;
    float* C_d;
    unsigned long long A_sz = m * k;
    unsigned long long B_sz = n * k;
    unsigned long long C_sz = m * n;
    unsigned long long MAX =  (unsigned long long )m* (unsigned long long) n / num_submatrix;
    
    MAX -= MAX % k;
    printf("MAX: %llu\n", MAX);
    printf("B_sz: %llu\n",B_sz);
    unsigned long long numSubMatrixB = B_sz / MAX;
    printf("SubmatriciesB: %llu\n", numSubMatrixB);
    unsigned long long SMB_sz = B_sz / numSubMatrixB;
    printf("SMB_sz: %llu\n", SMB_sz);
    unsigned long long subCols = B_sz / (numSubMatrixB * k);
    printf("subCols: %llu\n", subCols);
    
    unsigned long long numSubMatrixA = A_sz / MAX;
    unsigned long long SMA_sz = A_sz / numSubMatrixA;
    unsigned long long subRows = A_sz / (numSubMatrixA * k);
    printf("subrows: %llu\n", subRows);
    printf("SMA_sz: %llu\n", SMA_sz);
    printf("submatriciesA: %llu\n", numSubMatrixA);
    unsigned long long overflowA = m % subRows;
    unsigned long long overflowB = n % subCols;
    printf("overflowB: %llu\n", overflowB);
    printf("overflowA: %llu\n", overflowA);
    for(int i = 0; i < numStreams; ++i){
        cudaSetDevice(i);
        cublasCreate(&handles[i]);
        cudaStreamCreate(&streams[i]);
        cudaMalloc((void**) &b[i], sizeof(float) * subCols * k);
        cudaMalloc((void**) &a[i], sizeof(float) * subRows * k);
        cudaMalloc((void**) &c[i], sizeof(float) * subCols * subRows);
        cudaMallocHost((void**) &a_h[i], sizeof(float) * subRows * k);
        cudaMallocHost((void**) &c_h[i], sizeof(float) * subCols * subRows);
        threads_active[i] = 0;
    }

    
    for(unsigned long long i = 0; i < numSubMatrixB + 1; ++i){
        int count = 0;
        if(overflowB == 0 && i == numSubMatrixB){
            break;
        }
         
        int copynumB = i == numSubMatrixB ? overflowB : subCols;
        for(int j = 0; j < numStreams; ++j) {
            cudaSetDevice(j);    
            if(i == numSubMatrixB) {
                cudaMemsetAsync(a, 0, sizeof(float) * k * subCols, streams[j]);
            }    
            cudaMemcpy2DAsync(b[j], subCols * sizeof(float), B + (i * subCols), n * sizeof(float), 
                         copynumB*sizeof(float), k, cudaMemcpyHostToDevice, streams[j] );
        }    
        unsigned long long y = 0;
        int streamsActive = 0;
        while(y < numSubMatrixA + 1){
            if(overflowA == 0 && y == numSubMatrixA){
                break;
            }
            int copynumA = y == numSubMatrixA ? overflowA : subRows;
            cudaSetDevice(y % numStreams);
            if(y == numSubMatrixA) {
                cudaMemsetAsync(a, 0, sizeof(float) * k * subRows, streams[y % numStreams]);
            }    
            cudaMemcpy2DAsync(a[y % numStreams], k * sizeof(float), A + (k*y*subRows), k * sizeof(float), 
                         k * sizeof(float), copynumA, cudaMemcpyHostToDevice, streams[y % numStreams] );
            
            printf("sending multiply %llu,%llu to stream %d\n", y, i, y % numStreams);
            doMultiply2MatricesStreaming(subRows, k, a[y % numStreams], k, subCols, b[y % numStreams], c[y % numStreams], streams[y % numStreams], handles[y % numStreams], alpha);     
            cudaMemcpyAsync(c_h[y % numStreams], c[y % numStreams], sizeof(float)*subRows*subCols, cudaMemcpyDeviceToHost, streams[y % numStreams]);
                        
            streamsActive++;
            if(y % numStreams == numStreams - 1){
                for(int s = 0; s < numStreams; ++s){
                    cudaStreamSynchronize(streams[s]);
                    int currWork = count * numStreams + s;
                    // TODO: We can probably do a direct copy from the device to the appropriate output location on the host
                    // But we need to handle the beta term on the GPU
                    if(i == numSubMatrixB && currWork == numSubMatrixA){
                        copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
                    }else if(i == numSubMatrixB){
                        copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
                    }else if(currWork == numSubMatrixA){
                        copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
                    }else{
                        copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
                    }
                    streamsActive--;
                }
                ++count;
            }
            ++y;

        }
        
        for(int s = 0; s < streamsActive; ++s){
            cudaStreamSynchronize(streams[s]);
            int currWork = count * numStreams + s;
            if(i == numSubMatrixB && currWork == numSubMatrixA){
                copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
            }else if(i == numSubMatrixB){
                copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
            }else if(currWork == numSubMatrixA){
                copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
            }else{
                copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
            }

        }
        
        
        
    
    }

    for(int i = 0; i < numStreams; ++i){
        cudaSetDevice(i);    
        cudaFree(a[i]);
        cudaFree(c[i]);
        cudaFreeHost(a_h[i]);
        cudaFreeHost(c_h[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(b);
    
}





