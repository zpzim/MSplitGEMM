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
float* b = 0;
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
	printf("MAX: %d\n", MAX);
	printf("B_sz: %d\n",B_sz);
	unsigned long long numSubMatrixB = B_sz / MAX;
	printf("SubmatriciesB: %d\n", numSubMatrixB);
	unsigned long long SMB_sz = B_sz / numSubMatrixB;
	printf("SMB_sz: %d\n", SMB_sz);
	unsigned long long subCols = B_sz / (numSubMatrixB * k);
	printf("subCols: %d\n", subCols);
	
	unsigned long long numSubMatrixA = A_sz / MAX;
	unsigned long long SMA_sz = A_sz / numSubMatrixA;
	unsigned long long subRows = A_sz / (numSubMatrixA * k);
	printf("subrows: %d\n", subRows);
	printf("SMA_sz: %d\n", SMA_sz);
	printf("submatriciesA: %d\n", numSubMatrixA);
	unsigned long long overflowA = m % subRows;
	unsigned long long overflowB = n % subCols;
	printf("overflowB: %d\n", overflowB);
	printf("overflowA: %d\n", overflowA);
	cudaMalloc((void**) &b, sizeof(float) * subCols * k);
	for(int i = 0; i < numStreams; ++i){
		cublasCreate(&handles[i]);
		cudaStreamCreate(&streams[i]);
		cudaMalloc((void**) &a[i], sizeof(float) * subRows * k);
		cudaMalloc((void**) &c[i], sizeof(float) * subCols * subRows);
		cudaMallocHost((void**) &a_h[i], sizeof(float) * subRows * k);
		cudaMallocHost((void**) &c_h[i], sizeof(float) * subCols * subRows);
		threads_active[i] = 0;
	}

	float* temp3 = 0;
	
	cudaMallocHost((void**) &temp3, sizeof(float)*subCols * k );
	for(unsigned long long i = 0; i < numSubMatrixB + 1; ++i){
		int count = 0;
		if(overflowB == 0 && i == numSubMatrixB){
			break;
		}
	
		for(int j = 0; j < k; ++j){
			for(int x = 0; x < subCols; ++x){
				if(i * subCols + x < n){
					temp3[j * subCols + x] = B[j * n + (i*subCols + x)];
				}else{
					temp3[j *subCols + x] = 0;
				}
			}
		}
	
		cudaMemcpyAsync(b, temp3, sizeof(float)*subCols*k, cudaMemcpyHostToDevice, streams[0]);
		unsigned long long y = 0;
		int streamsActive = 0;
		while(y < numSubMatrixA + 1){
			if(overflowA == 0 && y == numSubMatrixA){
				break;
			}
			for(int j = 0; j < subRows; ++j){
				for(int x = 0; x < k; ++x){
					if(y * subRows + j < m){
						(a_h[y % numStreams])[j * k + x] = A[y*subRows*k + j*k + x];
					}else{
						(a_h[y % numStreams])[j * k + x] = 0;
					}
				}			
			}
			
			cudaMemcpyAsync(a[y % numStreams], a_h[y % numStreams], sizeof(float)*subRows*k, cudaMemcpyHostToDevice, streams[y % numStreams]);
			printf("sending multiply %d,%d to stream %d\n", y, i, y % numStreams);
			doMultiply2MatricesStreaming(subRows, k, a[y % numStreams], k, subCols, b, c[y % numStreams], streams[y % numStreams], handles[y % numStreams], alpha); 	
			cudaMemcpyAsync(c_h[y % numStreams], c[y % numStreams], sizeof(float)*subRows*subCols, cudaMemcpyDeviceToHost, streams[y % numStreams]);
						
			streamsActive++;
			if(y % numStreams == numStreams - 1){
				for(int s = 0; s < numStreams; ++s){
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
					streamsActive--;
				}
				++count;
			}
			++y;

		}
		PrintMatrix("C", m, n, C);
		printf("%d Streams Active Left over\n", streamsActive);
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
		cudaFree(a[i]);
		cudaFree(c[i]);
		cudaFreeHost(a_h[i]);
		cudaFreeHost(c_h[i]);
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(b);
	cudaFreeHost(temp3);
    
}





