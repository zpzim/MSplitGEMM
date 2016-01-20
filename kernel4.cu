#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <pthread.h>
#include <unistd.h>

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



float * doMultiply2Matrices(
        int a1Rows, int a1Cols,  float * A1,
        int a2Rows, int a2Cols,  float * A2,
	float* C, cudaStream_t cudaStream, cublasHandle_t handle)
{

    float alpha = 1.0;
    float beta =  0.0;

    cublasSetStream(handle, cudaStream) ;

    cublasStatus_t stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                  a2Cols, a1Rows, a1Cols,
                  &alpha,
                  A2, a2Cols,
                  A1, a1Cols,
                  &beta,
                  C, a2Cols );
    printf("cublas status = %d\n", stat);

    return C ;


}

void PrintMatrix(char name[], int rows, int cols, const float* m){
  printf("%s\n", name);
  for(int row = 0; row < rows; ++row){
	for(int col = 0; col < cols; ++col){
		printf("%f ", m[row * cols + col]);
	}
	printf("\n");
  }
}


void copyElements(float* out, float* entry, unsigned long long eRows, unsigned long long eCols, unsigned long long oRows, unsigned long long oCols, unsigned long long x, unsigned long long y){
	for(unsigned long long i = 0; i < eRows; ++i){
		for(unsigned long long j = 0; j < eCols; ++j){
			out[x*eRows*oCols + (i*oCols) + (y*eCols + j)] = entry[i*eCols + j];
		}

	}



}


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
	for(unsigned long long i = 0; i < numSubMatrixB; ++i){
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
		while(y < numSubMatrixA){
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
			doMultiply2Matrices(subRows, k, a[y % numStreams], k, subCols, b, c[y % numStreams], streams[y % numStreams], handles[y % numStreams]); 	
			cudaMemcpyAsync(c_h[y % numStreams], c[y % numStreams], sizeof(float)*subRows*subCols, cudaMemcpyDeviceToHost, streams[y % numStreams]);
			if(y % numStreams == numStreams - 1){
				cudaDeviceSynchronize();
				for(int s = 0; s < numStreams; ++s){
					//TODO:Currently does not work with overflowA != 0 or overflowB != 0
					copyElements(C, c_h[s], subRows, subCols, m, n, count * numStreams + s, i);
				}
				++count;
			}
			++y;

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





