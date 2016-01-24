#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <pthread.h>
#include <unistd.h>

#include "common.h"

const int num_submatrix = 16;
const int numStreams = 2;
const int num_threads = numStreams;


struct thread_args{
	int threadId;
	unsigned long long overflowA;
	unsigned long long overflowB;
	unsigned long long numSubMatrixB;
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

float ALPHA = 0;
float BETA = 0;
volatile int running_threads = 0;
pthread_mutex_t running_mutex = PTHREAD_MUTEX_INITIALIZER;


float* b = 0;
float* a[num_threads];
float* c[num_threads];
float* a_h[num_threads];
float* c_h[num_threads];
struct thread_args targs[num_threads];
pthread_t threads[num_threads];
char threads_active[num_threads];
cublasHandle_t handles[numStreams];


void* mult(void * threadArg){
	
	struct thread_args* data = (struct thread_args*) threadArg;
	int threadId = data -> threadId;
	unsigned long long overflowA = data -> overflowA;
	unsigned long long overflowB = data -> overflowB;
	unsigned long long numSubMatrixB = data -> numSubMatrixB;
	unsigned long long numSubMatrixA = data -> numSubMatrixA;
	unsigned long long subRows = data->subRows;
	unsigned long long subCols = data->subCols;
	unsigned long long m = data->m;
	unsigned long long n = data->n;
	unsigned long long k = data->k;
	unsigned long long y = data->y;
	unsigned long long i = data->i;
	float *C = data->C;
	float *A = data->A;
	if(overflowA == 0 && y == numSubMatrixA){
		pthread_exit(0);
	}
	for(int j = 0; j < subRows; ++j){
		for(int x = 0; x < k; ++x){
			//printf("(t,j,x) = (%d,%d,%d)\n",y,j,x);
			if(y * subRows + j < m){
				(a_h[threadId])[j * k + x] = A[y*subRows*k + j*k + x];
			}else{
				(a_h[threadId])[j * k + x] = 0;
			}
		}			
	}
	cudaMemcpyAsync(a[threadId], a_h[threadId], sizeof(float)*subRows*k, cudaMemcpyHostToDevice);
	doMultiply2MatricesStreaming(subRows, k, a[threadId], k, subCols, b, c[threadId], 0, handles[threadId], ALPHA); 	
	cudaMemcpyAsync(c_h[threadId], c[threadId], sizeof(float)*subRows*subCols, cudaMemcpyDeviceToHost);
	cudaStreamSynchronize(cudaStreamPerThread);
	if(i == numSubMatrixB && y == numSubMatrixA){
		copyElements(C,  c_h[threadId], subRows, subCols, m, n, y, i, overflowA, overflowB, BETA);
	}else if(i == numSubMatrixB){
		copyElements(C,  c_h[threadId], subRows, subCols, m, n, y, i, 0, overflowB, BETA);
	}else if(y == numSubMatrixA){
		copyElements(C,  c_h[threadId], subRows, subCols, m, n, y, i, overflowA, 0, BETA);
	}else{
		copyElements(C, c_h[threadId], subRows, subCols, m, n, y, i, 0, 0, BETA);
	}
	pthread_mutex_lock(&running_mutex);
   	running_threads--;
	threads_active[threadId] = 0;
   	pthread_mutex_unlock(&running_mutex);
	pthread_exit(0);
	


}


void msplitm(char transa, char transb, unsigned long long m, unsigned long long n, unsigned long long k, float alpha, float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    ALPHA = alpha;
    BETA = beta;
    printf("entering msplitm \n");
    float* A_d;
    float* B_d;
    float* C_d;
    unsigned long long A_sz = m * k;
    unsigned long long B_sz = n * k;
    unsigned long long C_sz = m * n;
    unsigned long long MAX =  (unsigned long long )m* (unsigned long long) n / num_submatrix;
    
    const unsigned int BLOCK_SIZE = 16;

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
		cudaMalloc((void**) &a[i], sizeof(float) * subRows * k);
		cudaMalloc((void**) &c[i], sizeof(float) * subCols * subRows);
		cudaMallocHost((void**) &a_h[i], sizeof(float) * subRows * k);
		cudaMallocHost((void**) &c_h[i], sizeof(float) * subCols * subRows);
		threads_active[i] = 0;
	}

	float* temp3 = 0;
	cudaMallocHost((void**) &temp3, sizeof(float)*subCols * k );
	for(unsigned long long i = 0; i < numSubMatrixB + 1; ++i){

		if(overflowB == 0 && i == numSubMatrixB){
			continue;
		}
	
		for(int j = 0; j < k; ++j){//subCols; ++j){
			for(int x = 0; x < subCols; ++x){
				if(i * subCols + x < n){
					temp3[j * subCols + x] = B[j * n + (i*subCols + x)];
				}else{
					temp3[j *subCols + x] = 0;
				}
			}
		}
	
		cudaMemcpyAsync(b, temp3, sizeof(float)*subCols*k, cudaMemcpyHostToDevice);
		unsigned long long y = 0;
		while(y < numSubMatrixA + 1){
			if(overflowA == 0 && y == numSubMatrixA){
				continue;
			}
			while(running_threads >= num_threads){
				//spinlock
			}
			int tid = 0;
			while(threads_active[tid]){
				tid++;
			}
			targs[tid].threadId = tid;
			targs[tid].y = y;
			targs[tid].numSubMatrixA = numSubMatrixA;
			targs[tid].numSubMatrixB = numSubMatrixB;
			targs[tid].overflowB = overflowB;
			targs[tid].overflowA = overflowA;
			targs[tid].C = C;
			targs[tid].A = A;
			targs[tid].subRows = subRows;
			targs[tid].subCols = subCols;
			targs[tid].m = m;
			targs[tid].n = n;
			targs[tid].k = k;
			targs[tid].i = i;
			printf("creating thread %d to multiply submatrix %d,%d\n",tid,y,i);
			
			 pthread_mutex_lock(&running_mutex);
			 running_threads++;
			 threads_active[tid] = 1;
			 pthread_mutex_unlock(&running_mutex);
			int rc = pthread_create(&threads[tid], NULL, mult, (void*) &targs[tid]);
			++y;
		}
		int ret = 0;
		for(int x = 0; x < num_threads; x++)
  			pthread_join(threads[x], NULL);
	
	
	}
	for(int i = 0; i < numStreams; ++i){
		cublasDestroy(handles[i]);
		cudaFree(a[i]);
		cudaFree(c[i]);
		cudaFreeHost(a_h[i]);
		cudaFreeHost(c_h[i]);
	}
	cudaFree(b);
	cudaFreeHost(temp3);
    
}





