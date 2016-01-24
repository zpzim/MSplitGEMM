#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "common.h"

const int num_submatrix = 8;

void copyElements(float* out, float* entry, unsigned long long eRows, unsigned long long eCols, unsigned long long oRows, unsigned long long oCols, unsigned long long x, unsigned long long y,
	unsigned long long ofA, unsigned long long ofB){
	unsigned long long counterRows = eRows;
	unsigned long long counterCols = eCols;
	if(ofA){
		counterRows = ofA;
	}
	if(ofB){
		counterCols = ofB;	
	}
	for(unsigned long long i = 0; i < counterRows; ++i){
		for(unsigned long long j = 0; j < counterCols; ++j){
			out[x*eRows*oCols + (i*oCols) + (y*eCols + j)] = entry[i*eCols + j];
		}

	}



}




void msplitm(char transa, char transb, unsigned long long m, unsigned long long n, unsigned long long k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
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
	float** B_split = (float**)malloc(sizeof(float*) * (numSubMatrixB + 1));
	for(int i = 0; i < numSubMatrixB + 1; ++i){
		float* temp = (float*) malloc( sizeof(float)*subCols * k );
		for(int j = 0; j < k; ++j){
			for(int x = 0; x < subCols; ++x){
				if(i * subCols + x < n){
					temp[j * subCols + x] = B[j * n + (i*subCols + x)];
				}else{
					temp[j *subCols + x] = 0;
				}
			}
		}
		cudaMalloc((void**) &B_split[i], sizeof(float) * subCols * k);
		cudaMemcpy(B_split[i], temp, sizeof(float)*subCols*k, cudaMemcpyHostToDevice);
		free(temp);
	}
	for(unsigned long long i = 0; i < numSubMatrixA + 1; ++i){
		if(overflowA == 0 && i == numSubMatrixA){
			break;
		}
		float* temp = (float*) malloc( sizeof(float)*subRows * k );
		for(int j = 0; j < subRows; ++j){
			for(int x = 0; x < k; ++x){
				if(i * subRows + j < m){
					temp[j * k + x] = A[i*subRows*k + j*k + x];
				}else{
					temp[j * k + x] = 0;
				}
			}			
		}
		float* temp2 = 0;
		float* temp3 = 0;
		cudaMalloc((void**) &temp2, sizeof(float) * subRows * k);
		cudaMalloc((void**) &temp3, sizeof(float) * subCols * subRows);
		cudaMemcpy(temp2, temp, sizeof(float)*subRows*k, cudaMemcpyHostToDevice);
		free(temp);

		printf("Running multiply for row group %d\n", i);
		temp = (float*)malloc(sizeof(float)*subRows*subCols);
		for(int x = 0; x < numSubMatrixB + 1; ++x){
				if(overflowB == 0 && x == numSubMatrixB){
					break;
				}
				doMultiply2Matrices(subRows, k, temp2, k, subCols, B_split[x], temp3, alpha);
				
				cudaMemcpy(temp, temp3, sizeof(float)*subRows*subCols,cudaMemcpyDeviceToHost);

			if(x == numSubMatrixB && i == numSubMatrixA){
				copyElements(C, temp, subRows, subCols, m, n, i, x, overflowA, overflowB, beta);
			}else if(x == numSubMatrixB){
				copyElements(C, temp, subRows, subCols, m, n, i, x, 0, overflowB, beta);
			}else if(i == numSubMatrixA){
				copyElements(C, temp, subRows, subCols, m, n, i, x, overflowA, 0, beta);
			}else{
				copyElements(C, temp, subRows, subCols, m, n, i, x, 0, 0, beta);
			}
		}
		
		cudaFree(temp2);
		cudaFree(temp3);
		
	}
}



