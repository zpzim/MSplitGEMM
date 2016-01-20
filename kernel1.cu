#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

const int num_submatrix = 8;

float * doMultiply2Matrices(
        int a1Rows, int a1Cols,  float * A1,
        int a2Rows, int a2Cols,  float * A2,
	float* C, float alpha, float beta)
{
    cublasHandle_t  handle;

    cublasCreate (&handle ) ;

    cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                  a2Cols, a1Rows, a1Cols,
                  &alpha,
                  A2, a2Cols,
                  A1, a1Cols,
                  &beta,
                  C, a2Cols );

    cublasDestroy ( handle ) ;

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
	for(unsigned long long i = 0; i < numSubMatrixB + 1; ++i){
		if(overflowB == 0 && i == numSubMatrixB){
			break;
		}
		float* b = 0;
		float* temp3 = (float*) malloc( sizeof(float)*subCols * k );
		for(int j = 0; j < k; ++j){
			for(int x = 0; x < subCols; ++x){
				if(i * subCols + x < n){
					temp3[j * subCols + x] = B[j * n + (i*subCols + x)];
				}else{
					temp3[j *subCols + x] = 0;
				}
			}
		}
		cudaMalloc((void**) &b, sizeof(float) * subCols * k);
		cudaMemcpy(b, temp3, sizeof(float)*subCols*k, cudaMemcpyHostToDevice);
		free(temp3);
		for(unsigned long long y = 0; y < numSubMatrixA + 1; ++y){
			if(overflowA == 0 && y == numSubMatrixA){
				break;
			}
			float * temp = (float*) malloc( sizeof(float)*subRows * k );
			for(int j = 0; j < subRows; ++j){
				for(int x = 0; x < k; ++x){
					if(y * subRows + j < m){
						temp[j * k + x] = A[y*subRows*k + j*k + x];
					}else{
						temp[j * k + x] = 0;
					}
				}			
			}
			float* a = 0;
			float* c = 0;
			cudaMalloc((void**) &a, sizeof(float) * subRows * k);
			cudaMalloc((void**) &c, sizeof(float) * subCols * subRows);
			cudaMemcpy(a, temp, sizeof(float)*subRows*k, cudaMemcpyHostToDevice);
			doMultiply2Matrices(subRows, k, a, k, subCols, b, c, alpha, beta); 			
			cudaMemcpy(temp, c, sizeof(float)*subRows*subCols, cudaMemcpyDeviceToHost);
			if(i == numSubMatrixB && y == numSubMatrixA){
				copyElements(C, temp, subRows, subCols, m, n, y, i, overflowA, overflowB);
			}else if(i == numSubMatrixB){
				copyElements(C, temp, subRows, subCols, m, n, y, i, 0, overflowB);
			}else if(y == numSubMatrixA){
				copyElements(C, temp, subRows, subCols, m, n, y, i, overflowA, 0);
			}else{
				copyElements(C, temp, subRows, subCols, m, n, y, i, 0, 0);
			}
			free(temp);
			cudaFree(a);
			cudaFree(c);
		
		}
		
		cudaFree(b);
	}
}





