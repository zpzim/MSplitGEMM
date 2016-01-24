#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "common.h"

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
	unsigned long long ofA, unsigned long long ofB, float beta){
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
			unsigned long long index = x*eRows*oCols + (i*oCols) + (y*eCols + j);
			out[index] = entry[i*eCols + j] + beta * out[index];
		}

	}
}

float * doMultiply2Matrices(
        int a1Rows, int a1Cols,  float * A1,
        int a2Rows, int a2Cols,  float * A2,
	float* C, float alpha)
{
    float beta = 0;
    cublasHandle_t  handle;

    cublasCreate (&handle) ;

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



float * doMultiply2MatricesStreaming(
        int a1Rows, int a1Cols,  float * A1,
        int a2Rows, int a2Cols,  float * A2,
	float* C, cudaStream_t cudaStream, cublasHandle_t handle, float alpha)
{

    //float alpha = 1.0;
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

