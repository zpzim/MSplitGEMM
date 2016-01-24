#ifndef _COMMON_H_
#define _COMMON_H_

#include <cuda_runtime.h>
#include "cublas_v2.h"

void PrintMatrix(char name[], int rows, int cols, const float* m);
void copyElements(float* out, float* entry, unsigned long long eRows, unsigned long long eCols, unsigned long long oRows, unsigned long long oCols, unsigned long long x, unsigned long long y,
	unsigned long long ofA, unsigned long long ofB, float beta);

float * doMultiply2Matrices(
        int a1Rows, int a1Cols,  float * A1,
        int a2Rows, int a2Cols,  float * A2,
	float* C, float beta);

float * doMultiply2MatricesStreaming(
        int a1Rows, int a1Cols,  float * A1,
        int a2Rows, int a2Cols,  float * A2,
	float* C, cudaStream_t cudaStream, cublasHandle_t h, float alpha);

#endif
