#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, float *C, unsigned int m, unsigned int k,
  unsigned int n) {

  const float relativeTolerance = 1e-5;

  for(int row = 0; row < m; ++row) {
    for(int col = 0; col < n; ++col) {
      float sum = 0;
      for(unsigned int i = 0; i < k; ++i) {
        sum += A[row*k + i]*B[i*n + col];
      }
      float relativeError = (sum - C[row*n + col])/sum;
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
	printf("(%d, %d) = %f, supposed to be %f\n", row, col, C[row*n + col], sum); 
        printf("TEST FAILED\n\n");
        exit(0);
      }
    }
  }
  printf("TEST PASSED\n\n");

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

