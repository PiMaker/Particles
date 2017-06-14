#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void kernel(int* a, int* b, int* out, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		out[i] = a[i] + b[i];
}

int main()
{
	return 0;
}