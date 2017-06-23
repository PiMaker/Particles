#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas.h"
#include <stdio.h>

#define SIZE 7
#define Opacity pi
#define X pi + 1
#define Y pi + 2
#define Z pi + 3
#define PrevX pi + 4
#define PrevY pi + 5
#define PrevZ pi + 6

__global__ void kernel(float* p, float* out, int N, float delta)
{
	int particle = blockDim.x * blockIdx.x + threadIdx.x;
	int pi = particle * SIZE;
	bool valid = particle < N && p[Opacity] > 0;

	if (valid)
	{
		// == Opacity ==
		out[Opacity] = p[Opacity] - 0.0002f * delta;

		// == "Gravity" ==
		p[Y] += 0.00000003 * delta;

		// == Inertia ==
		float x = p[X]*2-p[PrevX];
		float y = p[Y]*2-p[PrevY];
		float z = p[Z]*2-p[PrevZ];
		out[PrevX] = p[X];
		out[PrevY] = p[Y];
		out[PrevZ] = p[Z];
		out[X] = x;
		out[Y] = y;
		out[Z] = z;
	}
}

int main()
{
	return 0;
}

/*

p - Memory layout:

i = particle index * 11
i = opacity, active if > 0
i + 1 = Position X
i + 2 = Position Y
i + 3 = Position Z
i + 4 = Prev. Position X
i + 5 = Prev. Position Y
i + 6 = Prev. Position Z

*/