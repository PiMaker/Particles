#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas.h"
#include <stdio.h>

#define SIZE 10
#define Active pi
#define X pi + 1
#define Y pi + 2
#define PrevX pi + 3
#define PrevY pi + 4
#define Radius pi + 5

inline __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
inline __device__ float rsqrtf2(float x)
{
	return 1.0f / sqrtf(x);
}
inline __device__ float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __device__ float2 normalize(float2 v)
{
	float invLen = rsqrtf2(dot(v, v));
	return v * invLen;
}

__global__ void kernel(float* p, float* out, int N, float delta, float* area)
{
	int particle = blockDim.x * blockIdx.x + threadIdx.x;
	int pi = particle * SIZE;
	bool valid = particle < N && p[Active] > 0;

	if (valid)
	{
		// == Gravity ==
		//p[Y] += 0.0005 * delta;

		float2 c = make_float2(area[0] + area[2] / 2, area[1] + area[3] / 2);
		float2 pos = make_float2(p[X], p[Y]);
		float2 diff = c - pos;
		float2 gravity = normalize(diff) * 0.0005 * delta;
		p[Y] += gravity.y;
		p[X] += gravity.x;
	}

	// == Sync ==
	__syncthreads();

	if (valid)
	{
		// == Collision (naive) ==
		for (int j = 0; j < N; j++)
		{
			int pi2 = j * SIZE;

			if (pi == pi2)
			{
				continue;
			}

			float x = p[X] - p[pi2 + 1];
			float y = p[Y] - p[pi2 + 2];
			float slength = x*x + y*y;
			float length = sqrtf(slength);
			float target = p[Radius] + p[pi2 + 5];

			if (length < target)
			{
				float factor = (length - target) / length;
				p[X] -= x*factor*0.5;
				p[Y] -= y*factor*0.5;
				p[pi2 + 1] += x*factor*0.5;
				p[pi2 + 2] += y*factor*0.5;
			}
		}


		// == Border Collision ==
			
		// Bottom
		if (p[Y] + p[Radius] > area[3])
		{
			p[Y] = area[3] - p[Radius];
		}

		// Top
		if (p[Y] - p[Radius] < area[1])
		{
			p[Y] = area[1] + p[Radius];
		}

		// Left
		if (p[X] - p[Radius] < area[0])
		{
			p[X] = area[0] + p[Radius];
		}

		// Right
		if (p[X] + p[Radius] > area[2])
		{
			p[X] = area[2] - p[Radius];
		}


		// == Inertia ==
		float x = p[X] + (p[X] - p[PrevX]);
		float y = p[Y] + (p[Y] - p[PrevY]);
		out[PrevX] = p[X];
		out[PrevY] = p[Y];
		out[X] = x;
		out[Y] = y;
	}
}

int main()
{
	return 0;
}

/*

p - Memory layout:

i = particle index * 10
i = active if > 0
i + 1 = Position X
i + 2 = Position Y
i + 3 = Prev. Position X
i + 4 = Prev. Position Y
i + 5 = Radius
i + 6/7/8/9 = Color (RGBA)

*/