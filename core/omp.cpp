#include "omp.hpp"
#include <fstream>

void ParallelFill(float* datas, int resolution, FillFunc fillFunc) {
	#pragma omp parallel for
	for (int i = 0; i < resolution; i++)
		for (int j = 0; j < resolution; j++)
			for (int k = 0; k < resolution; k++)
			{
				float u, v, w;
				u = (i + 0.5) / resolution;
				v = (j + 0.5) / resolution;
				w = (k + 0.5) / resolution;
				float res = fillFunc(i, j, k, u, v, w);
				int index = (i * resolution + j) * resolution + k;
				datas[index] = res;
			}
}

void ParallelFill2(float* datas, int resolution, FillFunc fillFunc) {
	//#pragma omp parallel for

	int t = 0;

	for (int i = 0; i < resolution; i++)
		for (int j = 0; j < resolution; j++)
			for (int k = 0; k < resolution; k++)
			{
				static int test = 1184;
				if (t == test)
				{
					t += 0;
				}
				++t;
				//if (i == 34 && j == 62 && k == 17)
				//{
				//	i = i + 0;
				//}

				float u, v, w;
				u = (i + 0.5) / resolution;
				v = (j + 0.5) / resolution;
				w = (k + 0.5) / resolution;
				float res = fillFunc(i, j, k, u, v, w);

				static float t = 0.98f;
				if (res < t)
				{
					res *= 1.00000f;
				}

				//static float test = 0.1;
				//
				//if (test < res)
				//{
				//	res *= 1.0000000f;
				//}

				int index = (i * resolution + j) * resolution + k;
				datas[index] = res;
			}
}

void ParallelFor(float* result, int length, LoopFunc func) {
	#pragma omp parallel for
	for (int i = 0; i < length; i++) {
		result[i] = func(i);
	}	
}