#include <immintrin.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void printDistributedOutput(const float *array, const int size, const char *fileName)
{
	FILE *file = fopen(fileName, "a");
	if (file == NULL)
	{
		perror("Failed to open file");
		return;
	}
	for (int i = 0; i < size; i++)
	{
		fprintf(file, "%.1f\n", array[i]);
	}
	fclose(file);
}

void printDistributedOutput2(const float *array, const float *array2, const int size, const char *fileName)
{
	FILE *file = fopen(fileName, "w");
	if (file == NULL)
	{
		perror("Failed to open file");
		return;
	}
	for (int i = 0; i < size; i++)
	{
		fprintf(file, "Ref: %.3f New: %.3f\n", array[i], array2[i]);
	}
	fclose(file);
}

void printDistributedDiff(const float *array, const float *array2, const int size, const char *fileName)
{
	FILE *file = fopen(fileName, "w");
	if (file == NULL)
	{
		perror("Failed to open file");
		return;
	}
	for (int i = 0; i < size; i++)
	{
		float max_diff = 0.0;
		float sum = fabs(array[i] + array2[i]);
		float diff = fabs(array[i] - array2[i]);
		float res = 0.0f;
		if (sum == 0.0f)
			res = diff;
		else
			res = 2 * diff / sum;
		if (res > max_diff)
			max_diff = res;

		fprintf(file, "Diff at index %d: %f \n", i, max_diff);
	}
	fclose(file);
}

void printM256(__m256 vec)
{
	float values[8];
	_mm256_storeu_ps(values, vec);
	printf("__m256: ");
	for (int i = 0; i < 8; i++)
	{
		printf("%f ", values[i]);
	}
	printf("\n");
}

inline void printWeights(const float *array)
{
	printf("Normal: ");
	for (int i = 0; i < 8; i++)
	{
		printf("%f ", array[i]);
	}
	printf("\n");
}

inline void CheapFix(int m0, int k0, int badIndex, int offset, float *input_distributed, float *weights_distributed,
		     float *output_distributed)
{
	float res = 0.0f;
	for (int p0 = 0; p0 < k0; ++p0)
	{
		res += input_distributed[(p0 + badIndex) % m0] * weights_distributed[p0];
	}
	output_distributed[badIndex - offset] = res;
}