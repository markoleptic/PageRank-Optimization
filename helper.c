/*
  General helper functions for PDN class assignments.


  - richard.m.veras@ou.edu

*/

/*
  Helper functions
*/
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_float_mem(char *name, int vlen, float *src)
{

	printf("%s = [ ", name);
	for (int vid = 0; vid < vlen; ++vid)
	{
		if (src[vid] < 0.0f)
			printf(" x, " /*src[vid]*/);
		else
			printf("%2.f, ", src[vid]);
	}
	printf("]\n");
}

void print_float_matrix_mem(char *name, int m, int n, int rs, int cs, float *src)
{

	printf("%s =\n", name);
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if (src[i * cs + j * rs] < 0.0f)
				printf(" x, " /*src[i*cs + j*rs] */);
			else
				printf("%2.f, ", src[i * cs + j * rs]);
		}
		printf("\n");
	}
	printf("\n");
}

float max_pair_wise_diff_vect(int m, float *a, float *b)
{
	float max_diff = 0.0;

	for (int i = 0; i < m; ++i)
	{
		float sum = fabs(a[i] + b[i]);
		float diff = fabs(a[i] - b[i]);

		float res = 0.0f;

		if (sum == 0.0f)
			res = diff;
		else
			res = 2 * diff / sum;

		if (res > max_diff)
			max_diff = res;
	}

	return max_diff;
}

void fill_sequences(int size, int start, float *dst)
{
	for (int i = 0; i < size; ++i)
		dst[i] = (float)(i + start);
}

void fill_sequences_normalized(int size, int start, float *dst)
{
	float val = 1.0f / ((float)size);
	for (int i = 0; i < size; ++i)
		dst[i] = val;
}

void neg_xout_sequences(int size, float *dst)
{
	for (int i = 0; i < size; ++i)
	{
		dst[i] = -1.0f;
	}
}

void zero_out_sequences(int size, float *dst)
{
	for (int i = 0; i < size; ++i)
	{
		dst[i] = 0.0f;
	}
}

void fill_buffer_with_random(int num_elems, float *buff)
{
	long long range = RAND_MAX;
	// long long range = 1000;

	for (int i = 0; i < num_elems; ++i)
	{
		buff[i] = ((float)(rand() - ((range) / 2))) / ((float)range);
	}
}

void fill_buffer_with_value(int num_elems, float val, float *buff)
{
	for (int i = 0; i < num_elems; ++i)
		buff[i] = val;
}

float max_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b)
{
	float max_diff = 0.0;

	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
		{
			float sum = fabs(a[i * rs + j * cs] + b[i * rs + j * cs]);
			float diff = fabs(a[i * rs + j * cs] - b[i * rs + j * cs]);

			float res = 0.0f;

			if (sum == 0.0f)
				res = diff;
			else
				res = 2 * diff / sum;

			if (res > max_diff)
				max_diff = res;
		}

	return max_diff;
}

int scale_p_on_pos_ret_v_on_neg(int p, int v)
{
	if (v < 1)
		return -1 * v;
	else
		return v * p;
}
