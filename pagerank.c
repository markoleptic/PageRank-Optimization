/*
  This file contains the helpers for pagerank.

  - richard.m.veras@ou.edu

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "helper.h"
#include "pagerank.h"
#include "sparse.h"

// SOURCE:
// https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
int ipow(int base, int exp)
{
	int result = 1;
	for (;;)
	{
		if (exp & 1)
			result *= base;
		exp >>= 1;
		if (!exp)
			break;
		base *= base;
	}

	return result;
}

float get_kron_value_backend(int depth, int i, int j, dense_matrix_t *densities)
{
	int md = densities->m;
	int nd = densities->n;
	int rsd = densities->rs;
	int csd = densities->cs;

	float probability;

	if (depth == 0)
	{
		probability = 1.0f;
	}
	else
	{
		int depth_min_1 = depth - 1;

		int md_pow_depth_min_1 = ipow(md, depth_min_1);
		int nd_pow_depth_min_1 = ipow(nd, depth_min_1);

		int cur_i = i / md_pow_depth_min_1;
		int cur_j = j / nd_pow_depth_min_1;

		int next_i = i % md_pow_depth_min_1;
		int next_j = j % nd_pow_depth_min_1;

		probability = densities->values[cur_i * csd + cur_j * rsd] *
			      get_kron_value_backend(depth_min_1, next_i, next_j, densities);
	}

	return probability;
}

float get_kron_value(int m, int n, int i, int j, dense_matrix_t *densities)
{
	int md = densities->m;
	int nd = densities->n;

	int depth_estimate_m = (int)ceil(log(m) / log(md));
	int depth_estimate_n = (int)ceil(log(n) / log(nd));
	int depth;

	if (depth_estimate_m > depth_estimate_n)
		depth = depth_estimate_m;
	else
		depth = depth_estimate_n;

	return get_kron_value_backend(depth, i, j, densities);
}

dense_matrix_t *create_dense_matrix_with_kronecker_random_data(int m, int n, dense_matrix_t *densities)
{
	int cs = 1;
	int rs = m;

	// Initialize the matrix with zeros
	float *values = (float *)calloc(m * n, sizeof(float));

	// seed the random number generator. This is not the thread safe version
	srand(time(NULL));

	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
		{
			int rand_int = rand();

			// This will give us a value between [0,1]
			float rand_float = ((float)rand_int) / ((float)RAND_MAX);

			float kron_density = get_kron_value(m, n, i, j, densities);

			if (rand_float < kron_density)
			{
				values[i * rs + j * cs] = 1.0f; // kron_density;
			}
		}

	dense_matrix_t *A = create_dense_matrix_and_attach_existing_array(m, n, cs, rs, values);

	return A;
}

long *get_degree_vector_from_coo_matrix(coo_matrix_t *graph_input)
{

	// allocate and zero out degree
	long *degree = (long *)calloc(graph_input->m, sizeof(long));

	// count the number of outgoing edges
	for (int cur_nnz = 0; cur_nnz < graph_input->nnz; ++cur_nnz)
	{
		int i = graph_input->row_idx[cur_nnz];
		int j = graph_input->col_idx[cur_nnz];
		float val = graph_input->values[cur_nnz];

		degree[j]++;
	}

	return degree;
}

// Divide each nonzero in a column by the number of non-zeros in that column
void normalized_out_degree_of_coo_matrix(long *degree, coo_matrix_t *graph_input)
{
	for (int cur_nnz = 0; cur_nnz < graph_input->nnz; ++cur_nnz)
	{
		int i = graph_input->row_idx[cur_nnz];
		int j = graph_input->col_idx[cur_nnz];
		float val = graph_input->values[cur_nnz];

		graph_input->values[cur_nnz] = 1.0f / ((float)degree[j]);
	}
}

/*
  This function generates a random kronecker graph using the dimensions and provided densities.
*/
coo_matrix_t *create_coo_matrix_with_kronecker_random_data(int m, int n, dense_matrix_t *densities)
{
	// Initialize the matrix with zeros

	int cur_max = 2 * (m + n);

	int *col_idx = (int *)malloc(cur_max * sizeof(int));
	int *row_idx = (int *)malloc(cur_max * sizeof(int));
	float *values = (float *)malloc(cur_max * sizeof(float));

	int cur_nnz = 0;

	// seed the random number generator. This is not the thread safe version
	srand(time(NULL));

	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
		{
			int rand_int = rand();

			// This will give us a value between [0,1]
			float rand_float = ((float)rand_int) / ((float)RAND_MAX);

			// Now we need to figure out the probability of A_{i,j} existing.
			float kron_density = get_kron_value(m, n, i, j, densities);

			// If our random float is less than our predicted density then we have a non-zero.
			if (rand_float < kron_density)
			{

				if (cur_nnz + 1 > cur_max)
				{
					// update the current max number of elements
					cur_max *= 2;

					// reallocate the buffer and update the pointers
					col_idx = (int *)realloc(col_idx, cur_max * sizeof(int));
					row_idx = (int *)realloc(row_idx, cur_max * sizeof(int));
					values = (float *)realloc(values, cur_max * sizeof(float));

					// check if we were successful
					if (col_idx == NULL || row_idx == NULL || values == NULL)
					{
						printf("ERROR: Failed to reallocate COO buffers.\n");
						exit(1);
					}
				}

				// add entry to array
				row_idx[cur_nnz] = i;
				col_idx[cur_nnz] = j;
				values[cur_nnz] = 1.0f;

				// update the number of non-zero count
				cur_nnz++;
			}
		}

	// create the coo structure
	coo_matrix_t *A_coo = (coo_matrix_t *)malloc(sizeof(coo_matrix_t));

	A_coo->nnz = cur_nnz;
	A_coo->m = m;
	A_coo->n = n;
	A_coo->row_idx = row_idx;
	A_coo->col_idx = col_idx;
	A_coo->values = values;

	// return the structure
	return A_coo;
}

pagerank_data_t *initialize_pagerank_data(int m0, int k0)
{
	pagerank_data_t *pagerank_data_sequential = (pagerank_data_t *)malloc(sizeof(pagerank_data_t));

	pagerank_data_sequential->buff = (float *)calloc(sizeof(float), m0 * 2);

	pagerank_data_sequential->num_iterations = k0;
	pagerank_data_sequential->num_vertices = m0;

	// Fill the buffers
	pagerank_data_sequential->x = &(pagerank_data_sequential->buff[0]);
	pagerank_data_sequential->y = &(pagerank_data_sequential->buff[m0]);

	fill_sequences_normalized(m0, 1, pagerank_data_sequential->x);

	return pagerank_data_sequential;
}

void destroy_pagerank_data(pagerank_data_t *pagerank_data_sequential)
{
	free(pagerank_data_sequential->buff);
	free(pagerank_data_sequential);
}

void copy_pagerank_data(pagerank_data_t *pagerank_data_dst, pagerank_data_t *pagerank_data_src)
{
	int buff_size = sizeof(float) * 2 * pagerank_data_src->num_vertices;

	memcpy(pagerank_data_dst->buff, pagerank_data_src->buff, buff_size);
}

pagerank_data_t *clone_pagerank_data(pagerank_data_t *pagerank_data_src)
{

	int m0 = pagerank_data_src->num_vertices;
	int k0 = pagerank_data_src->num_iterations;

	pagerank_data_t *pagerank_data_dst = (pagerank_data_t *)malloc(sizeof(pagerank_data_t));

	pagerank_data_dst->num_vertices = pagerank_data_src->num_vertices;
	pagerank_data_dst->num_iterations = pagerank_data_src->num_iterations;

	pagerank_data_dst->buff = (float *)calloc(sizeof(float), m0 * 2);

	// adjust the x and y pointers
	pagerank_data_dst->x = &(pagerank_data_dst->buff[0]);
	pagerank_data_dst->y = &(pagerank_data_dst->buff[m0]);

	// copy over the data
	copy_pagerank_data(pagerank_data_dst, pagerank_data_src);

	return pagerank_data_dst;
}

multiformat_graph_t *initialize_multiformat_graph(coo_matrix_t *coo_graph)
{
	int m0 = coo_graph->m;
	int nnz = coo_graph->nnz;

	long *degree = get_degree_vector_from_coo_matrix(coo_graph);
	// pretty_print_coo_matrix( coo_graph );

	normalized_out_degree_of_coo_matrix(degree, coo_graph);
	// pretty_print_coo_matrix( coo_graph );

	multiformat_graph_t *multiformat_graph_sequential;
	multiformat_graph_sequential = (multiformat_graph_t *)malloc(sizeof(multiformat_graph_t));

	multiformat_graph_sequential->m = m0;
	multiformat_graph_sequential->n = m0;
	multiformat_graph_sequential->degree = degree;
	multiformat_graph_sequential->nnz = coo_graph->nnz;

	// NOTE: only coo view is populated
	multiformat_graph_sequential->graph_view_coo = coo_graph;
	multiformat_graph_sequential->graph_view_csr = NULL;
	multiformat_graph_sequential->graph_view_csc = NULL;
	multiformat_graph_sequential->graph_view_bcsr = NULL;

	return multiformat_graph_sequential;
}

void destroy_multiformat_graph(multiformat_graph_t *multiformat_graph_sequential)
{
	free(multiformat_graph_sequential->degree);
	free(multiformat_graph_sequential);
}
