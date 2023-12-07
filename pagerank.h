
#ifndef PAGERANK_H
#define PAGERANK_H

#include "sparse.h"

/*
   STUDENT_TODO: SORT OF ...

   Inside this struct are a bunch of pointers for different matrix formats.
   The idea is that if you want to change to a different format you won't
   have to change the function definitions for allocate,distribute,compute,
   collect, and free.

*/
typedef struct multiformat_graph_ts
{
	// Generic parameters
	int m;
	int n;
	int nnz;

	// This is the out degrees for all of the vertices.
	long *degree;

	// These are potentially different views of the same graph.
	coo_matrix_t *graph_view_coo;
	csr_matrix_t *graph_view_csr;
	csc_matrix_t *graph_view_csc;
	bcsr_matrix_t *graph_view_bcsr;
    indices *indices; // Row indices, split into simd and regular in split_indices function
} multiformat_graph_t;

typedef struct pagerank_data_ts
{
	// Number of iterations in the power method A^{num_iterations}
	int num_iterations;
	int num_vertices;

	// This is the buffer containing both x and y that we ping-pong between.
	float *buff;

	// These will point to locations in buff.
	float *x;
	float *y;
} pagerank_data_t;

int ipow(int base, int exp);
float get_kron_value_backend(int depth, int i, int j, dense_matrix_t *densities);
float get_kron_value(int m, int n, int i, int j, dense_matrix_t *densities);
dense_matrix_t *create_dense_matrix_with_kronecker_random_data(int m, int n, dense_matrix_t *densities);
long *get_degree_vector_from_coo_matrix(coo_matrix_t *graph_input);
void normalized_out_degree_of_coo_matrix(long *degree, coo_matrix_t *graph_input);
coo_matrix_t *create_coo_matrix_with_kronecker_random_data(int m, int n, dense_matrix_t *densities);
pagerank_data_t *initialize_pagerank_data(int m0, int k0);
void destroy_pagerank_data(pagerank_data_t *pagerank_data_sequential);
pagerank_data_t *clone_pagerank_data(pagerank_data_t *pagerank_data_src);
void copy_pagerank_data(pagerank_data_t *pagerank_data_dst, pagerank_data_t *pagerank_data_src);

multiformat_graph_t *initialize_multiformat_graph(coo_matrix_t *coo_graph);
void destroy_multiformat_graph(multiformat_graph_t *multiformat_graph_sequential);

#endif /* PAGERANK_H */
