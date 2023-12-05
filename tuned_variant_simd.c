/*
  These set of functions perform a PageRank computation on a graph stored as a sparse matrix.
  The bulk of this computation is an iterative matrix-times-vector product.

  Parameters:

    int num_vertices;    // Number of vertices in the graph
    int num_iterations;  // Number of iterations in the pagerank operation
    multiformat_graph_t *multiformat_graph_sequential; // The graph data coming from the test harness
    multiformat_graph_t *multiformat_graph_distributed; // The distributed copy of the data

    pagerank_data_t *pagerank_data_sequential; // The intermediate and final ranking data
    pagerank_data_t *pagerank_data_distributed; // The distributed copy of the ranking data


  Functions:

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across the system.
  COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to the sequential
  one for testing.
  DISTRIBUTED_FREE_NAME(...): Free the distributed buffers that were allocated


  - richard.m.veras@ou.edu

*/

#define DEBUG 0

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>

#include "helper.h"
#include "pagerank.h"
#include "sparse.h"
#include "utils.c"
#include <immintrin.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME baseline_collect
#endif

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif

/*
Prints row_idx, ind->reg_indices, and ind->simd_indices to indices.txt
*/
static void print_indices(int *row_idx, const indices *ind, const int nnz)
{
	FILE *file = fopen("indices.txt", "w");
	fprintf(file, "Regular Indices:\n");
	for (int i = 0; i < nnz; ++i)
	{
		fprintf(file, "%d: %d \n", i, row_idx[i]);
	}
	fprintf(file, "Regular Indices:\n");
	for (int i = 0; i < ind->reg_size; ++i)
	{
		fprintf(file, "Start: %d, End: %d\n", ind->reg_indices[i].start, ind->reg_indices[i].end);
	}
	fprintf(file, "SIMD Indices:\n");
	for (int i = 0; i < ind->simd_size; ++i)
	{
		fprintf(file, "Start: %d, End: %d\n", ind->simd_indices[i].start, ind->simd_indices[i].end);
	}
}

/*
Prints reference/test arrays side by side in text files
*/
static void print_test_diff(multiformat_graph_t *multiformat_graph_distributed,
			    pagerank_data_t *pagerank_data_distributed)
{
	coo_matrix_t *graph = multiformat_graph_distributed->graph_view_coo;
	float *reference = calloc(multiformat_graph_distributed->m, sizeof(float));
	FILE *file = fopen("test.txt", "w");
	for (int cur_pos = 0; cur_pos < graph->nnz; ++cur_pos)
	{
		int i = graph->row_idx[cur_pos];
		int j = graph->col_idx[cur_pos];
		fprintf(file, "%d %d %d\n", cur_pos, i, j);
		float val = graph->values[cur_pos];
		reference[i] += val * pagerank_data_distributed->x[j];
	}
	fclose(file);
	printDistributedDiff(reference, pagerank_data_distributed->y, multiformat_graph_distributed->m,
			     "test_compare.txt");
	printDistributedOutput2(reference, pagerank_data_distributed->y, multiformat_graph_distributed->m,
				"test_compare2.txt");
	free(reference);
}

/*
Adds a new entry into indices->simd_indices or indices->reg_indices
*/
static void add_index(indices *in, int bReg)
{
	if (bReg == 1)
	{
		in->reg_indices = realloc(in->reg_indices, (in->reg_size + 1) * sizeof(reg_index));
		in->reg_size += 1;
	}
	else
	{
		in->simd_indices = realloc(in->simd_indices, (in->simd_size + 1) * sizeof(simd_index));
		in->simd_size += 1;
	}
}

/*
Splits the row_idx array into indices->simd_indices and indices->reg_indices.
If the value of row_idx does not change for 8 in a row, it is added to
simd_indices, otherwise its added to reg_indices. */
/*
input row_idx array:
0: 0 --------|
1: 0         |
2: 0         |
3: 0         |simd_indices
4: 0         |
5: 0         |
6: 0         |
7: 0 --------|
8: 0 --------|
9: 0         |
10: 0        |
11: 0        |simd_indices
12: 0        |
13: 0        |
14: 0        |
15: 0 -------|
16: 0 -------|
17: 0        |reg_indices
18: 0 -------|
19: 1 -------|
20: 1        |
21: 1        |
22: 1        | simd_indices
23: 1        |
24: 1        |
25: 1        |
26: 1 -------|
27: 1        |
28: 1        |
29: 1        |
30: 1        | simd_indices
31: 1        |
32: 1        |
33: 1        |
34: 1 -------|
35: 1 -------|reg_indices


Regular Indices:
i: 0 Start: 16, End: 19
i: 1 Start: 35, End: 36

SIMD Indices:
i: 0 Start: 0, End: 8
i: 1 Start: 8, End: 16
i: 2 Start: 19, End: 27
i: 3 Start: 27, End: 35
*/
static void *split_indices(indices *indices, int *row_idx, int nnz)
{
	int start = 0;
	int end = 0;
	indices->reg_size = 0;
	indices->simd_size = 0;

	for (int i = 1; i < nnz; ++i)
	{
		if (row_idx[i] != row_idx[i - 1])
		{
			start = end;
			end = i;
			int consecutive = end - start;
            // Add chunk to reg_indices
			if (consecutive < 8)
			{
				add_index(indices, 1);
				int idx = indices->reg_size - 1;
				indices->reg_indices[idx].start = start;
				indices->reg_indices[idx].end = end;
			}
            // Add atleast one chunk of 8 to simd_indices
			else
			{
				int simd_end = (start + (consecutive - (consecutive % 8)));
				int num_to_add = (simd_end - start) / 8;
                // Add chunks of 8 to simd_indices until num_to_add
				for (int j = 0; j < num_to_add; ++j)
				{
					add_index(indices, 0);
					int idx = indices->simd_size - 1;
					indices->simd_indices[idx].start = start + j * 8;
					indices->simd_indices[idx].end = indices->simd_indices[idx].start + 8;
				}
                // Add the remaining chunk to reg_indices
				if (consecutive > simd_end - start)
				{
					add_index(indices, 1);
					int reg_idx = indices->reg_size - 1;
					indices->reg_indices[reg_idx].start = indices->simd_indices[indices->simd_size - 1].end;
					indices->reg_indices[reg_idx].end = end;
				}
			}
		}
	}
    // Add the section that spans from where the above left off up to in_size
	add_index(indices, 1);
	int idx = indices->reg_size - 1;
	indices->reg_indices[idx].start = end;
	indices->reg_indices[idx].end = nnz;

#if DEBUG
	print_indices(row_idx, indices, nnz);
#endif
}

/*
Sums the eight values from a __m256 type into one float value.
https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
*/
static float sum8(__m256 x)
{
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 sum = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(sum);
}

static void baseline_matrix_mul(int start, int end, multiformat_graph_t *graph, pagerank_data_t *pagerank)
{
	for (int cur_pos = start; cur_pos < end; ++cur_pos)
	{
		int row_idx = graph->graph_view_coo->row_idx[cur_pos];
		int col_idx = graph->graph_view_coo->col_idx[cur_pos];
		float val = graph->graph_view_coo->values[cur_pos];
		pagerank->y[row_idx] += val * pagerank->x[col_idx];
	}
}

static void matvec(multiformat_graph_t *graph, pagerank_data_t *pagerank_data)
{
	coo_matrix_t *graph_view = graph->graph_view_coo;
	indices *indices = graph->indices;

    // This loop parallelizes the x array by processing 8 elements at a time
	for (int index = 0; index < indices->simd_size; ++index)
	{
		int cur_pos = indices->simd_indices[index].start;
		int i = graph_view->row_idx[cur_pos];

		// loads 8 consecutive values from values array, starting at cur_pos
		__m256 val_vector = _mm256_loadu_ps(&graph_view->values[cur_pos]);

		// loads 8 consecutive column indices from col_idx array, starting at cur_pos
		__m256i col_idx_vector = _mm256_loadu_si256((__m256i *)(&graph_view->col_idx[cur_pos]));

		// loads 8 consecutive values from the x array, using indices from col_idx_vector
		__m256 x_values_for_row = _mm256_i32gather_ps(pagerank_data->x, col_idx_vector, sizeof(float));

		// multiply value array by x array, and sum all 8 elements into one float value
		pagerank_data->y[i] += sum8(_mm256_mul_ps(val_vector, x_values_for_row));
	}
    // This loop does everything that couldn't be done in the simd loop
	for (int i = 0; i < indices->reg_size; ++i)
	{
		baseline_matrix_mul(indices->reg_indices[i].start, indices->reg_indices[i].end, graph, pagerank_data);
	}
}

/*
  This function iteratively performs a matrix-vector product to compute
  the PageRank of a graph. Mathematically it is performing:
     x_{n-1} = A^{n} x_{0}
  where n is the number of iterations, x_{n-1} is the final ranking and
  x_{0} is a random vector.
  In this implementation, we keep two vector x and y that point to two halves
  of a buffer, and we ping pong at each iteration by pointing them to a different
  half of the buffer.
*/
static void page_rank(multiformat_graph_t *multiformat_graph_distributed, pagerank_data_t *pagerank_data_distributed)
{
    for (int t = 0; t < pagerank_data_distributed->num_iterations; ++t)
    {
        ////////////////////////////////////////////////////////////////////////////
        // Ping pong the buffers by changing the pointers                         //
        // when t is even x[0..m-1] = buff[0..m-1] and y[0..m-1] = buff[m..2m-1]  //
        // when t is odd  x[0..m-1] = buff[m..2m-1] and y[0..m-1] = buff[0..m-1]  //
        ////////////////////////////////////////////////////////////////////////////
        pagerank_data_distributed->x =
            &(pagerank_data_distributed->buff[((t + 0) % 2) * multiformat_graph_distributed->m]);
        pagerank_data_distributed->y =
            &(pagerank_data_distributed->buff[((t + 1) % 2) * multiformat_graph_distributed->m]);
        // zero out the output y
        for (int i = 0; i < multiformat_graph_distributed->m; ++i)
            pagerank_data_distributed->y[i] = 0.0f;
        matvec(multiformat_graph_distributed, pagerank_data_distributed);
#if DEBUG
        float res = max_pair_wise_diff_vect(multiformat_graph_distributed->m,
                            pagerank_data_distributed->x, pagerank_data_distributed->y);
        printf("diff[%i]: %f\n", t, res);
#endif
    }
}

/*
  STUDENT_TODO: ... if this is not the baseline. If you want to use a format other
		than coo, then it would be worth making a function that creates an
		instance of the new format from the original COO data.

  static csr_matrix_t* create_csr_matrix_from_coo_matrix( coo_matrix_t *src_coo ){}
  static csc_matrix_t* create_csc_matrix_from_coo_matrix( coo_matrix_t *src_coo ){}
  static bcsr_matrix_t* create_bcsr_matrix_from_coo_matrix( coo_matrix_t *src_coo ){}
*/

// given an input matrix in coordinate format this produces a new coordinate matrix.
static coo_matrix_t *create_coo_matrix_from_coo_matrix(coo_matrix_t *src_coo)
{
	// allocate the structure.
	coo_matrix_t *dst_coo = (coo_matrix_t *)malloc(sizeof(coo_matrix_t));

	// copy the metadata/
	dst_coo->m = src_coo->m;
	dst_coo->n = src_coo->n;
	dst_coo->nnz = src_coo->nnz;

	// Allocated the buffers.
	dst_coo->row_idx = (int *)malloc(sizeof(int) * src_coo->nnz);
	dst_coo->col_idx = (int *)malloc(sizeof(int) * src_coo->nnz);
	dst_coo->values = (float *)malloc(sizeof(float) * src_coo->nnz);

	// Copy the buffers.
	memcpy(dst_coo->row_idx, src_coo->row_idx, sizeof(int) * src_coo->nnz);
	memcpy(dst_coo->col_idx, src_coo->col_idx, sizeof(int) * src_coo->nnz);
	memcpy(dst_coo->values, src_coo->values, sizeof(float) * src_coo->nnz);

	// return the newly created and filled structure.
	return dst_coo;
}

void COMPUTE_NAME(int num_vertices, int num_iterations, multiformat_graph_t *multiformat_graph_distributed,
		  pagerank_data_t *pagerank_data_distributed)

{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{

		/////////////
		// COMPUTE //
		/////////////
		page_rank(multiformat_graph_distributed, pagerank_data_distributed);
	}
	else
	{
	}
}

// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME(int num_vertices, int num_iterations,
			       multiformat_graph_t **multiformat_graph_distributed,
			       pagerank_data_t **pagerank_data_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	/*
	  STUDENT_TODO: If this is not the baseline then you might want to change this
			especially if you intend on doing distributed memory work.

	 */

	if (rid == root_rid)
	{
		*multiformat_graph_distributed = (multiformat_graph_t *)malloc(sizeof(multiformat_graph_t));

		*pagerank_data_distributed = (pagerank_data_t *)malloc(sizeof(pagerank_data_t));
		(*pagerank_data_distributed)->buff = (float *)calloc(sizeof(float), num_vertices * 2);
	}
	else
	{
	}
}

/*
  This takes the sequential data located on the root node and distributes it
  to the nodes that will need it.
*/
void DISTRIBUTE_DATA_NAME(int num_vertices, int num_iterations, multiformat_graph_t *multiformat_graph_sequential,
			  pagerank_data_t *pagerank_data_sequential, multiformat_graph_t *multiformat_graph_distributed,
			  pagerank_data_t *pagerank_data_distributed)
{

	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{

		////////////////
		// DISTRIBUTE //
		////////////////

		//////////////////////////////
		// Copy the pagerank struct //
		//////////////////////////////
		int buff_size = sizeof(float) * 2 * pagerank_data_sequential->num_vertices;

		memcpy(pagerank_data_distributed->buff, pagerank_data_sequential->buff, buff_size);

		// Here we are just point x to the first half of buff and y to the second half.
		pagerank_data_distributed->x = &(pagerank_data_distributed->buff[0]);
		pagerank_data_distributed->y = &(pagerank_data_distributed->buff[multiformat_graph_sequential->m]);
		pagerank_data_distributed->num_iterations = pagerank_data_sequential->num_iterations;
		pagerank_data_distributed->num_vertices = pagerank_data_sequential->num_vertices;

		////////////////////////////////
		// Copy the multiformat graph //
		////////////////////////////////
		/*
		  STUDENT_TODO: If this is not the baseline and you want to use a matrix format other
				than Coordinate (COO) this is the place where you might want to make
				changes.
		*/
		// Copy the parameters of the graph from sequential to distributed
		multiformat_graph_distributed->m = multiformat_graph_sequential->m;
		multiformat_graph_distributed->n = multiformat_graph_sequential->n;
		multiformat_graph_distributed->nnz = multiformat_graph_sequential->nnz;

		// Copy over the degree of the nodes
		multiformat_graph_distributed->degree = (long *)malloc(sizeof(long) * multiformat_graph_sequential->m);

		memcpy(multiformat_graph_distributed->degree, multiformat_graph_sequential->degree,
		       sizeof(long) * multiformat_graph_sequential->m);
		/*
		  STUDENT_TODO: ... if you want ..

		  If this is not the baseline, then this is a good place to convert the
		  graph from Coordinate format (COO) to a different sparse matrix format.
		  Just make sure to free up any new structures you create when you are done.
		*/
		multiformat_graph_distributed->graph_view_coo =
		    create_coo_matrix_from_coo_matrix(multiformat_graph_sequential->graph_view_coo);

		/*
		   STUDENT_TODO: if this is not the baseline and it makes sense for your
				 implementation. (NOTE: you have done part of this in a hw).

		   csr_matrix_t* create_csr_matrix_from_coo_matrix( coo_matrix_t *src_coo );
		   csc_matrix_t* create_csc_matrix_from_coo_matrix( coo_matrix_t *src_coo );
		   bcsr_matrix_t* create_bcsr_matrix_from_coo_matrix( coo_matrix_t *src_coo );
		*/
		multiformat_graph_distributed->graph_view_csr = NULL;
		multiformat_graph_distributed->graph_view_csc = NULL;
		multiformat_graph_distributed->graph_view_bcsr = NULL;

		multiformat_graph_distributed->indices = malloc(sizeof(indices));
		multiformat_graph_distributed->indices->reg_indices = malloc(sizeof(reg_index));
		multiformat_graph_distributed->indices->simd_indices = malloc(sizeof(simd_index));
		split_indices(multiformat_graph_distributed->indices,
			      multiformat_graph_distributed->graph_view_coo->row_idx,
			      multiformat_graph_distributed->graph_view_coo->nnz);
	}
	else
	{
	}
}

/*
  This function gathers the resulting data distributed across potentially many
  nodes and copies it to its sequential counterpart.
*/

void COLLECT_DATA_NAME(int num_vertices, int num_iterations, pagerank_data_t *pagerank_data_distributed,
		       pagerank_data_t *pagerank_data_sequential)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{
		/*
		  STUDENT_TODO: if this is not the baseline and you have
				used distributed memory then you might need
				to do something here. However, because this
				is iteratively computing on the data you might
				need to be exchanging data at every iteration.
		 */
		copy_pagerank_data(pagerank_data_sequential, pagerank_data_distributed);
	}
	else
	{
	}
}

void DISTRIBUTED_FREE_NAME(int num_vertices, int num_iterations, multiformat_graph_t *multiformat_graph_distributed,
			   pagerank_data_t *pagerank_data_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{
		//////////
		// FREE //
		//////////
		free(pagerank_data_distributed->buff);
		free(pagerank_data_distributed);

		/*
		  STUDENT_TODO: if this is not the baseline and you are
				using something other than COO, then you
				might need to free that data up here.
		*/
		destroy_coo_matrix(multiformat_graph_distributed->graph_view_coo);
		// destroy_csr_matrix(multiformat_graph_distributed->graph_view_csr);
		// destroy_csc_matrix(multiformat_graph_distributed->graph_view_csc);
		// destroy_bcsr_matrix(multiformat_graph_distributed->graph_view_bcsr);
		free(multiformat_graph_distributed->indices->reg_indices);
		free(multiformat_graph_distributed->indices->simd_indices);
		free(multiformat_graph_distributed->indices);

		free(multiformat_graph_distributed->degree);
		free(multiformat_graph_distributed);
	}
	else
	{
	}
}
