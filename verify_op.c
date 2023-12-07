#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper.h"
#include "pagerank.h"
#include "sparse.h"

#define ERROR_THRESHOLD 1e-4

extern void COMPUTE_NAME_REF(int num_vertices, int num_iterations, multiformat_graph_t *multiformat_graph_distributed,
			     pagerank_data_t *pagerank_data_distributed);

extern void COMPUTE_NAME_TST(int num_vertices, int num_iterations, multiformat_graph_t *multiformat_graph_distributed,
			     pagerank_data_t *pagerank_data_distributed);

extern void DISTRIBUTED_ALLOCATE_NAME_REF(int num_vertices, int num_iterations,
					  multiformat_graph_t **multiformat_graph_distributed,
					  pagerank_data_t **pagerank_data_distributed);

extern void DISTRIBUTED_ALLOCATE_NAME_TST(int num_vertices, int num_iterations,
					  multiformat_graph_t **multiformat_graph_distributed,
					  pagerank_data_t **pagerank_data_distributed);

extern void DISTRIBUTE_DATA_NAME_REF(int num_vertices, int num_iterations,
				     multiformat_graph_t *multiformat_graph_sequential,
				     pagerank_data_t *pagerank_data_sequential,
				     multiformat_graph_t *multiformat_graph_distributed,
				     pagerank_data_t *pagerank_data_distributed);

extern void DISTRIBUTE_DATA_NAME_TST(int num_vertices, int num_iterations,
				     multiformat_graph_t *multiformat_graph_sequential,
				     pagerank_data_t *pagerank_data_sequential,
				     multiformat_graph_t *multiformat_graph_distributed,
				     pagerank_data_t *pagerank_data_distributed);

extern void COLLECT_DATA_NAME_REF(int num_vertices, int num_iterations, pagerank_data_t *pagerank_data_distributed,
				  pagerank_data_t *pagerank_data_sequential);

extern void COLLECT_DATA_NAME_TST(int num_vertices, int num_iterations, pagerank_data_t *pagerank_data_distributed,
				  pagerank_data_t *pagerank_data_sequential);

extern void DISTRIBUTED_FREE_NAME_REF(int num_vertices, int num_iterations,
				      multiformat_graph_t *multiformat_graph_distributed,
				      pagerank_data_t *pagerank_data_distributed);

extern void DISTRIBUTED_FREE_NAME_TST(int num_vertices, int num_iterations,
				      multiformat_graph_t *multiformat_graph_distributed,
				      pagerank_data_t *pagerank_data_distributed);

int main(int argc, char *argv[])
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	// What we will output to
	FILE *result_file;

	// Problem parameters
	int min_size;
	int max_size;
	int step_size;

	int in_num_vertices;
	int in_num_iterations;

	// Get command line arguments
	if (argc == 1)
	{
		min_size = 16;
		max_size = 256;
		step_size = 16;

		// defaults
		in_num_vertices = 1;
		in_num_iterations = -3;

		// default to printing to stdout
		result_file = stdout;
	}
	else if (argc == 5 + 1 || argc == 6 + 1)
	{
		min_size = atoi(argv[1]);
		max_size = atoi(argv[2]);
		step_size = atoi(argv[3]);

		in_num_vertices = atoi(argv[4]);
		in_num_iterations = atoi(argv[5]);

		// default to printing to stdout
		result_file = stdout;

		if (argc == 6 + 1)
		{
			// we don't want every node opening the same file
			// to write to.
			if (rid == 0)
			{
				result_file = fopen(argv[6], "w");
			}
			else
			{
				result_file = NULL;
			}
		}
	}
	else
	{
		printf("usage: %s min max step num_vertices num_iterations [filename]\n", argv[0]);
		exit(1);
	}

	// Print out the first line of the output in csv format
	if (rid == 0)
	{
		/*root node */
		fprintf(result_file, "num_ranks,num_vertices,num_iterations,error,result\n");
	}
	else
	{ /* all other nodes*/
	}

	for (int p = min_size; p < max_size; p += step_size)
	{

		/////////////////
		// input sizes //
		/////////////////

		// Number of vertices in the graph
		int num_vertices = scale_p_on_pos_ret_v_on_neg(p, in_num_vertices);
		// Number of time steps
		int num_iterations = scale_p_on_pos_ret_v_on_neg(p, in_num_iterations);

		// These structs contain different views of the graph
		multiformat_graph_t *multiformat_graph_sequential;
		multiformat_graph_t *multiformat_graph_sequential_ref;
		multiformat_graph_t *multiformat_graph_sequential_tst;

		// These structs hold the working dataset and the results
		pagerank_data_t *pagerank_data_sequential;
		pagerank_data_t *pagerank_data_sequential_ref;
		pagerank_data_t *pagerank_data_sequential_tst;

		// Allocate the buffers for the sequential inputs
		// and create the data only on the root node
		if (rid == 0)
		{ /* root node */
			// Create a stochastic kronecker graph to approximate a web graph.
			float kron_stochastic_buf[] = {0.91f, 0.75f, 0.63f, 0.52f};
			dense_matrix_t *kron_stochastic_densities =
			    create_dense_matrix_and_attach_existing_array(2, 2, 2, 1, kron_stochastic_buf);

			coo_matrix_t *coo_graph = create_coo_matrix_with_kronecker_random_data(
			    num_vertices, num_vertices, kron_stochastic_densities);

			destroy_dense_matrix_and_detach_existing_array(kron_stochastic_densities);

			// Create the structure containing multiple views of the graph.
			// NOTE: The only view is COO
			multiformat_graph_sequential = initialize_multiformat_graph(coo_graph);

			// Create the pagerank data structure
			pagerank_data_sequential = initialize_pagerank_data(num_vertices, num_iterations);

			// Do a create and copy
			pagerank_data_sequential_ref = clone_pagerank_data(pagerank_data_sequential);
			pagerank_data_sequential_tst = clone_pagerank_data(pagerank_data_sequential);
		}
		else
		{ /* all other nodes. */
		}

		// NOTE: We are going to only copy the pointer because this data should
		// be immutable.
		multiformat_graph_sequential_ref = multiformat_graph_sequential;
		multiformat_graph_sequential_tst = multiformat_graph_sequential;

		/*
		  Run the reference
		*/
		multiformat_graph_t *multiformat_graph_distributed_ref;
		pagerank_data_t *pagerank_data_distributed_ref;

		// Allocate distributed buffers for the reference
		DISTRIBUTED_ALLOCATE_NAME_REF(num_vertices, num_iterations, &multiformat_graph_distributed_ref,
					      &pagerank_data_distributed_ref);

		// Distribute the sequential buffers
		DISTRIBUTE_DATA_NAME_REF(num_vertices, num_iterations, multiformat_graph_sequential_ref,
					 pagerank_data_sequential_ref, multiformat_graph_distributed_ref,
					 pagerank_data_distributed_ref);

		// Perform the computation
		COMPUTE_NAME_REF(num_vertices, num_iterations, multiformat_graph_distributed_ref,
				 pagerank_data_distributed_ref);

		// Collect the distributed data and write it to a sequential buffer
		COLLECT_DATA_NAME_REF(num_vertices, num_iterations, pagerank_data_distributed_ref,
				      pagerank_data_sequential_ref);

		// Finally free the buffers
		DISTRIBUTED_FREE_NAME_REF(num_vertices, num_iterations, multiformat_graph_distributed_ref,
					  pagerank_data_distributed_ref);

		// run the test
		multiformat_graph_t *multiformat_graph_distributed_tst;
		pagerank_data_t *pagerank_data_distributed_tst;

		// Allocate distributed buffers for the reference
		DISTRIBUTED_ALLOCATE_NAME_TST(num_vertices, num_iterations, &multiformat_graph_distributed_tst,
					      &pagerank_data_distributed_tst);

		// Distribute the sequential buffers
		DISTRIBUTE_DATA_NAME_TST(num_vertices, num_iterations, multiformat_graph_sequential_tst,
					 pagerank_data_sequential_tst, multiformat_graph_distributed_tst,
					 pagerank_data_distributed_tst);

		// Perform the computation
		COMPUTE_NAME_TST(num_vertices, num_iterations, multiformat_graph_distributed_tst,
				 pagerank_data_distributed_tst);

		// Collect the distributed data and write it to a sequential buffer
		COLLECT_DATA_NAME_TST(num_vertices, num_iterations, pagerank_data_distributed_tst,
				      pagerank_data_sequential_tst);

		// Finally free the buffers
		DISTRIBUTED_FREE_NAME_TST(num_vertices, num_iterations, multiformat_graph_distributed_tst,
					  pagerank_data_distributed_tst);

		// We only need to verify the results sequentially
		if (rid == 0)
		{
			/* root node */
			float res = max_pair_wise_diff(num_vertices, 1, 1, 1, pagerank_data_sequential_ref->y,
						       pagerank_data_sequential_tst->y);

			fprintf(result_file, "%i,%i,%i,%f,", num_ranks, num_vertices, num_iterations, res);

			// if our error is greater than some threshold
			if (res > ERROR_THRESHOLD)
				fprintf(result_file, "FAIL\n");
			else
				fprintf(result_file, "PASS\n");

			destroy_coo_matrix(multiformat_graph_sequential->graph_view_coo);
			destroy_multiformat_graph(multiformat_graph_sequential);

			destroy_pagerank_data(pagerank_data_sequential);
			destroy_pagerank_data(pagerank_data_sequential_ref);
			destroy_pagerank_data(pagerank_data_sequential_tst);
		}
		else
		{ /* all other nodes */
		}
	}

	// Only needs to be done by root node
	if (rid == 0)
	{
		/* root node */
		fclose(result_file);
	}
	else
	{ /* all other nodes */
	}

	MPI_Finalize();
}
