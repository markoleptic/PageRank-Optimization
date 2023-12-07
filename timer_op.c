/*
  Timer harness for running a "function under test" for num_runs number of
  runs.


*/
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "helper.h"
#include "pagerank.h"
#include "sparse.h"
#include "timer.h"

// Function under test
extern void COMPUTE_NAME_REF(int m0, int k0, multiformat_graph_t *multiformat_graph_distributed,
			     pagerank_data_t *pagerank_data_distributed);

extern void COMPUTE_NAME_TST(int m0, int k0, multiformat_graph_t *multiformat_graph_distributed,
			     pagerank_data_t *pagerank_data_distributed);

extern void DISTRIBUTED_ALLOCATE_NAME_REF(int m0, int k0, multiformat_graph_t **multiformat_graph_distributed,
					  pagerank_data_t **pagerank_data_distributed);

extern void DISTRIBUTED_ALLOCATE_NAME_TST(int m0, int k0, multiformat_graph_t **multiformat_graph_distributed,
					  pagerank_data_t **pagerank_data_distributed);

extern void DISTRIBUTE_DATA_NAME_REF(int m0, int k0, multiformat_graph_t *multiformat_graph_sequential,
				     pagerank_data_t *pagerank_data_sequential,
				     multiformat_graph_t *multiformat_graph_distributed,
				     pagerank_data_t *pagerank_data_distributed);

extern void DISTRIBUTE_DATA_NAME_TST(int m0, int k0, multiformat_graph_t *multiformat_graph_sequential,
				     pagerank_data_t *pagerank_data_sequential,
				     multiformat_graph_t *multiformat_graph_distributed,
				     pagerank_data_t *pagerank_data_distributed);

extern void COLLECT_DATA_NAME_REF(int m0, int k0, pagerank_data_t *pagerank_data_distributed,
				  pagerank_data_t *pagerank_data_sequential);

extern void COLLECT_DATA_NAME_TST(int m0, int k0, pagerank_data_t *pagerank_data_distributed,
				  pagerank_data_t *pagerank_data_sequential);

extern void DISTRIBUTED_FREE_NAME_REF(int m0, int k0, multiformat_graph_t *multiformat_graph_distributed,
				      pagerank_data_t *pagerank_data_distributed);

extern void DISTRIBUTED_FREE_NAME_TST(int m0, int k0, multiformat_graph_t *multiformat_graph_distributed,
				      pagerank_data_t *pagerank_data_distributed);

long pick_min_in_list(int num_trials, long *results)
{
	long current_min = LONG_MAX;

	for (int i = 0; i < num_trials; ++i)
		if (results[i] < current_min)
			current_min = results[i];

	return current_min;
}

void flush_cache()
{

	int size = 1024 * 1024 * 8;

	int *buff = (int *)malloc(sizeof(int) * size);
	int i, result = 0;
	volatile int sink;
	for (i = 0; i < size; i++)
		result += buff[i];
	sink = result; /* So the compiler doesn't optimize away the loop */

	free(buff);
}

void time_function_under_test(int num_trials, int num_runs_per_trial,
			      long *results, // results from each trial
			      int m0, int k0, multiformat_graph_t *multiformat_graph_distributed,
			      pagerank_data_t *pagerank_data_distributed)
{
	// Initialize the start and stop variables.
	TIMER_INIT_COUNTERS(stop, start);

	// Click the timer a few times so the subsequent measurements are more accurate
	MPI_Barrier(MPI_COMM_WORLD);
	TIMER_WARMUP(stop, start);

	// flush the cache
	flush_cache();
	MPI_Barrier(MPI_COMM_WORLD);

	for (int trial = 0; trial < num_trials; ++trial)
	{

		/*
		  Time code.
		*/
		// start timer
		TIMER_GET_CLOCK(start);

		////////////////////////
		// Benchmark the code //
		////////////////////////

		for (int runs = 0; runs < num_runs_per_trial; ++runs)
		{
			COMPUTE_NAME_TST(m0, k0, multiformat_graph_distributed, pagerank_data_distributed);
		}

		////////////////////////
		// End Benchmark      //
		////////////////////////

		// stop timer
		TIMER_GET_CLOCK(stop);

		// subtract the start time from the stop time
		TIMER_GET_DIFF(start, stop, results[trial])
	}
}

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

	int num_trials = 1;
	int num_runs_per_trial = 1;

	// Problem parameters
	int min_size;
	int max_size;
	int step_size;

	int in_m0;
	int in_k0;

	// Get command line arguments
	if (argc == 1)
	{
		min_size = 16;
		max_size = 256;
		step_size = 16;

		// defaults
		in_m0 = 1;
		in_k0 = -3;

		// default to printing to stdout
		result_file = stdout;
	}
	else if (argc == 5 + 1 || argc == 6 + 1)
	{
		min_size = atoi(argv[1]);
		max_size = atoi(argv[2]);
		step_size = atoi(argv[3]);

		in_m0 = atoi(argv[4]);
		in_k0 = atoi(argv[5]);

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
		printf("usage: %s min max step m0 k0 [filename]\n", argv[0]);
		exit(1);
	}

	// Print out the first line of the output in csv format
	if (rid == 0)
	{
		/*root node */
		fprintf(result_file, "num_ranks,m0,k0,result\n");
	}
	else
	{ /* all other nodes*/
	}

	for (int p = min_size; p < max_size; p += step_size)
	{

		// input sizes
		int m0 = scale_p_on_pos_ret_v_on_neg(p, in_m0);
		int k0 = scale_p_on_pos_ret_v_on_neg(p, in_k0);

		// These structs contain different views of the graph
		multiformat_graph_t *multiformat_graph_sequential;
		multiformat_graph_t *multiformat_graph_sequential_tst;

		// These structs hold the working dataset and the results
		pagerank_data_t *pagerank_data_sequential;
		pagerank_data_t *pagerank_data_sequential_tst;

		if (rid == 0)
		{ /* root node */
			// Create a stochastic kronecker graph to approximate a web graph.
			float kron_stochastic_buf[] = {0.91f, 0.75f, 0.63f, 0.52f};
			dense_matrix_t *kron_stochastic_densities =
			    create_dense_matrix_and_attach_existing_array(2, 2, 2, 1, kron_stochastic_buf);

			coo_matrix_t *coo_graph =
			    create_coo_matrix_with_kronecker_random_data(m0, m0, kron_stochastic_densities);

			destroy_dense_matrix_and_detach_existing_array(kron_stochastic_densities);

			// Create the structure containing multiple views of the graph.
			// NOTE: The only view is COO
			multiformat_graph_sequential = initialize_multiformat_graph(coo_graph);

			// Create the pagerank data structure
			pagerank_data_sequential = initialize_pagerank_data(m0, k0);

			// create and copy
			pagerank_data_sequential_tst = clone_pagerank_data(pagerank_data_sequential);
		}
		else
		{ /* all other nodes. */
		}

		// run the test
		multiformat_graph_sequential_tst = multiformat_graph_sequential;

		multiformat_graph_t *multiformat_graph_distributed_tst;
		pagerank_data_t *pagerank_data_distributed_tst;

		// Allocate distributed buffers for the reference
		DISTRIBUTED_ALLOCATE_NAME_TST(m0, k0, &multiformat_graph_distributed_tst,
					      &pagerank_data_distributed_tst);

		// Distribute the sequential buffers
		DISTRIBUTE_DATA_NAME_TST(m0, k0, multiformat_graph_sequential_tst, pagerank_data_sequential_tst,
					 multiformat_graph_distributed_tst, pagerank_data_distributed_tst);

		// Perform the computation
		long *results = (long *)malloc(sizeof(long) * num_trials);

		time_function_under_test(num_trials, num_runs_per_trial,
					 results, // results from each trial
					 m0, k0, multiformat_graph_distributed_tst, pagerank_data_distributed_tst);

		long min_res = pick_min_in_list(num_trials, results);
		free(results);

		// Collect the distributed data and write it to a sequential buffer
		COLLECT_DATA_NAME_TST(m0, k0, pagerank_data_distributed_tst, pagerank_data_sequential_tst);

		// Finally free the buffers
		DISTRIBUTED_FREE_NAME_TST(m0, k0, multiformat_graph_distributed_tst, pagerank_data_distributed_tst);

		if (rid == 0)
		{
			/* root node */
			float nanoseconds = ((float)min_res) / (num_runs_per_trial);

			// Number of floating point operations
			long num_flops = 2 * k0 * multiformat_graph_sequential->nnz;

			// This gives us throughput as GFLOP/s
			float throughput = num_flops / nanoseconds;

			fprintf(result_file, "%i,%i,%i,%2.2f\n", num_ranks, m0, k0, throughput);

			// Free the sequential buffers
			destroy_coo_matrix(multiformat_graph_sequential->graph_view_coo);
			destroy_multiformat_graph(multiformat_graph_sequential);

			destroy_pagerank_data(pagerank_data_sequential);
			destroy_pagerank_data(pagerank_data_sequential_tst);
		}
		else
		{ /* all other nodes */
		}
	}

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
