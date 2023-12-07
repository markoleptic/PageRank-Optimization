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
  This operation performs a matrix vector multiplication, where the matrix is
  sparse and the vectors are dense. This implementation is using the Coordinate (COO)
  format, but it can (and should be) changed to a format that best fits the data
  and the hardware (for example CSR, CSC, BCSR, etc).

*/
static void matvec( multiformat_graph_t *multiformat_graph_distributed,
 		     pagerank_data_t *pagerank_data_distributed )
{
  /*
    STUDENT_TODO: If this is not the baseline feel free to use a different
                  sparse format (COO,CSR,CSC,BCSR) for the matrix-vector 
		  product. This operation is also where you will be doing
		  a lot of your optimizations and parallel transformations.
  */

  // Note: this is a Coordinate (COO) implementation of matrix-vector multiply
  //       but it could be any other format.
  for( int cur_pos = 0;
       cur_pos < multiformat_graph_distributed->graph_view_csr->nnz;
       ++cur_pos )
    {
      int i   = multiformat_graph_distributed->graph_view_csr->row_idx[cur_pos];
      int j   = multiformat_graph_distributed->graph_view_csr->col_idx[cur_pos];
      float  val = multiformat_graph_distributed->graph_view_csr->values[cur_pos];

      pagerank_data_distributed->y[i] += val * pagerank_data_distributed->x[j];
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
static void page_rank( multiformat_graph_t *multiformat_graph_distributed,
		        pagerank_data_t *pagerank_data_distributed )
{
  
  ////////////////////////////////////
  // The bulk of the computation    //
  // This is what will be optimized //
  ////////////////////////////////////
  /*
    STUDENT_TODO: If this is not the baseline feel free to use a different
                  sparse format (COO,CSR,CSC,BCSR) for the matrix-vector 
		  product. This operation is also where you will be doing
		  a lot of your optimizations and parallel transformations.
  */
  for( int t = 0;
       t < pagerank_data_distributed->num_iterations;
       ++t )
    {
      ////////////////////////////////////////////////////////////////////////////
      // Ping pong the buffers by changing the pointers                         //
      // when t is even x[0..m-1] = buff[0..m-1] and y[0..m-1] = buff[m..2m-1]  //
      // when t is odd  x[0..m-1] = buff[m..2m-1] and y[0..m-1] = buff[0..m-1]  //
      ////////////////////////////////////////////////////////////////////////////
      pagerank_data_distributed->x =
	&(pagerank_data_distributed->buff[((t+0)%2)*multiformat_graph_distributed->m]); 
      pagerank_data_distributed->y =
	&(pagerank_data_distributed->buff[((t+1)%2)*multiformat_graph_distributed->m]); 
      
      // zero out the output y
      for( int i = 0; i < multiformat_graph_distributed->m; ++i )
	pagerank_data_distributed->y[i] = 0.0f;

      ////////////////////////////
      // Matrix vector multiply //
      ////////////////////////////
      matvec( multiformat_graph_distributed,
	      pagerank_data_distributed );
      


      #if DEBUG
      float res = max_pair_wise_diff_vect(multiformat_graph_distributed->m,
				      pagerank_data_distributed->x,
				      pagerank_data_distributed->y);
      
      printf("diff[%i]: %f\n",t, res);
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

static csr_matrix_t* create_csr_matrix_from_coo_matrix( coo_matrix_t *src_coo ){
    
    csr_matrix_t *dst_csr = (csr_matrix_t *)malloc(sizeof(csr_matrix_t));

    // copy the metadata/
  dst_csr->m   = src_coo->m;
  dst_csr->n   = src_coo->n;
  dst_csr->nnz = src_coo->nnz;

  // Allocated the buffers.
  dst_csr->row_idx = (int *)malloc(sizeof(int)*src_coo->nnz);
  dst_csr->col_idx = (int *)malloc(sizeof(int)*src_coo->nnz);
  dst_csr->values  = (float *)malloc(sizeof(float)*src_coo->nnz);

  // Copy the buffers.
  memcpy(dst_csr->row_idx, src_coo->row_idx, sizeof(int)*src_coo->nnz);
  memcpy(dst_csr->col_idx, src_coo->col_idx, sizeof(int)*src_coo->nnz);
  memcpy(dst_csr->values, src_coo->values, sizeof(float)*src_coo->nnz);

    // // Copy the data
    // dst_csr->m = src_coo->m;
    // dst_csr->n = src_coo->n;
    // dst_csr->nnz = src_coo->nnz;

    // for (int i = 0; i < dst_csr->nnz; ++i) {
    //   dst_csr->row_idx[i] = 0;
    //   dst_csr->col_idx[i] = 0;
    //   dst_csr->values[i] = 0;
    // }

    // for (int i = 0; i < dst_csr->nnz; ++i) {
    //   dst_csr->values[i] = src_coo->values[i];
    //   dst_csr->col_idx[i] = src_coo->col_idx[i];
    //   dst_csr->row_idx[src_coo[i+1]]++;
    // }

    // for (int i = 0; i < src_coo->m; ++i) {
    //     dst_csr->row_idx[i + 1] += dst_csr[i];
    // }

    return dst_csr;    
} 

// given an input matrix in coordinate format this produces a new coordinate matrix.
static coo_matrix_t* create_coo_matrix_from_coo_matrix( coo_matrix_t *src_coo )
{
  // allocate the structure.
  coo_matrix_t *dst_coo = (coo_matrix_t *)malloc(sizeof(coo_matrix_t));

  // copy the metadata
  dst_coo->m   = src_coo->m;
  dst_coo->n   = src_coo->n;
  dst_coo->nnz = src_coo->nnz;

  // Allocated the buffers.
  dst_coo->row_idx = (int *)malloc(sizeof(int)*src_coo->nnz);
  dst_coo->col_idx = (int *)malloc(sizeof(int)*src_coo->nnz);
  dst_coo->values  = (float *)malloc(sizeof(float)*src_coo->nnz);

  // Copy the buffers.
  memcpy(dst_coo->row_idx, src_coo->row_idx, sizeof(int)*src_coo->nnz);
  memcpy(dst_coo->col_idx, src_coo->col_idx, sizeof(int)*src_coo->nnz);
  memcpy(dst_coo->values, src_coo->values, sizeof(float)*src_coo->nnz);

  // return the newly created and filled structure.
  return dst_coo;
}


void COMPUTE_NAME( int num_vertices, int num_iterations,
		   multiformat_graph_t *multiformat_graph_distributed,
		   pagerank_data_t *pagerank_data_distributed )

{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {

      /////////////
      // COMPUTE //
      /////////////
      page_rank( multiformat_graph_distributed,
		 pagerank_data_distributed );
     
    }
  else
    {
    }
}


// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME( int num_vertices, int num_iterations,
				multiformat_graph_t **multiformat_graph_distributed,
				pagerank_data_t     **pagerank_data_distributed )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  /*
    STUDENT_TODO: If this is not the baseline then you might want to change this
                  especially if you intend on doing distributed memory work.

   */
  
  if(rid == root_rid )
    {
      *multiformat_graph_distributed =
	(multiformat_graph_t *)malloc(sizeof(multiformat_graph_t));

      *pagerank_data_distributed = (pagerank_data_t *)malloc(sizeof(pagerank_data_t));
      (*pagerank_data_distributed)->buff = (float *)calloc(sizeof(float),
							   num_vertices*2);

    }
  else
    {
    }
}


/*
  This takes the sequential data located on the root node and distributes it
  to the nodes that will need it. 
*/
void DISTRIBUTE_DATA_NAME( int num_vertices, int num_iterations,
			   multiformat_graph_t *multiformat_graph_sequential,
			   pagerank_data_t *pagerank_data_sequential,
			   multiformat_graph_t *multiformat_graph_distributed,
			   pagerank_data_t *pagerank_data_distributed )
{

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {
      
      ////////////////
      // DISTRIBUTE //
      ////////////////



      //////////////////////////////      
      // Copy the pagerank struct //
      //////////////////////////////      
      int buff_size = sizeof(float)*2*pagerank_data_sequential->num_vertices;

      memcpy(pagerank_data_distributed->buff,
	     pagerank_data_sequential->buff,
	     buff_size);
           
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
      multiformat_graph_distributed->m   = multiformat_graph_sequential->m;
      multiformat_graph_distributed->n   = multiformat_graph_sequential->n;
      multiformat_graph_distributed->nnz = multiformat_graph_sequential->nnz;

      // Copy over the degree of the nodes
      multiformat_graph_distributed->degree =
	(long *)malloc(sizeof(long)*multiformat_graph_sequential->m);
      
      memcpy(multiformat_graph_distributed->degree,
	     multiformat_graph_sequential->degree,
	     sizeof(long)*multiformat_graph_sequential->m );
      /*
	STUDENT_TODO: ... if you want ..

	If this is not the baseline, then this is a good place to convert the
	graph from Coordinate format (COO) to a different sparse matrix format.
	Just make sure to free up any new structures you create when you are done.
      */
     
      // multiformat_graph_distributed->graph_view_coo =
	// create_coo_matrix_from_coo_matrix( multiformat_graph_sequential->graph_view_coo );

      multiformat_graph_distributed->graph_view_csr = create_csr_matrix_from_coo_matrix(multiformat_graph_sequential->graph_view_coo);

      /*	
         STUDENT_TODO: if this is not the baseline and it makes sense for your
                       implementation. (NOTE: you have done part of this in a hw).

         csr_matrix_t* create_csr_matrix_from_coo_matrix( coo_matrix_t *src_coo );
         csc_matrix_t* create_csc_matrix_from_coo_matrix( coo_matrix_t *src_coo );
         bcsr_matrix_t* create_bcsr_matrix_from_coo_matrix( coo_matrix_t *src_coo );
      */
      // multiformat_graph_distributed->graph_view_csr  = NULL;
      multiformat_graph_distributed->graph_view_csc  = NULL;
      multiformat_graph_distributed->graph_view_bcsr = NULL;

    }
  else
    {
    }
  
}

/*
  This function gathers the resulting data distributed across potentially many
  nodes and copies it to its sequential counterpart.
*/

void COLLECT_DATA_NAME( int num_vertices, int num_iterations,
			  pagerank_data_t *pagerank_data_distributed,
			  pagerank_data_t *pagerank_data_sequential )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {
      /*
	STUDENT_TODO: if this is not the baseline and you have
	              used distributed memory then you might need
		      to do something here. However, because this
		      is iteratively computing on the data you might
		      need to be exchanging data at every iteration.
       */
      copy_pagerank_data(pagerank_data_sequential,
			 pagerank_data_distributed );
    }
  else
    {
    }
  
}




void DISTRIBUTED_FREE_NAME( int num_vertices, int num_iterations,
			      multiformat_graph_t *multiformat_graph_distributed,
			      pagerank_data_t *pagerank_data_distributed )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {
      //////////
      // FREE //
      //////////
      free(pagerank_data_distributed->buff );
      free(pagerank_data_distributed);

      /*
	STUDENT_TODO: if this is not the baseline and you are
                      using something other than COO, then you
		      might need to free that data up here.
      */
      //destroy_coo_matrix(multiformat_graph_distributed->graph_view_coo);
      destroy_csr_matrix(multiformat_graph_distributed->graph_view_csr);
      // destroy_csc_matrix(multiformat_graph_distributed->graph_view_csc);
      // destroy_bcsr_matrix(multiformat_graph_distributed->graph_view_bcsr);
      
      free(multiformat_graph_distributed->degree);
      free(multiformat_graph_distributed);

    }
  else
    {
    }

}

