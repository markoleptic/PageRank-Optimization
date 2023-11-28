#ifndef SPARSE_H
#define SPARSE_H

/*
  Here are a collection of matrix data formats and their associated routines.

  TODO: I need to better encapsulate these structures and limit access to their
        contents outside of their functions.


  - richard.m.veras@ou.edu
*/



#include <stdlib.h>
#include <stdio.h>


typedef struct dense_matrix_ts
{
  int m;
  int n;
  int cs;
  int rs;
  float *values;
} dense_matrix_t;

typedef struct coo_matrix_ts
{
  int m;
  int n;
  int nnz;
  int   *col_idx;
  int   *row_idx;
  float *values;
} coo_matrix_t;


typedef struct csr_matrix_ts
{
  int m;
  int n;
  int nnz;
  int   *row_idx;// starting index of each row in col_idx and values
  int   *col_idx; // NNZ total number of indices
  float *values;  // NNZ total number of values
} csr_matrix_t;

typedef struct csc_matrix_ts
{
  int m;
  int n;
  int nnz;
  int   *col_idx;// starting index of each col in row_idx and values
  int   *row_idx; // NNZ total number of indices
  float *values;  // NNZ total number of values
} csc_matrix_t;

// Blocked Compressed Sparse Row
typedef struct bcsr_matrix_ts
{
  int m; // number of rows
  int n; // number of cols

  int mb; // Block size in m dimension
  int nb; // Block size in n dimension

  int bcs; // Block column stride
  int brs; // Block row stride
  
  int nnz_blocks; // non zero blocks
  int   *block_row_idx;// starting index of each row in col_idx and values
  int   *block_col_idx; // NNZ total number of indices
  float *block_values;  // NNZ total number of values
} bcsr_matrix_t;





// TODO: bcsrdense?
// This is probably called something else
typedef struct bcsrdense_matrix_ts
{
  int m; // number of rows
  int n; // number of cols

  int omb; // outer block size in m dimension
  int onb; // outer block size in n dimension

  int obcs; // outer block column stride
  int obrs; // outer block row stride

  int imb; // inner block size in m dimension
  int inb; // inner block size in n dimension

  int ibcs; // inner block column stride
  int ibrs; // inner block row stride

  int *nnz_blocks; // List of nnz for each (m/mb) x (n/nb)  
  int   **block_row_idx;// starting index of each row in col_idx and values
  int   **block_col_idx; // NNZ total number of indices
  float **block_values;  // NNZ total number of values
} bcsrdense_matrix_t;



void pretty_print_dense_matrix( dense_matrix_t *A );
int count_number_of_nonzero_blocks_in_dense_matrix( int mb, int nb,
						    dense_matrix_t* A );
dense_matrix_t* create_dense_matrix_and_attach_existing_array(
							      int m, // rows
							      int n, // columns
							      int cs, // column stride
							      int rs, // row stride
							      float *values // values
							      );


// Use this destroy if you created the dense matrix by attaching an existing
// array of values.
void destroy_dense_matrix_and_detach_existing_array( dense_matrix_t* A );
void destroy_dense_matrix( dense_matrix_t* A );

/*
  Compressed Sparse Row (CSR/CRS/Yale)
*/
void pretty_print_csr_matrix(csr_matrix_t *A);
dense_matrix_t *create_dense_matrix_from_csr_matrix(csr_matrix_t *src_csr);
void destroy_csr_matrix(csr_matrix_t *A);
void copy_dense_matrix_to_csr_matrix( dense_matrix_t* src_dense,
				      csr_matrix_t *dst_csr );
csr_matrix_t *create_csr_matrix_from_dense_matrix(dense_matrix_t* A_dense);

/*
  Compressed Sparse Column (CSC)
*/

void pretty_print_csc_matrix(csc_matrix_t *A);
dense_matrix_t *create_dense_matrix_from_csc_matrix(csc_matrix_t *src_csc);
void destroy_csc_matrix(csc_matrix_t *A);
void copy_dense_matrix_to_csc_matrix( dense_matrix_t* src_dense,
				      csc_matrix_t *dst_csc );
csc_matrix_t *create_csc_matrix_from_dense_matrix(dense_matrix_t* A_dense);

/*
  Coordinate Storage Format (COO)
*/

dense_matrix_t *create_dense_matrix_from_coo_matrix(coo_matrix_t *src_coo);
void pretty_print_coo_matrix(coo_matrix_t *A);
void copy_dense_matrix_to_coo_matrix( dense_matrix_t* src_dense,
				      coo_matrix_t *dst_coo );
coo_matrix_t *create_coo_matrix_from_dense_matrix(dense_matrix_t* A_dense);
void destroy_coo_matrix(coo_matrix_t *A);

/*
  Blocked Compressed Sparse Row (BCSR)
*/
void pretty_print_bcsr_matrix(bcsr_matrix_t *A);
dense_matrix_t *create_dense_matrix_from_bcsr_matrix(bcsr_matrix_t *src_bcsr);
void destroy_bcsr_matrix(bcsr_matrix_t *A);

void copy_dense_matrix_to_bcsr_matrix( int mb, int nb,
				       int bcs, int brs,
				       dense_matrix_t* src_dense,
				       bcsr_matrix_t *dst_bcsr );

bcsr_matrix_t *create_bcsr_matrix_from_dense_matrix(int mb, int nb,
						    int bcs, int brs,
						    dense_matrix_t* A_dense);


/*
  CSRDense or whatever
*/
#if 1
// TODO: csrdense?
// Dense on the outside csr in the blocks
// This is probably called something else
typedef struct csrdense_matrix_ts
{
  int m; // number of rows
  int n; // number of cols

  int omb; // outer block size in m dimension
  int onb; // outer block size in n dimension

  int obcs; // outer block column stride
  int obrs; // outer block row stride

  int *nnz; // List of nnz for each (m/mb) x (n/nb)  
  int   **row_idx;// starting index of each row in col_idx and values
  int   **col_idx; // NNZ total number of indices
  float **values;  // NNZ total number of values
} csrdense_matrix_t;

#endif


#if 0
void pretty_print_csrdense_matrix(csrdense_matrix_t *A);
dense_matrix_t *create_dense_matrix_from_csrdense_matrix(csrdense_matrix_t *src_csrdense);
void destroy_csrdense_matrix(csrdense_matrix_t *A);
void copy_dense_matrix_to_csrdense_matrix( dense_matrix_t* src_dense,
					   csrdense_matrix_t *dst_csrdense );
csr_matrix_t *create_csr_matrix_from_dense_matrix(dense_matrix_t* A_dense);
#endif


#endif /* SPARSE_H */
