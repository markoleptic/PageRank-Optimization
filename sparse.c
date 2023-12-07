/*
  Here are a collection of matrix data formats and their associated routines.

  TODO: I need to better encapsulate these structures and limit access to their
	contents outside of their functions.


  - richard.m.veras@ou.edu
*/

#include <stdio.h>
#include <stdlib.h>

#include "sparse.h"

void pretty_print_dense_matrix(dense_matrix_t *A)
{

	int m = A->m;
	int n = A->n;
	int cs = A->cs;
	int rs = A->rs;
	float *values = A->values;

	printf("    ");
	for (int j = 0; j < n; ++j)
		printf("__%2i", j);
	printf("\n");

	for (int i = 0; i < m; ++i)
	{
		printf("%2i |", i);
		for (int j = 0; j < n; ++j)
		{
			float val = values[i * cs + j * rs];

			if (val < 0.0f)
			{
				printf("  x,");
			}
			else if (val == 0.0f)
			{
				printf("  .,");
			}
			else
			{
				printf("%3.f,", val);
			}
		}
		printf("\n");
	}
	printf("\n");
}

int count_number_of_nonzero_blocks_in_dense_matrix(int mb, int nb, dense_matrix_t *A)
{
	int m = A->m;
	int n = A->n;
	int cs = A->cs;
	int rs = A->rs;
	float *values = A->values;

	if (m % mb != 0 || n % nb != 0)
	{
		// TODO: deal with fringe
		printf("Can't currently deal with non multiples of block sizes.\n");
		exit(1);
	}

	int nnz_blocks = 0;
	for (int io = 0; io < m; io += mb)
		for (int jo = 0; jo < n; jo += nb)
		{
			// Check the current block for non-zeros
			int nnz = 0;
			for (int ii = 0; ii < mb; ++ii)
				for (int ji = 0; ji < nb; ++ji)
				{
					float val = values[(io + ii) * cs + (jo + ji) * rs];
					if (val != 0.0f)
						nnz++;
				}

			// This block had at least one non-zero
			if (nnz > 0)
				nnz_blocks++;
		}

	return nnz_blocks;
}

dense_matrix_t *create_dense_matrix_and_attach_existing_array(int m,	    // rows
							      int n,	    // columns
							      int cs,	    // column stride
							      int rs,	    // row stride
							      float *values // values
)
{
	dense_matrix_t *A = (dense_matrix_t *)malloc(sizeof(dense_matrix_t));

	A->m = m;
	A->n = n;
	A->rs = rs;
	A->cs = cs;
	A->values = values;

	return A;
}

// Use this destroy if you created the dense matrix by attaching an existing
// array of values.
void destroy_dense_matrix_and_detach_existing_array(dense_matrix_t *A) { free(A); }

void destroy_dense_matrix(dense_matrix_t *A)
{
	free(A->values);
	free(A);
}

/*
  Compressed Sparse Row (CSR/CRS/Yale)
*/
void pretty_print_csr_matrix(csr_matrix_t *A)
{
	printf("idx --->  ");
	for (int idx = 0; idx < A->nnz; ++idx)
		printf("__%2i", idx);
	printf("\n");

	printf("row_idx = [");
	for (int i = 0; i < A->m + 1; ++i)
		printf("%3i,", A->row_idx[i]);
	printf("]\n");

	printf("col_idx = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
		printf("%3i,", A->col_idx[cur_nnz]);
	printf("]\n");

	printf("values  = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
		printf("%3.f,", A->values[cur_nnz]);
	printf("]\n");
}

dense_matrix_t *create_dense_matrix_from_csr_matrix(csr_matrix_t *src_csr)
{
	// allocate the actual structure with the metadata
	dense_matrix_t *dst_dense = (dense_matrix_t *)malloc(sizeof(dense_matrix_t));

	// Copy the metadata over
	dst_dense->m = src_csr->m;
	dst_dense->n = src_csr->n;

	// Default to column major
	dst_dense->rs = dst_dense->m;
	dst_dense->cs = 1;

	// Create a buffer for the values that is MxN
	dst_dense->values = (float *)malloc(sizeof(float) * dst_dense->m * dst_dense->n);

	// Zero all the values out
	for (int i = 0; i < dst_dense->m; ++i)
		for (int j = 0; j < dst_dense->n; ++j)
			dst_dense->values[i * dst_dense->cs + j * dst_dense->rs] = 0.0f;

	// Now we copy the values from the csr matrix to the dense one.
	for (int i = 0; i < src_csr->m; ++i)
		// This is the only tricky indirection part because we need the segment
		// of column indices that correspond to the nonzeros in the row.
		for (int cur_nz = src_csr->row_idx[i]; cur_nz < src_csr->row_idx[i + 1]; ++cur_nz)
		{
			int j = src_csr->col_idx[cur_nz];
			float val = src_csr->values[cur_nz];
			dst_dense->values[i * dst_dense->cs + j * dst_dense->rs] = val;
		}

	return dst_dense;
}

void destroy_csr_matrix(csr_matrix_t *A)
{
	// Free the buffers
	free(A->col_idx);
	free(A->row_idx);
	free(A->values);

	// Free the structure
	free(A);
}

void copy_dense_matrix_to_csr_matrix(dense_matrix_t *src_dense, csr_matrix_t *dst_csr)
{
	int m = src_dense->m;
	int n = src_dense->n;
	int cs = src_dense->cs;
	int rs = src_dense->rs;
	float *src_values = src_dense->values;

	int cur_nnz = 0;
	for (int i = 0; i < m; ++i)
	{
		// We need to know where each row starts in in the col_idx and val_idx
		dst_csr->row_idx[i] = cur_nnz;
		for (int j = 0; j < n; ++j)
		{
			float val = src_values[i * cs + j * rs];

			if (val != 0.0f)
			{
				dst_csr->col_idx[cur_nnz] = j;
				dst_csr->values[cur_nnz] = val;

				cur_nnz++;
			}
		}
	}
	dst_csr->row_idx[m] = cur_nnz;
}

csr_matrix_t *create_csr_matrix_from_dense_matrix(dense_matrix_t *A_dense)
{
	// First pass count all of the non-zeros
	int mb = 1;
	int nb = 1;
	int nnz = count_number_of_nonzero_blocks_in_dense_matrix(mb, nb, A_dense);

	// Allocate the buffers
	csr_matrix_t *A_csr = (csr_matrix_t *)malloc(sizeof(csr_matrix_t));

	// assign the values for the metadata
	A_csr->nnz = nnz;
	A_csr->m = A_dense->m;
	A_csr->n = A_dense->n;

	// allocate the arrays used by csr
	A_csr->row_idx = (int *)malloc(sizeof(int) * (A_csr->m + 1));
	A_csr->col_idx = (int *)malloc(sizeof(int) * nnz);
	A_csr->values = (float *)malloc(sizeof(float) * nnz);

	// Second pass to copy the values
	copy_dense_matrix_to_csr_matrix(A_dense, A_csr);

	return A_csr;
}

/*
  Compressed Sparse Column (CSC)
*/

void pretty_print_csc_matrix(csc_matrix_t *A)
{
	printf("idx --->  ");
	for (int idx = 0; idx < A->nnz; ++idx)
		printf("__%2i", idx);
	printf("\n");

	printf("col_idx = [");
	for (int j = 0; j < A->n + 1; ++j)
		printf("%3i,", A->col_idx[j]);
	printf("]\n");

	printf("row_idx = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
		printf("%3i,", A->row_idx[cur_nnz]);
	printf("]\n");

	printf("values  = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
		printf("%3.f,", A->values[cur_nnz]);
	printf("]\n");
}

dense_matrix_t *create_dense_matrix_from_csc_matrix(csc_matrix_t *src_csc)
{
	// allocate the actual structure with the metadata
	dense_matrix_t *dst_dense = (dense_matrix_t *)malloc(sizeof(dense_matrix_t));

	// Copy the metadata over
	dst_dense->m = src_csc->m;
	dst_dense->n = src_csc->n;

	// Default to column major
	dst_dense->rs = dst_dense->m;
	dst_dense->cs = 1;

	// Create a buffer for the values that is MxN
	dst_dense->values = (float *)malloc(sizeof(float) * dst_dense->m * dst_dense->n);

	// Zero all the values out
	for (int i = 0; i < dst_dense->m; ++i)
		for (int j = 0; j < dst_dense->n; ++j)
			dst_dense->values[i * dst_dense->cs + j * dst_dense->rs] = 0.0f;

	// Now we copy the values from the csc matrix to the dense one.
	for (int j = 0; j < src_csc->n; ++j)
		// This is the only tricky indirection part because we need the segment
		// of row indices that correspond to the nonzeros in the row.
		for (int cur_nz = src_csc->col_idx[j]; cur_nz < src_csc->col_idx[j + 1]; ++cur_nz)
		{
			int i = src_csc->row_idx[cur_nz];
			float val = src_csc->values[cur_nz];
			dst_dense->values[i * dst_dense->cs + j * dst_dense->rs] = val;
		}

	return dst_dense;
}

void destroy_csc_matrix(csc_matrix_t *A)
{
	// Free the buffers
	free(A->col_idx);
	free(A->row_idx);
	free(A->values);

	// Free the structure
	free(A);
}

void copy_dense_matrix_to_csc_matrix(dense_matrix_t *src_dense, csc_matrix_t *dst_csc)
{
	int m = src_dense->m;
	int n = src_dense->n;
	int cs = src_dense->cs;
	int rs = src_dense->rs;
	float *src_values = src_dense->values;

	int cur_nnz = 0;
	for (int j = 0; j < n; ++j)
	{
		// We need to know where each row starts in in the col_idx and val_idx
		dst_csc->col_idx[j] = cur_nnz;
		for (int i = 0; i < m; ++i)
		{
			float val = src_values[i * cs + j * rs];

			if (val != 0.0f)
			{
				dst_csc->row_idx[cur_nnz] = i;
				dst_csc->values[cur_nnz] = val;

				cur_nnz++;
			}
		}
	}
	dst_csc->col_idx[n] = cur_nnz;
}

csc_matrix_t *create_csc_matrix_from_dense_matrix(dense_matrix_t *A_dense)
{
	// First pass count all of the non-zeros
	int mb = 1;
	int nb = 1;
	int nnz = count_number_of_nonzero_blocks_in_dense_matrix(mb, nb, A_dense);

	// Allocate the buffers
	csc_matrix_t *A_csc = (csc_matrix_t *)malloc(sizeof(csc_matrix_t));

	// assign the values for the metadata
	A_csc->nnz = nnz;
	A_csc->m = A_dense->m;
	A_csc->n = A_dense->n;

	// allocate the arrays used by csc
	A_csc->col_idx = (int *)malloc(sizeof(int) * (A_csc->n + 1));
	A_csc->row_idx = (int *)malloc(sizeof(int) * nnz);
	A_csc->values = (float *)malloc(sizeof(float) * nnz);

	// Second pass to copy the values
	copy_dense_matrix_to_csc_matrix(A_dense, A_csc);

	return A_csc;
}

/*
  Coordinate Storage Format (COO)
*/

dense_matrix_t *create_dense_matrix_from_coo_matrix(coo_matrix_t *src_coo)
{
	// allocate the actual structure with the metadata
	dense_matrix_t *dst_dense = (dense_matrix_t *)malloc(sizeof(dense_matrix_t));

	// Copy the metadata over
	dst_dense->m = src_coo->m;
	dst_dense->n = src_coo->n;

	// Default to column major
	dst_dense->rs = dst_dense->m;
	dst_dense->cs = 1;

	// Create a buffer for the values that is MxN
	dst_dense->values = (float *)malloc(sizeof(float) * dst_dense->m * dst_dense->n);

	// Zero all the values out
	for (int i = 0; i < dst_dense->m; ++i)
		for (int j = 0; j < dst_dense->n; ++j)
			dst_dense->values[i * dst_dense->cs + j * dst_dense->rs] = 0.0f;

	// Now we copy the values from the coo matrix to the dense one.
	for (int cur_nnz = 0; cur_nnz < src_coo->nnz; ++cur_nnz)
	{
		int i = src_coo->row_idx[cur_nnz];
		int j = src_coo->col_idx[cur_nnz];
		float val = src_coo->values[cur_nnz];

		dst_dense->values[i * dst_dense->cs + j * dst_dense->rs] = val;
	}

	return dst_dense;
}

void pretty_print_coo_matrix(coo_matrix_t *A)
{
	printf("idx --->  ");
	for (int idx = 0; idx < A->nnz; ++idx)
		printf("__%2i", idx);
	printf("\n");

	printf("row_idx = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
		printf("%3i,", A->row_idx[cur_nnz]);
	printf("]\n");

	printf("col_idx = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
		printf("%3i,", A->col_idx[cur_nnz]);
	printf("]\n");

	printf("values  = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
		printf("%3.2f,", A->values[cur_nnz]);
	printf("]\n");
}

void copy_dense_matrix_to_coo_matrix(dense_matrix_t *src_dense, coo_matrix_t *dst_coo)
{
	int m = src_dense->m;
	int n = src_dense->n;
	int cs = src_dense->cs;
	int rs = src_dense->rs;
	float *src_values = src_dense->values;

	int cur_nnz = 0;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			float val = src_values[i * cs + j * rs];

			if (val != 0.0f)
			{
				dst_coo->row_idx[cur_nnz] = i;
				dst_coo->col_idx[cur_nnz] = j;
				dst_coo->values[cur_nnz] = val;

				cur_nnz++;
			}
		}
	}
}

coo_matrix_t *create_coo_matrix_from_dense_matrix(dense_matrix_t *A_dense)
{

	// First pass count all of the non-zeros
	int mb = 1;
	int nb = 1;
	int nnz = count_number_of_nonzero_blocks_in_dense_matrix(mb, nb, A_dense);

	// Allocate the buffers
	coo_matrix_t *A_coo = (coo_matrix_t *)malloc(sizeof(coo_matrix_t));

	// assign the values for the metadata
	A_coo->nnz = nnz;
	A_coo->m = A_dense->m;
	A_coo->n = A_dense->n;

	// allocate the arrays used by coo
	A_coo->col_idx = (int *)malloc(sizeof(int) * nnz);
	A_coo->row_idx = (int *)malloc(sizeof(int) * nnz);
	A_coo->values = (float *)malloc(sizeof(float) * nnz);

	// Second pass to copy the values
	copy_dense_matrix_to_coo_matrix(A_dense, A_coo);

	return A_coo;
}

void destroy_coo_matrix(coo_matrix_t *A)
{
	// Free the buffers
	free(A->col_idx);
	free(A->row_idx);
	free(A->values);

	// Free the structure
	free(A);
}

/*
  Blocked Compressed Sparse Row (BCSR)
*/
void pretty_print_bcsr_matrix(bcsr_matrix_t *A)
{
	printf("row_idx = [");
	for (int i = 0; i < (A->m) / (A->mb) + 1; ++i)
	{
		printf("%3i,", A->block_row_idx[i]);
	}
	printf("]\n");

	printf("col_idx = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz_blocks; ++cur_nnz)
		printf("%3i,", A->block_col_idx[cur_nnz]);
	printf("]\n");

	printf("values  = [");
	for (int cur_nnz = 0; cur_nnz < A->nnz_blocks * A->nb * A->mb; ++cur_nnz)
	{
		// TODO: pretty print as matrix
		printf("%3.f,", A->block_values[cur_nnz]);
	}
	printf("]\n");
}

dense_matrix_t *create_dense_matrix_from_bcsr_matrix(bcsr_matrix_t *src_bcsr)
{
	// allocate the actual structure with the metadata
	dense_matrix_t *dst_dense = (dense_matrix_t *)malloc(sizeof(dense_matrix_t));

	int mb = src_bcsr->mb;
	int nb = src_bcsr->nb;
	int brs = src_bcsr->brs;
	int bcs = src_bcsr->bcs;

	// Copy the metadata over
	int m = src_bcsr->m;
	int n = src_bcsr->n;

	dst_dense->m = m;
	dst_dense->n = n;

	// Default to column major
	int rs = dst_dense->m;
	int cs = 1;

	dst_dense->rs = rs;
	dst_dense->cs = cs;

	// Create a buffer for the values that is MxN
	dst_dense->values = (float *)malloc(sizeof(float) * dst_dense->m * dst_dense->n);

	// Zero all the values out
	for (int i = 0; i < dst_dense->m; ++i)
		for (int j = 0; j < dst_dense->n; ++j)
			dst_dense->values[i * dst_dense->cs + j * dst_dense->rs] = 0.0f;

	// Now we copy the values from the csr matrix to the dense one.
	for (int io = 0; io < m; io += mb)
	{
		int io_ind = io / mb;
		// This is the only tricky indirection part because we need the segment
		// of column indices that correspond to the nonzeros in the row.
		for (int cur_nz = src_bcsr->block_row_idx[io_ind]; cur_nz < src_bcsr->block_row_idx[io_ind + 1];
		     ++cur_nz)
		{
			int jo = src_bcsr->block_col_idx[cur_nz];
			float *block_val = &src_bcsr->block_values[cur_nz * mb * nb];

			for (int ii = 0; ii < mb; ++ii)
				for (int jj = 0; jj < nb; ++jj)
				{
					int i = ii + io;
					int j = jj + jo;
					float val = block_val[ii * bcs + jj * brs];

					dst_dense->values[i * cs + j * rs] = val;
				}
		}
	}

	return dst_dense;
}

void destroy_bcsr_matrix(bcsr_matrix_t *A)
{
	// Free the buffers
	free(A->block_col_idx);
	free(A->block_row_idx);
	free(A->block_values);

	// Free the structure
	free(A);
}

void copy_dense_matrix_to_bcsr_matrix(int mb, int nb, int bcs, int brs, dense_matrix_t *src_dense,
				      bcsr_matrix_t *dst_bcsr)
{
	// Get parameters from the dense matrix
	int m = src_dense->m;
	int n = src_dense->n;
	int cs = src_dense->cs;
	int rs = src_dense->rs;
	float *src_values = src_dense->values;

	int cur_nnz = 0;
	for (int io = 0; io < m; io += mb)
	{
		// We need to know where each row starts in in the col_idx and val_idx
		dst_bcsr->block_row_idx[io / mb] = cur_nnz;

		for (int jo = 0; jo < n; jo += nb)
		{

			int nnz_in_block = 0;

			// Scan to see if block has non zeros
			// Note this work can be incorporated in the first scan of the matrix
			for (int ii = 0; ii < mb; ++ii)
				for (int jj = 0; jj < nb; ++jj)
				{
					int i = ii + io;
					int j = jj + jo;
					float val = src_dense->values[i * cs + j * rs];

					if (val != 0.0f)
					{
						nnz_in_block++;
						break;
					}
				}

			// Add the elements into the block
			if (nnz_in_block > 0)
			{
				dst_bcsr->block_col_idx[cur_nnz] = jo;

				// This is where we will copy the block of values
				int block_idx = cur_nnz * mb * nb;
				float *block = &(dst_bcsr->block_values[block_idx]);

				// Now we need to copy over the block
				for (int ii = 0; ii < mb; ++ii)
					for (int jj = 0; jj < nb; ++jj)
					{
						int i = ii + io;
						int j = jj + jo;
						float val = src_dense->values[i * cs + j * rs];

						block[ii * bcs + jj * brs] = val;
					}

				// We have added a new block so we need to increment our current nonzero position.
				cur_nnz++;
			}
		}
	}

	dst_bcsr->block_row_idx[m / mb] = cur_nnz;
}

bcsr_matrix_t *create_bcsr_matrix_from_dense_matrix(int mb, int nb, int bcs, int brs, dense_matrix_t *A_dense)
{
	// First pass count all of the non-zeros
	int nnz = count_number_of_nonzero_blocks_in_dense_matrix(mb, nb, A_dense);

	// Allocate the buffers
	bcsr_matrix_t *A_bcsr = (bcsr_matrix_t *)malloc(sizeof(bcsr_matrix_t));

	// assign the values for the metadata
	int m = A_dense->m;
	int n = A_dense->n;

	A_bcsr->nnz_blocks = nnz;
	A_bcsr->mb = mb;
	A_bcsr->nb = nb;
	A_bcsr->bcs = bcs;
	A_bcsr->brs = brs;
	A_bcsr->m = m;
	A_bcsr->n = n;

	// allocate the arrays used by bcsr
	A_bcsr->block_row_idx = (int *)malloc(sizeof(int) * (m / mb + 1));
	A_bcsr->block_col_idx = (int *)malloc(sizeof(int) * nnz);
	A_bcsr->block_values = (float *)malloc(sizeof(float) * nnz * mb * nb);

	// Second pass to copy the values
	copy_dense_matrix_to_bcsr_matrix(mb, nb, bcs, brs, A_dense, A_bcsr);

	return A_bcsr;
}

/*
  CSRDense or whatever
*/

#if 0
void pretty_print_csrdense_matrix(csrdense_matrix_t *A)
{
  printf("idx --->  ");
  for( int idx = 0; idx < A->nnz; ++idx )
    printf("__%2i",idx);
  printf("\n");

  printf("row_idx = [");
  for(int i = 0; i < A->m+1; ++i)
    printf("%3i,", A->row_idx[i]);
  printf("]\n");

  printf("col_idx = [");
  for(int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
    printf("%3i,", A->col_idx[cur_nnz]);
  printf("]\n");

  printf("values  = [");
  for(int cur_nnz = 0; cur_nnz < A->nnz; ++cur_nnz)
    printf("%3.f,", A->values[cur_nnz]);
  printf("]\n");
  
}
#endif

#if 0
dense_matrix_t *create_dense_matrix_from_csrdense_matrix(csrdense_matrix_t *src_csrdense)
{
  // allocate the actual structure with the metadata
  dense_matrix_t *dst_dense = (dense_matrix_t *)malloc(sizeof(dense_matrix_t));

  // Copy the metadata over
  dst_dense->m = src_csrdense->m;
  dst_dense->n = src_csrdense->n;

  // Default to column major
  dst_dense->rs = dst_dense->m;
  dst_dense->cs = 1;

  // Create a buffer for the values that is MxN
  dst_dense->values = (float *)malloc(sizeof(float) * dst_dense->m * dst_dense->n);

  // Zero all the values out
  for(int i = 0; i < dst_dense->m; ++i )
    for(int j = 0; j < dst_dense->n; ++j )
      dst_dense->values[i*dst_dense->cs + j*dst_dense->rs] = 0.0f;


  // Now we copy the values from the csrdense matrix to the dense one.
  for(int i = 0; i < src_csrdense->m; ++i )
    // This is the only tricky indirection part because we need the segment
    // of column indices that correspond to the nonzeros in the row.
    for(int cur_nz = src_csrdense->row_idx[i];
	cur_nz < src_csrdense->row_idx[i+1];
	++cur_nz )
      {
	int j     = src_csrdense->col_idx[cur_nz];
	float val = src_csrdense->values[cur_nz];
	dst_dense->values[i*dst_dense->cs + j*dst_dense->rs] = val;
      }
  
  return dst_dense;

}
#endif

#if 0
void destroy_csrdense_matrix(csrdense_matrix_t *A)
{
  // Free the buffers
  free(A->col_idx);
  free(A->row_idx);
  free(A->values);

  // Free the structure
  free(A);

}
#endif

#if 0
void copy_dense_matrix_to_csrdense_matrix( dense_matrix_t* src_dense,
					   csrdense_matrix_t *dst_csrdense )
{
  int m  = src_dense->m;
  int n  = src_dense->n;
  int cs = src_dense->cs;
  int rs = src_dense->rs;
  float *src_values = src_dense->values;

  int cur_nnz = 0;
  for( int i = 0; i < m; ++i )
    {
      // We need to know where each row starts in in the col_idx and val_idx
      dst_csrdense->row_idx[i] = cur_nnz;
      for( int j = 0; j < n; ++j )
	{
	  float val = src_values[i*cs + j*rs];
	  
	  if ( val != 0.0f )
	    {
	      dst_csrdense->col_idx[cur_nnz] = j;
	      dst_csrdense->values[cur_nnz]  = val;
	      
	      cur_nnz++;
	    }
	}
    }
  dst_csrdense->row_idx[m] = cur_nnz;
}
#endif

#if 0
csr_matrix_t *create_csr_matrix_from_dense_matrix(dense_matrix_t* A_dense)
{
  // First pass count all of the non-zeros
  int mb = 1;
  int nb = 1;
  int nnz = count_number_of_nonzero_blocks_in_dense_matrix( mb, nb, A_dense );

  // Allocate the buffers
  csr_matrix_t *A_csr = (csr_matrix_t *)malloc(sizeof(csr_matrix_t));

  // assign the values for the metadata
  A_csr->nnz = nnz;
  A_csr->m = A_dense->m;
  A_csr->n = A_dense->n;

  // allocate the arrays used by csr
  A_csr->row_idx = (int *)malloc(sizeof(int)*(A_csr->m+1));
  A_csr->col_idx = (int *)malloc(sizeof(int)*nnz);
  A_csr->values  = (float *)malloc(sizeof(float)*nnz);
  
  // Second pass to copy the values
  copy_dense_matrix_to_csr_matrix( A_dense, A_csr );
  
  return A_csr;

}
#endif
