#ifndef HELPER_H
#define HELPER_H

/*
  General helper functions for PDN class assignments.


  - richard.m.veras@ou.edu

*/


void print_float_mem(char *name, int vlen, float *src);
void print_float_matrix_mem(char *name, int m, int n, int rs, int cs, float *src);
float max_pair_wise_diff_vect(int m, float *a, float *b);


void fill_sequences( int size, int start, float *dst );
void fill_sequences_normalized( int size, int start, float *dst );
void neg_xout_sequences( int size, float *dst );
void zero_out_sequences( int size, float *dst );

void fill_buffer_with_random( int num_elems, float *buff );
void fill_buffer_with_value( int num_elems, float val, float *buff );
float max_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b);
int scale_p_on_pos_ret_v_on_neg(int p, int v);

#endif /* HELPER_H */
