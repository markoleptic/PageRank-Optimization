# PageRank Optimization

This repository implements various parallel optimization techniques for the PageRank algorithm. The starting point for all variants is `tuned_variant_baseline.c`.

## Optimization Techniques

- **tuned_variant_csr.c:** Uses compressed sparse row format (CSR) instead of default Coordinate format (COO)
- **tuned_variant_openmp.c:** Uses OpenMP with 4 threads and reduction
- **tuned_variant_simd.c:** Applies AVX2 to operate on eight columns at once
- **tuned_variant_openmp_and_simd.c:** Combines OpenMP and SIMD

## File Descriptions

- **tuned_variant_baseline.c:** The starting point for all variants.
- **writeup.pdf:** Contains a full description and results of the implemented optimization techniques.

Feel free to explore each file for detailed implementation and optimization strategies.

