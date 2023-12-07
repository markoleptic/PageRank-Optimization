#!/usr/bin/env bash

############################################
# CHANGE THESE LINES:                      #
# Replace the filenames with your variants #
############################################
OP_SUBMISSION_VAR01_FILE="tuned_variant_simd.c"
OP_SUBMISSION_VAR02_FILE="tuned_variant_openmp.c"
OP_SUBMISSION_VAR03_FILE="tuned_variant_openmp_and_simd.c"  #########################################
OP_SUBMISSION_VAR04_FILE="tuned_variant_openmp_and_simd_2.c"   # NOTE: If you add variants here, then  #
OP_SUBMISSION_VAR05_FILE="tuned_variant05_op.c" # append them to the OP_SUBMISSION_LIST #
# ..                                             #########################################
#OP_SUBMISSION_VARXX_FILE="tuned_variantXX_op.c"



########################################
# List of variants to build and run    #
# Add to this if you add more variants #
########################################
OP_SUBMISSION_LIST="${OP_SUBMISSION_VAR01_FILE} ${OP_SUBMISSION_VAR02_FILE} ${OP_SUBMISSION_VAR03_FILE}" #${OP_SUBMISSION_VAR04_FILE} ${OP_SUBMISSION_VAR05_FILE}"

##########################################
# These are the defaults for running the #
# experiments.                           #
##########################################
VAR_LOCAL_NUMRANKS=4
VAR_LOCAL_MIN=128
VAR_LOCAL_MAX=2048
VAR_LOCAL_STEP=128
VAR_LOCAL_SCALE_M=1
VAR_LOCAL_SCALE_K=-10
VAR_LOCAL_OUTPUT_FILE_PREFIX=result_local




VAR_REMOTE_NUMRANKS= # This is set in the sbatch file
VAR_REMOTE_MIN=128
VAR_REMOTE_MAX=2048
VAR_REMOTE_STEP=128
VAR_REMOTE_SCALE_M=1
VAR_REMOTE_SCALE_K=-10
VAR_REMOTE_OUTPUT_FILE_PREFIX=result_local


############################################################
# NOTE: TODO: these should be overides and the environment #
# specific values should be in a .env file that's in a     #
# .gitignore                                               #
############################################################

VAR_LOCAL_MODULE_MPI="mpi/mpich-x86_64" # This may be different on your system
# VAR_LOCAL_MODULE_MPI="OpenMPI" # This is the schooner value
VAR_LOCAL_MODULE_SCIKIT=
VAR_LOCAL_MODULE_MATPLOT=


# These are correct for schooner
VAR_REMOTE_MODULE_MPI="OpenMPI"
VAR_REMOTE_MODULE_SCIKIT="scikit-learn/0.23.1-foss-2020a-Python-3.8.2"
VAR_REMOTE_MODULE_MATPLOT="matplotlib/3.2.1-foss-2019b-Python-3.8.2"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC=mpicc
# CFLAGS="-std=c99 -O2"
CFLAGS="-std=c99 -O3 -mavx2 -mfma -lm -fopenmp"
#CFLAGS="-std=c99 -O0 -mavx2 -g -mfma -lm"


#####################################################
# DO NOT CHANGE THE FOLLOWING LINES                 #
OP_BASELINE_FILE="baseline_op.c"                    #
                                                    #
COMPUTE_NAME_REF="baseline"                         #
DISTRIBUTED_ALLOCATE_NAME_REF="baseline_allocate"   #
DISTRIBUTED_FREE_NAME_REF="baseline_free"           #
DISTRIBUTE_DATA_NAME_REF="baseline_distribute"      #
COLLECT_DATA_NAME_REF="baseline_collect"            #
                                                    #
COMPUTE_NAME_TST="test"                             #
DISTRIBUTED_ALLOCATE_NAME_TST="test_allocate"       #
DISTRIBUTED_FREE_NAME_TST="test_free"               #
DISTRIBUTE_DATA_NAME_TST="test_distribute"          #
COLLECT_DATA_NAME_TST="test_collect"                #
#####################################################

