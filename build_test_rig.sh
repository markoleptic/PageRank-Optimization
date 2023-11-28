#!/usr/bin/env bash
#
# This file builds the verifier and bench code based on the configured vars.
# You should not need to modify this.
#
# - richard.m.veras@ou.edu

# Turn on command echo for debugging 
#set -x


MODE=${1}
TEST_NAME=${2}
TEST_RIG=${3}

source op_dispatch_vars.sh

if [[ "${MODE}" == "local" ]]; then
    MODULE_MPI=${VAR_LOCAL_MODULE_MPI}

elif [[ "${MODE}" == "remote" ]]; then
    MODULE_MPI=${VAR_REMOTE_MODULE_MPI}
fi


echo $OP_BASELINE_FILE
echo $CC
echo $CFLAGS


module load ${MODULE_MPI}

#################################################################
# Build the test rig                                            #
# NOTE: need gnu99/gnu11 to get the POSIX compliance for timing #
#################################################################
${CC} ${CFLAGS} -c \
    -DCOMPUTE_NAME_REF=${COMPUTE_NAME_REF} \
    -DDISTRIBUTED_ALLOCATE_NAME_REF=${DISTRIBUTED_ALLOCATE_NAME_REF} \
    -DDISTRIBUTED_FREE_NAME_REF=${DISTRIBUTED_FREE_NAME_REF} \
    -DDISTRIBUTE_DATA_NAME_REF=${DISTRIBUTE_DATA_NAME_REF} \
    -DCOLLECT_DATA_NAME_REF=${COLLECT_DATA_NAME_REF} \
    -DCOMPUTE_NAME_TST=${COMPUTE_NAME_TST} \
    -DDISTRIBUTED_ALLOCATE_NAME_TST=${DISTRIBUTED_ALLOCATE_NAME_TST} \
    -DDISTRIBUTED_FREE_NAME_TST=${DISTRIBUTED_FREE_NAME_TST} \
    -DDISTRIBUTE_DATA_NAME_TST=${DISTRIBUTE_DATA_NAME_TST} \
    -DCOLLECT_DATA_NAME_TST=${COLLECT_DATA_NAME_TST} \
    ${TEST_RIG} -static -fPIC -o ${TEST_RIG}.o


############################
# Build all of the helpers #
############################
${CC} ${CFLAGS} -c helper.c
${CC} ${CFLAGS} -c sparse.c
${CC} ${CFLAGS} -c pagerank.c

HELPER_OBJS="helper.o sparse.o pagerank.o"

################################
# Build the reference baseline #
################################
${CC} ${CFLAGS} -c \
    -DCOMPUTE_NAME=${COMPUTE_NAME_REF} \
    -DDISTRIBUTE_DATA_NAME=${DISTRIBUTE_DATA_NAME_REF} \
    -DCOLLECT_DATA_NAME=${COLLECT_DATA_NAME_REF} \
    -DDISTRIBUTED_ALLOCATE_NAME=${DISTRIBUTED_ALLOCATE_NAME_REF}\
    -DDISTRIBUTED_FREE_NAME=${DISTRIBUTED_FREE_NAME_REF}\
    ${OP_BASELINE_FILE} -o ${OP_BASELINE_FILE}.ref.o


######################
# Build the variants #
######################
for SUBMISSION_VARIANT in ${OP_SUBMISSION_LIST}
do
    echo $SUBMISSION_VARIANT

    # build the variants
    ${CC} ${CFLAGS} -c \
    -DCOMPUTE_NAME=${COMPUTE_NAME_TST} \
    -DDISTRIBUTE_DATA_NAME=${DISTRIBUTE_DATA_NAME_TST} \
    -DCOLLECT_DATA_NAME=${COLLECT_DATA_NAME_TST} \
    -DDISTRIBUTED_ALLOCATE_NAME=${DISTRIBUTED_ALLOCATE_NAME_TST}\
    -DDISTRIBUTED_FREE_NAME=${DISTRIBUTED_FREE_NAME_TST}\
    ${SUBMISSION_VARIANT} -o ${SUBMISSION_VARIANT}.o

    # link together the test rig
    ${CC} ${CFLAGS} ${HELPER_OBJS} ${TEST_RIG}.o ${OP_BASELINE_FILE}.ref.o ${SUBMISSION_VARIANT}.o -o ./run_${TEST_NAME}_${SUBMISSION_VARIANT}.x
done
