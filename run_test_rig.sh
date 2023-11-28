#!/usr/bin/env bash
#
# This file runs the test based on the configured vars.
# You should not need to modify this.
#
# - richard.m.veras@ou.edu

# Turn on command echo for debugging 
#set -x



MODE=${1}
TEST_NAME=${2}
TEST_RIG=${3}

source op_dispatch_vars.sh

# NOTE: Change this for new operations
if [[ "${MODE}" == "local" ]]; then
    NUMRANKS=${VAR_LOCAL_NUMRANKS}
    MIN=${VAR_LOCAL_MIN}
    MAX=${VAR_LOCAL_MAX}
    STEP=${VAR_LOCAL_STEP}
    SCALE_M=${VAR_LOCAL_SCALE_M}
    SCALE_K=${VAR_LOCAL_SCALE_K}
    OUTPUT_FILE_PREFIX=${VAR_LOCAL_OUTPUT_FILE_PREFIX}
    MODULE_MPI=${VAR_LOCAL_MODULE_MPI}
    MODULE_SCIKIT=${VAR_LOCAL_MODULE_SCIKIT}
    MODULE_MATPLOT=${VAR_LOCAL_MODULE_MATPLOT}

elif [[ "${MODE}" == "remote" ]]; then
    NUMRANKS=${VAR_REMOTE_NUMRANKS}
    MIN=${VAR_REMOTE_MIN}
    MAX=${VAR_REMOTE_MAX}
    STEP=${VAR_REMOTE_STEP}
    SCALE_M=${VAR_REMOTE_SCALE_M}
    SCALE_K=${VAR_REMOTE_SCALE_K}
    OUTPUT_FILE_PREFIX=${VAR_REMOTE_OUTPUT_FILE_PREFIX}
    MODULE_MPI=${VAR_REMOTE_MODULE_MPI}
    MODULE_SCIKIT=${VAR_REMOTE_MODULE_SCIKIT}
    MODULE_MATPLOT=${VAR_REMOTE_MODULE_MATPLOT}
fi



echo "TEST_NAME=${TEST_NAME}"
echo "TEST_RIG=${TEST_RIG}"
echo "MODE=${MODE}"
echo "NUMRANKS=${NUMRANKS}"
echo "MIN=${MIN}"
echo "MAX=${MAX}"
echo "STEP=${STEP}"
echo "SCALE_M=${SCALE_M}"
echo "SCALE_K=${SCALE_K}"
echo "OUTPUT_FILE_PREFIX=${OUTPUT_FILE_PREFIX}"
echo "MODULE_MPI=${MODULE_MPI}"
echo "MODULE_SCIKIT=${MODULE_SCIKIT}"
echo "MODULE_MATPLOT=${MODULE_MATPLOT}"


OUTPUT_FILE_LONG_PREFIX="${OUTPUT_FILE_PREFIX}_${MODE}_${TEST_NAME}"

######################
# Initializaton step #
######################
module load ${MODULE_MPI}


#########################
# Loop through variants #
#########################
for SUBMISSION_VARIANT in ${OP_SUBMISSION_LIST}
do

    if [[ "${MODE}" == "local" ]]; then


	OUTPUT_FILE_NAME="${OUTPUT_FILE_LONG_PREFIX}_${SUBMISSION_VARIANT}.csv"
	CURRENT_TEST_RIG="./run_${TEST_NAME}_${SUBMISSION_VARIANT}.x"
	
	mpiexec -n ${NUMRANKS} ${CURRENT_TEST_RIG} ${MIN} ${MAX} ${STEP} ${SCALE_M} ${SCALE_K} ${OUTPUT_FILE_NAME}

	
    elif [[ "${MODE}" == "remote" ]]; then
	OUTPUT_FILE_NAME="${OUTPUT_FILE_LONG_PREFIX}_${SUBMISSION_VARIANT}.csv"
	CURRENT_TEST_RIG="./run_${TEST_NAME}_${SUBMISSION_VARIANT}.x"
	
	mpiexec ${CURRENT_TEST_RIG} ${MIN} ${MAX} ${STEP} ${SCALE_M} ${SCALE_K} ${OUTPUT_FILE_NAME}

    fi
done


##############################
# Final step through results #
##############################
if [[ "${TEST_NAME}" == "test" ]]; then
    echo "Number of FAILS: `grep "FAIL" ${OUTPUT_FILE_LONG_PREFIX}_*.csv|wc -l`"
elif [[ "${TEST_NAME}" == "bench" ]]; then

    PLOT_NAME="Results_${MODE}_${TEST_NAME}.png"

    if [[ "${MODE}" == "local" ]]; then
	# Note much setup to do here
	echo ""

    elif [[ "${MODE}" == "remote" ]]; then

	module unload ${MODULE_MPI}
	module load ${MODULE_SCIKIT}
	module load ${MODULE_MATPLOT}

    fi

    ./plotter_multi.py "Local Results" ${PLOT_NAME} `ls ${OUTPUT_FILE_LONG_PREFIX}_*.csv`
fi
