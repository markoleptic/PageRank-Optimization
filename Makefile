# This Makefile orchestrates the building, verification and timing of
# your implementations.
#
#
# Run modes:
# 1. execute on machine with mpi already installed and loaded.
#    make all-local
#
# 2. If you are on a system with mpi loaded through module, then
#    you can run locally on that node for debugging by typing:
#
#    module load OpenMPI
#    make all-local
#
#
# 3. execute remotely on Schooner.
#    make all-schooner
#


clean:
	rm -f *.x *~ *.o

cleanall: clean
	rm -f *.csv *.png


all-schooner: build-verifier-schooner  build-bench-schooner 
	sbatch parallel-prob.sbatch

build-verifier-schooner:
	./build_test_rig.sh remote test verify_op.c

build-bench-schooner:
	./build_test_rig.sh remote bench timer_op.c

all-local: run-verifier-local run-bench-local

run-verifier-local: build-verifier-local
	./run_test_rig.sh local test verify_op.c

run-bench-local: build-bench-local
	./run_test_rig.sh local bench timer_op.c

build-verifier-local:
	./build_test_rig.sh local test verify_op.c

build-bench-local:
	./build_test_rig.sh local bench timer_op.c
