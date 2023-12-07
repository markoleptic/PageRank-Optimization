#!/bin/bash
#SBATCH --partition=debug_5min
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --output=./jobname_%J_stdout.txt
#SBATCH --error=./jobname_%J_stderr.txt
#SBATCH --time=5:00
#SBATCH --job-name=cs4473_cs5473_lab02
#SBATCH --mail-user=youremailaddress@yourinstitution.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=./


# Verify
./run_test_rig.sh remote test verify_op.c

# Bench
./run_test_rig.sh remote bench timer_op.c

