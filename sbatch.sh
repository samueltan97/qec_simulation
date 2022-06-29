#!/bin/bash

# Submit this script with: sbatch sbatch.sh

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=any
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J burst-error-threshold   # job name
#SBATCH --mail-user=stan@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load singularity/3.8.0
singularity exec /home/stan/stim_container.sif python cluster.py