#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** Serial Job in Small Queue***
# 
# Last revised: 22 June 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch clx.serial.slurm" on a Frontera login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J H2proxy         # Job name
#SBATCH -o h2proxy.o%j     # Name of stdout output file
#SBATCH -e h2proxy.e%j     # Name of stderr error file
#SBATCH -p gpu-a100        # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=end    # Send email at begin and end of job
#SBATCH -A EAR23030        # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=misaelmorales@utexas.edu

# Any other commands must follow all #SBATCH directives...
pwd
date

# Launch serial code...
python main.py         # Do not use ibrun or any other MPI launcher