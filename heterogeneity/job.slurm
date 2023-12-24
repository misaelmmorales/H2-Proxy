#!/bin/bash

#SBATCH -J H2proxy         # Job name
#SBATCH -o h2proxy.o%j     # Name of stdout output file
#SBATCH -e h2proxy.e%j     # Name of stderr error file
#SBATCH -p gpu-h100        # Queue (partition) name
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
