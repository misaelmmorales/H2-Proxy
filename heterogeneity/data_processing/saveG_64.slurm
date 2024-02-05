#!/bin/bash

#SBATCH -J saveG64        # Job name
#SBATCH -o out_saveG64.%j # Name of stdout output file
#SBATCH -e err_saveG64.%j # Name of stderr error file
#SBATCH -p normal         # Queue (partition) name
#SBATCH -N 1              # Total # of nodes (must be 1 for serial)
#SBATCH -n 20             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00       # Run time (hh:mm:ss)
#SBATCH -A EAR23030       # Project/Allocation name
#SBATCH --mail-type=all   # Send email at begin and end of job
#SBATCH --mail-user=misaelmorales@utexas.edu

# Any other commands must follow all #SBATCH directives...
pwd
date

# Launch serial code...
python3 saveG_64.py