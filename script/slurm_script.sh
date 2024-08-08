#!/bin/bash

#SBATCH --nodelist=##node to work with ex.: node-02 (CHANGE)
#SBATCH --ntasks=1
#SBATCH --get-user-env            ## Exports all local SHELL vars
#SBATCH --time=72:00:00 ##(CHANGE)
#SBATCH --cpus-per-task=##number_of_cpus ex.: 32 (CHANGE)
#SBATCH --mem-per-cpu=##RAM per cpu ex.:4GB (CHANGE)
#SBATCH --partition=##partition name (CHANGE)
##SBATCH --qos=long ##uncoment for runs longer than 10 days
#SBATCH --job-name=##JOB_NAME (CHANGE)
#SBATCH --output=slurm-%j.out 
#SBATCH --error=slurm-%j.err

# [DO NOT CHANGE - MANDATORY]
# Use '/scratch' FS - file system
# -------------------------------
source scratchfs
# -------------------------------
# SYNC DATA: uwd SlurmJobName.out
# -------------------------------

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## -------- BEFORE SEND IT TO THE NODE  --------
## Activate our environment in terminal
## module load anaconda3/2023.3
## conda activate rl_rht_py3.9

## Go to the folder from which we want to execute our code (check current with pwd)

## Set the path to our work directory (`pwd`) in `global_vars.py`:
## WORKING_DIR = ## directory_with_the_code_to_employ (CHANGE)

## ------------------ EXECUTE  ------------------
## Execute our code with sbatch: sbatch ./script/slurm_script.sh
python ./src/RL_simulations/RHT_simulation_RUNS.py ##(CHANGE)