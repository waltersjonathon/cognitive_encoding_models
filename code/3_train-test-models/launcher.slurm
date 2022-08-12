#!/bin/bash
#
#SBATCH -J null_models
#SBATCH --array=1-920%128

#SBATCH --time=00:30:00
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

#SBATCH -p russpold,owners,normal
#SBATCH --export=NONE

# Outputs ----------------------------------
#SBATCH -o slurm_output/%A-%a.out
#SBATCH -e slurm_output/%A-%a.err
#SBATCH --mail-user=waltersj@stanford.edu
#SBATCH --mail-type=ALL
# ------------------------------------------


##module load system                          # Only Sherlock2

source $HOME/.bashrc
source activate multitask_env

eval $( sed "${SLURM_ARRAY_TASK_ID}q;d" em_cmds_first_level_bySession_analysis-3_RidgeCV_NULLS.sh )