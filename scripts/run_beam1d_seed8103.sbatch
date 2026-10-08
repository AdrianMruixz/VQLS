#!/bin/bash
#SBATCH --job-name=b1d_s8103
#SBATCH --account=owner-guest
#SBATCH --partition=kingspeak-guest
#SBATCH --qos=kingspeak-guest
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --mail-type=ALL,TIME_LIMIT_90
#SBATCH --mail-user=u1662280@utah.edu
#SBATCH -o b1d_seed8103_%j.out
#SBATCH -e b1d_seed8103_%j.err

echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | Seed: 8103 | CPUs: ${SLURM_CPUS_PER_TASK}"

source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate mitiq_py311_device
cd "$HOME/beam1d_optimized"
bash run_beam1d_full.sh 8103
