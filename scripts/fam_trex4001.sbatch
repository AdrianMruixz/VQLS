#!/bin/bash
#SBATCH --job-name=fam_trex4001
#SBATCH --account=newell
#SBATCH --partition=granite-guest
#SBATCH --qos=granite-guest
#SBATCH --requeue
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --mail-type=ALL,TIME_LIMIT_90
#SBATCH --mail-user=u1662280@utah.edu
#SBATCH -o fam_trex4001_%j.out
#SBATCH -e fam_trex4001_%j.err
echo "Job $SLURM_JOB_ID | Node $(hostname) | CPUs $SLURM_CPUS_PER_TASK"
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate mitiq_py311_device
cd "$HOME/other_families_trex_campaign"
bash run_families_seed.sh 4001 trex
