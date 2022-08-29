#!/bin/bash -ex

#SBATCH --mail-user=xinyang@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8      # 10 MPI processes per node
#SBATCH --time=7-00:00:00
#SBATCH --job=PaiNN-training
#SBATCH --output=runner_output.log
#SBATCH --gres=gpu:RTX3090:1

#module load ASE/3.22.0-intel-2020b
#module load Python/3.8.6-GCCcore-10.2.0

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nvidia-smi > gpu_info
ulimit -s unlimited
python3 md_run.py
