#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=2-23:58:00
#SBATCH --partition=mini
#SBATCH --job-name=CRLZ
#SBATCH --error=err_CRLZ
#SBATCH --output=out_CRLZ

cd "/scratch/mukund_n.iitr/CropLizer/v3"

export PATH="/home/mukund_n.iitr/.conda/envs/mukundn/bin:$PATH"

python HyperparameterTuning.py>CRLZOptimization.txt
