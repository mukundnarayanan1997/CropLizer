#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=2-23:58:00
#SBATCH --partition=mini
#SBATCH --job-name=Sen1
#SBATCH --error=err_Sen1
#SBATCH --output=out_Sen1

cd "/scratch/mukund_n.iitr/CropLizer/v3"

export PATH="/home/mukund_n.iitr/.conda/envs/mukundn/bin:$PATH"

python CRLZSensitivity.py>CRLZSensitivity.txt
