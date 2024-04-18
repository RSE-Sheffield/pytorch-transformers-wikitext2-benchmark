#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1

# 24 CPU cores (1/2 of the node) and 1 GPUs worth of memory < 1/2th of the node)
#SBATCH --cpus-per-task=24
#SBATCH --mem=82G

# Path to the compiled image
APPTAINER_IMAGE_PATH=/mnt/parscratch/users/$USER/pytorch-transformers-wikitext2-benchmark/transformers-benchmark-23.07.sif 

# Output the node this was executed on
echo "HOSTNAME=${HOSTNAME}"

# Output some GPU/CPU information into the Log
nvidia-smi
nproc

# Run the benchmark
apptainer run -c -e --bind $(pwd):/mnt --bind ${TMPDIR}:/tmp --env "HF_HOME=/mnt/hf_home/${SLURM_JOB_ID}" --env "TMPDIR=/tmp/${SLURM_JOB_ID}" --nv ${APPTAINER_IMAGE_PATH}
