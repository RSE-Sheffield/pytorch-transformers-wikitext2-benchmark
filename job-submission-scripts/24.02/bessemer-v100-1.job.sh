#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1

# 10 CPU cores (1/4th of the node) and 1 GPUs worth of memory < 1/4th of the node)
#SBATCH --cpus-per-task=10
#SBATCH --mem=34G

# Path to the compiled image
APPTAINER_IMAGE_PATH=/fastdata/$USER/transformers-benchmark-24.02.sif 

# Output the node this was executed on
echo "HOSTNAME=${HOSTNAME}"

# Output some GPU/CPU information into the Log
nvidia-smi
nproc

# Run the benchmark
apptainer run -c -e --bind $(pwd):/mnt --bind ${TMPDIR}:/tmp --env "HF_HOME=/mnt/hf_home/${SLURM_JOB_ID}" --env "TMPDIR=/tmp/${SLURM_JOB_ID}" --nv ${APPTAINER_IMAGE_PATH}
