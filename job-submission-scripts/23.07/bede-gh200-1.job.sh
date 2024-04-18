#!/bin/bash
#SBATCH --account=<project> # specify the <project>
#SBATCH --time=1:00:0
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Path to the compiled image
APPTAINER_IMAGE_PATH=/nobackup/projects/${SLURM_JOB_ACCOUNT}/${USER}/aarch64/pytorch-transformers-wikitext2-benchmark/transformers-benchmark-23.07.sif 

# Output the node this was executed on
echo "HOSTNAME=${HOSTNAME}"

# Output some GPU/CPU information into the Log
nvidia-smi
nproc

# Run the benchmark
apptainer run -c -e --bind $(pwd):/mnt --bind ${TMPDIR}:/tmp --env "HF_HOME=/mnt/hf_home/${SLURM_JOB_ID}" --env "TMPDIR=/tmp/${SLURM_JOB_ID}" --nv ${APPTAINER_IMAGE_PATH}
