#! /usr/bin/bash

# Ensure expected number of arguments
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  echo 1>&2 "Usage:n\n$0 <RUN_CLM_PATH> [REPS] [BATCH_SIZE]"
  exit 1
fi

RUN_CLM=$1
# Defaults to 3
REPS=${2:-3}
# Defaults to 8
BATCH_SIZE=${3:-8}

# Ensure the run_clm.py script exists, else error.
if [ ! -f ${RUN_CLM} ]; then
    echo "run_clm.py not found at ${RUN_CLM}"
    exit 1
fi

set -e

nvidia-smi
nvidia-smi -L

echo "RUN_CLM=${RUN_CLM}"
echo "REPS=${REPS}"
echo "BATCH_SIZE=${BATCH_SIZE}"

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.get_arch_list())"
python3 -m pip list 

# FP32
for rep in $(seq $REPS); do
    python3  ${RUN_CLM} \
        --model_name_or_path gpt2 \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --do_train \
        --do_eval \
        --overwrite_output_dir \
        --output_dir ${TMPDIR:-/tmp}/test-clm-${rep}
done

# FP16
for rep in $(seq $REPS); do
    python3  ${RUN_CLM} \
        --model_name_or_path gpt2 \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --do_train \
        --do_eval \
        --fp16 \
        --fp16_full_eval \
        --fp16_opt_level O3 \
        --overwrite_output_dir \
        --output_dir ${TMPDIR:-/tmp}/test-clm-${rep}
done
