#! /usr/bin/env python3

import pathlib
import argparse
import re
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description="Extract usage information from run_clm stdout/stderr logging")
    parser.add_argument("-i", "--input", nargs="+", type=pathlib.Path, help="input file(s)", required=True)
    parser.add_argument("-o", "--output", type=pathlib.Path, help="output csv path. Prints if omitted")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing output files")

    args = parser.parse_args()
    
    METRIC_PATTERN = r"^[ ]+(epoch|train_[a-z_]+|eval_[a-z_]+|perplexity)[ ]+=[ ]+([0-9\.:]+)"
    metric_re = re.compile(METRIC_PATTERN)
    RUNTIME_PATTERN = r"^([0-9]{1,2}):([0-9]{2}):([0-9]{2})\.([0-9]+)$"
    runtime_re = re.compile(RUNTIME_PATTERN)
    DRIVER_PATTERN = r"^\|.+Driver Version: ([0-9]+\.[0-9]+\.[0-9]+).+$"
    driver_re = re.compile(DRIVER_PATTERN)
    NVIDIA_SMI_TABLE_GPU_PATTERN = r"^\|[ ]+[0-9]+[ ]+(.+?)[ ]+(On|Off).+\|$"
    gpu_re = re.compile(NVIDIA_SMI_TABLE_GPU_PATTERN)
    dicts = []
    for input_path in args.input:
        if not input_path.is_file():
            raise Exception(f"File does not exist: '{input_path}'")
        with open(input_path) as fp:
            # There is a potential stdout/stderr race, so can't just grab sequential lines safely
            # So sketchily compare each line pulling out metrics. This is probably brittle, but I didn't plan enough when I set up the original scripts.
            train_data = []
            eval_data = []
            in_train = False
            in_eval = False
            precision = 32
            cuda_version = None
            torch_version = None
            nvidia_driver = None
            transformers_version = None
            gpu = None
            for line in fp.readlines():
                if line.startswith("torch "):
                    torch_version = line.strip().split(" ")[-1]
                elif m := driver_re.match(line):
                    nvidia_driver = m.group(1)
                # Didn't use nvidia-smi -L, so getting the gpu name is a bit flaky.
                elif m := gpu_re.match(line):
                    gpu = m.group(1)
                # cupy-cuda seems to be the simplest way to get the actual cuda version used in the container. 
                elif line.startswith("cupy-cuda"):
                    cuda_version = line.strip().split(" ")[-1]
                elif line.startswith("***** train metrics *****"):
                    in_train = True
                    in_eval = False
                    train_data.append({"precision": precision})
                elif line.startswith("***** eval metrics *****"):
                    in_train = False
                    in_eval = True
                    eval_data.append({"precision": precision})
                elif line.startswith("fp16="):
                    precision= 16 if  line.startswith("fp16=True") else 32
                elif line.startswith('  "transformers_version": "'):
                    transformers_version = line.strip().split(" ")[-1].replace(",", "").replace('"',"")
                elif m := metric_re.match(line):
                    k = m.group(1)
                    v = m.group(2)
                    if k.endswith("_runtime"):
                        if rt_match := runtime_re.match(v):
                            v = f"{(int(rt_match.group(1)) * 60) + (int(rt_match.group(2)) * 60) + int(rt_match.group(3))}.{rt_match.group(4)}"
                        else:
                            raise Exception("Error during runtime string conversion for {v}")
                    if in_train:
                        train_data[-1][k] = v
                    elif in_eval:
                        eval_data[-1][k] = v
                    else:
                        raise Exception("train/data regex match without header")

            # @todo - put more version info into the script output, gpu model etc. 
            label = input_path.name
            gpu_label = label.split("-slurm")[0]

            for t, e in zip(train_data, eval_data):
                combined = {
                    "label": label,
                    "gpu_label": gpu_label,
                    "gpu": gpu,
                    "nvidia_driver": nvidia_driver,
                    "cuda_version": cuda_version,
                    "torch_version": torch_version,
                    "transformers_version": transformers_version,
                    **t,
                    **e
                }
                dicts.append(combined)
    df = pd.DataFrame(dicts)

    if args.output:
        if args.output.is_dir():
            raise Exception(f"{args.output} is a directory")
        elif args.output.is_file() and not args.force:
            raise Exception(f"{args.output.is_file} is an existing file. Use -f/--force to overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
    else:
        print(df.to_csv(index=False))

if __name__ == "__main__":
    main()