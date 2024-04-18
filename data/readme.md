# Extraction and plotting

## csv

### `ngc-pytorch-23.07.csv`

```bash
cd ../post-processing
source .venv/bin/activate
./extract_metrics.py -i ../data/logs/ngc-pytorch-23.07/bessemer* ../data/logs/ngc-pytorch-23.07/stanage-a100* ../data/logs/ngc-pytorch-23.07/stanage-h100* ../data/logs/ngc-pytorch-23.07/bede* -o ../data/csv/ngc-pytorch-23.07.csv -f
```

### `ngc-pytorch-24.02.csv`

```bash
cd ../post-processing
source .venv/bin/activate
./extract_metrics.py -i ../data/logs/ngc-pytorch-24.02/bessemer* ../data/logs/ngc-pytorch-24.02/stanage-a100* ../data/logs/ngc-pytorch-24.02/stanage-h100* ../data/logs/ngc-pytorch-24.02/bede* -o ../data/csv/ngc-pytorch-24.02.csv -f
```

## png

### `png/ngc-pytorch-23.07*.png`

```bash
cd ../post-processing
source .venv/bin/activate
./plot.py -f --dpi 300 --no-xlabel --fp16 --metric samples_per_second --bar-label-type edge --bar-label-fmt="%.1f" --sharey --legend-loc "upper left" --title-prefix "PyTorch NGC 23.07:" -i ../data/csv/ngc-pytorch-23.07.csv -o ../data/png/ngc-23.07/ngc-pytorch-23.07-fp16-samples-per-second.png
./plot.py -f --dpi 300 --no-xlabel --fp32 --metric samples_per_second --bar-label-type edge --bar-label-fmt="%.1f" --sharey --legend-loc "upper left" --title-prefix "PyTorch NGC 23.07:" -i ../data/csv/ngc-pytorch-23.07.csv -o ../data/png/ngc-23.07/ngc-pytorch-23.07-fp32-samples-per-second.png
./plot.py -f --dpi 300 --no-xlabel --fp16 --metric runtime --bar-label-type edge --bar-label-fmt="%.1f" --legend-loc "upper right" --title-prefix "PyTorch NGC 23.07:" -i ../data/csv/ngc-pytorch-23.07.csv -o ../data/png/ngc-23.07/ngc-pytorch-23.07-fp16-runtime.png
./plot.py -f --dpi 300 --no-xlabel --fp32 --metric runtime --bar-label-type edge --bar-label-fmt="%.1f" --legend-loc "upper right" --title-prefix "PyTorch NGC 23.07:" -i ../data/csv/ngc-pytorch-23.07.csv -o ../data/png/ngc-23.07/ngc-pytorch-23.07-fp32-runtime.png
```

### `png/ngc-pytorch-24.02*.png`

```bash
cd ../post-processing
source .venv/bin/activate
./plot.py -f --dpi 300 --no-xlabel --fp16 --metric samples_per_second --bar-label-type edge --bar-label-fmt="%.1f" --sharey --legend-loc "upper left" --title-prefix "PyTorch NGC 24.02:" -i ../data/csv/ngc-pytorch-24.02.csv -o ../data/png/ngc-24.02/ngc-pytorch-24.02-fp16-samples-per-second.png
./plot.py -f --dpi 300 --no-xlabel --fp32 --metric samples_per_second --bar-label-type edge --bar-label-fmt="%.1f" --sharey --legend-loc "upper left" --title-prefix "PyTorch NGC 24.02:" -i ../data/csv/ngc-pytorch-24.02.csv -o ../data/png/ngc-24.02/ngc-pytorch-24.02-fp32-samples-per-second.png
./plot.py -f --dpi 300 --no-xlabel --fp16 --metric runtime --bar-label-type edge --bar-label-fmt="%.1f" --legend-loc "upper right" --title-prefix "PyTorch NGC 24.02:" -i ../data/csv/ngc-pytorch-24.02.csv -o ../data/png/ngc-24.02/ngc-pytorch-24.02-fp16-runtime.png
./plot.py -f --dpi 300 --no-xlabel --fp32 --metric runtime --bar-label-type edge --bar-label-fmt="%.1f" --legend-loc "upper right" --title-prefix "PyTorch NGC 24.02:" -i ../data/csv/ngc-pytorch-24.02.csv -o ../data/png/ngc-24.02/ngc-pytorch-24.02-fp32-runtime.png
```
