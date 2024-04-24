#! /usr/bin/env python3

import pathlib
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MONOSPACE_FONT_FAMILY = "DejaVu Sans Mono"

class Config:
    def __init__(self):
        self.SNS_CONTEXT = "talk"
        self.SNS_STYLE = "darkgrid"
        self.SNS_PALETTE = "Dark2"
        self.FIGSIZE_INCHES = (16, 9)
        self.DPI = 96
        self.X = None
        self.XLABEL = None
        self.HUE = None
        self.HUELABEL = None
        self.LEGEND_BORDER_PAD = 0.5
        self.EXTERNAL_LEGEND = False
        self.XTICK_ANGLE = None
        self.BAR_LABEL_TYPE = None #"edge" # edge or center
        self.BAR_LABEL_FMT = "%.3f"
        self.ERRORBAR=None


def plot(args, df, config):
    # Do some plotting stuff.
    sns.set_context(config.SNS_CONTEXT, rc={"lines.linewidth": 2.5})  
    sns.set_style(config.SNS_STYLE)
    huecount = len(df[config.HUE].unique()) if config.HUE is not None else 1 
    if huecount == 0:
        huecount = 1
    palette = sns.color_palette(config.SNS_PALETTE, huecount)
    sns.set_palette(palette)

    metric = args.metric if args.metric is not None else "samples_per_second"
    metric_label = metric.replace("_", " ")
    if args.metric == "runtime":
        metric_label = "Runtime (s)"
    elif args.metric == "samples_per_second":
        metric_label = "Samples per Second"
    elif args.metric == "steps_per_second":
        metric_label = "Steps per Second"

    train_samples = int(df["train_samples"].mean())
    eval_samples = int(df["eval_samples"].mean())

    subplots = [
        {
            "title": f"Training ({train_samples} samples)",
            "Y": f"train_{metric}",
            "YLABEL": f"{metric_label}",
        },
        {
            "title": f"Inferencing ({eval_samples} samples)",
            "Y": f"eval_{metric}",
            "YLABEL": f"{metric_label}",
        }
    ]

    fig, axes = plt.subplots(nrows=1, ncols=len(subplots), constrained_layout=True, sharey=args.sharey)
    fig.set_size_inches(config.FIGSIZE_INCHES[0], config.FIGSIZE_INCHES[1])

    # compute the legend label
    legend_title = f"{config.HUELABEL}"

    for idx, (ax, sp) in enumerate(zip(axes, subplots)):

        g = sns.barplot(
            data=df, 
            # x=df.index,
            x = config.X,
            y=sp["Y"], 
            hue=config.HUE, 
            ax=ax,
            palette=palette,
            errorbar=config.ERRORBAR,
            capsize=.1,
            err_kws={'linewidth': 1.8},
            legend="full",
        )

        # Axis settings
        if config.XLABEL:
            ax.set(xlabel=config.XLABEL)
        if args.no_xlabel:
            ax.set(xlabel=None)
        if sp["YLABEL"]:
            ax.set(ylabel=sp["YLABEL"])

        if config.XTICK_ANGLE:
            ax.tick_params(axis='x', labelrotation=config.XTICK_ANGLE)

        if config.BAR_LABEL_TYPE:
            for container in ax.containers:
                ax.bar_label(container, label_type=config.BAR_LABEL_TYPE, fmt=config.BAR_LABEL_FMT)

        # Set the subtitle.
        ax.set_title(sp["title"])

        # Remove legend from all axes, or all but the last one. Otherwise tweak it.
        if config.EXTERNAL_LEGEND:
            if ax.get_legend():
                ax.get_legend().remove()
        else:
            # Remove all but the last one, unless samples per second then remove first
            if (metric == "runtime" and idx != min(len(axes), len(subplots)) - 1) or (metric == "samples_per_second" and idx != 0):
                if ax.get_legend():
                    ax.get_legend().remove()

            if ax.get_legend() is not None:
                legend = ax.get_legend()
                legend.set_title(legend_title)
                if args.legend_loc:
                    ax.legend(loc=args.legend_loc)
                plt.setp(legend.texts)#, family=MONOSPACE_FONT_FAMILY)

    # Compute the figure title
    prec = "FP16" if args.fp16 else "FP32" if args.fp32 else ""
    title = f"GPT2 Wikitext-2 fine-tuning with batchsize 8 in {prec}"
    if args.title_prefix is not None and args.title_prefix != "":
        title = f"{args.title_prefix} {title}"
    if args.title is not None and args.title != "":
        title = args.title
    plt.suptitle(title)

    # plt.tight_layout()

    # If using an external legend, do external placement. This is experimental.
    if config.EXTERNAL_LEGEND:
        # Set legend placement if not internal.
        loc = "upper left"
        # @todo - y offset should be LEGEND_BORDER_PAD transformed from font units to bbox.
        bbox_to_anchor = (1, 1 - 0.0)
        handles, labels = ax.get_legend_handles_labels()
        # add an invisible patch with the appropriate label, like how seaborn does if multiple values are provided.
        handles.insert(0, mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label=legend_title))
        labels.insert(0, legend_title)
        ax.legend(handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, borderaxespad=config.LEGEND_BORDER_PAD)
        legend = ax.get_legend()
        plt.setp(legend.texts)#, family=MONOSPACE_FONT_FAMILY)
    else:
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            legend.set_title(legend_title)
            plt.setp(legend.texts)#, family=MONOSPACE_FONT_FAMILY)

    if args.output:
        if args.output.is_dir():
            raise Exception(f"{args.output} is a directory")
        elif args.output.is_file() and not args.force:
            raise Exception(f"{args.is_file} is an existing file. Use -f/--force to overwrite")
        # Save the figure to disk, creating the parent dir if needed.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=config.DPI, bbox_inches='tight')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot CSV data comparing pytorch benchmark results")
    parser.add_argument("-i", "--input", nargs="+", type=pathlib.Path, help="input csv file(s)", required=True)
    parser.add_argument("-o", "--output", type=pathlib.Path, help="output image path. shows if omitted")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--title", type=str, help="Figure title")
    parser.add_argument("--metric", type=str, help="train/eval metric to plot. Must be valid for both, e.g. samples_per_second, steps_per_second, runtime, samples")
    parser.add_argument("--sharey", action="store_true", help="If train and eval should share the y axis")
    parser.add_argument("--errorbar", action="store_true", help="Include error bars")
    parser.add_argument("--no-xlabel", action="store_true", help="Remove X label(s)")
    parser.add_argument("--bar-label-type", type=str, choices=["edge", "center"], help="Add values to bar labels at the specific position")
    parser.add_argument("--bar-label-fmt", type=str, help="Format string for bar labels. E.g %%.3f")
    parser.add_argument("--title-prefix", type=str, help="Prefix for generated figure title")
    parser.add_argument("--legend-loc", type=str, help="Legend location, e.g. 'auto', 'upper left', 'upper right'")
    parser.add_argument("--dpi", type=int, help="Output image DPI")

    precision_group = parser.add_argument_group('precision', description="Control which data should be plotted based on precision. Implicitly plots all data.")
    precision_group.add_argument("--fp16", action="store_true", help="Include FP16 data")
    precision_group.add_argument("--fp32", action="store_true", help="Include FP32 data")

    args = parser.parse_args()

    # Read in dataframes
    dfs = []
    for input_file in args.input:
        df = pd.read_csv(input_file)
        dfs.append(df)

    df = pd.concat(dfs)

    # Filter data by precision if required.
    if (args.fp16 and not args.fp32):
        df = df.query("precision == 16").copy()
    elif (not args.fp16 and args.fp32):
        df = df.query("precision == 32").copy()

    # Make GPU names prettier.
    df.replace({"gpu": {
        # "Tesla V100-SXM2-32GB": "V100 SXM2 32GB",
        "Tesla V100-SXM2-32GB": "V100 SXM2",
        "NVIDIA A100-SXM...": "A100 SXM4",
        "NVIDIA H100 PCIe": "H100 PCIe",
        "NVIDIA GH200 480GB": "GH200 480GB",
    }}, inplace=True)

    df["gpu-short"] = df["gpu"]
    df.replace({"gpu-short": {
        "V100 SXM2 32GB": "V100",
        "V100 SXM2": "V100",
        "A100 SXM4": "A100",
        "H100 PCIe": "H100",
        "GH200 480GB": "GH200",
    }}, inplace=True)

    # Add a new column to the dataframe which is GPU combined with container/pytorch versions
    df["ngcver"] = df["torch_version"]
    df.replace({"ngcver": {
        "2.1.0a0+b5021ba": "23.07",
        "2.3.0a0+ebedce2": "24.02",
    }}, inplace=True)
    df["gpu-ngcver"] = df["gpu-short"] + " " + df["ngcver"]


    # Error if output is a dir, or existing file without force 
    if args.output:
        if args.output.is_dir():
            raise Exception(f"{args.output} is a directory")
        elif args.output.is_file() and not args.force:
            raise Exception(f"{args.is_file} is an existing file. Use -f/--force to overwrite")

    config = Config()
    # Tweak the default config a little depending on what data is present
    torch_version_count = len(df["torch_version"].unique())
    if torch_version_count == 1:
        config.X = "gpu"
        config.XLABEL = "GPU"
        config.HUE = "gpu"
        config.HUELABEL="GPU"
        config.XTICK_ANGLE = 0
    else:
        config.X = "gpu-ngcver"
        config.XLABEL = "GPU + Pytorch"
        config.HUE = "gpu-ngcver"
        config.HUELABEL="GPU + Pytorch"
        config.XTICK_ANGLE = 45

    if args.errorbar:
        config.ERRORBAR = ("pi", 100)
    if args.bar_label_type:
        config.BAR_LABEL_TYPE = args.bar_label_type
    if args.bar_label_fmt:
        config.BAR_LABEL_FMT = args.bar_label_fmt
    if args.dpi:
        config.DPI = args.dpi

    # Plot data
    plot(args, df, config)


if __name__ == "__main__":
    main()