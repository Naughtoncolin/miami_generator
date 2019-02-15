#!/usr/bin/env python


import sys
import logging
import argparse
from os import path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__author__ = "Louis-Philippe Lemieux Perreault"
__copyright__ = "Copyright 2019, Beaulieu-Saucier Pharmacogenomics Centre"
__credits__ = ["Louis-Philippe Lemieux Perreault"]
__license__ = "MIT"
__maintainer__ = "Louis-Philippe Lemieux Perreault"
__email__ = "louis-philippe.lemieux.perreault@statgen.org"
__status__ = "Development"
__version__ = "0.1.0"


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("miami_generator")


def main():
    # Getting the arguments and options
    args = parse_args()
    check_args(args)

    # Getting the data
    data = read_data(args)

    # Finding the positions' relative start, so that chromosomes align
    max_position, relative_start = find_chrom_relative_start(*data, args)

    # Plotting the data
    create_miami_plot(data, max_position, relative_start, args)


def create_miami_plot(dfs, pos_max, rel_start, args):
    # Creating the figure and axe
    logger.info("Generating the miami plot")
    figure, axe = plt.subplots(
        1, 1, figsize=(args.graph_width, args.graph_height),
    )

    # Plotting the upper part
    plot_group(dfs[0], rel_start, axe, args, colors=args.up_colors)

    # Plotting the lower part
    plot_group(dfs[1], rel_start, axe, args, colors=args.down_colors,
               multiplier=1)

    # Adding the boxes around the even chromosomes
    add_plot_boxes(pos_max, rel_start, axe, args)

    # Adding the chromosome labels
    add_chromosome_labels(pos_max, rel_start, axe, args)

    # Removing some of the spines
    axe.spines["top"].set_visible(False)
    axe.spines["right"].set_visible(False)
    axe.spines["bottom"].set_visible(False)
    axe.xaxis.set_ticks_position("none")

    # Adding axes labels
    axe.set_xlabel("Chromosome", fontsize=args.label_text_size)
    axe.set_ylabel(r"$-\log_{10}(p)$", fontsize=args.label_text_size)

    # Changing the tick size for the Y axis
    for tick in axe.yaxis.get_major_ticks():
        tick.label.set_fontsize(args.axis_text_size)

    # Changing so that they are equal on both sides
    y_min, y_max = axe.get_ylim()
    final_y = max(abs(y_min), y_max)
    axe.set_ylim(-final_y, final_y)

    # Making sure the y tick labels are all positive
    positive_ytick_labels(axe)

    # Creating the ablines
    for abline in args.abline:
        abline = -np.log10(abline)
        if abline <= final_y:
            axe.axhline(y=abline, color="black", ls="--", lw=1, zorder=2)
            axe.axhline(y=-abline, color="black", ls="--", lw=1, zorder=2)

    # Adding the group labels
    add_group_labels(args.up_data_label, args.down_data_label,
                     args.group_text_size, axe)

    # Adding the title (if one)
    if args.graph_title is not None:
        axe.set_title(args.graph_title, weight="bold")

    # Saving the figure
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")


def positive_ytick_labels(axe):
    # Getting the current Y ticks
    yticks = axe.get_yticks()

    # Are they all integer?
    all_integer = True
    for tick in yticks:
        if int(tick) != tick:
            all_integer = False
            break

    new_yticklabels = map(abs, yticks)
    if all_integer:
        new_yticklabels = map(int, new_yticklabels)

    axe.set_yticklabels(new_yticklabels)


def add_group_labels(up_label, down_label, fontsize, axe):
    # Adding the group labels
    axe.text(
        x=0.01, y=0.99, s=f"{up_label}", ha="left", va="top",
        fontsize=fontsize, transform=axe.transAxes,
    )
    axe.text(
        x=0.01, y=0.01, s=f"{down_label}", ha="left", va="bottom",
        fontsize=fontsize, transform=axe.transAxes,
    )


def add_chromosome_labels(pos_max, rel_start, axe, args):
    # The ticks and labels
    xticks = []
    xticks_labels = []

    # Labels goes in the middle of the chromosomes
    for chrom in pos_max.keys():
        min_pos, max_pos = find_chrom_box_start_end(
            pos_max[chrom], rel_start[chrom],
        )

        # The tick position and the labels for this chromosome
        xticks.append((max_pos + min_pos) / 2)
        xticks_labels.append(str(chrom))

    # Changing the ticks
    axe.set_xticks(xticks)
    axe.set_xticklabels(xticks_labels, fontsize=args.chr_text_size)


def add_plot_boxes(pos_max, rel_start, axe, args):
    # Boxes goes to even chromosomes
    for chrom_i, chrom in enumerate(pos_max.keys()):
        if (chrom_i + 1) % 2 == 0:
            min_pos, max_pos = find_chrom_box_start_end(
                pos_max[chrom], rel_start[chrom],
                padding=args.chromosome_spacing / 2,
            )
            axe.axvspan(
                xmin=min_pos, xmax=max_pos, color=args.chromosome_box_color,
                zorder=1,
            )


def find_chrom_box_start_end(pos_max, rel_start, padding=0):
    relative_pos_min = rel_start - padding
    relative_pos_max = pos_max + rel_start + padding

    return relative_pos_min, relative_pos_max


def plot_group(df, rel_start, axe, args, colors, multiplier=-1):
    # Plotting per chromosome
    for chrom_i, (chrom, df) in enumerate(df.groupby(args.chrom, sort=True)):
        # Encoding the chromosome
        chrom = chrom_as_int(chrom)

        # Sorting by positions
        df = df.sort_values(args.pos)

        # Removing points with to high of a p-value
        df = df.loc[df.loc[:, args.p] < args.max_p_value, :]

        # Sampling the data
        if args.p_value_sampling:
            df = df.loc[
                (df.loc[:, args.p] < args.p_sampling_t) |
                (np.random.random(size=df.shape[0]) < args.p_sampling_prop), :
            ]

        # Plotting the normal points
        axe.scatter(
            df.loc[:, args.pos] + rel_start[chrom],
            (np.log10(df.loc[:, args.p])) * multiplier,
            s=args.point_size,
            c=colors[0] if (chrom_i + 1) % 2 == 1 else colors[1],
            zorder=3,
        )

        # Plotting the significant points
        sig_points = df.loc[
            df.loc[:, args.p] < args.significant_threshold, :
        ]
        axe.scatter(
            sig_points.loc[:, args.pos] + rel_start[chrom],
            (np.log10(sig_points.loc[:, args.p])) * multiplier,
            s=args.significant_point_size,
            c=args.significant_color,
            zorder=3,
        )


def find_chrom_relative_start(df1, df2, args):
    # The biggest positions of all chromosome
    chrom_max = defaultdict(int)

    # The first data set
    for chrom, df in df1.groupby(args.chrom, sort=True):
        chrom = chrom_as_int(chrom)
        chrom_max[chrom] = df.loc[:, args.pos].max()

    # The second data set
    for chrom, df in df2.groupby(args.chrom, sort=True):
        chrom = chrom_as_int(chrom)
        chrom_max[chrom] = max(chrom_max[chrom], df.loc[:, args.pos].max())

    # Computing the relative starting position
    relative_start = {1: 0}
    for chrom in sorted(chrom_max.keys()):
        if chrom != 1:
            # Finding the previous chromosome
            previous_chrom = chrom - 1
            while previous_chrom not in chrom_max and previous_chrom > 1:
                previous_chrom -= 1

            relative_start[chrom] = (
                relative_start[previous_chrom] +
                chrom_max.get(previous_chrom, 0) +
                args.chromosome_spacing
            )

    return chrom_max, relative_start


def read_data(args):
    # The up and down dataset
    df_1 = None
    df_2 = None

    # The column to read (to speed up the IO)
    cols = [args.chrom, args.pos, args.p]

    # Dataset is in a single file
    if args.data is not None:
        # We need the group column
        logger.info(f"Reading '{args.data}'")
        cols.append(args.strata)

        # Reading the file
        df = None
        try:
            df = pd.read_csv(args.data, sep=args.sep, low_memory=False,
                             usecols=cols)
        except ValueError as e:
            logger.error(f"{args.data}: {str(e)}")
            sys.exit(1)

        # Checking the number of group
        unique_group = df.loc[:, args.strata].unique()
        if len(unique_group) != 2:
            logger.error(f"{args.data}: need exactly two groups "
                         f"'{args.strata}'")
            sys.exit(1)

        # Splitting
        df_1 = df.loc[df.loc[:, args.strata] == unique_group[0], :]
        df_2 = df.loc[df.loc[:, args.strata] == unique_group[1], :]

        # Setting the label if not already set by the user
        if args.up_data_label is None:
            args.up_data_label = unique_group[0]
        if args.down_data_label is None:
            args.down_data_label = unique_group[1]

    # There is two datasets
    else:
        # Reading the first file
        logger.info(f"Reading '{args.up_data}'")
        try:
            df_1 = pd.read_csv(args.up_data, sep=args.sep, low_memory=False,
                               usecols=cols)
        except ValueError as e:
            logger.error(f"{args.up_data}: {str(e)}")
            sys.exit(1)

        # Reading the second file
        logger.info(f"Reading '{args.down_data}'")
        try:
            df_2 = pd.read_csv(args.down_data, sep=args.sep, low_memory=False,
                               usecols=cols)
        except ValueError as e:
            logger.error(f"{args.up_data}: {str(e)}")
            sys.exit(1)

    return df_1, df_2


def chrom_as_int(chrom):
    if chrom == "X":
        return 23
    if chrom == "Y":
        return 24
    if chrom == "XY":
        return 25
    if chrom == "MT":
        return 26

    try:
        return int(chrom)
    except ValueError():
        logger.error(f"{chrom}: invalid chromosome")
        sys.exit(1)


def check_args(args):
    # Checking a single data file
    if args.data is not None:
        # No other dataset is specified
        if args.up_data is not None or args.down_data is not None:
            logger.error("The '--data' option cannot be used with either "
                         "'--up-data' or '--down-data'.")
            sys.exit(1)

        # The file exists
        if not path.isfile(args.data):
            logger.error(f"{args.data}: no such file")
            sys.exit(1)

        # The column for grouping is set
        if args.strata is None:
            logger.error(f"{args.data}: missing grouping column (--group)")
            sys.exit(1)

    # Checking both data files
    elif args.up_data is not None and args.down_data is not None:
        for fn in (args.up_data, args.down_data):
            if not path.isfile(fn):
                logger.error(f"{fn}: no such file")
                sys.exit(1)
        if args.up_data_label is None or args.down_data_label is None:
            logger.error("No dataset labels were provided (use "
                         "'--up-data-label' and '--down-data-label').")
            sys.exit(1)

    # Problem with input file(s)
    else:
        logger.error("Missing input file(s).")
        sys.exit(1)

    # Checking the output file type
    if not args.output.endswith(args.format):
        args.output += "." + args.format

    # Checking if there is a horizontal line at the significant threshold
    if args.abline is None:
        args.abline = [args.significant_threshold]
    elif args.significant_threshold not in args.abline:
        args.abline.append(args.significant_threshold)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Creates a Miami plot from two datasets.",
    )

    # The input files
    group = parser.add_argument_group("Input options")
    group.add_argument(
        "--data", type=str, metavar="FILE",
        help="A file containing two datasets. You need to specify the column "
             "which stratifies the two datasets.",
    )
    group.add_argument(
        "--up-data", type=str, metavar="FILE",
        help="A file containing one of the two datasets. This dataset will be "
             "at the top of the Miami plot.",
    )
    group.add_argument(
        "--up-data-label", type=str, metavar="LABEL",
        help="The label for the datasets that will be at the top of the "
             "Miami plot. If this value is not set and the data comes from a "
             "single file, the label is automatically generated.",
    )
    group.add_argument(
        "--down-data", type=str, metavar="FILE",
        help="A file containing one of the two datasets. This dataset will be "
             "at the bottom of the Miami plot.",
    )
    group.add_argument(
        "--down-data-label", type=str, metavar="LABEL",
        help="The label for the datasets that will be at the bottom of the "
             "Miami plot. If this value is not set and the data comes from a "
             "single file, the label is automatically generated.",
    )
    # The columns to gather information
    group = parser.add_argument_group("Data format")
    group.add_argument(
        "--separator", type=str, metavar="SEP", default="\t", dest="sep",
        help="The file(s) field separator. [TAB]",
    )
    group.add_argument(
        "--chromosome", type=str, metavar="CHR", default="chr", dest="chrom",
        help="The column containing the chromosomes. [%(default)s]",
    )
    group.add_argument(
        "--position", type=str, metavar="POS", default="pos", dest="pos",
        help="The column containing the positions. [%(default)s]",
    )
    group.add_argument(
        "--p-value", type=str, metavar="P", default="p", dest="p",
        help="The column containing the p-values. [%(default)s]",
    )
    group.add_argument(
        "--group", type=str, metavar="COL", dest="strata",
        help="The column containing the groups (for single file dataset).",
    )

    # The output options
    group = parser.add_argument_group("Output options")
    group.add_argument(
        "--output", type=str, default="miami", metavar="FILE",
        help="The name of the output file. The extension will be added if "
             "missing. [%(default)s]",
    )
    format_choices = ["pdf", "png"]
    group.add_argument(
        "--format", type=str, default="png", metavar="FORMAT",
        choices=format_choices,
        help="The format of the plot ({}). "
             "[%(default)s]".format(", ".join(format_choices)),
    )
    group.add_argument(
        "--dpi", type=int, default=600, metavar="INT",
        help="The quality of the output (in dpi). [%(default)d]",
    )

    # Plot coloring
    group = parser.add_argument_group("Graph color")
    group.add_argument(
        "--up-colors", type=str, metavar="COL", nargs=2,
        default=("#0099CC", "#33B5E5"),
        help="The two colors (odd and even chromosomes) for the upper part of "
             "the Miami plot.",
    )
    group.add_argument(
        "--down-colors", type=str, metavar="COL", nargs=2,
        default=("#669900", "#99CC00"),
        help="The two colors (odd and even chromosomes) for the upper part of "
             "the Miami plot.",
    )
    group.add_argument(
        "--chromosome-box-color", type=str, default="#E5E5E5", metavar="COLOR",
        help="The COLOR for the box surrounding even chromosome numbers. "
             "[%(default)s]",
    )
    group.add_argument(
        "--significant-color", type=str, default="#FF0000", metavar="COLOR",
        help="The color for points representing significant points. "
             "[%(default)s]",
    )

    # Plot text and style
    group = parser.add_argument_group("Graph text")
    group.add_argument(
        "--graph-title", type=str, dest='graph_title', metavar="TITLE",
        help="The TITLE of the graph.",
    )
    group.add_argument(
        "--axis-text-size", type=int, default=10, metavar="INT",
        help="The axis font size. [%(default)d]",
    )
    group.add_argument(
        "--chr-text-size", type=int, default=10, metavar="INT",
        help="The axis font size. [%(default)d]",
    )
    group.add_argument(
        "--label-text-size", type=int, default=12, metavar="INT",
        help="The axis font size. [%(default)d]",
    )
    group.add_argument(
        "--group-text-size", type=int, default=8, metavar="INT",
        help="The group label font size. [%(default)d]",
    )

    # The look of the graph
    group = parser.add_argument_group("Graph look")
    group.add_argument(
        "--graph-width", type=float, default=14, metavar="WIDTH",
        help="The width of the graph, in inches. [%(default)d]",
    )
    group.add_argument(
        "--graph-height", type=float, default=4, metavar="HEIGHT",
        help="The height of the graph, in inches. [%(default)d]",
    )
    group.add_argument(
        "--abline", type=float, metavar="P", nargs="+",
        help="The p-value(s) at which a horizontal line should be drawn.",
    )
    group.add_argument(
        "--point-size", type=float, default=0.5, metavar="SIZE",
        help="The size of each points. [%(default).1f]",
    )
    group.add_argument(
        "--significant-point-size", type=float, default=4.0, metavar="SIZE",
        help="The size of each significant points. [%(default).1f]",
    )
    group.add_argument(
        "--chromosome-spacing", type=float, metavar="VALUE", default=25e6,
        help="The spacing between two chromosomes. [%(default).1E]",
    )
    group.add_argument(
        "--significant-threshold", type=float, default=5e-8, metavar="FLOAT",
        help="The significant threshold for linkage or association. "
             "[%(default).1E]",
    )

    # Optimization options (to make the plot faster
    group = parser.add_argument_group("Optimization")
    group.add_argument(
        "--max-p-value", type=float, metavar="FLOAT", default=1.0,
        help="Prevent plotting p-values higher than this cutoff. "
             "[%(default).1f]",
    )
    group.add_argument(
        "--p-value-sampling", action="store_true",
        help="Sample a subset of points for specific p-values (see "
             "'--p-value-sampling-threshold' and "
             "'--p-value-sampling-proportion'). [%(default)s]",
    )
    group.add_argument(
        "--p-value-sampling-threshold", type=float, metavar="FLOAT",
        default=0.01, dest="p_sampling_t",
        help="Points with p-values equal or higher than this threshold will "
             "be sampled. [%(default).2f]",
    )
    group.add_argument(
        "--p-value-sampling-proportion", type=float, metavar="FLOAT",
        default=0.1, dest="p_sampling_prop",
        help="The proportion of points with a p-values equal or higher than "
             "the threshold which will be kept for plotting. "
             "[%(default).1f]",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
