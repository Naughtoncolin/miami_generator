#!/usr/bin/env python
"""Create miami plots from two categories."""


import csv
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
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


DEFAULT_DELIMITERS = [",", "\t", ";", "|", " "]

COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "chrom": (
        "chr",
        "chrom",
        "chromosome",
        "chromosomeid",
        "chromosome_name",
        "chromosomecode",
    ),
    "pos": (
        "pos",
        "position",
        "bp",
        "basepair",
        "base_pair",
        "bp_position",
        "bp_pos",
        "bpcoordinate",
        "pos_bp",
        "position_bp",
    ),
    "p": (
        "p",
        "pvalue",
        "p_value",
        "pval",
        "pval_nominal",
        "pvalue_nominal",
        "pvalueall",
        "pvaluemeta",
        "pvaluefixed",
        "pvaluerandom",
        "pvaluegc",
    ),
    "log10p": (
        "log10p",
        "log10pvalue",
        "log10pval",
        "logp",
        "logpvalue",
        "minuslog10p",
        "minuslogp",
        "neglog10p",
        "neglogp",
        "nlog10p",
        "mlogp",
        "mlog10p",
        "neglog10",
        "neglog",
    ),
    "effect_allele": (
        "effect_allele",
        "effectallele",
        "allele1",
        "a1",
        "alt",
        "alt_allele",
        "ea",
    ),
    "other_allele": (
        "other_allele",
        "otherallele",
        "allele2",
        "a2",
        "ref",
        "ref_allele",
        "oa",
    ),
    "group": (
        "group",
        "strata",
        "cohort",
        "dataset",
        "category",
        "phenotype",
        "trait",
        "label",
    ),
}


@dataclass
class DatasetFormat:
    """Metadata describing how to read a dataset."""

    separator: str
    column_map: Dict[str, Optional[str]]
    original_columns: Tuple[str, ...]


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _sniff_separator(file_path: str, provided: Optional[str]) -> str:
    """Detect the separator for a file."""

    if provided:
        return provided

    with open(file_path, "r", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=DEFAULT_DELIMITERS)
            return dialect.delimiter
        except csv.Error:
            if "\t" in sample:
                return "\t"
            if "," in sample:
                return ","
            return "\t"


def _build_column_overrides(args) -> Dict[str, Optional[str]]:
    """Capture user requested column names before detection mutates args."""

    overrides = {
        "chrom": getattr(args, "chrom", None),
        "pos": getattr(args, "pos", None),
        "p": getattr(args, "p", None),
        "log10p": getattr(args, "logp", None),
        "other_allele": getattr(args, "other_allele", None),
        "effect_allele": getattr(args, "effect_allele", None),
        "group": getattr(args, "strata", None),
    }

    return {key: value for key, value in overrides.items() if value}


def _find_matching_column(header: Iterable[str], canonical: str) -> Optional[str]:
    aliases = set(COLUMN_ALIASES.get(canonical, [])) | {canonical}
    normalized_aliases = {_normalize_column_name(alias) for alias in aliases}

    for col in header:
        if _normalize_column_name(col) in normalized_aliases:
            return col
    return None


def detect_dataset_format(
    file_path: str,
    overrides: Dict[str, str],
    include_optional: Iterable[str],
    require_group: bool,
    provided_separator: Optional[str],
) -> DatasetFormat:
    """Detect the dataset separator and relevant column names."""

    separator = _sniff_separator(file_path, provided_separator)

    with open(file_path, "r", newline="") as handle:
        reader = csv.reader(handle, delimiter=separator)
        try:
            header = next(reader)
        except StopIteration:
            logger.error(f"{file_path}: file is empty")
            sys.exit(1)

    header = [column.strip() for column in header]
    column_map: Dict[str, Optional[str]] = {}
    required = ["chrom", "pos"]
    optional = {"p", "log10p"}
    optional.update(include_optional)
    if require_group or "group" in overrides:
        optional.add("group")

    used_columns = set()

    def _assign_column(canonical: str, column: Optional[str]):
        if column is not None:
            column_map[canonical] = column
            used_columns.add(column)
        else:
            column_map[canonical] = None

    # First honour overrides.
    for canonical, column in overrides.items():
        if canonical not in required and canonical not in optional:
            optional.add(canonical)
        if column not in header:
            logger.error(f"{file_path}: column '{column}' not found")
            sys.exit(1)
        _assign_column(canonical, column)

    # Detect required columns.
    for canonical in required + sorted(optional):
        if canonical in column_map and column_map[canonical] is not None:
            continue
        match = _find_matching_column(header, canonical)
        if match and match not in used_columns:
            _assign_column(canonical, match)
        else:
            _assign_column(canonical, None)

    if column_map["chrom"] is None or column_map["pos"] is None:
        logger.error(
            f"{file_path}: unable to detect chromosome and position columns"
        )
        sys.exit(1)

    if column_map.get("p") is None and column_map.get("log10p") is None:
        logger.error(
            f"{file_path}: unable to detect either a p-value or log10(p) column"
        )
        sys.exit(1)

    for canonical in include_optional:
        if column_map.get(canonical) is None:
            logger.error(
                f"{file_path}: unable to detect required column '{canonical}'"
            )
            sys.exit(1)

    if require_group and column_map.get("group") is None:
        logger.error(f"{file_path}: unable to detect grouping column")
        sys.exit(1)

    return DatasetFormat(
        separator=separator,
        column_map=column_map,
        original_columns=tuple(header),
    )


def load_dataset(
    file_path: str,
    overrides: Dict[str, str],
    include_optional: Iterable[str],
    require_group: bool,
    provided_separator: Optional[str],
) -> Tuple[pd.DataFrame, DatasetFormat]:
    """Load a dataset with automatically detected format information."""

    format_info = detect_dataset_format(
        file_path,
        overrides=overrides,
        include_optional=include_optional,
        require_group=require_group,
        provided_separator=provided_separator,
    )

    use_columns = [
        column
        for column in format_info.column_map.values()
        if column is not None
    ]

    try:
        df = pd.read_csv(
            file_path,
            sep=format_info.separator,
            low_memory=False,
            usecols=use_columns if use_columns else None,
        )
    except ValueError as exc:
        logger.error(f"{file_path}: {exc}")
        sys.exit(1)

    rename_map = {
        original: canonical
        for canonical, original in format_info.column_map.items()
        if original is not None
    }
    df = df.rename(columns=rename_map)

    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str)

    if "pos" in df.columns:
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
        if df["pos"].isnull().any():
            logger.error(f"{file_path}: non numeric positions detected")
            sys.exit(1)

    if "p" in df.columns:
        df["p"] = pd.to_numeric(df["p"], errors="coerce")

    if "log10p" in df.columns:
        df["log10p"] = pd.to_numeric(df["log10p"], errors="coerce")

    if "log10p" not in df.columns:
        if "p" not in df.columns:
            logger.error(
                f"{file_path}: unable to compute log10(p) without p-values"
            )
            sys.exit(1)
        with np.errstate(divide="ignore"):
            df["log10p"] = -np.log10(df["p"])

    if "p" not in df.columns:
        with np.errstate(over="ignore"):
            df["p"] = np.power(10.0, -df["log10p"])

    if "effect_allele" in df.columns:
        df["effect_allele"] = df["effect_allele"].astype(str)
    if "other_allele" in df.columns:
        df["other_allele"] = df["other_allele"].astype(str)

    if require_group and "group" in df.columns:
        df["group"] = df["group"].astype(str)

    return df, format_info

def main():
    """Create miami plots from two categories."""
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
    """Create the Miami plot."""
    # Creating the figure and axe
    logger.info("Generating the miami plot")
    figure, axe = plt.subplots(
        1, 1, figsize=(args.graph_width, args.graph_height),
    )

    # Join the two datasets and find the dataset specific variants if needed.
    if args.show_dataset_specific_variants:
        dfs = check_significance_in_datasets(*dfs, args)

    # Plotting the upper part
    plot_group(dfs[0], rel_start, axe, args, up=True)

    # Plotting the lower part
    plot_group(dfs[1], rel_start, axe, args, up=False)

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
    "Changing the Y labels to positive numbers."
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
    """Add the group name (label) in each subplot."""
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
    """Add the chromosome labels in the X axis."""
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
    """Add alternating boxes (to highlight chromosome boundaries)."""
    # Boxes goes to even chromosomes indexes because we want alternating colors
    # even between 23 and 25 (24 is mostly never analyzed)
    for chrom_i, chrom in enumerate(sorted(pos_max.keys())):
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
    """Finds the chromosome box start."""
    relative_pos_min = rel_start - padding
    relative_pos_max = pos_max + rel_start + padding

    return relative_pos_min, relative_pos_max


def plot_group(df, rel_start, axe, args, up):
    """Generate the scatter plot for a single group (each chromosome)."""
    if up:
        multiplier = -1
        colors = args.up_colors
    else:
        multiplier = 1
        colors = args.down_colors

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

        if args.show_dataset_specific_variants:
            scatter_with_dataset_specific_significance(
                df, rel_start, axe, args, colors, multiplier, chrom, chrom_i
            )
        else:
            scatter_standard(
                df, rel_start, axe, args, colors, multiplier, chrom, chrom_i
            )


def scatter_with_dataset_specific_significance(
    df, rel_start, axe, args, colors, multiplier, chrom, chrom_i
):
    """Generate the scatter plot for a single group (specific significance)."""
    doing_dataset_1 = (multiplier == -1)

    markers = {
        "NON_SIGNIFICANT": "o",
        "1_ONLY": "^",
        "2_ONLY": "v",
        "BOTH": "o",
    }

    for level, marker in markers.items():
        color = colors[0] if (chrom_i + 1) % 2 == 1 else colors[1]

        if (level == "NON_SIGNIFICANT"):
            # The variant is not significant anywhere, we use default color and
            # size.
            s = args.point_size

        elif (
            ((doing_dataset_1) and level == "2_ONLY") or
            ((not doing_dataset_1) and level == "1_ONLY")
        ):
            # The is significant in the other dataset only.
            # We use the regular color, but still make the marker larger to
            # make sure that it is visible.
            s = args.significant_point_size

        else:
            # The variant is significant in the current dataset.
            s = args.significant_point_size
            color = args.significant_color

        cur = df.loc[df.significance == level, :]
        axe.scatter(
            cur[args.pos] + rel_start[chrom],
            cur[args.logp] * (-multiplier),
            s=s,
            marker=marker,
            c=color,
            zorder=3
        )


def scatter_standard(df, rel_start, axe, args, colors, multiplier, chrom,
                     chrom_i):
    """Generate the scatter plot for a single group."""
    # Plotting the normal points
    axe.scatter(
        df.loc[:, args.pos] + rel_start[chrom],
        df.loc[:, args.logp] * (-multiplier),
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
        sig_points.loc[:, args.logp] * (-multiplier),
        s=args.significant_point_size,
        c=args.significant_color,
        zorder=3,
    )


def find_chrom_relative_start(df1, df2, args):
    """Find chromosome relative start (according to previous chromosome)."""
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


def read_data(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the data file (splitting a single file in two if necessary)."""

    separator_override = None
    if getattr(args, "sep", None) not in (None, "auto"):
        separator_override = args.sep

    overrides = _build_column_overrides(args)
    optional_columns = []
    if args.show_dataset_specific_variants:
        optional_columns.extend(["other_allele", "effect_allele"])

    args.logp = "log10p"

    # Dataset is in a single file
    if args.data is not None:
        logger.info(f"Reading '{args.data}'")

        df, _ = load_dataset(
            args.data,
            overrides=overrides,
            include_optional=optional_columns,
            require_group=True,
            provided_separator=separator_override,
        )

        args.chrom = "chrom"
        args.pos = "pos"
        args.p = "p"
        args.strata = "group"

        # Checking the number of group
        unique_group = df.loc[:, args.strata].unique()
        if len(unique_group) != 2:
            logger.error(
                f"{args.data}: need exactly two groups '{args.strata}'"
            )
            sys.exit(1)

        # Splitting
        df_1 = df.loc[df.loc[:, args.strata] == unique_group[0], :].copy()
        df_2 = df.loc[df.loc[:, args.strata] == unique_group[1], :].copy()

        # Setting the label if not already set by the user
        if args.up_data_label is None:
            args.up_data_label = unique_group[0]
        if args.down_data_label is None:
            args.down_data_label = unique_group[1]

        return df_1, df_2

    # There are two datasets
    logger.info(f"Reading '{args.up_data}'")
    df_1, _ = load_dataset(
        args.up_data,
        overrides=overrides,
        include_optional=optional_columns,
        require_group=False,
        provided_separator=separator_override,
    )

    logger.info(f"Reading '{args.down_data}'")
    df_2, _ = load_dataset(
        args.down_data,
        overrides=overrides,
        include_optional=optional_columns,
        require_group=False,
        provided_separator=separator_override,
    )

    args.chrom = "chrom"
    args.pos = "pos"
    args.p = "p"

    if args.show_dataset_specific_variants:
        missing_columns = []
        for column in ["other_allele", "effect_allele"]:
            if column not in df_1.columns or column not in df_2.columns:
                missing_columns.append(column)
        if missing_columns:
            logger.error(
                "Missing columns in dataset specific variant detection: "
                + ", ".join(missing_columns)
            )
            sys.exit(1)

    # Ensure both datasets contain the same set of columns
    if set(df_1.columns) != set(df_2.columns):
        diff = sorted(set(df_1.columns).symmetric_difference(set(df_2.columns)))
        logger.error(
            "Input files must contain matching columns. Mismatched columns: "
            + ", ".join(diff)
        )
        sys.exit(1)

    return df_1, df_2


def check_significance_in_datasets(
    df1,
    df2,
    args
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Checks significance of variants in the two datasets after joining.

    The returned series contains factors with levels:
        - 1_ONLY
        - 2_ONLY
        - BOTH
        - NON_SIGNIFICANT

    """
    # Create unique identifiers for variants to allow joining.
    def _create_id(df) -> pd.Series:
        alleles_df = df[[args.other_allele, args.effect_allele]]\
            .astype(str)

        return (
            df[args.chrom].astype(str) + ":" +
            df[args.pos].astype(str) + "_" +
            alleles_df.min(axis=1) + "/" +
            alleles_df.max(axis=1)
        )

    threshold = -np.log10(args.significant_threshold)

    df1["_id"] = _create_id(df1)
    df2["_id"] = _create_id(df2)

    # Compute -log10(p) for both datasets.
    df1["nlogp_1"] = df1[args.logp]
    df2["nlogp_2"] = df2[args.logp]

    merged = pd.merge(df1, df2, how="outer", on="_id")

    # Check the different possible types of significance.
    sig_1 = merged.nlogp_1 > threshold
    sig_2 = merged.nlogp_2 > threshold
    sig_both = sig_1 & sig_2
    sig_1_only = sig_1 & (~sig_2)
    sig_2_only = sig_2 & (~sig_1)
    ns = (~sig_1) & (~sig_2)

    significance_df = pd.DataFrame({
        "BOTH": sig_both,
        "1_ONLY": sig_1_only,
        "2_ONLY": sig_2_only,
        "NON_SIGNIFICANT": ns
    })

    merged["significance"] = significance_df.idxmax(axis=1).astype("category")
    merged = merged[["_id", "significance"]]

    return (
        pd.merge(df1, merged, how="left", on="_id"),
        pd.merge(df2, merged, how="left", on="_id"),
    )


def chrom_as_int(chrom):
    """Convert string chromosome to integer."""
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
    """Verify arguments and options."""
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


def parse_args(argv=None):
    """Parse arguments and options."""
    parser = argparse.ArgumentParser(
        description="Creates a Miami plot from two datasets.",
    )
    parser.set_defaults(logp=None)

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
        "--separator", type=str, metavar="SEP", default="auto", dest="sep",
        help="The file(s) field separator. Automatically detected when set to"
             " 'auto'. [%(default)s]",
    )
    group.add_argument(
        "--chromosome", type=str, metavar="CHR", default=None, dest="chrom",
        help="The column containing the chromosomes. [auto]",
    )
    group.add_argument(
        "--position", type=str, metavar="POS", default=None, dest="pos",
        help="The column containing the positions. [auto]",
    )
    group.add_argument(
        "--other-allele", type=str, metavar="OTHER_ALLELE", default="",
        help="Reference/non-coded/other allele to be used when computing "
             "dataset specific significance."
    )
    group.add_argument(
        "--effect-allele", type=str, metavar="EFFECT_ALLELE", default="",
        help="Reference/non-coded/other allele to be used when computing "
             "dataset specific significance."
    )
    group.add_argument(
        "--p-value", type=str, metavar="P", default=None, dest="p",
        help="The column containing the p-values. [auto]",
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
    group.add_argument(
        "--show-dataset-specific-variants",
        action="store_true",
        help="Use a different color for variants that are significant in the "
             "top only, bottom only or both datasets."
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

    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
