"""Benchmark Polars-based pipeline against the legacy pandas implementation.

This script generates synthetic GWAS-like datasets of varying sizes, runs the
core Miami plot preparation routines using both the Polars-powered
``miami_generator`` module and a reimplementation of the original pandas/
numpy-based logic, and reports the elapsed time for each approach.

The generated CSV files are stored in a temporary directory and removed at the
end of the run to avoid consuming disk space.
"""

from __future__ import annotations

import argparse
import copy
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl

try:  # pragma: no cover - optional dependency for baseline benchmark
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The benchmarking script requires both pandas and numpy to reproduce "
        "the legacy pipeline. Please install them before running the script."
    ) from exc

import miami_generator


DEFAULT_SIZES: Tuple[int, ...] = (500_000, 1_000_000, 10_000_000, 20_000_000)
ALLELES: Tuple[str, ...] = ("A", "C", "G", "T")
CHROMS: Tuple[str, ...] = tuple(str(i) for i in range(1, 23)) + ("X",)


def generate_dataset(num_rows: int, destination: Path) -> None:
    """Create a synthetic dataset with two strata and realistic columns."""
    half = num_rows // 2
    remainder = num_rows - half

    chroms = random.choices(CHROMS, k=num_rows)
    positions = [random.randint(1, 200_000_000) for _ in range(num_rows)]
    p_values = [random.random() for _ in range(num_rows)]
    other_alleles = random.choices(ALLELES, k=num_rows)
    effect_alleles = random.choices(ALLELES, k=num_rows)
    strata = ["UP"] * half + ["DOWN"] * remainder
    random.shuffle(strata)

    pl.DataFrame(
        {
            "chrom": chroms,
            "pos": positions,
            "p": p_values,
            "other": other_alleles,
            "effect": effect_alleles,
            "group": strata,
        }
    ).write_csv(destination)


def create_args(csv_path: Path) -> argparse.Namespace:
    """Construct the minimal namespace required by the pipelines."""
    return argparse.Namespace(
        data=str(csv_path),
        up_data=None,
        down_data=None,
        up_data_label=None,
        down_data_label=None,
        show_dataset_specific_variants=True,
        other_allele="other",
        effect_allele="effect",
        chrom="chrom",
        pos="pos",
        p="p",
        strata="group",
        sep=",",
        chromosome_spacing=25e6,
        significant_threshold=5e-8,
        max_p_value=1.0,
        p_value_sampling=False,
        p_sampling_t=0.01,
        p_sampling_prop=0.1,
    )


def polars_pipeline(args: argparse.Namespace) -> None:
    """Run the Polars-backed pipeline used by ``miami_generator``."""
    df1, df2 = miami_generator.read_data(args)
    if args.show_dataset_specific_variants:
        df1, df2 = miami_generator.check_significance_in_datasets(df1, df2, args)
    miami_generator.find_chrom_relative_start(df1, df2, args)


def pandas_read_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols: List[str] = [args.chrom, args.pos, args.p]
    if args.show_dataset_specific_variants:
        cols.extend([args.other_allele, args.effect_allele])

    df_1: pd.DataFrame | None = None
    df_2: pd.DataFrame | None = None

    if args.data is not None:
        cols_with_group = cols + [args.strata]
        df = pd.read_csv(args.data, sep=args.sep, low_memory=False, usecols=cols_with_group)
        unique_group = df.loc[:, args.strata].unique()
        df_1 = df.loc[df.loc[:, args.strata] == unique_group[0], :]
        df_2 = df.loc[df.loc[:, args.strata] == unique_group[1], :]
    else:
        df_1 = pd.read_csv(args.up_data, sep=args.sep, low_memory=False, usecols=cols)
        df_2 = pd.read_csv(args.down_data, sep=args.sep, low_memory=False, usecols=cols)

    return df_1, df_2


def pandas_check_significance(
    df1: pd.DataFrame, df2: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _create_id(df: pd.DataFrame) -> pd.Series:
        alleles_df = df[[args.other_allele, args.effect_allele]].astype(str)
        return (
            df[args.chrom].astype(str)
            + ":"
            + df[args.pos].astype(str)
            + "_"
            + alleles_df.min(axis=1)
            + "/"
            + alleles_df.max(axis=1)
        )

    threshold = -np.log10(args.significant_threshold)

    df1 = df1.copy()
    df2 = df2.copy()
    df1["_id"] = _create_id(df1)
    df2["_id"] = _create_id(df2)
    df1["nlogp_1"] = -np.log10(df1[args.p])
    df2["nlogp_2"] = -np.log10(df2[args.p])

    merged = pd.merge(df1, df2, how="outer", on="_id")
    sig_1 = merged.nlogp_1 > threshold
    sig_2 = merged.nlogp_2 > threshold
    significance_df = pd.DataFrame(
        {
            "BOTH": sig_1 & sig_2,
            "1_ONLY": sig_1 & (~sig_2),
            "2_ONLY": sig_2 & (~sig_1),
            "NON_SIGNIFICANT": (~sig_1) & (~sig_2),
        }
    )
    merged["significance"] = significance_df.idxmax(axis=1).astype("category")
    merged = merged[["_id", "significance"]]

    df1 = pd.merge(df1, merged, how="left", on="_id").drop(columns=["_id", "nlogp_1"])
    df2 = pd.merge(df2, merged, how="left", on="_id").drop(columns=["_id", "nlogp_2"])
    return df1, df2


def pandas_find_chrom_relative_start(
    df1: pd.DataFrame, df2: pd.DataFrame, args: argparse.Namespace
) -> None:
    chrom_max: dict[int, float] = defaultdict(int)

    for chrom, df in df1.groupby(args.chrom, sort=True):
        chrom_max[miami_generator.chrom_as_int(chrom)] = df.loc[:, args.pos].max()

    for chrom, df in df2.groupby(args.chrom, sort=True):
        chrom_int = miami_generator.chrom_as_int(chrom)
        chrom_max[chrom_int] = max(chrom_max[chrom_int], df.loc[:, args.pos].max())

    relative_start = {1: 0.0}
    for chrom in sorted(chrom_max.keys()):
        if chrom == 1:
            continue
        previous_chrom = chrom - 1
        while previous_chrom not in chrom_max and previous_chrom > 1:
            previous_chrom -= 1
        relative_start[chrom] = (
            relative_start.get(previous_chrom, 0.0)
            + chrom_max.get(previous_chrom, 0.0)
            + args.chromosome_spacing
        )

    return None


def pandas_pipeline(args: argparse.Namespace) -> None:
    df1, df2 = pandas_read_data(args)
    if args.show_dataset_specific_variants:
        df1, df2 = pandas_check_significance(df1, df2, args)
    pandas_find_chrom_relative_start(df1, df2, args)


def run_benchmark(sizes: Iterable[int], seed: int) -> List[Tuple[int, float, float]]:
    random.seed(seed)
    results: List[Tuple[int, float, float]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        for size in sizes:
            csv_path = temp_path / f"synthetic_{size}.csv"
            print(f"Generating dataset with {size:,} rows...")
            generate_dataset(size, csv_path)

            polars_args = create_args(csv_path)
            pandas_args = copy.deepcopy(polars_args)

            print("  Running Polars pipeline...")
            start = time.perf_counter()
            polars_pipeline(polars_args)
            polars_time = time.perf_counter() - start

            print("  Running pandas pipeline...")
            start = time.perf_counter()
            pandas_pipeline(pandas_args)
            pandas_time = time.perf_counter() - start

            results.append((size, polars_time, pandas_time))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Dataset sizes (number of rows) to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed used for reproducible dataset generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(args.sizes, args.seed)

    print("\nBenchmark results (seconds):")
    print(f"{'Rows':>12} | {'Polars':>10} | {'Pandas':>10} | {'Speedup':>8}")
    print("-" * 50)
    for size, polars_time, pandas_time in results:
        speedup = pandas_time / polars_time if polars_time else float('inf')
        print(
            f"{size:>12,} | {polars_time:>10.2f} | {pandas_time:>10.2f} | {speedup:>8.2f}"
        )


if __name__ == "__main__":
    main()
