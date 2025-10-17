import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "miami_generator.py"
SPEC = importlib.util.spec_from_file_location("miami_generator", MODULE_PATH)
miami_generator = importlib.util.module_from_spec(SPEC)
sys.modules["miami_generator"] = miami_generator
SPEC.loader.exec_module(miami_generator)  # type: ignore[union-attr]


def _data_path(name: str) -> Path:
    return Path(__file__).parent / "data" / name


def test_read_data_single_file_detection():
    args = miami_generator.parse_args([
        "--data",
        str(_data_path("combined.tsv")),
    ])

    df_up, df_down = miami_generator.read_data(args)

    assert set(df_up.columns) == {"chrom", "pos", "p", "log10p", "group"}
    assert set(df_down.columns) == {"chrom", "pos", "p", "log10p", "group"}
    assert args.chrom == "chrom"
    assert args.pos == "pos"
    assert args.p == "p"
    assert args.logp == "log10p"
    assert args.strata == "group"

    assert args.up_data_label == "CohortA"
    assert args.down_data_label == "CohortB"

    np.testing.assert_allclose(df_up["log10p"].values, [5.0, 0.69897])
    np.testing.assert_allclose(df_down["log10p"].values, [4.0, 1.5228787])


def test_read_data_two_files_detection():
    args = miami_generator.parse_args([
        "--up-data",
        str(_data_path("up.csv")),
        "--down-data",
        str(_data_path("down.tsv")),
        "--up-data-label",
        "Up",
        "--down-data-label",
        "Down",
    ])

    df_up, df_down = miami_generator.read_data(args)

    assert set(df_up.columns) == {"chrom", "pos", "p", "log10p"}
    assert set(df_down.columns) == {"chrom", "pos", "p", "log10p"}
    assert args.chrom == "chrom"
    assert args.pos == "pos"
    assert args.p == "p"
    assert args.logp == "log10p"

    np.testing.assert_allclose(df_up["log10p"].values, [6.0, 3.0, 1.30103])
    np.testing.assert_allclose(
        df_down["log10p"].values,
        -np.log10(df_down["p"].values),
    )
