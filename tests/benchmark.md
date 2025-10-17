# Benchmarking the Polars migration

The `tests/benchmark_miami.py` helper script generates temporary synthetic
GWAS-style datasets, runs the `miami_generator` data-processing pipeline using
both the new Polars implementation and a re-created copy of the legacy
pandas/numpy logic, and prints a timing table. The CSV files are written to a
`TemporaryDirectory` and automatically deleted after the benchmark finishes.

## Running the benchmark

1. Ensure the optional baseline dependencies are installed:
   ```bash
   pip install pandas numpy
   ```
2. Run the benchmark script (feel free to customise the dataset sizes and
   random seed):
   ```bash
   python tests/benchmark_miami.py --sizes 500000 1000000 10000000 20000000 --seed 2024
   ```
3. Review the printed timing table. The `Speedup` column expresses how many
   times faster the Polars pipeline executed compared to the pandas baseline.

> **Note**
> Large dataset sizes (10M and 20M rows) can require significant RAM and
> execution time. Reduce the `--sizes` values when experimenting on smaller
> machines.

The script is designed for manual benchmarking and does not run automatically
as part of the test suite; invoke it when evaluating future performance
changes.
