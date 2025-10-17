# Test Instructions

This directory contains automated tests for the Miami plot generator. To run the
full test suite, execute the helper script from the repository root:

```bash
./tests/run_tests.sh
```

Additional arguments are forwarded to `pytest`, so you can run an individual
file like this:

```bash
./tests/run_tests.sh tests/test_format_detection.py
```
