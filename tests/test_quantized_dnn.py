"""Tests for the quantized DNN benchmark binary.

The option tests run without a GPU, because the options are checked before
CUDA starts. The GPU tests are skipped when no CUDA device is available. Set
BENCHMARKS_BUILD_DIR to use a build directory other than build/.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

BUILD_DIRECTORY = Path(
    os.environ.get(
        "BENCHMARKS_BUILD_DIR", Path(__file__).resolve().parents[1] / "build"
    )
)
BINARY = BUILD_DIRECTORY / "bench_quantized_dnn"
TIMEOUT_SECONDS = 300
NO_DEVICE_MARKERS = ("cudaGetDeviceCount", "requested CUDA device 0")

# A small network keeps the measure test short on every GPU.
SMALL_NETWORK = [
    "--batch",
    "64",
    "--input-features",
    "64",
    "--hidden-features",
    "128",
    "--hidden-layers",
    "1",
    "--outputs",
    "64",
]


def run_benchmark(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Runs the benchmark and captures its output."""
    return subprocess.run(
        [str(BINARY), *arguments],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )


def parse_output(text: str) -> dict[str, list[str]]:
    """Maps every KEY of the KEY=VALUE lines to its values, in order."""
    values: dict[str, list[str]] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values.setdefault(key, []).append(value)
    return values


@unittest.skipUnless(BINARY.is_file(), f"{BINARY} is not built")
class OptionTests(unittest.TestCase):
    """Check the help and the option errors, which need no GPU."""

    def test_help(self) -> None:
        """--help prints every option and exits with 0."""
        result = run_benchmark("--help")
        self.assertEqual(result.returncode, 0)
        for option in ("--precision", "--mode", "--batch", "--seed"):
            self.assertIn(option, result.stdout)

    def test_invalid_options(self) -> None:
        """Invalid options end with STATUS=ERROR and exit code 1."""
        cases = {
            ("--bogus", "1"): "unknown option: --bogus",
            ("--precision", "fp8"): "--precision must be fp32, fp16 or int8",
            ("--mode", "x"): "--mode must be measure or verify",
            ("--batch", "100"): "positive multiples of 64",
            ("--hidden-layers", "-1"): "--hidden-layers must be non-negative",
            ("--duration", "0"): "measurement must be positive",
            ("--seed", "-1"): "--seed must be non-negative",
            ("--batch", "abc"): "invalid integer for --batch: abc",
        }
        for arguments, message in cases.items():
            with self.subTest(arguments=arguments):
                result = run_benchmark(*arguments)
                self.assertEqual(result.returncode, 1)
                errors = parse_output(result.stderr)
                self.assertEqual(errors["STATUS"], ["ERROR"])
                self.assertIn(message, errors["ERROR"][0])


@unittest.skipUnless(BINARY.is_file(), f"{BINARY} is not built")
class DeviceTests(unittest.TestCase):
    """Run the benchmark on the GPU."""

    def run_on_device(self, *arguments: str) -> dict[str, list[str]]:
        """Runs the benchmark, skips without a GPU and checks STATUS=OK."""
        result = run_benchmark(*arguments)
        if result.returncode != 0 and any(
            marker in result.stderr for marker in NO_DEVICE_MARKERS
        ):
            self.skipTest("no CUDA device is available")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = parse_output(result.stdout)
        self.assertEqual(output["STATUS"], ["OK"])
        return output

    def test_verify(self) -> None:
        """Every precision matches the host reference within its tolerance."""
        for seed in ("2024", "7"):
            with self.subTest(seed=seed):
                output = self.run_on_device("--mode", "verify", "--seed", seed)
                for precision in ("FP32", "FP16", "INT8"):
                    error = float(output[f"VERIFY_{precision}_REL_L2_ERROR"][0])
                    tolerance = float(
                        output[f"VERIFY_{precision}_TOLERANCE"][0]
                    )
                    self.assertEqual(output[f"VERIFY_{precision}"], ["OK"])
                    self.assertLessEqual(error, tolerance)
                self.assertEqual(output["VERIFY"], ["OK"])

    def test_verify_is_deterministic(self) -> None:
        """The same seed gives the same layer errors."""
        first = self.run_on_device("--mode", "verify")
        second = self.run_on_device("--mode", "verify")
        for precision in ("FP32", "FP16", "INT8"):
            key = f"VERIFY_{precision}_REL_L2_ERROR"
            self.assertEqual(first[key], second[key])

    def test_measure_phases_and_accuracy(self) -> None:
        """Each precision runs its phases in order and reports its accuracy."""
        for precision in ("fp32", "fp16", "int8"):
            with self.subTest(precision=precision):
                output = self.run_on_device(
                    "--precision",
                    precision,
                    "--warmup-seconds",
                    "0.2",
                    "--duration",
                    "0.5",
                    "--cooldown-seconds",
                    "0",
                    *SMALL_NETWORK,
                )
                self.assertEqual(output["PRECISION"], [precision])
                self.assertEqual(
                    output["PHASE"],
                    ["warmup", "measurement", "cooldown", "complete"],
                )
                self.assertEqual(output["LAYERS"], ["2"])
                self.assertGreater(int(output["MEASUREMENT_FORWARDS"][0]), 0)
                error = float(output["REL_L2_ERROR_VS_FP32"][0])
                agreement = float(output["TOP1_AGREEMENT_VS_FP32_PCT"][0])
                if precision == "fp32":
                    self.assertEqual(error, 0.0)
                    self.assertEqual(agreement, 100.0)
                else:
                    self.assertLess(error, 0.1)
                    self.assertGreaterEqual(agreement, 50.0)


if __name__ == "__main__":
    unittest.main()
