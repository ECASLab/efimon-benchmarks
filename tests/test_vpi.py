"""Tests for the NVIDIA VPI benchmark binary.

bench_vpi is only built when CMake finds VPI 4, so every test is skipped
without it. The option tests run without a GPU, because the options are
checked before CUDA starts. The GPU tests are skipped when no CUDA device is
available. Set BENCHMARKS_BUILD_DIR to use a build directory other than
build/.
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
BINARY = BUILD_DIRECTORY / "bench_vpi"
TIMEOUT_SECONDS = 300
NO_DEVICE_MARKERS = ("cudaGetDeviceCount", "requested CUDA device 0")
ALGORITHMS = ("gaussian", "rescale", "harris", "stereo", "pipeline")
VERIFY_CHECKS = (
    "GAUSSIAN_MAX_ABS_DIFF_VS_CPU",
    "GAUSSIAN_ABOVE_1_FRACTION_VS_CPU",
    "RESCALE_MAX_ABS_DIFF_VS_CPU",
    "RESCALE_ABOVE_1_FRACTION_VS_CPU",
    "BGR_TO_GREY_MAX_ABS_DIFF_VS_CPU",
    "BGR_TO_GREY_ABOVE_1_FRACTION_VS_CPU",
    "HARRIS_PRECISION",
    "HARRIS_RECALL",
    "STEREO_WITHIN_1PX_FRACTION",
)


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
        for option in ("--algorithm", "--backend", "--mode", "--width"):
            self.assertIn(option, result.stdout)

    def test_invalid_options(self) -> None:
        """Invalid options end with STATUS=ERROR and exit code 1."""
        cases = {
            ("--bogus", "1"): "unknown option: --bogus",
            ("--algorithm", "sobel"): "--algorithm must be gaussian",
            ("--backend", "gpu"): "--backend must be cuda, cpu, pva",
            ("--mode", "x"): "--mode must be measure or verify",
            ("--width", "101"): "--width and --height must be even",
            ("--algorithm", "stereo", "--width", "90"): "stereo frames",
            ("--duration", "0"): "measurement must be positive",
            ("--seed", "-1"): "--seed must be non-negative",
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
        """Every algorithm matches the CPU backend or the scene's truth."""
        output = self.run_on_device("--mode", "verify")
        for check in VERIFY_CHECKS:
            with self.subTest(check=check):
                self.assertEqual(output[f"VERIFY_{check}_STATUS"], ["OK"])
        self.assertEqual(output["VERIFY"], ["OK"])

    def test_measure_phases(self) -> None:
        """Each algorithm runs its phases in order and processes frames."""
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                output = self.run_on_device(
                    "--algorithm",
                    algorithm,
                    "--warmup-seconds",
                    "0.2",
                    "--duration",
                    "0.5",
                    "--cooldown-seconds",
                    "0",
                )
                self.assertEqual(output["ALGORITHM"], [algorithm])
                self.assertEqual(
                    output["PHASE"],
                    ["warmup", "measurement", "cooldown", "complete"],
                )
                self.assertGreater(int(output["MEASUREMENT_FRAMES"][0]), 0)

    def test_measure_results(self) -> None:
        """Harris finds corners and stereo recovers the true disparity."""
        harris = self.run_on_device(
            "--algorithm",
            "harris",
            "--duration",
            "0.2",
            "--warmup-seconds",
            "0",
            "--cooldown-seconds",
            "0",
        )
        self.assertGreater(int(harris["KEYPOINTS_LAST_FRAME"][0]), 0)
        stereo = self.run_on_device(
            "--algorithm",
            "stereo",
            "--duration",
            "0.2",
            "--warmup-seconds",
            "0",
            "--cooldown-seconds",
            "0",
        )
        self.assertGreaterEqual(
            float(stereo["DISPARITY_WITHIN_1PX_PCT"][0]), 90.0
        )


if __name__ == "__main__":
    unittest.main()
