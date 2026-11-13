# efimon-benchmarks
Benchmarks to stress CPU and GPU for energy accounting training

## Installation

This project uses CMake as the building system. Please, check the following
instructions according to your needs:

### CMake and compiler

```bash
# Ubuntu
sudo apt install build-essential cmake
# Fedora
sudo dnf install gcc-c++ cmake
```

### Dependencies

It requires the following dependencies:

* CMake >= 3.21
* C++17 compiler
* CUDA Toolkit
* Python >= 3.9 (only for the power sweep scripts)

The CUDA Toolkit can be downloaded from https://developer.nvidia.com/cuda-downloads

Optional: matplotlib, to plot the power sweep results.

## Compiling efimon-benchmarks

Compiling efimon-benchmarks follows the same process as any CMake project:

```bash
cmake -S . -B build $OPTIONS
cmake --build build -j
```

The binaries will be available under `build/`.

For the `$OPTIONS`, you can select the following:

| Option | Default Value | Values | Description |
|---|---|---|---|
| CMAKE_CUDA_ARCHITECTURES | nvcc default | [70, 80, 87, 90, 110, ...] | GPU architecture (70 for V100, 110 for Thor) |
| CMAKE_BUILD_TYPE | empty | [Release, Debug] | Build type |

## Usage

All the benchmarks have these options:

| Option | Default Value | Description |
|---|---|---|
| --device | 0 | CUDA device |
| --duration | 10 | Seconds of measurement |
| --warmup | 5 | Warm-up iterations |
| --block-size | 256 | Threads per CUDA block |
| --help | | Show the help |

### Idle

Measures the GPU without load, with or without a CUDA context.

```bash
./build/bench_idle --duration 10 --mode system
./build/bench_idle --duration 10 --mode cuda-context
```

### Compute

Stresses the FP32 units of the GPU.

```bash
./build/bench_compute --duration 10 --iterations 10000
```

### Memory

Stresses the GPU memory with a full copy or with a stride.

```bash
./build/bench_memory --duration 10 --size 536870912
./build/bench_stride --duration 10 --size 536870912 --stride 8
```

### Transfer

Stresses the copies between the host and the GPU.

```bash
./build/bench_transfer --duration 10 --direction h2d --memory pinned
./build/bench_transfer --duration 10 --direction d2h --memory pageable
```

### Power cycle

Runs a warm-up, a measurement and a cooldown phase in one process. In each
period, the GPU works for the duty cycle and sleeps for the rest, so the power
can be set between idle and full load. A small marker kernel starts each
phase, so the phases can be found in the CUPTI kernel trace.

```bash
./build/bench_power_cycle --workload compute --duty-cycle 50 \
  --warmup-seconds 15 --duration 45 --cooldown-seconds 15
```

It has these options instead of `--warmup`:

| Option | Default Value | Description |
|---|---|---|
| --workload | compute | idle, compute, memory or mixed |
| --duty-cycle | 100 | Active part of each period, from 0 to 100 |
| --period-ms | 100 | Length of each period |
| --warmup-seconds | 15 | Warm-up phase |
| --duration | 45 | Measurement phase |
| --cooldown-seconds | 15 | Cooldown phase, without GPU work |
| --iterations | 10000 | FMA iterations of the compute kernel |
| --size | 268435456 | Bytes per buffer of the memory kernel |
| --mixed-compute-pct | 50 | Compute share of the mixed workload |

The idle workload requires `--duty-cycle 0`.

### Output

Each benchmark prints one `KEY=VALUE` per line. It ends with `STATUS=OK` or
`STATUS=ERROR`.

## Power Sweeps

A power sweep runs `bench_power_cycle` at several duty cycles through the
`nvidia-process-metrics-launcher` of the profiler, and then summarizes the
power of each phase. The conditions are in `config/power_sweep.default.json`.

Check the plan without running the GPU:

```bash
python3 scripts/run_power_sweep.py --config config/power_sweep.default.json \
  --mode pilot --output-dir results/pilot --plan-only
```

Run the pilot:

```bash
python3 scripts/run_power_sweep.py --config config/power_sweep.default.json \
  --mode pilot --launcher "$(command -v nvidia-process-metrics-launcher)" \
  --benchmark build/bench_power_cycle --output-dir results/pilot
```

Add `--socket PATH` and `--library PATH` if the profiler is not installed in
the default paths. Add `--resume` to continue an interrupted sweep.

Analyze the results:

```bash
python3 scripts/analyze_power_sweep.py --study-dir results/pilot \
  --target-step-w 20 --plot --strict
```

Run the production sweep with the power levels chosen by the pilot:

```bash
python3 scripts/run_power_sweep.py --config config/power_sweep.default.json \
  --mode production \
  --levels-file results/pilot/summary/recommended_power_levels.csv \
  --launcher "$(command -v nvidia-process-metrics-launcher)" \
  --benchmark build/bench_power_cycle --output-dir results/production
```

Run the tests of the scripts:

```bash
python3 -m unittest discover -s tests
```

## Platforms

efimon-benchmarks has been tested in the following platforms:

* AlmaLinux 9 with NVIDIA Tesla V100 (CUDA 12.8)
* Ubuntu 24.04 with NVIDIA Jetson Thor (CUDA 13.2)

## Additional Information

This project is given to you under the Apache License 2.0.

## Official Repository

Github: https://github.com/ECASLab/efimon-benchmarks
