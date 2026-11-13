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

The CUDA Toolkit can be downloaded from https://developer.nvidia.com/cuda-downloads

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

### Output

Each benchmark prints one `KEY=VALUE` per line. It ends with `STATUS=OK` or
`STATUS=ERROR`.

## Platforms

efimon-benchmarks has been tested in the following platforms:

* AlmaLinux 9 with NVIDIA Tesla V100 (CUDA 12.8)
* Ubuntu 24.04 with NVIDIA Jetson Thor (CUDA 13.2)

## Additional Information

This project is given to you under the Apache License 2.0.

## Official Repository

Github: https://github.com/ECASLab/efimon-benchmarks
