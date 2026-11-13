/**
 * @file bench_molecular_dynamics.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Lennard-Jones molecular dynamics benchmark
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

// Simulates a Lennard-Jones liquid in reduced units with the settings of the
// LAMMPS "in.lj" benchmark: an FCC lattice at density 0.8442 melted from
// T = 1.44, cutoff 2.5 sigma, neighbor skin 0.3 sigma, neighbor list rebuilt
// every 20 steps without a displacement check, NVE velocity Verlet with
// dt = 0.005 tau. Each step runs four kernels (first half-kick and drift,
// optional cell binning and neighbor-list build, pair forces, second
// half-kick), so a capture shows a realistic mix of compute-bound force
// kernels and memory-bound integration kernels.
//
// Like bench_power_cycle, the run is split into warm-up, measurement and
// cooldown phases delimited by empty marker kernels. --mode verify instead
// checks the GPU forces against an exact O(N^2) host computation.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "benchmarks/benchmark_config.hpp"
#include "benchmarks/benchmark_utils.hpp"
#include "benchmarks/cuda_utils.hpp"
#include "benchmarks/status.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kDefaultWarmupSeconds = 10.0;
constexpr double kDefaultMeasurementSeconds = 40.0;
constexpr double kDefaultCooldownSeconds = 10.0;
// 4 * 32^3 = 131072 atoms, the size of the LAMMPS in.lj benchmark x4.
constexpr int kDefaultLatticeCells = 32;
constexpr int kMinLatticeCells = 2;
constexpr int kMaxLatticeCells = 128;
constexpr double kDefaultDensity = 0.8442;
constexpr double kDefaultTemperature = 1.44;
constexpr double kDefaultCutoff = 2.5;
constexpr double kDefaultSkin = 0.3;
constexpr double kDefaultTimestep = 0.005;
constexpr int kDefaultNeighborEvery = 20;
constexpr int kDefaultVerifySteps = 200;
constexpr std::int64_t kDefaultSeed = 12345;

// Fixed capacities; overflow is detected and reported, never silent. At the
// default density a cell holds about 19 atoms and an atom about 78 neighbors.
constexpr int kCellCapacity = 96;
constexpr int kMaxNeighbors = 192;
constexpr int kStepsBetweenChecks = 20;
constexpr int kReductionBlock = 256;
constexpr int kVerifyMaxAtoms = 32000;
constexpr double kVerifyToleranceFp32 = 1e-4;
constexpr double kVerifyToleranceFp64 = 1e-9;
// Width, in units of the Real epsilon times cutoff^2, of the band around the
// cutoff where the GPU and the host may disagree on whether a pair counts.
constexpr double kCutoffBandUlps = 100.0;
// Ambiguous pairs per atom tried both ways; about one is expected at most.
constexpr std::size_t kMaxAmbiguousPairs = 8;

/** Overflow flag set when a cell holds more than kCellCapacity atoms */
constexpr int kCellOverflowFlag = 1;
/** Overflow flag set when an atom has more than kMaxNeighbors neighbors */
constexpr int kNeighborOverflowFlag = 2;

/** Floating-point type of positions, velocities and forces */
enum class Precision { kFp32, kFp64 };
/** Timed measurement or force check against the host */
enum class Mode { kMeasure, kVerify };

/**
 * @brief Options of one simulation
 */
struct Config {
  /** CUDA device index */
  int device = benchmarks::kDefaultDevice;
  /** Threads per CUDA block */
  int block_size = benchmarks::kDefaultBlockSize;
  /** Duration of the warm-up phase */
  double warmup_seconds = kDefaultWarmupSeconds;
  /** Duration of the measurement phase */
  double measurement_seconds = kDefaultMeasurementSeconds;
  /** Duration of the context-idle cooldown phase */
  double cooldown_seconds = kDefaultCooldownSeconds;
  /** FCC unit cells per side of the box; atoms = 4 N^3 */
  int lattice_cells = kDefaultLatticeCells;
  /** Reduced number density */
  double density = kDefaultDensity;
  /** Initial reduced temperature */
  double temperature = kDefaultTemperature;
  /** Pair cutoff in sigma */
  double cutoff = kDefaultCutoff;
  /** Neighbor-list skin in sigma */
  double skin = kDefaultSkin;
  /** Integration step in tau */
  double timestep = kDefaultTimestep;
  /** Steps between neighbor-list rebuilds */
  int neighbor_every = kDefaultNeighborEvery;
  /** Steps before the force check, rounded up to a rebuild step */
  int verify_steps = kDefaultVerifySteps;
  /** Seed of the initial velocities */
  std::int64_t seed = kDefaultSeed;
  /** Type of the simulation state */
  Precision precision = Precision::kFp32;
  /** What the run does */
  Mode mode = Mode::kMeasure;
};

/**
 * @brief Counters of one timed phase
 */
struct PhaseStats {
  /** Wall time of the phase */
  double elapsed_seconds = 0.0;
  /** Integration steps in the phase */
  std::int64_t steps = 0;
};

/**
 * @brief Energies per atom and temperature of the system
 */
struct Energies {
  /** Lennard-Jones potential energy per atom */
  double potential_per_atom = 0.0;
  /** Kinetic energy per atom */
  double kinetic_per_atom = 0.0;
  /** Potential plus kinetic energy per atom */
  double total_per_atom = 0.0;
  /** Instantaneous temperature, with 3N - 3 degrees of freedom */
  double temperature = 0.0;
};

/**
 * @brief Plain view of the device state, passed to kernels by value
 */
template <typename Real>
struct SystemView {
  /** Number of atoms */
  int atoms = 0;
  /** Neighbor cells per side of the box */
  int cells_per_side = 0;
  /** Side of the cubic box */
  Real box = 0;
  /** Inverse of the box side */
  Real inverse_box = 0;
  /** Cells per unit of length */
  Real cells_per_length = 0;
  /** Square of the pair cutoff */
  Real cutoff_sq = 0;
  /** Square of the cutoff plus the skin */
  Real list_cutoff_sq = 0;
  /** Integration step */
  Real timestep = 0;
  /** Positions, one array per axis */
  Real* x = nullptr;
  Real* y = nullptr;
  Real* z = nullptr;
  /** Velocities, one array per axis */
  Real* vx = nullptr;
  Real* vy = nullptr;
  Real* vz = nullptr;
  /** Forces, one array per axis */
  Real* fx = nullptr;
  Real* fy = nullptr;
  Real* fz = nullptr;
  /** Atoms binned in each cell */
  int* cell_counts = nullptr;
  /** Atom indices per cell, kCellCapacity slots each */
  int* cell_atoms = nullptr;
  /** Neighbors stored per atom */
  int* neighbor_counts = nullptr;
  /** neighbors[k * atoms + i], coalesced over i */
  int* neighbors = nullptr;
  /** Bitwise OR of the overflow flags */
  int* overflow_flags = nullptr;
  /** Largest neighbor count seen, before clamping */
  int* max_neighbors = nullptr;
};

// The marker kernels keep their snake_case names on purpose: the analysis
// scripts find the phase boundaries by these names in the CUPTI kernel trace,
// and other workloads emit the same names. Renaming them breaks that format.

/** Marks the start of the warm-up phase in the kernel trace */
__global__ void benchmark_warmup_start_marker() {}
/** Marks the start of the measurement phase in the kernel trace */
__global__ void benchmark_measurement_start_marker() {}
/** Marks the start of the cooldown phase in the kernel trace */
__global__ void benchmark_cooldown_start_marker() {}
/** Marks the end of the benchmark in the kernel trace */
__global__ void benchmark_complete_marker() {}

/**
 * @brief Wraps a coordinate into the periodic box
 */
template <typename Real>
__device__ Real WrapPosition(Real position, const SystemView<Real>& system) {
  return position - system.box * floor(position * system.inverse_box);
}

/**
 * @brief Returns the minimum-image distance along one axis
 */
template <typename Real>
__device__ Real MinimumImage(Real delta, const SystemView<Real>& system) {
  return delta - system.box * rint(delta * system.inverse_box);
}

/**
 * @brief Returns the cell index of a coordinate along one axis
 */
template <typename Real>
__device__ int CellCoordinate(Real position, const SystemView<Real>& system) {
  const int cell = static_cast<int>(position * system.cells_per_length);
  return min(max(cell, 0), system.cells_per_side - 1);
}

/**
 * @brief First half-kick and drift of velocity Verlet, with periodic wrap
 *
 * @param system Device state, one thread per atom
 */
template <typename Real>
__global__ void MdIntegrateFirstHalfKernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const Real half_step = static_cast<Real>(0.5) * system.timestep;
  const Real vx = system.vx[i] + half_step * system.fx[i];
  const Real vy = system.vy[i] + half_step * system.fy[i];
  const Real vz = system.vz[i] + half_step * system.fz[i];
  system.vx[i] = vx;
  system.vy[i] = vy;
  system.vz[i] = vz;
  system.x[i] = WrapPosition(system.x[i] + system.timestep * vx, system);
  system.y[i] = WrapPosition(system.y[i] + system.timestep * vy, system);
  system.z[i] = WrapPosition(system.z[i] + system.timestep * vz, system);
}

/**
 * @brief Second half-kick of velocity Verlet
 *
 * @param system Device state, one thread per atom
 */
template <typename Real>
__global__ void MdIntegrateSecondHalfKernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const Real half_step = static_cast<Real>(0.5) * system.timestep;
  system.vx[i] += half_step * system.fx[i];
  system.vy[i] += half_step * system.fy[i];
  system.vz[i] += half_step * system.fz[i];
}

/**
 * @brief Bins every atom into its cell; cell_counts must start at zero
 *
 * @param system Device state, one thread per atom
 */
template <typename Real>
__global__ void MdBinAtomsKernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const int n = system.cells_per_side;
  const int cell = CellCoordinate(system.x[i], system) +
                   n * (CellCoordinate(system.y[i], system) +
                        n * CellCoordinate(system.z[i], system));
  const int slot = atomicAdd(&system.cell_counts[cell], 1);
  if (slot < kCellCapacity) {
    system.cell_atoms[cell * kCellCapacity + slot] = i;
  } else {
    atomicOr(system.overflow_flags, kCellOverflowFlag);
  }
}

/**
 * @brief Builds the full neighbor list from the 27 surrounding cells
 *
 * Each pair is stored twice, so the forces need no atomics and a step is
 * deterministic for a given list order.
 *
 * @param system Device state, one thread per atom
 */
template <typename Real>
__global__ void MdBuildNeighborListKernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const int n = system.cells_per_side;
  const Real xi = system.x[i];
  const Real yi = system.y[i];
  const Real zi = system.z[i];
  const int cx = CellCoordinate(xi, system);
  const int cy = CellCoordinate(yi, system);
  const int cz = CellCoordinate(zi, system);
  int count = 0;
  for (int oz = -1; oz <= 1; ++oz) {
    const int z = (cz + oz + n) % n;
    for (int oy = -1; oy <= 1; ++oy) {
      const int y = (cy + oy + n) % n;
      for (int ox = -1; ox <= 1; ++ox) {
        const int cell = (cx + ox + n) % n + n * (y + n * z);
        const int members = min(system.cell_counts[cell], kCellCapacity);
        for (int slot = 0; slot < members; ++slot) {
          const int j = system.cell_atoms[cell * kCellCapacity + slot];
          if (j == i) continue;
          const Real dx = MinimumImage(system.x[j] - xi, system);
          const Real dy = MinimumImage(system.y[j] - yi, system);
          const Real dz = MinimumImage(system.z[j] - zi, system);
          if (dx * dx + dy * dy + dz * dz < system.list_cutoff_sq) {
            if (count < kMaxNeighbors) {
              system.neighbors[static_cast<std::size_t>(count) * system.atoms +
                               i] = j;
            }
            ++count;
          }
        }
      }
    }
  }
  system.neighbor_counts[i] = min(count, kMaxNeighbors);
  if (count > kMaxNeighbors) {
    atomicOr(system.overflow_flags, kNeighborOverflowFlag);
  }
  atomicMax(system.max_neighbors, count);
}

/**
 * @brief Computes the Lennard-Jones pair forces from the neighbor list
 *
 * F_i = sum_j 48 r^-8 (r^-6 - 1/2) (r_i - r_j), epsilon = sigma = mass = 1.
 *
 * @param system Device state, one thread per atom
 */
template <typename Real>
__global__ void MdLjForceKernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const Real xi = system.x[i];
  const Real yi = system.y[i];
  const Real zi = system.z[i];
  Real fx = 0;
  Real fy = 0;
  Real fz = 0;
  const int count = system.neighbor_counts[i];
  for (int k = 0; k < count; ++k) {
    const int j =
        system.neighbors[static_cast<std::size_t>(k) * system.atoms + i];
    const Real dx = MinimumImage(xi - system.x[j], system);
    const Real dy = MinimumImage(yi - system.y[j], system);
    const Real dz = MinimumImage(zi - system.z[j], system);
    const Real r_sq = dx * dx + dy * dy + dz * dz;
    if (r_sq < system.cutoff_sq) {
      const Real inverse_r2 = static_cast<Real>(1) / r_sq;
      const Real inverse_r6 = inverse_r2 * inverse_r2 * inverse_r2;
      const Real force_over_r = static_cast<Real>(48) * inverse_r6 *
                                (inverse_r6 - static_cast<Real>(0.5)) *
                                inverse_r2;
      fx += force_over_r * dx;
      fy += force_over_r * dy;
      fz += force_over_r * dz;
    }
  }
  system.fx[i] = fx;
  system.fy[i] = fy;
  system.fz[i] = fz;
}

/**
 * @brief Sums the potential and kinetic energy of each block
 *
 * The sums are accumulated in double for both precisions, so the drift
 * reflects the dynamics and not the reduction.
 *
 * @param system Device state, launched with kReductionBlock threads per block
 * @param partials Output, potential and kinetic sum per block
 */
template <typename Real>
__global__ void MdEnergyKernel(SystemView<Real> system, double* partials) {
  __shared__ double potential[kReductionBlock];
  __shared__ double kinetic[kReductionBlock];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double atom_potential = 0.0;
  double atom_kinetic = 0.0;
  if (i < system.atoms) {
    const int count = system.neighbor_counts[i];
    for (int k = 0; k < count; ++k) {
      const int j =
          system.neighbors[static_cast<std::size_t>(k) * system.atoms + i];
      const Real dx = MinimumImage(system.x[i] - system.x[j], system);
      const Real dy = MinimumImage(system.y[i] - system.y[j], system);
      const Real dz = MinimumImage(system.z[i] - system.z[j], system);
      const Real r_sq = dx * dx + dy * dy + dz * dz;
      if (r_sq < system.cutoff_sq) {
        const double inverse_r6 =
            1.0 / (static_cast<double>(r_sq) * r_sq * r_sq);
        // Each pair appears twice in the full list.
        atom_potential += 2.0 * inverse_r6 * (inverse_r6 - 1.0);
      }
    }
    const double vx = system.vx[i];
    const double vy = system.vy[i];
    const double vz = system.vz[i];
    atom_kinetic = 0.5 * (vx * vx + vy * vy + vz * vz);
  }
  potential[threadIdx.x] = atom_potential;
  kinetic[threadIdx.x] = atom_kinetic;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      potential[threadIdx.x] += potential[threadIdx.x + stride];
      kinetic[threadIdx.x] += kinetic[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[2 * blockIdx.x] = potential[0];
    partials[2 * blockIdx.x + 1] = kinetic[0];
  }
}

/**
 * @brief Allocates count elements of device memory
 *
 * @param count Number of elements
 * @param buffer Output that owns the allocation. Must not be null
 * @return Error if the allocation fails
 */
template <typename T>
benchmarks::Status AllocateDevice(std::size_t count,
                                  benchmarks::DeviceBuffer<T>* buffer) {
  T* pointer = nullptr;
  BENCHMARKS_CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  buffer->reset(pointer);
  return benchmarks::Status();
}

/**
 * @brief Returns the number of blocks that cover items threads
 */
int GridSize(int items, int block_size) {
  return (items + block_size - 1) / block_size;
}

/**
 * @brief Owns the device buffers of one simulation and advances it
 */
template <typename Real>
class Simulation {
 public:
  Simulation() = default;
  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;

  /**
   * @brief Builds the lattice, the first neighbor list and the first forces
   *
   * @param config Simulation options
   * @return Error if the box is too small, a CUDA call fails or a capacity
   * is exceeded
   */
  benchmarks::Status Init(const Config& config) {
    block_size_ = config.block_size;
    neighbor_every_ = config.neighbor_every;
    const int n = config.lattice_cells;
    view_.atoms = 4 * n * n * n;
    const double lattice = std::cbrt(4.0 / config.density);
    const double box = n * lattice;
    const double list_cutoff = config.cutoff + config.skin;
    view_.cells_per_side = static_cast<int>(std::floor(box / list_cutoff));
    if (view_.cells_per_side < 3) {
      return benchmarks::Status::Error(
          "box is smaller than three neighbor cells; raise --lattice-cells");
    }
    view_.box = static_cast<Real>(box);
    view_.inverse_box = static_cast<Real>(1.0 / box);
    view_.cells_per_length = static_cast<Real>(view_.cells_per_side / box);
    view_.cutoff_sq = static_cast<Real>(config.cutoff * config.cutoff);
    view_.list_cutoff_sq = static_cast<Real>(list_cutoff * list_cutoff);
    view_.timestep = static_cast<Real>(config.timestep);
    box_length_ = box;
    BENCHMARKS_RETURN_IF_ERROR(Allocate());
    BENCHMARKS_RETURN_IF_ERROR(Initialize(config, lattice));
    BENCHMARKS_RETURN_IF_ERROR(Rebuild());
    BENCHMARKS_RETURN_IF_ERROR(LaunchForces());
    return Check();
  }

  /**
   * @brief Queues one velocity Verlet step
   *
   * The neighbor list is rebuilt every neighbor_every steps before the force
   * evaluation, as LAMMPS does.
   *
   * @return Error if a launch fails
   */
  benchmarks::Status Step() {
    const int grid = GridSize(view_.atoms, block_size_);
    MdIntegrateFirstHalfKernel<<<grid, block_size_>>>(view_);
    ++steps_;
    if (steps_ % neighbor_every_ == 0) {
      BENCHMARKS_RETURN_IF_ERROR(Rebuild());
    }
    BENCHMARKS_RETURN_IF_ERROR(LaunchForces());
    MdIntegrateSecondHalfKernel<<<grid, block_size_>>>(view_);
    BENCHMARKS_CUDA_CHECK(cudaGetLastError());
    return benchmarks::Status();
  }

  /**
   * @brief Waits for the queued work and checks the fixed capacities
   *
   * @return Error if a capacity was exceeded or a CUDA call fails
   */
  benchmarks::Status Check() {
    int flags = 0;
    BENCHMARKS_CUDA_CHECK(cudaMemcpy(&flags, view_.overflow_flags, sizeof(int),
                                     cudaMemcpyDeviceToHost));
    if ((flags & kCellOverflowFlag) != 0) {
      return benchmarks::Status::Error(
          "cell capacity exceeded (kCellCapacity)");
    }
    if ((flags & kNeighborOverflowFlag) != 0) {
      return benchmarks::Status::Error(
          "neighbor capacity exceeded (kMaxNeighbors)");
    }
    return benchmarks::Status();
  }

  /**
   * @brief Computes the energies per atom and the temperature
   *
   * @param energies Output. Must not be null
   * @return Error if a CUDA call fails
   */
  benchmarks::Status ComputeEnergies(Energies* energies) {
    const int blocks = GridSize(view_.atoms, kReductionBlock);
    MdEnergyKernel<<<blocks, kReductionBlock>>>(view_, partials_.get());
    BENCHMARKS_CUDA_CHECK(cudaGetLastError());
    std::vector<double> host(2 * static_cast<std::size_t>(blocks));
    BENCHMARKS_CUDA_CHECK(cudaMemcpy(host.data(), partials_.get(),
                                     host.size() * sizeof(double),
                                     cudaMemcpyDeviceToHost));
    double potential = 0.0;
    double kinetic = 0.0;
    for (int block = 0; block < blocks; ++block) {
      potential += host[2 * block];
      kinetic += host[2 * block + 1];
    }
    energies->potential_per_atom = potential / view_.atoms;
    energies->kinetic_per_atom = kinetic / view_.atoms;
    energies->total_per_atom =
        energies->potential_per_atom + energies->kinetic_per_atom;
    energies->temperature = 2.0 * kinetic / (3.0 * (view_.atoms - 1));
    return benchmarks::Status();
  }

  /**
   * @brief Reads the largest neighbor count seen so far
   *
   * @param value Output. Must not be null
   * @return Error if the copy fails
   */
  benchmarks::Status MaxNeighbors(int* value) const {
    BENCHMARKS_CUDA_CHECK(cudaMemcpy(value, view_.max_neighbors, sizeof(int),
                                     cudaMemcpyDeviceToHost));
    return benchmarks::Status();
  }

  /**
   * @brief Copies positions and forces to the host as double
   *
   * @param positions Output, x y z per atom. Must not be null
   * @param forces Output, x y z per atom. Must not be null
   * @return Error if a copy fails
   */
  benchmarks::Status Download(std::vector<double>* positions,
                              std::vector<double>* forces) const {
    positions->assign(3 * static_cast<std::size_t>(view_.atoms), 0.0);
    forces->assign(3 * static_cast<std::size_t>(view_.atoms), 0.0);
    const Real* sources[6] = {view_.x,  view_.y,  view_.z,
                              view_.fx, view_.fy, view_.fz};
    std::vector<Real> buffer(view_.atoms);
    for (int axis = 0; axis < 6; ++axis) {
      BENCHMARKS_CUDA_CHECK(cudaMemcpy(buffer.data(), sources[axis],
                                       buffer.size() * sizeof(Real),
                                       cudaMemcpyDeviceToHost));
      std::vector<double>* target = axis < 3 ? positions : forces;
      for (int i = 0; i < view_.atoms; ++i) {
        (*target)[3 * static_cast<std::size_t>(i) + axis % 3] = buffer[i];
      }
    }
    return benchmarks::Status();
  }

  /** Number of atoms */
  int atoms() const { return view_.atoms; }
  /** Neighbor cells per side of the box */
  int cells_per_side() const { return view_.cells_per_side; }
  /** Side of the cubic box */
  double box_length() const { return box_length_; }
  /** Neighbor-list rebuilds so far */
  std::int64_t rebuilds() const { return rebuilds_; }

 private:
  /**
   * @brief Allocates every device buffer and clears the flags
   */
  benchmarks::Status Allocate() {
    const std::size_t atoms = view_.atoms;
    const std::size_t cells = static_cast<std::size_t>(view_.cells_per_side) *
                              view_.cells_per_side * view_.cells_per_side;
    Real** views[kAtomArrays] = {&view_.x,  &view_.y,  &view_.z,
                                 &view_.vx, &view_.vy, &view_.vz,
                                 &view_.fx, &view_.fy, &view_.fz};
    for (int array = 0; array < kAtomArrays; ++array) {
      BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(atoms, &atom_arrays_[array]));
      *views[array] = atom_arrays_[array].get();
    }
    BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(cells, &cell_counts_));
    BENCHMARKS_RETURN_IF_ERROR(
        AllocateDevice(cells * kCellCapacity, &cell_atoms_));
    BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(atoms, &neighbor_counts_));
    BENCHMARKS_RETURN_IF_ERROR(
        AllocateDevice(atoms * kMaxNeighbors, &neighbors_));
    BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(1, &overflow_flags_));
    BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(1, &max_neighbors_));
    BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(
        2 * static_cast<std::size_t>(GridSize(view_.atoms, kReductionBlock)),
        &partials_));
    view_.cell_counts = cell_counts_.get();
    view_.cell_atoms = cell_atoms_.get();
    view_.neighbor_counts = neighbor_counts_.get();
    view_.neighbors = neighbors_.get();
    view_.overflow_flags = overflow_flags_.get();
    view_.max_neighbors = max_neighbors_.get();
    cells_ = cells;
    BENCHMARKS_CUDA_CHECK(cudaMemset(view_.overflow_flags, 0, sizeof(int)));
    BENCHMARKS_CUDA_CHECK(cudaMemset(view_.max_neighbors, 0, sizeof(int)));
    return benchmarks::Status();
  }

  /**
   * @brief Places the atoms on an FCC lattice with random velocities
   *
   * The velocities have zero total momentum and are scaled to the requested
   * temperature (3N - 3 degrees of freedom).
   */
  benchmarks::Status Initialize(const Config& config, double lattice) {
    const int n = config.lattice_cells;
    const std::size_t atoms = view_.atoms;
    std::vector<double> host[6];
    for (auto& values : host) values.reserve(atoms);
    constexpr double kBasis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          for (const auto& basis : kBasis) {
            host[0].push_back((i + basis[0]) * lattice);
            host[1].push_back((j + basis[1]) * lattice);
            host[2].push_back((k + basis[2]) * lattice);
          }
        }
      }
    }
    std::mt19937_64 generator(static_cast<std::uint64_t>(config.seed));
    std::uniform_real_distribution<double> uniform(-0.5, 0.5);
    double momentum[3] = {0.0, 0.0, 0.0};
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      for (int axis = 0; axis < 3; ++axis) {
        host[3 + axis].push_back(uniform(generator));
        momentum[axis] += host[3 + axis].back();
      }
    }
    double kinetic_twice = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      for (double& velocity : host[3 + axis]) {
        velocity -= momentum[axis] / atoms;
        kinetic_twice += velocity * velocity;
      }
    }
    const double scale =
        std::sqrt(config.temperature * 3.0 * (atoms - 1) / kinetic_twice);
    Real* targets[6] = {view_.x,  view_.y,  view_.z,
                        view_.vx, view_.vy, view_.vz};
    std::vector<Real> buffer(atoms);
    for (int array = 0; array < 6; ++array) {
      const double factor = array < 3 ? 1.0 : scale;
      for (std::size_t atom = 0; atom < atoms; ++atom) {
        buffer[atom] = static_cast<Real>(host[array][atom] * factor);
      }
      BENCHMARKS_CUDA_CHECK(cudaMemcpy(targets[array], buffer.data(),
                                       atoms * sizeof(Real),
                                       cudaMemcpyHostToDevice));
    }
    return benchmarks::Status();
  }

  /**
   * @brief Queues the cell binning and the neighbor-list build
   */
  benchmarks::Status Rebuild() {
    const int grid = GridSize(view_.atoms, block_size_);
    BENCHMARKS_CUDA_CHECK(
        cudaMemsetAsync(view_.cell_counts, 0, cells_ * sizeof(int)));
    MdBinAtomsKernel<<<grid, block_size_>>>(view_);
    MdBuildNeighborListKernel<<<grid, block_size_>>>(view_);
    BENCHMARKS_CUDA_CHECK(cudaGetLastError());
    ++rebuilds_;
    return benchmarks::Status();
  }

  /**
   * @brief Queues the force kernel
   */
  benchmarks::Status LaunchForces() {
    MdLjForceKernel<<<GridSize(view_.atoms, block_size_), block_size_>>>(view_);
    BENCHMARKS_CUDA_CHECK(cudaGetLastError());
    return benchmarks::Status();
  }

  /** Positions, velocities and forces: x y z vx vy vz fx fy fz */
  static constexpr int kAtomArrays = 9;

  /** Raw pointers into the buffers below, passed to the kernels */
  SystemView<Real> view_;
  /** Owners of the per-atom arrays of view_ */
  benchmarks::DeviceBuffer<Real> atom_arrays_[kAtomArrays];
  /** Owner of view_.cell_counts */
  benchmarks::DeviceBuffer<int> cell_counts_;
  /** Owner of view_.cell_atoms */
  benchmarks::DeviceBuffer<int> cell_atoms_;
  /** Owner of view_.neighbor_counts */
  benchmarks::DeviceBuffer<int> neighbor_counts_;
  /** Owner of view_.neighbors */
  benchmarks::DeviceBuffer<int> neighbors_;
  /** Owner of view_.overflow_flags */
  benchmarks::DeviceBuffer<int> overflow_flags_;
  /** Owner of view_.max_neighbors */
  benchmarks::DeviceBuffer<int> max_neighbors_;
  /** Per-block energy sums of MdEnergyKernel */
  benchmarks::DeviceBuffer<double> partials_;
  /** Number of neighbor cells */
  std::size_t cells_ = 0;
  /** Side of the cubic box, in double */
  double box_length_ = 0.0;
  /** Threads per block of the per-atom kernels */
  int block_size_ = 0;
  /** Steps between neighbor-list rebuilds */
  int neighbor_every_ = 0;
  /** Steps queued so far */
  std::int64_t steps_ = 0;
  /** Neighbor-list rebuilds queued so far */
  std::int64_t rebuilds_ = 0;
};

/**
 * @brief Compares the GPU forces with an exact O(N^2) host computation
 *
 * The force is discontinuous at the cutoff, and the GPU decides in Real
 * precision whether a pair is inside it. A pair whose r^2 lies within
 * kCutoffBandUlps of the cutoff is therefore ambiguous: the host tries the
 * force with and without each ambiguous pair and keeps the closest one, so
 * a rounding difference at the cutoff is not reported as a force error.
 *
 * @param positions x y z per atom
 * @param forces GPU forces, x y z per atom
 * @param box Side of the cubic box
 * @param cutoff Pair cutoff
 * @return Largest |F_gpu - F_exact| over atoms, relative to the largest
 * |F_exact|
 */
template <typename Real>
double MaxRelativeForceError(const std::vector<double>& positions,
                             const std::vector<double>& forces, double box,
                             double cutoff) {
  const std::size_t atoms = positions.size() / 3;
  const double cutoff_sq = cutoff * cutoff;
  const double band =
      kCutoffBandUlps * std::numeric_limits<Real>::epsilon() * cutoff_sq;
  double max_error = 0.0;
  double max_force = 0.0;
  for (std::size_t i = 0; i < atoms; ++i) {
    std::array<double, 3> exact = {0.0, 0.0, 0.0};
    std::vector<std::array<double, 3>> ambiguous;
    for (std::size_t j = 0; j < atoms; ++j) {
      if (j == i) continue;
      double delta[3];
      for (int axis = 0; axis < 3; ++axis) {
        const double d = positions[3 * i + axis] - positions[3 * j + axis];
        delta[axis] = d - box * std::nearbyint(d / box);
      }
      const double r_sq =
          delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
      if (r_sq >= cutoff_sq + band) continue;
      const double inverse_r2 = 1.0 / r_sq;
      const double inverse_r6 = inverse_r2 * inverse_r2 * inverse_r2;
      const double force_over_r =
          48.0 * inverse_r6 * (inverse_r6 - 0.5) * inverse_r2;
      std::array<double, 3> pair_force;
      for (int axis = 0; axis < 3; ++axis) {
        pair_force[axis] = force_over_r * delta[axis];
      }
      const bool near_cutoff = r_sq > cutoff_sq - band;
      if (near_cutoff && ambiguous.size() < kMaxAmbiguousPairs) {
        ambiguous.push_back(pair_force);
        continue;
      }
      // Pairs beyond kMaxAmbiguousPairs keep the plain r < cutoff rule.
      if (r_sq < cutoff_sq) {
        for (int axis = 0; axis < 3; ++axis) exact[axis] += pair_force[axis];
      }
    }
    double best_error_sq = std::numeric_limits<double>::infinity();
    double best_force_sq = 0.0;
    const unsigned subsets = 1U << ambiguous.size();
    for (unsigned subset = 0; subset < subsets; ++subset) {
      std::array<double, 3> candidate = exact;
      for (std::size_t index = 0; index < ambiguous.size(); ++index) {
        if ((subset >> index & 1U) == 0) continue;
        for (int axis = 0; axis < 3; ++axis) {
          candidate[axis] += ambiguous[index][axis];
        }
      }
      double error_sq = 0.0;
      double force_sq = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double difference = forces[3 * i + axis] - candidate[axis];
        error_sq += difference * difference;
        force_sq += candidate[axis] * candidate[axis];
      }
      if (error_sq < best_error_sq) {
        best_error_sq = error_sq;
        best_force_sq = force_sq;
      }
    }
    max_error = std::max(max_error, std::sqrt(best_error_sq));
    max_force = std::max(max_force, std::sqrt(best_force_sq));
  }
  return max_force > 0.0 ? max_error / max_force : max_error;
}

/**
 * @brief Prints the usage and the options of the benchmark
 *
 * @param program Name of the executable, from argv[0]
 */
void PrintHelp(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Lennard-Jones molecular dynamics (LAMMPS in.lj settings).\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --precision <name>        fp32 or fp64 (default: fp32)\n"
      << "  --mode <name>             measure or verify (default: measure)\n"
      << "  --lattice-cells <N>       FCC cells per side; atoms = 4 N^3\n"
      << "                            (default: 32, 131072 atoms)\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 10)\n"
      << "  --duration <seconds>      Measurement phase (default: 40)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 10)\n"
      << "  --density <value>         Reduced density (default: 0.8442)\n"
      << "  --temperature <value>     Initial temperature (default: 1.44)\n"
      << "  --cutoff <value>          Pair cutoff in sigma (default: 2.5)\n"
      << "  --skin <value>            Neighbor skin in sigma (default: 0.3)\n"
      << "  --timestep <value>        Step in tau (default: 0.005)\n"
      << "  --neighbor-every <N>      Rebuild period in steps (default: 20)\n"
      << "  --verify-steps <N>        Steps before the force check\n"
      << "                            (default: 200)\n"
      << "  --seed <N>                Velocity seed (default: 12345)\n"
      << "  --block-size <threads>    CUDA block size (default: 256)\n"
      << "  --help                    Show this help\n";
}

/**
 * @brief Reads and validates the options of the benchmark
 *
 * @param arguments Parsed command line
 * @param config Output. Must not be null
 * @return Error if an option is unknown, invalid or out of range
 */
benchmarks::Status ParseConfig(const benchmarks::ArgumentParser& arguments,
                               Config* config) {
  BENCHMARKS_RETURN_IF_ERROR(arguments.ValidateOptions(
      {"--device", "--precision", "--mode", "--lattice-cells",
       "--warmup-seconds", "--duration", "--cooldown-seconds", "--density",
       "--temperature", "--cutoff", "--skin", "--timestep", "--neighbor-every",
       "--verify-steps", "--seed", "--block-size"}));

  *config = Config();
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--device", config->device, &config->device));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--block-size", config->block_size, &config->block_size));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--warmup-seconds", config->warmup_seconds, &config->warmup_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--duration", config->measurement_seconds, &config->measurement_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue("--cooldown-seconds",
                                                   config->cooldown_seconds,
                                                   &config->cooldown_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--lattice-cells", config->lattice_cells, &config->lattice_cells));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.DoubleValue("--density", config->density, &config->density));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--temperature", config->temperature, &config->temperature));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.DoubleValue("--cutoff", config->cutoff, &config->cutoff));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.DoubleValue("--skin", config->skin, &config->skin));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.DoubleValue("--timestep", config->timestep, &config->timestep));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--neighbor-every", config->neighbor_every, &config->neighbor_every));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--verify-steps", config->verify_steps, &config->verify_steps));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.Int64Value("--seed", config->seed, &config->seed));

  const std::string precision = arguments.StringValue("--precision", "fp32");
  if (precision == "fp32") {
    config->precision = Precision::kFp32;
  } else if (precision == "fp64") {
    config->precision = Precision::kFp64;
  } else {
    return benchmarks::Status::Error("--precision must be fp32 or fp64");
  }
  const std::string mode = arguments.StringValue("--mode", "measure");
  if (mode == "measure") {
    config->mode = Mode::kMeasure;
  } else if (mode == "verify") {
    config->mode = Mode::kVerify;
  } else {
    return benchmarks::Status::Error("--mode must be measure or verify");
  }

  if (config->device < 0) {
    return benchmarks::Status::Error("--device must be non-negative");
  }
  if (config->block_size <= 0 || config->block_size > 1024) {
    return benchmarks::Status::Error(
        "--block-size must be in the range [1, 1024]");
  }
  if (!(config->warmup_seconds >= 0.0) ||
      !(config->measurement_seconds > 0.0) ||
      !(config->cooldown_seconds >= 0.0)) {
    return benchmarks::Status::Error(
        "phase durations must be finite; measurement must be positive");
  }
  if (config->lattice_cells < kMinLatticeCells ||
      config->lattice_cells > kMaxLatticeCells) {
    return benchmarks::Status::Error(
        "--lattice-cells must be in the range [2, 128]");
  }
  if (!(config->density > 0.0) || !(config->temperature > 0.0) ||
      !(config->cutoff > 0.0) || !(config->skin >= 0.0) ||
      !(config->timestep > 0.0)) {
    return benchmarks::Status::Error(
        "density, temperature, cutoff and timestep must be positive; skin "
        "must be non-negative");
  }
  if (config->neighbor_every < 1 || config->verify_steps < 0) {
    return benchmarks::Status::Error(
        "--neighbor-every must be positive and --verify-steps non-negative");
  }
  if (config->seed < 0) {
    return benchmarks::Status::Error("--seed must be non-negative");
  }
  // The force check needs a freshly built list, so it runs on a rebuild step.
  config->verify_steps = (config->verify_steps + config->neighbor_every - 1) /
                         config->neighbor_every * config->neighbor_every;
  return benchmarks::Status();
}

/**
 * @brief Returns the --precision name of a precision
 */
const char* PrecisionName(Precision precision) {
  return precision == Precision::kFp32 ? "fp32" : "fp64";
}

/**
 * @brief Prints the configuration and the device as KEY=VALUE lines
 */
void PrintConfig(const Config& config, const cudaDeviceProp& properties) {
  std::cout << std::setprecision(12) << "BENCHMARK=molecular_dynamics\n"
            << "POTENTIAL=lennard_jones\n"
            << "PRECISION=" << PrecisionName(config.precision) << '\n'
            << "MODE=" << (config.mode == Mode::kMeasure ? "measure" : "verify")
            << '\n'
            << "DEVICE=" << config.device << '\n'
            << "DEVICE_NAME=" << properties.name << '\n'
            << "COMPUTE_CAPABILITY=" << properties.major << '.'
            << properties.minor << '\n'
            << "LATTICE_CELLS=" << config.lattice_cells << '\n'
            << "DENSITY=" << config.density << '\n'
            << "TEMPERATURE=" << config.temperature << '\n'
            << "CUTOFF=" << config.cutoff << '\n'
            << "SKIN=" << config.skin << '\n'
            << "TIMESTEP=" << config.timestep << '\n'
            << "NEIGHBOR_EVERY=" << config.neighbor_every << '\n'
            << "BLOCK_SIZE=" << config.block_size << '\n'
            << "WARMUP_SECONDS=" << config.warmup_seconds << '\n'
            << "MEASUREMENT_SECONDS=" << config.measurement_seconds << '\n'
            << "COOLDOWN_SECONDS=" << config.cooldown_seconds << '\n';
}

/** Phase boundary marked in the kernel trace */
enum class PhaseMarker { kWarmup, kMeasurement, kCooldown, kComplete };

/**
 * @brief Launches the marker kernel of a phase and prints PHASE=<name>
 *
 * @param phase Name printed after PHASE=
 * @param marker Marker kernel to launch
 * @return Error if the launch or the synchronization fails
 */
benchmarks::Status EmitPhase(const char* phase, PhaseMarker marker) {
  switch (marker) {
    case PhaseMarker::kWarmup:
      benchmark_warmup_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kMeasurement:
      benchmark_measurement_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kCooldown:
      benchmark_cooldown_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kComplete:
      benchmark_complete_marker<<<1, 1>>>();
      break;
  }
  BENCHMARKS_CUDA_CHECK(cudaGetLastError());
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "PHASE=" << phase << '\n' << std::flush;
  return benchmarks::Status();
}

/**
 * @brief Advances the simulation until the phase time runs out
 *
 * @param simulation Simulation to advance. Must not be null
 * @param seconds Duration of the phase
 * @param stats Output. Must not be null
 * @return Error if a step fails or a capacity is exceeded
 */
template <typename Real>
benchmarks::Status RunPhase(Simulation<Real>* simulation, double seconds,
                            PhaseStats* stats) {
  *stats = PhaseStats();
  const auto start = Clock::now();
  const auto deadline = start + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(seconds));
  while (Clock::now() < deadline) {
    for (int step = 0; step < kStepsBetweenChecks; ++step) {
      BENCHMARKS_RETURN_IF_ERROR(simulation->Step());
    }
    BENCHMARKS_RETURN_IF_ERROR(simulation->Check());
    stats->steps += kStepsBetweenChecks;
  }
  stats->elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  return benchmarks::Status();
}

/**
 * @brief Prints the energies as <prefix>_<NAME>=<value> lines
 */
void PrintEnergies(const char* prefix, const Energies& energies) {
  std::cout << std::scientific << std::setprecision(10) << prefix
            << "_POTENTIAL_PER_ATOM=" << energies.potential_per_atom << '\n'
            << prefix << "_KINETIC_PER_ATOM=" << energies.kinetic_per_atom
            << '\n'
            << prefix << "_TOTAL_ENERGY_PER_ATOM=" << energies.total_per_atom
            << '\n'
            << prefix << "_TEMPERATURE=" << energies.temperature << '\n';
}

/**
 * @brief Prints the size of the simulated system
 */
void PrintSystem(int atoms, int cells_per_side, double box_length) {
  std::cout << std::setprecision(12) << "ATOMS=" << atoms << '\n'
            << "BOX_LENGTH=" << box_length << '\n'
            << "CELLS_PER_SIDE=" << cells_per_side << '\n';
}

/**
 * @brief Runs --verify-steps steps and checks the forces on the host
 *
 * @param config Simulation options
 * @return Error if the force error exceeds the tolerance or a step fails
 */
template <typename Real>
benchmarks::Status Verify(const Config& config) {
  const int n = config.lattice_cells;
  if (4 * n * n * n > kVerifyMaxAtoms) {
    return benchmarks::Status::Error(
        "verify mode needs at most 32000 atoms; use --lattice-cells 20 or "
        "fewer");
  }
  Simulation<Real> simulation;
  BENCHMARKS_RETURN_IF_ERROR(simulation.Init(config));
  PrintSystem(simulation.atoms(), simulation.cells_per_side(),
              simulation.box_length());
  Energies initial;
  BENCHMARKS_RETURN_IF_ERROR(simulation.ComputeEnergies(&initial));
  for (int step = 0; step < config.verify_steps; ++step) {
    BENCHMARKS_RETURN_IF_ERROR(simulation.Step());
  }
  BENCHMARKS_RETURN_IF_ERROR(simulation.Check());
  Energies final_energies;
  BENCHMARKS_RETURN_IF_ERROR(simulation.ComputeEnergies(&final_energies));

  std::vector<double> positions;
  std::vector<double> forces;
  BENCHMARKS_RETURN_IF_ERROR(simulation.Download(&positions, &forces));
  int max_neighbors = 0;
  BENCHMARKS_RETURN_IF_ERROR(simulation.MaxNeighbors(&max_neighbors));
  const double error = MaxRelativeForceError<Real>(
      positions, forces, simulation.box_length(), config.cutoff);
  const double tolerance = config.precision == Precision::kFp32
                               ? kVerifyToleranceFp32
                               : kVerifyToleranceFp64;
  const bool passed = std::isfinite(error) && error <= tolerance;

  PrintEnergies("INITIAL", initial);
  PrintEnergies("FINAL", final_energies);
  std::cout << std::scientific << std::setprecision(6)
            << "VERIFY_STEPS=" << config.verify_steps << '\n'
            << "MAX_NEIGHBORS=" << max_neighbors << '\n'
            << "VERIFY_MAX_RELATIVE_FORCE_ERROR=" << error << '\n'
            << "VERIFY_TOLERANCE=" << tolerance << '\n'
            << "VERIFY_RELATIVE_ENERGY_DRIFT="
            << (final_energies.total_per_atom - initial.total_per_atom) /
                   std::abs(initial.total_per_atom)
            << '\n'
            << "VERIFY=" << (passed ? "OK" : "FAILED") << '\n';
  if (!passed) {
    return benchmarks::Status::Error(
        "force error exceeds the verification tolerance");
  }
  benchmarks::PrintExecutionSummary(0.0, config.verify_steps);
  return benchmarks::Status();
}

/**
 * @brief Runs the warm-up, measurement and cooldown phases
 *
 * @param config Simulation options
 * @return Error if a step fails or a capacity is exceeded
 */
template <typename Real>
benchmarks::Status Measure(const Config& config) {
  Simulation<Real> simulation;
  BENCHMARKS_RETURN_IF_ERROR(simulation.Init(config));
  PrintSystem(simulation.atoms(), simulation.cells_per_side(),
              simulation.box_length());

  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("warmup", PhaseMarker::kWarmup));
  PhaseStats warmup;
  BENCHMARKS_RETURN_IF_ERROR(
      RunPhase(&simulation, config.warmup_seconds, &warmup));
  // Energy reductions sit outside the measured phase.
  Energies start;
  BENCHMARKS_RETURN_IF_ERROR(simulation.ComputeEnergies(&start));

  BENCHMARKS_RETURN_IF_ERROR(
      EmitPhase("measurement", PhaseMarker::kMeasurement));
  const std::int64_t rebuilds_before = simulation.rebuilds();
  PhaseStats measurement;
  BENCHMARKS_RETURN_IF_ERROR(
      RunPhase(&simulation, config.measurement_seconds, &measurement));
  const std::int64_t rebuilds = simulation.rebuilds() - rebuilds_before;

  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("cooldown", PhaseMarker::kCooldown));
  Energies end;
  BENCHMARKS_RETURN_IF_ERROR(simulation.ComputeEnergies(&end));
  const auto cooldown_start = Clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.cooldown_seconds));
  const double cooldown_elapsed =
      std::chrono::duration<double>(Clock::now() - cooldown_start).count();
  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("complete", PhaseMarker::kComplete));

  int max_neighbors = 0;
  BENCHMARKS_RETURN_IF_ERROR(simulation.MaxNeighbors(&max_neighbors));
  const double atom_steps =
      static_cast<double>(measurement.steps) * simulation.atoms();
  PrintEnergies("MEASUREMENT_START", start);
  PrintEnergies("MEASUREMENT_END", end);
  std::cout << std::fixed << std::setprecision(6)
            << "WARMUP_ELAPSED_SECONDS=" << warmup.elapsed_seconds << '\n'
            << "WARMUP_STEPS=" << warmup.steps << '\n'
            << "MEASUREMENT_ELAPSED_SECONDS=" << measurement.elapsed_seconds
            << '\n'
            << "MEASUREMENT_STEPS=" << measurement.steps << '\n'
            << "MEASUREMENT_NEIGHBOR_REBUILDS=" << rebuilds << '\n'
            << "STEPS_PER_SECOND="
            << measurement.steps / measurement.elapsed_seconds << '\n'
            << "MS_PER_STEP="
            << 1000.0 * measurement.elapsed_seconds / measurement.steps << '\n'
            << "ATOM_STEPS_PER_SECOND="
            << atom_steps / measurement.elapsed_seconds << '\n'
            << "MAX_NEIGHBORS=" << max_neighbors << '\n'
            << std::scientific << std::setprecision(6)
            << "RELATIVE_ENERGY_DRIFT="
            << (end.total_per_atom - start.total_per_atom) /
                   std::abs(start.total_per_atom)
            << '\n'
            << std::fixed << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed
            << '\n';
  benchmarks::PrintExecutionSummary(measurement.elapsed_seconds,
                                    measurement.steps);
  return benchmarks::Status();
}

/**
 * @brief Runs the selected mode with the given precision
 */
template <typename Real>
benchmarks::Status RunSimulation(const Config& config) {
  return config.mode == Mode::kVerify ? Verify<Real>(config)
                                      : Measure<Real>(config);
}

/**
 * @brief Parses the options, prints the configuration and runs the
 * simulation
 *
 * @param argc Number of arguments, as received by main()
 * @param argv Arguments, as received by main()
 * @return Error if an option is invalid, a CUDA call fails or the force
 * check fails
 */
benchmarks::Status Run(int argc, char** argv) {
  benchmarks::ArgumentParser arguments;
  BENCHMARKS_RETURN_IF_ERROR(arguments.Parse(argc, argv));
  if (arguments.HelpRequested()) {
    PrintHelp(argv[0]);
    return benchmarks::Status();
  }
  Config config;
  BENCHMARKS_RETURN_IF_ERROR(ParseConfig(arguments, &config));
  cudaDeviceProp properties;
  BENCHMARKS_RETURN_IF_ERROR(
      benchmarks::InitializeCudaDevice(config.device, &properties));
  if (config.block_size > properties.maxThreadsPerBlock) {
    return benchmarks::Status::Error("--block-size exceeds the device limit");
  }
  PrintConfig(config, properties);
  return config.precision == Precision::kFp32 ? RunSimulation<float>(config)
                                              : RunSimulation<double>(config);
}

}  // namespace

int main(int argc, char** argv) {
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
