/**
 * @file bench_quantized_dnn.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Quantized deep neural network inference benchmark
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

// Runs batched inference of a deep multilayer perceptron (default
// 1024 -> 4096 -> 4096 -> 4096 -> 1024, ReLU between layers) in one of three
// precisions with the same tiled GEMM on CUDA cores:
//
//   fp32  float weights and activations, one FMA per 32-bit word
//   fp16  half weights and activations, __hfma2 (two MACs per word), partial
//         sums promoted to float every tile
//   int8  symmetric INT8 weights (per output channel) and activations (per
//         layer), __dp4a (four MACs per word), INT32 accumulation, float
//         dequantization and INT8 requantization in the epilogue
//
// Every precision moves the same number of 32-bit words per tile, so the
// comparison isolates arithmetic density and memory footprint. Tensor cores
// are not used: the V100 has none for INT8, and __hfma2 and __dp4a exist on
// every GPU from compute capability 6.1, including Jetson Thor.
//
// The INT8 activation scales are calibrated on the fp32 activations of the
// input batch. Each run reports the logit error and top-1 agreement against
// fp32. Like bench_power_cycle, the run has warm-up, measurement and cooldown
// phases delimited by empty marker kernels. --mode verify checks one layer of
// each precision against a host reference.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
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
constexpr int kDefaultBatch = 256;
constexpr int kDefaultInputFeatures = 1024;
constexpr int kDefaultHiddenFeatures = 4096;
constexpr int kDefaultHiddenLayers = 3;
constexpr int kDefaultOutputs = 1024;
constexpr std::int64_t kDefaultSeed = 2024;
constexpr int kForwardsBetweenSynchronizations = 5;

// GEMM tiling: a 256-thread block computes a 64 x 64 output tile, 4 x 4
// outputs per thread, reading 16 32-bit words of K per tile.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileWords = 16;
constexpr int kThreads = 256;
constexpr int kThreadsPerRow = 16;
constexpr int kOutputsPerThread = 4;
constexpr int kDimensionMultiple = 64;
constexpr int kInt8Max = 127;

constexpr int kVerifyRows = 64;
constexpr int kVerifyColumns = 128;
constexpr int kVerifyDepth = 512;
constexpr double kVerifyToleranceFp32 = 1e-5;
constexpr double kVerifyToleranceFp16 = 5e-3;
constexpr double kVerifyToleranceInt8 = 1e-5;

// Weights and activations travel as packed 32-bit words. They are raw bit
// patterns of one to four values, so unsigned int is the right type, and it
// is the type that __uint_as_float, __dp4a and the half2 casts take.

/** Type of weights and activations */
enum class Precision { kFp32, kFp16, kInt8 };
/** Timed measurement or check of one layer against the host */
enum class Mode { kMeasure, kVerify };

/**
 * @brief Options of one benchmark run
 */
struct Config {
  /** CUDA device index */
  int device = benchmarks::kDefaultDevice;
  /** Duration of the warm-up phase */
  double warmup_seconds = kDefaultWarmupSeconds;
  /** Duration of the measurement phase */
  double measurement_seconds = kDefaultMeasurementSeconds;
  /** Duration of the context-idle cooldown phase */
  double cooldown_seconds = kDefaultCooldownSeconds;
  /** Samples per forward pass */
  int batch = kDefaultBatch;
  /** Width of the input layer */
  int input_features = kDefaultInputFeatures;
  /** Width of every hidden layer */
  int hidden_features = kDefaultHiddenFeatures;
  /** Number of hidden layers */
  int hidden_layers = kDefaultHiddenLayers;
  /** Width of the output layer */
  int outputs = kDefaultOutputs;
  /** Seed of the weights and the input batch */
  std::int64_t seed = kDefaultSeed;
  /** Type of weights and activations */
  Precision precision = Precision::kInt8;
  /** What the run does */
  Mode mode = Mode::kMeasure;
};

/**
 * @brief Epilogue inputs of one layer; the scales are used by INT8 only
 */
struct LayerParams {
  /** Bias per output channel */
  const float* bias = nullptr;
  /** Weight scale per output channel */
  const float* weight_scale = nullptr;
  /** Scale of the layer input */
  float input_scale = 1.0F;
  /** Inverse scale of the layer output, used to requantize */
  float inverse_output_scale = 1.0F;
  /** Whether ReLU follows the layer */
  bool relu = true;
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
 * @brief Reinterprets a 32-bit word as two half values
 */
__device__ __forceinline__ __half2 BitsAsHalf2(unsigned int bits) {
  __half2 value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

/**
 * @brief float weights and activations, one FMA per word
 */
struct Fp32Policy {
  using Value = float;
  using Acc = float;
  using Sum = float;
  static constexpr int kValuesPerWord = 1;
  static const char* Name() { return "fp32"; }
  __device__ static Acc ZeroAcc() { return 0.0F; }
  __device__ static Sum ZeroSum() { return 0.0F; }
  __device__ static Acc Mac(Acc acc, unsigned int a, unsigned int w) {
    return fmaf(__uint_as_float(a), __uint_as_float(w), acc);
  }
  __device__ static void Flush(Sum* sum, Acc* acc) {
    *sum += *acc;
    *acc = 0.0F;
  }
  __device__ static float Dequantize(Sum sum, const LayerParams&, int) {
    return sum;
  }
  __device__ static Value Quantize(float value, const LayerParams&) {
    return value;
  }
};

/**
 * @brief half weights and activations, two MACs per word with __hfma2
 */
struct Fp16Policy {
  using Value = __half;
  using Acc = __half2;
  using Sum = float;
  static constexpr int kValuesPerWord = 2;
  static const char* Name() { return "fp16"; }
  __device__ static Acc ZeroAcc() { return __float2half2_rn(0.0F); }
  __device__ static Sum ZeroSum() { return 0.0F; }
  __device__ static Acc Mac(Acc acc, unsigned int a, unsigned int w) {
    return __hfma2(BitsAsHalf2(a), BitsAsHalf2(w), acc);
  }
  // Each half lane holds at most 16 products before it is promoted.
  __device__ static void Flush(Sum* sum, Acc* acc) {
    *sum += __low2float(*acc) + __high2float(*acc);
    *acc = ZeroAcc();
  }
  __device__ static float Dequantize(Sum sum, const LayerParams&, int) {
    return sum;
  }
  __device__ static Value Quantize(float value, const LayerParams&) {
    return __float2half_rn(value);
  }
};

/**
 * @brief Symmetric INT8 weights and activations, four MACs per word with
 * __dp4a
 */
struct Int8Policy {
  using Value = std::int8_t;
  using Acc = int;
  using Sum = int;
  static constexpr int kValuesPerWord = 4;
  static const char* Name() { return "int8"; }
  __device__ static Acc ZeroAcc() { return 0; }
  __device__ static Sum ZeroSum() { return 0; }
  __device__ static Acc Mac(Acc acc, unsigned int a, unsigned int w) {
    return __dp4a(static_cast<int>(a), static_cast<int>(w), acc);
  }
  __device__ static void Flush(Sum* sum, Acc* acc) {
    *sum += *acc;
    *acc = 0;
  }
  __device__ static float Dequantize(Sum sum, const LayerParams& params,
                                     int column) {
    return static_cast<float>(sum) * params.input_scale *
           params.weight_scale[column];
  }
  __device__ static Value Quantize(float value, const LayerParams& params) {
    const int level = __float2int_rn(value * params.inverse_output_scale);
    return static_cast<Value>(min(max(level, -kInt8Max), kInt8Max));
  }
};

/**
 * @brief Tiled GEMM with the epilogue of one layer
 *
 * output[m][n] = act(dequantize(sum_k input[m][k] * weights[n][k]) + bias[n]).
 * input and weights are row-major with K contiguous, packed in 32-bit words.
 *
 * @tparam Policy Precision of the layer
 * @tparam kFinal Writes float logits instead of the next layer's activations
 * @param input Activations, rows x k_words
 * @param weights Weights, columns x k_words
 * @param output Next activations or logits, rows x columns
 * @param columns Output width
 * @param k_words Input width in 32-bit words
 * @param params Epilogue inputs
 */
template <typename Policy, bool kFinal>
__global__ __launch_bounds__(kThreads) void QdnnLinearKernel(
    const unsigned int* __restrict__ input,
    const unsigned int* __restrict__ weights, void* __restrict__ output,
    int columns, int k_words, LayerParams params) {
  __shared__ unsigned int input_tile[kTileWords][kTileM + 1];
  __shared__ unsigned int weight_tile[kTileWords][kTileN + 1];
  const int tx = threadIdx.x % kThreadsPerRow;
  const int ty = threadIdx.x / kThreadsPerRow;
  const int row0 = blockIdx.y * kTileM;
  const int column0 = blockIdx.x * kTileN;

  typename Policy::Acc acc[kOutputsPerThread][kOutputsPerThread];
  typename Policy::Sum sum[kOutputsPerThread][kOutputsPerThread];
  for (int i = 0; i < kOutputsPerThread; ++i) {
    for (int j = 0; j < kOutputsPerThread; ++j) {
      acc[i][j] = Policy::ZeroAcc();
      sum[i][j] = Policy::ZeroSum();
    }
  }

  for (int k0 = 0; k0 < k_words; k0 += kTileWords) {
    for (int element = threadIdx.x; element < kTileM * kTileWords;
         element += kThreads) {
      const int row = element / kTileWords;
      const int word = element % kTileWords;
      input_tile[word][row] =
          input[static_cast<std::size_t>(row0 + row) * k_words + k0 + word];
      weight_tile[word][row] =
          weights[static_cast<std::size_t>(column0 + row) * k_words + k0 +
                  word];
    }
    __syncthreads();
#pragma unroll
    for (int word = 0; word < kTileWords; ++word) {
      unsigned int a[kOutputsPerThread];
      unsigned int w[kOutputsPerThread];
#pragma unroll
      for (int i = 0; i < kOutputsPerThread; ++i) {
        a[i] = input_tile[word][ty * kOutputsPerThread + i];
        w[i] = weight_tile[word][tx * kOutputsPerThread + i];
      }
#pragma unroll
      for (int i = 0; i < kOutputsPerThread; ++i) {
#pragma unroll
        for (int j = 0; j < kOutputsPerThread; ++j) {
          acc[i][j] = Policy::Mac(acc[i][j], a[i], w[j]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kOutputsPerThread; ++i) {
#pragma unroll
      for (int j = 0; j < kOutputsPerThread; ++j) {
        Policy::Flush(&sum[i][j], &acc[i][j]);
      }
    }
  }

  for (int i = 0; i < kOutputsPerThread; ++i) {
    const int row = row0 + ty * kOutputsPerThread + i;
    for (int j = 0; j < kOutputsPerThread; ++j) {
      const int column = column0 + tx * kOutputsPerThread + j;
      float value =
          Policy::Dequantize(sum[i][j], params, column) + params.bias[column];
      if (params.relu) value = fmaxf(value, 0.0F);
      const std::size_t index =
          static_cast<std::size_t>(row) * columns + column;
      if constexpr (kFinal) {
        static_cast<float*>(output)[index] = value;
      } else {
        static_cast<typename Policy::Value*>(output)[index] =
            Policy::Quantize(value, params);
      }
    }
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
 * @brief Allocates device memory and copies values into it
 *
 * @param values Host values
 * @param buffer Output that owns the allocation. Must not be null
 * @return Error if the allocation or the copy fails
 */
template <typename T>
benchmarks::Status CopyToDevice(const std::vector<T>& values,
                                benchmarks::DeviceBuffer<T>* buffer) {
  BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(values.size(), buffer));
  BENCHMARKS_CUDA_CHECK(cudaMemcpy(buffer->get(), values.data(),
                                   values.size() * sizeof(T),
                                   cudaMemcpyHostToDevice));
  return benchmarks::Status();
}

/**
 * @brief Packs host values of one precision into 32-bit words
 */
template <typename Value>
std::vector<unsigned int> PackWords(const std::vector<Value>& values) {
  std::vector<unsigned int> words(values.size() * sizeof(Value) /
                                  sizeof(unsigned int));
  std::memcpy(words.data(), values.data(), words.size() * sizeof(unsigned int));
  return words;
}

/**
 * @brief Rounds every value to half
 */
std::vector<__half> ToHalf(const std::vector<float>& values) {
  std::vector<__half> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[i] = __float2half_rn(values[i]);
  }
  return result;
}

/**
 * @brief Returns the largest absolute value
 */
float MaxAbs(const std::vector<float>& values) {
  float result = 0.0F;
  for (float value : values) result = std::max(result, std::abs(value));
  return result;
}

/**
 * @brief Rounds value / scale to the nearest INT8 level in [-127, 127]
 */
std::int8_t QuantizeLevel(float value, float inverse_scale) {
  const std::int64_t level = std::llround(value * inverse_scale);
  return static_cast<std::int8_t>(
      std::clamp<std::int64_t>(level, -kInt8Max, kInt8Max));
}

/**
 * @brief Quantizes weights to symmetric INT8 with one scale per row
 *
 * @param weights rows x depth, row-major
 * @param rows Output channels
 * @param depth Input width
 * @param levels Output, INT8 levels. Must not be null
 * @param scales Output, one scale per row. Must not be null
 */
void QuantizeWeights(const std::vector<float>& weights, int rows, int depth,
                     std::vector<std::int8_t>* levels,
                     std::vector<float>* scales) {
  levels->resize(weights.size());
  scales->resize(rows);
  for (int row = 0; row < rows; ++row) {
    const auto begin = weights.begin() + static_cast<std::size_t>(row) * depth;
    float largest = 0.0F;
    for (auto value = begin; value != begin + depth; ++value) {
      largest = std::max(largest, std::abs(*value));
    }
    const float scale = largest > 0.0F ? largest / kInt8Max : 1.0F;
    (*scales)[row] = scale;
    for (int k = 0; k < depth; ++k) {
      const std::size_t index = static_cast<std::size_t>(row) * depth + k;
      (*levels)[index] = QuantizeLevel(weights[index], 1.0F / scale);
    }
  }
}

/**
 * @brief Quantizes activations to INT8 with one scale
 */
std::vector<std::int8_t> QuantizeActivations(const std::vector<float>& values,
                                             float scale) {
  std::vector<std::int8_t> levels(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    levels[i] = QuantizeLevel(values[i], 1.0F / scale);
  }
  return levels;
}

/**
 * @brief fp32 master weights and input batch, generated deterministically
 */
struct HostNetwork {
  /** Samples per forward pass */
  int batch = 0;
  /** Widths: input, hidden..., outputs */
  std::vector<int> widths;
  /** Weights per layer, out x in, row-major */
  std::vector<std::vector<float>> weights;
  /** Bias per layer */
  std::vector<std::vector<float>> biases;
  /** Input batch, batch x input width */
  std::vector<float> input;
};

/**
 * @brief Returns the number of layers of the network
 */
int LayerCount(const HostNetwork& network) {
  return static_cast<int>(network.weights.size());
}

/**
 * @brief Returns the multiply-accumulates of one sample
 */
double MacsPerSample(const HostNetwork& network) {
  double macs = 0.0;
  for (int layer = 0; layer < LayerCount(network); ++layer) {
    macs +=
        static_cast<double>(network.widths[layer]) * network.widths[layer + 1];
  }
  return macs;
}

/**
 * @brief Builds the network with He-uniform weights and a random input
 */
HostNetwork MakeNetwork(const Config& config) {
  HostNetwork network;
  network.batch = config.batch;
  network.widths.push_back(config.input_features);
  for (int layer = 0; layer < config.hidden_layers; ++layer) {
    network.widths.push_back(config.hidden_features);
  }
  network.widths.push_back(config.outputs);
  std::mt19937_64 generator(static_cast<std::uint64_t>(config.seed));
  std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
  for (std::size_t layer = 0; layer + 1 < network.widths.size(); ++layer) {
    const int depth = network.widths[layer];
    const int rows = network.widths[layer + 1];
    // He-uniform initialization keeps ReLU activations at a stable scale.
    const float limit = std::sqrt(6.0F / depth);
    std::vector<float> weights(static_cast<std::size_t>(rows) * depth);
    for (float& weight : weights) weight = limit * uniform(generator);
    std::vector<float> bias(rows);
    for (float& value : bias) value = 0.01F * uniform(generator);
    network.weights.push_back(std::move(weights));
    network.biases.push_back(std::move(bias));
  }
  network.input.resize(static_cast<std::size_t>(config.batch) *
                       config.input_features);
  for (float& value : network.input) value = uniform(generator);
  return network;
}

/**
 * @brief Device copy of the network in one precision; runs forward passes
 */
template <typename Policy>
class DeviceModel {
 public:
  using Value = typename Policy::Value;

  DeviceModel() = default;
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /**
   * @brief Copies the network to the device in this precision
   *
   * @param network fp32 master network
   * @param input_scales INT8 scale of the input of each layer. Ignored by
   * the other precisions
   * @return Error if an allocation or a copy fails
   */
  benchmarks::Status Init(const HostNetwork& network,
                          const std::vector<float>& input_scales) {
    batch_ = network.batch;
    widths_ = network.widths;
    const int widest = *std::max_element(widths_.begin(), widths_.end());
    const std::size_t activation_bytes =
        static_cast<std::size_t>(batch_) * widest * sizeof(Value);
    for (auto& buffer : activations_) {
      BENCHMARKS_RETURN_IF_ERROR(
          AllocateDevice(activation_bytes / sizeof(unsigned int), &buffer));
    }
    BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(
        static_cast<std::size_t>(batch_) * widths_.back(), &logits_));
    const int layer_count = LayerCount(network);
    for (int layer = 0; layer < layer_count; ++layer) {
      Layer device_layer;
      device_layer.rows = widths_[layer + 1];
      device_layer.depth = widths_[layer];
      BENCHMARKS_RETURN_IF_ERROR(
          CopyToDevice(network.biases[layer], &device_layer.bias));
      device_layer.params.bias = device_layer.bias.get();
      device_layer.params.relu = layer + 1 < layer_count;
      if constexpr (std::is_same_v<Policy, Int8Policy>) {
        std::vector<std::int8_t> levels;
        std::vector<float> scales;
        QuantizeWeights(network.weights[layer], device_layer.rows,
                        device_layer.depth, &levels, &scales);
        BENCHMARKS_RETURN_IF_ERROR(
            CopyToDevice(PackWords(levels), &device_layer.weights));
        BENCHMARKS_RETURN_IF_ERROR(
            CopyToDevice(scales, &device_layer.weight_scale));
        device_layer.params.weight_scale = device_layer.weight_scale.get();
        device_layer.params.input_scale = input_scales[layer];
        if (layer + 1 < layer_count) {
          device_layer.params.inverse_output_scale =
              1.0F / input_scales[layer + 1];
        }
      } else if constexpr (std::is_same_v<Policy, Fp16Policy>) {
        BENCHMARKS_RETURN_IF_ERROR(CopyToDevice(
            PackWords(ToHalf(network.weights[layer])), &device_layer.weights));
      } else {
        BENCHMARKS_RETURN_IF_ERROR(CopyToDevice(
            PackWords(network.weights[layer]), &device_layer.weights));
      }
      layers_.push_back(std::move(device_layer));
    }
    std::vector<unsigned int> input_words;
    if constexpr (std::is_same_v<Policy, Int8Policy>) {
      input_words =
          PackWords(QuantizeActivations(network.input, input_scales[0]));
    } else if constexpr (std::is_same_v<Policy, Fp16Policy>) {
      input_words = PackWords(ToHalf(network.input));
    } else {
      input_words = PackWords(network.input);
    }
    return CopyToDevice(input_words, &input_);
  }

  /**
   * @brief Queues one batched forward pass
   *
   * @param capture When not null, the fp32 activations entering every layer
   * after the first are copied back, to calibrate the INT8 scales
   * @return Error if a launch or a copy fails
   */
  benchmarks::Status Forward(
      std::vector<std::vector<float>>* capture = nullptr) {
    const unsigned int* input = input_.get();
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
      const Layer& current = layers_[layer];
      const dim3 grid(current.rows / kTileN, batch_ / kTileM);
      const int k_words = current.depth / Policy::kValuesPerWord;
      const bool final_layer = layer + 1 == layers_.size();
      unsigned int* output = activations_[layer % 2].get();
      if (final_layer) {
        QdnnLinearKernel<Policy, true>
            <<<grid, kThreads>>>(input, current.weights.get(), logits_.get(),
                                 current.rows, k_words, current.params);
      } else {
        QdnnLinearKernel<Policy, false>
            <<<grid, kThreads>>>(input, current.weights.get(), output,
                                 current.rows, k_words, current.params);
      }
      BENCHMARKS_CUDA_CHECK(cudaGetLastError());
      if (capture != nullptr && !final_layer) {
        if constexpr (std::is_same_v<Policy, Fp32Policy>) {
          std::vector<float> values(static_cast<std::size_t>(batch_) *
                                    current.rows);
          BENCHMARKS_CUDA_CHECK(cudaMemcpy(values.data(), output,
                                           values.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost));
          capture->push_back(std::move(values));
        }
      }
      input = output;
    }
    return benchmarks::Status();
  }

  /**
   * @brief Copies the logits of the last forward pass to the host
   *
   * @param values Output, batch x outputs. Must not be null
   * @return Error if the copy fails
   */
  benchmarks::Status Logits(std::vector<float>* values) const {
    values->resize(static_cast<std::size_t>(batch_) * widths_.back());
    BENCHMARKS_CUDA_CHECK(cudaMemcpy(values->data(), logits_.get(),
                                     values->size() * sizeof(float),
                                     cudaMemcpyDeviceToHost));
    return benchmarks::Status();
  }

 private:
  /**
   * @brief Device data of one layer
   */
  struct Layer {
    /** Output channels */
    int rows = 0;
    /** Input width */
    int depth = 0;
    /** Packed weights, rows x depth */
    benchmarks::DeviceBuffer<unsigned int> weights;
    /** Bias per output channel */
    benchmarks::DeviceBuffer<float> bias;
    /** INT8 weight scale per output channel */
    benchmarks::DeviceBuffer<float> weight_scale;
    /** Epilogue inputs, pointing into the buffers above */
    LayerParams params;
  };

  /** Samples per forward pass */
  int batch_ = 0;
  /** Widths: input, hidden..., outputs */
  std::vector<int> widths_;
  /** Device layers */
  std::vector<Layer> layers_;
  /** Ping-pong buffers of the hidden activations */
  benchmarks::DeviceBuffer<unsigned int> activations_[2];
  /** Packed input batch */
  benchmarks::DeviceBuffer<unsigned int> input_;
  /** float logits of the last layer */
  benchmarks::DeviceBuffer<float> logits_;
};

/**
 * @brief Runs the fp32 network and computes the INT8 input scales
 *
 * The scale of each layer input is max |activation| / 127 of its fp32 value.
 *
 * @param network fp32 master network
 * @param scales Output, one scale per layer. Must not be null
 * @param fp32_logits Output, logits of the fp32 network. Must not be null
 * @return Error if a CUDA call fails
 */
benchmarks::Status Calibrate(const HostNetwork& network,
                             std::vector<float>* scales,
                             std::vector<float>* fp32_logits) {
  DeviceModel<Fp32Policy> reference;
  BENCHMARKS_RETURN_IF_ERROR(reference.Init(network, {}));
  std::vector<std::vector<float>> hidden;
  BENCHMARKS_RETURN_IF_ERROR(reference.Forward(&hidden));
  BENCHMARKS_RETURN_IF_ERROR(reference.Logits(fp32_logits));
  scales->clear();
  scales->push_back(MaxAbs(network.input) / kInt8Max);
  for (const auto& values : hidden) {
    const float largest = MaxAbs(values);
    scales->push_back(largest > 0.0F ? largest / kInt8Max : 1.0F);
  }
  return benchmarks::Status();
}

/**
 * @brief Accuracy of a network against the fp32 network
 */
struct Accuracy {
  /** Relative L2 error of the logits */
  double relative_l2_error = 0.0;
  /** Samples with the same arg-max class, in percent */
  double top1_agreement_pct = 0.0;
};

/**
 * @brief Compares logits against the fp32 reference logits
 */
Accuracy CompareLogits(const std::vector<float>& reference,
                       const std::vector<float>& logits, int batch) {
  const std::size_t classes = reference.size() / batch;
  double error_sq = 0.0;
  double reference_sq = 0.0;
  int agreements = 0;
  for (int sample = 0; sample < batch; ++sample) {
    const auto first = reference.begin() + sample * classes;
    const auto other = logits.begin() + sample * classes;
    if (std::max_element(first, first + classes) - first ==
        std::max_element(other, other + classes) - other) {
      ++agreements;
    }
    for (std::size_t c = 0; c < classes; ++c) {
      const double difference = other[c] - first[c];
      error_sq += difference * difference;
      reference_sq += static_cast<double>(first[c]) * first[c];
    }
  }
  return {std::sqrt(error_sq / reference_sq), 100.0 * agreements / batch};
}

/**
 * @brief Checks one final layer (no ReLU) against a host reference
 *
 * The reference uses the same rounded or quantized inputs, so INT8 sums are
 * exact on the host.
 *
 * @param seed Seed of the layer data
 * @param error Output, relative L2 error. Must not be null
 * @return Error if a CUDA call fails
 */
template <typename Policy>
benchmarks::Status VerifyLayer(std::int64_t seed, double* error) {
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed));
  std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
  std::vector<float> input(static_cast<std::size_t>(kVerifyRows) *
                           kVerifyDepth);
  std::vector<float> weights(static_cast<std::size_t>(kVerifyColumns) *
                             kVerifyDepth);
  std::vector<float> bias(kVerifyColumns);
  for (float& value : input) value = uniform(generator);
  for (float& value : weights) value = uniform(generator) / 16.0F;
  for (float& value : bias) value = uniform(generator);

  std::vector<unsigned int> input_words;
  std::vector<unsigned int> weight_words;
  std::vector<double> input_used(input.begin(), input.end());
  std::vector<double> weight_used(weights.begin(), weights.end());
  std::vector<float> weight_scales(kVerifyColumns, 1.0F);
  float input_scale = 1.0F;
  if constexpr (std::is_same_v<Policy, Int8Policy>) {
    input_scale = MaxAbs(input) / kInt8Max;
    const auto input_levels = QuantizeActivations(input, input_scale);
    std::vector<std::int8_t> weight_levels;
    QuantizeWeights(weights, kVerifyColumns, kVerifyDepth, &weight_levels,
                    &weight_scales);
    input_words = PackWords(input_levels);
    weight_words = PackWords(weight_levels);
    input_used.assign(input_levels.begin(), input_levels.end());
    weight_used.assign(weight_levels.begin(), weight_levels.end());
  } else if constexpr (std::is_same_v<Policy, Fp16Policy>) {
    const auto input_half = ToHalf(input);
    const auto weight_half = ToHalf(weights);
    input_words = PackWords(input_half);
    weight_words = PackWords(weight_half);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input_used[i] = __half2float(input_half[i]);
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weight_used[i] = __half2float(weight_half[i]);
    }
  } else {
    input_words = PackWords(input);
    weight_words = PackWords(weights);
  }

  benchmarks::DeviceBuffer<unsigned int> device_input;
  BENCHMARKS_RETURN_IF_ERROR(CopyToDevice(input_words, &device_input));
  benchmarks::DeviceBuffer<unsigned int> device_weights;
  BENCHMARKS_RETURN_IF_ERROR(CopyToDevice(weight_words, &device_weights));
  benchmarks::DeviceBuffer<float> device_bias;
  BENCHMARKS_RETURN_IF_ERROR(CopyToDevice(bias, &device_bias));
  benchmarks::DeviceBuffer<float> device_scales;
  BENCHMARKS_RETURN_IF_ERROR(CopyToDevice(weight_scales, &device_scales));
  benchmarks::DeviceBuffer<float> device_output;
  BENCHMARKS_RETURN_IF_ERROR(AllocateDevice(
      static_cast<std::size_t>(kVerifyRows) * kVerifyColumns, &device_output));
  LayerParams params;
  params.bias = device_bias.get();
  params.weight_scale = device_scales.get();
  params.input_scale = input_scale;
  params.relu = false;
  QdnnLinearKernel<Policy, true>
      <<<dim3(kVerifyColumns / kTileN, kVerifyRows / kTileM), kThreads>>>(
          device_input.get(), device_weights.get(), device_output.get(),
          kVerifyColumns, kVerifyDepth / Policy::kValuesPerWord, params);
  BENCHMARKS_CUDA_CHECK(cudaGetLastError());
  std::vector<float> output(static_cast<std::size_t>(kVerifyRows) *
                            kVerifyColumns);
  BENCHMARKS_CUDA_CHECK(cudaMemcpy(output.data(), device_output.get(),
                                   output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));

  double error_sq = 0.0;
  double reference_sq = 0.0;
  for (int row = 0; row < kVerifyRows; ++row) {
    for (int column = 0; column < kVerifyColumns; ++column) {
      double sum = 0.0;
      for (int k = 0; k < kVerifyDepth; ++k) {
        sum += input_used[static_cast<std::size_t>(row) * kVerifyDepth + k] *
               weight_used[static_cast<std::size_t>(column) * kVerifyDepth + k];
      }
      const double expected =
          sum * input_scale * weight_scales[column] + bias[column];
      const double difference =
          output[static_cast<std::size_t>(row) * kVerifyColumns + column] -
          expected;
      error_sq += difference * difference;
      reference_sq += expected * expected;
    }
  }
  *error = std::sqrt(error_sq / reference_sq);
  return benchmarks::Status();
}

/**
 * @brief Prints the usage and the options of the benchmark
 *
 * @param program Name of the executable, from argv[0]
 */
void PrintHelp(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Quantized deep neural network (MLP) inference: fp32, fp16 or "
         "int8.\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --precision <name>        fp32, fp16 or int8 (default: int8)\n"
      << "  --mode <name>             measure or verify (default: measure)\n"
      << "  --batch <N>               Samples per forward pass (default: 256)\n"
      << "  --input-features <N>      Input width (default: 1024)\n"
      << "  --hidden-features <N>     Hidden width (default: 4096)\n"
      << "  --hidden-layers <N>       Hidden layers (default: 3)\n"
      << "  --outputs <N>             Output width (default: 1024)\n"
      << "                            Widths and batch must be multiples of "
         "64\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 10)\n"
      << "  --duration <seconds>      Measurement phase (default: 40)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 10)\n"
      << "  --seed <N>                Weight and input seed (default: 2024)\n"
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
      {"--device", "--precision", "--mode", "--batch", "--input-features",
       "--hidden-features", "--hidden-layers", "--outputs", "--warmup-seconds",
       "--duration", "--cooldown-seconds", "--seed"}));
  *config = Config();
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--device", config->device, &config->device));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--batch", config->batch, &config->batch));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--input-features", config->input_features, &config->input_features));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--hidden-features", config->hidden_features, &config->hidden_features));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--hidden-layers", config->hidden_layers, &config->hidden_layers));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--outputs", config->outputs, &config->outputs));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--warmup-seconds", config->warmup_seconds, &config->warmup_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--duration", config->measurement_seconds, &config->measurement_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue("--cooldown-seconds",
                                                   config->cooldown_seconds,
                                                   &config->cooldown_seconds));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.Int64Value("--seed", config->seed, &config->seed));

  const std::string precision = arguments.StringValue("--precision", "int8");
  if (precision == "fp32") {
    config->precision = Precision::kFp32;
  } else if (precision == "fp16") {
    config->precision = Precision::kFp16;
  } else if (precision == "int8") {
    config->precision = Precision::kInt8;
  } else {
    return benchmarks::Status::Error("--precision must be fp32, fp16 or int8");
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
  for (int value : {config->batch, config->input_features,
                    config->hidden_features, config->outputs}) {
    if (value <= 0 || value % kDimensionMultiple != 0) {
      return benchmarks::Status::Error(
          "--batch and every layer width must be positive multiples of 64");
    }
  }
  if (config->hidden_layers < 0) {
    return benchmarks::Status::Error("--hidden-layers must be non-negative");
  }
  if (!(config->warmup_seconds >= 0.0) ||
      !(config->measurement_seconds > 0.0) ||
      !(config->cooldown_seconds >= 0.0)) {
    return benchmarks::Status::Error(
        "phase durations must be finite; measurement must be positive");
  }
  if (config->seed < 0) {
    return benchmarks::Status::Error("--seed must be non-negative");
  }
  return benchmarks::Status();
}

/**
 * @brief Returns the --precision name of a precision
 */
const char* PrecisionName(Precision precision) {
  switch (precision) {
    case Precision::kFp32:
      return Fp32Policy::Name();
    case Precision::kFp16:
      return Fp16Policy::Name();
    case Precision::kInt8:
      return Int8Policy::Name();
  }
  return "unknown";
}

/**
 * @brief Prints the configuration and the device as KEY=VALUE lines
 */
void PrintConfig(const Config& config, const cudaDeviceProp& properties) {
  std::cout << std::setprecision(12) << "BENCHMARK=quantized_dnn\n"
            << "MODEL=mlp\n"
            << "PRECISION=" << PrecisionName(config.precision) << '\n'
            << "MODE=" << (config.mode == Mode::kMeasure ? "measure" : "verify")
            << '\n'
            << "DEVICE=" << config.device << '\n'
            << "DEVICE_NAME=" << properties.name << '\n'
            << "COMPUTE_CAPABILITY=" << properties.major << '.'
            << properties.minor << '\n'
            << "BATCH_SIZE=" << config.batch << '\n'
            << "INPUT_FEATURES=" << config.input_features << '\n'
            << "HIDDEN_FEATURES=" << config.hidden_features << '\n'
            << "HIDDEN_LAYERS=" << config.hidden_layers << '\n'
            << "OUTPUTS=" << config.outputs << '\n'
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
 * @brief Counters of one timed phase
 */
struct PhaseStats {
  /** Wall time of the phase */
  double elapsed_seconds = 0.0;
  /** Batched forward passes in the phase */
  std::int64_t forwards = 0;
};

/**
 * @brief Runs forward passes until the phase time runs out
 *
 * @param model Model to run. Must not be null
 * @param seconds Duration of the phase
 * @param stats Output. Must not be null
 * @return Error if a CUDA call fails
 */
template <typename Policy>
benchmarks::Status RunPhase(DeviceModel<Policy>* model, double seconds,
                            PhaseStats* stats) {
  *stats = PhaseStats();
  const auto start = Clock::now();
  const auto deadline = start + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(seconds));
  while (Clock::now() < deadline) {
    for (int i = 0; i < kForwardsBetweenSynchronizations; ++i) {
      BENCHMARKS_RETURN_IF_ERROR(model->Forward());
    }
    BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
    stats->forwards += kForwardsBetweenSynchronizations;
  }
  stats->elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  return benchmarks::Status();
}

/**
 * @brief Builds the model, then runs the warm-up, measurement and cooldown
 * phases
 *
 * @param config Benchmark options
 * @return Error if a CUDA call fails
 */
template <typename Policy>
benchmarks::Status Measure(const Config& config) {
  const HostNetwork network = MakeNetwork(config);
  std::vector<float> scales;
  std::vector<float> fp32_logits;
  BENCHMARKS_RETURN_IF_ERROR(Calibrate(network, &scales, &fp32_logits));
  DeviceModel<Policy> model;
  BENCHMARKS_RETURN_IF_ERROR(model.Init(network, scales));
  BENCHMARKS_RETURN_IF_ERROR(model.Forward());
  std::vector<float> logits;
  BENCHMARKS_RETURN_IF_ERROR(model.Logits(&logits));
  const Accuracy accuracy = CompareLogits(fp32_logits, logits, config.batch);

  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("warmup", PhaseMarker::kWarmup));
  PhaseStats warmup;
  BENCHMARKS_RETURN_IF_ERROR(RunPhase(&model, config.warmup_seconds, &warmup));
  BENCHMARKS_RETURN_IF_ERROR(
      EmitPhase("measurement", PhaseMarker::kMeasurement));
  PhaseStats measurement;
  BENCHMARKS_RETURN_IF_ERROR(
      RunPhase(&model, config.measurement_seconds, &measurement));
  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("cooldown", PhaseMarker::kCooldown));
  const auto cooldown_start = Clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.cooldown_seconds));
  const double cooldown_elapsed =
      std::chrono::duration<double>(Clock::now() - cooldown_start).count();
  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("complete", PhaseMarker::kComplete));

  const double samples =
      static_cast<double>(measurement.forwards) * config.batch;
  const double macs_per_sample = MacsPerSample(network);
  const double macs = samples * macs_per_sample;
  std::cout << std::fixed << std::setprecision(6)
            << "LAYERS=" << LayerCount(network) << '\n'
            << "MACS_PER_SAMPLE=" << std::setprecision(0) << macs_per_sample
            << '\n'
            << std::setprecision(6)
            << "WARMUP_ELAPSED_SECONDS=" << warmup.elapsed_seconds << '\n'
            << "WARMUP_FORWARDS=" << warmup.forwards << '\n'
            << "MEASUREMENT_ELAPSED_SECONDS=" << measurement.elapsed_seconds
            << '\n'
            << "MEASUREMENT_FORWARDS=" << measurement.forwards << '\n'
            << "MEASUREMENT_SAMPLES=" << std::setprecision(0) << samples << '\n'
            << std::setprecision(6)
            << "SAMPLES_PER_SECOND=" << samples / measurement.elapsed_seconds
            << '\n'
            << "MS_PER_BATCH="
            << 1000.0 * measurement.elapsed_seconds / measurement.forwards
            << '\n'
            << "TERA_MACS_PER_SECOND="
            << macs / measurement.elapsed_seconds / 1e12 << '\n'
            << std::scientific
            << "REL_L2_ERROR_VS_FP32=" << accuracy.relative_l2_error << '\n'
            << std::fixed << std::setprecision(3)
            << "TOP1_AGREEMENT_VS_FP32_PCT=" << accuracy.top1_agreement_pct
            << '\n'
            << std::setprecision(6)
            << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed << '\n';
  benchmarks::PrintExecutionSummary(measurement.elapsed_seconds,
                                    measurement.forwards);
  return benchmarks::Status();
}

/**
 * @brief Checks one layer of each precision against the host
 *
 * @param config Benchmark options
 * @return Error if a precision exceeds its tolerance or a CUDA call fails
 */
benchmarks::Status Verify(const Config& config) {
  double errors[3] = {0.0, 0.0, 0.0};
  BENCHMARKS_RETURN_IF_ERROR(VerifyLayer<Fp32Policy>(config.seed, &errors[0]));
  BENCHMARKS_RETURN_IF_ERROR(VerifyLayer<Fp16Policy>(config.seed, &errors[1]));
  BENCHMARKS_RETURN_IF_ERROR(VerifyLayer<Int8Policy>(config.seed, &errors[2]));
  const double tolerances[3] = {kVerifyToleranceFp32, kVerifyToleranceFp16,
                                kVerifyToleranceInt8};
  const char* names[3] = {"FP32", "FP16", "INT8"};
  bool passed = true;
  std::cout << std::scientific << std::setprecision(6);
  for (int i = 0; i < 3; ++i) {
    const bool ok = std::isfinite(errors[i]) && errors[i] <= tolerances[i];
    passed = passed && ok;
    std::cout << "VERIFY_" << names[i] << "_REL_L2_ERROR=" << errors[i] << '\n'
              << "VERIFY_" << names[i] << "_TOLERANCE=" << tolerances[i] << '\n'
              << "VERIFY_" << names[i] << '=' << (ok ? "OK" : "FAILED") << '\n';
  }
  std::cout << "VERIFY=" << (passed ? "OK" : "FAILED") << '\n';
  if (!passed) {
    return benchmarks::Status::Error(
        "layer error exceeds the verification tolerance");
  }
  benchmarks::PrintExecutionSummary(0.0, 3);
  return benchmarks::Status();
}

/**
 * @brief Parses the options, prints the configuration and runs the selected
 * mode
 *
 * @param argc Number of arguments, as received by main()
 * @param argv Arguments, as received by main()
 * @return Error if an option is invalid, a CUDA call fails or the check
 * fails
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
  PrintConfig(config, properties);
  if (config.mode == Mode::kVerify) return Verify(config);
  switch (config.precision) {
    case Precision::kFp32:
      return Measure<Fp32Policy>(config);
    case Precision::kFp16:
      return Measure<Fp16Policy>(config);
    case Precision::kInt8:
      return Measure<Int8Policy>(config);
  }
  return benchmarks::Status::Error("unknown precision");
}

}  // namespace

int main(int argc, char** argv) {
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
