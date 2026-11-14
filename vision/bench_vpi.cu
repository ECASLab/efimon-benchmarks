/**
 * @file bench_vpi.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief NVIDIA VPI computer vision benchmark
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

// Streams synthetic frames through one VPI algorithm or a small pipeline on a
// selectable VPI backend:
//
//   gaussian  7x7 Gaussian filter, U8, 3840x2160 (stencil, memory bound)
//   rescale   bilinear downscale by two, U8, 3840x2160 -> 1920x1080
//   harris    Harris corner detector, U8, 1920x1080 (several kernels and a
//             keypoint compaction)
//   stereo    semi-global matching stereo disparity, U8, 1920x1080, 64
//             disparities (compute bound)
//   pipeline  BGR8 3840x2160 -> grey U8 -> 5x5 Gaussian -> downscale by two
//             -> Harris corners, the front end of a feature tracker
//
// The CUDA backend runs on every NVIDIA GPU VPI supports, including the
// Tesla V100 (sm_70) and Jetson Thor. On Jetson the PVA, VIC and OFA backends
// are separate accelerators: their work is invisible to CUPTI, so a profiler
// built on CUPTI attributes none of it to kernels.
//
// Like bench_power_cycle, the run has warm-up, measurement and cooldown phases
// delimited by empty marker kernels launched with the CUDA runtime. The input
// frames are synthetic and deterministic: rectangles on a textured background
// with known corners, and a random-texture stereo pair whose right view is
// the left view shifted by a known disparity. --mode verify checks each
// algorithm on the CUDA backend against the CPU backend or against that
// ground truth.

#include <cuda_runtime.h>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/Version.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianFilter.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
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
constexpr std::int64_t kDefaultSeed = 2024;
constexpr int kFramesBetweenSynchronizations = 8;

constexpr int kLargeWidth = 3840;
constexpr int kLargeHeight = 2160;
constexpr int kSmallWidth = 1920;
constexpr int kSmallHeight = 1080;

constexpr int kGaussianKernelSize = 7;
constexpr float kGaussianSigma = 1.7F;
constexpr int kPipelineKernelSize = 5;
constexpr float kPipelineSigma = 1.0F;
constexpr int kKeypointCapacity = 16384;
// Harris strength thresholds at which each scene's square corners are found:
// VPI's default (20) misses a third of them on the harris scene and all of
// them once the pipeline has blurred and downscaled the frame.
constexpr float kHarrisStrength = 10.0F;
constexpr float kPipelineHarrisStrength = 5.0F;
constexpr int kStereoMaxDisparity = 64;
constexpr int kStereoTrueDisparity = 20;
constexpr int kStereoFixedPointScale = 32;  // Q10.5 disparity output

// Scene: bright squares on a darker textured background.
constexpr int kSquareSize = 48;
constexpr int kSquareSpacing = 96;
constexpr int kBackgroundLevel = 40;
constexpr int kSquareLevel = 210;
constexpr int kSceneNoise = 24;
constexpr int kVerifyHarrisNoise = 4;

// Verification tolerances.
// CPU and CUDA accumulate the filter taps with different arithmetic, so a
// 49-tap Gaussian can round two steps apart on a few pixels; a real defect
// shows up as larger or widespread differences.
constexpr int kVerifyMaxPixelDifference = 2;
constexpr double kVerifyMaxAboveOneFraction = 1e-4;
constexpr double kVerifyCornerRadius = 3.0;
constexpr double kVerifyMinCornerFraction = 0.9;
// Harris needs pixels on both sides of a corner: its gradient and block
// windows (3 x 3 by default) and its non-maximum suppression (8 px) do not fit
// at the image border. Corners closer than this to the border, such as those
// of a square that ends on the last row, are left out of the Harris check.
constexpr float kVerifyBorderMargin = 8.0F;
constexpr double kVerifyMinDisparityFraction = 0.9;
constexpr int kStereoBorder = 8;

/** VPI workload streamed through the backend */
enum class Algorithm { kGaussian, kRescale, kHarris, kStereo, kPipeline };
/** Timed measurement or check against the CPU backend and ground truth */
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
  /** Seed of the synthetic frames */
  std::int64_t seed = kDefaultSeed;
  /** Workload */
  Algorithm algorithm = Algorithm::kPipeline;
  /** --backend value */
  std::string backend_name = "cuda";
  /** VPI backend flag; VPI flags are a uint64_t bit mask */
  std::uint64_t backend = VPI_BACKEND_CUDA;
  /** Frame width; 0 selects the algorithm's default */
  int width = 0;
  /** Frame height; 0 selects the algorithm's default */
  int height = 0;
  /** Harris strength threshold; negative selects the workload's default */
  float harris_strength = -1.0F;
  /** What the run does */
  Mode mode = Mode::kMeasure;
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
 * @brief Converts a VPI status into a Status with VPI's last message
 *
 * @param status Result of a VPI call
 * @param expression Text of the call, for the message
 * @return Error unless status is VPI_SUCCESS
 */
benchmarks::Status CheckVpi(VPIStatus status, const char* expression) {
  if (status == VPI_SUCCESS) return benchmarks::Status();
  char message[VPI_MAX_STATUS_MESSAGE_LENGTH];
  vpiGetLastStatusMessage(message, sizeof(message));
  return benchmarks::Status::Error(std::string(expression) + " failed: " +
                                   vpiStatusGetName(status) + ": " + message);
}

/**
 * @brief Returns from the enclosing function if a VPI call fails
 */
#define BENCHMARKS_VPI_CHECK(expression) \
  BENCHMARKS_RETURN_IF_ERROR(CheckVpi((expression), #expression))

/**
 * @brief Owns one VPI handle and destroys it with the matching VPI function
 */
template <typename Handle, void (*Destroy)(Handle)>
class VpiHandle {
 public:
  VpiHandle() = default;
  ~VpiHandle() { Reset(); }
  VpiHandle(const VpiHandle&) = delete;
  VpiHandle& operator=(const VpiHandle&) = delete;
  VpiHandle(VpiHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  VpiHandle& operator=(VpiHandle&& other) noexcept {
    if (this != &other) {
      Reset();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  /** Owned handle, or null */
  Handle get() const { return handle_; }

  /**
   * @brief Destroys the owned handle and returns where to create a new one
   */
  Handle* Out() {
    Reset();
    return &handle_;
  }

 private:
  void Reset() {
    if (handle_ != nullptr) Destroy(handle_);
    handle_ = nullptr;
  }

  Handle handle_ = nullptr;
};

using Image = VpiHandle<VPIImage, vpiImageDestroy>;
using Array = VpiHandle<VPIArray, vpiArrayDestroy>;
using Payload = VpiHandle<VPIPayload, vpiPayloadDestroy>;
using Stream = VpiHandle<VPIStream, vpiStreamDestroy>;

/**
 * @brief Host copy of a single-plane pitch-linear image
 */
template <typename T>
struct HostImage {
  /** Width in pixels */
  int width = 0;
  /** Height in pixels */
  int height = 0;
  /** Interleaved channels per pixel */
  int channels = 1;
  /** Pixels, row-major and interleaved */
  std::vector<T> pixels;

  HostImage() = default;
  HostImage(int image_width, int image_height, int image_channels)
      : width(image_width),
        height(image_height),
        channels(image_channels),
        pixels(static_cast<std::size_t>(image_width) * image_height *
               image_channels) {}

  /** Value of one channel of one pixel */
  T& At(int x, int y, int channel = 0) {
    return pixels[(static_cast<std::size_t>(y) * width + x) * channels +
                  channel];
  }
  /** Value of one channel of one pixel */
  const T& At(int x, int y, int channel = 0) const {
    return pixels[(static_cast<std::size_t>(y) * width + x) * channels +
                  channel];
  }
};

/**
 * @brief Keeps an image locked for host access while the object lives
 */
class HostLock {
 public:
  HostLock() = default;
  ~HostLock() {
    if (image_ != nullptr) vpiImageUnlock(image_);
  }
  HostLock(const HostLock&) = delete;
  HostLock& operator=(const HostLock&) = delete;

  /**
   * @brief Locks a single-plane image for host access
   *
   * @param image Image to lock
   * @param mode Read or write access
   * @return Error if the lock fails or the image has several planes
   */
  benchmarks::Status Lock(VPIImage image, VPILockMode mode) {
    BENCHMARKS_VPI_CHECK(vpiImageLockData(
        image, mode, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data_));
    image_ = image;
    if (data_.buffer.pitch.numPlanes != 1) {
      return benchmarks::Status::Error(
          "only single-plane images are supported");
    }
    return benchmarks::Status();
  }

  /** First byte of row y */
  VPIByte* Row(int y) const {
    const VPIImagePlanePitchLinear& plane = data_.buffer.pitch.planes[0];
    return plane.pBase + plane.offsetBytes +
           static_cast<std::ptrdiff_t>(y) * plane.pitchBytes;
  }

 private:
  /** Locked image, or null */
  VPIImage image_ = nullptr;
  /** Host view of the locked image */
  VPIImageData data_{};
};

/**
 * @brief Copies a host image into a VPI image of the same size
 */
template <typename T>
benchmarks::Status Upload(VPIImage image, const HostImage<T>& host) {
  HostLock lock;
  BENCHMARKS_RETURN_IF_ERROR(lock.Lock(image, VPI_LOCK_WRITE));
  const std::size_t row_bytes =
      static_cast<std::size_t>(host.width) * host.channels * sizeof(T);
  for (int y = 0; y < host.height; ++y) {
    std::memcpy(lock.Row(y), &host.At(0, y), row_bytes);
  }
  return benchmarks::Status();
}

/**
 * @brief Copies a VPI image to the host
 *
 * @param image Image to copy
 * @param channels Interleaved channels per pixel
 * @param host Output. Must not be null
 * @return Error if a VPI call fails
 */
template <typename T>
benchmarks::Status Download(VPIImage image, int channels, HostImage<T>* host) {
  int width = 0;
  int height = 0;
  BENCHMARKS_VPI_CHECK(vpiImageGetSize(image, &width, &height));
  *host = HostImage<T>(width, height, channels);
  HostLock lock;
  BENCHMARKS_RETURN_IF_ERROR(lock.Lock(image, VPI_LOCK_READ));
  const std::size_t row_bytes =
      static_cast<std::size_t>(width) * channels * sizeof(T);
  for (int y = 0; y < height; ++y) {
    std::memcpy(&host->At(0, y), lock.Row(y), row_bytes);
  }
  return benchmarks::Status();
}

/**
 * @brief Copies the keypoints of a VPI array to the host
 *
 * @param array Keypoint array
 * @param keypoints Output. Must not be null
 * @return Error if the lock fails
 */
benchmarks::Status DownloadKeypoints(VPIArray array,
                                     std::vector<VPIKeypointF32>* keypoints) {
  VPIArrayData data{};
  BENCHMARKS_VPI_CHECK(
      vpiArrayLockData(array, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &data));
  const auto* first = static_cast<const VPIKeypointF32*>(data.buffer.aos.data);
  keypoints->assign(first, first + *data.buffer.aos.sizePointer);
  vpiArrayUnlock(array);
  return benchmarks::Status();
}

/**
 * @brief Deterministic per-pixel noise in [0, 256), independent of the
 * evaluation order
 *
 * The seed and the result are hashed bit patterns, so they are unsigned.
 */
std::uint32_t HashNoise(std::uint64_t seed, int x, int y, int channel) {
  std::uint64_t value = seed ^ (static_cast<std::uint64_t>(x) << 1) ^
                        (static_cast<std::uint64_t>(y) << 21) ^
                        (static_cast<std::uint64_t>(channel) << 42);
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  value ^= value >> 31;
  return static_cast<std::uint32_t>(value & 0xFFU);
}

/**
 * @brief Returns whether a pixel lies inside one of the scene's squares
 */
bool InsideSquare(int x, int y) {
  const int offset = kSquareSpacing / 2;
  if (x < offset || y < offset) return false;
  return (x - offset) % kSquareSpacing < kSquareSize &&
         (y - offset) % kSquareSpacing < kSquareSize;
}

/**
 * @brief Draws squares on a textured background
 *
 * @param noise Amplitude of the texture
 */
HostImage<std::uint8_t> MakeScene(int width, int height, int channels,
                                  int noise, std::uint64_t seed) {
  HostImage<std::uint8_t> image(width, height, channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int base = InsideSquare(x, y) ? kSquareLevel : kBackgroundLevel;
      for (int c = 0; c < channels; ++c) {
        const int texture =
            static_cast<int>(HashNoise(seed, x, y, c) * noise / 256);
        image.At(x, y, c) =
            static_cast<std::uint8_t>(std::min(255, base + texture));
      }
    }
  }
  return image;
}

/**
 * @brief Returns the corners of the scene's squares that lie inside the image
 *
 * A square cut by the right or bottom border still has its left or top
 * corners, so those are included too.
 */
std::vector<VPIKeypointF32> SceneCorners(int width, int height) {
  std::vector<VPIKeypointF32> corners;
  const int offset = kSquareSpacing / 2;
  for (int y0 = offset; y0 < height; y0 += kSquareSpacing) {
    for (int x0 = offset; x0 < width; x0 += kSquareSpacing) {
      const int x1 = x0 + kSquareSize - 1;
      const int y1 = y0 + kSquareSize - 1;
      for (const int x : {x0, x1}) {
        for (const int y : {y0, y1}) {
          if (x < width && y < height) {
            corners.push_back(
                VPIKeypointF32{static_cast<float>(x), static_cast<float>(y)});
          }
        }
      }
    }
  }
  return corners;
}

/**
 * @brief Builds a random-texture stereo pair
 *
 * right(x, y) = left(x + disparity, y), so every left pixel with
 * x >= disparity has the true disparity.
 */
void MakeStereoPair(int width, int height, int disparity, std::uint64_t seed,
                    HostImage<std::uint8_t>* left,
                    HostImage<std::uint8_t>* right) {
  *left = HostImage<std::uint8_t>(width, height, 1);
  *right = HostImage<std::uint8_t>(width, height, 1);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      left->At(x, y) = static_cast<std::uint8_t>(HashNoise(seed, x, y, 0));
      right->At(x, y) = static_cast<std::uint8_t>(
          x + disparity < width ? HashNoise(seed, x + disparity, y, 0)
                                : HashNoise(seed ^ 1U, x, y, 0));
    }
  }
}

/**
 * @brief Returns the image flags of a backend; the CPU flag allows the host
 * upload
 */
std::uint64_t ImageFlags(std::uint64_t backend) {
  return backend | VPI_BACKEND_CPU;
}

/**
 * @brief One algorithm or pipeline with its images, arrays and payloads
 */
class Workload {
 public:
  Workload() = default;
  virtual ~Workload() = default;
  Workload(const Workload&) = delete;
  Workload& operator=(const Workload&) = delete;

  /**
   * @brief Creates the VPI objects and uploads the input frame
   */
  virtual benchmarks::Status Init() = 0;

  /**
   * @brief Queues one frame on the stream
   */
  virtual benchmarks::Status Submit(VPIStream stream) = 0;

  /** Input pixels of one frame */
  virtual double InputPixels() const = 0;

  /** --algorithm name of the workload */
  virtual const char* Name() const = 0;

  /**
   * @brief Prints the per-frame result after the measurement as KEY=VALUE
   * lines
   */
  virtual benchmarks::Status PrintResult() const {
    return benchmarks::Status();
  }
};

/**
 * @brief 7x7 Gaussian filter of a U8 frame
 */
class GaussianWorkload : public Workload {
 public:
  GaussianWorkload(std::uint64_t backend, int width, int height,
                   std::uint64_t seed)
      : backend_(backend), width_(width), height_(height), seed_(seed) {}

  benchmarks::Status Init() override {
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                                        ImageFlags(backend_), input_.Out()));
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                                        ImageFlags(backend_), output_.Out()));
    return Upload(input_.get(),
                  MakeScene(width_, height_, 1, kSceneNoise, seed_));
  }

  benchmarks::Status Submit(VPIStream stream) override {
    BENCHMARKS_VPI_CHECK(vpiSubmitGaussianFilter(
        stream, backend_, input_.get(), output_.get(), kGaussianKernelSize,
        kGaussianKernelSize, kGaussianSigma, kGaussianSigma, VPI_BORDER_CLAMP));
    return benchmarks::Status();
  }
  double InputPixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* Name() const override { return "gaussian"; }
  /** Filtered frame */
  VPIImage output() const { return output_.get(); }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  std::uint64_t seed_;
  Image input_;
  Image output_;
};

/**
 * @brief Bilinear downscale of a U8 frame by two
 */
class RescaleWorkload : public Workload {
 public:
  RescaleWorkload(std::uint64_t backend, int width, int height,
                  std::uint64_t seed)
      : backend_(backend), width_(width), height_(height), seed_(seed) {}

  benchmarks::Status Init() override {
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                                        ImageFlags(backend_), input_.Out()));
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_ / 2, height_ / 2,
                                        VPI_IMAGE_FORMAT_U8,
                                        ImageFlags(backend_), output_.Out()));
    return Upload(input_.get(),
                  MakeScene(width_, height_, 1, kSceneNoise, seed_));
  }

  benchmarks::Status Submit(VPIStream stream) override {
    BENCHMARKS_VPI_CHECK(vpiSubmitRescale(stream, backend_, input_.get(),
                                          output_.get(), VPI_INTERP_LINEAR,
                                          VPI_BORDER_CLAMP, 0));
    return benchmarks::Status();
  }
  double InputPixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* Name() const override { return "rescale"; }
  /** Downscaled frame */
  VPIImage output() const { return output_.get(); }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  std::uint64_t seed_;
  Image input_;
  Image output_;
};

/**
 * @brief Harris corner detector on a U8 frame
 */
class HarrisWorkload : public Workload {
 public:
  HarrisWorkload(std::uint64_t backend, int width, int height, int noise,
                 float strength, std::uint64_t seed)
      : backend_(backend),
        width_(width),
        height_(height),
        noise_(noise),
        strength_(strength),
        seed_(seed) {}

  benchmarks::Status Init() override {
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                                        ImageFlags(backend_), input_.Out()));
    BENCHMARKS_VPI_CHECK(
        vpiArrayCreate(kKeypointCapacity, VPI_ARRAY_TYPE_KEYPOINT_F32,
                       ImageFlags(backend_), keypoints_.Out()));
    BENCHMARKS_VPI_CHECK(vpiArrayCreate(kKeypointCapacity, VPI_ARRAY_TYPE_U32,
                                        ImageFlags(backend_), scores_.Out()));
    BENCHMARKS_VPI_CHECK(vpiCreateHarrisCornerDetector(
        backend_, width_, height_, payload_.Out()));
    BENCHMARKS_VPI_CHECK(vpiInitHarrisCornerDetectorParams(&params_));
    if (strength_ >= 0.0F) params_.strengthThresh = strength_;
    return Upload(input_.get(), MakeScene(width_, height_, 1, noise_, seed_));
  }

  benchmarks::Status Submit(VPIStream stream) override {
    BENCHMARKS_VPI_CHECK(vpiSubmitHarrisCornerDetector(
        stream, backend_, payload_.get(), input_.get(), keypoints_.get(),
        scores_.get(), &params_));
    return benchmarks::Status();
  }
  double InputPixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* Name() const override { return "harris"; }
  benchmarks::Status PrintResult() const override {
    std::vector<VPIKeypointF32> keypoints;
    BENCHMARKS_RETURN_IF_ERROR(Keypoints(&keypoints));
    std::cout << "KEYPOINTS_LAST_FRAME=" << keypoints.size() << '\n';
    return benchmarks::Status();
  }

  /**
   * @brief Copies the keypoints of the last frame to the host
   */
  benchmarks::Status Keypoints(std::vector<VPIKeypointF32>* keypoints) const {
    return DownloadKeypoints(keypoints_.get(), keypoints);
  }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  int noise_;
  float strength_;
  std::uint64_t seed_;
  Image input_;
  Array keypoints_;
  Array scores_;
  Payload payload_;
  VPIHarrisCornerDetectorParams params_{};
};

/**
 * @brief Semi-global matching stereo disparity of a U8 pair
 */
class StereoWorkload : public Workload {
 public:
  StereoWorkload(std::uint64_t backend, int width, int height,
                 std::uint64_t seed)
      : backend_(backend), width_(width), height_(height), seed_(seed) {}

  benchmarks::Status Init() override {
    for (Image* image : {&left_, &right_}) {
      BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                                          ImageFlags(backend_), image->Out()));
    }
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_S16,
                                        ImageFlags(backend_),
                                        disparity_.Out()));
    VPIStereoDisparityEstimatorCreationParams creation{};
    BENCHMARKS_VPI_CHECK(
        vpiInitStereoDisparityEstimatorCreationParams(&creation));
    creation.maxDisparity = kStereoMaxDisparity;
    BENCHMARKS_VPI_CHECK(vpiCreateStereoDisparityEstimator(
        backend_, width_, height_, VPI_IMAGE_FORMAT_U8, &creation,
        payload_.Out()));
    BENCHMARKS_VPI_CHECK(vpiInitStereoDisparityEstimatorParams(&params_));
    params_.maxDisparity = kStereoMaxDisparity;
    HostImage<std::uint8_t> left;
    HostImage<std::uint8_t> right;
    MakeStereoPair(width_, height_, kStereoTrueDisparity, seed_, &left, &right);
    BENCHMARKS_RETURN_IF_ERROR(Upload(left_.get(), left));
    return Upload(right_.get(), right);
  }

  benchmarks::Status Submit(VPIStream stream) override {
    BENCHMARKS_VPI_CHECK(vpiSubmitStereoDisparityEstimator(
        stream, backend_, payload_.get(), left_.get(), right_.get(),
        disparity_.get(), nullptr, &params_));
    return benchmarks::Status();
  }
  double InputPixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* Name() const override { return "stereo"; }
  benchmarks::Status PrintResult() const override {
    double accuracy = 0.0;
    BENCHMARKS_RETURN_IF_ERROR(Accuracy(&accuracy));
    std::cout << std::fixed << std::setprecision(3)
              << "DISPARITY_WITHIN_1PX_PCT=" << 100.0 * accuracy << '\n'
              << "TRUE_DISPARITY=" << kStereoTrueDisparity << '\n';
    return benchmarks::Status();
  }

  /**
   * @brief Fraction of interior pixels whose disparity is within one pixel
   * of the true disparity
   *
   * Pixels with x < max disparity have no full search range.
   */
  benchmarks::Status Accuracy(double* fraction) const {
    HostImage<std::int16_t> disparity;
    BENCHMARKS_RETURN_IF_ERROR(Download(disparity_.get(), 1, &disparity));
    std::int64_t good = 0;
    std::int64_t total = 0;
    for (int y = kStereoBorder; y < height_ - kStereoBorder; ++y) {
      for (int x = kStereoMaxDisparity;
           x < width_ - kStereoTrueDisparity - kStereoBorder; ++x) {
        const double value =
            static_cast<double>(disparity.At(x, y)) / kStereoFixedPointScale;
        if (std::abs(value - kStereoTrueDisparity) <= 1.0) ++good;
        ++total;
      }
    }
    *fraction = total == 0 ? 0.0 : static_cast<double>(good) / total;
    return benchmarks::Status();
  }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  std::uint64_t seed_;
  Image left_;
  Image right_;
  Image disparity_;
  Payload payload_;
  VPIStereoDisparityEstimatorParams params_{};
};

/**
 * @brief BGR8 frame to grey, 5x5 Gaussian, downscale by two and Harris
 * corners
 */
class PipelineWorkload : public Workload {
 public:
  PipelineWorkload(std::uint64_t backend, int width, int height, float strength,
                   std::uint64_t seed)
      : backend_(backend),
        width_(width),
        height_(height),
        strength_(strength),
        seed_(seed) {}

  benchmarks::Status Init() override {
    const std::uint64_t flags = ImageFlags(backend_);
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_BGR8,
                                        flags, colour_.Out()));
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                                        flags, grey_.Out()));
    BENCHMARKS_VPI_CHECK(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                                        flags, blurred_.Out()));
    BENCHMARKS_VPI_CHECK(vpiImageCreate(
        width_ / 2, height_ / 2, VPI_IMAGE_FORMAT_U8, flags, small_.Out()));
    BENCHMARKS_VPI_CHECK(vpiArrayCreate(kKeypointCapacity,
                                        VPI_ARRAY_TYPE_KEYPOINT_F32, flags,
                                        keypoints_.Out()));
    BENCHMARKS_VPI_CHECK(vpiArrayCreate(kKeypointCapacity, VPI_ARRAY_TYPE_U32,
                                        flags, scores_.Out()));
    BENCHMARKS_VPI_CHECK(vpiCreateHarrisCornerDetector(
        backend_, width_ / 2, height_ / 2, harris_.Out()));
    BENCHMARKS_VPI_CHECK(vpiInitHarrisCornerDetectorParams(&params_));
    if (strength_ >= 0.0F) params_.strengthThresh = strength_;
    return Upload(colour_.get(),
                  MakeScene(width_, height_, 3, kSceneNoise, seed_));
  }

  benchmarks::Status Submit(VPIStream stream) override {
    BENCHMARKS_VPI_CHECK(vpiSubmitConvertImageFormat(
        stream, backend_, colour_.get(), grey_.get(), nullptr));
    BENCHMARKS_VPI_CHECK(vpiSubmitGaussianFilter(
        stream, backend_, grey_.get(), blurred_.get(), kPipelineKernelSize,
        kPipelineKernelSize, kPipelineSigma, kPipelineSigma, VPI_BORDER_CLAMP));
    BENCHMARKS_VPI_CHECK(vpiSubmitRescale(stream, backend_, blurred_.get(),
                                          small_.get(), VPI_INTERP_LINEAR,
                                          VPI_BORDER_CLAMP, 0));
    BENCHMARKS_VPI_CHECK(vpiSubmitHarrisCornerDetector(
        stream, backend_, harris_.get(), small_.get(), keypoints_.get(),
        scores_.get(), &params_));
    return benchmarks::Status();
  }
  double InputPixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* Name() const override { return "pipeline"; }
  benchmarks::Status PrintResult() const override {
    std::vector<VPIKeypointF32> keypoints;
    BENCHMARKS_RETURN_IF_ERROR(DownloadKeypoints(keypoints_.get(), &keypoints));
    std::cout << "KEYPOINTS_LAST_FRAME=" << keypoints.size() << '\n';
    return benchmarks::Status();
  }
  /** Grey frame, after the format conversion */
  VPIImage grey() const { return grey_.get(); }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  float strength_;
  std::uint64_t seed_;
  Image colour_;
  Image grey_;
  Image blurred_;
  Image small_;
  Array keypoints_;
  Array scores_;
  Payload harris_;
  VPIHarrisCornerDetectorParams params_{};
};

/**
 * @brief Returns the --algorithm name of an algorithm
 */
const char* AlgorithmName(Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::kGaussian:
      return "gaussian";
    case Algorithm::kRescale:
      return "rescale";
    case Algorithm::kHarris:
      return "harris";
    case Algorithm::kStereo:
      return "stereo";
    case Algorithm::kPipeline:
      return "pipeline";
  }
  return "unknown";
}

/**
 * @brief Returns whether the algorithm defaults to 3840x2160 frames
 */
bool UsesLargeFrames(Algorithm algorithm) {
  return algorithm == Algorithm::kGaussian ||
         algorithm == Algorithm::kRescale || algorithm == Algorithm::kPipeline;
}

/**
 * @brief Returns the frame width, from --width or the algorithm's default
 */
int FrameWidth(const Config& config) {
  if (config.width > 0) return config.width;
  return UsesLargeFrames(config.algorithm) ? kLargeWidth : kSmallWidth;
}

/**
 * @brief Returns the frame height, from --height or the algorithm's default
 */
int FrameHeight(const Config& config) {
  if (config.height > 0) return config.height;
  return UsesLargeFrames(config.algorithm) ? kLargeHeight : kSmallHeight;
}

/**
 * @brief Creates and initializes the workload of the configuration
 *
 * @param config Benchmark options
 * @param workload Output. Must not be null
 * @return Error if a VPI call fails
 */
benchmarks::Status MakeWorkload(const Config& config,
                                std::unique_ptr<Workload>* workload) {
  const int width = FrameWidth(config);
  const int height = FrameHeight(config);
  const auto seed = static_cast<std::uint64_t>(config.seed);
  const auto strength = [&config](float workload_default) {
    return config.harris_strength >= 0.0F ? config.harris_strength
                                          : workload_default;
  };
  switch (config.algorithm) {
    case Algorithm::kGaussian:
      *workload = std::make_unique<GaussianWorkload>(config.backend, width,
                                                     height, seed);
      break;
    case Algorithm::kRescale:
      *workload = std::make_unique<RescaleWorkload>(config.backend, width,
                                                    height, seed);
      break;
    case Algorithm::kHarris:
      *workload = std::make_unique<HarrisWorkload>(
          config.backend, width, height, kSceneNoise, strength(kHarrisStrength),
          seed);
      break;
    case Algorithm::kStereo:
      *workload =
          std::make_unique<StereoWorkload>(config.backend, width, height, seed);
      break;
    case Algorithm::kPipeline:
      *workload = std::make_unique<PipelineWorkload>(
          config.backend, width, height, strength(kPipelineHarrisStrength),
          seed);
      break;
  }
  if (*workload == nullptr) {
    return benchmarks::Status::Error("unknown algorithm");
  }
  return (*workload)->Init();
}

/**
 * @brief Prints the usage and the options of the benchmark
 *
 * @param program Name of the executable, from argv[0]
 */
void PrintHelp(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "NVIDIA VPI computer-vision algorithms on synthetic frames.\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --algorithm <name>        gaussian, rescale, harris, stereo or\n"
      << "                            pipeline (default: pipeline)\n"
      << "  --backend <name>          cuda, cpu, pva, vic or ofa (default: "
         "cuda)\n"
      << "  --mode <name>             measure or verify (default: measure)\n"
      << "  --width <N>               Frame width (default: 3840 for "
         "gaussian,\n"
      << "                            rescale and pipeline, otherwise 1920)\n"
      << "  --height <N>              Frame height (default: 2160 or 1080)\n"
      << "  --harris-strength <val>   Harris strength threshold (default: 10\n"
      << "                            for harris, 5 for pipeline)\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 10)\n"
      << "  --duration <seconds>      Measurement phase (default: 40)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 10)\n"
      << "  --seed <N>                Synthetic frame seed (default: 2024)\n"
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
      {"--device", "--algorithm", "--backend", "--mode", "--width", "--height",
       "--harris-strength", "--warmup-seconds", "--duration",
       "--cooldown-seconds", "--seed"}));
  *config = Config();
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--device", config->device, &config->device));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--width", config->width, &config->width));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--height", config->height, &config->height));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--warmup-seconds", config->warmup_seconds, &config->warmup_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--duration", config->measurement_seconds, &config->measurement_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue("--cooldown-seconds",
                                                   config->cooldown_seconds,
                                                   &config->cooldown_seconds));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.Int64Value("--seed", config->seed, &config->seed));
  double harris_strength = 0.0;
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--harris-strength", config->harris_strength, &harris_strength));
  config->harris_strength = static_cast<float>(harris_strength);

  const std::string algorithm =
      arguments.StringValue("--algorithm", "pipeline");
  bool known = false;
  for (Algorithm candidate :
       {Algorithm::kGaussian, Algorithm::kRescale, Algorithm::kHarris,
        Algorithm::kStereo, Algorithm::kPipeline}) {
    if (algorithm == AlgorithmName(candidate)) {
      config->algorithm = candidate;
      known = true;
    }
  }
  if (!known) {
    return benchmarks::Status::Error(
        "--algorithm must be gaussian, rescale, harris, stereo or pipeline");
  }

  config->backend_name = arguments.StringValue("--backend", "cuda");
  if (config->backend_name == "cuda") {
    config->backend = VPI_BACKEND_CUDA;
  } else if (config->backend_name == "cpu") {
    config->backend = VPI_BACKEND_CPU;
  } else if (config->backend_name == "pva") {
    config->backend = VPI_BACKEND_PVA;
  } else if (config->backend_name == "vic") {
    config->backend = VPI_BACKEND_VIC;
  } else if (config->backend_name == "ofa") {
    config->backend = VPI_BACKEND_OFA;
  } else {
    return benchmarks::Status::Error(
        "--backend must be cuda, cpu, pva, vic or ofa");
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
  if (config->width < 0 || config->height < 0 || config->width % 2 != 0 ||
      config->height % 2 != 0) {
    return benchmarks::Status::Error(
        "--width and --height must be even and positive");
  }
  if (config->algorithm == Algorithm::kStereo &&
      FrameWidth(*config) <=
          kStereoMaxDisparity + kStereoTrueDisparity + kStereoBorder) {
    return benchmarks::Status::Error("stereo frames are too narrow");
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
 * @brief Prints the configuration, the VPI version and the device as
 * KEY=VALUE lines
 */
void PrintConfig(const Config& config, const cudaDeviceProp& properties) {
  const std::int32_t version = vpiGetVersion();
  std::cout << std::setprecision(12) << "BENCHMARK=vpi\n"
            << "ALGORITHM=" << AlgorithmName(config.algorithm) << '\n'
            << "BACKEND=" << config.backend_name << '\n'
            << "MODE=" << (config.mode == Mode::kMeasure ? "measure" : "verify")
            << '\n'
            << "VPI_VERSION=" << version / 1000000 << '.'
            << version / 10000 % 100 << '.' << version / 100 % 100 << '\n'
            << "DEVICE=" << config.device << '\n'
            << "DEVICE_NAME=" << properties.name << '\n'
            << "COMPUTE_CAPABILITY=" << properties.major << '.'
            << properties.minor << '\n'
            << "FRAME_WIDTH=" << FrameWidth(config) << '\n'
            << "FRAME_HEIGHT=" << FrameHeight(config) << '\n'
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
  /** Frames processed in the phase */
  std::int64_t frames = 0;
};

/**
 * @brief Submits frames until the phase time runs out
 *
 * @param workload Workload to run. Must not be null
 * @param stream VPI stream
 * @param seconds Duration of the phase
 * @param stats Output. Must not be null
 * @return Error if a VPI call fails
 */
benchmarks::Status RunPhase(Workload* workload, VPIStream stream,
                            double seconds, PhaseStats* stats) {
  *stats = PhaseStats();
  const auto start = Clock::now();
  const auto deadline = start + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(seconds));
  while (Clock::now() < deadline) {
    for (int i = 0; i < kFramesBetweenSynchronizations; ++i) {
      BENCHMARKS_RETURN_IF_ERROR(workload->Submit(stream));
    }
    BENCHMARKS_VPI_CHECK(vpiStreamSync(stream));
    stats->frames += kFramesBetweenSynchronizations;
  }
  stats->elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  return benchmarks::Status();
}

/**
 * @brief Builds the workload, then runs the warm-up, measurement and
 * cooldown phases
 *
 * @param config Benchmark options
 * @return Error if a VPI or CUDA call fails
 */
benchmarks::Status Measure(const Config& config) {
  Stream stream;
  BENCHMARKS_VPI_CHECK(vpiStreamCreate(config.backend, stream.Out()));
  std::unique_ptr<Workload> workload;
  BENCHMARKS_RETURN_IF_ERROR(MakeWorkload(config, &workload));
  // The first frame pays for lazy allocations and JIT compilation.
  BENCHMARKS_RETURN_IF_ERROR(workload->Submit(stream.get()));
  BENCHMARKS_VPI_CHECK(vpiStreamSync(stream.get()));

  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("warmup", PhaseMarker::kWarmup));
  PhaseStats warmup;
  BENCHMARKS_RETURN_IF_ERROR(
      RunPhase(workload.get(), stream.get(), config.warmup_seconds, &warmup));
  BENCHMARKS_RETURN_IF_ERROR(
      EmitPhase("measurement", PhaseMarker::kMeasurement));
  PhaseStats measurement;
  BENCHMARKS_RETURN_IF_ERROR(RunPhase(
      workload.get(), stream.get(), config.measurement_seconds, &measurement));
  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("cooldown", PhaseMarker::kCooldown));
  const auto cooldown_start = Clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.cooldown_seconds));
  const double cooldown_elapsed =
      std::chrono::duration<double>(Clock::now() - cooldown_start).count();
  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("complete", PhaseMarker::kComplete));

  const double frames = static_cast<double>(measurement.frames);
  std::cout << std::fixed << std::setprecision(6)
            << "WARMUP_ELAPSED_SECONDS=" << warmup.elapsed_seconds << '\n'
            << "WARMUP_FRAMES=" << warmup.frames << '\n'
            << "MEASUREMENT_ELAPSED_SECONDS=" << measurement.elapsed_seconds
            << '\n'
            << "MEASUREMENT_FRAMES=" << measurement.frames << '\n'
            << "FRAMES_PER_SECOND=" << frames / measurement.elapsed_seconds
            << '\n'
            << "MS_PER_FRAME=" << 1000.0 * measurement.elapsed_seconds / frames
            << '\n'
            << "MEGAPIXELS_PER_SECOND="
            << frames * workload->InputPixels() / measurement.elapsed_seconds /
                   1e6
            << '\n';
  BENCHMARKS_RETURN_IF_ERROR(workload->PrintResult());
  std::cout << std::fixed << std::setprecision(6)
            << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed << '\n';
  benchmarks::PrintExecutionSummary(measurement.elapsed_seconds,
                                    measurement.frames);
  return benchmarks::Status();
}

/**
 * @brief Initializes a workload and runs one frame on its own stream
 */
benchmarks::Status InitAndRunOnce(Workload* workload) {
  BENCHMARKS_RETURN_IF_ERROR(workload->Init());
  Stream stream;
  BENCHMARKS_VPI_CHECK(vpiStreamCreate(0, stream.Out()));
  BENCHMARKS_RETURN_IF_ERROR(workload->Submit(stream.get()));
  BENCHMARKS_VPI_CHECK(vpiStreamSync(stream.get()));
  return benchmarks::Status();
}

/**
 * @brief Pixel differences between two U8 images
 */
struct PixelDifference {
  /** Largest absolute difference */
  int max_abs = 0;
  /** Mean absolute difference */
  double mean_abs = 0.0;
  /** Fraction of pixels that differ by more than 1 */
  double above_one_fraction = 0.0;
};

/**
 * @brief Compares two U8 images of the same size
 *
 * @param difference Output. Must not be null
 * @return Error if the sizes differ or the images are empty
 */
benchmarks::Status CompareImages(const HostImage<std::uint8_t>& a,
                                 const HostImage<std::uint8_t>& b,
                                 PixelDifference* difference) {
  if (a.pixels.size() != b.pixels.size() || a.pixels.empty()) {
    return benchmarks::Status::Error("verification images differ in size");
  }
  *difference = PixelDifference();
  std::int64_t sum = 0;
  std::int64_t above_one = 0;
  for (std::size_t i = 0; i < a.pixels.size(); ++i) {
    const int pixel_difference =
        std::abs(static_cast<int>(a.pixels[i]) - static_cast<int>(b.pixels[i]));
    difference->max_abs = std::max(difference->max_abs, pixel_difference);
    sum += pixel_difference;
    if (pixel_difference > 1) ++above_one;
  }
  difference->mean_abs = static_cast<double>(sum) / a.pixels.size();
  difference->above_one_fraction =
      static_cast<double>(above_one) / a.pixels.size();
  return benchmarks::Status();
}

/**
 * @brief Keeps the points at least kVerifyBorderMargin away from the border
 */
std::vector<VPIKeypointF32> AwayFromBorder(
    const std::vector<VPIKeypointF32>& points, int width, int height) {
  std::vector<VPIKeypointF32> kept;
  for (const VPIKeypointF32& point : points) {
    if (point.x >= kVerifyBorderMargin && point.y >= kVerifyBorderMargin &&
        point.x <= width - 1 - kVerifyBorderMargin &&
        point.y <= height - 1 - kVerifyBorderMargin) {
      kept.push_back(point);
    }
  }
  return kept;
}

/**
 * @brief Fraction of the points in from that have a point of to within
 * radius
 */
double MatchedFraction(const std::vector<VPIKeypointF32>& from,
                       const std::vector<VPIKeypointF32>& to, double radius) {
  if (from.empty()) return 0.0;
  std::size_t matched = 0;
  for (const VPIKeypointF32& a : from) {
    for (const VPIKeypointF32& b : to) {
      if (std::hypot(a.x - b.x, a.y - b.y) <= radius) {
        ++matched;
        break;
      }
    }
  }
  return static_cast<double>(matched) / from.size();
}

/**
 * @brief Prints each verification check and remembers whether all passed
 */
class VerifyReport {
 public:
  VerifyReport() = default;
  VerifyReport(const VerifyReport&) = delete;
  VerifyReport& operator=(const VerifyReport&) = delete;

  /**
   * @brief Checks a value against a lower or an upper limit
   *
   * @param name Suffix of the VERIFY_ keys
   * @param value Measured value
   * @param limit Allowed limit
   * @param at_least Whether limit is a minimum instead of a maximum
   */
  void Check(const std::string& name, double value, double limit,
             bool at_least) {
    const bool ok =
        std::isfinite(value) && (at_least ? value >= limit : value <= limit);
    passed_ = passed_ && ok;
    std::cout << std::setprecision(6) << "VERIFY_" << name << '=' << value
              << '\n'
              << "VERIFY_" << name << (at_least ? "_MIN=" : "_MAX=") << limit
              << '\n'
              << "VERIFY_" << name << "_STATUS=" << (ok ? "OK" : "FAILED")
              << '\n';
  }

  /**
   * @brief Checks the pixel differences between the CPU and CUDA outputs
   *
   * @return Error if a download fails or the sizes differ
   */
  benchmarks::Status CheckPixels(const std::string& name, VPIImage cpu,
                                 VPIImage cuda) {
    HostImage<std::uint8_t> cpu_pixels;
    BENCHMARKS_RETURN_IF_ERROR(Download(cpu, 1, &cpu_pixels));
    HostImage<std::uint8_t> cuda_pixels;
    BENCHMARKS_RETURN_IF_ERROR(Download(cuda, 1, &cuda_pixels));
    PixelDifference difference;
    BENCHMARKS_RETURN_IF_ERROR(
        CompareImages(cpu_pixels, cuda_pixels, &difference));
    std::cout << std::setprecision(6) << "VERIFY_" << name
              << "_MEAN_ABS_DIFF_VS_CPU=" << difference.mean_abs << '\n';
    Check(name + "_MAX_ABS_DIFF_VS_CPU", difference.max_abs,
          kVerifyMaxPixelDifference, false);
    Check(name + "_ABOVE_1_FRACTION_VS_CPU", difference.above_one_fraction,
          kVerifyMaxAboveOneFraction, false);
    return benchmarks::Status();
  }

  /** Whether every check passed */
  bool passed() const { return passed_; }

 private:
  bool passed_ = true;
};

/**
 * @brief Checks every algorithm on the CUDA backend
 *
 * Gaussian, rescale and grey conversion are compared with the CPU backend;
 * Harris and stereo with the ground truth of the synthetic scene.
 *
 * @param config Benchmark options
 * @return Error if a check fails or a VPI call fails
 */
benchmarks::Status Verify(const Config& config) {
  const auto seed = static_cast<std::uint64_t>(config.seed);
  VerifyReport report;
  const int large_width = config.width > 0 ? config.width : kLargeWidth;
  const int large_height = config.height > 0 ? config.height : kLargeHeight;
  const int small_width = config.width > 0 ? config.width : kSmallWidth;
  const int small_height = config.height > 0 ? config.height : kSmallHeight;

  {
    GaussianWorkload cpu(VPI_BACKEND_CPU, large_width, large_height, seed);
    GaussianWorkload cuda(VPI_BACKEND_CUDA, large_width, large_height, seed);
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cpu));
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cuda));
    BENCHMARKS_RETURN_IF_ERROR(
        report.CheckPixels("GAUSSIAN", cpu.output(), cuda.output()));
  }
  {
    RescaleWorkload cpu(VPI_BACKEND_CPU, large_width, large_height, seed);
    RescaleWorkload cuda(VPI_BACKEND_CUDA, large_width, large_height, seed);
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cpu));
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cuda));
    BENCHMARKS_RETURN_IF_ERROR(
        report.CheckPixels("RESCALE", cpu.output(), cuda.output()));
  }
  {
    PipelineWorkload cpu(VPI_BACKEND_CPU, large_width, large_height, -1.0F,
                         seed);
    PipelineWorkload cuda(VPI_BACKEND_CUDA, large_width, large_height, -1.0F,
                          seed);
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cpu));
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cuda));
    BENCHMARKS_RETURN_IF_ERROR(
        report.CheckPixels("BGR_TO_GREY", cpu.grey(), cuda.grey()));
  }
  {
    // Low texture so that the only corners are the squares' corners.
    HarrisWorkload cuda(VPI_BACKEND_CUDA, small_width, small_height,
                        kVerifyHarrisNoise, -1.0F, seed);
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cuda));
    std::vector<VPIKeypointF32> keypoints;
    BENCHMARKS_RETURN_IF_ERROR(cuda.Keypoints(&keypoints));
    const std::vector<VPIKeypointF32> detected =
        AwayFromBorder(keypoints, small_width, small_height);
    const std::vector<VPIKeypointF32> truth = AwayFromBorder(
        SceneCorners(small_width, small_height), small_width, small_height);
    std::cout << "VERIFY_HARRIS_KEYPOINTS=" << detected.size() << '\n'
              << "VERIFY_HARRIS_TRUE_CORNERS=" << truth.size() << '\n';
    report.Check("HARRIS_PRECISION",
                 MatchedFraction(detected, truth, kVerifyCornerRadius),
                 kVerifyMinCornerFraction, true);
    report.Check("HARRIS_RECALL",
                 MatchedFraction(truth, detected, kVerifyCornerRadius),
                 kVerifyMinCornerFraction, true);
  }
  {
    StereoWorkload cuda(VPI_BACKEND_CUDA, small_width, small_height, seed);
    BENCHMARKS_RETURN_IF_ERROR(InitAndRunOnce(&cuda));
    double accuracy = 0.0;
    BENCHMARKS_RETURN_IF_ERROR(cuda.Accuracy(&accuracy));
    report.Check("STEREO_WITHIN_1PX_FRACTION", accuracy,
                 kVerifyMinDisparityFraction, true);
  }

  std::cout << "VERIFY=" << (report.passed() ? "OK" : "FAILED") << '\n';
  if (!report.passed()) {
    return benchmarks::Status::Error("a verification check failed");
  }
  benchmarks::PrintExecutionSummary(0.0, 5);
  return benchmarks::Status();
}

/**
 * @brief Parses the options, prints the configuration and runs the selected
 * mode
 *
 * @param argc Number of arguments, as received by main()
 * @param argv Arguments, as received by main()
 * @return Error if an option is invalid, a VPI or CUDA call fails or a check
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
  return config.mode == Mode::kVerify ? Verify(config) : Measure(config);
}

}  // namespace

int main(int argc, char** argv) {
  // On Jetson Thor (L4T driver 595), libEGL_nvidia, which libnvvpi loads,
  // initializes its external window-system platforms in its exit-time
  // destructor and aborts with a double free when the shared CUDA runtime is
  // in use. The benchmark draws nothing, so it disables those platforms. A
  // value already set by the user is kept.
  setenv("__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS", "", 0);
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
