#if ENABLE_ROCM

#include "stream_pool.hpp"
#include "../../logger/logger.hpp"

#include <algorithm>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

StreamPool::StreamPool() = default;

StreamPool::~StreamPool() {
  Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

StreamPool::StreamPool(StreamPool&& other) noexcept
    : streams_(std::move(other.streams_)),
      device_index_(other.device_index_),
      initialized_(other.initialized_) {
  other.initialized_ = false;
}

StreamPool& StreamPool::operator=(StreamPool&& other) noexcept {
  if (this != &other) {
    Cleanup();
    streams_ = std::move(other.streams_);
    device_index_ = other.device_index_;
    initialized_ = other.initialized_;
    other.initialized_ = false;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация
// ════════════════════════════════════════════════════════════════════════════

bool StreamPool::Initialize(int count, int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    Cleanup();
  }

  device_index_ = device_index;

  // hipSetDevice — привязка текущего потока к конкретному GPU
  hipError_t err = hipSetDevice(device_index);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index, "StreamPool",
        "hipSetDevice failed: " + std::string(hipGetErrorString(err)));
    return false;
  }

  if (count <= 0) {
    count = 2;  // По умолчанию 2 stream'а
  }

  for (int i = 0; i < count; ++i) {
    hipStream_t stream = nullptr;
    // hipStreamCreateWithFlags: non-blocking позволяет параллельное
    // выполнение с default stream (не блокирует другие потоки)
    err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
      DRVGPU_LOG_ERROR_GPU(device_index, "StreamPool",
          "hipStreamCreate failed for stream " + std::to_string(i) +
          ": " + std::string(hipGetErrorString(err)));
      continue;
    }
    streams_.push_back(stream);
  }

  initialized_ = !streams_.empty();

  if (initialized_) {
    DRVGPU_LOG_INFO_GPU(device_index, "StreamPool",
        "Initialized with " + std::to_string(streams_.size()) + " streams");
  }

  return initialized_;
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка
// ════════════════════════════════════════════════════════════════════════════

void StreamPool::Cleanup() {
  // NB: mutex НЕ захватываем здесь — вызывается из деструктора.
  // Initialize() захватывает mutex перед вызовом Cleanup().
  for (auto& stream : streams_) {
    if (stream) {
      hipStreamDestroy(stream);
    }
  }
  streams_.clear();
  initialized_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// Доступ к streams
// ════════════════════════════════════════════════════════════════════════════

hipStream_t StreamPool::GetStream(size_t index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (streams_.empty()) {
    return nullptr;
  }
  return streams_[index % streams_.size()];
}

size_t StreamPool::GetStreamCount() const {
  return streams_.size();
}

// ════════════════════════════════════════════════════════════════════════════
// Синхронизация
// ════════════════════════════════════════════════════════════════════════════

void StreamPool::SynchronizeAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& stream : streams_) {
    if (stream) {
      hipStreamSynchronize(stream);
    }
  }
}

} // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
