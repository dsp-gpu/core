#pragma once

/**
 * @file buffer_set.hpp
 * @brief BufferSet<N> — compile-time fixed array of GPU buffers with lazy alloc + reuse
 *
 * Part of Ref03 Unified Architecture (Layer 4).
 *
 * Usage:
 *   class MyOp : public GpuKernelOp {
 *     enum Buf { kInput, kOutput, kTemp, kCount };
 *     BufferSet<kCount> bufs_;
 *
 *     void Execute() {
 *       auto* in  = bufs_.Require(kInput,  n * sizeof(float), stream_);
 *       auto* out = bufs_.Require(kOutput, n * sizeof(float), stream_);
 *       // ... launch kernels ...
 *     }
 *   };
 *
 * Properties:
 * - Zero heap allocation (stack array of N entries)
 * - Lazy alloc: buffer created on first Require(), reused if big enough
 * - Trivial move semantics (memcpy + zero source)
 * - RAII: destructor frees all GPU memory
 * - Thread-safe by design: per-instance, no shared state
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#if ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace drv_gpu_lib {

/**
 * @brief Single GPU buffer entry (pointer + allocated size)
 */
struct GpuBufferEntry {
  void* ptr = nullptr;
  size_t size = 0;
};

/**
 * @brief Compile-time fixed array of GPU buffers
 * @tparam N Number of buffer slots (use enum { ..., kCount } for type-safe indexing)
 */
template<size_t N>
class BufferSet {
public:
  BufferSet() = default;
  ~BufferSet() { ReleaseAll(); }

  // Move: transfer ownership
  BufferSet(BufferSet&& other) noexcept {
    std::memcpy(entries_, other.entries_, sizeof(entries_));
    std::memset(other.entries_, 0, sizeof(other.entries_));
  }

  BufferSet& operator=(BufferSet&& other) noexcept {
    if (this != &other) {
      ReleaseAll();
      std::memcpy(entries_, other.entries_, sizeof(entries_));
      std::memset(other.entries_, 0, sizeof(other.entries_));
    }
    return *this;
  }

  // No copy
  BufferSet(const BufferSet&) = delete;
  BufferSet& operator=(const BufferSet&) = delete;

  /**
   * @brief Get or allocate buffer at index idx
   * @param idx Buffer index (use enum value)
   * @param bytes Required size in bytes
   * @return Device pointer (hipDeviceptr_t)
   *
   * If buffer exists and is large enough — reuse (no allocation).
   * If buffer is too small or doesn't exist — free old, allocate new.
   */
  void* Require(size_t idx, size_t bytes) {
    static_assert(N > 0, "BufferSet must have at least 1 slot");
    if (idx >= N) {
      throw std::out_of_range("BufferSet::Require: idx=" + std::to_string(idx) +
                               " >= N=" + std::to_string(N));
    }

    auto& e = entries_[idx];
    if (e.size >= bytes && e.ptr != nullptr) {
      return e.ptr;  // reuse existing buffer
    }

    // Free old buffer if exists
    FreeEntry(e);

    // Allocate new
    if (bytes > 0) {
#if ENABLE_ROCM
      hipError_t err = hipMalloc(&e.ptr, bytes);
      if (err != hipSuccess) {
        throw std::runtime_error("BufferSet::Require: hipMalloc(" +
                                  std::to_string(bytes) + " bytes) failed: " +
                                  hipGetErrorString(err));
      }
      e.size = bytes;  // размер запоминаем только если выделение успешно
#endif
    }
    return e.ptr;
  }

  /**
   * @brief Get existing buffer pointer (no allocation)
   * @return nullptr if not allocated
   */
  void* Get(size_t idx) const {
    return (idx < N) ? entries_[idx].ptr : nullptr;
  }

  /**
   * @brief Get allocated size of buffer at idx
   */
  size_t Size(size_t idx) const {
    return (idx < N) ? entries_[idx].size : 0;
  }

  /**
   * @brief Release all GPU buffers
   */
  void ReleaseAll() {
    for (size_t i = 0; i < N; ++i) {
      FreeEntry(entries_[i]);
    }
  }

  /**
   * @brief Number of buffer slots
   */
  static constexpr size_t Count() { return N; }

  /**
   * @brief Number of currently allocated (non-null) buffers
   */
  size_t AllocatedCount() const {
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
      if (entries_[i].ptr) ++count;
    }
    return count;
  }

private:
  GpuBufferEntry entries_[N] = {};

  static void FreeEntry(GpuBufferEntry& e) {
    if (e.ptr) {
#if ENABLE_ROCM
      (void)hipFree(e.ptr);
#endif
      e.ptr = nullptr;
      e.size = 0;
    }
  }
};

/**
 * @brief Specialization for N=0 (operations with no private buffers)
 */
template<>
class BufferSet<0> {
public:
  BufferSet() = default;
  ~BufferSet() = default;
  BufferSet(BufferSet&&) noexcept = default;
  BufferSet& operator=(BufferSet&&) noexcept = default;
  BufferSet(const BufferSet&) = delete;
  BufferSet& operator=(const BufferSet&) = delete;

  void ReleaseAll() {}
  static constexpr size_t Count() { return 0; }
  size_t AllocatedCount() const { return 0; }
};

}  // namespace drv_gpu_lib
