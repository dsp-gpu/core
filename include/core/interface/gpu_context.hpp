#pragma once

/**
 * @file gpu_context.hpp
 * @brief GpuContext — per-module shared state for GPU operations
 *
 * Part of Ref03 Unified Architecture (Layer 1).
 *
 * Each module (StatisticsProcessor, FilterProcessor, etc.) creates its own
 * GpuContext. This provides:
 *   - backend + stream access (from IBackend)
 *   - Kernel compilation via hiprtc (one CompileModule call for ALL kernels)
 *   - Kernel lookup by name (GetKernel)
 *   - Shared GPU buffers (used by multiple Ops within the module)
 *   - Disk cache via KernelCacheService
 *   - WARP_SIZE determination by GPU architecture
 *
 * Thread safety: per-module instance → no shared mutable state between modules.
 * Operations within one module are sequential (same stream).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include <core/services/buffer_set.hpp>
#include <core/services/console_output.hpp>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

// Forward declarations
namespace drv_gpu_lib {
class IBackend;
class KernelCacheService;
}

namespace drv_gpu_lib {

class GpuContext {
public:
  // ═══════════════════════════════════════════════════════════════════════
  // Shared GPU buffer pool — module-wide buffers reused by multiple Ops.
  //
  // GpuContext is generic infrastructure: it does NOT know what each slot
  // means. Each module defines its own slot assignments, e.g.:
  //   statistics module → statistics::shared_buf::{kInput, kMagnitudes, ...}
  //   fft module        → fft::shared_buf::{kInput, kOutput, ...}
  //
  // Max slots per module is kMaxSharedBuffers (currently 8).
  // ═══════════════════════════════════════════════════════════════════════

  /// Maximum shared GPU buffer slots available per module.
  static constexpr size_t kMaxSharedBuffers = 8;

  // ═══════════════════════════════════════════════════════════════════════
  // Construction / Destruction
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Construct GpuContext for a module
   * @param backend Non-owning pointer to IBackend (must be ROCm, initialized)
   * @param module_name Human-readable name for logging (e.g. "Statistics")
   * @param cache_dir Disk cache directory for compiled HSACO (e.g. "modules/statistics/kernels")
   */
  GpuContext(IBackend* backend,
             const char* module_name,
             const std::string& cache_dir = "");

  ~GpuContext();

  // No copy
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

  // Move
  GpuContext(GpuContext&& other) noexcept;
  GpuContext& operator=(GpuContext&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════════════
  // Immutable accessors (thread-safe reads)
  // ═══════════════════════════════════════════════════════════════════════

  IBackend* backend() const { return backend_; }
  hipStream_t stream() const { return stream_; }
  const char* module_name() const { return module_name_; }
  int warp_size() const { return warp_size_; }
  const std::string& arch_name() const { return arch_name_; }

  // ═══════════════════════════════════════════════════════════════════════
  // Kernel compilation (lazy, one-time per module)
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Compile all kernels for this module in one hiprtc call
   * @param source HIP C++ source (from kernels::GetXxxKernelSource())
   * @param kernel_names List of __global__ function names to extract
   * @param extra_defines Additional -D flags (e.g. "-DBLOCK_SIZE=256")
   *
   * Uses disk cache (KernelCacheService) when available.
   * Sets warp_size_ based on GPU architecture (gfx9* → 64, else → 32).
   * Idempotent: second call is a no-op.
   */
  void CompileModule(const char* source,
                     const std::vector<std::string>& kernel_names,
                     const std::vector<std::string>& extra_defines = {});

  /**
   * @brief Get compiled kernel function by name
   * @throws std::runtime_error if kernel not found or not compiled
   */
  hipFunction_t GetKernel(const char* name) const;

  /// True if CompileModule() has been called successfully
  bool IsCompiled() const { return module_ != nullptr; }

  // ═══════════════════════════════════════════════════════════════════════
  // rocBLAS handle — ленивая инициализация (только для ROCm-модулей с BLAS)
  //
  // Handle создаётся при первом вызове GetRocblasHandleRaw() и привязывается
  // к stream_ данного GpuContext (= к конкретному GPU).
  //
  // Возвращает void* — caller кастует в rocblas_handle сам:
  //   auto blas = static_cast<rocblas_handle>(ctx_->GetRocblasHandleRaw());
  //
  // Thread-safe: защита уникальным мьютексом per-GpuContext.
  //
  // ⚠️ OPT-IN: Требует -DENABLE_ROCBLAS=1 при сборке модуля-потребителя.
  // core/CMakeLists.txt НЕ определяет ENABLE_ROCBLAS — это делает модуль,
  // которому нужен rocBLAS (например, linalg):
  //   find_package(rocblas REQUIRED)
  //   target_compile_definitions(MyLib PRIVATE ENABLE_ROCBLAS=1)
  //   target_link_libraries(MyLib PRIVATE roc::rocblas)
  // Без ENABLE_ROCBLAS вызов GetRocblasHandleRaw() бросит runtime_error.
  // ═══════════════════════════════════════════════════════════════════════
  void* GetRocblasHandleRaw() const;

  // ═══════════════════════════════════════════════════════════════════════
  // Shared GPU buffers (module-wide, used by multiple Ops)
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Get or allocate shared buffer
   * @param id Slot index (defined by the module, e.g. statistics::shared_buf::kInput)
   * @param bytes Required size in bytes
   * @return Device pointer (reused if existing buffer is large enough)
   */
  void* RequireShared(size_t id, size_t bytes) {
    return shared_.Require(id, bytes);
  }

  /// Get existing shared buffer (no allocation, nullptr if not allocated)
  void* GetShared(size_t id) const {
    return shared_.Get(id);
  }

  /// Release all shared buffers
  void ReleaseShared() { shared_.ReleaseAll(); }

private:
  // Backend (non-owning)
  IBackend* backend_ = nullptr;
  hipStream_t stream_ = nullptr;
  const char* module_name_ = "Unknown";

  // Architecture info (set in constructor)
  std::string arch_name_;
  int warp_size_ = 32;

  // Compiled kernels
  hipModule_t module_ = nullptr;
  std::unordered_map<std::string, hipFunction_t> kernels_;

  // Shared buffers (kMaxSharedBuffers slots; modules use a subset)
  BufferSet<kMaxSharedBuffers> shared_;

  // Disk cache (optional)
  std::unique_ptr<KernelCacheService> kernel_cache_;

  // rocBLAS handle — ленивая инициализация, уничтожается в деструкторе
  // unique_ptr<mutex> необходим для movability GpuContext
  mutable void*                       blas_handle_ = nullptr;
  mutable std::unique_ptr<std::mutex> blas_mutex_  = std::make_unique<std::mutex>();

  /// Release compiled module
  void ReleaseModule();
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
