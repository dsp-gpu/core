#pragma once

/**
 * @file i_gpu_operation.hpp
 * @brief IGpuOperation — minimal contract for all GPU operations
 *
 * Part of Ref03 Unified Architecture (Layer 2).
 *
 * Every concrete GPU operation (MeanReductionOp, FirFilterOp, GemmStep, etc.)
 * implements this interface. The Facade (StatisticsProcessor, FilterProcessor)
 * owns Op instances and calls Initialize/Release via GpuContext.
 *
 * Contract:
 *   - Name()       → human-readable identifier (for profiling & logging)
 *   - Initialize()  → bind to GpuContext (kernels, stream, shared buffers)
 *   - IsReady()     → true after successful Initialize()
 *   - Release()     → free private GPU resources (BufferSet, temp state)
 *
 * Note: Execute() is NOT part of this interface — each Op has its own
 * signature with specific params (beam_count, n_point, flags, etc.).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

namespace drv_gpu_lib {

// Forward declaration — full definition in gpu_context.hpp
class GpuContext;

class IGpuOperation {
public:
  virtual ~IGpuOperation() = default;

  /// Human-readable name for profiling and logging
  virtual const char* Name() const = 0;

  /// Bind to GpuContext (get kernel handles, stream, shared buffers)
  virtual void Initialize(GpuContext& ctx) = 0;

  /// True after successful Initialize()
  virtual bool IsReady() const = 0;

  /// Release private GPU resources (BufferSet, cached state)
  virtual void Release() = 0;
};

}  // namespace drv_gpu_lib
