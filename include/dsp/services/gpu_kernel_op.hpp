#pragma once

/**
 * @file gpu_kernel_op.hpp
 * @brief GpuKernelOp — base class for operations that use compiled GPU kernels
 *
 * Part of Ref03 Unified Architecture (Layer 3).
 *
 * Provides:
 *   - ctx_ pointer to GpuContext (set by Initialize)
 *   - kernel(name) — lookup compiled kernel from GpuContext (throws if not found)
 *   - stream() — proxy to ctx_->stream()
 *   - OnInitialize() / OnRelease() — hooks for subclass-specific init/cleanup
 *
 * Concrete Ops (Layer 5) inherit from this and add:
 *   - BufferSet<N> for private GPU buffers
 *   - Execute(...) with operation-specific parameters
 *
 * IMPORTANT: GpuKernelOp does NOT compile kernels. The Facade compiles once
 * via GpuContext::CompileModule(), and all Ops share the compiled module.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#include "interface/i_gpu_operation.hpp"

#if ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace drv_gpu_lib {

class GpuContext;  // forward — full definition in gpu_context.hpp

class GpuKernelOp : public IGpuOperation {
public:
  ~GpuKernelOp() override = default;

  // ── IGpuOperation implementation ──────────────────────────────────────

  void Initialize(GpuContext& ctx) override {
    ctx_ = &ctx;
    OnInitialize();
  }

  bool IsReady() const override {
    return ctx_ != nullptr;
  }

  void Release() override {
    OnRelease();
    ctx_ = nullptr;
  }

protected:
  GpuContext* ctx_ = nullptr;

  /// Override for custom initialization (called after ctx_ is set)
  virtual void OnInitialize() {}

  /// Override for custom cleanup (called before ctx_ is cleared)
  virtual void OnRelease() {}

#if ENABLE_ROCM
  /// Get compiled kernel by name from GpuContext (throws if not found)
  hipFunction_t kernel(const char* name) const;

  /// Proxy to ctx_->stream()
  hipStream_t stream() const;
#endif
};

}  // namespace drv_gpu_lib
