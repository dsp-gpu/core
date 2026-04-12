/**
 * @file gpu_kernel_op.cpp
 * @brief GpuKernelOp implementation — kernel/stream proxy methods
 *
 * Part of Ref03 Unified Architecture (Layer 3).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"

namespace drv_gpu_lib {

hipFunction_t GpuKernelOp::kernel(const char* name) const {
  return ctx_->GetKernel(name);
}

hipStream_t GpuKernelOp::stream() const {
  return ctx_->stream();
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
