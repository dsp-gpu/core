#pragma once

/**
 * @file py_gpu_context.hpp
 * @brief ROCmGPUContext / HybridGPUContext — общие для Python биндингов dsp-gpu
 *
 * Подключается из:
 *   - core/python/dsp_core_module.cpp     — регистрация классов
 *   - spectrum/python/py_fft_processor_rocm.hpp — аргумент конструктора
 *   - stats/python/, signal_generators/python/ — аналогично
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-16
 */

#if ENABLE_ROCM

#include <core/backends/rocm/rocm_backend.hpp>
#include <core/backends/hybrid/hybrid_backend.hpp>
#include <core/interface/i_backend.hpp>

#include <memory>
#include <string>

// ============================================================================
// ROCmGPUContext — Python-обёртка над ROCmBackend
// ============================================================================

class ROCmGPUContext {
public:
  explicit ROCmGPUContext(int device_index = 0)
      : backend_(std::make_unique<drv_gpu_lib::ROCmBackend>()) {
    backend_->Initialize(device_index);
  }

  ~ROCmGPUContext() = default;
  ROCmGPUContext(const ROCmGPUContext&) = delete;
  ROCmGPUContext& operator=(const ROCmGPUContext&) = delete;

  drv_gpu_lib::IBackend* backend() { return backend_.get(); }
  std::string device_name() const { return backend_->GetDeviceName(); }
  int device_index() const { return backend_->GetDeviceIndex(); }

private:
  std::unique_ptr<drv_gpu_lib::ROCmBackend> backend_;
};

// ============================================================================
// HybridGPUContext — Python-обёртка над HybridBackend (OpenCL + ROCm)
// ============================================================================

class HybridGPUContext {
public:
  explicit HybridGPUContext(int device_index = 0)
      : backend_(std::make_unique<drv_gpu_lib::HybridBackend>()) {
    backend_->Initialize(device_index);
  }

  ~HybridGPUContext() = default;
  HybridGPUContext(const HybridGPUContext&) = delete;
  HybridGPUContext& operator=(const HybridGPUContext&) = delete;

  drv_gpu_lib::HybridBackend* backend() { return backend_.get(); }
  std::string device_name() const { return backend_->GetDeviceName(); }
  int device_index() const { return backend_->GetDeviceIndex(); }

  std::string opencl_device_name() const {
    auto* ocl = backend_->GetOpenCL();
    if (ocl && ocl->IsInitialized()) return ocl->GetDeviceName();
    return "Unknown";
  }

  std::string rocm_device_name() const {
    auto* rocm = backend_->GetROCm();
    if (rocm && rocm->IsInitialized()) return rocm->GetDeviceName();
    return "Unknown";
  }

  std::string zero_copy_method() const {
    auto method = backend_->GetBestZeroCopyMethod();
    return drv_gpu_lib::ZeroCopyMethodToString(method);
  }

  bool is_zero_copy_supported() const {
    return backend_->GetBestZeroCopyMethod() != drv_gpu_lib::ZeroCopyMethod::NONE;
  }

private:
  std::unique_ptr<drv_gpu_lib::HybridBackend> backend_;
};

#endif  // ENABLE_ROCM
