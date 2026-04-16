/**
 * @file gpu_context.cpp
 * @brief GpuContext implementation — kernel compilation, disk cache, shared buffers
 *
 * Part of Ref03 Unified Architecture (Layer 1).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include <core/interface/gpu_context.hpp>
#include <core/interface/i_backend.hpp>
#include <core/backends/rocm/rocm_backend.hpp>
#include <core/services/kernel_cache_service.hpp>
#include <core/services/console_output.hpp>

#ifdef ENABLE_ROCBLAS
#include <rocblas/rocblas.h>
#endif

#include <cstring>
#include <algorithm>

namespace drv_gpu_lib {

// ═════════════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═════════════════════════════════════════════════════════════════════════════

GpuContext::GpuContext(IBackend* backend,
                       const char* module_name,
                       const std::string& cache_dir)
    : backend_(backend)
    , module_name_(module_name) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: backend is null or not initialized");
  }

  if (backend_->GetType() != BackendType::ROCm) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: requires ROCm backend");
  }

  // Get HIP stream from backend
  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: failed to get HIP stream");
  }

  // Determine GPU architecture and warp size from hipDeviceProp_t (авторитетный источник)
  auto* rocm_backend = dynamic_cast<ROCmBackend*>(backend_);
  if (!rocm_backend) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ +
        "]: backend is not ROCmBackend (dynamic_cast failed)");
  }
  arch_name_ = rocm_backend->GetCore().GetArchName();
  warp_size_ = rocm_backend->GetCore().GetWarpSize();

  // Disk cache for compiled HSACO.
  // Per-arch подкаталог (gfx908/gfx1100/…) добавляется внутри KernelCacheService.
  // Это КРИТИЧНО для multi-GPU: HSACO скомпилированный для gfx908 не будет
  // использован на gfx1100 — изоляция через подкаталог по arch.
  if (!cache_dir.empty()) {
    kernel_cache_ = std::make_unique<KernelCacheService>(
        cache_dir, BackendType::ROCm, arch_name_);  // ← arch_name_ определён выше
  }
}

GpuContext::~GpuContext() {
  ReleaseShared();
  ReleaseModule();
#ifdef ENABLE_ROCBLAS
  if (blas_handle_) {
    rocblas_destroy_handle(static_cast<rocblas_handle>(blas_handle_));
    blas_handle_ = nullptr;
  }
#endif
}

GpuContext::GpuContext(GpuContext&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , module_name_(other.module_name_)
    , arch_name_(std::move(other.arch_name_))
    , warp_size_(other.warp_size_)
    , module_(other.module_)
    , kernels_(std::move(other.kernels_))
    , shared_(std::move(other.shared_))
    , kernel_cache_(std::move(other.kernel_cache_))
    , blas_handle_(other.blas_handle_)
    , blas_mutex_(std::move(other.blas_mutex_)) {
  other.backend_      = nullptr;
  other.stream_       = nullptr;
  other.module_       = nullptr;
  other.blas_handle_  = nullptr;
}

GpuContext& GpuContext::operator=(GpuContext&& other) noexcept {
  if (this != &other) {
    ReleaseShared();
    ReleaseModule();
#ifdef ENABLE_ROCBLAS
    if (blas_handle_) {
      rocblas_destroy_handle(static_cast<rocblas_handle>(blas_handle_));
      blas_handle_ = nullptr;
    }
#endif

    backend_      = other.backend_;
    stream_       = other.stream_;
    module_name_  = other.module_name_;
    arch_name_    = std::move(other.arch_name_);
    warp_size_    = other.warp_size_;
    module_       = other.module_;
    kernels_      = std::move(other.kernels_);
    shared_       = std::move(other.shared_);
    kernel_cache_ = std::move(other.kernel_cache_);
    blas_handle_  = other.blas_handle_;
    blas_mutex_   = std::move(other.blas_mutex_);

    other.backend_     = nullptr;
    other.stream_      = nullptr;
    other.module_      = nullptr;
    other.blas_handle_ = nullptr;
  }
  return *this;
}

// ═════════════════════════════════════════════════════════════════════════════
// Kernel Compilation
// ═════════════════════════════════════════════════════════════════════════════

void GpuContext::CompileModule(const char* source,
                               const std::vector<std::string>& kernel_names,
                               const std::vector<std::string>& extra_defines) {
  if (module_) return;  // already compiled

  auto& con = ConsoleOutput::GetInstance();
  const std::string cache_name = std::string(module_name_) + "_kernels";

  // Helper: extract all kernel functions from loaded module
  auto extractKernels = [&]() {
    for (const auto& name : kernel_names) {
      hipFunction_t func = nullptr;
      hipError_t err = hipModuleGetFunction(&func, module_, name.c_str());
      if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("GpuContext[") + module_name_ + "]: hipModuleGetFunction(" +
            name + ") failed: " + hipGetErrorString(err));
      }
      kernels_[name] = func;
    }
  };

  const int gpu_id = backend_->GetDeviceIndex();

  // ─── Try loading from disk cache ──────────────────────────────────────
  if (kernel_cache_) {
    auto entry = kernel_cache_->Load(cache_name);  // nullopt = cache miss
    if (entry && entry->has_binary()) {
      hipError_t err = hipModuleLoadData(&module_, entry->binary.data());
      if (err == hipSuccess) {
        extractKernels();
        con.Print(gpu_id, module_name_, "kernels loaded from cache (HSACO)");
        return;
      }
      // Cache might be stale (different arch) — fall through to compile
      if (module_) { hipModuleUnload(module_); module_ = nullptr; }
    }
  }

  // ─── Compile via hiprtc ───────────────────────────────────────────────
  hiprtcProgram prog;
  std::string filename = std::string(module_name_) + "_kernels.hip";
  hiprtcResult rtc = hiprtcCreateProgram(&prog, source, filename.c_str(),
                                          0, nullptr, nullptr);
  if (rtc != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: hiprtcCreateProgram failed: " +
        std::to_string(static_cast<int>(rtc)));
  }

  // Build compiler flags
  std::string warp_def  = "-DWARP_SIZE=" + std::to_string(warp_size_);
  std::string arch_flag = arch_name_.empty() ? "" : ("--offload-arch=" + arch_name_);

  std::vector<std::string> opts_storage = {"-O3", "-std=c++17", warp_def};
  for (const auto& def : extra_defines) {
    opts_storage.push_back(def);
  }
  if (!arch_flag.empty()) {
    opts_storage.push_back(arch_flag);
  }

  std::vector<const char*> opts;
  opts.reserve(opts_storage.size());
  for (const auto& o : opts_storage) {
    opts.push_back(o.c_str());
  }

  rtc = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
  if (rtc != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: compilation failed:\n" + log);
  }

  // Extract compiled binary (HSACO)
  size_t code_size = 0;
  hiprtcGetCodeSize(prog, &code_size);
  std::vector<char> code(code_size);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  // Load module into GPU
  hipError_t hip_err = hipModuleLoadData(&module_, code.data());
  if (hip_err != hipSuccess) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: hipModuleLoadData failed: " +
        hipGetErrorString(hip_err));
  }

  // Extract kernel functions
  extractKernels();

  con.Print(gpu_id, module_name_,
            "kernels compiled (" + std::to_string(code_size) + " bytes HSACO" +
            (arch_name_.empty() ? "" : ", " + arch_name_) + ")");

  // ─── Save to disk cache ───────────────────────────────────────────────
  if (kernel_cache_) {
    try {
      std::vector<uint8_t> binary(code.begin(), code.end());
      kernel_cache_->Save(cache_name, source, binary, "",
                          std::string(module_name_) + " hiprtc kernels");
    } catch (...) {
      // Non-fatal: cache save failure doesn't stop execution
    }
  }
}

hipFunction_t GpuContext::GetKernel(const char* name) const {
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: kernel '" + name +
        "' not found. CompileModule() not called or name misspelled.");
  }
  return it->second;
}

// ═════════════════════════════════════════════════════════════════════════════
// Private
// ═════════════════════════════════════════════════════════════════════════════

void GpuContext::ReleaseModule() {
  if (module_) {
    hipModuleUnload(module_);
    module_ = nullptr;
  }
  kernels_.clear();
}

// ═════════════════════════════════════════════════════════════════════════════
// rocBLAS handle — ленивая инициализация
// ═════════════════════════════════════════════════════════════════════════════

void* GpuContext::GetRocblasHandleRaw() const {
#ifdef ENABLE_ROCBLAS
  std::lock_guard<std::mutex> lock(*blas_mutex_);
  if (!blas_handle_) {
    // hipSetDevice обязателен: при 10-15 GPU текущий device в потоке
    // может не совпадать с device этого GpuContext.
    hipSetDevice(backend_->GetDeviceIndex());

    rocblas_handle h;
    rocblas_status status = rocblas_create_handle(&h);
    if (status != rocblas_status_success) {
      throw std::runtime_error(
          std::string("GpuContext[") + module_name_ +
          "]: rocblas_create_handle failed (" +
          std::to_string(static_cast<int>(status)) + ")");
    }
    rocblas_set_stream(h, stream_);
    blas_handle_ = static_cast<void*>(h);

    ConsoleOutput::GetInstance().Print(
        backend_->GetDeviceIndex(), module_name_, "rocBLAS handle created");
  }
  return blas_handle_;
#else
  throw std::runtime_error(
      std::string("GpuContext[") + module_name_ +
      "]: rocBLAS not available (ENABLE_ROCBLAS not defined)");
#endif
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
