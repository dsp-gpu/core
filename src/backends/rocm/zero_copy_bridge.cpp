/**
 * @file zero_copy_bridge.cpp
 * @brief Реализация ZeroCopyBridge (OpenCL ↔ ROCm)
 *
 * Стратегии ImportFromOpenCl (в порядке приоритета):
 * A. HSA Probe — GPU VA из cl_mem (TRUE zero-copy, 0 копий)
 * B. DMA-BUF через OpenCL extension
 * C. GPU Copy Kernel — cl_mem → coarse-grain SVM (VRAM→VRAM, ~8мс для 4ГБ)
 * D. SVM fallback (fine-grain, через CPU, секунды для 4ГБ)
 *
 * Программное переключение: ZeroCopyStrategy (AUTO, FORCE_HSA_PROBE, ...)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-24
 */

#if ENABLE_ROCM

#include "zero_copy_bridge.hpp"
#include "hsa_interop.hpp"
#include "../opencl/gpu_copy_kernel.hpp"
#include "../../logger/logger.hpp"

#include <hip/hip_runtime.h>
#include <cinttypes>  // PRIxPTR
#include <unistd.h>   // close() для fd cleanup при DMA-BUF ошибке

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Helper
// ════════════════════════════════════════════════════════════════════════════

static std::string PtrHex(const void* ptr) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%" PRIxPTR, reinterpret_cast<uintptr_t>(ptr));
  return buf;
}

static void CheckHip(hipError_t err, const char* operation) {
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("ZeroCopyBridge: HIP error in ") + operation +
        ": " + hipGetErrorString(err) +
        " (code " + std::to_string(static_cast<int>(err)) + ")");
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor / Move
// ════════════════════════════════════════════════════════════════════════════

ZeroCopyBridge::ZeroCopyBridge()
    : ext_mem_(nullptr)
    , hip_ptr_(nullptr)
    , size_(0)
    , method_(ZeroCopyMethod::NONE)
    , owns_memory_(false)
    , svm_cl_context_(nullptr) {
}

ZeroCopyBridge::~ZeroCopyBridge() {
  Release();
}

ZeroCopyBridge::ZeroCopyBridge(ZeroCopyBridge&& other) noexcept
    : ext_mem_(other.ext_mem_)
    , hip_ptr_(other.hip_ptr_)
    , size_(other.size_)
    , method_(other.method_)
    , owns_memory_(other.owns_memory_)
    , svm_cl_context_(other.svm_cl_context_) {
  other.ext_mem_ = nullptr;
  other.hip_ptr_ = nullptr;
  other.size_ = 0;
  other.method_ = ZeroCopyMethod::NONE;
  other.owns_memory_ = false;
  other.svm_cl_context_ = nullptr;
}

ZeroCopyBridge& ZeroCopyBridge::operator=(ZeroCopyBridge&& other) noexcept {
  if (this != &other) {
    Release();
    ext_mem_ = other.ext_mem_;
    hip_ptr_ = other.hip_ptr_;
    size_ = other.size_;
    method_ = other.method_;
    owns_memory_ = other.owns_memory_;
    svm_cl_context_ = other.svm_cl_context_;
    other.ext_mem_ = nullptr;
    other.hip_ptr_ = nullptr;
    other.size_ = 0;
    other.method_ = ZeroCopyMethod::NONE;
    other.owns_memory_ = false;
    other.svm_cl_context_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Import: HSA Probe (стратегия A — TRUE zero-copy)
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::ImportFromHsaProbe(cl_mem cl_buffer, size_t buffer_size) {
  if (IsActive()) Release();

  if (!cl_buffer) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromHsaProbe: cl_buffer is nullptr");
  }

  auto probe = ProbeGpuVA(cl_buffer, buffer_size);
  if (!probe.valid || !probe.gpu_va) {
    throw std::runtime_error(
        "ZeroCopyBridge::ImportFromHsaProbe: GPU VA not found in cl_mem "
        "(probe scanned " + std::to_string(kMaxProbeBytes) + " bytes)");
  }

  // GPU VA из cl_mem напрямую доступен в HIP (unified VA space, HSA runtime)
  hip_ptr_ = probe.gpu_va;
  size_ = buffer_size;
  method_ = ZeroCopyMethod::HSA_PROBE;
  owns_memory_ = false;  // Память принадлежит OpenCL cl_mem

  DRVGPU_LOG_INFO("ZeroCopyBridge",
      "Imported via HSA Probe (GPU VA=0x" +
      PtrHex(probe.gpu_va) +
      ", offset=+" + std::to_string(probe.offset) +
      ", size=" + std::to_string(buffer_size) + " bytes, TRUE zero-copy)");
}

// ════════════════════════════════════════════════════════════════════════════
// Import: DMA-BUF
// ════════════════════════════════════════════════════════════════════════════

hipError_t ZeroCopyBridge::ImportFromDmaBuf(int dma_buf_fd, size_t buffer_size) {
  if (IsActive()) Release();

  if (dma_buf_fd < 0) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromDmaBuf: invalid fd (" +
                             std::to_string(dma_buf_fd) + ")");
  }
  if (buffer_size == 0) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromDmaBuf: buffer_size must be > 0");
  }

  size_ = buffer_size;

  hipExternalMemoryHandleDesc ext_mem_desc = {};
#ifdef hipExternalMemoryHandleTypeDmaBufFd
  ext_mem_desc.type = hipExternalMemoryHandleTypeDmaBufFd;
#else
  // ROCm 7.2: нет dedicated DmaBuf type, OpaqueFd работает для dma-buf fd
  ext_mem_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
#endif
  ext_mem_desc.handle.fd = dma_buf_fd;
  ext_mem_desc.size = buffer_size;
  ext_mem_desc.flags = 0;

  hipError_t err = hipImportExternalMemory(&ext_mem_, &ext_mem_desc);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR("ZeroCopyBridge", "hipImportExternalMemory failed: " +
                     std::string(hipGetErrorString(err)));
    return err;
  }

  hipExternalMemoryBufferDesc buf_desc = {};
  buf_desc.offset = 0;
  buf_desc.size = buffer_size;
  buf_desc.flags = 0;

  err = hipExternalMemoryGetMappedBuffer(&hip_ptr_, ext_mem_, &buf_desc);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR("ZeroCopyBridge", "hipExternalMemoryGetMappedBuffer failed");
    (void)hipDestroyExternalMemory(ext_mem_);
    ext_mem_ = nullptr;
    return err;
  }

  method_ = ZeroCopyMethod::DMA_BUF;
  owns_memory_ = true;

  DRVGPU_LOG_INFO("ZeroCopyBridge", "Imported via DMA-BUF (fd=" +
                  std::to_string(dma_buf_fd) + ", size=" +
                  std::to_string(buffer_size) + " bytes)");
  return hipSuccess;
}

// ════════════════════════════════════════════════════════════════════════════
// Import: GPU VA (прямой указатель)
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::ImportFromGpuVA(void* gpu_va, size_t buffer_size) {
  if (IsActive()) Release();
  if (!gpu_va) throw std::runtime_error("ZeroCopyBridge::ImportFromGpuVA: gpu_va is nullptr");
  if (buffer_size == 0) throw std::runtime_error("ZeroCopyBridge::ImportFromGpuVA: buffer_size must be > 0");

  hip_ptr_ = gpu_va;
  size_ = buffer_size;
  method_ = ZeroCopyMethod::HSA_PROBE;  // Прямой VA в unified address space
  owns_memory_ = false;

  DRVGPU_LOG_INFO("ZeroCopyBridge", "Imported via direct GPU VA (ptr=0x" +
                  PtrHex(gpu_va) +
                  ", size=" + std::to_string(buffer_size) + " bytes)");
}

// ════════════════════════════════════════════════════════════════════════════
// Import: SVM pointer
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::ImportFromSVM(void* svm_ptr, size_t buffer_size) {
  if (IsActive()) Release();
  if (!svm_ptr) throw std::runtime_error("ZeroCopyBridge::ImportFromSVM: svm_ptr is nullptr");
  if (buffer_size == 0) throw std::runtime_error("ZeroCopyBridge::ImportFromSVM: buffer_size must be > 0");

  hip_ptr_ = svm_ptr;
  size_ = buffer_size;
  method_ = ZeroCopyMethod::SVM;
  owns_memory_ = false;
  svm_cl_context_ = nullptr;

  DRVGPU_LOG_INFO("ZeroCopyBridge", "Imported via SVM (ptr=0x" +
                  PtrHex(svm_ptr) +
                  ", size=" + std::to_string(buffer_size) + " bytes, zero-copy)");
}

// ════════════════════════════════════════════════════════════════════════════
// Universal Import: автоопределение стратегии
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::ImportFromOpenCl(cl_mem cl_buffer, size_t buffer_size,
                                       cl_device_id cl_device,
                                       ZeroCopyStrategy strategy) {
  if (IsActive()) Release();

  if (!cl_buffer) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromOpenCl: cl_buffer is nullptr");
  }

  const bool try_all = (strategy == ZeroCopyStrategy::AUTO);

  // ══════════════════════════════════════════════════════════════════════
  // Стратегия A: HSA Probe — TRUE zero-copy (приоритет 1)
  // GPU VA из cl_mem напрямую в HIP. 0 копий, 0 доп. памяти, ~μs.
  // ══════════════════════════════════════════════════════════════════════
  if (try_all || strategy == ZeroCopyStrategy::FORCE_HSA_PROBE) {
    if (IsHsaAvailable()) {
      auto probe = ProbeGpuVA(cl_buffer, buffer_size);
      if (probe.valid && probe.gpu_va) {
        hip_ptr_ = probe.gpu_va;
        size_ = buffer_size;
        method_ = ZeroCopyMethod::HSA_PROBE;
        owns_memory_ = false;

        DRVGPU_LOG_INFO("ZeroCopyBridge",
            "Imported via HSA Probe (GPU VA=0x" +
            PtrHex(probe.gpu_va) +
            ", offset=+" + std::to_string(probe.offset) +
            ", size=" + std::to_string(buffer_size) + ", TRUE zero-copy)");
        return;
      }
      if (try_all) {
        DRVGPU_LOG_WARNING("ZeroCopyBridge", "HSA Probe failed, trying DMA-BUF...");
      }
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  // Стратегия B: DMA-BUF через OpenCL extension
  // ══════════════════════════════════════════════════════════════════════
  if (try_all || strategy == ZeroCopyStrategy::FORCE_DMA_BUF) {
    if (SupportsDmaBufExport(cl_device)) {
      int fd = ExportClBufferToFd(cl_buffer);
      if (fd >= 0) {
        hipError_t err = ImportFromDmaBuf(fd, buffer_size);
        if (err == hipSuccess) {
#ifdef hipExternalMemoryHandleTypeDmaBufFd
          close(fd);  // DmaBufFd: HIP не берёт ownership
#endif
          DRVGPU_LOG_INFO("ZeroCopyBridge", "Imported via DMA-BUF");
          return;
        }
        close(fd);
        if (try_all) {
          DRVGPU_LOG_WARNING("ZeroCopyBridge", "DMA-BUF failed, trying GPU Copy...");
        }
      }
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  // Стратегия C: GPU Copy Kernel — cl_mem → coarse-grain SVM (VRAM→VRAM)
  // Данные не покидают GPU. ~8-15мс для 4ГБ.
  // ══════════════════════════════════════════════════════════════════════
  if (try_all || strategy == ZeroCopyStrategy::FORCE_GPU_COPY) {
    if (SupportsSVMCoarseGrain(cl_device)) {
      cl_context ctx = nullptr;
      cl_int ctx_err = clGetMemObjectInfo(cl_buffer, CL_MEM_CONTEXT,
                                           sizeof(ctx), &ctx, nullptr);
      if (ctx_err == CL_SUCCESS && ctx) {
        // Coarse-grain SVM: VRAM на dGPU (без FINE_GRAIN флага)
        void* svm_ptr = clSVMAlloc(ctx, CL_MEM_READ_WRITE, buffer_size, 0);
        if (svm_ptr) {
          cl_int q_err;
          cl_command_queue temp_q = clCreateCommandQueueWithProperties(
              ctx, cl_device, nullptr, &q_err);
          if (q_err == CL_SUCCESS && temp_q) {
            bool ok = GpuCopyClMemToSVM(temp_q, ctx, cl_buffer, svm_ptr, buffer_size);
            clReleaseCommandQueue(temp_q);

            if (ok) {
              clRetainContext(ctx);
              hip_ptr_ = svm_ptr;
              size_ = buffer_size;
              method_ = ZeroCopyMethod::GPU_COPY;
              owns_memory_ = true;
              svm_cl_context_ = ctx;

              DRVGPU_LOG_INFO("ZeroCopyBridge",
                  "Imported via GPU Copy Kernel (VRAM→VRAM, size=" +
                  std::to_string(buffer_size) + " bytes)");
              return;
            }
            if (try_all) {
              DRVGPU_LOG_WARNING("ZeroCopyBridge",
                  "GPU Copy Kernel failed, trying SVM fallback...");
            }
          } else {
            if (temp_q) clReleaseCommandQueue(temp_q);
          }
          clSVMFree(ctx, svm_ptr);
        }
      }
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  // Стратегия D: SVM fallback — fine-grain SVM + копия через CPU
  // Медленно (секунды для 4ГБ), но работает на всех ROCm системах.
  // ══════════════════════════════════════════════════════════════════════
  if (try_all || strategy == ZeroCopyStrategy::FORCE_SVM) {
    cl_device_svm_capabilities svm_caps = 0;
    cl_int svm_err = clGetDeviceInfo(cl_device, CL_DEVICE_SVM_CAPABILITIES,
                                      sizeof(svm_caps), &svm_caps, nullptr);
    if (svm_err == CL_SUCCESS && (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
      cl_context ctx = nullptr;
      cl_int ctx_err = clGetMemObjectInfo(cl_buffer, CL_MEM_CONTEXT,
                                           sizeof(ctx), &ctx, nullptr);
      if (ctx_err == CL_SUCCESS && ctx) {
        void* svm_ptr = clSVMAlloc(ctx,
            CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE,
            buffer_size, 0);
        if (svm_ptr) {
          cl_command_queue temp_q = clCreateCommandQueueWithProperties(
              ctx, cl_device, nullptr, &svm_err);
          if (svm_err == CL_SUCCESS && temp_q) {
            svm_err = clEnqueueReadBuffer(temp_q, cl_buffer, CL_TRUE,
                0, buffer_size, svm_ptr, 0, nullptr, nullptr);
            clReleaseCommandQueue(temp_q);

            if (svm_err == CL_SUCCESS) {
              clRetainContext(ctx);
              hip_ptr_ = svm_ptr;
              size_ = buffer_size;
              method_ = ZeroCopyMethod::SVM;
              owns_memory_ = true;
              svm_cl_context_ = ctx;

              DRVGPU_LOG_INFO("ZeroCopyBridge",
                  "Imported via SVM fallback (CPU copy, size=" +
                  std::to_string(buffer_size) + " bytes)");
              return;
            }
          } else {
            if (temp_q) clReleaseCommandQueue(temp_q);
          }
          clSVMFree(ctx, svm_ptr);
        }
      }
    }
  }

  throw std::runtime_error(
      "ZeroCopyBridge::ImportFromOpenCl: no ZeroCopy method available"
      " (strategy=" + std::to_string(static_cast<int>(strategy)) +
      "). Detected: " + std::string(ZeroCopyMethodToString(
          DetectBestZeroCopyMethod(cl_device))));
}

// ════════════════════════════════════════════════════════════════════════════
// Release
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::Release() {
  if (owns_memory_) {
    if ((method_ == ZeroCopyMethod::SVM || method_ == ZeroCopyMethod::GPU_COPY)
        && svm_cl_context_ && hip_ptr_) {
      clSVMFree(svm_cl_context_, hip_ptr_);
      clReleaseContext(svm_cl_context_);
      DRVGPU_LOG_INFO("ZeroCopyBridge", "Released SVM memory");
    } else if (ext_mem_) {
      if (hip_ptr_) {
        (void)hipFree(hip_ptr_);  // mapped buffer от hipExternalMemoryGetMappedBuffer
      }
      (void)hipDestroyExternalMemory(ext_mem_);
      DRVGPU_LOG_INFO("ZeroCopyBridge", "Released external memory");
    }
  }
  // HSA_PROBE, GPU VA, external SVM: owns_memory_=false, ничего не освобождаем

  ext_mem_ = nullptr;
  hip_ptr_ = nullptr;
  size_ = 0;
  method_ = ZeroCopyMethod::NONE;
  owns_memory_ = false;
  svm_cl_context_ = nullptr;
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
