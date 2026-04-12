#pragma once

/// @file gpu_transfer.hpp
/// @brief ReadGpuBuffer, PeekGpuBuffer — GPU→CPU трансфер (OpenCL + ROCm).
/// Заменяет ~40 мест с 5-строчным clEnqueueReadBuffer паттерном.
/// Review R1: hipMemcpyAsync + hipStreamSynchronize (не hipMemcpy!).
/// Review R9: PeekGpuBuffer с offset для чтения части буфера.

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>

// DrvGPU includes
#include "interface/i_backend.hpp"

// OpenCL — всегда доступен
#include <CL/cl.h>

// ROCm — условная компиляция
#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace gpu_test_utils {

// ══════════════════════════════════════════════════════════════════
// OpenCL: cl_mem → vector<T>
// ══════════════════════════════════════════════════════════════════

template<typename T>
inline std::vector<T>
ReadClBuffer(cl_command_queue queue, cl_mem buffer, size_t count,
             size_t offset = 0, bool release = true)
{
  std::vector<T> result(count);
  cl_int err = clEnqueueReadBuffer(
      queue, buffer, CL_TRUE, offset * sizeof(T),
      count * sizeof(T), result.data(),
      0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    if (release) clReleaseMemObject(buffer);
    throw std::runtime_error(
        "clEnqueueReadBuffer failed, error=" + std::to_string(err));
  }
  if (release) clReleaseMemObject(buffer);
  return result;
}

template<typename T>
inline void WriteClBuffer(cl_command_queue queue, cl_mem buffer,
                          const std::vector<T>& data)
{
  cl_int err = clEnqueueWriteBuffer(
      queue, buffer, CL_TRUE, 0,
      data.size() * sizeof(T), data.data(),
      0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error(
        "clEnqueueWriteBuffer failed, error=" + std::to_string(err));
}

// ══════════════════════════════════════════════════════════════════
// ROCm: void* → vector<T>  (Review R1: hipMemcpyAsync + StreamSync)
// ══════════════════════════════════════════════════════════════════

#ifdef ENABLE_ROCM

template<typename T>
inline std::vector<T>
ReadHipBuffer(void* native_queue, void* device_ptr, size_t count,
              size_t offset = 0, bool free_buffer = true)
{
  auto stream = static_cast<hipStream_t>(native_queue);
  auto src = static_cast<char*>(device_ptr) + offset * sizeof(T);
  std::vector<T> result(count);

  hipError_t err = hipMemcpyAsync(
      result.data(), src,
      count * sizeof(T), hipMemcpyDeviceToHost, stream);
  if (err != hipSuccess) {
    if (free_buffer) hipFree(device_ptr);
    throw std::runtime_error(
        std::string("hipMemcpyAsync D2H failed: ") + hipGetErrorString(err));
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    if (free_buffer) hipFree(device_ptr);
    throw std::runtime_error(
        std::string("hipStreamSynchronize failed: ") + hipGetErrorString(err));
  }
  if (free_buffer) hipFree(device_ptr);
  return result;
}

template<typename T>
inline void WriteHipBuffer(void* native_queue, void* device_ptr,
                           const std::vector<T>& data)
{
  auto stream = static_cast<hipStream_t>(native_queue);
  hipError_t err = hipMemcpyAsync(
      device_ptr, data.data(),
      data.size() * sizeof(T), hipMemcpyHostToDevice, stream);
  if (err != hipSuccess)
    throw std::runtime_error(
        std::string("hipMemcpyAsync H2D failed: ") + hipGetErrorString(err));
  hipStreamSynchronize(stream);
}

#endif // ENABLE_ROCM

// ══════════════════════════════════════════════════════════════════
// Backend-agnostic: автоопределение OpenCL vs ROCm
// ══════════════════════════════════════════════════════════════════

/// Основная функция для тестов: auto result = ReadGpuBuffer<cf>(backend, buf, n);
template<typename T>
inline std::vector<T>
ReadGpuBuffer(drv_gpu_lib::IBackend* backend, void* buffer, size_t count,
              size_t offset = 0, bool release = true)
{
#ifdef ENABLE_ROCM
  if (backend->GetType() == drv_gpu_lib::BackendType::ROCm) {
    return ReadHipBuffer<T>(backend->GetNativeQueue(), buffer, count,
                            offset, release);
  }
#endif
  auto queue = static_cast<cl_command_queue>(backend->GetNativeQueue());
  auto cl_buf = static_cast<cl_mem>(buffer);
  return ReadClBuffer<T>(queue, cl_buf, count, offset, release);
}

/// Читать БЕЗ освобождения (буфер переиспользуется).
/// Пример: auto ant3 = PeekGpuBuffer<cf>(backend, buf, 4096, 3*4096);
template<typename T>
inline std::vector<T>
PeekGpuBuffer(drv_gpu_lib::IBackend* backend, void* buffer, size_t count,
              size_t offset = 0)
{
  return ReadGpuBuffer<T>(backend, buffer, count, offset, /*release=*/false);
}

} // namespace gpu_test_utils
