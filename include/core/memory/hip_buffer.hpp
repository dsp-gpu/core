#pragma once

/**
 * @file hip_buffer.hpp
 * @brief HIPBuffer - Non-owning обёртка для HIP/ROCm GPU памяти
 *
 * Отличие от GPUBuffer<T>:
 * - GPUBuffer<T> — backend-агностичный RAII буфер (через IBackend),
 *   ВСЕГДА владеет памятью и освобождает её в деструкторе.
 *
 * - HIPBuffer — non-owning обёртка для ГОТОВЫХ hipDeviceptr_t указателей.
 *   Используется в ZeroCopy сценариях и при работе с внешними HIP буферами.
 *   Не освобождает память в деструкторе (не владеет ею).
 *
 * Для обычной работы используйте GPUBuffer<T> через MemoryManager:
 *   auto buf = mem_mgr->CreateBuffer<float>(N);
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-24
 */

#if ENABLE_ROCM

#include <hip/hip_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: HIPBuffer — Non-owning wrapper для hipDeviceptr_t
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class HIPBuffer
 * @brief Non-owning обёртка для существующих HIP device указателей
 *
 * Используется для:
 * - ZeroCopy буферов (hipHostMalloc → hipMemGetAddressRange)
 * - Внешних HIP аллокаций (от hipFFT, rocBLAS и пр.)
 * - Wrapping'а указателей от hipMalloc без передачи владения
 *
 * НЕ вызывает hipFree в деструкторе.
 *
 * @tparam T Тип элементов
 */
template<typename T>
class HIPBuffer {
public:
  // ═══════════════════════════════════════════════════════════════
  // Конструкторы
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Создать non-owning HIPBuffer из готового device указателя
   * @param hip_ptr  Указатель на GPU память (hipDeviceptr_t, приведённый к void*)
   * @param num_elements Количество элементов типа T
   * @param stream   HIP stream для асинхронных операций (nullptr = default stream)
   */
  HIPBuffer(void* hip_ptr, size_t num_elements, hipStream_t stream = nullptr)
      : ptr_(hip_ptr),
        num_elements_(num_elements),
        size_bytes_(num_elements * sizeof(T)),
        stream_(stream) {
    if (!ptr_) {
      throw std::invalid_argument("HIPBuffer: hip_ptr must not be null");
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Деструктор — НЕ освобождает память (non-owning)
  // ═══════════════════════════════════════════════════════════════

  ~HIPBuffer() = default;

  // ═══════════════════════════════════════════════════════════════
  // Запрет копирования, разрешение перемещения
  // ═══════════════════════════════════════════════════════════════

  HIPBuffer(const HIPBuffer&) = delete;
  HIPBuffer& operator=(const HIPBuffer&) = delete;

  HIPBuffer(HIPBuffer&& other) noexcept
      : ptr_(other.ptr_),
        num_elements_(other.num_elements_),
        size_bytes_(other.size_bytes_),
        stream_(other.stream_) {
    other.ptr_ = nullptr;
  }

  HIPBuffer& operator=(HIPBuffer&& other) noexcept {
    if (this != &other) {
      ptr_ = other.ptr_;
      num_elements_ = other.num_elements_;
      size_bytes_ = other.size_bytes_;
      stream_ = other.stream_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  // ═══════════════════════════════════════════════════════════════
  // Операции с данными (через hipMemcpy напрямую)
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Записать данные Host -> Device (синхронно)
   * @param host_data Указатель на данные на host
   * @param size_bytes Размер данных в байтах (0 = весь буфер)
   */
  void Write(const void* host_data, size_t size_bytes = 0) {
    if (!ptr_ || !host_data) {
      throw std::runtime_error("HIPBuffer::Write: null pointer");
    }
    size_t bytes = (size_bytes == 0) ? size_bytes_ : size_bytes;
    if (bytes > size_bytes_) {
      throw std::runtime_error("HIPBuffer::Write: size exceeds buffer capacity");
    }

    hipError_t err;
    if (stream_) {
      err = hipMemcpyHtoDAsync(ptr_, const_cast<void*>(host_data), bytes, stream_);
      if (err == hipSuccess) hipStreamSynchronize(stream_);
    } else {
      err = hipMemcpyHtoD(ptr_, const_cast<void*>(host_data), bytes);
    }

    if (err != hipSuccess) {
      throw std::runtime_error(
          std::string("HIPBuffer::Write failed: ") + hipGetErrorString(err));
    }
  }

  /**
   * @brief Записать std::vector<T> Host -> Device
   */
  void Write(const std::vector<T>& data) {
    Write(data.data(), data.size() * sizeof(T));
  }

  /**
   * @brief Прочитать данные Device -> Host (синхронно)
   * @param host_data Указатель на буфер на host
   * @param size_bytes Размер данных в байтах (0 = весь буфер)
   */
  void Read(void* host_data, size_t size_bytes = 0) const {
    if (!ptr_ || !host_data) {
      throw std::runtime_error("HIPBuffer::Read: null pointer");
    }
    size_t bytes = (size_bytes == 0) ? size_bytes_ : size_bytes;
    if (bytes > size_bytes_) {
      throw std::runtime_error("HIPBuffer::Read: size exceeds buffer capacity");
    }

    hipError_t err;
    if (stream_) {
      err = hipMemcpyDtoHAsync(host_data, ptr_, bytes, stream_);
      if (err == hipSuccess) hipStreamSynchronize(stream_);
    } else {
      err = hipMemcpyDtoH(host_data, ptr_, bytes);
    }

    if (err != hipSuccess) {
      throw std::runtime_error(
          std::string("HIPBuffer::Read failed: ") + hipGetErrorString(err));
    }
  }

  /**
   * @brief Прочитать в std::vector<T>
   */
  std::vector<T> Read() const {
    std::vector<T> result(num_elements_);
    Read(result.data(), size_bytes_);
    return result;
  }

  /**
   * @brief Копировать из другого HIPBuffer (Device -> Device)
   */
  void CopyFrom(const HIPBuffer<T>& other) {
    if (other.GetSizeBytes() > size_bytes_) {
      throw std::runtime_error("HIPBuffer::CopyFrom: source buffer is too large");
    }
    hipError_t err;
    if (stream_) {
      err = hipMemcpyDtoDAsync(ptr_, other.GetDevicePtr(), other.GetSizeBytes(), stream_);
      if (err == hipSuccess) hipStreamSynchronize(stream_);
    } else {
      err = hipMemcpyDtoD(ptr_, other.GetDevicePtr(), other.GetSizeBytes());
    }

    if (err != hipSuccess) {
      throw std::runtime_error(
          std::string("HIPBuffer::CopyFrom failed: ") + hipGetErrorString(err));
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Информация о буфере
  // ═══════════════════════════════════════════════════════════════

  /** @brief Получить указатель на GPU память (void* = hipDeviceptr_t) */
  void* GetDevicePtr() const { return ptr_; }

  /** @brief Получить указатель на GPU память */
  void* GetPtr() const { return ptr_; }

  /** @brief Количество элементов */
  size_t GetNumElements() const { return num_elements_; }

  /** @brief Размер в байтах */
  size_t GetSizeBytes() const { return size_bytes_; }

  /** @brief Проверить валидность */
  bool IsValid() const { return ptr_ != nullptr; }

  /** @brief HIP stream */
  hipStream_t GetStream() const { return stream_; }

private:
  void* ptr_;              ///< Указатель на GPU память (non-owning)
  size_t num_elements_;    ///< Количество элементов
  size_t size_bytes_;      ///< Размер в байтах
  hipStream_t stream_;     ///< HIP stream (nullptr = default)
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
