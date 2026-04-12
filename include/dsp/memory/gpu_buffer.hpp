#pragma once

/**
 * @file gpu_buffer.hpp
 * @brief GPUBuffer - RAII обёртка для GPU памяти
 * 
 * Типобезопасный буфер с автоматическим управлением памятью.
 * Backend-агностичный (работает с любым IBackend).
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "../interface/i_backend.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: GPUBuffer - Типобезопасный GPU буфер
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class GPUBuffer
 * @brief RAII обёртка для GPU памяти с типобезопасностью
 * 
 * @tparam T Тип элементов в буфере
 * 
 * Особенности:
 * - RAII (автоматическое освобождение в деструкторе)
 * - Типобезопасность (шаблонный класс)
 * - Backend-агностичный (работает через IBackend)
 * - Move semantics (нельзя копировать, можно перемещать)
 * 
 * Использование:
 * @code
 * GPUBuffer<float> buffer(ptr, 1024, backend);
 * 
 * // Записать данные
 * std::vector<float> data(1024, 1.0f);
 * buffer.Write(data.data(), 1024 * sizeof(float));
 * 
 * // Прочитать данные
 * std::vector<float> result(1024);
 * buffer.Read(result.data(), 1024 * sizeof(float));
 * @endcode
 */
template<typename T>
class GPUBuffer {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать GPUBuffer из существующего указателя
     * @param ptr Указатель на GPU память
     * @param num_elements Количество элементов
     * @param backend Указатель на бэкенд
     */
    GPUBuffer(void* ptr, size_t num_elements, IBackend* backend)
        : ptr_(ptr), 
          num_elements_(num_elements),
          size_bytes_(num_elements * sizeof(T)),
          backend_(backend)
    {
        if (!ptr_ || !backend_) {
            throw std::invalid_argument("GPUBuffer: ptr and backend must not be null");
        }
    }
    
    /**
     * @brief Деструктор (RAII - освобождает память)
     */
    ~GPUBuffer() {
        if (ptr_ && backend_) {
            backend_->Free(ptr_);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    
    GPUBuffer(GPUBuffer&& other) noexcept
        : ptr_(other.ptr_),
          num_elements_(other.num_elements_),
          size_bytes_(other.size_bytes_),
          backend_(other.backend_)
    {
        other.ptr_ = nullptr;
        other.backend_ = nullptr;
    }
    
    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            // Free сначала — иначе текущая GPU память останется без владельца (утечка).
            if (ptr_ && backend_) {
                backend_->Free(ptr_);
            }

            // Переместить ресурсы
            ptr_ = other.ptr_;
            num_elements_ = other.num_elements_;
            size_bytes_ = other.size_bytes_;
            backend_ = other.backend_;
            
            other.ptr_ = nullptr;
            other.backend_ = nullptr;
        }
        return *this;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Операции с данными
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Записать данные Host -> Device
     * @param host_data Указатель на данные на host
     * @param size_bytes Размер данных в байтах
     */
    void Write(const void* host_data, size_t size_bytes) {
        if (size_bytes > size_bytes_) {
            throw std::runtime_error("GPUBuffer::Write: size exceeds buffer capacity");
        }
        backend_->MemcpyHostToDevice(ptr_, host_data, size_bytes);
    }
    
    /**
     * @brief Записать std::vector
     */
    void Write(const std::vector<T>& data) {
        Write(data.data(), data.size() * sizeof(T));
    }
    
    /**
     * @brief Прочитать данные Device -> Host
     * @param host_data Указатель на буфер на host
     * @param size_bytes Размер данных в байтах
     */
    void Read(void* host_data, size_t size_bytes) const {
        if (size_bytes > size_bytes_) {
            throw std::runtime_error("GPUBuffer::Read: size exceeds buffer capacity");
        }
        backend_->MemcpyDeviceToHost(host_data, ptr_, size_bytes);
    }
    
    /**
     * @brief Прочитать в std::vector
     */
    std::vector<T> Read() const {
        std::vector<T> result(num_elements_);
        Read(result.data(), size_bytes_);
        return result;
    }
    
    /**
     * @brief Копировать данные из другого буфера (Device -> Device)
     */
    void CopyFrom(const GPUBuffer<T>& other) {
        if (other.GetSizeBytes() > size_bytes_) {
            throw std::runtime_error("GPUBuffer::CopyFrom: source buffer is too large");
        }
        backend_->MemcpyDeviceToDevice(ptr_, other.GetPtr(), other.GetSizeBytes());
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Информация о буфере
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить указатель на GPU память
     */
    void* GetPtr() const { return ptr_; }
    
    /**
     * @brief Получить количество элементов
     */
    size_t GetNumElements() const { return num_elements_; }
    
    /**
     * @brief Получить размер в байтах
     */
    size_t GetSizeBytes() const { return size_bytes_; }
    
    /**
     * @brief Проверить валидность буфера
     */
    bool IsValid() const { return ptr_ != nullptr && backend_ != nullptr; }

private:
    void* ptr_;              ///< Указатель на GPU память (hipDeviceptr_t или cl_mem void*, владеем)
    size_t num_elements_;    ///< Количество элементов типа T
    size_t size_bytes_;      ///< Кешировано: num_elements_ * sizeof(T), избегаем пересчёта
    IBackend* backend_;      ///< Не владеет — lifetime backend должен превышать lifetime буфера
};

} // namespace drv_gpu_lib
