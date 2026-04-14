#pragma once

#if !ENABLE_ROCM  // OpenCL-only: ROCm uses UVA (hipMalloc) instead of SVM

/**
 * @file svm_buffer.hpp
 * @brief RAII обёртка для OpenCL SVM (Shared Virtual Memory)
 * 
 * Поддерживает:
 * - Coarse-Grained SVM (map/unmap required)
 * - Fine-Grained SVM (optional atomics)
 * - RAII для автоматического освобождения памяти
 * - Move semantics
 * 
 * @author Codo (AI Assistant)
 * @date 2026-01-19
 */

#include "i_memory_buffer.hpp"
#include "svm_capabilities.hpp"
#include "memory_type.hpp"
#include "../logger/logger.hpp"
#include <CL/cl.h>
#include <complex>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: SVMBuffer - RAII обёртка для SVM памяти
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class SVMBuffer
 * @brief RAII управление SVM памятью OpenCL
 * 
 * Особенности:
 * - Автоматическое освобождение через clSVMFree в деструкторе
 * - Map/Unmap для coarse-grained SVM
 * - Zero-copy операции где возможно
 * - Thread-safe (но не concurrent access!)
 * 
 * @code
 * SVMBuffer buffer(context, queue, 1024, MemoryStrategy::SVM_COARSE_GRAIN);
 * buffer.Write(data);  // Автоматический map/unmap внутри
 * auto result = buffer.Read();
 * @endcode
 */
class SVMBuffer : public IMemoryBuffer {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать SVM буфер
     * @param context OpenCL-контекст
     * @param queue Command queue для операций
     * @param num_elements Количество complex<float> элементов
     * @param strategy SVM стратегия (COARSE или FINE)
     * @param mem_type Тип памяти (READ_ONLY, WRITE_ONLY, READ_WRITE)
     * @throws std::runtime_error если SVM allocation failed
     */
    SVMBuffer(
        cl_context context,
        cl_command_queue queue,
        size_t num_elements,
        MemoryStrategy strategy = MemoryStrategy::SVM_COARSE_GRAIN,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    /**
     * @brief Создать SVM буфер с начальными данными
     */
    SVMBuffer(
        cl_context context,
        cl_command_queue queue,
        const ComplexVector& initial_data,
        MemoryStrategy strategy = MemoryStrategy::SVM_COARSE_GRAIN,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    // ═══════════════════════════════════════════════════════════════
    // Деструктор (RAII)
    // ═══════════════════════════════════════════════════════════════
    
    ~SVMBuffer() override;
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение move
    // ═══════════════════════════════════════════════════════════════
    
    SVMBuffer(const SVMBuffer&) = delete;
    SVMBuffer& operator=(const SVMBuffer&) = delete;
    
    SVMBuffer(SVMBuffer&& other) noexcept;
    SVMBuffer& operator=(SVMBuffer&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IMemoryBuffer интерфейса
    // ═══════════════════════════════════════════════════════════════
    
    // --- Чтение/Запись ---
    void Write(const ComplexVector& data) override;
    void WriteRaw(const void* data, size_t size_bytes) override;
    ComplexVector Read() override;
    ComplexVector ReadPartial(size_t num_elements) override;
    void ReadRaw(void* dest, size_t size_bytes) override;
    
    // --- Асинхронные операции ---
    cl_event WriteAsync(const ComplexVector& data) override;
    cl_event ReadAsync(ComplexVector& out_data) override;
    
    // --- OpenCL ресурсы ---
    // SVM не создаёт cl_mem объект — аллокация через clSVMAlloc возвращает void*, не cl_mem.
    // Kernel-аргументы SVM ставятся через clSetKernelArgSVMPointer, а не clSetKernelArg.
    cl_mem GetCLMem() const override { return nullptr; }
    void* GetSVMPointer() const override { return svm_ptr_; }
    void SetAsKernelArg(cl_kernel kernel, cl_uint arg_index) override;
    
    // --- Информация ---
    size_t GetNumElements() const override { return num_elements_; }
    size_t GetSizeBytes() const override { return size_bytes_; }
    MemoryType GetMemoryType() const override { return mem_type_; }
    MemoryStrategy GetStrategy() const override { return strategy_; }
    // SVM память выделяется нами через clSVMAlloc → мы владеем, освобождаем в FreeSVM().
    // External = false, потому что cl_mem адаптер (ExternalCLBufferAdapter) — внешний,
    // а SVMBuffer всегда создаёт и владеет своей памятью.
    bool IsExternal() const override { return false; }
    bool IsSVM() const override { return true; }
    BufferInfo GetInfo() const override;
    void PrintStats() const override;
    
    // --- SVM операции ---
    void Map(bool write = true, bool read = true) override;
    void Unmap() override;
    bool IsMapped() const override { return is_mapped_; }

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    cl_context       context_      = nullptr;
    cl_command_queue queue_        = nullptr;
    void*            svm_ptr_      = nullptr;
    size_t           num_elements_ = 0;
    size_t           size_bytes_   = 0;
    MemoryStrategy   strategy_     = MemoryStrategy::SVM_COARSE_GRAIN;
    MemoryType       mem_type_     = MemoryType::GPU_READ_WRITE;
    bool             is_mapped_    = false;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    void AllocateSVM();
    void FreeSVM();
    
    cl_svm_mem_flags GetSVMFlags() const;
    
    static void CheckCLError(cl_int err, const std::string& operation);
};

// ════════════════════════════════════════════════════════════════════════════
// Реализация (inline в заголовке или в .cpp)
// ════════════════════════════════════════════════════════════════════════════

inline SVMBuffer::SVMBuffer(
    cl_context context,
    cl_command_queue queue,
    size_t num_elements,
    MemoryStrategy strategy,
    MemoryType mem_type)
    : context_(context),
      queue_(queue),
      num_elements_(num_elements),
      size_bytes_(num_elements * sizeof(ComplexFloat)),
      strategy_(strategy),
      mem_type_(mem_type) {
    
    if (!context_ || !queue_) {
        throw std::invalid_argument("SVMBuffer: context and queue must not be null");
    }
    
    if (num_elements_ == 0) {
        throw std::invalid_argument("SVMBuffer: num_elements must be > 0");
    }
    
    AllocateSVM();
}

inline SVMBuffer::SVMBuffer(
    cl_context context,
    cl_command_queue queue,
    const ComplexVector& initial_data,
    MemoryStrategy strategy,
    MemoryType mem_type)
    : SVMBuffer(context, queue, initial_data.size(), strategy, mem_type) {
    
    Write(initial_data);
}

inline SVMBuffer::~SVMBuffer() {
    FreeSVM();
}

inline SVMBuffer::SVMBuffer(SVMBuffer&& other) noexcept
    : context_(other.context_),
      queue_(other.queue_),
      svm_ptr_(other.svm_ptr_),
      num_elements_(other.num_elements_),
      size_bytes_(other.size_bytes_),
      strategy_(other.strategy_),
      mem_type_(other.mem_type_),
      is_mapped_(other.is_mapped_) {
    
    // Инвалидируем источник
    other.svm_ptr_ = nullptr;
    other.is_mapped_ = false;
}

inline SVMBuffer& SVMBuffer::operator=(SVMBuffer&& other) noexcept {
    if (this != &other) {
        // Освобождаем текущие ресурсы
        FreeSVM();
        
        // Перемещаем из other
        context_      = other.context_;
        queue_        = other.queue_;
        svm_ptr_      = other.svm_ptr_;
        num_elements_ = other.num_elements_;
        size_bytes_   = other.size_bytes_;
        strategy_     = other.strategy_;
        mem_type_     = other.mem_type_;
        is_mapped_    = other.is_mapped_;
        
        // Инвалидируем источник
        other.svm_ptr_ = nullptr;
        other.is_mapped_ = false;
    }
    return *this;
}

inline void SVMBuffer::AllocateSVM() {
    cl_svm_mem_flags flags = GetSVMFlags();

    // clSVMAlloc: OpenCL 2.0+. Alignment=0 → реализация выбирает сама
    // (обычно 64 байта для векторных операций на GPU).
    // Возвращает nullptr при неудаче (не бросает, в отличие от clCreateBuffer).
    svm_ptr_ = clSVMAlloc(context_, flags, size_bytes_, 0);
    
    if (!svm_ptr_) {
        throw std::runtime_error(
            "SVMBuffer: clSVMAlloc failed for " + 
            std::to_string(size_bytes_) + " bytes"
        );
    }
}

inline void SVMBuffer::FreeSVM() {
    if (svm_ptr_) {
        // Unmap ПЕРЕД clSVMFree — обязательно для coarse-grain SVM.
        // clSVMFree на mapped память → undefined behavior (OpenCL spec 2.0, §5.6.1).
        // Для fine-grain Unmap() — no-op, но вызов безопасен.
        if (is_mapped_) {
            Unmap();
        }

        clSVMFree(context_, svm_ptr_);
        svm_ptr_ = nullptr;
    }
}

inline cl_svm_mem_flags SVMBuffer::GetSVMFlags() const {
    cl_svm_mem_flags flags = 0;
    
    // Базовые флаги по стратегии
    switch (strategy_) {
        case MemoryStrategy::SVM_FINE_GRAIN:
            flags = CL_MEM_SVM_FINE_GRAIN_BUFFER;
            break;
        case MemoryStrategy::SVM_FINE_SYSTEM:
            // OpenCL не имеет отдельного флага для Fine-Grain System —
            // он использует те же CL_MEM_SVM_FINE_GRAIN_BUFFER. Различие только
            // в том, что Fine-Grain System память физически единая с хостом (iGPU/APU).
            // Детектируется через CL_DEVICE_SVM_FINE_GRAIN_SYSTEM в SVMCapabilities.
            flags = CL_MEM_SVM_FINE_GRAIN_BUFFER;
            break;
        case MemoryStrategy::SVM_COARSE_GRAIN:
        default:
            // Coarse-grain: нет специальных флагов для clSVMAlloc.
            // Требует явного Map/Unmap для синхронизации видимости CPU↔GPU.
            flags = 0;
            break;
    }
    
    // Добавляем флаги чтения/записи
    switch (mem_type_) {
        case MemoryType::GPU_READ_ONLY:
            flags |= CL_MEM_READ_ONLY;
            break;
        case MemoryType::GPU_WRITE_ONLY:
            flags |= CL_MEM_WRITE_ONLY;
            break;
        case MemoryType::GPU_READ_WRITE:
        default:
            flags |= CL_MEM_READ_WRITE;
            break;
    }
    
    return flags;
}

inline void SVMBuffer::Map(bool write, bool read) {
    if (is_mapped_) {
        return;  // Уже отображён
    }
    
    // Fine-grained SVM: CPU может обращаться к памяти напрямую без map —
    // аппаратура сама поддерживает когерентность. Map → no-op (просто ставим флаг).
    if (strategy_ == MemoryStrategy::SVM_FINE_GRAIN ||
        strategy_ == MemoryStrategy::SVM_FINE_SYSTEM) {
        is_mapped_ = true;
        return;
    }

    // Coarse-grained требует явного map через OpenCL runtime.
    // До clEnqueueSVMMap CPU не должен читать/писать svm_ptr_ — UB!
    cl_map_flags map_flags = 0;
    if (write) map_flags |= CL_MAP_WRITE;
    if (read)  map_flags |= CL_MAP_READ;
    
    cl_int err = clEnqueueSVMMap(
        queue_,
        CL_TRUE,  // Blocking
        map_flags,
        svm_ptr_,
        size_bytes_,
        0, nullptr, nullptr
    );
    
    CheckCLError(err, "clEnqueueSVMMap");
    is_mapped_ = true;
}

inline void SVMBuffer::Unmap() {
    if (!is_mapped_) {
        return;  // Не отображён
    }
    
    // Fine-grained SVM не требует явного unmap
    if (strategy_ == MemoryStrategy::SVM_FINE_GRAIN || 
        strategy_ == MemoryStrategy::SVM_FINE_SYSTEM) {
        is_mapped_ = false;
        return;
    }
    
    // Coarse-grained требует явного unmap
    cl_int err = clEnqueueSVMUnmap(
        queue_,
        svm_ptr_,
        0, nullptr, nullptr
    );
    
    CheckCLError(err, "clEnqueueSVMUnmap");
    
    // Flush to ensure unmap completes
    clFlush(queue_);
    
    is_mapped_ = false;
}

inline void SVMBuffer::Write(const ComplexVector& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error(
            "SVMBuffer::Write: data size exceeds buffer capacity"
        );
    }
    
    WriteRaw(data.data(), data.size() * sizeof(ComplexFloat));
}

inline void SVMBuffer::WriteRaw(const void* data, size_t size_bytes) {
    if (size_bytes > size_bytes_) {
        throw std::runtime_error(
            "SVMBuffer::WriteRaw: size exceeds buffer capacity"
        );
    }
    
    bool was_mapped = is_mapped_;

    // Паттерн: временно маппируем если не был mapped.
    // Если caller уже сделал Map() — не трогаем (он управляет lifetime map-периода).
    if (!is_mapped_) {
        Map(true, false);  // map write-only: не нужно читать GPU данные
    }

    std::memcpy(svm_ptr_, data, size_bytes);

    if (!was_mapped) {
        Unmap();  // Unmap сигнализирует GPU что CPU завершил запись
    }
}

inline ComplexVector SVMBuffer::Read() {
    return ReadPartial(num_elements_);
}

inline ComplexVector SVMBuffer::ReadPartial(size_t num_elements) {
    if (num_elements > num_elements_) {
        throw std::runtime_error(
            "SVMBuffer::ReadPartial: requested elements exceed buffer size"
        );
    }
    
    ComplexVector result(num_elements);
    ReadRaw(result.data(), num_elements * sizeof(ComplexFloat));
    return result;
}

inline void SVMBuffer::ReadRaw(void* dest, size_t size_bytes) {
    if (size_bytes > size_bytes_) {
        throw std::runtime_error(
            "SVMBuffer::ReadRaw: size exceeds buffer capacity"
        );
    }
    
    bool was_mapped = is_mapped_;
    
    if (!is_mapped_) {
        Map(false, true);  // Отображение для чтения
    }
    
    // Прямое копирование памяти (без лишнего копирования)
    std::memcpy(dest, svm_ptr_, size_bytes);
    
    if (!was_mapped) {
        Unmap();
    }
}

inline cl_event SVMBuffer::WriteAsync(const ComplexVector& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error(
            "SVMBuffer::WriteAsync: data size exceeds buffer capacity"
        );
    }
    
    cl_event event = nullptr;
    
    // Для SVM используем clEnqueueSVMMemcpy
    cl_int err = clEnqueueSVMMemcpy(
        queue_,
        CL_FALSE,  // Неблокирующий режим
        svm_ptr_,
        data.data(),
        data.size() * sizeof(ComplexFloat),
        0, nullptr,
        &event
    );
    
    CheckCLError(err, "clEnqueueSVMMemcpy (write)");
    return event;
}

inline cl_event SVMBuffer::ReadAsync(ComplexVector& out_data) {
    if (out_data.size() < num_elements_) {
        out_data.resize(num_elements_);
    }
    
    cl_event event = nullptr;
    
    cl_int err = clEnqueueSVMMemcpy(
        queue_,
        CL_FALSE,  // Неблокирующий режим
        out_data.data(),
        svm_ptr_,
        num_elements_ * sizeof(ComplexFloat),
        0, nullptr,
        &event
    );
    
    CheckCLError(err, "clEnqueueSVMMemcpy (read)");
    return event;
}

inline void SVMBuffer::SetAsKernelArg(cl_kernel kernel, cl_uint arg_index) {
    cl_int err = clSetKernelArgSVMPointer(kernel, arg_index, svm_ptr_);
    CheckCLError(err, "clSetKernelArgSVMPointer");
}

inline BufferInfo SVMBuffer::GetInfo() const {
    BufferInfo info;
    info.num_elements = num_elements_;
    info.size_bytes   = size_bytes_;
    info.memory_type  = mem_type_;
    info.strategy     = strategy_;
    info.is_external  = false;
    info.is_mapped    = is_mapped_;
    return info;
}

inline void SVMBuffer::PrintStats() const {
    std::ostringstream ss;
    ss << "\n" << std::string(50, '-') << "\n";
    ss << "SVMBuffer Statistics\n";
    ss << std::string(50, '-') << "\n";
    ss << std::left << std::setw(20) << "Elements:" << num_elements_ << "\n";
    ss << std::left << std::setw(20) << "Size:"
       << std::fixed << std::setprecision(2)
       << (size_bytes_ / (1024.0 * 1024.0)) << " MB\n";
    ss << std::left << std::setw(20) << "Strategy:"
       << MemoryStrategyToString(strategy_) << "\n";
    ss << std::left << std::setw(20) << "Mapped:"
       << (is_mapped_ ? "YES" : "NO") << "\n";
    ss << std::left << std::setw(20) << "SVM Pointer:"
       << svm_ptr_ << "\n";
    ss << std::string(50, '-');
    DRVGPU_LOG_INFO("SVMBuffer", ss.str());
}

inline void SVMBuffer::CheckCLError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "OpenCL Error in " + operation + ": " + std::to_string(err)
        );
    }
}

} // namespace drv_gpu_lib

#endif  // !ENABLE_ROCM

