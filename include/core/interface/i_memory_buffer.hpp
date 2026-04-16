#pragma once

/**
 * @file i_memory_buffer.hpp
 * @brief Абстрактный интерфейс для GPU буферов
 * 
 * Определяет общий интерфейс для всех типов буферов:
 * - RegularBuffer (традиционный cl_mem)
 * - SVMBuffer (Shared Virtual Memory)
 * - HybridBuffer (автовыбор стратегии)
 * 
 * @author Codo (AI Assistant)
 * @date 2026-01-19
 */

#include <core/memory/svm_capabilities.hpp>
#include <core/memory/memory_type.hpp>
#include <CL/cl.h>
#include <complex>
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace drv_gpu_lib {

// Предварительные объявления
class IMemoryBuffer;

// ════════════════════════════════════════════════════════════════════════════
// Псевдонимы типов
// ════════════════════════════════════════════════════════════════════════════

using ComplexFloat = std::complex<float>;
using ComplexVector = std::vector<ComplexFloat>;

// ════════════════════════════════════════════════════════════════════════════
// Struct: BufferInfo - информация о буфере
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct BufferInfo
 * @brief Информация о буфере для диагностики
 */
struct BufferInfo {
    size_t          num_elements   = 0;
    size_t          size_bytes     = 0;
    MemoryType      memory_type    = MemoryType::GPU_READ_WRITE;
    MemoryStrategy  strategy       = MemoryStrategy::REGULAR_BUFFER;
    bool            is_external    = false;
    bool            is_mapped      = false;
    
    std::string ToString() const;
};

// ════════════════════════════════════════════════════════════════════════════
// Interface: IMemoryBuffer - абстрактный интерфейс для GPU буферов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class IMemoryBuffer
 * @brief Абстрактный интерфейс для работы с GPU памятью
 * 
 * Все реализации буферов должны наследовать этот интерфейс.
 * Это позволяет использовать разные стратегии (SVM/Regular) 
 * через единый полиморфный интерфейс.
 * 
 * Паттерн: Strategy Pattern + RAII
 * 
 * @code
 * std::unique_ptr<IMemoryBuffer> buffer = factory.CreateBuffer(size);
 * buffer->Write(data);
 * kernel.SetArg(0, buffer.get());
 * // ... kernel execution ...
 * auto result = buffer->Read();
 * @endcode
 */
class IMemoryBuffer {
public:
    // ═══════════════════════════════════════════════════════════════
    // Виртуальный деструктор (RAII)
    // ═══════════════════════════════════════════════════════════════
    
    virtual ~IMemoryBuffer() = default;
    
    // ═══════════════════════════════════════════════════════════════
    // Основные операции чтения/записи
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Записать данные в буфер (синхронно)
     * @param data Вектор данных для записи
     * @throws std::runtime_error если размер превышает буфер
     */
    virtual void Write(const ComplexVector& data) = 0;
    
    /**
     * @brief Записать raw данные в буфер
     * @param data Указатель на данные
     * @param size_bytes Размер в байтах
     */
    virtual void WriteRaw(const void* data, size_t size_bytes) = 0;
    
    /**
     * @brief Прочитать все данные из буфера (синхронно)
     * @return Вектор данных
     */
    virtual ComplexVector Read() = 0;
    
    /**
     * @brief Прочитать часть данных из буфера
     * @param num_elements Количество элементов для чтения
     * @return Вектор данных
     */
    virtual ComplexVector ReadPartial(size_t num_elements) = 0;
    
    /**
     * @brief Прочитать raw данные из буфера
     * @param dest Указатель на приёмник
     * @param size_bytes Размер в байтах
     */
    virtual void ReadRaw(void* dest, size_t size_bytes) = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Асинхронные операции
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Записать данные асинхронно
     * @param data Вектор данных
     * @return cl_event для синхронизации
     */
    virtual cl_event WriteAsync(const ComplexVector& data) = 0;
    
    /**
     * @brief Прочитать данные асинхронно
     * @param out_data Указатель на выходной вектор (должен быть выделен!)
     * @return cl_event для синхронизации
     */
    virtual cl_event ReadAsync(ComplexVector& out_data) = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Доступ к OpenCL ресурсам
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить cl_mem handle (для traditional buffers)
     * @return cl_mem или nullptr для SVM
     */
    virtual cl_mem GetCLMem() const = 0;
    
    /**
     * @brief Получить SVM указатель (для SVM buffers)
     * @return void* или nullptr для traditional
     */
    virtual void* GetSVMPointer() const = 0;
    
    /**
     * @brief Установить как аргумент kernel
     * @param kernel OpenCL kernel
     * @param arg_index Индекс аргумента
     */
    virtual void SetAsKernelArg(cl_kernel kernel, cl_uint arg_index) = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Информация о буфере
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество элементов
     */
    virtual size_t GetNumElements() const = 0;
    
    /**
     * @brief Получить размер в байтах
     */
    virtual size_t GetSizeBytes() const = 0;
    
    /**
     * @brief Получить тип памяти (READ_ONLY, WRITE_ONLY, READ_WRITE)
     */
    virtual MemoryType GetMemoryType() const = 0;
    
    /**
     * @brief Получить стратегию памяти (REGULAR, SVM_COARSE, etc.)
     */
    virtual MemoryStrategy GetStrategy() const = 0;
    
    /**
     * @brief Проверить, является ли буфер внешним (non-owning)
     */
    virtual bool IsExternal() const = 0;
    
    /**
     * @brief Проверить, использует ли буфер SVM
     */
    virtual bool IsSVM() const = 0;
    
    /**
     * @brief Получить полную информацию о буфере
     */
    virtual BufferInfo GetInfo() const = 0;
    
    /**
     * @brief Вывести статистику буфера
     */
    virtual void PrintStats() const = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // SVM-специфичные операции (no-op для traditional buffers)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Map буфер для доступа с хоста (SVM)
     * @param write Разрешить запись
     * @param read Разрешить чтение
     */
    virtual void Map(bool write = true, bool read = true) = 0;
    
    /**
     * @brief Unmap буфер (SVM)
     */
    virtual void Unmap() = 0;
    
    /**
     * @brief Проверить, замаплен ли буфер
     */
    virtual bool IsMapped() const = 0;

protected:
    // Защищённый конструктор (только для наследников)
    IMemoryBuffer() = default;
    
    // Запрет копирования
    IMemoryBuffer(const IMemoryBuffer&) = delete;
    IMemoryBuffer& operator=(const IMemoryBuffer&) = delete;
};

// ════════════════════════════════════════════════════════════════════════════
// Inline реализация BufferInfo::ToString
// ════════════════════════════════════════════════════════════════════════════

inline std::string BufferInfo::ToString() const {
    std::ostringstream oss;
    oss << "BufferInfo:\n";
    oss << "  Elements:   " << num_elements << "\n";
    oss << "  Size:       " << (size_bytes / (1024.0 * 1024.0)) << " MB\n";
    oss << "  Strategy:   " << MemoryStrategyToString(strategy) << "\n";
    oss << "  External:   " << (is_external ? "YES" : "NO") << "\n";
    oss << "  Mapped:     " << (is_mapped ? "YES" : "NO") << "\n";
    return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// RAII Guard для Map/Unmap (SVM)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ScopedMap
 * @brief RAII guard для автоматического unmap SVM буфера
 * 
 * @code
 * {
 *     ScopedMap guard(buffer);
 *     // Работаем с buffer->GetSVMPointer()
 * } // Автоматический unmap
 * @endcode
 */
class ScopedMap {
public:
    explicit ScopedMap(IMemoryBuffer* buffer, bool write = true, bool read = true)
        : buffer_(buffer) {
        if (buffer_ && buffer_->IsSVM()) {
            buffer_->Map(write, read);
        }
    }
    
    ~ScopedMap() {
        if (buffer_ && buffer_->IsSVM() && buffer_->IsMapped()) {
            buffer_->Unmap();
        }
    }
    
    // Запрет копирования
    ScopedMap(const ScopedMap&) = delete;
    ScopedMap& operator=(const ScopedMap&) = delete;
    
    // Move разрешён
    ScopedMap(ScopedMap&& other) noexcept : buffer_(other.buffer_) {
        other.buffer_ = nullptr;
    }
    
    ScopedMap& operator=(ScopedMap&& other) noexcept {
        if (this != &other) {
            if (buffer_ && buffer_->IsSVM() && buffer_->IsMapped()) {
                buffer_->Unmap();
            }
            buffer_ = other.buffer_;
            other.buffer_ = nullptr;
        }
        return *this;
    }

private:
    IMemoryBuffer* buffer_;
};

} // namespace drv_gpu_lib

