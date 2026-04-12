#pragma once

/**
 * @file external_cl_buffer_adapter.hpp
 * @brief Адаптер для работы с ВНЕШНИМИ cl_mem буферами
 * 
 * КЛЮЧЕВАЯ ФУНКЦИОНАЛЬНОСТЬ: Загрузка/Выгрузка данных из внешних OpenCL буферов
 * 
 * Сценарий: обмен данными между DrvGPU и вашим существующим OpenCL-кодом
 * 
 * @author DrvGPU Team
 * @date 2026-02-01
 */

#include <CL/cl.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <memory>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: ExternalCLBufferAdapter - адаптер для внешних cl_mem буферов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ExternalCLBufferAdapter
 * @brief RAII адаптер для работы с внешними cl_mem буферами
 * 
 * @tparam T Тип элементов в буфере (float, int, double, etc.)
 * 
 * ОСОБЕННОСТИ:
 * - НЕ владеет cl_mem буфером (по умолчанию)
 * - Типобезопасность через шаблоны
 * - Простые методы Read() / Write()
 * - RAII для опционального владения
 * 
 * Пример использования:
 * @code
 * // У вас есть cl_mem буфер из другого класса
 * cl_mem your_buffer = external_class->GetBuffer();
 * cl_command_queue your_queue = external_class->GetQueue();
 * 
 * // Создаем адаптер (тип float, 1024 элемента)
 * ExternalCLBufferAdapter<float> adapter(your_buffer, 1024, your_queue);
 * 
 * // ЗАГРУЗИТЬ данные с GPU -> Host
 * std::vector<float> data_from_gpu = adapter.Read();
 * 
 * // Обработать данные на CPU
 * for (auto& val : data_from_gpu) {
 *     val *= 2.0f;
 * }
 * 
 * // ВЫГРУЗИТЬ обработанные данные Host -> GPU
 * adapter.Write(data_from_gpu);
 * 
 * // Адаптер НЕ уничтожит your_buffer (owns_buffer = false по умолчанию)
 * @endcode
 */
template<typename T>
class ExternalCLBufferAdapter {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать адаптер для внешнего cl_mem буфера
     * @param external_buffer Внешний cl_mem (ваш существующий буфер)
     * @param num_elements Количество элементов типа T в буфере
     * @param queue Command queue для операций чтения/записи
     * @param owns_buffer false (по умолчанию) - НЕ владеет буфером
     * 
     * ВАЖНО: Если owns_buffer = false, буфер НЕ будет уничтожен в деструкторе!
     */
    ExternalCLBufferAdapter(
        cl_mem external_buffer,
        size_t num_elements,
        cl_command_queue queue,
        bool owns_buffer = false
    );

    /**
     * @brief Деструктор (RAII)
     * Уничтожает cl_mem ТОЛЬКО если owns_buffer = true
     */
    ~ExternalCLBufferAdapter();

    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    ExternalCLBufferAdapter(const ExternalCLBufferAdapter&) = delete;
    ExternalCLBufferAdapter& operator=(const ExternalCLBufferAdapter&) = delete;

    ExternalCLBufferAdapter(ExternalCLBufferAdapter&& other) noexcept;
    ExternalCLBufferAdapter& operator=(ExternalCLBufferAdapter&& other) noexcept;

    // ═══════════════════════════════════════════════════════════════
    // Операции чтения/записи (КЛЮЧЕВАЯ ФУНКЦИОНАЛЬНОСТЬ)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief ЗАГРУЗИТЬ все данные с GPU -> Host (синхронно)
     * @return std::vector с данными из GPU
     * @throws std::runtime_error при ошибке OpenCL
     * 
     * Сценарий: чтение результатов обработки на GPU
     * @code
     * // GPU обработал данные, теперь читаем результат
     * std::vector<float> result = adapter.Read();
     * 
     * // Анализируем на CPU
     * float sum = std::accumulate(result.begin(), result.end(), 0.0f);
     * @endcode
     */
    std::vector<T> Read();

    /**
     * @brief ЗАГРУЗИТЬ часть данных с GPU -> Host
     * @param num_elements Количество элементов для чтения
     * @return std::vector с частью данных
     */
    std::vector<T> ReadPartial(size_t num_elements);

    /**
     * @brief ЗАГРУЗИТЬ данные в существующий буфер
     * @param host_dest Указатель на буфер CPU (должен быть выделен!)
     * @param num_elements Количество элементов для чтения
     */
    void ReadTo(T* host_dest, size_t num_elements);

    /**
     * @brief ВЫГРУЗИТЬ данные с Host -> GPU (синхронно)
     * @param data Вектор данных для записи
     * @throws std::runtime_error если data.size() > num_elements_
     * 
     * Сценарий: передача обработанных данных обратно на GPU
     * @code
     * // Подготовили данные на CPU
     * std::vector<float> processed_data(1024);
     * // ... заполнить данные ...
     * 
     * // Отправляем на GPU
     * adapter.Write(processed_data);
     * @endcode
     */
    void Write(const std::vector<T>& data);

    /**
     * @brief ВЫГРУЗИТЬ данные из raw указателя
     * @param host_data Указатель на данные CPU
     * @param num_elements Количество элементов для записи
     */
    void WriteFrom(const T* host_data, size_t num_elements);

    /**
     * @brief Асинхронное чтение (возвращает event)
     * @param out_data Выходной вектор (будет изменен размер)
     * @return cl_event для синхронизации
     */
    cl_event ReadAsync(std::vector<T>& out_data);

    /**
     * @brief Асинхронная запись (возвращает event)
     * @param data Данные для записи
     * @return cl_event для синхронизации
     */
    cl_event WriteAsync(const std::vector<T>& data);

    // ═══════════════════════════════════════════════════════════════
    // Информация о буфере
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество элементов
     */
    size_t GetNumElements() const { return num_elements_; }

    /**
     * @brief Получить размер в байтах
     */
    size_t GetSizeBytes() const { return size_bytes_; }

    /**
     * @brief Получить cl_mem хэндл
     */
    cl_mem GetCLMem() const { return buffer_; }

    /**
     * @brief Проверить, владеет ли адаптер буфером
     */
    bool OwnsBuffer() const { return owns_buffer_; }

    /**
     * @brief Синхронизировать очередь (дождаться завершения операций)
     */
    void Synchronize();

    /**
     * @brief Flush очереди
     */
    void Flush();

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    cl_mem buffer_;              ///< Внешний cl_mem буфер
    size_t num_elements_;        ///< Количество элементов типа T
    size_t size_bytes_;          ///< Размер в байтах
    cl_command_queue queue_;     ///< Command queue для операций
    bool owns_buffer_;           ///< Владеет ли адаптер буфером

    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Проверка OpenCL ошибок
     */
    static void CheckCLError(cl_int err, const std::string& operation);
};

// ════════════════════════════════════════════════════════════════════════════
// Шаблонная реализация (inline для header-only)
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
ExternalCLBufferAdapter<T>::ExternalCLBufferAdapter(
    cl_mem external_buffer,
    size_t num_elements,
    cl_command_queue queue,
    bool owns_buffer)
    : buffer_(external_buffer)
    , num_elements_(num_elements)
    , size_bytes_(num_elements * sizeof(T))
    , queue_(queue)
    , owns_buffer_(owns_buffer)
{
    if (!buffer_) {
        throw std::invalid_argument("ExternalCLBufferAdapter: buffer is null");
    }

    if (!queue_) {
        throw std::invalid_argument("ExternalCLBufferAdapter: queue is null");
    }

    if (num_elements_ == 0) {
        throw std::invalid_argument("ExternalCLBufferAdapter: num_elements must be > 0");
    }

    std::cout << "[ExternalCLBufferAdapter] Created adapter for " 
              << num_elements_ << " elements (" 
              << (size_bytes_ / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "[ExternalCLBufferAdapter] Owns buffer: " 
              << (owns_buffer_ ? "YES" : "NO") << "\n";
}

template<typename T>
ExternalCLBufferAdapter<T>::~ExternalCLBufferAdapter() {
    if (owns_buffer_ && buffer_) {
        std::cout << "[ExternalCLBufferAdapter] Releasing owned cl_mem buffer\n";
        clReleaseMemObject(buffer_);
    }
    buffer_ = nullptr;
}

template<typename T>
ExternalCLBufferAdapter<T>::ExternalCLBufferAdapter(ExternalCLBufferAdapter&& other) noexcept
    : buffer_(other.buffer_)
    , num_elements_(other.num_elements_)
    , size_bytes_(other.size_bytes_)
    , queue_(other.queue_)
    , owns_buffer_(other.owns_buffer_)
{
    other.buffer_ = nullptr;
    other.owns_buffer_ = false;
}

template<typename T>
ExternalCLBufferAdapter<T>& ExternalCLBufferAdapter<T>::operator=(
    ExternalCLBufferAdapter&& other) noexcept
{
    if (this != &other) {
        // Освободить текущий буфер если владеем
        if (owns_buffer_ && buffer_) {
            clReleaseMemObject(buffer_);
        }

        // Переместить ресурсы
        buffer_ = other.buffer_;
        num_elements_ = other.num_elements_;
        size_bytes_ = other.size_bytes_;
        queue_ = other.queue_;
        owns_buffer_ = other.owns_buffer_;

        // Инвалидируем источник
        other.buffer_ = nullptr;
        other.owns_buffer_ = false;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// ЗАГРУЗИТЬ данные с GPU -> Host
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
std::vector<T> ExternalCLBufferAdapter<T>::Read() {
    std::vector<T> result(num_elements_);
    ReadTo(result.data(), num_elements_);
    return result;
}

template<typename T>
std::vector<T> ExternalCLBufferAdapter<T>::ReadPartial(size_t num_elements) {
    if (num_elements > num_elements_) {
        throw std::runtime_error("ReadPartial: requested elements exceed buffer size");
    }

    std::vector<T> result(num_elements);
    ReadTo(result.data(), num_elements);
    return result;
}

template<typename T>
void ExternalCLBufferAdapter<T>::ReadTo(T* host_dest, size_t num_elements) {
    if (!host_dest) {
        throw std::invalid_argument("ReadTo: host_dest is null");
    }

    if (num_elements > num_elements_) {
        throw std::runtime_error("ReadTo: requested elements exceed buffer size");
    }

    // Синхронное чтение GPU -> Host
    cl_int err = clEnqueueReadBuffer(
        queue_,
        buffer_,
        CL_TRUE,                        // блокирующий режим
        0,                              // смещение
        num_elements * sizeof(T),       // размер
        host_dest,                      // указатель приёмника
        0,                              // num_events_in_wait_list
        nullptr,                        // event_wait_list
        nullptr                         // event
    );

    CheckCLError(err, "ReadTo (clEnqueueReadBuffer)");
}

// ════════════════════════════════════════════════════════════════════════════
// ВЫГРУЗИТЬ данные с Host -> GPU
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
void ExternalCLBufferAdapter<T>::Write(const std::vector<T>& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error(
            "Write: data size (" + std::to_string(data.size()) + 
            ") exceeds buffer capacity (" + std::to_string(num_elements_) + ")"
        );
    }

    WriteFrom(data.data(), data.size());
}

template<typename T>
void ExternalCLBufferAdapter<T>::WriteFrom(const T* host_data, size_t num_elements) {
    if (!host_data) {
        throw std::invalid_argument("WriteFrom: host_data is null");
    }

    if (num_elements > num_elements_) {
        throw std::runtime_error("WriteFrom: requested elements exceed buffer size");
    }

    // Синхронная запись Host -> GPU
    cl_int err = clEnqueueWriteBuffer(
        queue_,
        buffer_,
        CL_TRUE,                        // блокирующий режим
        0,                              // смещение
        num_elements * sizeof(T),       // размер
        host_data,                      // указатель источника
        0,                              // num_events_in_wait_list
        nullptr,                        // event_wait_list
        nullptr                         // event
    );

    CheckCLError(err, "WriteFrom (clEnqueueWriteBuffer)");
}

// ════════════════════════════════════════════════════════════════════════════
// Асинхронные операции
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
cl_event ExternalCLBufferAdapter<T>::ReadAsync(std::vector<T>& out_data) {
    if (out_data.size() < num_elements_) {
        out_data.resize(num_elements_);
    }

    cl_event event = nullptr;
    cl_int err = clEnqueueReadBuffer(
        queue_,
        buffer_,
        CL_FALSE,                       // неблокирующий режим
        0,
        num_elements_ * sizeof(T),
        out_data.data(),
        0,
        nullptr,
        &event
    );

    CheckCLError(err, "ReadAsync (clEnqueueReadBuffer)");
    return event;
}

template<typename T>
cl_event ExternalCLBufferAdapter<T>::WriteAsync(const std::vector<T>& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error("WriteAsync: data size exceeds buffer capacity");
    }

    cl_event event = nullptr;
    cl_int err = clEnqueueWriteBuffer(
        queue_,
        buffer_,
        CL_FALSE,                       // неблокирующий режим
        0,
        data.size() * sizeof(T),
        data.data(),
        0,
        nullptr,
        &event
    );

    CheckCLError(err, "WriteAsync (clEnqueueWriteBuffer)");
    return event;
}

// ════════════════════════════════════════════════════════════════════════════
// Утилиты
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
void ExternalCLBufferAdapter<T>::Synchronize() {
    if (queue_) {
        clFinish(queue_);
    }
}

template<typename T>
void ExternalCLBufferAdapter<T>::Flush() {
    if (queue_) {
        clFlush(queue_);
    }
}

template<typename T>
void ExternalCLBufferAdapter<T>::CheckCLError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "OpenCL Error in " + operation + ": error code " + std::to_string(err)
        );
    }
}

} // namespace drv_gpu_lib
