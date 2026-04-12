#pragma once

/**
 * @file command_queue_pool.hpp
 * @brief Pool of OpenCL command queues - пул командных очередей OpenCL
 * 
 * CommandQueuePool управляет множеством cl_command_queue для:
 * - Параллельного выполнения команд на разных очередях
 * - Балансировки нагрузки между очередями
 * - Избежания блокировок при ожидании завершения операций
 * 
 * Архитектура:
 * - Thread-safe доступ к очередям через мьютекс
 * - Round-robin выбор очереди (циклический перебор)
 * - RAII освобождение ресурсов в деструкторе
 * 
 * Пример использования:
 * @code
 * // Создание пула с 4 очередями
 * CommandQueuePool pool;
 * pool.Initialize(context, device, 4);
 * 
 * // Получение очереди для параллельной работы
 * cl_command_queue q1 = pool.GetQueue(0);
 * cl_command_queue q2 = pool.GetQueue(1);
 * 
 * // Распараллеливание операций
 * clEnqueueNDRangeKernel(q1, kernel1, ...);  // Первая очередь
 * clEnqueueNDRangeKernel(q2, kernel2, ...);  // Вторая очередь
 * 
 * // Синхронизация всех очередей
 * pool.Synchronize();
 * @endcode
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include <CL/cl.h>

#include <memory>
#include <vector>
#include <mutex>

namespace drv_gpu_lib {

/**
 * @class CommandQueuePool
 * @brief Менеджер пула OpenCL командных очередей
 * 
 * Ответственность:
 * - Создание и управление несколькими cl_command_queue
 * - Thread-safe доступ к очередям
 * - Автоматическая очистка ресурсов (RAII)
 * 
 * Особенности:
 * - Ленивое создание очередей (только при Initialize)
 * - Round-robin распределение при GetQueue(index)
 * - Синхронизация всех очередей одним вызовом
 */
class CommandQueuePool {
public:
    /**
     * @brief Создать пустой CommandQueuePool
     */
    CommandQueuePool();
    
    /**
     * @brief Деструктор - освобождает все очереди
     */
    ~CommandQueuePool();
    
    /**
     * @brief Инициализировать пул очередями
     * @param context OpenCL контекст
     * @param device OpenCL устройство
     * @param num_queues Количество очередей (0 = авто, 2 по умолчанию)
     * @param device_index Индекс GPU (0, 1, 8, ...) для логов в DRVGPU_XX
     * @return true если успешно созданы хотя бы одна очередь
     */
    bool Initialize(cl_context context, cl_device_id device, size_t num_queues = 0, int device_index = 0);
    
    /**
     * @brief Очистить все очереди и освободить ресурсы
     */
    void Cleanup();
    
    /**
     * @brief Получить очередь по индексу
     * @param index Индекс очереди (0-based)
     * @return cl_command_queue или nullptr если индекс вне диапазона
     * 
     * Использует modulo для циклического доступа:
     * index % queue_count_
     */
    cl_command_queue GetQueue(size_t index = 0);
    
    /**
     * @brief Получить количество созданных очередей
     */
    size_t GetQueueCount() const;
    
    /**
     * @brief Синхронизировать все очереди
     * Ожидает завершения всех команд во всех очередях
     */
    void Synchronize();
    
private:
    std::vector<cl_command_queue> queues_;  ///< Список созданных очередей
    cl_context context_;                     ///< OpenCL контекст (не владеет)
    cl_device_id device_;                    ///< OpenCL устройство (не владеет)
    int device_index_;                       ///< Индекс GPU для логов (DRVGPU_XX)
    bool initialized_;                       ///< Флаг инициализации
    mutable std::mutex mutex_;               ///< Мьютекс для thread-safety
};

} // namespace drv_gpu_lib
