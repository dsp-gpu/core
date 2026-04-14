#include <core/backends/opencl/command_queue_pool.hpp>
#include <core/backends/opencl/opencl_core.hpp>
#include <core/logger/logger.hpp>
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создать пустой пул очередей
 * 
 * Инициализирует все члены значениями по умолчанию:
 * - context_ = nullptr (будет установлен при Initialize)
 * - device_ = nullptr (будет установлен при Initialize)
 * - initialized_ = false (флаг готовности)
 */
CommandQueuePool::CommandQueuePool()
    : context_(nullptr),
      device_(nullptr),
      device_index_(0),
      initialized_(false) {
}

/**
 * @brief Деструктор - автоматическая очистка ресурсов
 * 
 * Вызывает Cleanup() для освобождения всех созданных очередей.
 * Это обеспечивает RAII - ресурсы освобождаются даже при исключениях.
 */
CommandQueuePool::~CommandQueuePool() {
    Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать пул командными очередями
 * 
 * Процесс инициализации:
 * 1. Проверяем, не инициализирован ли уже пул
 * 2. Сохраняем контекст и устройство
 * 3. Определяем количество очередей (0 -> 2 по умолчанию)
 * 4. Создаём указанное количество cl_command_queue
 * 5. Проверяем успешность создания хотя бы одной очереди
 * 
 * @param context OpenCL контекст для создания очередей
 * @param device OpenCL устройство
 * @param num_queues Количество очередей (0 = авто = 2)
 * @return true если создана хотя бы одна очередь, false иначе
 * 
 * @note Каждая очередь создаётся с флагами по умолчанию (0)
 * @note При ошибке создания одной очереди - пропускаем и пробуем следующую
 */
bool CommandQueuePool::Initialize(cl_context context, cl_device_id device, size_t num_queues, int device_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Если уже инициализирован - очищаем старые очереди
    if (initialized_) {
        Cleanup();
    }
    
    // Сохраняем параметры
    context_ = context;
    device_ = device;
    device_index_ = device_index;
    
    // Определяем количество очередей
    if (num_queues == 0) {
        num_queues = 2;  // Значение по умолчанию: 2 очереди
    }
    
    cl_int err;
    // Создаём очереди в цикле
    for (size_t i = 0; i < num_queues; ++i) {
        // clCreateCommandQueue - создаёт очередь команд для устройства
        // Параметры:
        //   context - контекст OpenCL
        //   device - устройство
        //   properties - свойства очереди (0 = по умолчанию)
        //   err - код ошибки
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        
        if (err != CL_SUCCESS) {
            // Логируем ошибку, но продолжаем создавать остальные очереди
            DRVGPU_LOG_ERROR_GPU(device_index_, "CommandQueuePool", "Failed to create command queue: " + std::to_string(err));
            continue;  // Пропускаем эту очередь
        }
        
        // Успешно создана - добавляем в пул
        queues_.push_back(queue);
    }
    
    // Устанавливаем флаг инициализации если есть хотя бы одна очередь
    initialized_ = !queues_.empty();
    return initialized_;
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Освободить все ресурсы пула
 * 
 * Процесс очистки:
 * 1. Захватываем мьютекс для thread-safety
 * 2. Для каждой очереди вызываем clReleaseCommandQueue()
 * 3. Очищаем вектор очередей
 * 4. Сбрасываем флаг инициализации
 * 
 * @note Безопасно вызывать даже если пул не был инициализирован
 */
void CommandQueuePool::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Освобождаем каждую очередь
    for (auto& queue : queues_) {
        if (queue) {
            // clReleaseCommandQueue - уменьшает счётчик ссылок очереди
            // Если счётчик достиг 0 - очередь удаляется
            clReleaseCommandQueue(queue);
        }
    }
    
    // Очищаем вектор (освобождает память)
    queues_.clear();
    
    // Сбрасываем флаг
    initialized_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// Доступ к очередям
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить очередь по индексу
 * 
 * Использует modulo для циклического доступа:
 * Если запрошен индекс 5, а всего 4 очереди - вернёт очередь 1 (5 % 4 = 1)
 * Это обеспечивает round-robin распределение нагрузки.
 * 
 * @param index Индекс очереди (0-based)
 * @return cl_command_queue или nullptr если очередей нет
 * 
 * @code
 * // Пример: параллельное выполнение на разных очередях
 * cl_command_queue q1 = pool.GetQueue(0);  // Первая очередь
 * cl_command_queue q2 = pool.GetQueue(1);  // Вторая очередь
 * 
 * // Эти два вызова будут использовать разные очереди
 * clEnqueueNDRangeKernel(q1, kernel1, ...);
 * clEnqueueNDRangeKernel(q2, kernel2, ...);
 * @endcode
 */
cl_command_queue CommandQueuePool::GetQueue(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Проверяем наличие очередей
    if (queues_.empty()) {
        return nullptr;
    }
    
    // Round-robin: index % queue_count
    return queues_[index % queues_.size()];
}

/**
 * @brief Получить количество созданных очередей
 * @return Количество очередей в пуле (0 если не инициализирован)
 */
size_t CommandQueuePool::GetQueueCount() const {
    return queues_.size();
}

// ════════════════════════════════════════════════════════════════════════════
// Синхронизация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Синхронизировать все очереди
 * 
 * Ожидает завершения всех команд во всех очередях пула.
 * Использует clFinish() для каждой очереди.
 * 
 * @code
 * // Пример: асинхронная работа с последующей синхронизацией
 * cl_command_queue q1 = pool.GetQueue(0);
 * cl_command_queue q2 = pool.GetQueue(1);
 * 
 * // Запускаем kernel'ы асинхронно
 * clEnqueueNDRangeKernel(q1, kernel1, ...);  // Не ждём завершения
 * clEnqueueNDRangeKernel(q2, kernel2, ...);  // Не ждём завершения
 * 
 * // Ждём завершения всех операций
 * pool.Synchronize();  // Блокирует до полного завершения
 * 
 * // Теперь результаты гарантированно готовы
 * @endcode
 * 
 * @note Это блокирующая операция (ожидает завершения всех команд)
 */
void CommandQueuePool::Synchronize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Для каждой очереди вызываем clFinish()
    for (auto& queue : queues_) {
        if (queue) {
            // clFinish - ожидает завершения всех команд в очереди
            clFinish(queue);
        }
    }
}

} // namespace drv_gpu_lib
