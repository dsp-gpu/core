#pragma once

/**
 * @file drv_gpu.hpp
 * @brief Главный класс DrvGPU - абстракция GPU устройства (Multi-Instance!)
 * 
 * ВАЖНО: DrvGPU НЕ является Singleton!
 * Для Multi-GPU используйте GPUManager (см. gpu_manager.hpp)
 * 
 * Архитектура:
 * - Backend Abstraction через IBackend интерфейс
 * - RAII управление ресурсами
 * - Потокобезопасные операции
 * - Поддержка OpenCL (расширяемо на CUDA/Vulkan)
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "interface/i_backend.hpp"
#include "common/backend_type.hpp"
#include "common/gpu_device_info.hpp"
#include "memory/memory_manager.hpp"
#include "module_registry.hpp"
#include <CL/cl.h>
#if ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif
#include <memory>
#include <string>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: DrvGPU - Главный класс библиотеки (НЕ Singleton!)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class DrvGPU
 * @brief Абстракция GPU устройства с поддержкой разных бэкендов
 * 
 * DrvGPU предоставляет единый интерфейс для работы с GPU через различные
 * бэкенды (OpenCL, CUDA, Vulkan). Класс НЕ является Singleton - вы можете
 * создать экземпляр для каждой GPU.
 * 
 * Для Multi-GPU сценариев используйте GPUManager:
 * @code
 * // Multi-GPU (правильный способ)
 * GPUManager manager;
 * manager.InitializeAll(BackendType::OPENCL);
 * auto gpu0 = manager.GetGPU(0);
 * auto gpu1 = manager.GetGPU(1);
 * 
 * // Single GPU (можно напрямую)
 * DrvGPU gpu(BackendType::OPENCL, 0);
 * @endcode
 * 
 * Основные возможности:
 * - Backend-агностичный интерфейс
 * - Управление памятью (MemoryManager)
 * - Регистр compute модулей (ModuleRegistry)
 * - RAII для автоматической очистки
 * - Thread-safe
 * 
 * Паттерны:
 * - Bridge Pattern (абстракция бэкенда)
 * - Facade Pattern (упрощённый интерфейс)
 * - RAII (автоматическое управление ресурсами)
 *
 * @ingroup grp_drvgpu
 */
class DrvGPU {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать DrvGPU для конкретного устройства
     * @param backend_type Тип бэкенда (OPENCL, CUDA, VULKAN)
     * @param device_index Индекс GPU устройства (0-based)
     * @throws std::runtime_error если устройство недоступно
     */
    explicit DrvGPU(BackendType backend_type, int device_index = 0);
    
    /**
     * @brief Деструктор (RAII - автоматическая очистка)
     */
    ~DrvGPU();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    
    DrvGPU(DrvGPU&& other) noexcept;
    DrvGPU& operator=(DrvGPU&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация и очистка
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать GPU
     * @throws std::runtime_error при ошибке инициализации
     */
    void Initialize();

    // ═══════════════════════════════════════════════════════════════
    // Static factory: External Context Integration
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Создать DrvGPU из внешнего OpenCL контекста
     *
     * OpenCL context/device/queue уже созданы вызывающим кодом.
     * DrvGPU НЕ освобождает их при уничтожении.
     *
     * @param device_index   Индекс GPU (для логирования и GetDeviceIndex())
     * @param context        Внешний cl_context
     * @param device         Внешний cl_device_id
     * @param queue          Внешняя cl_command_queue
     * @return               Готовый к работе DrvGPU (initialized=true)
     *
     * @code
     * cl_context ctx = ...; cl_device_id dev = ...; cl_command_queue q = ...;
     * auto gpu = DrvGPU::CreateFromExternalOpenCL(0, ctx, dev, q);
     * // gpu.Initialize() вызывать НЕ нужно — уже инициализирован
     * @endcode
     */
    static DrvGPU CreateFromExternalOpenCL(
        int device_index,
        cl_context context,
        cl_device_id device,
        cl_command_queue queue);

#if ENABLE_ROCM
    /**
     * @brief Создать DrvGPU из внешнего HIP stream
     *
     * hipStream_t уже создан вызывающим кодом (hipBLAS, hipFFT, MIOpen и т.п.).
     * DrvGPU НЕ вызывает hipStreamDestroy при уничтожении.
     *
     * @param device_index    Индекс AMD GPU (0..N-1)
     * @param stream          Внешний hipStream_t
     * @return                Готовый к работе DrvGPU (initialized=true)
     *
     * @code
     * hipStream_t s; hipStreamCreate(&s);
     * auto gpu = DrvGPU::CreateFromExternalROCm(0, s);
     * // gpu.Initialize() вызывать НЕ нужно — уже инициализирован
     * @endcode
     */
    static DrvGPU CreateFromExternalROCm(
        int device_index,
        hipStream_t stream);

    /**
     * @brief Создать HybridBackend DrvGPU из внешних OpenCL + HIP ресурсов
     *
     * Оба контекста уже существуют. DrvGPU НЕ освобождает ни один из них.
     * Поддерживает ZeroCopy между cl_mem и hipStream_t (общее VRAM AMD).
     *
     * @param device_index   Индекс GPU (одинаков для OpenCL и ROCm)
     * @param context        Внешний cl_context
     * @param device         Внешний cl_device_id
     * @param queue          Внешняя cl_command_queue
     * @param stream         Внешний hipStream_t
     * @return               Готовый к работе DrvGPU с HybridBackend (initialized=true)
     *
     * @code
     * auto gpu = DrvGPU::CreateFromExternalHybrid(0, cl_ctx, cl_dev, cl_q, hip_s);
     * auto& hybrid = static_cast<HybridBackend&>(gpu.GetBackend());
     * auto bridge = hybrid.CreateZeroCopyBridge(cl_buf, size);
     * @endcode
     */
    static DrvGPU CreateFromExternalHybrid(
        int device_index,
        cl_context context,
        cl_device_id device,
        cl_command_queue queue,
        hipStream_t stream);
#endif  // ENABLE_ROCM
    
    /**
     * @brief Проверить, инициализирован ли GPU
     */
    bool IsInitialized() const { return initialized_; }
    
    /**
     * @brief Очистить все ресурсы (вызывается автоматически в деструкторе)
     */
    void Cleanup();
    
    // ═══════════════════════════════════════════════════════════════
    // Информация об устройстве
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить информацию об устройстве
     */
    GPUDeviceInfo GetDeviceInfo() const;
    
    /**
     * @brief Получить индекс устройства
     */
    int GetDeviceIndex() const { return device_index_; }
    
    /**
     * @brief Получить тип бэкенда
     */
    BackendType GetBackendType() const { return backend_type_; }
    
    /**
     * @brief Получить название устройства
     */
    std::string GetDeviceName() const;
    
    /**
     * @brief Вывести информацию об устройстве
     */
    void PrintDeviceInfo() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Доступ к подсистемам
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить менеджер памяти
     */
    MemoryManager& GetMemoryManager();
    const MemoryManager& GetMemoryManager() const;
    
    /**
     * @brief Получить регистр модулей
     */
    ModuleRegistry& GetModuleRegistry();
    const ModuleRegistry& GetModuleRegistry() const;
    
    /**
     * @brief Получить бэкенд (для прямого доступа)
     * ВНИМАНИЕ: Используйте только если абстракции недостаточно!
     */
    IBackend& GetBackend();
    const IBackend& GetBackend() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Дождаться завершения всех операций на GPU
     */
    void Synchronize();
    
    /**
     * @brief Сброс буфера команд (без ожидания завершения)
     */
    void Flush();
    
    // ═══════════════════════════════════════════════════════════════
    // Статистика и отладка
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вывести статистику использования
     */
    void PrintStatistics() const;
    
    /**
     * @brief Получить строку со статистикой
     */
    std::string GetStatistics() const;
    
    /**
     * @brief Сбросить статистику
     */
    void ResetStatistics();

private:
    // ═══════════════════════════════════════════════════════════════
    // Приватный tagged constructor для static factory methods
    // ═══════════════════════════════════════════════════════════════

    // Тег для различения приватного конструктора от публичного.
    // Предотвращает случайный вызов: DrvGPU(ExternalInitTag{}, ...) невозможен снаружи класса.
    struct ExternalInitTag {};

    // Принимает уже инициализированный backend; не вызывает CreateBackend() / backend_->Initialize().
    // Вызывается только из static factory methods (CreateFromExternal*).
    DrvGPU(ExternalInitTag, BackendType type, int device_index,
           std::unique_ptr<IBackend> backend);

    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════

    BackendType backend_type_;
    int device_index_;
    bool initialized_;
    
    // Бэкенд (паттерн Bridge)
    std::unique_ptr<IBackend> backend_;
    
    // Подсистемы
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<ModuleRegistry> module_registry_;
    
    // Потокобезопасность
    mutable std::mutex mutex_;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать бэкенд на основе типа
     */
    void CreateBackend();
    
    /**
     * @brief Инициализировать подсистемы
     */
    void InitializeSubsystems();
};

} // namespace drv_gpu_lib
