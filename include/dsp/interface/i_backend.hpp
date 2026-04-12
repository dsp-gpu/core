#pragma once

/**
 * @file i_backend.hpp
 * @brief Абстрактный интерфейс для бэкендов (OpenCL, CUDA, ROCm)
 * 
 * IBackend - ключевая абстракция в DrvGPU, реализующая Bridge Pattern.
 * Позволяет переключаться между бэкендами без изменения клиентского кода.
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "backend_type.hpp"
#include "gpu_device_info.hpp"

#include <string>
#include <cstddef>

namespace drv_gpu_lib {
class MemoryManager;

// ════════════════════════════════════════════════════════════════════════════
// Interface: IBackend - абстракция GPU бэкенда
// ════════════════════════════════════════════════════════════════════════════

/**
 * @interface IBackend
 * @brief Абстрактный интерфейс для всех GPU бэкендов
 * 
 * Каждый бэкенд (OpenCL, CUDA, ROCm) реализует этот интерфейс,
 * предоставляя единообразный API для DrvGPU.
 * 
 * Паттерн: Bridge (отделяет абстракцию от реализации)
 * 
 * Основные методы:
 * - Initialize/Cleanup - жизненный цикл
 * - GetNativeHandle - доступ к нативным объектам
 * - Allocate/Free - управление памятью
 * - Synchronize/Flush - синхронизация
 * 
 * Реализации:
 * - OpenCLBackend (см. opencl_backend.hpp)
 * - CUDABackend (будущее)
 * - VulkanBackend (будущее)
 *
 * @ingroup grp_drvgpu
 */
class IBackend {
public:
    virtual ~IBackend() = default;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Инициализация и жизненный цикл
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать бэкенд для конкретного устройства
     * @param device_index Индекс GPU устройства
     * @throws std::runtime_error при ошибке инициализации
     */
    virtual void Initialize(int device_index) = 0;
    
    /**
     * @brief Проверить, инициализирован ли бэкенд
     */
    virtual bool IsInitialized() const = 0;
    
    /**
     * @brief Очистить ресурсы бэкенда
     * 
     * ✅ ВАЖНО: Учитывает владение ресурсами (owns_resources_).
     * Если backend создал ресурсы сам - освобождает их.
     * Если ресурсы пришли извне - только обнуляет указатели.
     */
    virtual void Cleanup() = 0;
    
    // ═══════════════════════════════════════════════════════════════════════
    // ✅ НОВОЕ: Управление владением ресурсами (для внешней интеграции)
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Установить режим владения ресурсами
     * 
     * @param owns true = backend создал ресурсы сам и должен их освободить
     *             false = ресурсы пришли извне, backend только использует их
     * 
     * Примеры использования:
     * 
     * @code
     * // Сценарий 1: Backend создаёт контекст сам
     * auto backend = std::make_unique<OpenCLBackend>();
     * backend->Initialize(0);  // owns_resources_ = true (по умолчанию)
     * // Backend освободит контекст при Cleanup()
     * 
     * // Сценарий 2: Используем внешний контекст
     * auto backend = std::make_unique<OpenCLBackendExternal>();
     * backend->InitializeFromExternalContext(ctx, dev, queue);
     * // owns_resources_ = false автоматически
     * // Backend НЕ освободит контекст при Cleanup()
     * 
     * // Сценарий 3: Явное управление
     * backend->SetOwnsResources(false);  // Принудительно non-owning
     * @endcode
     */
    virtual void SetOwnsResources(bool owns) = 0;
    
    /**
     * @brief Проверить, владеет ли backend ресурсами
     * 
     * @return true если backend создал ресурсы и освободит их при Cleanup()
     *         false если ресурсы внешние и backend их не освобождает
     * 
     * Используется для отладки и проверки корректности интеграции:
     * 
     * @code
     * if (backend->OwnsResources()) {
     *     std::cout << "Backend will release resources\n";
     * } else {
     *     std::cout << "External code must release resources\n";
     * }
     * @endcode
     */
    virtual bool OwnsResources() const = 0;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Информация об устройстве
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить тип бэкенда
     */
    virtual BackendType GetType() const = 0;
    
    /**
     * @brief Получить информацию об устройстве
     */
    virtual GPUDeviceInfo GetDeviceInfo() const = 0;
    
    /**
     * @brief Получить индекс устройства
     */
    virtual int GetDeviceIndex() const = 0;
    
    /**
     * @brief Получить название устройства
     */
    virtual std::string GetDeviceName() const = 0;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Нативные хэндлы (для прямого доступа к API)
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить нативный context
     * OpenCL: возвращает cl_context
     * CUDA: возвращает CUcontext
     * Vulkan: возвращает VkDevice
     * ROCm: возвращает hipCtx_t
     */
    virtual void* GetNativeContext() const = 0;
    
    /**
     * @brief Получить нативный device
     * OpenCL: возвращает cl_device_id
     * CUDA: возвращает CUdevice
     * Vulkan: возвращает VkPhysicalDevice
     * ROCm: возвращает hipDevice_t
     */
    virtual void* GetNativeDevice() const = 0;
    
    /**
     * @brief Получить нативную command queue/stream
     * OpenCL: возвращает cl_command_queue
     * CUDA: возвращает CUstream
     * Vulkan: возвращает VkQueue
     * ROCm: возвращает hipStream_t
     */
    virtual void* GetNativeQueue() const = 0;
    
    /**
     * @brief Получить менеджер памяти (для создания буферов)
     * @return Указатель на MemoryManager или nullptr если не поддерживается
     */
    virtual MemoryManager* GetMemoryManager() { return nullptr; }
    virtual const MemoryManager* GetMemoryManager() const { return nullptr; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Управление памятью (базовые операции)
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Выделить память на GPU
     * @param size_bytes Размер в байтах
     * @param flags Флаги (backend-specific)
     * @return Указатель на выделенную память
     */
    virtual void* Allocate(size_t size_bytes, unsigned int flags = 0) = 0;

    /**
     * @brief Выделить unified memory (CPU+GPU доступ без явного hipMemcpy)
     *
     * ROCm: hipMallocManaged — полезно для отладки (CPU читает без D2H).
     * OpenCL: не поддерживается, возвращает nullptr.
     * Освобождать через Free() (hipFree совместим с managed memory).
     *
     * @param size_bytes Размер в байтах
     * @return Указатель на managed memory; nullptr если не поддерживается
     */
    virtual void* AllocateManaged(size_t size_bytes) { (void)size_bytes; return nullptr; }
    
    /**
     * @brief Освободить память на GPU
     * @param ptr Указатель на память
     */
    virtual void Free(void* ptr) = 0;
    
    /**
     * @brief Копировать данные Host -> Device
     */
    virtual void MemcpyHostToDevice(void* dst, const void* src,
                                   size_t size_bytes) = 0;
    
    /**
     * @brief Копировать данные Device -> Host
     */
    virtual void MemcpyDeviceToHost(void* dst, const void* src,
                                   size_t size_bytes) = 0;
    
    /**
     * @brief Копировать данные Device -> Device
     */
    virtual void MemcpyDeviceToDevice(void* dst, const void* src,
                                     size_t size_bytes) = 0;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Синхронизировать (ждать завершения всех операций)
     */
    virtual void Synchronize() = 0;
    
    /**
     * @brief Flush команд (без ожидания)
     */
    virtual void Flush() = 0;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Возможности устройства
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Поддерживается ли SVM (Shared Virtual Memory)
     */
    virtual bool SupportsSVM() const = 0;
    
    /**
     * @brief Поддерживается ли double precision
     */
    virtual bool SupportsDoublePrecision() const = 0;
    
    /**
     * @brief Максимальный размер work group
     */
    virtual size_t GetMaxWorkGroupSize() const = 0;
    
    /**
     * @brief Глобальная память (bytes)
     */
    virtual size_t GetGlobalMemorySize() const = 0;

    /**
     * @brief Свободная память GPU (bytes)
     *
     * Для NVIDIA: использует CL_DEVICE_MEMORY_FREE_NV (расширение)
     * Для AMD: использует расширение AMD или эвристику
     * Fallback: GetGlobalMemorySize() * 0.9
     */
    virtual size_t GetFreeMemorySize() const = 0;

    /**
     * @brief Локальная память (bytes)
     */
    virtual size_t GetLocalMemorySize() const = 0;
};

} // namespace drv_gpu_lib
