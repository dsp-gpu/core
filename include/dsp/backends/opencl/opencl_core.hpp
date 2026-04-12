#pragma once

#include <CL/cl.h>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <array>
#include <vector>

// Forward declaration для SVMCapabilities
namespace drv_gpu_lib { struct SVMCapabilities; }

#include "gpu_copy_kernel.hpp"  // GpuCopyKernels — per-GPU кеш copy kernels

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// ENUM для типа девайса
// ════════════════════════════════════════════════════════════════════════════

enum class DeviceType {
    GPU,  // CL_DEVICE_TYPE_GPU
    CPU   // CL_DEVICE_TYPE_CPU
};

// ════════════════════════════════════════════════════════════════════════════
// OpenCLCore - Per-Device контекст OpenCL (Multi-GPU поддержка)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class OpenCLCore
 * @brief Управляет OpenCL контекстом для КОНКРЕТНОГО устройства
 *
 * ✅ MULTI-GPU ARCHITECTURE:
 * Каждый экземпляр OpenCLCore владеет СВОИМ устройством по device_index.
 * Это позволяет создавать несколько backend'ов для разных GPU.
 *
 * Ответственность:
 * - Инициализация платформы и девайса по индексу
 * - Создание и владение контекстом OpenCL
 * - Информация о девайсе
 * - Thread-safe доступ к контексту
 *
 * НЕ управляет:
 * - Command queues (это делает CommandQueuePool или OpenCLBackend)
 * - Программы (это делает KernelProgram)
 * - Буферы (это делает GPUMemoryBuffer)
 *
 * @code
 * // Multi-GPU использование:
 * auto core0 = std::make_unique<OpenCLCore>(0, DeviceType::GPU);  // GPU 0
 * auto core1 = std::make_unique<OpenCLCore>(1, DeviceType::GPU);  // GPU 1
 *
 * core0->Initialize();
 * core1->Initialize();
 *
 * cl_context ctx0 = core0->GetContext();  // Контекст GPU 0
 * cl_context ctx1 = core1->GetContext();  // Контекст GPU 1 (РАЗНЫЕ!)
 * @endcode
 */
class OpenCLCore {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Создать OpenCLCore для конкретного устройства
     * @param device_index Индекс устройства (0, 1, 2, ...)
     * @param device_type Тип устройства: GPU или CPU
     */
    explicit OpenCLCore(int device_index = 0, DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Деструктор - освобождает ресурсы OpenCL
     */
    ~OpenCLCore();

    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    OpenCLCore(const OpenCLCore&) = delete;
    OpenCLCore& operator=(const OpenCLCore&) = delete;
    OpenCLCore(OpenCLCore&& other) noexcept;
    OpenCLCore& operator=(OpenCLCore&& other) noexcept;

    // ═══════════════════════════════════════════════════════════════
    // Инициализация
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать OpenCL контекст для этого устройства
     * @throws std::runtime_error если инициализация не удалась
     */
    void Initialize();

    /**
     * @brief Очистить ресурсы
     */
    void Cleanup();

    /**
     * @brief Проверить инициализацию
     */
    bool IsInitialized() const { return initialized_; }

    // ═══════════════════════════════════════════════════════════════
    // Getters для OpenCL объектов
    // ═══════════════════════════════════════════════════════════════

    cl_context GetContext() const { return context_; }
    cl_device_id GetDevice() const { return device_; }
    cl_platform_id GetPlatform() const { return platform_; }
    int GetDeviceIndex() const { return device_index_; }
    DeviceType GetDeviceType() const { return device_type_; }

    // ═══════════════════════════════════════════════════════════════
    // Информация о девайсе
    // ═══════════════════════════════════════════════════════════════

    std::string GetDeviceInfo() const;
    std::string GetDeviceName() const;
    std::string GetVendor() const;
    std::string GetDriverVersion() const;
    std::string GetPlatformName() const;
    size_t GetGlobalMemorySize() const;
    size_t GetFreeMemorySize() const;
    size_t GetLocalMemorySize() const;
    cl_uint GetComputeUnits() const;
    size_t GetMaxWorkGroupSize() const;
    std::array<size_t, 3> GetMaxWorkItemSizes() const;

    // ═══════════════════════════════════════════════════════════════
    // SVM (Shared Virtual Memory) информация - OpenCL 2.0+
    // ═══════════════════════════════════════════════════════════════

    cl_uint GetOpenCLVersionMajor() const;
    cl_uint GetOpenCLVersionMinor() const;
    bool IsSVMSupported() const;
    SVMCapabilities GetSVMCapabilities() const;
    std::string GetSVMInfo() const;

    // ═══════════════════════════════════════════════════════════════
    // СТАТИЧЕСКИЕ МЕТОДЫ для обнаружения GPU (Multi-GPU support)
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить количество доступных устройств
     * @param device_type Тип устройства (GPU или CPU)
     * @return Количество устройств данного типа
     */
    static int GetAvailableDeviceCount(DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Получить все доступные устройства
     * @param device_type Тип устройства (GPU или CPU)
     * @return Вектор пар (platform_id, device_id)
     */
    static std::vector<std::pair<cl_platform_id, cl_device_id>>
        GetAllDevices(DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Получить информацию о всех устройствах (для вывода)
     * @param device_type Тип устройства
     * @return Строка с информацией о всех устройствах
     */
    static std::string GetAllDevicesInfo(DeviceType device_type = DeviceType::GPU);

    // ═══════════════════════════════════════════════════════════════
    // GPU Copy Kernels — per-GPU кеш (Ref04: заменяет singleton)
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить или скомпилировать copy kernels для ЭТОГО GPU
     *
     * Per-instance кеш: компилируется один раз, хранится до ReleaseResources().
     * Никакого shared state между GPU — каждый OpenCLCore свой.
     *
     * @return Указатель на кешированные kernels, nullptr при ошибке
     */
    GpuCopyKernels* GetOrCompileCopyKernels();

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════

    int device_index_;
    DeviceType device_type_;
    bool initialized_;

    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;

    // Per-GPU кеш copy kernels (Ref04: вместо singleton GpuCopyKernelCache)
    GpuCopyKernels copy_kernels_{};       // Compile-on-first-use, release in ReleaseResources()
    bool copy_kernels_compiled_ = false;

    mutable std::mutex mutex_;

    // ═══════════════════════════════════════════════════════════════
    // Приватные методы инициализации
    // ═══════════════════════════════════════════════════════════════

    void InitializeOpenCL();
    void ReleaseResources();

    // Утилиты для информации о девайсе
    template<typename T>
    T GetDeviceInfoValue(cl_device_info param) const;

    std::string GetDeviceInfoString(cl_device_info param) const;
};

// ════════════════════════════════════════════════════════════════════════════
// Утилита: Проверка OpenCL ошибок (inline для удобства)
// ════════════════════════════════════════════════════════════════════════════

inline void CheckCLError(cl_int error, const std::string& operation) {
    if (error != CL_SUCCESS) {
        std::string error_msg = "OpenCL Error [" + std::to_string(error) + "] in " + operation;
        throw std::runtime_error(error_msg);
    }
}

} // namespace drv_gpu_lib
