#pragma once

/**
 * @file gpu_manager.hpp
 * @brief GPUManager - центральный координатор для Multi-GPU
 *
 * ✅ MULTI-GPU (v2.0):
 * Теперь использует OpenCLCore::GetAvailableDeviceCount() для
 * реального обнаружения GPU в системе!
 *
 * GPUManager управляет множественными экземплярами DrvGPU и предоставляет:
 * - ✅ Автоматическое обнаружение ВСЕХ GPU (реальное!)
 * - Балансировка нагрузки (Round-Robin, наименее загруженная, вручную)
 * - Централизованное управление ресурсами
 * - Thread-safe доступ к GPU
 *
 * @author DrvGPU Team
 * @date 2026-02-06
 */

#include <core/drv_gpu.hpp>
#include <core/common/backend_type.hpp>
#include <core/common/load_balancing.hpp>
#include <core/logger/logger.hpp>
#include <core/backends/opencl/opencl_core.hpp>  // ✅ MULTI-GPU: Для реального обнаружения устройств
#include <core/backends/opencl/opencl_backend.hpp>
#include <core/services/gpu_profiler.hpp>  // ✅ AUTO-FILL: Для автоматической передачи GPU info

#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: GPUManager - Координатор для Multi-GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class GPUManager
 * @brief Facade для управления множественными GPU
 * 
 * GPUManager - это "Single entry point" для работы с несколькими GPU.
 * Он создаёт и управляет экземплярами DrvGPU для каждого устройства.
 * 
 * Примеры использования:
 * 
 * @code
 * // Инициализация всех GPU
 * GPUManager manager;
 * manager.InitializeAll(BackendType::OPENCL);
 * 
 * // Round-Robin распределение
 * for (int i = 0; i < 100; ++i) {
 *     auto& gpu = manager.GetNextGPU();
 *     gpu.GetMemoryManager().Allocate(...);
 * }
 * 
 * // Явный выбор GPU
 * auto& gpu0 = manager.GetGPU(0);
 * auto& gpu1 = manager.GetGPU(1);
 * 
 * // Балансировка нагрузки
 * auto& least_loaded = manager.GetLeastLoadedGPU();
 * @endcode
 * 
 * Паттерны:
 * - Facade (упрощение работы с Multi-GPU)
 * - Factory (создание DrvGPU экземпляров)
 * - Strategy (стратегии балансировки нагрузки)
 */
class GPUManager {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать GPUManager (без инициализации GPU)
     */
    GPUManager();
    
    /**
     * @brief Деструктор (освободит все GPU)
     */
    ~GPUManager();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;
    GPUManager(GPUManager&& other) noexcept;
    GPUManager& operator=(GPUManager&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать все доступные GPU
     * @param backend_type Тип бэкенда (OPENCL, CUDA, VULKAN)
     * @throws std::runtime_error если не найдено ни одной GPU
     */
    void InitializeAll(BackendType backend_type);
    
    /**
     * @brief Инициализировать конкретные GPU по индексам
     * @param backend_type Тип бэкенда
     * @param device_indices Список индексов GPU для инициализации
     */
    void InitializeSpecific(BackendType backend_type, 
                           const std::vector<int>& device_indices);
    
    /**
     * @brief Проверить, инициализирован ли менеджер
     */
    bool IsInitialized() const { return !gpus_.empty(); }
    
    /**
     * @brief Очистить все GPU и освободить ресурсы
     */
    void Cleanup();
    
    // ═══════════════════════════════════════════════════════════════
    // Доступ к GPU
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить GPU по индексу
     * @param index Индекс GPU (0 до GetGPUCount()-1)
     * @throws std::out_of_range если индекс некорректен
     */
    DrvGPU& GetGPU(size_t index);
    const DrvGPU& GetGPU(size_t index) const;
    
    /**
     * @brief Получить следующую GPU (Round-Robin)
     * Thread-safe, автоматически инкрементирует счётчик
     */
    DrvGPU& GetNextGPU();
    
    /**
     * @brief Получить наименее загруженную GPU
     * Использует метрику: количество активных задач
     */
    DrvGPU& GetLeastLoadedGPU();
    
    /**
     * @brief Получить все GPU
     */
    std::vector<DrvGPU*> GetAllGPUs();
    std::vector<const DrvGPU*> GetAllGPUs() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Информация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество доступных GPU
     */
    size_t GetGPUCount() const { return gpus_.size(); }
    
    /**
     * @brief Получить тип бэкенда
     */
    BackendType GetBackendType() const { return backend_type_; }
    
    /**
     * @brief Вывести информацию обо всех GPU
     */
    void PrintAllDevices() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Load Balancing
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Установить стратегию балансировки нагрузки
     */
    void SetLoadBalancingStrategy(LoadBalancingStrategy strategy);
    
    /**
     * @brief Получить текущую стратегию
     */
    LoadBalancingStrategy GetLoadBalancingStrategy() const { 
        return lb_strategy_; 
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Синхронизировать все GPU (ждать завершения всех операций)
     */
    void SynchronizeAll();
    
    /**
     * @brief Сброс буфера команд всех GPU
     */
    void FlushAll();
    
    // ═══════════════════════════════════════════════════════════════
    // Статистика
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вывести статистику по всем GPU
     */
    void PrintStatistics() const;
    
    /**
     * @brief Получить строку со статистикой
     */
    std::string GetStatistics() const;

    /**
     * @brief Получить информацию о драйверах всех GPU
     * @return Вектор map с версиями драйверов для каждой GPU
     */
    std::vector<std::map<std::string, std::string>> GetDriverSet() const;

    /**
     * @brief Получить GPUReportInfo для конкретной GPU (для профайлера)
     * @param gpu_id Индекс GPU
     * @return GPUReportInfo с заполненными drivers[] из реальной системы
     *
     * Использование:
     *   auto info = manager.GetGPUReportInfo(0);
     *   profiler.SetGPUInfo(0, info);
     */
    GPUReportInfo GetGPUReportInfo(int gpu_id) const;

    /**
     * @brief Сбросить статистику всех GPU
     */
    void ResetStatistics();
    
    // ═══════════════════════════════════════════════════════════════
    // Утилиты
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество доступных GPU в системе (статический метод)
     * @param backend_type Тип бэкенда для запроса
     */
    static int GetAvailableGPUCount(BackendType backend_type);

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    BackendType backend_type_;
    LoadBalancingStrategy lb_strategy_;
    
    // GPU экземпляры (владение через unique_ptr)
    std::vector<std::unique_ptr<DrvGPU>> gpus_;
    
    // Счётчик Round-Robin (потокобезопасный)
    std::atomic<size_t> round_robin_index_;
    
    // Учёт нагрузки (метрика: количество задач, защищено мьютексом)
    std::vector<size_t> gpu_task_count_;
    
    // Потокобезопасность
    mutable std::mutex mutex_;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Обнаружить все доступные GPU
     */
    int DiscoverGPUs(BackendType backend_type);
    
    /**
     * @brief Инициализировать GPU по индексу
     */
    void InitializeGPU(int device_index);
    
    /**
     * @brief Получить индекс наименее загруженной GPU
     */
    size_t GetLeastLoadedGPUIndex() const;
    
    /**
     * @brief Внутренний метод очистки БЕЗ блокировки mutex
     * ✅ DEADLOCK FIX: используется изнутри методов, которые уже держат lock
     */
    void CleanupInternal();
};

// ════════════════════════════════════════════════════════════════════════════
// Inline-реализация GPUManager (только заголовки)
// ════════════════════════════════════════════════════════════════════════════

inline GPUManager::GPUManager()
    : backend_type_(BackendType::OPENCL)
    , lb_strategy_(LoadBalancingStrategy::ROUND_ROBIN)
    , round_robin_index_(0) {
}

inline GPUManager::~GPUManager() {
    // ✅ Деструктор вызывает публичный Cleanup (с блокировкой)
    Cleanup();
}

inline GPUManager::GPUManager(GPUManager&& other) noexcept
    : backend_type_(other.backend_type_)
    , lb_strategy_(other.lb_strategy_)
    , gpus_(std::move(other.gpus_))
    , round_robin_index_(other.round_robin_index_.load())
    , gpu_task_count_(std::move(other.gpu_task_count_)) {
}

inline GPUManager& GPUManager::operator=(GPUManager&& other) noexcept {
    if (this != &other) {
        // ✅ FIX: Вызываем публичный Cleanup (с блокировкой)
        Cleanup();
        
        backend_type_ = other.backend_type_;
        lb_strategy_ = other.lb_strategy_;
        gpus_ = std::move(other.gpus_);
        round_robin_index_ = other.round_robin_index_.load();
        gpu_task_count_ = std::move(other.gpu_task_count_);
    }
    
    return *this;
}

inline void GPUManager::InitializeAll(BackendType backend_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    backend_type_ = backend_type;
    
    // ✅ FIX: Вызываем ВНУТРЕННИЙ метод (БЕЗ блокировки)
    CleanupInternal();
    
    int gpu_count = DiscoverGPUs(backend_type);
    if (gpu_count == 0) {
        throw std::runtime_error("No GPUs available for backend type");
    }
    
    for (int i = 0; i < gpu_count; ++i) {
        InitializeGPU(i);
    }
    
    DRVGPU_LOG_INFO_GPU(0, "GPUManager", "Initialized " + std::to_string(gpus_.size()) + " GPU(s)");
}

inline void GPUManager::InitializeSpecific(BackendType backend_type,
                                          const std::vector<int>& device_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    backend_type_ = backend_type;
    
    // ✅ FIX: Вызываем ВНУТРЕННИЙ метод (БЕЗ блокировки)
    CleanupInternal();
    
    for (int index : device_indices) {
        InitializeGPU(index);
    }
    
    DRVGPU_LOG_INFO_GPU(0, "GPUManager", "Initialized " + std::to_string(gpus_.size()) + " specific GPU(s)");
}

// ════════════════════════════════════════════════════════════════════════════
// ✅ DEADLOCK FIX: Два варианта метода Cleanup
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Внутренний метод очистки БЕЗ блокировки
 * Используется изнутри других методов, которые уже держат lock
 */
inline void GPUManager::CleanupInternal() {
    // ✅ БЕЗ std::lock_guard - предполагается что mutex уже заблокирован!
    
    std::vector<std::unique_ptr<DrvGPU>> temp_gpus;
    
    // Перемещаем gpus_ во временный вектор
    temp_gpus = std::move(gpus_);
    
    // Очищаем метаданные
    gpu_task_count_.clear();
    round_robin_index_ = 0;
    
    DRVGPU_LOG_INFO_GPU(0, "GPUManager", "CleanupInternal: GPU instances moved, will be destroyed on scope exit");
    
    // temp_gpus автоматически очистится при выходе из scope
    // Деструкторы ~DrvGPU() вызовутся здесь
}

/**
 * @brief Публичный метод очистки С блокировкой
 * Используется извне (из деструктора, пользовательского кода)
 */
inline void GPUManager::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Вызываем внутренний метод
    CleanupInternal();
}

// ════════════════════════════════════════════════════════════════════════════
// Остальные методы
// ════════════════════════════════════════════════════════════════════════════

inline DrvGPU& GPUManager::GetGPU(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (index >= gpus_.size()) {
        throw std::out_of_range("GPU index out of range");
    }
    
    return *gpus_[index];
}

inline const DrvGPU& GPUManager::GetGPU(size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (index >= gpus_.size()) {
        throw std::out_of_range("GPU index out of range");
    }
    
    return *gpus_[index];
}

inline DrvGPU& GPUManager::GetNextGPU() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (gpus_.empty()) {
        throw std::runtime_error("No GPUs initialized");
    }
    
    size_t index = round_robin_index_++ % gpus_.size();
    return *gpus_[index];
}

inline DrvGPU& GPUManager::GetLeastLoadedGPU() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t least_loaded_idx = GetLeastLoadedGPUIndex();
    return *gpus_[least_loaded_idx];
}

inline std::vector<DrvGPU*> GPUManager::GetAllGPUs() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<DrvGPU*> result;
    result.reserve(gpus_.size());
    for (auto& gpu : gpus_) {
        result.push_back(gpu.get());
    }
    
    return result;
}

inline std::vector<const DrvGPU*> GPUManager::GetAllGPUs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<const DrvGPU*> result;
    result.reserve(gpus_.size());
    for (auto& gpu : gpus_) {
        result.push_back(gpu.get());
    }
    
    return result;
}

inline void GPUManager::PrintAllDevices() const {
    std::cout << "\n--- GPU Devices ---\n";
    size_t idx = 0;
    for (const auto& gpu : gpus_) {
        std::cout << "GPU " << idx << ": " << gpu->GetDeviceName() << "\n";
        ++idx;
    }
    std::cout << "------------------\n";
}

inline void GPUManager::SetLoadBalancingStrategy(LoadBalancingStrategy strategy) {
    std::lock_guard<std::mutex> lock(mutex_);
    lb_strategy_ = strategy;
}

inline void GPUManager::SynchronizeAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& gpu : gpus_) {
        gpu->Synchronize();
    }
}

inline void GPUManager::FlushAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& gpu : gpus_) {
        gpu->Flush();
    }
}

inline void GPUManager::PrintStatistics() const {
    std::cout << "\n=== GPU Manager Statistics ===\n";
    std::cout << "Total GPUs: " << gpus_.size() << "\n";
    size_t idx = 0;
    for (const auto& gpu : gpus_) {
        std::cout << "GPU " << idx << ": " << gpu->GetDeviceName() << "\n";
        std::cout << gpu->GetStatistics();
        ++idx;
    }
    std::cout << "==============================\n\n";
}

inline std::string GPUManager::GetStatistics() const {
    std::ostringstream oss;
    oss << "GPU Manager Statistics:\n";
    oss << "  Total GPUs: " << gpus_.size() << "\n";
    oss << "  Load Balancing: " << LoadBalancingStrategyToString(lb_strategy_) << "\n";
    return oss.str();
}

inline std::vector<std::map<std::string, std::string>> GPUManager::GetDriverSet() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::map<std::string, std::string>> result;

    for (size_t i = 0; i < gpus_.size(); ++i) {
        auto info = gpus_[i]->GetDeviceInfo();
        std::map<std::string, std::string> gpu_info;
        gpu_info["gpu_id"] = std::to_string(i);
        gpu_info["gpu_name"] = info.name;
        gpu_info["driver_version"] = info.driver_version;
        gpu_info["opencl_version"] = info.opencl_version;
        gpu_info["vendor"] = info.vendor;
        result.push_back(gpu_info);
    }

    return result;
}

inline GPUReportInfo GPUManager::GetGPUReportInfo(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    GPUReportInfo report_info;

    if (gpu_id < 0 || static_cast<size_t>(gpu_id) >= gpus_.size()) {
        return report_info;  // Пустая структура если GPU не найдена
    }

    auto device_info = gpus_[gpu_id]->GetDeviceInfo();
    report_info.gpu_name = device_info.name;
    report_info.backend_type = backend_type_;
    report_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);

    // drivers[0] = OpenCL info (читаем из реальной системы)
    if (backend_type_ == BackendType::OPENCL ||
        backend_type_ == BackendType::OPENCLandROCm ||
        backend_type_ == BackendType::AUTO) {

        std::map<std::string, std::string> opencl_driver;
        opencl_driver["driver_type"] = "OpenCL";
        opencl_driver["version"] = device_info.opencl_version;
        opencl_driver["driver_version"] = device_info.driver_version;
        opencl_driver["vendor"] = device_info.vendor;

        // Platform name из OpenCL backend
        try {
            auto* backend = dynamic_cast<OpenCLBackend*>(&gpus_[gpu_id]->GetBackend());
            if (backend && backend->GetCore().IsInitialized()) {
                opencl_driver["platform_name"] = backend->GetCore().GetPlatformName();
            }
        } catch (...) {
            // Игнорируем ошибки приведения типа
        }

        report_info.drivers.push_back(opencl_driver);
    }

    // drivers[1] = ROCm info (ЗАКОММЕНТИРОВАНО - нет ROCm драйверов)
    /*
    if (backend_type_ == BackendType::ROCm ||
        backend_type_ == BackendType::OPENCLandROCm) {
        std::map<std::string, std::string> rocm_driver;
        rocm_driver["driver_type"] = "ROCm";
        // ... читаем из /opt/rocm/.info/version и HIP API
        report_info.drivers.push_back(rocm_driver);
    }
    */

    return report_info;
}

inline void GPUManager::ResetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& gpu : gpus_) {
        gpu->ResetStatistics();
    }
}

inline int GPUManager::DiscoverGPUs(BackendType backend_type) {
    DRVGPU_LOG_DEBUG_GPU(0, "GPUManager", "Discovering GPUs...");

    // ═══════════════════════════════════════════════════════════════════════
    // ✅ MULTI-GPU: Реальное обнаружение устройств!
    // ═══════════════════════════════════════════════════════════════════════

    int device_count = 0;

    switch (backend_type) {
        case BackendType::OPENCL:
        case BackendType::OPENCLandROCm:
        case BackendType::AUTO: {
            // Используем OpenCLCore для обнаружения GPU
            device_count = OpenCLCore::GetAvailableDeviceCount(DeviceType::GPU);

            DRVGPU_LOG_INFO_GPU(0, "GPUManager",
                "Found " + std::to_string(device_count) + " OpenCL GPU(s)");

            // Выводим информацию о найденных устройствах
            if (device_count > 0) {
                std::string devices_info = OpenCLCore::GetAllDevicesInfo(DeviceType::GPU);
                DRVGPU_LOG_DEBUG_GPU(0, "GPUManager", devices_info);
            }
            break;
        }

        case BackendType::ROCm: {
            // TODO: Реализовать для ROCm
            DRVGPU_LOG_WARNING_GPU(0, "GPUManager", "ROCm discovery not implemented yet");
            // Пока используем OpenCL discovery (ROCm поддерживает OpenCL)
            device_count = OpenCLCore::GetAvailableDeviceCount(DeviceType::GPU);
            break;
        }

        default:
            DRVGPU_LOG_ERROR_GPU(0, "GPUManager", "Unknown backend type");
            device_count = 0;
            break;
    }

    return device_count;
}

inline void GPUManager::InitializeGPU(int device_index) {
    try {
        auto gpu = std::make_unique<DrvGPU>(backend_type_, device_index);
        gpu->Initialize();

        // ═══════════════════════════════════════════════════════════════
        // AUTO-FILL: Автоматически передаём GPU info в профайлер
        // Формируем drivers[] vector с информацией о драйверах
        // ═══════════════════════════════════════════════════════════════
        auto device_info = gpu->GetDeviceInfo();
        GPUReportInfo report_info;
        report_info.gpu_name = device_info.name;
        report_info.backend_type = backend_type_;
        report_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);

        // drivers[0] = OpenCL info (читаем из реальной системы)
        if (backend_type_ == BackendType::OPENCL ||
            backend_type_ == BackendType::OPENCLandROCm ||
            backend_type_ == BackendType::AUTO) {
            std::map<std::string, std::string> opencl_driver;
            opencl_driver["driver_type"] = "OpenCL";
            opencl_driver["version"] = device_info.opencl_version;
            opencl_driver["driver_version"] = device_info.driver_version;
            opencl_driver["vendor"] = device_info.vendor;

            // Platform name из OpenCL backend
            try {
                auto* backend = dynamic_cast<OpenCLBackend*>(&gpu->GetBackend());
                if (backend && backend->GetCore().IsInitialized()) {
                    opencl_driver["platform_name"] = backend->GetCore().GetPlatformName();
                }
            } catch (...) {
                // Ignore cast errors
            }

            report_info.drivers.push_back(opencl_driver);
        }

        // ═══════════════════════════════════════════════════════════════
        // drivers[1] = ROCm info (ЗАКОММЕНТИРОВАНО - нет ROCm драйверов)
        // Раскомментировать когда будут установлены ROCm/HIP
        // ═══════════════════════════════════════════════════════════════
        /*
        if (backend_type_ == BackendType::ROCm ||
            backend_type_ == BackendType::OPENCLandROCm) {
            std::map<std::string, std::string> rocm_driver;
            rocm_driver["driver_type"] = "ROCm";

            // Версия ROCm из /opt/rocm/.info/version
            std::ifstream f("/opt/rocm/.info/version");
            if (f.is_open()) {
                std::string version;
                std::getline(f, version);
                rocm_driver["version"] = version;  // "5.4.3"
            }

            // HIP версия через hipRuntimeGetVersion
            // int hipVersion = 0;
            // hipRuntimeGetVersion(&hipVersion);
            // rocm_driver["hip_version"] = std::to_string(hipVersion);

            // int driverVersion = 0;
            // hipDriverGetVersion(&driverVersion);
            // rocm_driver["hip_driver"] = std::to_string(driverVersion);

            report_info.drivers.push_back(rocm_driver);
        }
        */

        GPUProfiler::GetInstance().SetGPUInfo(device_index, report_info);
        // ═══════════════════════════════════════════════════════════════

        gpus_.push_back(std::move(gpu));
        gpu_task_count_.emplace_back(0);
        DRVGPU_LOG_INFO_GPU(device_index, "GPUManager", "Initialized GPU " + std::to_string(device_index));
    } catch (const std::exception& e) {
        DRVGPU_LOG_ERROR_GPU(device_index, "GPUManager", "Failed to initialize GPU " + std::to_string(device_index) + ": " + e.what());
    }
}

inline size_t GPUManager::GetLeastLoadedGPUIndex() const {
    size_t min_tasks = SIZE_MAX;
    size_t min_idx = 0;
    for (size_t i = 0; i < gpu_task_count_.size(); ++i) {
        size_t tasks = gpu_task_count_[i];
        if (tasks < min_tasks) {
            min_tasks = tasks;
            min_idx = i;
        }
    }
    return min_idx;
}

inline int GPUManager::GetAvailableGPUCount(BackendType backend_type) {
    // ✅ MULTI-GPU: Реальное обнаружение!
    switch (backend_type) {
        case BackendType::OPENCL:
        case BackendType::OPENCLandROCm:
        case BackendType::AUTO:
            return OpenCLCore::GetAvailableDeviceCount(DeviceType::GPU);

        case BackendType::ROCm:
            // Пока используем OpenCL discovery
            return OpenCLCore::GetAvailableDeviceCount(DeviceType::GPU);

        default:
            return 0;
    }
}

} // namespace drv_gpu_lib
