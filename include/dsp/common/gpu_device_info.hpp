#pragma once

/**
 * @file gpu_device_info.hpp
 * @brief Информация о GPU устройстве
 */

#include <string>
#include <cstddef>

namespace drv_gpu_lib {

/**
 * @struct GPUDeviceInfo
 * @brief Структура с информацией о GPU устройстве
 * 
 * Backend-независимая структура для хранения информации о GPU.
 */
struct GPUDeviceInfo {
    // ═══════════════════════════════════════════════════════════════
    // Основная информация
    // ═══════════════════════════════════════════════════════════════
    
    std::string name;              ///< Название устройства
    std::string vendor;            ///< Производитель
    std::string driver_version;    ///< Версия драйвера
    std::string opencl_version;    ///< Версия OpenCL (если применимо)
    int device_index;              ///< Индекс устройства
    
    // ═══════════════════════════════════════════════════════════════
    // Память
    // ═══════════════════════════════════════════════════════════════
    
    size_t global_memory_size;     ///< Глобальная память (bytes)
    size_t local_memory_size;      ///< Локальная память (bytes)
    size_t max_mem_alloc_size;     ///< Максимальный размер аллокации
    
    // ═══════════════════════════════════════════════════════════════
    // Вычислительные возможности
    // ═══════════════════════════════════════════════════════════════
    
    size_t max_compute_units;      ///< Количество compute units
    size_t max_work_group_size;    ///< Максимальный размер work group
    size_t max_clock_frequency;    ///< Частота (MHz)
    
    // ═══════════════════════════════════════════════════════════════
    // Возможности
    // ═══════════════════════════════════════════════════════════════
    
    bool supports_svm;             ///< Shared Virtual Memory
    bool supports_double;          ///< Double precision
    bool supports_half;            ///< Half precision (fp16)
    bool supports_unified_memory;  ///< Unified memory
    
    // ═══════════════════════════════════════════════════════════════
    // Утилиты
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вывести информацию в читаемом виде
     */
    std::string ToString() const;
    
    /**
     * @brief Получить размер глобальной памяти в GB
     */
    double GetGlobalMemoryGB() const {
        return static_cast<double>(global_memory_size) / (1024.0 * 1024.0 * 1024.0);
    }
};

} // namespace drv_gpu_lib
