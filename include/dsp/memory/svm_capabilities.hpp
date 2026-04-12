#pragma once

/**
 * @file svm_capabilities.hpp
 * @brief OpenCL SVM (Shared Virtual Memory) capabilities detection
 * 
 * Определяет возможности SVM для конкретного GPU устройства.
 * Поддерживает OpenCL 2.0+ с fallback для OpenCL 1.x
 * 
 * @author Codo (AI Assistant)
 * @date 2026-01-19
 */

#include <CL/cl.h>
#include <cstdio>
#include <string>
#include <sstream>
#include <iomanip>
#include <type_traits>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Enum: MemoryStrategy - стратегия выделения памяти
// ════════════════════════════════════════════════════════════════════════════

/**
 * @enum MemoryStrategy
 * @brief Определяет стратегию работы с GPU памятью
 * 
 * Выбор стратегии зависит от:
 * - Возможностей GPU (SVM support)
 * - Размера буфера
 * - Паттерна использования (частота read/write)
 */
enum class MemoryStrategy {
    REGULAR_BUFFER,      ///< Традиционный cl_mem + clEnqueueRead/Write
    SVM_COARSE_GRAIN,    ///< SVM Coarse-Grained Buffer (map/unmap required)
    SVM_FINE_GRAIN,      ///< SVM Fine-Grained Buffer (atomics optional)
    SVM_FINE_SYSTEM,     ///< SVM Fine-Grained System (unified memory)
    AUTO                 ///< Автоматический выбор на основе эвристик
};

/**
 * @brief Преобразовать стратегию в строку
 */
inline std::string MemoryStrategyToString(MemoryStrategy strategy) {
    switch (strategy) {
        case MemoryStrategy::REGULAR_BUFFER:   return "REGULAR_BUFFER";
        case MemoryStrategy::SVM_COARSE_GRAIN: return "SVM_COARSE_GRAIN";
        case MemoryStrategy::SVM_FINE_GRAIN:   return "SVM_FINE_GRAIN";
        case MemoryStrategy::SVM_FINE_SYSTEM:  return "SVM_FINE_SYSTEM";
        case MemoryStrategy::AUTO:             return "AUTO";
        default:                               return "UNKNOWN";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Struct: SVMCapabilities - возможности SVM устройства
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct SVMCapabilities
 * @brief Хранит информацию о поддержке SVM на устройстве
 * 
 * Использование:
 * @code
 * auto caps = SVMCapabilities::Query(device);
 * if (caps.coarse_grain_buffer) {
 *     // Использовать SVM
 * }
 * @endcode
 */
struct SVMCapabilities {
    // ═══════════════════════════════════════════════════════════════
    // Флаги поддержки
    // ═══════════════════════════════════════════════════════════════
    
    bool coarse_grain_buffer = false;  ///< CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
    bool fine_grain_buffer   = false;  ///< CL_DEVICE_SVM_FINE_GRAIN_BUFFER
    bool fine_grain_system   = false;  ///< CL_DEVICE_SVM_FINE_GRAIN_SYSTEM
    bool atomics             = false;  ///< CL_DEVICE_SVM_ATOMICS
    
    // ═══════════════════════════════════════════════════════════════
    // Дополнительная информация
    // ═══════════════════════════════════════════════════════════════
    
    cl_uint opencl_major_version = 0;  ///< Мажорная версия OpenCL
    cl_uint opencl_minor_version = 0;  ///< Минорная версия OpenCL
    bool    svm_supported        = false;  ///< SVM поддерживается вообще
    
    // ═══════════════════════════════════════════════════════════════
    // Статический метод: запросить возможности устройства
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Запросить SVM capabilities устройства
     * @param device OpenCL device ID
     * @return SVMCapabilities структура с флагами
     */
    static SVMCapabilities Query(cl_device_id device) {
        SVMCapabilities caps;
        
        if (!device) {
            // Логируем ошибку для диагностики
            std::fprintf(stderr, "[SVM Capabilities] ERROR: device is null in Query() - returning empty capabilities\n");
            return caps;  // Пустые capabilities
        }
        
        // 1. Получить версию OpenCL
        char version_str[256] = {0};
        cl_int err = clGetDeviceInfo(
            device, 
            CL_DEVICE_VERSION, 
            sizeof(version_str), 
            version_str, 
            nullptr
        );
        
        if (err == CL_SUCCESS) {
            // Формат: "OpenCL X.Y ..."
            int major = 0, minor = 0;
            if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) == 2) {
                caps.opencl_major_version = static_cast<cl_uint>(major);
                caps.opencl_minor_version = static_cast<cl_uint>(minor);
            }
        }
        
        // 2. Проверить SVM capabilities (только для OpenCL 2.0+)
        if (caps.opencl_major_version >= 2) {
            cl_device_svm_capabilities svm_caps = 0;
            err = clGetDeviceInfo(
                device,
                CL_DEVICE_SVM_CAPABILITIES,
                sizeof(svm_caps),
                &svm_caps,
                nullptr
            );
            
            if (err == CL_SUCCESS && svm_caps != 0) {
                caps.svm_supported = true;
                caps.coarse_grain_buffer = (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) != 0;
                caps.fine_grain_buffer   = (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
                caps.fine_grain_system   = (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) != 0;
                caps.atomics             = (svm_caps & CL_DEVICE_SVM_ATOMICS) != 0;
            }
        }
        
        return caps;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Методы-помощники
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Проверить поддержку любого типа SVM
     */
    bool HasAnySVM() const {
        return svm_supported && (coarse_grain_buffer || fine_grain_buffer || fine_grain_system);
    }
    
    /**
     * @brief Получить лучшую доступную стратегию SVM
     */
    MemoryStrategy GetBestSVMStrategy() const {
        if (fine_grain_system) {
            return MemoryStrategy::SVM_FINE_SYSTEM;
        }
        if (fine_grain_buffer) {
            return MemoryStrategy::SVM_FINE_GRAIN;
        }
        if (coarse_grain_buffer) {
            return MemoryStrategy::SVM_COARSE_GRAIN;
        }
        return MemoryStrategy::REGULAR_BUFFER;
    }
    
    /**
     * @brief Получить рекомендуемую стратегию на основе размера буфера
     * 
     * Эвристика:
     * - Маленькие буферы (< 1MB): Regular buffer быстрее из-за overhead SVM
     * - Средние буферы (1MB - 64MB): SVM coarse-grain если доступен
     * - Большие буферы (> 64MB): SVM предпочтительнее для zero-copy
     * 
     * @param size_bytes Размер буфера в байтах
     * @return Рекомендуемая стратегия
     */
    MemoryStrategy RecommendStrategy(size_t size_bytes) const {
        // Пороговые значения (можно настроить)
        constexpr size_t SMALL_BUFFER_THRESHOLD  = 1 * 1024 * 1024;     // 1 MB
        [[maybe_unused]] constexpr size_t MEDIUM_BUFFER_THRESHOLD = 64 * 1024 * 1024;    // 64 MB
        
        // Если SVM не поддерживается - только regular buffer
        if (!svm_supported) {
            return MemoryStrategy::REGULAR_BUFFER;
        }
        
        // Маленькие буферы: regular buffer (overhead SVM не оправдан)
        if (size_bytes < SMALL_BUFFER_THRESHOLD) {
            return MemoryStrategy::REGULAR_BUFFER;
        }
        
        // Средние и большие буферы: SVM если доступен
        if (size_bytes >= SMALL_BUFFER_THRESHOLD) {
            // Предпочитаем coarse-grain для большинства случаев
            // (fine-grain может быть медленнее на дискретных GPU)
            if (coarse_grain_buffer) {
                return MemoryStrategy::SVM_COARSE_GRAIN;
            }
            if (fine_grain_buffer) {
                return MemoryStrategy::SVM_FINE_GRAIN;
            }
        }
        
        return MemoryStrategy::REGULAR_BUFFER;
    }
    
    /**
     * @brief Получить строковое представление capabilities
     */
    std::string ToString() const {
        std::ostringstream oss;
        
        oss << "\n" << std::string(60, '=') << "\n";
        oss << "SVM Capabilities\n";
        oss << std::string(60, '=') << "\n\n";
        
        oss << std::left << std::setw(25) << "OpenCL Version:" 
            << opencl_major_version << "." << opencl_minor_version << "\n";
        oss << std::left << std::setw(25) << "SVM Supported:" 
            << (svm_supported ? "YES ✅" : "NO ❌") << "\n\n";
        
        if (svm_supported) {
            oss << "SVM Types:\n";
            oss << "  " << std::left << std::setw(23) << "Coarse-Grain Buffer:" 
                << (coarse_grain_buffer ? "YES ✅" : "NO ❌") << "\n";
            oss << "  " << std::left << std::setw(23) << "Fine-Grain Buffer:" 
                << (fine_grain_buffer ? "YES ✅" : "NO ❌") << "\n";
            oss << "  " << std::left << std::setw(23) << "Fine-Grain System:" 
                << (fine_grain_system ? "YES ✅" : "NO ❌") << "\n";
            oss << "  " << std::left << std::setw(23) << "Atomics:" 
                << (atomics ? "YES ✅" : "NO ❌") << "\n";
        }
        
        oss << "\n" << std::left << std::setw(25) << "Recommended Strategy:" 
            << MemoryStrategyToString(GetBestSVMStrategy()) << "\n";
        
        oss << std::string(60, '=') << "\n";
        
        return oss.str();
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Struct: BufferUsageHint - подсказка по использованию буфера
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct BufferUsageHint
 * @brief Подсказки для оптимального выбора стратегии
 */
struct BufferUsageHint {
    bool frequent_host_read  = false;  ///< Частое чтение с хоста
    bool frequent_host_write = false;  ///< Частая запись с хоста
    bool gpu_only            = false;  ///< Буфер используется только GPU
    bool requires_atomics    = false;  ///< Нужны атомарные операции
    
    /**
     * @brief Создать hint для буфера только GPU
     */
    static BufferUsageHint GPUOnly() {
        BufferUsageHint hint;
        hint.gpu_only = true;
        return hint;
    }
    
    /**
     * @brief Создать hint для частого обмена хост-GPU
     */
    static BufferUsageHint FrequentTransfer() {
        BufferUsageHint hint;
        hint.frequent_host_read = true;
        hint.frequent_host_write = true;
        return hint;
    }
    
    /**
     * @brief Создать hint по умолчанию
     */
    static BufferUsageHint Default() {
        return BufferUsageHint();
    }
};

} // namespace 

