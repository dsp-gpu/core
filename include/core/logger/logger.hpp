#pragma once

/**
 * @file logger.hpp
 * @brief Logger - Главный фасад системы логирования DrvGPU
 * 
 * Предоставляет:
 * - Макросы DRVGPU_LOG_* для удобного логирования
 * - Фабрику Logger для установки своего логера
 * - Условную компиляцию (отключение в Release)
 * 
 * Уровни логирования:
 * - DEBUG: Подробная отладка (только в Debug сборке)
 * - INFO: Информационные сообщения
 * - WARNING: Предупреждения
 * - ERROR: Ошибки
 * 
 * Пример использования:
 * @code
 * #include "common/logger.hpp"
 * 
 * DRVGPU_LOG_INFO("DrvGPU", "Initialized successfully");
 * DRVGPU_LOG_WARNING("OpenCL", "Memory allocation warning");
 * DRVGPU_LOG_ERROR("Backend", "Failed to create context");
 * 
 * // В продакшене можно установить свой логер:
 * Logger::SetInstance(my_company_logger);
 * @endcode
 * 
 * @author DrvGPU Team
 * @date 2026-02-01
 */

#include <core/interface/i_logger.hpp>
#include <core/logger/config_logger.hpp>
#include <string>
// PRIVATE: реализация default_logger через PIMPL — сюда НЕ инклудим,
// клиенты получают ILogger* через LoggerFactory или Logger::GetInstance().

namespace drv_gpu_lib {
/**    
 * - Получить текущий логер
 * - Установить свой логер (для продакшена)
 * - Быстрое логирование через статические методы
 */
class Logger {
public:
    /**
     * @brief Получить логер по умолчанию (GPU 0 или установленный через SetInstance)
     * @return Ссылка на ILogger
     */
    static ILogger& GetInstance();

    /**
     * @brief Получить логер для конкретного GPU (per-GPU: log_path/Logs/DRVGPU_XX/...)
     * @param gpu_id Номер GPU (0, 1, 2, ...)
     * @return Ссылка на ILogger для этого GPU
     */
    static ILogger& GetInstance(int gpu_id);

    /**
     * @brief Установить свой логер (для продакшена, только для GetInstance() без аргументов)
     * @param logger Умный указатель на ILogger
     */
    static void SetInstance(ILoggerPtr logger);

    /**
     * @brief Сбросить на стандартный логер
     */
    static void ResetToDefault();

    /**
     * @brief Логировать отладочное сообщение (в логер по умолчанию / GPU 0)
     */
    static void Debug(const std::string& component, const std::string& message);

    /**
     * @brief Логировать отладочное сообщение в лог указанного GPU
     */
    static void Debug(int gpu_id, const std::string& component, const std::string& message);

    /**
     * @brief Логировать информационное сообщение (в логер по умолчанию / GPU 0)
     */
    static void Info(const std::string& component, const std::string& message);

    /**
     * @brief Логировать информационное сообщение в лог указанного GPU
     */
    static void Info(int gpu_id, const std::string& component, const std::string& message);

    /**
     * @brief Логировать предупреждение (в логер по умолчанию / GPU 0)
     */
    static void Warning(const std::string& component, const std::string& message);

    /**
     * @brief Логировать предупреждение в лог указанного GPU
     */
    static void Warning(int gpu_id, const std::string& component, const std::string& message);

    /**
     * @brief Логировать ошибку (в логер по умолчанию / GPU 0)
     */
    static void Error(const std::string& component, const std::string& message);

    /**
     * @brief Логировать ошибку в лог указанного GPU
     */
    static void Error(int gpu_id, const std::string& component, const std::string& message);

    /**
     * @brief Проверить, включено ли логирование
     */
    static bool IsEnabled();

    /**
     * @brief Включить логирование
     */
    static void Enable();

    /**
     * @brief Выключить логирование (production mode)
     */
    static void Disable();

private:
    /// Текущий логер (по умолчанию DefaultLogger)
    static ILoggerPtr current_logger_;
};

// ════════════════════════════════════════════════════════════════════════════
// Макросы логирования
// ═══════════════════════════════════════════════════════════════════════════=

#ifdef NDEBUG
    // ═══════════════════════════════════════════════════════════════════════
    // Release сборка: DEBUG отключён, остальные уровни активны
    // ═══════════════════════════════════════════════════════════════════════

    #define DRVGPU_LOG_DEBUG(component, message) ((void)0)
    #define DRVGPU_LOG_DEBUG_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Debug(gpu_id, component, message); } while (0)
    #define DRVGPU_LOG_INFO(component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Info(component, message); } while (0)
    #define DRVGPU_LOG_INFO_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Info(gpu_id, component, message); } while (0)
    #define DRVGPU_LOG_WARNING(component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Warning(component, message); } while (0)
    #define DRVGPU_LOG_WARNING_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Warning(gpu_id, component, message); } while (0)
    #define DRVGPU_LOG_ERROR(component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Error(component, message); } while (0)
    #define DRVGPU_LOG_ERROR_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Error(gpu_id, component, message); } while (0)

#else
    // ═══════════════════════════════════════════════════════════════════════
    // Debug сборка: все уровни активны
    // ═══════════════════════════════════════════════════════════════════════

    #define DRVGPU_LOG_DEBUG(component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Debug(component, message); } while (0)
    #define DRVGPU_LOG_DEBUG_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Debug(gpu_id, component, message); } while (0)
    #define DRVGPU_LOG_INFO(component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Info(component, message); } while (0)
    #define DRVGPU_LOG_INFO_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Info(gpu_id, component, message); } while (0)
    #define DRVGPU_LOG_WARNING(component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Warning(component, message); } while (0)
    #define DRVGPU_LOG_WARNING_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Warning(gpu_id, component, message); } while (0)
    #define DRVGPU_LOG_ERROR(component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Error(component, message); } while (0)
    #define DRVGPU_LOG_ERROR_GPU(gpu_id, component, message) \
        do { if (drv_gpu_lib::Logger::IsEnabled()) drv_gpu_lib::Logger::Error(gpu_id, component, message); } while (0)

#endif // NDEBUG

// ════════════════════════════════════════════════════════════════════════════
// Устаревшие макросы (для совместимости)
// ═══════════════════════════════════════════════════════════════════════════=

// TODO(GPUProfiler-refactor): удалить deprecated alias после OOP рефакторинга
// Не используется в коде (проверено grep 2026-04-16)
// #define DRVGPU_LOG DRVGPU_LOG_INFO

} // namespace drv_gpu_lib
