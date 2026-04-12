#pragma once

/**
 * @file default_logger.hpp
 * @brief DefaultLogger - Реализация ILogger на основе plog
 *
 * Логирование ТОЛЬКО в файл с использованием plog (header-only).
 * Автоматически создаёт структуру папок для логов.
 *
 * Поведение:
 * - GPUConfig::IsLoggingEnabled(gpu_id) == true  -> пишем в файл
 * - GPUConfig::IsLoggingEnabled(gpu_id) == false -> не логируем вообще
 * - Флаг is_logger читается из configGPU.json
 *
 * ЗАМЕНА spdlog → plog:
 * - plog — header-only, нет зависимостей (нет fmt)
 * - plog — стабильный, кросс-платформенный (Windows/Linux/macOS)
 * - plog — простой API, rolling файлы, потокобезопасность
 *
 * @author DrvGPU Team
 * @date 2026-02-01
 * @modified 2026-02-07 (spdlog → plog migration)
 */

#include "../interface/i_logger.hpp"
#include "../logger/config_logger.hpp"

// ═══════════════════════════════════════════════════════════════════════════
// plog — header-only библиотека логирования
// https://github.com/SergiusTheBest/plog
// ═══════════════════════════════════════════════════════════════════════════
#include <plog/Log.h>
#include <plog/Initializers/RollingFileInitializer.h>

#include <memory>
#include <mutex>
#include <string>
#include <map>

namespace drv_gpu_lib {

/// Максимальное количество GPU для отдельных логеров (plog требует compile-time instance ID)
constexpr int kMaxGpuLogInstances = 32;

// ═══════════════════════════════════════════════════════════════════════════
// Уровни логирования plog:
//   plog::verbose  = 6 (самый детальный)
//   plog::debug    = 5
//   plog::info     = 4
//   plog::warning  = 3
//   plog::error    = 2
//   plog::fatal    = 1
//   plog::none     = 0 (отключено)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @class DefaultLogger
 * @brief Реализация ILogger на основе plog (файловое логирование)
 *
 * Использует plog для:
 * - Логирования в файл с автоматическим созданием структуры папок
 * - Потокобезопасное логирование
 * - Rolling файлов (автоматическая ротация по размеру)
 *
 * Пример использования:
 * @code
 * // Включить логирование (по умолчанию включено)
 * ConfigLogger::GetInstance().Enable();
 *
 * // Логировать сообщения (пишутся в файл)
 * DRVGPU_LOG_INFO("DrvGPU", "Initialized successfully");
 * DRVGPU_LOG_WARNING("OpenCL", "Memory low");
 * DRVGPU_LOG_ERROR("Backend", "Failed to allocate");
 *
 * // Отключить логирование (ничего не пишется)
 * ConfigLogger::GetInstance().Disable();
 * @endcode
 */
class DefaultLogger : public ILogger {
public:
    /// Получить логер для GPU (по умолчанию gpu_id=0). Per-GPU: отдельный инстанс на каждый gpu_id.
    static DefaultLogger& GetInstance(int gpu_id = 0);

    // ═══════════════════════════════════════════════════════════════════════
    // Реализация ILogger
    // ═══════════════════════════════════════════════════════════════════════

    void Debug(const std::string& component, const std::string& message) override;
    void Info(const std::string& component, const std::string& message) override;
    void Warning(const std::string& component, const std::string& message) override;
    void Error(const std::string& component, const std::string& message) override;

    bool IsDebugEnabled() const override;
    bool IsInfoEnabled() const override;
    bool IsWarningEnabled() const override;
    bool IsErrorEnabled() const override;

    void Reset() override;

    // ═══════════════════════════════════════════════════════════════════════
    // Дополнительные методы
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Форматировать сообщение с компонентом
     * @param component Имя компонента
     * @param message Текст сообщения
     * @return Отформатированное сообщение
     */
    static std::string FormatMessage(const std::string& component,
                                      const std::string& message);

    /**
     * @brief Проверить, инициализирован ли логер
     * @return true если логер инициализирован
     */
    bool IsInitialized() const;

    /// Деструктор
    ~DefaultLogger();

    /// Конструктор (приватный; используйте GetInstance(gpu_id))
    explicit DefaultLogger(int gpu_id);

private:

    /// Инициализировать plog для этого GPU
    void Initialize();

    /// Очистить plog
    void Shutdown();

    /// Номер GPU (путь: Logs/DRVGPU_XX/)
    int gpu_id_;

    /// Флаг инициализации
    bool initialized_;

    /// Мьютекс для потокобезопасности
    mutable std::mutex mutex_;

    /// Текущий уровень логирования (plog severity)
    plog::Severity current_level_;

    /// Хранилище инстансов по gpu_id (синглтон на GPU)
    static std::map<int, std::unique_ptr<DefaultLogger>> instances_;
    static std::mutex instances_mutex_;
};

} // namespace drv_gpu_lib
