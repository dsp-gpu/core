#pragma once

/**
 * @file default_logger.hpp
 * @brief DefaultLogger — реализация ILogger на базе plog (PRIVATE)
 *
 * ============================================================================
 * ВАЖНО:
 *   Этот заголовок PRIVATE (живёт в src/, не публикуется через include/).
 *   Клиенты ядра (spectrum/stats/...) НЕ включают его — они работают с
 *   ILogger* + LoggerFactory::CreateDefault().
 *
 *   plog скрыт через PIMPL: ни одного <plog/...> в этом заголовке.
 *   plog виден ТОЛЬКО в default_logger.cpp.
 *
 * ПАТТЕРНЫ:
 *   - SOLID: SRP (одна реализация — один файл), DIP (наследует ILogger)
 *   - GoF:   Bridge / PIMPL (реализация скрыта за указателем)
 * ============================================================================
 *
 * @author DrvGPU Team
 * @date 2026-02-01
 * @modified 2026-04-14 (PIMPL — plog убран из заголовка)
 */

#include <core/interface/i_logger.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace drv_gpu_lib {

/// Максимальное количество GPU для отдельных логеров (plog требует compile-time instance ID)
constexpr int kMaxGpuLogInstances = 32;

/**
 * @class DefaultLogger
 * @brief Реализация ILogger через plog. plog скрыт PIMPL'ом.
 */
class DefaultLogger : public ILogger {
public:
    /// Получить per-GPU логер. Внутри карта инстансов по gpu_id.
    static DefaultLogger& GetInstance(int gpu_id = 0);

    // ─── Реализация ILogger ───────────────────────────────────────────────
    void Debug(const std::string& component, const std::string& message) override;
    void Info(const std::string& component, const std::string& message) override;
    void Warning(const std::string& component, const std::string& message) override;
    void Error(const std::string& component, const std::string& message) override;

    bool IsDebugEnabled() const override;
    bool IsInfoEnabled() const override;
    bool IsWarningEnabled() const override;
    bool IsErrorEnabled() const override;

    void Reset() override;

    // ─── Дополнительные методы ────────────────────────────────────────────
    static std::string FormatMessage(const std::string& component,
                                      const std::string& message);

    bool IsInitialized() const;

    /// Деструктор обязан быть в .cpp (где видна полная Impl) — PIMPL правило
    ~DefaultLogger() override;

    /// Конструктор (используйте GetInstance(gpu_id))
    explicit DefaultLogger(int gpu_id);

private:
    /// Инициализировать plog (детали в .cpp)
    void Initialize();

    /// Очистить plog (детали в .cpp)
    void Shutdown();

    /// Номер GPU (путь: Logs/DRVGPU_XX/)
    int gpu_id_;

    /// Флаг инициализации
    bool initialized_;

    /// Мьютекс для потокобезопасности
    mutable std::mutex mutex_;

    // ─── PIMPL: plog::Severity и прочие plog-детали скрыты ────────────────
    class Impl;
    std::unique_ptr<Impl> impl_;

    /// Хранилище инстансов по gpu_id (синглтон на GPU)
    static std::map<int, std::unique_ptr<DefaultLogger>> instances_;
    static std::mutex instances_mutex_;
};

} // namespace drv_gpu_lib
