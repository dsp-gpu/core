#pragma once

/**
 * @file i_data_sink.hpp
 * @brief IDataSink — универсальный интерфейс приёмников данных
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Единый интерфейс для всех приёмников вывода данных:
 *   - Консоль (ConsoleOutput)
 *   - Файловое логирование (Logger / DefaultLogger)
 *   - Профилирование (GPUProfiler)
 *   - БД (будущий DBSink)
 *
 * ПАТТЕРН: Strategy + Observer
 *   У сервисов может быть несколько приёмников.
 *   Каждый приёмник обрабатывает данные независимо.
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   class MyCustomSink : public IDataSink {
 *       void Write(const DataRecord& record) override {
 *           // Отправка в вашу систему мониторинга
 *           myMonitor.send(record.gpu_id, record.message);
 *       }
 *   };
 *
 *   // Подключение к логеру:
 *   Logger::AddSink(std::make_shared<MyCustomSink>());
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <string>
#include <cstdint>
#include <chrono>
#include <memory>

namespace drv_gpu_lib {

// ============================================================================
// DataRecord — универсальная запись данных для всех приёмников
// ============================================================================

/**
 * @struct DataRecord
 * @brief Универсальная запись данных, передаваемая во все реализации IDataSink
 *
 * Содержит всё необходимое для любого типа приёмника:
 * - Идентификация GPU (gpu_id)
 * - Имя модуля-источника
 * - Уровень логирования
 * - Текст сообщения
 * - Временная метка
 * - Опциональные числовые данные (для профилирования)
 */
struct DataRecord {
    /// Индекс устройства GPU (с 0, -1 = без привязки к GPU)
    int gpu_id = -1;

    /// Имя модуля-источника (напр., "AntennaFFT", "OpenCLBackend", "MemoryManager")
    std::string module_name;

    /// Уровень логирования / тип записи
    enum class Level : uint8_t {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
        PROFILING = 4,  // Уровень для данных профилирования
        METRIC = 5      // Уровень для числовых метрик
    };
    Level level = Level::INFO;

    /// Текст сообщения (для человека)
    std::string message;

    /// Временная метка (устанавливается при создании)
    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();

    /// Опциональное числовое значение (для профилирования: duration_ms, memory_bytes и т.д.)
    double value = 0.0;

    /// Опциональное имя события (для профилирования: "FFT", "MemAlloc", "KernelExec")
    std::string event_name;
};

// ============================================================================
// IDataSink — абстрактный интерфейс вывода данных
// ============================================================================

/**
 * @interface IDataSink
 * @brief Абстрактный интерфейс для всех приёмников вывода данных
 *
 * Реализации:
 * - ConsoleSink (ConsoleOutput) — форматированный вывод в stdout
 * - FileSink (DefaultLogger) — вывод в файл через plog
 * - ProfilingSink (GPUProfiler) — агрегация данных профилирования
 * - DBSink (будущее) — вывод в БД
 *
 * Потокобезопасность:
 *   Реализации ДОЛЖНЫ быть потокобезопасны, т.к. могут вызываться
 *   из нескольких рабочих потоков GPU одновременно.
 */
class IDataSink {
public:
    /// Виртуальный деструктор
    virtual ~IDataSink() = default;

    /**
     * @brief Записать запись данных в этот приёмник
     * @param record Запись данных для обработки
     *
     * ВАЖНО: Метод должен быть потокобезопасным!
     * Вызывается из рабочего потока асинхронного сервиса,
     * несколько приёмников могут вызываться параллельно.
     */
    virtual void Write(const DataRecord& record) = 0;

    /**
     * @brief Сбросить буферизованные данные
     *
     * Вызывается при остановке сервиса или когда нужен
     * немедленный вывод.
     */
    virtual void Flush() = 0;

    /**
     * @brief Получить человекочитаемое имя приёмника
     * @return Имя приёмника (напр., "ConsoleSink", "FileSink_GPU_00")
     */
    virtual std::string GetName() const = 0;

    /**
     * @brief Проверить, включён ли приёмник
     * @return true, если приёмник активен и обрабатывает записи
     */
    virtual bool IsEnabled() const = 0;

    /**
     * @brief Включить или отключить приёмник
     * @param enabled true — включить, false — отключить
     */
    virtual void SetEnabled(bool enabled) = 0;
};

/// Умный указатель на IDataSink
using IDataSinkPtr = std::shared_ptr<IDataSink>;

} // namespace drv_gpu_lib
