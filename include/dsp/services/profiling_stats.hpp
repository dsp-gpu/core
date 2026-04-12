#pragma once

/**
 * @file profiling_stats.hpp
 * @brief Структуры статистики профилирования GPU
 *
 * Содержит:
 * - ProfilingMessage — сообщение для очереди профилирования
 * - DetailedTimingStats — статистика для одного поля времени
 * - EventStats — агрегированная статистика для события
 * - ModuleStats — статистика для модуля
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-08
 */

#include "profiling_types.hpp"

#include <string>
#include <chrono>
#include <cstdint>
#include <map>
#include <limits>
#include <algorithm>

namespace drv_gpu_lib {

// ============================================================================
// ProfilingMessage - Message type for profiling queue
// ============================================================================

/**
 * @struct ProfilingMessage
 * @brief Одна запись: gpu_id, module, event, time_ (variant OpenCL/ROCm)
 */
struct ProfilingMessage {
    int gpu_id = 0;
    std::string module_name;
    std::string event_name;
    ProfilingTimeVariant time_;
    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();
};

// ============================================================================
// DetailedTimingStats - Aggregated statistics for one timing field
// ============================================================================

/**
 * @struct DetailedTimingStats
 * @brief Агрегированная статистика для одного поля времени (avg/min/max в мс)
 */
struct DetailedTimingStats {
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    uint64_t count = 0;

    double GetAvgMs() const {
        return count > 0 ? total_ms / static_cast<double>(count) : 0.0;
    }

    void Update(uint64_t value_ns) {
        double ms = value_ns * 1e-6;
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        count++;
    }

    double GetMinSafe() const {
        return min_ms == std::numeric_limits<double>::max() ? 0.0 : min_ms;
    }
};

// ============================================================================
// EventStats - Aggregated statistics for one event type
// ============================================================================

/**
 * @struct EventStats
 * @brief Aggregated statistics for a specific event
 *
 * Tracks min/max/avg/total for an event like "FFT_Execute"
 * Включает все 5 полей ProfilingDataBase + производные задержки
 */
struct EventStats {
    /// Event name
    std::string event_name;

    /// Total number of calls
    uint64_t total_calls = 0;

    /// Total accumulated time (ms) - основная метрика (end - start)
    double total_time_ms = 0.0;

    /// Minimum duration (ms)
    double min_time_ms = std::numeric_limits<double>::max();

    /// Maximum duration (ms)
    double max_time_ms = 0.0;

    // ========== 5 полей ProfilingDataBase (агрегированные) ==========
    DetailedTimingStats queued;     ///< Q: время постановки в очередь
    DetailedTimingStats submit;     ///< S: время отправки на устройство
    DetailedTimingStats start;      ///< St: время начала выполнения
    DetailedTimingStats end;        ///< E: время окончания выполнения
    DetailedTimingStats complete;   ///< C: время полного завершения

    // ========== Производные задержки ==========
    DetailedTimingStats queue_delay;   ///< QD: submit - queued (задержка в очереди)
    DetailedTimingStats submit_delay;  ///< SD: start - submit (задержка запуска)
    DetailedTimingStats exec_time;     ///< Ex: end - start (время выполнения)
    DetailedTimingStats complete_delay;///< CD: complete - end (задержка завершения)

    // ========== ROCm-специфичные поля (если есть) ==========
    bool has_rocm_data = false;
    uint64_t total_bytes = 0;          ///< Суммарный объём данных (bytes)
    std::string last_kernel_name;      ///< Последнее имя ядра
    std::string last_op_string;        ///< Последняя операция

    // ROCm дополнительные поля (последние значения)
    uint32_t last_domain = 0;          ///< Область профилирования (0=HIP API, 1=Activity, 2=HSA)
    uint32_t last_kind = 0;            ///< Тип операции (0=kernel, 1=copy, 2=barrier)
    uint32_t last_op = 0;              ///< Код HIP операции
    uint64_t last_correlation_id = 0;  ///< Correlation ID
    int last_device_id = 0;            ///< ID устройства
    uint64_t last_queue_id = 0;        ///< ID очереди/потока

    /// Average duration (ms) - computed on request
    double GetAvgTimeMs() const {
        return total_calls > 0 ? total_time_ms / static_cast<double>(total_calls) : 0.0;
    }

    /// Update with new measurement (legacy - только duration)
    void Update(double duration_ms) {
        total_calls++;
        total_time_ms += duration_ms;
        min_time_ms = std::min(min_time_ms, duration_ms);
        max_time_ms = std::max(max_time_ms, duration_ms);
    }

    /// Update with full ProfilingDataBase (5 полей времени)
    void UpdateFull(const ProfilingDataBase& data) {
        total_calls++;
        double duration_ms = (data.end_ns - data.start_ns) * 1e-6;
        total_time_ms += duration_ms;
        min_time_ms = std::min(min_time_ms, duration_ms);
        max_time_ms = std::max(max_time_ms, duration_ms);

        // 5 полей времени
        queued.Update(data.queued_ns);
        submit.Update(data.submit_ns);
        start.Update(data.start_ns);
        end.Update(data.end_ns);
        complete.Update(data.complete_ns);

        // Производные задержки (в наносекундах, конвертируем).
        // Защитные проверки >= нужны: некоторые драйверы возвращают 0 или некорректный
        // порядок для временных меток событий, которые не прошли все фазы очереди GPU.
        if (data.submit_ns >= data.queued_ns)
            queue_delay.Update(data.submit_ns - data.queued_ns);
        if (data.start_ns >= data.submit_ns)
            submit_delay.Update(data.start_ns - data.submit_ns);
        if (data.end_ns >= data.start_ns)
            exec_time.Update(data.end_ns - data.start_ns);
        if (data.complete_ns >= data.end_ns)
            complete_delay.Update(data.complete_ns - data.end_ns);
    }

    /// Update with ROCm data (база + доп. поля)
    void UpdateROCm(const ROCmProfilingData& data) {
        UpdateFull(data);  // 5 полей времени
        has_rocm_data = true;
        total_bytes += data.bytes;
        if (!data.kernel_name.empty()) last_kernel_name = data.kernel_name;
        if (!data.op_string.empty()) last_op_string = data.op_string;

        // Сохраняем ROCm-специфичные поля
        last_domain = data.domain;
        last_kind = data.kind;
        last_op = data.op;
        last_correlation_id = data.correlation_id;
        last_device_id = data.device_id;
        last_queue_id = data.queue_id;
    }

    double GetMinSafe() const {
        return min_time_ms == std::numeric_limits<double>::max() ? 0.0 : min_time_ms;
    }
};

// ============================================================================
// ModuleStats - Statistics for one module on one GPU
// ============================================================================

/**
 * @struct ModuleStats
 * @brief Aggregated statistics for a module on a specific GPU
 *
 * Contains per-event statistics for one module (e.g., "AntennaFFT" on GPU 0)
 */
struct ModuleStats {
    /// Module name
    std::string module_name;

    /// Per-event statistics: event_name -> EventStats
    std::map<std::string, EventStats> events;

    /// Total accumulated time across all events (сумма всех замеров)
    double GetTotalTimeMs() const {
        double total = 0.0;
        for (const auto& [name, stats] : events) {
            total += stats.total_time_ms;
        }
        return total;
    }

    /// Average time per one full run = sum of avg times across all events
    /// Это "сколько в среднем занимает один прогон модуля целиком"
    double GetAvgRunTimeMs() const {
        double total = 0.0;
        for (const auto& [name, stats] : events) {
            total += stats.GetAvgTimeMs();
        }
        return total;
    }

    /// Total calls across all events
    uint64_t GetTotalCalls() const {
        uint64_t total = 0;
        for (const auto& [name, stats] : events) {
            total += stats.total_calls;
        }
        return total;
    }

    // ПРЕДПОЛОЖЕНИЕ: все события имеют одинаковое число вызовов.
    // Верно для бенчмарков, где каждая фаза (Init→Process→Cleanup) замеряется ровно n раз.
    // Если события вызывались с разной частотой — GetRunCount() даст неверный результат.
    uint64_t GetRunCount() const {
        if (events.empty()) return 0;
        return events.begin()->second.total_calls;
    }
};

} // namespace drv_gpu_lib
