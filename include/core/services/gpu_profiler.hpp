#pragma once

/**
 * @file gpu_profiler.hpp
 * @brief GPUProfiler — асинхронный синглтон сбора данных профилирования GPU
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Централизованный сбор и агрегация данных профилирования GPU.
 *   Модули отправляют записи (время ядер, операции с памятью и т.д.)
 *   через неблокирующий Enqueue. Фоновый поток агрегирует статистику.
 *
 * АРХИТЕКТУРА:
 *   Модуль GPU --> Profiler::Record(gpu_id, "FFT", 12.5ms) --> Enqueue() --+
 *                                                                           |
 *                                                                    [Рабочий поток]
 *                                                                           |
 *                                                              Агрегация (min/max/avg)
 *                                                              Экспорт в JSON
 *                                                              Уведомление наблюдателей
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   GPUProfiler::GetInstance().Start();
 *
 *   // Из любого потока GPU (неблокирующе):
 *   GPUProfiler::GetInstance().Record(0, "AntennaFFT", "FFT_Execute", data_opencl);
 *   GPUProfiler::GetInstance().Record(0, "AntennaFFT", "Padding_Kernel", data_opencl);
 *   GPUProfiler::GetInstance().Record(1, "VectorOps", "VectorAdd", data_rocm);
 *
 *   // Получить агрегированную статистику:
 *   auto stats = GPUProfiler::GetInstance().GetStats(0);
 *   auto all_stats = GPUProfiler::GetInstance().GetAllStats();
 *
 *   // Экспорт в JSON:
 *   GPUProfiler::GetInstance().ExportJSON("./Results/Profiler/2026-02-07_14-30-00.json");
 *
 *   GPUProfiler::GetInstance().Stop();
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "async_service_base.hpp"
#include "profiling_types.hpp"
#include "profiling_stats.hpp"

#include <string>
#include <chrono>
#include <map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace drv_gpu_lib {

// ============================================================================
// GPUProfiler — асинхронный сервис профилирования
// ============================================================================

/**
 * @class GPUProfiler
 * @brief Синглтон-сервис сбора данных профилирования GPU
 *
 * Наследует AsyncServiceBase<ProfilingMessage>:
 * - Фоновый рабочий поток для агрегации
 * - Неблокирующий Record() для потоков GPU
 * - Потокобезопасный доступ к статистике через GetStats()
 *
 * @ingroup grp_drvgpu
 */
class GPUProfiler : public AsyncServiceBase<ProfilingMessage> {
public:
    // ========================================================================
    // Синглтон
    // ========================================================================

    /**
     * @brief Получить экземпляр синглтона
     */
    static GPUProfiler& GetInstance() {
        static GPUProfiler instance;
        return instance;
    }

    /**
     * @brief Получить текущую дату и время в формате "YYYY-MM-DD HH:MM:SS"
     */
    static std::string GetCurrentDateTimeString() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    // ⚠️ КРИТИЧНО: Stop() в деструкторе ПРОИЗВОДНОГО класса!
    // Если Stop() вызывается только из ~AsyncServiceBase(), vtable уже
    // переключён на базовый → ProcessMessage() = pure virtual call → UB.
    // ConsoleOutput делает то же самое — каждый наследник ОБЯЗАН.
    ~GPUProfiler() {
        Stop();
    }

    // Запрет копирования (синглтон)
    GPUProfiler(const GPUProfiler&) = delete;
    GPUProfiler& operator=(const GPUProfiler&) = delete;

    // ========================================================================
    // API записи (неблокирующий)
    // ========================================================================

    /**
     * @brief Записать событие профилирования (OpenCL: 5 параметров cl_profiling_info)
     * Конвертация в duration_ms выполняется в рабочем потоке.
     */
    void Record(int gpu_id, const std::string& module,
                const std::string& event, const OpenCLProfilingData& data) {
        if (!enabled_.load(std::memory_order_acquire)) return;
        ProfilingMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.event_name = event;
        msg.time_ = data;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Записать событие профилирования (ROCm/HIP: база + доп. параметры)
     */
    void Record(int gpu_id, const std::string& module,
                const std::string& event, const ROCmProfilingData& data) {
        if (!enabled_.load(std::memory_order_acquire)) return;
        ProfilingMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.event_name = event;
        msg.time_ = data;
        Enqueue(std::move(msg));
    }

    // ========================================================================
    // Доступ к статистике (потокобезопасное чтение)
    // ========================================================================

    /**
     * @brief Получить статистику для указанной GPU
     * @param gpu_id Индекс устройства GPU
     * @return Отображение имя_модуля -> ModuleStats
     */
    std::map<std::string, ModuleStats> GetStats(int gpu_id) const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto it = stats_.find(gpu_id);
        if (it != stats_.end()) {
            return it->second;
        }
        return {};
    }

    /**
     * @brief Получить статистику по всем GPU
     * @return Отображение gpu_id -> (имя_модуля -> ModuleStats)
     */
    std::map<int, std::map<std::string, ModuleStats>> GetAllStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

    /**
     * @brief Сбросить всю собранную статистику
     */
    void Reset() {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.clear();
    }

    // ========================================================================
    // Включение/выключение
    // ========================================================================

    /**
     * @brief Включить или выключить профилирование глобально
     */
    void SetEnabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Проверить, включено ли профилирование глобально
     */
    bool IsEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Включить или выключить профилирование для указанной GPU (из конфига: is_prof)
     * @param gpu_id Индекс устройства GPU
     * @param enabled true — записывать профиль для этого GPU, false — отключить
     */
    void SetGPUEnabled(int gpu_id, bool enabled) {
        std::lock_guard<std::mutex> lock(profile_filter_mutex_);
        if (enabled) {
            disabled_gpus_.erase(gpu_id);
        } else {
            disabled_gpus_.insert(gpu_id);
        }
    }

    /**
     * @brief Проверить, включено ли профилирование для данной GPU
     */
    bool IsGPUEnabled(int gpu_id) const {
        std::lock_guard<std::mutex> lock(profile_filter_mutex_);
        return disabled_gpus_.find(gpu_id) == disabled_gpus_.end();
    }

    // ========================================================================
    // GPU Info (для шапки отчёта)
    // ========================================================================

    /**
     * @brief Установить информацию о GPU для отчёта
     * @param gpu_id Индекс устройства GPU
     * @param info Структура с информацией о GPU
     */
    void SetGPUInfo(int gpu_id, const GPUReportInfo& info) {
        std::lock_guard<std::mutex> lock(gpu_info_mutex_);
        gpu_info_[gpu_id] = info;
    }

    /**
     * @brief Получить информацию о GPU
     * @param gpu_id Индекс устройства GPU
     * @return GPUReportInfo или пустая структура, если не задано
     */
    GPUReportInfo GetGPUInfo(int gpu_id) const {
        std::lock_guard<std::mutex> lock(gpu_info_mutex_);
        auto it = gpu_info_.find(gpu_id);
        if (it != gpu_info_.end()) {
            return it->second;
        }
        return GPUReportInfo{};
    }

    /**
     * @brief Получить информацию обо всех GPU
     */
    std::map<int, GPUReportInfo> GetAllGPUInfo() const {
        std::lock_guard<std::mutex> lock(gpu_info_mutex_);
        return gpu_info_;
    }

    // ========================================================================
    // Обнаружение ROCm
    // ========================================================================

    /**
     * @brief Проверить, есть ли ROCm данные для указанной GPU
     */
    bool HasAnyROCmData(int gpu_id) const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto it = stats_.find(gpu_id);
        if (it == stats_.end()) return false;
        for (const auto& [mod, mstats] : it->second) {
            for (const auto& [evt, estats] : mstats.events) {
                if (estats.has_rocm_data) return true;
            }
        }
        return false;
    }

    /**
     * @brief Проверить, есть ли ROCm данные в модуле (внутренняя, без блокировки)
     * @param mod_stats Статистика модуля
     * @return true если хотя бы одно событие имеет ROCm данные
     */
    static bool HasModuleROCmData(const ModuleStats& mod_stats) {
        for (const auto& [evt_name, evt_stats] : mod_stats.events) {
            if (evt_stats.has_rocm_data) return true;
        }
        return false;
    }

    /**
     * @brief Проверить, есть ли ROCm данные для ЛЮБОЙ GPU (публичная)
     */
    bool HasAnyROCmDataGlobal() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return HasAnyROCmDataGlobal_NoLock();
    }

    // ========================================================================
    // Экспорт
    // ========================================================================

    /**
     * @brief Экспорт данных профилирования в JSON-файл
     * @param file_path Путь к выходному JSON-файлу
     * @return true при успешном экспорте
     */
    bool ExportJSON(const std::string& file_path) const;

    /**
     * @brief Вывести сводку профилирования в stdout
     */
    void PrintSummary() const;

    /**
     * @brief Вывести отчёт профилирования с шапкой с информацией о GPU
     *
     * Таблица со всеми 5 полями времени ProfilingDataBase:
     * Очередь, Отправка, Старт, Конец, Готово
     */
    void PrintReport() const;
    void PrintLegend() const;
    bool ExportMarkdown(const std::string& file_path) const;

protected:
    void ProcessMessage(const ProfilingMessage& msg) override;
    std::string GetServiceName() const override;

private:
    GPUProfiler() : enabled_(true) {}
    bool HasAnyROCmDataGlobal_NoLock() const;

    // ========================================================================
    // Приватные члены
    // ========================================================================

    /// Агрегированная статистика: gpu_id -> имя_модуля -> ModuleStats
    std::map<int, std::map<std::string, ModuleStats>> stats_;
    mutable std::mutex stats_mutex_;

    /// Множество отключённых по GPU
    std::unordered_set<int> disabled_gpus_;
    mutable std::mutex profile_filter_mutex_;

    /// Информация о GPU для шапок отчётов
    std::map<int, GPUReportInfo> gpu_info_;
    mutable std::mutex gpu_info_mutex_;

    std::atomic<bool> enabled_;
};

} // namespace drv_gpu_lib
