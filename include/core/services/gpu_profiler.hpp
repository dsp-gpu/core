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
    bool ExportJSON(const std::string& file_path) const {
        std::scoped_lock lock(stats_mutex_, gpu_info_mutex_);

        // Экранирование спецсимволов JSON в строках (защита от injection)
        auto escJson = [](const std::string& s) -> std::string {
            std::string out;
            out.reserve(s.size() + 8);
            for (char c : s) {
                if      (c == '"')  out += "\\\"";
                else if (c == '\\') out += "\\\\";
                else if (c == '\n') out += "\\n";
                else if (c == '\r') out += "\\r";
                else if (c == '\t') out += "\\t";
                else                out += c;
            }
            return out;
        };

        auto fmtD = [](double val) -> std::string {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << val;
            return oss.str();
        };

        try {
            std::ofstream file(file_path, std::ios::out | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "[GPUProfiler] Cannot create file: " << file_path << "\n";
                return false;
            }

            file << "{\n";
            file << "  \"timestamp\": \"" << GetCurrentDateTimeString() << "\",\n";

            // GPU array
            file << "  \"gpus\": [\n";
            bool first_gpu = true;
            for (const auto& [gpu_id, modules] : stats_) {
                if (!first_gpu) file << ",\n";
                first_gpu = false;

                file << "    {\n";
                file << "      \"gpu_id\": " << gpu_id << ",\n";

                // GPU info
                GPUReportInfo info;
                auto it = gpu_info_.find(gpu_id);
                if (it != gpu_info_.end()) {
                    info = it->second;
                }
                file << "      \"gpu_name\": \"" << escJson(info.gpu_name.empty() ? "Unknown" : info.gpu_name) << "\",\n";
                file << "      \"memory_mb\": " << info.global_mem_mb << ",\n";

                // Drivers array
                file << "      \"drivers\": [\n";
                bool first_drv = true;
                for (const auto& drv : info.drivers) {
                    if (!first_drv) file << ",\n";
                    first_drv = false;
                    file << "        {";
                    bool first_field = true;
                    for (const auto& [key, val] : drv) {
                        if (!first_field) file << ", ";
                        first_field = false;
                        file << "\"" << escJson(key) << "\": \"" << escJson(val) << "\"";
                    }
                    file << "}";
                }
                file << "\n      ],\n";

                // Modules
                file << "      \"modules\": [\n";
                bool first_module = true;
                for (const auto& [mod_name, mod_stats] : modules) {
                    if (!first_module) file << ",\n";
                    first_module = false;

                    file << "        {\n";
                    file << "          \"name\": \"" << escJson(mod_name) << "\",\n";
                    file << "          \"run_count\": " << mod_stats.GetRunCount() << ",\n";
                    file << "          \"avg_run_time_ms\": " << fmtD(mod_stats.GetAvgRunTimeMs()) << ",\n";

                    // Events
                    file << "          \"events\": [\n";
                    bool first_event = true;
                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        if (!first_event) file << ",\n";
                        first_event = false;

                        file << "            {\n";
                        file << "              \"name\": \"" << escJson(evt_name) << "\",\n";
                        file << "              \"calls\": " << evt_stats.total_calls << ",\n";
                        file << "              \"total_ms\": " << fmtD(evt_stats.total_time_ms) << ",\n";
                        file << "              \"avg_ms\": " << fmtD(evt_stats.GetAvgTimeMs()) << ",\n";
                        file << "              \"min_ms\": " << fmtD(evt_stats.min_time_ms == std::numeric_limits<double>::max() ? 0.0 : evt_stats.min_time_ms) << ",\n";
                        file << "              \"max_ms\": " << fmtD(evt_stats.max_time_ms) << ",\n";
                        // Детальные тайминги
                        file << "              \"queue_delay_avg_ms\": " << fmtD(evt_stats.queue_delay.GetAvgMs()) << ",\n";
                        file << "              \"submit_delay_avg_ms\": " << fmtD(evt_stats.submit_delay.GetAvgMs()) << ",\n";
                        file << "              \"exec_time_avg_ms\": " << fmtD(evt_stats.exec_time.GetAvgMs()) << ",\n";
                        file << "              \"complete_delay_avg_ms\": " << fmtD(evt_stats.complete_delay.GetAvgMs()) << "\n";
                        file << "            }";
                    }
                    file << "\n          ]\n";
                    file << "        }";
                }
                file << "\n      ]\n";
                file << "    }";
            }
            file << "\n  ]\n";
            file << "}\n";

            file.close();
            return true;

        } catch (const std::exception& e) {
            std::cerr << "[GPUProfiler] Export error: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * @brief Вывести сводку профилирования в stdout
     */
    void PrintSummary() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════╗\n";
        std::cout << "║              GPU Profiling Summary                  ║\n";
        std::cout << "╚══════════════════════════════════════════════════════╝\n";

        for (const auto& [gpu_id, modules] : stats_) {
            std::cout << "\n  GPU " << gpu_id << ":\n";

            for (const auto& [mod_name, mod_stats] : modules) {
                std::cout << "    Module: " << mod_name
                          << " (total: " << std::fixed << std::setprecision(1)
                          << mod_stats.GetTotalTimeMs() << " ms, "
                          << mod_stats.GetTotalCalls() << " calls)\n";

                for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                    std::cout << "      " << std::left << std::setw(25) << evt_name
                              << " calls=" << std::setw(6) << evt_stats.total_calls
                              << " avg=" << std::setw(8) << std::fixed << std::setprecision(2)
                              << evt_stats.GetAvgTimeMs() << "ms"
                              << " min=" << std::setw(8)
                              << (evt_stats.min_time_ms == std::numeric_limits<double>::max()
                                  ? 0.0 : evt_stats.min_time_ms) << "ms"
                              << " max=" << std::setw(8) << evt_stats.max_time_ms << "ms\n";
                }
            }
        }
        std::cout << "\n";
    }

    /**
     * @brief Вывести отчёт профилирования с шапкой с информацией о GPU
     *
     * Таблица со всеми 5 полями времени ProfilingDataBase:
     * Очередь, Отправка, Старт, Конец, Готово
     */
    void PrintReport() const {
        std::scoped_lock lock(stats_mutex_, gpu_info_mutex_);

        const int W = 110;

        auto pad = [](const std::string& s, int width) -> std::string {
            if (static_cast<int>(s.size()) >= width) return s.substr(0, width);
            return s + std::string(width - s.size(), ' ');
        };

        auto fmtD = [](double val, int prec = 3) -> std::string {
            if (val == 0.0) return "-";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec) << val;
            return oss.str();
        };

        std::cout << "\n";
        std::cout << "+" << std::string(W - 2, '=') << "+\n";
        std::cout << "|" << pad("              ОТЧЁТ ПРОФИЛИРОВАНИЯ GPU", W - 2) << "|\n";
        std::cout << "|" << pad("  Дата: " + GetCurrentDateTimeString(), W - 2) << "|\n";
        std::cout << "+" << std::string(W - 2, '=') << "+\n";

        for (const auto& [gpu_id, modules] : stats_) {
            GPUReportInfo info;
            auto it = gpu_info_.find(gpu_id);
            if (it != gpu_info_.end()) {
                info = it->second;
            }

            // Шапка GPU
            std::cout << "|" << pad("  GPU " + std::to_string(gpu_id) + ": " +
                (info.gpu_name.empty() ? "Unknown" : info.gpu_name), W - 2) << "|\n";
            if (info.global_mem_mb > 0) {
                std::cout << "|" << pad("  Память: " + std::to_string(info.global_mem_mb) + " MB", W - 2) << "|\n";
            }

            // Драйверы из vector<map>
            std::cout << "|" << pad("  Драйверы:", W - 2) << "|\n";
            for (const auto& drv : info.drivers) {
                std::string drv_line = "    ";
                auto it_type = drv.find("driver_type");
                if (it_type != drv.end()) {
                    drv_line += "[" + it_type->second + "] ";
                    auto it_ver = drv.find("version");
                    if (it_ver != drv.end()) drv_line += "Версия: " + it_ver->second + " | ";
                    auto it_drv = drv.find("driver_version");
                    if (it_drv != drv.end()) drv_line += "Драйвер: " + it_drv->second;
                    if (it_type->second == "ROCm") {
                        auto it_hip = drv.find("hip_version");
                        if (it_hip != drv.end()) drv_line += " | HIP: " + it_hip->second;
                    } else {
                        auto it_plat = drv.find("platform_name");
                        if (it_plat != drv.end()) {
                            std::string plat = it_plat->second;
                            if (plat.length() > 25) plat = plat.substr(0, 25) + "...";
                            drv_line += " | Платформа: " + plat;
                        }
                    }
                }
                std::cout << "|" << pad(drv_line, W - 2) << "|\n";
            }
            if (info.drivers.empty()) {
                std::cout << "|" << pad("    [нет информации о драйверах]", W - 2) << "|\n";
            }
            std::cout << "+" << std::string(W - 2, '-') << "+\n";

            // Строки данных — разделение по типу таблицы (OpenCL и ROCm)
            for (const auto& [mod_name, mod_stats] : modules) {
                bool is_rocm_module = HasModuleROCmData(mod_stats);

                if (is_rocm_module) {
                    // ========== ROCm ТАБЛИЦА (расширенная) ==========
                    std::cout << "| [ROCm] Модуль: " << pad(mod_name, W - 19) << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";

                    // Шапка ROCm (расширенная)
                    std::cout << "| " << std::left << std::setw(16) << "Событие"
                              << "| " << std::setw(5) << "N"
                              << "| " << std::setw(6) << "Домен"
                              << "| " << std::setw(5) << "Тип"
                              << "| " << std::setw(5) << "Опер"
                              << "| " << std::setw(7) << "УстрID"
                              << "| " << std::setw(8) << "ОчерID"
                              << "| " << std::setw(10) << "Байты(MB)"
                              << "| " << std::setw(10) << "Старт(мс)"
                              << "| " << std::setw(10) << "Конец(мс)"
                              << "| " << std::setw(10) << "Время(мс)"
                              << "|\n";
                    std::cout << "+" << std::string(17, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(7, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(8, '-')
                              << "+" << std::string(9, '-')
                              << "+" << std::string(11, '-')
                              << "+" << std::string(11, '-')
                              << "+" << std::string(11, '-')
                              << "+" << std::string(11, '-')
                              << "+\n";

                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        // Основная строка ROCm
                        std::cout << "| " << std::left << std::setw(16) << evt_name.substr(0, 15)
                                  << "| " << std::right << std::setw(4) << evt_stats.total_calls << " "
                                  << "| " << std::setw(5) << evt_stats.last_domain << " "
                                  << "| " << std::setw(4) << evt_stats.last_kind << " "
                                  << "| " << std::setw(4) << evt_stats.last_op << " "
                                  << "| " << std::setw(6) << evt_stats.last_device_id << " "
                                  << "| " << std::setw(7) << evt_stats.last_queue_id << " "
                                  << "| " << std::setw(9) << (evt_stats.total_bytes / (1024*1024)) << " "
                                  << "| " << std::setw(9) << fmtD(evt_stats.start.GetAvgMs()) << " "
                                  << "| " << std::setw(9) << fmtD(evt_stats.end.GetAvgMs()) << " "
                                  << "| " << std::setw(9) << fmtD(evt_stats.exec_time.GetAvgMs()) << " "
                                  << "|\n";

                        // Имя кернела (если есть)
                        if (!evt_stats.last_kernel_name.empty()) {
                            std::cout << "|   Ядро: " << pad(evt_stats.last_kernel_name, W - 12) << "|\n";
                        }
                        // Строка операции (если есть)
                        if (!evt_stats.last_op_string.empty()) {
                            std::cout << "|   Опер: " << pad(evt_stats.last_op_string, W - 12) << "|\n";
                        }
                    }

                    // Промежуточный итог по модулю (ROCm)
                    // N = число прогонов; Время = среднее одного прогона (сумма avg по событиям)
                    std::cout << "| " << std::left << std::setw(16) << "--- ИТОГО ---"
                              << "| " << std::right << std::setw(4) << mod_stats.GetRunCount() << " "
                              << "| " << std::string(5, ' ') << " "
                              << "| " << std::string(4, ' ') << " "
                              << "| " << std::string(4, ' ') << " "
                              << "| " << std::string(6, ' ') << " "
                              << "| " << std::string(7, ' ') << " "
                              << "| " << std::string(9, ' ') << " "
                              << "| " << std::string(9, ' ') << " "
                              << "| " << std::string(9, ' ') << " "
                              << "| " << std::setw(9) << fmtD(mod_stats.GetAvgRunTimeMs(), 3) << " "
                              << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";

                } else {
                    // ========== OpenCL ТАБЛИЦА (стандартная, 5 полей) ==========
                    std::cout << "| [OpenCL] Модуль: " << pad(mod_name, W - 21) << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";

                    // Шапка OpenCL (дельты между этапами, мс)
                    std::cout << "| " << std::left << std::setw(16) << "Событие"
                              << "| " << std::setw(5) << "N"
                              << "| " << std::setw(12) << "В очереди"
                              << "| " << std::setw(12) << "Запуск"
                              << "| " << std::setw(12) << "Выполн."
                              << "| " << std::setw(12) << "Заверш."
                              << "| " << std::setw(12) << "Всего"
                              << "|\n";
                    std::cout << "+" << std::string(17, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+\n";

                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        // Основная строка OpenCL: дельты между этапами (мс)
                        std::cout << "| " << std::left << std::setw(16) << evt_name.substr(0, 15)
                                  << "| " << std::right << std::setw(4) << evt_stats.total_calls << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.queue_delay.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.submit_delay.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.exec_time.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.complete_delay.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.GetAvgTimeMs()) << " "
                                  << "|\n";
                    }

                    // Промежуточный итог по модулю (OpenCL)
                    // N = число прогонов; Всего = среднее время одного прогона (сумма avg по событиям)
                    std::cout << "| " << std::left << std::setw(16) << "--- ИТОГО ---"
                              << "| " << std::right << std::setw(4) << mod_stats.GetRunCount() << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::setw(11) << fmtD(mod_stats.GetAvgRunTimeMs(), 3) << " "
                              << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";
                }
            }

            std::cout << "+" << std::string(W - 2, '=') << "+\n";
        }

        PrintLegend();
    }

    /**
     * @brief Печать легенды (расшифровка колонок)
     */
    void PrintLegend() const {
        std::cout << "\n";
        std::cout << "+--- ЛЕГЕНДА ---+\n";
        std::cout << "| Время в миллисекундах (мс), усреднённое значение                           |\n";
        std::cout << "+---------------+------------------------------------------------------------+\n";
        std::cout << "| В очереди     | Ожидание в очереди хоста (submit - queued)                 |\n";
        std::cout << "| Запуск        | Задержка запуска на GPU (start - submit)                   |\n";
        std::cout << "| Выполн.       | Время выполнения кернела (end - start)                     |\n";
        std::cout << "| Заверш.       | Задержка завершения (complete - end)                       |\n";
        std::cout << "| Всего         | Среднее время операции (end - start) на один вызов         |\n";
        std::cout << "| ИТОГО N       | Число прогонов бенчмарка                                   |\n";
        std::cout << "| ИТОГО Всего   | Среднее время одного прогона (сумма Всего по событиям)     |\n";
        std::cout << "+---------------+------------------------------------------------------------+\n";

        // ROCm легенда (показываем если есть ROCm данные)
        // Используем _NoLock версию, т.к. PrintReport() уже держит stats_mutex_
        if (HasAnyROCmDataGlobal_NoLock()) {
            std::cout << "| [ROCm поля]                                                                |\n";
            std::cout << "+---------------+------------------------------------------------------------+\n";
            std::cout << "| Домен         | Область профилирования (0=HIP API, 1=Activity, 2=HSA)      |\n";
            std::cout << "| Тип           | Тип операции (0=kernel, 1=copy, 2=barrier, 3=marker)       |\n";
            std::cout << "| Опер          | Код HIP операции                                           |\n";
            std::cout << "| КоррID        | Correlation ID - связь API вызова и выполнения             |\n";
            std::cout << "| УстрID        | ID устройства GPU                                          |\n";
            std::cout << "| ОчерID        | ID очереди/потока (stream)                                 |\n";
            std::cout << "| Байты         | Объём переданных данных                                    |\n";
            std::cout << "| Ядро          | Имя кернела                                                |\n";
            std::cout << "+---------------+------------------------------------------------------------+\n";
        }
    }

    /**
     * @brief Экспорт отчёта профилирования в Markdown-файл
     * @param file_path Путь к выходному .md файлу
     * @return true при успешном экспорте
     */
    bool ExportMarkdown(const std::string& file_path) const {
        std::scoped_lock lock(stats_mutex_, gpu_info_mutex_);

        auto fmtD = [](double val, int prec = 3) -> std::string {
            if (val == 0.0) return "-";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec) << val;
            return oss.str();
        };

        try {
            std::ofstream file(file_path, std::ios::out | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "[GPUProfiler] Cannot create MD file: " << file_path << "\n";
                return false;
            }

            file << "# Отчёт профилирования GPU\n\n";
            file << "**Дата:** " << GetCurrentDateTimeString() << "\n\n";

            for (const auto& [gpu_id, modules] : stats_) {
                GPUReportInfo info;
                auto it = gpu_info_.find(gpu_id);
                if (it != gpu_info_.end()) {
                    info = it->second;
                }

                file << "## GPU " << gpu_id << "\n\n";
                file << "### Информация о системе\n\n";
                file << "| Параметр | Значение |\n";
                file << "|----------|----------|\n";
                file << "| **GPU** | " << (info.gpu_name.empty() ? "Unknown" : info.gpu_name) << " |\n";
                if (info.global_mem_mb > 0)
                    file << "| **Память** | " << info.global_mem_mb << " MB |\n";

                // Драйверы
                for (size_t i = 0; i < info.drivers.size(); ++i) {
                    const auto& drv = info.drivers[i];
                    auto it_type = drv.find("driver_type");
                    if (it_type != drv.end()) {
                        std::string drv_info = "[" + it_type->second + "] ";
                        auto it_ver = drv.find("version");
                        if (it_ver != drv.end()) drv_info += "Версия: " + it_ver->second;
                        auto it_drv = drv.find("driver_version");
                        if (it_drv != drv.end()) drv_info += " \\| Драйвер: " + it_drv->second;
                        if (it_type->second == "ROCm") {
                            auto it_hip = drv.find("hip_version");
                            if (it_hip != drv.end()) drv_info += " \\| HIP: " + it_hip->second;
                        } else {
                            auto it_plat = drv.find("platform_name");
                            if (it_plat != drv.end()) drv_info += " \\| Платформа: " + it_plat->second;
                        }
                        file << "| **Драйвер " << (i + 1) << "** | " << drv_info << " |\n";
                    }
                }
                file << "\n";

                // Таблица
                file << "### Результаты профилирования\n\n";
                file << "| Модуль | Событие | N | В очереди | Запуск | Выполн. | Заверш. | Всего |\n";
                file << "|--------|---------|--:|----------:|-------:|--------:|--------:|------:|\n";

                for (const auto& [mod_name, mod_stats] : modules) {
                    bool first_event = true;
                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        file << "| " << (first_event ? mod_name : "")
                             << " | " << evt_name
                             << " | " << evt_stats.total_calls
                             << " | " << fmtD(evt_stats.queue_delay.GetAvgMs())
                             << " | " << fmtD(evt_stats.submit_delay.GetAvgMs())
                             << " | " << fmtD(evt_stats.exec_time.GetAvgMs())
                             << " | " << fmtD(evt_stats.complete_delay.GetAvgMs())
                             << " | " << fmtD(evt_stats.GetAvgTimeMs())
                             << " |\n";
                        first_event = false;
                    }
                    file << "| | **ИТОГО** | " << mod_stats.GetRunCount()
                         << " | | | | | " << std::fixed << std::setprecision(3) << mod_stats.GetAvgRunTimeMs()
                         << " |\n";
                }

                file << "\n";
            }

            // Легенда
            file << "---\n\n";
            file << "## Легенда\n\n";
            file << "| Колонка | Описание |\n";
            file << "|---------|----------|\n";
            file << "| **N** | Количество вызовов |\n";
            file << "| **В очереди** | Ожидание в очереди хоста (submit - queued) |\n";
            file << "| **Запуск** | Задержка запуска на GPU (start - submit) |\n";
            file << "| **Выполн.** | Время выполнения кернела (end - start) |\n";
            file << "| **Заверш.** | Задержка завершения (complete - end) |\n";
            file << "| **Всего** | Общее время операции (end - start) |\n";
            file << "\n*Время в миллисекундах (мс), усреднённое значение*\n";

            file.close();
            return true;

        } catch (const std::exception& e) {
            std::cerr << "[GPUProfiler] MD Export error: " << e.what() << "\n";
            return false;
        }
    }

protected:
    // ========================================================================
    // Реализация AsyncServiceBase
    // ========================================================================

    /**
     * @brief Обработать одно сообщение профилирования (выполняется в рабочем потоке)
     */
    void ProcessMessage(const ProfilingMessage& msg) override {
        {
            std::lock_guard<std::mutex> lock(profile_filter_mutex_);
            if (disabled_gpus_.find(msg.gpu_id) != disabled_gpus_.end()) return;
        }

        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto& module_stats = stats_[msg.gpu_id][msg.module_name];
        module_stats.module_name = msg.module_name;
        auto& event_stats = module_stats.events[msg.event_name];
        event_stats.event_name = msg.event_name;

        std::visit([&event_stats](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ROCmProfilingData>) {
                event_stats.UpdateROCm(arg);
            } else {
                event_stats.UpdateFull(arg);
            }
        }, msg.time_);
    }

    /**
     * @brief Имя сервиса для диагностики
     */
    std::string GetServiceName() const override {
        return "GPUProfiler";
    }

private:
    // ========================================================================
    // Приватный конструктор (синглтон)
    // ========================================================================

    GPUProfiler() : enabled_(true) {}

    // Вызывать ТОЛЬКО когда stats_mutex_ уже захвачен!
    bool HasAnyROCmDataGlobal_NoLock() const {
        for (const auto& [gpu_id, modules] : stats_) {
            for (const auto& [mod, mstats] : modules) {
                for (const auto& [evt, estats] : mstats.events) {
                    if (estats.has_rocm_data) return true;
                }
            }
        }
        return false;
    }

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
