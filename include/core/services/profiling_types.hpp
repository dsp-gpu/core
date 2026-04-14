#pragma once

/**
 * @file profiling_types.hpp
 * @brief Типы данных для профилирования GPU (OpenCL, ROCm)
 *
 * Содержит:
 * - ProfilingDataBase — общие 5 полей времени
 * - OpenCLProfilingData — нативный формат OpenCL
 * - ROCmProfilingData — нативный формат ROCm/HIP
 * - GPUReportInfo — информация о GPU для отчёта
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-08
 */

#include "../common/backend_type.hpp"

#include <string>
#include <cstdint>
#include <map>
#include <vector>
#include <variant>

namespace drv_gpu_lib {

// ============================================================================
// Profiling data types (OpenCL 5 + ROCm, наследование)
// ============================================================================

/**
 * @struct ProfilingDataBase
 * @brief Общие 5 полей времени для OpenCL и ROCm (одна таблица)
 *
 * Соответствие полей OpenCL:
 * - queued_ns   = CL_PROFILING_COMMAND_QUEUED
 * - submit_ns   = CL_PROFILING_COMMAND_SUBMIT
 * - start_ns    = CL_PROFILING_COMMAND_START
 * - end_ns      = CL_PROFILING_COMMAND_END
 * - complete_ns = CL_PROFILING_COMMAND_COMPLETE
 */
struct ProfilingDataBase {
    uint64_t queued_ns   = 0;  ///< Команда попала в очередь хоста
    uint64_t submit_ns   = 0;  ///< Команда отправлена на GPU
    uint64_t start_ns    = 0;  ///< Кернел начал выполняться
    uint64_t end_ns      = 0;  ///< Кернел закончил выполняться
    uint64_t complete_ns = 0;  ///< Данные выгружены/доступны
};

/**
 * @struct OpenCLProfilingData
 * @brief Нативный формат OpenCL (5 параметров cl_profiling_info)
 */
struct OpenCLProfilingData : ProfilingDataBase {};

/**
 * @struct ROCmProfilingData
 * @brief Нативный формат ROCm/HIP: база + дополнительные параметры
 *
 * Дополнительные поля:
 * - domain — область профилирования (0=HIP API, 1=HIP Activity, 2=HSA)
 * - kind — тип операции (0=кернел, 1=копирование, 2=барьер, 3=маркер)
 * - op — код конкретной HIP операции
 * - correlation_id — связь между API вызовом и выполнением
 * - device_id — ID устройства GPU
 * - queue_id — ID очереди/потока (stream)
 * - bytes — объём переданных данных
 * - kernel_name — имя кернела
 * - op_string — строка операции
 * - counters — аппаратные счётчики производительности
 */
struct ROCmProfilingData : ProfilingDataBase {
    uint32_t domain = 0;
    uint32_t kind   = 0;
    uint32_t op     = 0;
    uint64_t correlation_id = 0;
    int      device_id      = 0;
    uint64_t queue_id       = 0;
    size_t   bytes          = 0;
    std::string kernel_name;
    std::string op_string;
    std::map<std::string, double> counters;
};

/// Вариант измерения: OpenCL или ROCm (конвертация в воркере)
using ProfilingTimeVariant = std::variant<OpenCLProfilingData, ROCmProfilingData>;

/**
 * @brief Собрать OpenCLProfilingData из длительности в мс
 * @param duration_ms Длительность в миллисекундах
 * @return OpenCLProfilingData с заполненными полями
 *
 * Используется для тестов и fallback без cl_event.
 *
 * ВАЖНО: все поля кроме start_ns/end_ns устанавливаются в end_ns.
 * В результате все производные задержки (queue_delay, submit_delay, complete_delay)
 * в EventStats::UpdateFull() будут равны нулю — только exec_time будет ненулевым.
 */
inline OpenCLProfilingData MakeOpenCLFromDurationMs(double duration_ms) {
    OpenCLProfilingData d{};
    d.end_ns = static_cast<uint64_t>(duration_ms * 1e6);
    d.start_ns = 0;
    d.queued_ns = d.submit_ns = d.complete_ns = d.end_ns;
    return d;
}

// ============================================================================
// GPU Report Info - информация о GPU для шапки отчёта
// ============================================================================

/**
 * @struct GPUReportInfo
 * @brief Информация о GPU для заголовка отчёта профилирования
 *
 * На ОДНОЙ GPU могут работать сразу OpenCL И ROCm!
 * Информация о драйверах хранится в векторе drivers:
 *   drivers[0] = OpenCL info
 *   drivers[1] = ROCm info (если есть)
 *
 * Заполняется при инициализации библиотеки и используется в PrintReport/ExportMarkdown
 */
struct GPUReportInfo {
    std::string gpu_name;           ///< "AMD Radeon RX 6700 XT"
    BackendType backend_type = BackendType::OPENCL;  ///< OpenCL / ROCm / OPENCLandROCm
    size_t global_mem_mb = 0;       ///< Глобальная память (MB)

    /**
     * @brief Вектор драйверов: drivers[0]=OpenCL, drivers[1]=ROCm, ...
     *
     * Формат map для OpenCL:
     *   map["driver_type"] = "OpenCL"
     *   map["version"] = "3.0"
     *   map["driver_version"] = "23.10.2"
     *   map["platform_name"] = "AMD Accelerated Parallel Processing"
     *   map["vendor"] = "AMD"
     *
     * Формат map для ROCm:
     *   map["driver_type"] = "ROCm"
     *   map["version"] = "5.4.3"
     *   map["driver_version"] = "amdgpu 6.1.0"
     *   map["hip_version"] = "5.4.22801"
     *   map["hip_runtime"] = "5.4.22801-1"
     */
    std::vector<std::map<std::string, std::string>> drivers;

    /// Получить строку драйверов для отчёта
    std::string GetDriversString() const {
        std::string result;
        for (const auto& drv : drivers) {
            if (!result.empty()) result += "\n";
            auto it_type = drv.find("driver_type");
            if (it_type != drv.end()) {
                result += "[" + it_type->second + "] ";
                auto it_ver = drv.find("version");
                if (it_ver != drv.end()) result += "Версия: " + it_ver->second + " | ";
                auto it_drv = drv.find("driver_version");
                if (it_drv != drv.end()) result += "Драйвер: " + it_drv->second;
                if (it_type->second == "ROCm") {
                    auto it_hip = drv.find("hip_version");
                    if (it_hip != drv.end()) result += " | HIP: " + it_hip->second;
                } else {
                    auto it_plat = drv.find("platform_name");
                    if (it_plat != drv.end()) {
                        std::string plat = it_plat->second;
                        if (plat.length() > 20) plat = plat.substr(0, 20) + "...";
                        result += " | Платформа: " + plat;
                    }
                }
            }
        }
        return result;
    }

    /// Получить строку backend для отчёта (краткая)
    std::string GetBackendString() const {
        std::string result;
        for (const auto& drv : drivers) {
            auto it_type = drv.find("driver_type");
            if (it_type != drv.end()) {
                if (!result.empty()) result += " + ";
                result += it_type->second;
                auto it_ver = drv.find("version");
                if (it_ver != drv.end()) result += " " + it_ver->second;
            }
        }
        if (result.empty()) {
            switch (backend_type) {
                case BackendType::OPENCL: result = "OpenCL"; break;
                case BackendType::ROCm: result = "ROCm"; break;
                case BackendType::OPENCLandROCm: result = "OpenCL + ROCm"; break;
                default: result = "Auto";
            }
        }
        return result;
    }
};

} // namespace drv_gpu_lib
