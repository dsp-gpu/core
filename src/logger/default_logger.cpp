#include "default_logger.hpp"
#include "../config/gpu_config.hpp"
#include <plog/Record.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <iostream>
#include <iomanip>

namespace plog {

// ════════════════════════════════════════════════════════════════════════════
// DrvGPUFormatter — кастомный форматтер для plog
// Формат: "YYYY-MM-DD HH:MM:SS.mmm LEVEL [Component] Message"
// Убраны: [ThreadID] [@InstanceID] (не нужны для GPU логов)
// ════════════════════════════════════════════════════════════════════════════
class DrvGPUFormatter {
public:
    static util::nstring header() {
        return util::nstring();
    }

    static util::nstring format(const Record& record) {
        tm t;
        util::localtime_s(&t, &record.getTime().time);

        util::nostringstream ss;
        // Дата: YYYY-MM-DD
        ss << t.tm_year + 1900 << PLOG_NSTR("-")
           << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_mon + 1 << PLOG_NSTR("-")
           << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_mday << PLOG_NSTR(" ");
        // Время: HH:MM:SS.mmm
        ss << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_hour << PLOG_NSTR(":")
           << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_min << PLOG_NSTR(":")
           << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_sec << PLOG_NSTR(".")
           << std::setfill(PLOG_NSTR('0')) << std::setw(3) << static_cast<int>(record.getTime().millitm) << PLOG_NSTR(" ");
        // Уровень: DEBUG/INFO/WARN/ERROR
        ss << std::setfill(PLOG_NSTR(' ')) << std::setw(5) << std::left << severityToString(record.getSeverity()) << PLOG_NSTR(" ");
        // Сообщение
        ss << record.getMessage() << PLOG_NSTR("\n");

        return ss.str();
    }
};

} // namespace plog

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// DefaultLogger Implementation - Per-GPU файловое логирование на plog
// ════════════════════════════════════════════════════════════════════════════

std::map<int, std::unique_ptr<DefaultLogger>> DefaultLogger::instances_;
std::mutex DefaultLogger::instances_mutex_;

/**
 * @brief Получить логер для данного GPU (отдельный инстанс на каждый gpu_id)
 * @param gpu_id Номер GPU (0, 1, 2, ...). Путь: log_path/Logs/DRVGPU_XX/{YYYY-MM-DD}/
 */
DefaultLogger& DefaultLogger::GetInstance(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= kMaxGpuLogInstances) {
        gpu_id = 0;
    }
    std::lock_guard<std::mutex> lock(instances_mutex_);
    auto it = instances_.find(gpu_id);
    if (it == instances_.end()) {
        it = instances_.emplace(gpu_id, std::make_unique<DefaultLogger>(gpu_id)).first;
    }
    return *it->second;
}

/**
 * @brief Конструктор DefaultLogger для конкретного GPU
 * @param gpu_id Номер GPU (путь лога: Logs/DRVGPU_XX/)
 */
DefaultLogger::DefaultLogger(int gpu_id)
    : gpu_id_(gpu_id)
    , initialized_(false)
    , current_level_(plog::debug) {
    Initialize();
}

/**
 * @brief Деструктор DefaultLogger
 *
 * Вызывает Shutdown() для корректного завершения.
 */
DefaultLogger::~DefaultLogger() {
    Shutdown();
}

// Диспетчеризация plog::init по instance ID с кастомным форматтером DrvGPUFormatter
// Формат: "YYYY-MM-DD HH:MM:SS.mmm LEVEL [Component] Message" (без ThreadID, без InstanceID)
#define DRVGPU_PLOG_INIT_CASE(N) \
    case N: plog::init<plog::DrvGPUFormatter, N>(plog::debug, log_file_path.c_str(), kMaxFileSize, kMaxFiles); break;

/**
 * @brief Инициализировать plog file logger для этого GPU
 *
 * Путь: log_path/Logs/DRVGPU_XX/{YYYY-MM-DD}/{HH-MM-SS}.log
 * Использует GetLogFilePathForGPU(gpu_id_) и CreateLogDirectoryForGPU(gpu_id_).
 */
void DefaultLogger::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return;
    }

    // Проверяем is_logger из configGPU.json для данного GPU
    if (!GPUConfig::GetInstance().IsLoggingEnabled(gpu_id_)) {
        initialized_ = true;
        return;
    }

    try {
        ConfigLogger::GetInstance().CreateLogDirectoryForGPU(gpu_id_);
        std::string log_file_path = ConfigLogger::GetInstance().GetLogFilePathForGPU(gpu_id_);

        static const size_t kMaxFileSize = 5 * 1024 * 1024;  // 5 MB
        static const int    kMaxFiles    = 3;

        switch (gpu_id_) {
            DRVGPU_PLOG_INIT_CASE(0)  DRVGPU_PLOG_INIT_CASE(1)  DRVGPU_PLOG_INIT_CASE(2)  DRVGPU_PLOG_INIT_CASE(3)
            DRVGPU_PLOG_INIT_CASE(4)  DRVGPU_PLOG_INIT_CASE(5)  DRVGPU_PLOG_INIT_CASE(6)  DRVGPU_PLOG_INIT_CASE(7)
            DRVGPU_PLOG_INIT_CASE(8)  DRVGPU_PLOG_INIT_CASE(9)  DRVGPU_PLOG_INIT_CASE(10) DRVGPU_PLOG_INIT_CASE(11)
            DRVGPU_PLOG_INIT_CASE(12) DRVGPU_PLOG_INIT_CASE(13) DRVGPU_PLOG_INIT_CASE(14) DRVGPU_PLOG_INIT_CASE(15)
            DRVGPU_PLOG_INIT_CASE(16) DRVGPU_PLOG_INIT_CASE(17) DRVGPU_PLOG_INIT_CASE(18) DRVGPU_PLOG_INIT_CASE(19)
            DRVGPU_PLOG_INIT_CASE(20) DRVGPU_PLOG_INIT_CASE(21) DRVGPU_PLOG_INIT_CASE(22) DRVGPU_PLOG_INIT_CASE(23)
            DRVGPU_PLOG_INIT_CASE(24) DRVGPU_PLOG_INIT_CASE(25) DRVGPU_PLOG_INIT_CASE(26) DRVGPU_PLOG_INIT_CASE(27)
            DRVGPU_PLOG_INIT_CASE(28) DRVGPU_PLOG_INIT_CASE(29) DRVGPU_PLOG_INIT_CASE(30) DRVGPU_PLOG_INIT_CASE(31)
            default: break;
        }

        initialized_ = true;
    } catch (const std::exception& e) {
        (void)e;
        initialized_ = true;
    }
}

#undef DRVGPU_PLOG_INIT_CASE

namespace {
template<int InstanceId>
void WriteToPlogInstance(plog::Severity severity, const std::string& message) {
    plog::Logger<InstanceId>* log = plog::get<InstanceId>();
    if (log && log->checkSeverity(severity)) {
        (*log) += plog::Record(severity, "", 0, "", nullptr, InstanceId).ref() << message;
    }
}
#define DRVGPU_PLOG_WRITE_CASE(N) case N: WriteToPlogInstance<N>(severity, formatted); break;
void WriteToPlogByGpuId(int gpu_id, plog::Severity severity, const std::string& formatted) {
    switch (gpu_id) {
        DRVGPU_PLOG_WRITE_CASE(0)  DRVGPU_PLOG_WRITE_CASE(1)  DRVGPU_PLOG_WRITE_CASE(2)  DRVGPU_PLOG_WRITE_CASE(3)
        DRVGPU_PLOG_WRITE_CASE(4)  DRVGPU_PLOG_WRITE_CASE(5)  DRVGPU_PLOG_WRITE_CASE(6)  DRVGPU_PLOG_WRITE_CASE(7)
        DRVGPU_PLOG_WRITE_CASE(8)  DRVGPU_PLOG_WRITE_CASE(9)  DRVGPU_PLOG_WRITE_CASE(10) DRVGPU_PLOG_WRITE_CASE(11)
        DRVGPU_PLOG_WRITE_CASE(12) DRVGPU_PLOG_WRITE_CASE(13) DRVGPU_PLOG_WRITE_CASE(14) DRVGPU_PLOG_WRITE_CASE(15)
        DRVGPU_PLOG_WRITE_CASE(16) DRVGPU_PLOG_WRITE_CASE(17) DRVGPU_PLOG_WRITE_CASE(18) DRVGPU_PLOG_WRITE_CASE(19)
        DRVGPU_PLOG_WRITE_CASE(20) DRVGPU_PLOG_WRITE_CASE(21) DRVGPU_PLOG_WRITE_CASE(22) DRVGPU_PLOG_WRITE_CASE(23)
        DRVGPU_PLOG_WRITE_CASE(24) DRVGPU_PLOG_WRITE_CASE(25) DRVGPU_PLOG_WRITE_CASE(26) DRVGPU_PLOG_WRITE_CASE(27)
        DRVGPU_PLOG_WRITE_CASE(28) DRVGPU_PLOG_WRITE_CASE(29) DRVGPU_PLOG_WRITE_CASE(30) DRVGPU_PLOG_WRITE_CASE(31)
        default: WriteToPlogInstance<0>(severity, formatted); break;
    }
}
#undef DRVGPU_PLOG_WRITE_CASE
}

/**
 * @brief Очистить и завершить работу plog
 *
 * plog не требует явного shutdown — ресурсы освобождаются автоматически.
 * Метод оставлен для совместимости с интерфейсом.
 */
void DefaultLogger::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
}

/**
 * @brief Логировать отладочное сообщение
 * @param component Имя компонента (например: "OpenCL", "Memory")
 * @param message Текст сообщения
 */
void DefaultLogger::Debug(const std::string& component, const std::string& message) {
    if (!initialized_) return;
    std::string formatted = FormatMessage(component, message);
    WriteToPlogByGpuId(gpu_id_, plog::debug, formatted);
}

void DefaultLogger::Info(const std::string& component, const std::string& message) {
    if (!initialized_) return;
    std::string formatted = FormatMessage(component, message);
    WriteToPlogByGpuId(gpu_id_, plog::info, formatted);
}

void DefaultLogger::Warning(const std::string& component, const std::string& message) {
    if (!initialized_) return;
    std::string formatted = FormatMessage(component, message);
    WriteToPlogByGpuId(gpu_id_, plog::warning, formatted);
}

void DefaultLogger::Error(const std::string& component, const std::string& message) {
    if (!initialized_) return;
    std::string formatted = FormatMessage(component, message);
    WriteToPlogByGpuId(gpu_id_, plog::error, formatted);
}

/**
 * @brief Проверить, активен ли уровень DEBUG
 * @return true если DEBUG активен
 */
bool DefaultLogger::IsDebugEnabled() const {
    return initialized_ && current_level_ >= plog::debug;
}

/**
 * @brief Проверить, активен ли уровень INFO
 * @return true если INFO активен
 */
bool DefaultLogger::IsInfoEnabled() const {
    return initialized_ && current_level_ >= plog::info;
}

/**
 * @brief Проверить, активен ли уровень WARNING
 * @return true если WARNING активен
 */
bool DefaultLogger::IsWarningEnabled() const {
    return initialized_ && current_level_ >= plog::warning;
}

/**
 * @brief Проверить, активен ли уровень ERROR
 * @return true если ERROR активен
 */
bool DefaultLogger::IsErrorEnabled() const {
    return initialized_ && current_level_ >= plog::error;
}

/**
 * @brief Сбросить состояние логера
 *
 * Вызывает Shutdown() + Initialize() для переинициализации.
 */
void DefaultLogger::Reset() {
    Shutdown();
    Initialize();
}

/**
 * @brief Форматировать сообщение с компонентом
 * @param component Имя компонента
 * @param message Текст сообщения
 * @return Отформатированное сообщение "[component] message"
 */
std::string DefaultLogger::FormatMessage(const std::string& component,
                                          const std::string& message) {
    return "[" + component + "] " + message;
}

/**
 * @brief Проверить, инициализирован ли логер
 * @return true если логер инициализирован
 */
bool DefaultLogger::IsInitialized() const {
    return initialized_;
}

} // namespace drv_gpu_lib
