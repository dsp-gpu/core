#include <core/logger/config_logger.hpp>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// ConfigLogger Implementation - Реализация конфигурации логирования
// ═══════════════════════════════════════════════════════════════════════════=

/**
 * @brief Получить единственный экземпляр ConfigLogger (Singleton)
 * @return Ссылка на статический экземпляр
 * 
 * Потокобезопасная инициализация через статическую локальную переменную.
 * Гарантирует создание только одного экземпляра.
 */
ConfigLogger& ConfigLogger::GetInstance() {
    static ConfigLogger instance;
    return instance;
}

/**
 * @brief Приватный конструктор
 * 
 * Инициализирует:
 * - log_path_ пустой строкой (используется путь по умолчанию)
 * - enabled_ = true (логирование включено)
 */
ConfigLogger::ConfigLogger()
    : log_path_("")
    , enabled_(true) {
}

/**
 * @brief Установить путь к директории логов
 * @param path Путь к директории (может быть пустой строкой)
 * 
 * Если path пустой, используется путь по умолчанию:
 *   ./Logs/DRVGPU/YYYY-MM-DD/HH-MM-SS.log
 * 
 * Пример:
 * @code
 * ConfigLogger::GetInstance().SetLogPath("C:/MyApp/logs");
 * @endcode
 */
void ConfigLogger::SetLogPath(const std::string& path) {
    log_path_ = path;
}

/**
 * @brief Получить путь к директории логов
 * @return Путь к директории (пустая строка = по умолчанию)
 */
std::string ConfigLogger::GetLogPath() const {
    return log_path_;
}

/**
 * @brief Получить полный путь к файлу лога с автоматическим созданием структуры папок
 * @return Полный путь к файлу лога
 * 
 * Структура пути:
 *   {log_path}/Logs/DRVGPU/{YYYY-MM-DD}/{HH-MM-SS}.log
 * 
 * Если log_path_ пустой, используется текущая директория.
 *
 * @note Создаёт уникальный файл для каждого запуска на основе времени.
 */
std::string ConfigLogger::GetLogFilePath() const {
    // Получаем текущее время с высоким разрешением
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;
    
    // Кроссплатформенное получение локального времени
#if defined(_WIN32)
    localtime_s(&now_tm, &now_time);
#else
    localtime_r(&now_time, &now_tm);
#endif

    // Формируем строку с датой: YYYY-MM-DD
    std::ostringstream date_ss;
    date_ss << std::put_time(&now_tm, "%Y-%m-%d");
    std::string date_str = date_ss.str();

    // Формируем строку со временем: HH-MM-SS (с дефисами для безопасности файловой системы)
    std::ostringstream time_ss;
    time_ss << std::put_time(&now_tm, "%H-%M-%S");
    std::string time_str = time_ss.str();

    // Определяем базовый путь
    std::string base_path = log_path_;
    if (base_path.empty()) {
        base_path = std::filesystem::current_path().string();
    }

    // Формируем полный путь: {base_path}/Logs/DRVGPU/{date}/{time}.log
    std::ostringstream path_ss;
    path_ss << base_path;
    
    // Добавляем разделитель пути (кроссплатформенно)
    if (!base_path.empty() && base_path.back() != '/' && base_path.back() != '\\') {
        path_ss << std::filesystem::path::preferred_separator;
    }
    
    path_ss << kLogsDir << std::filesystem::path::preferred_separator;
    path_ss << kLogSubdir << std::filesystem::path::preferred_separator;
    path_ss << date_str << std::filesystem::path::preferred_separator;
    path_ss << time_str << ".log";

    return path_ss.str();
}

/**
 * @brief Включить или выключить логирование
 * @param enabled true = включить, false = выключить
 * 
 * Когда логирование отключено:
 * - Сообщения не пишутся в файл
 * - DefaultLogger не создаёт file sink
 * - Это повышает производительность в production
 */
void ConfigLogger::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

/**
 * @brief Проверить, включено ли логирование
 * @return true если логирование включено
 */
bool ConfigLogger::IsEnabled() const {
    return enabled_;
}

/**
 * @brief Включить логирование (shortcut)
 * 
 * Эквивалентно SetEnabled(true).
 */
void ConfigLogger::Enable() {
    enabled_ = true;
}

/**
 * @brief Выключить логирование (shortcut, production mode)
 * 
 * Эквивалентно SetEnabled(false).
 * Рекомендуется использовать в production для повышения производительности.
 */
void ConfigLogger::Disable() {
    enabled_ = false;
}

/**
 * @brief Создать директорию для логов (если не существует)
 * @return true если успешно создан или уже существует, false при ошибке
 * 
 * Создаёт все промежуточные директории в структуре:
 *   {log_path}/Logs/DRVGPU/{YYYY-MM-DD}/
 * 
 * @note Вызывается автоматически DefaultLogger при инициализации.
 */
bool ConfigLogger::CreateLogDirectory() const {
    std::string file_path = GetLogFilePath();
    
    // Извлекаем директорию из полного пути к файлу
    std::filesystem::path log_file_path(file_path);
    std::filesystem::path log_dir = log_file_path.parent_path();

    try {
        // Создаём директорию (и все родительские), если не существует
        if (!std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        // BOOTSTRAP: std::cerr INTENTIONAL — Logger ещё не инициализирован,
        // это код самого логера. DRVGPU_LOG_* здесь использовать нельзя.
        std::cerr << "[ConfigLogger] Failed to create log directory: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Получить полный путь к файлу лога для конкретного GPU
 * @param gpu_id Индекс GPU устройства (0-based)
 * @return Полный путь к файлу лога
 *
 * Создаёт структуру с ID GPU (двузначный номер с ведущим нулём):
 *   {log_path}/Logs/DRVGPU_00/{YYYY-MM-DD}/{HH-MM-SS}.log
 *   {log_path}/Logs/DRVGPU_01/{YYYY-MM-DD}/{HH-MM-SS}.log
 *   {log_path}/Logs/DRVGPU_13/{YYYY-MM-DD}/{HH-MM-SS}.log
 *
 * Используется для Multi-GPU: каждый GPU пишет в свою директорию.
 * Формат поддиректории: DRVGPU_XX (XX — двузначный номер GPU).
 */
std::string ConfigLogger::GetLogFilePathForGPU(int gpu_id) const {
    // Получаем текущее время
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;

#if defined(_WIN32)
    localtime_s(&now_tm, &now_time);
#else
    localtime_r(&now_time, &now_tm);
#endif

    // Дата: YYYY-MM-DD
    std::ostringstream date_ss;
    date_ss << std::put_time(&now_tm, "%Y-%m-%d");
    std::string date_str = date_ss.str();

    // Время: HH-MM-SS
    std::ostringstream time_ss;
    time_ss << std::put_time(&now_tm, "%H-%M-%S");
    std::string time_str = time_ss.str();

    // Формат поддиректории: DRVGPU_XX (двузначный номер GPU)
    std::ostringstream subdir_ss;
    subdir_ss << kLogSubdir << "_" << std::setfill('0') << std::setw(2) << gpu_id;
    std::string gpu_subdir = subdir_ss.str();

    // Базовый путь
    std::filesystem::path base_path_fs;
    if (log_path_.empty()) {
        base_path_fs = std::filesystem::current_path();
    } else {
        base_path_fs = log_path_;
    }

    // Полный путь: {base_path}/Logs/DRVGPU_XX/{date}/{time}.log
    // Используем std::filesystem::path с operator/= для правильной конкатенации
    // (избегаем проблемы с preferred_separator который выводится как число в ostringstream)
    std::filesystem::path full_path = base_path_fs;
    full_path /= kLogsDir;                    // "Logs"
    full_path /= gpu_subdir;                  // "DRVGPU_00"
    full_path /= date_str;                    // "2026-02-09"
    full_path /= (time_str + ".log");         // "21-17-52.log"

    return full_path.string();
}

/**
 * @brief Создать директорию для логов конкретного GPU
 * @param gpu_id Индекс GPU устройства
 * @return true если успешно создан или уже существует, false при ошибке
 *
 * Создаёт структуру:
 *   {log_path}/Logs/DRVGPU_XX/{YYYY-MM-DD}/
 *
 * Используется совместно с GetLogFilePathForGPU().
 */
bool ConfigLogger::CreateLogDirectoryForGPU(int gpu_id) const {
    std::string file_path = GetLogFilePathForGPU(gpu_id);

    // Извлекаем директорию из полного пути к файлу
    std::filesystem::path log_file_path(file_path);
    std::filesystem::path log_dir = log_file_path.parent_path();

    try {
        if (!std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        // BOOTSTRAP: std::cerr INTENTIONAL — Logger ещё не инициализирован
        std::cerr << "[ConfigLogger] Failed to create GPU log directory: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Сбросить настройки на значения по умолчанию
 *
 * Сбрасывает:
 * - log_path_ = "" (путь по умолчанию)
 * - enabled_ = true (логирование включено)
 */
void ConfigLogger::Reset() {
    log_path_ = "";
    enabled_ = true;
}

} // namespace drv_gpu_lib
