#pragma once

/**
 * @file config_logger.hpp
 * @brief ConfigLogger - Конфигурация логирования для DrvGPU
 * 
 * Хранит настройки логирования:
 * - Путь к директории логов
 * - Флаг отключения логов (production mode)
 * - Автоматическое создание структуры папок
 * 
 * Структура папок логов:
 *   logs_dir/Logs/DRVGPU/YYYY-MM-DD/HH-MM-SS.log
 * 
 * @author DrvGPU Team
 * @date 2026-02-01
 */

#include <string>
#include <atomic>

namespace drv_gpu_lib {

/**
 * @class ConfigLogger
 * @brief Конфигурация логирования
 * 
 * Singleton для хранения настроек логера.
 * 
 * Пример использования:
 * @code
 * // Отключить логирование (production mode)
 * ConfigLogger::GetInstance().SetEnabled(false);
 * 
 * // Установить свой путь для логов
 * ConfigLogger::GetInstance().SetLogPath("C:/MyApp/logs");
 * 
 * // Получить путь к логам
 * auto path = ConfigLogger::GetInstance().GetLogPath();
 * @endcode
 */
class ConfigLogger {
public:
    /// Получить единственный экземпляр ConfigLogger
    static ConfigLogger& GetInstance();

    // ═══════════════════════════════════════════════════════════════════════
    // Запрет копирования
    // ═══════════════════════════════════════════════════════════════════════
    
    ConfigLogger(const ConfigLogger&) = delete;
    ConfigLogger& operator=(const ConfigLogger&) = delete;

    // ═══════════════════════════════════════════════════════════════════════
    // Настройки пути
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Установить путь к директории логов
     * @param path Путь к директории ("" = автоматически ./Logs)
     * 
     * Если path пустой, логи будут создаваться в:
     *   ./Logs/DRVGPU/YYYY-MM-DD/HH-MM-SS.log
     * 
     * Если path указан, логи будут создаваться в:
     *   ${path}/Logs/DRVGPU/YYYY-MM-DD/HH-MM-SS.log
     */
    void SetLogPath(const std::string& path);

    /**
     * @brief Получить путь к директории логов
     * @return Путь к директории
     */
    std::string GetLogPath() const;

    /**
     * @brief Получить полный путь к файлу лога с датой и временем
     * @return Полный путь к файлу лога
     *
     * Создаёт структуру:
     *   {log_path}/Logs/DRVGPU/{YYYY-MM-DD}/{HH-MM-SS}.log
     */
    std::string GetLogFilePath() const;

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
     */
    std::string GetLogFilePathForGPU(int gpu_id) const;

    /**
     * @brief Создать директорию для логов конкретного GPU
     * @param gpu_id Индекс GPU устройства
     * @return true если успешно создан или уже существует
     */
    bool CreateLogDirectoryForGPU(int gpu_id) const;

    // ═══════════════════════════════════════════════════════════════════════
    // Настройки включения/выключения
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Включить/выключить логирование
     * @param enabled true = логировать, false = не логировать
     * 
     * Пример:
     * @code
     * // Production mode - логи отключены
     * ConfigLogger::GetInstance().SetEnabled(false);
     * 
     * // Development mode - логи включены
     * ConfigLogger::GetInstance().SetEnabled(true);
     * @endcode
     */
    void SetEnabled(bool enabled);

    /**
     * @brief Проверить, включено ли логирование
     * @return true если логирование включено
     */
    bool IsEnabled() const;

    /**
     * @brief Включить логирование (shortcut)
     */
    void Enable();

    /**
     * @brief Выключить логирование (shortcut, production mode)
     */
    void Disable();

    // ═══════════════════════════════════════════════════════════════════════
    // Утилиты
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Создать директорию для логов (если не существует)
     * @return true если успешно создан или уже существует
     */
    bool CreateLogDirectory() const;

    /**
     * @brief Сбросить настройки по умолчанию
     */
    void Reset();

private:
    /// Приватный конструктор (Singleton)
    ConfigLogger();

    /// Путь к логам (пустое значение = использовать ./Logs)
    std::string log_path_;

    /// Флаг включения логирования
    std::atomic<bool> enabled_;

    /// Имя поддиректории DRVGPU
    static constexpr const char* kLogSubdir = "DRVGPU";

    /// Имя директории Logs
    static constexpr const char* kLogsDir = "Logs";
};

} // namespace drv_gpu_lib
