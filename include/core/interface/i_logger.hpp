#pragma once

/**
 * @file logger_interface.hpp
 * @brief ILogger - Интерфейс логирования для DrvGPU
 * 
 * Абстрактный интерфейс логирования. Позволяет подключить любой логер
 * (DefaultLogger с plog, custom logger из продакшена, etc.)
 * 
 * @author DrvGPU Team
 * @date 2026-02-01
 */

#include <string>
#include <memory>

namespace drv_gpu_lib {

/**
 * @class ILogger
 * @brief Интерфейс логирования
 * 
 * Все логеры должны реализовывать этот интерфейс.
 * Позволяет заменить DefaultLogger на любой другой в продакшене.
 * 
 * Пример реализации своего логера:
 * @code
 * class CustomLogger : public ILogger {
 * public:
 *     void Debug(const std::string& component, const std::string& message) override {
 *         // Ваша логика логирования
 *         my_company_logger.log("DEBUG", component, message);
 *     }
 *     
 *     void Info(const std::string& component, const std::string& message) override {
 *         my_company_logger.log("INFO", component, message);
 *     }
 *     
 *     void Warning(const std::string& component, const std::string& message) override {
 *         my_company_logger.log("WARNING", component, message);
 *     }
 *     
 *     void Error(const std::string& component, const std::string& message) override {
 *         my_company_logger.log("ERROR", component, message);
 *     }
 * };
 * 
 * // В продакшене:
 * ILogger* custom_logger = new CustomLogger();
 * Logger::SetInstance(custom_logger);
 * @endcode
 */
class ILogger {
public:
    /// Виртуальный деструктор
    virtual ~ILogger() = default;

    /**
     * @brief Логирование отладочного сообщения
     * @param component Компонент (например: "DrvGPU", "OpenCLBackend")
     * @param message Текст сообщения
     */
    virtual void Debug(const std::string& component, const std::string& message) = 0;

    /**
     * @brief Логирование информационного сообщения
     * @param component Компонент
     * @param message Текст сообщения
     */
    virtual void Info(const std::string& component, const std::string& message) = 0;

    /**
     * @brief Логирование предупреждения
     * @param component Компонент
     * @param message Текст сообщения
     */
    virtual void Warning(const std::string& component, const std::string& message) = 0;

    /**
     * @brief Логирование ошибки
     * @param component Компонент
     * @param message Текст сообщения
     */
    virtual void Error(const std::string& component, const std::string& message) = 0;

    /**
     * @brief Проверить, активен ли уровень DEBUG
     * @return true если DEBUG активен
     */
    virtual bool IsDebugEnabled() const = 0;

    /**
     * @brief Проверить, активен ли уровень INFO
     * @return true если INFO активен
     */
    virtual bool IsInfoEnabled() const = 0;

    /**
     * @brief Проверить, активен ли уровень WARNING
     * @return true если WARNING активен
     */
    virtual bool IsWarningEnabled() const = 0;

    /**
     * @brief Проверить, активен ли уровень ERROR
     * @return true если ERROR активен
     */
    virtual bool IsErrorEnabled() const = 0;

    /**
     * @brief Сбросить состояние логера (вызывается при переинициализации)
     */
    virtual void Reset() = 0;
};

/// Умный указатель на ILogger
using ILoggerPtr = std::shared_ptr<ILogger>;

} // namespace drv_gpu_lib
