#pragma once

/**
 * @file console_output.hpp
 * @brief ConsoleOutput — потокобезопасный синглтон вывода в консоль с нескольких GPU
 *
 * ============================================================================
 * ПРОБЛЕМА:
 *   При одновременной записи в stdout с 8 GPU вывод перемешивается.
 *   Сообщения от разных GPU чередуются непредсказуемо.
 *
 * РЕШЕНИЕ:
 *   ConsoleOutput — сервис-синглтон с:
 *   - Выделенным фоновым потоком для всего вывода в консоль
 *   - Очередью сообщений (потоки GPU только делают Enqueue — почти без задержки)
 *   - Форматированием: [ЧЧ:ММ:СС.ммм] [GPU_XX] [Модуль] сообщение
 *   - Включением/отключением по каждому GPU через configGPU.json (флаг is_console)
 *
 * АРХИТЕКТУРА:
 *   GPU Thread 0 --> Print(0, "FFT", "Done") --> Enqueue() --+
 *   GPU Thread 1 --> Print(1, "FFT", "Done") --> Enqueue() --+--> [Очередь] --> Worker --> stdout
 *   GPU Thread N --> Print(N, "FFT", "Done") --> Enqueue() --+
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   ConsoleOutput::GetInstance().Start();
 *   ConsoleOutput::GetInstance().Print(0, "FFT", "Processing 1024 beams...");
 *   ConsoleOutput::GetInstance().PrintError(0, "FFT", "Failed to allocate!");
 *   ConsoleOutput::GetInstance().Stop();
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "async_service_base.hpp"

#include <string>
#include <chrono>
#include <cstdint>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace drv_gpu_lib {

// ============================================================================
// ConsoleMessage — тип сообщения для очереди вывода в консоль
// ============================================================================

/**
 * @struct ConsoleMessage
 * @brief Одно сообщение для вывода в консоль
 */
struct ConsoleMessage {
    /// Индекс устройства GPU (-1 = системное сообщение, без префикса GPU)
    int gpu_id = -1;

    /// Имя модуля-источника (напр., "FFT", "MemManager", "Backend")
    std::string module_name;

    /// Уровень серьёзности сообщения
    enum class Level : uint8_t {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERRLEVEL = 3  // ERRLEVEL, чтобы не конфликтовать с макросом Windows ERROR
    };
    Level level = Level::INFO;

    /// Текст сообщения
    std::string message;

    /// Временная метка (устанавливается при создании)
    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();
};

// ============================================================================
// ConsoleOutput — потокобезопасный сервис вывода в консоль
// ============================================================================

/**
 * @class ConsoleOutput
 * @brief Синглтон для потокобезопасного вывода в консоль
 *
 * Наследуется от AsyncServiceBase<ConsoleMessage>:
 * - Фоновый рабочий поток
 * - Неблокирующий Enqueue() для потоков GPU
 * - Упорядоченный форматированный вывод в stdout
 */
class ConsoleOutput : public AsyncServiceBase<ConsoleMessage> {
public:
    // ========================================================================
    // Singleton
    // ========================================================================

    /**
     * @brief Получить экземпляр синглтона
     */
    static ConsoleOutput& GetInstance() {
        static ConsoleOutput instance;
        return instance;
    }

    // Запрет копирования (синглтон)
    ConsoleOutput(const ConsoleOutput&) = delete;
    ConsoleOutput& operator=(const ConsoleOutput&) = delete;

    /**
     * @brief Деструктор — останавливает рабочий поток ДО сброса vtable.
     *
     * КРИТИЧНО: Stop() должен быть вызван в деструкторе ПРОИЗВОДНОГО класса.
     * Если Stop() вызывается только в ~AsyncServiceBase(), vtable уже переключён
     * на базовый класс и ProcessMessage() (pure virtual) вызывает terminate().
     */
    ~ConsoleOutput() {
        Stop();
    }

    // ========================================================================
    // Удобный API (неблокирующий)
    // ========================================================================

    /**
     * @brief Вывести информационное сообщение в консоль
     * @param gpu_id Индекс устройства GPU (-1 для системных сообщений)
     * @param module Имя модуля-источника
     * @param message Текст сообщения
     */
    void Print(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::INFO;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Вывести предупреждение в консоль
     */
    void PrintWarning(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::WARNING;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Вывести сообщение об ошибке в консоль
     */
    void PrintError(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::ERRLEVEL;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Вывести отладочное сообщение в консоль
     */
    void PrintDebug(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::DEBUG;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Вывести системное сообщение (без префикса GPU)
     */
    void PrintSystem(const std::string& module, const std::string& message) {
        Print(-1, module, message);
    }

    // ========================================================================
    // Per-GPU Enable/Disable
    // ========================================================================

    /**
     * @brief Включить или отключить вывод в консоль глобально
     */
    void SetEnabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Проверить, включён ли глобально вывод в консоль
     */
    bool IsEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Включить или отключить вывод в консоль для конкретного GPU
     * @param gpu_id Индекс устройства GPU
     * @param enabled true — включить, false — отключить
     */
    void SetGPUEnabled(int gpu_id, bool enabled) {
        std::lock_guard<std::mutex> lock(gpu_filter_mutex_);
        if (enabled) {
            disabled_gpus_.erase(gpu_id);
        } else {
            disabled_gpus_.insert(gpu_id);
        }
    }

    /**
     * @brief Проверить, включён ли вывод в консоль для данного GPU
     */
    bool IsGPUEnabled(int gpu_id) const {
        std::lock_guard<std::mutex> lock(gpu_filter_mutex_);
        return disabled_gpus_.find(gpu_id) == disabled_gpus_.end();
    }

protected:
    // ========================================================================
    // AsyncServiceBase implementation
    // ========================================================================

    /**
     * @brief Обработать одно консольное сообщение (выполняется в рабочем потоке)
     *
     * Форматирует и выводит сообщение в stdout.
     * Формат: [ЧЧ:ММ:СС.ммм] [GPU_XX] [Модуль] сообщение
     */
    void ProcessMessage(const ConsoleMessage& msg) override {
        // Проверка глобального включения
        if (!enabled_.load(std::memory_order_acquire)) {
            return;
        }

        // Проверка включения для данного GPU
        if (msg.gpu_id >= 0) {
            std::lock_guard<std::mutex> lock(gpu_filter_mutex_);
            if (disabled_gpus_.find(msg.gpu_id) != disabled_gpus_.end()) {
                return;
            }
        }

        // Форматирование времени: ЧЧ:ММ:СС.ммм
        auto time_t = std::chrono::system_clock::to_time_t(msg.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            msg.timestamp.time_since_epoch()) % 1000;

        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif

        std::ostringstream oss;

        // Timestamp
        oss << "[" << std::setfill('0')
            << std::setw(2) << tm_buf.tm_hour << ":"
            << std::setw(2) << tm_buf.tm_min << ":"
            << std::setw(2) << tm_buf.tm_sec << "."
            << std::setw(3) << ms.count() << "] ";

        // Level prefix
        switch (msg.level) {
            case ConsoleMessage::Level::DEBUG:
                oss << "[DBG] ";
                break;
            case ConsoleMessage::Level::INFO:
                oss << "[INF] ";
                break;
            case ConsoleMessage::Level::WARNING:
                oss << "[WRN] ";
                break;
            case ConsoleMessage::Level::ERRLEVEL:
                oss << "[ERR] ";
                break;
        }

        // GPU prefix
        if (msg.gpu_id >= 0) {
            oss << "[GPU_" << std::setfill('0') << std::setw(2) << msg.gpu_id << "] ";
        } else {
            oss << "[SYSTEM] ";
        }

        // Module
        if (!msg.module_name.empty()) {
            oss << "[" << msg.module_name << "] ";
        }

        // Message
        oss << msg.message;

        // Вывод в stdout (или stderr для ошибок)
        if (msg.level == ConsoleMessage::Level::ERRLEVEL) {
            std::cerr << oss.str() << "\n";
        } else {
            std::cout << oss.str() << "\n";
        }
    }

    /**
     * @brief Имя сервиса для диагностики
     */
    std::string GetServiceName() const override {
        return "ConsoleOutput";
    }

private:
    // ========================================================================
    // Приватный конструктор (синглтон)
    // ========================================================================

    ConsoleOutput() : enabled_(true) {}

    // ========================================================================
    // Приватные члены
    // ========================================================================

    /// Глобальный флаг включения
    std::atomic<bool> enabled_;

    /// Множество отключённых GPU
    std::unordered_set<int> disabled_gpus_;

    /// Мьютекс для множества disabled_gpus_
    mutable std::mutex gpu_filter_mutex_;
};

} // namespace drv_gpu_lib
