#pragma once

/**
 * @file service_manager.hpp
 * @brief ServiceManager — централизованный запуск/останов всех фоновых сервисов
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Единая точка инициализации, запуска и остановки всех асинхронных сервисов
 *   DrvGPU: Logger, Profiler, ConsoleOutput.
 *
 *   Читает configGPU.json и включает/отключает сервисы по каждому GPU.
 *
 * ЖИЗНЕННЫЙ ЦИКЛ:
 *   1. GPUManager создаёт GPU
 *   2. ServiceManager::InitializeFromConfig(config) — настройка сервисов из JSON
 *   3. ServiceManager::StartAll() — запуск фоновых потоков
 *   4. ... работа GPU, модули вызывают Enqueue() ...
 *   5. ServiceManager::StopAll() — освобождение очередей, join потоков
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   // После GPUManager::InitializeAll():
 *   auto& sm = ServiceManager::GetInstance();
 *   sm.InitializeFromConfig("configGPU.json");
 *   sm.StartAll();
 *
 *   // ... обработка на GPU ...
 *
 *   // Перед выходом:
 *   sm.StopAll();
 *
 * ПОТОКОБЕЗОПАСНОСТЬ:
 *   - Initialize/Start/Stop не предназначены для параллельного вызова
 *   - Вызываются один раз из главного потока
 *   - API отдельных сервисов (Enqueue) потокобезопасны
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "console_output.hpp"
#include "gpu_profiler.hpp"
#include "../config/gpu_config.hpp"
#include "../logger/config_logger.hpp"
#include "../logger/logger.hpp"

#include <string>
#include <iostream>
#include <vector>

namespace drv_gpu_lib {

// ============================================================================
// ServiceManager — централизованное управление жизненным циклом сервисов
// ============================================================================

/**
 * @class ServiceManager
 * @brief Синглтон, управляющий жизненным циклом всех фоновых сервисов
 *
 * Обязанности:
 * - Чтение configGPU.json и настройка сервисов
 * - Запуск/останов потоков ConsoleOutput, GPUProfiler
 * - Настройка путей логера по каждому GPU
 * - Удобный API для статуса сервисов
 */
class ServiceManager {
public:
    // ========================================================================
    // Singleton
    // ========================================================================

    /**
     * @brief Получить экземпляр синглтона
     */
    static ServiceManager& GetInstance() {
        static ServiceManager instance;
        return instance;
    }

    // Запрет копирования
    ServiceManager(const ServiceManager&) = delete;
    ServiceManager& operator=(const ServiceManager&) = delete;

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Инициализировать сервисы из configGPU.json
     * @param config_file Путь к configGPU.json
     * @return true при успешной загрузке конфигурации
     *
     * Читает JSON-конфиг и применяет настройки:
     * - is_console  -> включение/отключение ConsoleOutput по каждому GPU
     * - is_prof     -> флаг включения GPUProfiler
     * - is_logger   -> пути логов ConfigLogger по каждому GPU
     * - log_level   -> уровень логирования
     *
     * Сервисы не запускает (для этого вызвать StartAll()).
     */
    bool InitializeFromConfig(const std::string& config_file) {
        // Загрузка конфига (или создание по умолчанию при отсутствии)
        bool ok = GPUConfig::GetInstance().LoadOrCreate(config_file);
        if (!ok) {
            std::cerr << "[ServiceManager] WARNING: Failed to load config, using defaults\n";
        }

        const auto& data = GPUConfig::GetInstance().GetData();

        // Настройка ConsoleOutput по каждому GPU
        for (const auto& gpu : data.gpus) {
            ConsoleOutput::GetInstance().SetGPUEnabled(gpu.id, gpu.is_console);
        }

        // Настройка GPUProfiler: глобально и по каждому GPU (is_prof)
        bool any_profiling = false;
        for (const auto& gpu : data.gpus) {
            if (gpu.is_prof) {
                any_profiling = true;
            }
            GPUProfiler::GetInstance().SetGPUEnabled(gpu.id, gpu.is_prof);
        }
        GPUProfiler::GetInstance().SetEnabled(any_profiling);

        // Настройка путей логера по каждому GPU и предсоздание инстансов логеров
        for (const auto& gpu : data.gpus) {
            if (gpu.is_logger) {
                ConfigLogger::GetInstance().CreateLogDirectoryForGPU(gpu.id);
                (void)Logger::GetInstance(gpu.id);  // создать логер для DRVGPU_XX при старте
            }
        }

        initialized_ = true;

        std::cout << "[ServiceManager] Configured "
                  << data.gpus.size() << " GPU(s) from: "
                  << config_file << "\n";

        return true;
    }

    /**
     * @brief Инициализация настройками по умолчанию (без файла конфига)
     *
     * Создаёт конфиг по умолчанию для одного GPU со всеми сервисами.
     * Удобно для тестов и разработки.
     */
    void InitializeDefaults() {
        // GPUConfig уже имеет значения по умолчанию из конструктора
        // Включаем всё
        ConsoleOutput::GetInstance().SetEnabled(true);
        GPUProfiler::GetInstance().SetEnabled(true);
        ConfigLogger::GetInstance().Enable();

        initialized_ = true;

        std::cout << "[ServiceManager] Initialized with defaults\n";
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * @brief Запустить все фоновые потоки сервисов
     *
     * Запускает:
     * - рабочий поток ConsoleOutput
     * - рабочий поток GPUProfiler
     *
     * Logger (plog) в отдельном потоке не нужен (работа с файлом).
     *
     * ВАЖНО: Сначала вызвать InitializeFromConfig() или InitializeDefaults()!
     */
    void StartAll() {
        if (!initialized_) {
            std::cerr << "[ServiceManager] WARNING: Not initialized, calling InitializeDefaults()\n";
            InitializeDefaults();
        }

        // Запуск фонового потока ConsoleOutput
        ConsoleOutput::GetInstance().Start();

        // Запуск фонового потока GPUProfiler
        if (GPUProfiler::GetInstance().IsEnabled()) {
            GPUProfiler::GetInstance().Start();
        }

        running_ = true;

        ConsoleOutput::GetInstance().PrintSystem("ServiceManager", "All services started");
    }

    /**
     * @brief Остановить все фоновые потоки сервисов
     *
     * Опустошает очереди сообщений, затем присоединяет рабочие потоки.
     * После вызова новые сообщения не обрабатываются.
     *
     * Безопасно вызывать несколько раз.
     */
    void StopAll() {
        if (!running_) return;

        ConsoleOutput::GetInstance().PrintSystem("ServiceManager", "Stopping all services...");

        // Сначала останавливаем GPUProfiler (может ещё писать во время остановки)
        GPUProfiler::GetInstance().Stop();

        // ConsoleOutput останавливаем последним (чтобы сервисы могли залогировать остановку)
        ConsoleOutput::GetInstance().Stop();

        running_ = false;

        // Вывод итога после остановки консоли (напрямую в stdout)
        std::cout << "[ServiceManager] All services stopped.\n";
    }

    /**
     * @brief Проверить, запущены ли сервисы
     */
    bool IsRunning() const { return running_; }

    /**
     * @brief Проверить, инициализированы ли сервисы
     */
    bool IsInitialized() const { return initialized_; }

    // ========================================================================
    // Удобный API
    // ========================================================================

    /**
     * @brief Экспорт данных профилирования в JSON-файл
     * @param file_path Путь к выходному файлу
     * @return true при успехе
     *
     * Обёртка над GPUProfiler::ExportJSON().
     * Создаёт родительские директории при необходимости.
     */
    bool ExportProfiling(const std::string& file_path) const {
        return GPUProfiler::GetInstance().ExportJSON(file_path);
    }

    /**
     * @brief Вывести сводку профилирования в консоль
     *
     * Обёртка над GPUProfiler::PrintSummary().
     */
    void PrintProfilingSummary() const {
        GPUProfiler::GetInstance().PrintSummary();
    }

    /**
     * @brief Вывести конфигурацию GPU в консоль
     *
     * Обёртка над GPUConfig::Print().
     */
    void PrintConfig() const {
        GPUConfig::GetInstance().Print();
    }

    /**
     * @brief Получить строку со статистикой сервисов
     * @return Человекочитаемый статус сервисов
     */
    std::string GetStatus() const {
        std::ostringstream oss;
        oss << "ServiceManager Status:\n";
        oss << "  Initialized: " << (initialized_ ? "YES" : "NO") << "\n";
        oss << "  Running: " << (running_ ? "YES" : "NO") << "\n";
        oss << "  ConsoleOutput: "
            << (ConsoleOutput::GetInstance().IsRunning() ? "running" : "stopped")
            << " (processed: " << ConsoleOutput::GetInstance().GetProcessedCount()
            << ", queue: " << ConsoleOutput::GetInstance().GetQueueSize() << ")\n";
        oss << "  GPUProfiler: "
            << (GPUProfiler::GetInstance().IsRunning() ? "running" : "stopped")
            << " (enabled: " << (GPUProfiler::GetInstance().IsEnabled() ? "YES" : "NO")
            << ", processed: " << GPUProfiler::GetInstance().GetProcessedCount() << ")\n";
        return oss.str();
    }

private:
    // ========================================================================
    // Приватный конструктор (синглтон)
    // ========================================================================

    ServiceManager() : initialized_(false), running_(false) {}

    ~ServiceManager() {
        // Автоостановка при уничтожении
        if (running_) {
            StopAll();
        }
    }

    // ========================================================================
    // Приватные члены
    // ========================================================================

    /// Флаг инициализации
    bool initialized_;

    /// Флаг работы
    bool running_;
};

} // namespace drv_gpu_lib
