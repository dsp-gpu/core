#pragma once

/**
 * @file gpu_config.hpp
 * @brief GPUConfig — Singleton для управления конфигурацией GPU (configGPU.json)
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Централизованное управление конфигурацией GPU:
 *   - Загрузка/сохранение configGPU.json
 *   - Конфигурация по каждому GPU (is_prof, is_logger и т.д.)
 *   - Автосоздание конфига по умолчанию, если файл отсутствует
 *   - Потокобезопасный доступ
 *
 * ФОРМАТ JSON (configGPU.json):
 *   {
 *     "version": "1.0",
 *     "description": "GPU Configuration for DrvGPU",
 *     "gpus": [
 *       {
 *         "id": 0,
 *         "name": "Alex",
 *         "is_prof": true,
 *         "is_logger": true,
 *         "is_console": true
 *       },
 *       {
 *         "id": 1,
 *         "name": "Evgeni"
 *       }
 *     ]
 *   }
 *
 * ОТСУТСТВУЮЩИЕ ПОЛЯ:
 *   Если поле отсутствует в JSON, используется значение по умолчанию из GPUConfigEntry.
 *   Пример: GPU id=1 выше имеет is_prof=false (по умолчанию).
 *
 * АВТОСОЗДАНИЕ:
 *   Если configGPU.json не существует, создаётся конфиг с одним GPU:
 *   { id: 0, name: "TEST", is_prof: true, is_logger: true }
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   // Загрузить конфигурацию
 *   GPUConfig::GetInstance().Load("./configGPU.json");
 *
 *   // Получить конфиг для конкретного GPU
 *   auto& cfg = GPUConfig::GetInstance().GetConfig(0);
 *   if (cfg.is_prof) { ... }
 *
 *   // Получить все конфиги
 *   auto& all = GPUConfig::GetInstance().GetAllConfigs();
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "config_types.hpp"

#include <string>
#include <vector>
#include <mutex>
#include <optional>

namespace drv_gpu_lib {

// ============================================================================
// GPUConfig — Singleton конфигурации GPU
// ============================================================================

/**
 * @class GPUConfig
 * @brief Менеджер-Singleton конфигурации GPU
 *
 * Обеспечивает централизованный доступ к конфигурации GPU из configGPU.json.
 * Потокобезопасен при параллельном чтении из нескольких потоков GPU.
 *
 * Жизненный цикл:
 * 1. GPUConfig::GetInstance().Load(path) или LoadOrCreate(path)
 * 2. GPUConfig::GetInstance().GetConfig(gpu_id) — доступ по GPU
 * 3. Опционально: Save() для сохранения изменений
 */
class GPUConfig {
public:
    // ========================================================================
    // Доступ к Singleton
    // ========================================================================

    /**
     * @brief Получить единственный экземпляр
     * @return Ссылка на глобальный GPUConfig
     */
    static GPUConfig& GetInstance();

    // Запрет копирования/перемещения (singleton)
    GPUConfig(const GPUConfig&) = delete;
    GPUConfig& operator=(const GPUConfig&) = delete;

    // ========================================================================
    // Загрузка и сохранение
    // ========================================================================

    /**
     * @brief Загрузить конфигурацию из JSON-файла
     * @param file_path Путь к configGPU.json
     * @return true при успешной загрузке, false при ошибке
     *
     * При ошибке загрузки (файл не найден, ошибка разбора) возвращается false,
     * текущая конфигурация не изменяется.
     */
    bool Load(const std::string& file_path);

    /**
     * @brief Загрузить конфигурацию из файла или создать по умолчанию, если не найден
     * @param file_path Путь к configGPU.json
     * @return true при успешной загрузке или создании
     *
     * Если файл не существует:
     * 1. Создаётся конфигурация по умолчанию (один GPU: id=0, name="TEST", is_prof=true, is_logger=true)
     * 2. Сохраняется в file_path
     * 3. Возвращается true
     *
     * Если файл есть, но с ошибками:
     * 1. Возвращается false
     * 2. Конфигурация не изменяется
     */
    bool LoadOrCreate(const std::string& file_path);

    /**
     * @brief Сохранить текущую конфигурацию в JSON-файл
     * @param file_path Путь для сохранения (пусто = последний загруженный путь)
     * @return true при успешном сохранении
     */
    bool Save(const std::string& file_path = "");

    /**
     * @brief Проверить, загружена ли конфигурация
     * @return true если Load() или LoadOrCreate() были вызваны успешно
     */
    bool IsLoaded() const;

    // ========================================================================
    // Доступ к конфигурации (потокобезопасно)
    // ========================================================================

    /**
     * @brief Получить конфигурацию для конкретного GPU
     * @param gpu_id Индекс устройства GPU
     * @return Ссылка на GPUConfigEntry для этого GPU
     *
     * Если gpu_id не найден в конфигурации, возвращается запись по умолчанию с указанным id.
     */
    const GPUConfigEntry& GetConfig(int gpu_id) const;

    /**
     * @brief Получить все конфигурации GPU
     * @return Ссылка на вектор всех GPUConfigEntry
     */
    const std::vector<GPUConfigEntry>& GetAllConfigs() const;

    /**
     * @brief Получить корневые данные конфигурации (версия, описание)
     * @return Ссылка на GPUConfigData
     */
    const GPUConfigData& GetData() const;

    /**
     * @brief Получить список активных ID GPU (is_active == true)
     * @return Вектор ID GPU, которые следует инициализировать
     */
    std::vector<int> GetActiveGPUIDs() const;

    /**
     * @brief Проверить, включено ли профилирование для данного GPU
     * @param gpu_id Индекс устройства GPU
     * @return true если is_prof == true для этого GPU
     */
    bool IsProfilingEnabled(int gpu_id) const;

    /**
     * @brief Проверить, включено ли логирование для данного GPU
     * @param gpu_id Индекс устройства GPU
     * @return true если is_logger == true для этого GPU
     */
    bool IsLoggingEnabled(int gpu_id) const;

    /**
     * @brief Проверить, включён ли вывод в консоль для данного GPU
     * @param gpu_id Индекс устройства GPU
     * @return true если is_console == true для этого GPU
     */
    bool IsConsoleEnabled(int gpu_id) const;

    /**
     * @brief Получить максимальный процент памяти для GPU
     * @param gpu_id Индекс устройства GPU
     * @return Лимит памяти в процентах (напр., 70 означает 70%)
     */
    size_t GetMaxMemoryPercent(int gpu_id) const;

    // ========================================================================
    // Изменение конфигурации
    // ========================================================================

    /**
     * @brief Установить или обновить конфигурацию для GPU
     * @param entry Новая запись конфигурации
     *
     * Если GPU с таким id уже есть — запись заменяется.
     * Иначе добавляется новая запись.
     */
    void SetConfig(const GPUConfigEntry& entry);

    /**
     * @brief Сбросить к конфигурации по умолчанию
     * Создаётся один GPU: id=0, name="TEST", is_prof=true, is_logger=true
     */
    void ResetToDefault();

    // ========================================================================
    // Утилиты
    // ========================================================================

    /**
     * @brief Получить путь к файлу, последний раз использованный для Load/Save
     * @return Строка пути (пустая, если не загружали)
     */
    std::string GetFilePath() const;

    /**
     * @brief Вывести конфигурацию в stdout (для отладки)
     */
    void Print() const;

private:
    // ========================================================================
    // Приватный конструктор (Singleton)
    // ========================================================================

    GPUConfig();

    // ========================================================================
    // Приватные методы
    // ========================================================================

    /// Создать данные конфигурации по умолчанию
    GPUConfigData CreateDefaultConfig() const;

    /// Найти запись конфигурации по ID GPU (возвращает nullptr, если не найдено)
    const GPUConfigEntry* FindConfig(int gpu_id) const;

    // ========================================================================
    // Приватные члены
    // ========================================================================

    /// Корневые данные конфигурации
    GPUConfigData data_;

    /// Запись по умолчанию при ненайденном ID GPU
    mutable GPUConfigEntry default_entry_;

    /// Путь к файлу конфигурации
    std::string file_path_;

    /// Флаг загрузки конфигурации
    bool loaded_ = false;

    /// Мьютекс для потокобезопасного доступа
    mutable std::mutex mutex_;
};

} // namespace drv_gpu_lib
