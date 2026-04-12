/**
 * @file gpu_config.cpp
 * @brief Реализация GPUConfig — менеджер конфигурации GPU на основе JSON
 *
 * ============================================================================
 * ЗАМЕТКИ ПО РЕАЛИЗАЦИИ:
 *
 * Стратегия десериализации JSON:
 *   Используется nlohmann/json с методом value() для опциональных полей.
 *   value(key, default) возвращает default, если ключ отсутствует.
 *   Это обеспечивает прямую совместимость при добавлении новых полей.
 *
 * Потокобезопасность:
 *   Все публичные методы захватывают mutex перед доступом к data_.
 *   Методы только для чтения используют const lock мьютекса.
 *
 * Обработка ошибок:
 *   - Ошибки разбора: логируются в stderr, возвращается false
 *   - Отсутствующие поля: используются значения по умолчанию из GPUConfigEntry
 *   - Отсутствующий файл в LoadOrCreate: создаётся конфиг по умолчанию и сохраняется
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "gpu_config.hpp"

// nlohmann/json — header-only библиотека JSON
#include "../../third_party/nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace drv_gpu_lib {

// ============================================================================
// Вспомогательные функции сериализации / десериализации JSON
// ============================================================================

/**
 * @brief Десериализовать GPUConfigEntry из JSON-объекта
 *
 * Используется value() с значениями по умолчанию для КАЖДОГО поля.
 * Если поле отсутствует в JSON, подставляется значение по умолчанию из GPUConfigEntry.
 *
 * Пример:
 *   JSON: { "id": 1, "name": "Evgeni" }
 *   Результат: { id=1, name="Evgeni", is_prof=false, is_logger=false, ... }
 */
static GPUConfigEntry ParseGPUEntry(const json& j) {
    GPUConfigEntry entry;

    // Идентификация
    entry.id                = j.value("id", entry.id);
    entry.name              = j.value("name", entry.name);

    // Флаги возможностей (по умолчанию false)
    entry.is_prof           = j.value("is_prof", entry.is_prof);
    entry.is_logger         = j.value("is_logger", entry.is_logger);
    entry.is_console        = j.value("is_console", entry.is_console);
    entry.is_active         = j.value("is_active", entry.is_active);
    entry.is_db             = j.value("is_db", entry.is_db);

    // Лимиты ресурсов
    entry.max_memory_percent = j.value("max_memory_percent", entry.max_memory_percent);

    // Настройки логирования
    entry.log_level         = j.value("log_level", entry.log_level);

    return entry;
}

/**
 * @brief Сериализовать GPUConfigEntry в JSON-объект
 *
 * Записывает ВСЕ поля в JSON (включая значения по умолчанию).
 * Это делает файл конфигурации самодокументируемым.
 */
static json SerializeGPUEntry(const GPUConfigEntry& entry) {
    json j;

    j["id"]                 = entry.id;
    j["name"]               = entry.name;
    j["is_prof"]            = entry.is_prof;
    j["is_logger"]          = entry.is_logger;
    j["is_console"]         = entry.is_console;
    j["is_active"]          = entry.is_active;
    j["is_db"]              = entry.is_db;
    j["max_memory_percent"] = entry.max_memory_percent;
    j["log_level"]          = entry.log_level;

    return j;
}

/**
 * @brief Сериализовать полные GPUConfigData в JSON
 */
static json SerializeConfigData(const GPUConfigData& data) {
    json root;

    root["version"]     = data.version;
    root["description"] = data.description;

    json gpus_array = json::array();
    for (const auto& entry : data.gpus) {
        gpus_array.push_back(SerializeGPUEntry(entry));
    }
    root["gpus"] = gpus_array;

    return root;
}

// ============================================================================
// Реализация Singleton
// ============================================================================

GPUConfig& GPUConfig::GetInstance() {
    static GPUConfig instance;
    return instance;
}

GPUConfig::GPUConfig() {
    // Инициализация конфигом по умолчанию
    data_ = CreateDefaultConfig();
}

// ============================================================================
// Загрузка и сохранение
// ============================================================================

bool GPUConfig::Load(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        // Открыть и разобрать JSON-файл
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "[GPUConfig] ERROR: Cannot open file: " << file_path << "\n";
            return false;
        }

        json root = json::parse(file);

        // Разбор корневых полей
        GPUConfigData new_data;
        new_data.version     = root.value("version", new_data.version);
        new_data.description = root.value("description", new_data.description);

        // Разбор записей GPU
        if (root.contains("gpus") && root["gpus"].is_array()) {
            for (const auto& gpu_json : root["gpus"]) {
                new_data.gpus.push_back(ParseGPUEntry(gpu_json));
            }
        }

        // Проверка: хотя бы один GPU
        if (new_data.gpus.empty()) {
            std::cerr << "[GPUConfig] WARNING: No GPUs in config, adding default\n";
            GPUConfigEntry default_gpu;
            default_gpu.id = 0;
            default_gpu.name = "TEST";
            default_gpu.is_prof = true;
            default_gpu.is_logger = true;
            new_data.gpus.push_back(default_gpu);
        }

        // Применить
        data_ = std::move(new_data);
        file_path_ = file_path;
        loaded_ = true;

        std::cout << "[GPUConfig] Loaded " << data_.gpus.size()
                  << " GPU config(s) from: " << file_path << "\n";

        return true;

    } catch (const json::parse_error& e) {
        std::cerr << "[GPUConfig] JSON parse error: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[GPUConfig] Load error: " << e.what() << "\n";
        return false;
    }
}

bool GPUConfig::LoadOrCreate(const std::string& file_path) {
    // Попытка загрузить существующий файл
    if (fs::exists(file_path)) {
        return Load(file_path);
    }

    // Файл не найден — создаём конфиг по умолчанию
    std::cout << "[GPUConfig] Config file not found, creating default: " << file_path << "\n";

    {
        std::lock_guard<std::mutex> lock(mutex_);
        data_ = CreateDefaultConfig();
        file_path_ = file_path;
        loaded_ = true;
    }

    // Сохранить конфиг по умолчанию в файл
    return Save(file_path);
}

bool GPUConfig::Save(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string path = file_path.empty() ? file_path_ : file_path;
    if (path.empty()) {
        std::cerr << "[GPUConfig] ERROR: No file path specified for Save()\n";
        return false;
    }

    try {
        // Создать родительские директории при необходимости
        fs::path dir = fs::path(path).parent_path();
        if (!dir.empty() && !fs::exists(dir)) {
            fs::create_directories(dir);
        }

        // Сериализация в JSON
        json root = SerializeConfigData(data_);

        // Запись в файл (с отступами в 2 пробела)
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[GPUConfig] ERROR: Cannot create file: " << path << "\n";
            return false;
        }

        file << root.dump(2) << "\n";
        file.close();

        file_path_ = path;

        std::cout << "[GPUConfig] Saved " << data_.gpus.size()
                  << " GPU config(s) to: " << path << "\n";

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[GPUConfig] Save error: " << e.what() << "\n";
        return false;
    }
}

bool GPUConfig::IsLoaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaded_;
}

// ============================================================================
// Доступ к конфигурации
// ============================================================================

const GPUConfigEntry& GPUConfig::GetConfig(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    if (found) {
        return *found;
    }

    // Вернуть запись по умолчанию с запрошенным id
    default_entry_ = GPUConfigEntry();
    default_entry_.id = gpu_id;
    return default_entry_;
}

const std::vector<GPUConfigEntry>& GPUConfig::GetAllConfigs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_.gpus;
}

const GPUConfigData& GPUConfig::GetData() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_;
}

std::vector<int> GPUConfig::GetActiveGPUIDs() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<int> active_ids;
    for (const auto& entry : data_.gpus) {
        if (entry.is_active) {
            active_ids.push_back(entry.id);
        }
    }
    return active_ids;
}

bool GPUConfig::IsProfilingEnabled(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->is_prof : false;
}

bool GPUConfig::IsLoggingEnabled(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->is_logger : false;
}

bool GPUConfig::IsConsoleEnabled(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->is_console : false;
}

size_t GPUConfig::GetMaxMemoryPercent(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->max_memory_percent : 70;
}

// ============================================================================
// Изменение конфигурации
// ============================================================================

void GPUConfig::SetConfig(const GPUConfigEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Найти существующую запись с тем же id
    for (auto& existing : data_.gpus) {
        if (existing.id == entry.id) {
            existing = entry;
            return;
        }
    }

    // Не найдено — добавить новую запись
    data_.gpus.push_back(entry);
}

void GPUConfig::ResetToDefault() {
    std::lock_guard<std::mutex> lock(mutex_);
    data_ = CreateDefaultConfig();
    loaded_ = false;
    file_path_.clear();
}

// ============================================================================
// Утилиты
// ============================================================================

std::string GPUConfig::GetFilePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return file_path_;
}

void GPUConfig::Print() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║              GPU Configuration                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    std::cout << "  Version: " << data_.version << "\n";
    std::cout << "  File: " << (file_path_.empty() ? "(not saved)" : file_path_) << "\n";
    std::cout << "  GPUs: " << data_.gpus.size() << "\n";
    std::cout << "\n";

    for (const auto& entry : data_.gpus) {
        std::cout << "  ┌─ GPU " << entry.id << ": \"" << entry.name << "\"\n";
        std::cout << "  │  Active:  " << (entry.is_active ? "YES" : "NO") << "\n";
        std::cout << "  │  Prof:    " << (entry.is_prof ? "ON" : "off") << "\n";
        std::cout << "  │  Logger:  " << (entry.is_logger ? "ON" : "off") << "\n";
        std::cout << "  │  Console: " << (entry.is_console ? "ON" : "off") << "\n";
        std::cout << "  │  DB:      " << (entry.is_db ? "ON" : "off") << "\n";
        std::cout << "  │  MaxMem:  " << entry.max_memory_percent << "%\n";
        std::cout << "  │  LogLvl:  " << entry.log_level << "\n";
        std::cout << "  └───────────────────────────────────\n";
    }
    std::cout << "\n";
}

// ============================================================================
// Приватные методы
// ============================================================================

GPUConfigData GPUConfig::CreateDefaultConfig() const {
    GPUConfigData data;
    data.version = "1.0";
    data.description = "GPU Configuration for DrvGPU";

    // По умолчанию: один GPU с включённым профилированием и логированием
    GPUConfigEntry default_gpu;
    default_gpu.id = 0;
    default_gpu.name = "TEST";
    default_gpu.is_prof = true;
    default_gpu.is_logger = true;
    default_gpu.is_console = true;
    default_gpu.is_active = true;

    data.gpus.push_back(default_gpu);

    return data;
}

const GPUConfigEntry* GPUConfig::FindConfig(int gpu_id) const {
    // ВАЖНО: вызывающий код должен удерживать mutex_!
    for (const auto& entry : data_.gpus) {
        if (entry.id == gpu_id) {
            return &entry;
        }
    }
    return nullptr;
}

} // namespace drv_gpu_lib
