/**
 * @file gpu_config.cpp
 * @brief Реализация GPUConfig — менеджер конфигурации GPU
 *
 * ============================================================================
 * АРХИТЕКТУРА (после рефакторинга 2026-04-14):
 *
 *   GPUConfig не зависит от конкретного JSON-парсера. Использует только
 *   IConfigReader / IConfigWriter (абстракции) + ConfigSerializerFactory
 *   для создания конкретной реализации.
 *
 *   Замена nlohmann → rapidjson / yaml-cpp / toml11 сводится к замене ОДНОГО
 *   файла (`src/config/json_config_serializer.cpp`) — GPUConfig не трогаем.
 *
 *   Паттерны: SOLID (DIP), GoF (Factory Method, Composite).
 *
 * Потокобезопасность:
 *   Все публичные методы захватывают mutex перед доступом к data_.
 *
 * Обработка ошибок:
 *   - Ошибки разбора: логируются в stderr, возвращается false
 *   - Отсутствующие поля: используются значения по умолчанию из GPUConfigEntry
 *   - Отсутствующий файл в LoadOrCreate: создаётся конфиг по умолчанию и сохраняется
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 * @modified 2026-04-14 (nlohmann → IConfigReader/Writer через фабрику)
 */

#include <core/config/gpu_config.hpp>

#include <core/config/config_serializer_factory.hpp>
#include <core/interface/i_config_reader.hpp>
#include <core/interface/i_config_writer.hpp>
#include <core/logger/logger.hpp>

#include <iostream>  // std::cout for Print() formatted output
#include <filesystem>

namespace fs = std::filesystem;

namespace drv_gpu_lib {

// ============================================================================
// Вспомогательные функции (работают через абстракции, а не через json)
// ============================================================================

/// Прочитать одну запись GPUConfigEntry из sub-reader'а (один элемент массива).
static GPUConfigEntry ParseGPUEntry(const IConfigReader& reader) {
    GPUConfigEntry entry;

    // Идентификация
    entry.id                = reader.GetInt   ("id",                  entry.id);
    entry.name              = reader.GetString("name",                entry.name);

    // Флаги возможностей
    entry.is_prof           = reader.GetBool  ("is_prof",             entry.is_prof);
    entry.is_logger         = reader.GetBool  ("is_logger",           entry.is_logger);
    entry.is_console        = reader.GetBool  ("is_console",          entry.is_console);
    entry.is_active         = reader.GetBool  ("is_active",           entry.is_active);
    entry.is_db             = reader.GetBool  ("is_db",               entry.is_db);

    // Лимиты ресурсов
    entry.max_memory_percent = static_cast<size_t>(
        reader.GetInt("max_memory_percent",
                      static_cast<int>(entry.max_memory_percent)));

    // Настройки логирования — уровень лога как строка ("DEBUG"/"INFO"/...)
    entry.log_level         = reader.GetString("log_level",           entry.log_level);

    return entry;
}

/// Записать одну запись GPUConfigEntry в sub-writer (элемент массива).
static void SerializeGPUEntry(IConfigWriter& writer, const GPUConfigEntry& entry) {
    writer.SetInt   ("id",                 entry.id);
    writer.SetString("name",               entry.name);
    writer.SetBool  ("is_prof",            entry.is_prof);
    writer.SetBool  ("is_logger",          entry.is_logger);
    writer.SetBool  ("is_console",         entry.is_console);
    writer.SetBool  ("is_active",          entry.is_active);
    writer.SetBool  ("is_db",              entry.is_db);
    writer.SetInt   ("max_memory_percent", static_cast<int>(entry.max_memory_percent));
    writer.SetString("log_level",          entry.log_level);
}

// ============================================================================
// Singleton
// ============================================================================

GPUConfig& GPUConfig::GetInstance() {
    static GPUConfig instance;
    return instance;
}

GPUConfig::GPUConfig() {
    data_ = CreateDefaultConfig();
}

// ============================================================================
// Загрузка и сохранение
// ============================================================================

bool GPUConfig::Load(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    // DIP: зависим от абстракции, фабрика скрывает что внутри nlohmann.
    auto reader = ConfigSerializerFactory::CreateJsonReader();
    if (!reader || !reader->LoadFromFile(file_path)) {
        // BOOTSTRAP: std::cerr — Logger может вызвать ConfigLogger → GPUConfig (circular)
        std::cerr << "[GPUConfig] ERROR: Cannot load file: " << file_path << "\n";
        return false;
    }

    GPUConfigData new_data;
    new_data.version     = reader->GetString("version",     new_data.version);
    new_data.description = reader->GetString("description", new_data.description);

    // Composite: массив gpus — вектор sub-reader'ов на элементы
    for (const auto& gpu_reader : reader->GetArray("gpus")) {
        if (gpu_reader) {
            new_data.gpus.push_back(ParseGPUEntry(*gpu_reader));
        }
    }

    // Защита: хотя бы один GPU
    if (new_data.gpus.empty()) {
        // BOOTSTRAP: std::cerr — Logger ↔ GPUConfig circular dependency
        std::cerr << "[GPUConfig] WARNING: No GPUs in config, adding default\n";
        GPUConfigEntry default_gpu;
        default_gpu.id        = 0;
        default_gpu.name      = "TEST";
        default_gpu.is_prof   = true;
        default_gpu.is_logger = true;
        new_data.gpus.push_back(default_gpu);
    }

    data_ = std::move(new_data);
    file_path_ = file_path;
    loaded_ = true;

    // BOOTSTRAP: std::cout — Logger ↔ GPUConfig circular dependency
    std::cout << "[GPUConfig] Loaded " << data_.gpus.size()
              << " GPU config(s) from: " << file_path << "\n";

    return true;
}

bool GPUConfig::LoadOrCreate(const std::string& file_path) {
    if (fs::exists(file_path)) {
        return Load(file_path);
    }

    // BOOTSTRAP: std::cout — Logger ↔ GPUConfig circular dependency
    std::cout << "[GPUConfig] Config file not found, creating default: " << file_path << "\n";

    {
        std::lock_guard<std::mutex> lock(mutex_);
        data_ = CreateDefaultConfig();
        file_path_ = file_path;
        loaded_ = true;
    }

    return Save(file_path);
}

bool GPUConfig::Save(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string path = file_path.empty() ? file_path_ : file_path;
    if (path.empty()) {
        // BOOTSTRAP: std::cerr — Logger ↔ GPUConfig circular dependency
        std::cerr << "[GPUConfig] ERROR: No file path specified for Save()\n";
        return false;
    }

    // DIP: через фабрику — nlohmann невиден
    auto writer = ConfigSerializerFactory::CreateJsonWriter();
    if (!writer) return false;

    writer->SetString("version",     data_.version);
    writer->SetString("description", data_.description);

    // Composite: массив gpus — для каждого GPU получаем sub-writer и заполняем
    for (const auto& entry : data_.gpus) {
        auto gpu_writer = writer->AppendArrayItem("gpus");
        if (gpu_writer) {
            SerializeGPUEntry(*gpu_writer, entry);
        }
    }

    if (!writer->SaveToFile(path)) {
        // BOOTSTRAP: std::cerr — Logger ↔ GPUConfig circular dependency
        std::cerr << "[GPUConfig] ERROR: Cannot save file: " << path << "\n";
        return false;
    }

    file_path_ = path;

    // BOOTSTRAP: std::cout — Logger ↔ GPUConfig circular dependency
    std::cout << "[GPUConfig] Saved " << data_.gpus.size()
              << " GPU config(s) to: " << path << "\n";

    return true;
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

    for (auto& existing : data_.gpus) {
        if (existing.id == entry.id) {
            existing = entry;
            return;
        }
    }

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
