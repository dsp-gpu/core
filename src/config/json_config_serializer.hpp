#pragma once

/**
 * @file json_config_serializer.hpp
 * @brief JsonConfigReader / JsonConfigWriter — реализация на nlohmann/json
 *
 * ============================================================================
 * ВАЖНО:
 *   Этот заголовок PRIVATE (живёт в src/, не публикуется). Клиенты ядра
 *   никогда его не включают — они работают через IConfigReader/Writer
 *   + ConfigSerializerFactory.
 *
 *   Здесь nlohmann/json скрыт через PIMPL — ни заголовок, ни фабрика
 *   не имеют #include <nlohmann/...>.
 *
 * ПАТТЕРНЫ:
 *   - SOLID: SRP (один класс — один формат)
 *   - GoF:   Bridge / PIMPL (реализация скрыта за указателем)
 *   - Low Coupling: nlohmann виден ТОЛЬКО в .cpp
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-04-14
 */

#include <core/interface/i_config_reader.hpp>
#include <core/interface/i_config_writer.hpp>

#include <memory>
#include <string>

namespace drv_gpu_lib {

// ─── Reader ───────────────────────────────────────────────────────────────

/**
 * @class JsonConfigReader
 * @brief IConfigReader на базе nlohmann/json. nlohmann скрыт через PIMPL.
 */
class JsonConfigReader : public IConfigReader {
public:
    JsonConfigReader();
    ~JsonConfigReader() override;

    // IConfigReader
    bool LoadFromFile(const std::string& path) override;
    bool LoadFromString(const std::string& content) override;

    bool        Has(const std::string& key) const override;
    std::string GetString(const std::string& key, const std::string& def = "") const override;
    int         GetInt(const std::string& key, int def = 0) const override;
    double      GetDouble(const std::string& key, double def = 0.0) const override;
    bool        GetBool(const std::string& key, bool def = false) const override;

    IConfigReaderPtr              GetObject(const std::string& key) const override;
    std::vector<IConfigReaderPtr> GetArray(const std::string& key) const override;

private:
    class Impl;                         // forward — nlohmann невиден
    std::unique_ptr<Impl> impl_;        // PIMPL

    // Внутренний конструктор для sub-reader'ов (GetObject / GetArray)
    explicit JsonConfigReader(std::unique_ptr<Impl> impl);
};

// ─── Writer ───────────────────────────────────────────────────────────────

/**
 * @class JsonConfigWriter
 * @brief IConfigWriter на базе nlohmann/json. nlohmann скрыт через PIMPL.
 */
class JsonConfigWriter : public IConfigWriter {
public:
    JsonConfigWriter();
    ~JsonConfigWriter() override;

    // IConfigWriter
    void SetString(const std::string& key, const std::string& value) override;
    void SetInt(const std::string& key, int value) override;
    void SetDouble(const std::string& key, double value) override;
    void SetBool(const std::string& key, bool value) override;

    IConfigWriterPtr BeginObject(const std::string& key) override;
    IConfigWriterPtr AppendArrayItem(const std::string& key) override;

    bool        SaveToFile(const std::string& path) const override;
    std::string SaveToString() const override;

private:
    class Impl;                         // forward — nlohmann невиден
    std::unique_ptr<Impl> impl_;        // PIMPL

    explicit JsonConfigWriter(std::unique_ptr<Impl> impl);
};

} // namespace drv_gpu_lib
