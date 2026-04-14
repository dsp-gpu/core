#pragma once

/**
 * @file i_config_reader.hpp
 * @brief IConfigReader — универсальный интерфейс чтения конфигурации (JSON/YAML/TOML)
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Абстракция поверх конкретной реализации парсера конфига. Клиенты ядра
 *   (spectrum, stats, radar и т.д.) работают ТОЛЬКО через этот интерфейс и
 *   никогда не видят nlohmann/rapidjson/yaml-cpp.
 *
 * ПАТТЕРНЫ:
 *   - SOLID: SRP (чтение), OCP (новый формат = новый класс), DIP (клиент
 *            зависит от IConfigReader, не от реализации)
 *   - GoF:   Composite (sub-reader для массивов/объектов),
 *            Factory Method (см. ConfigSerializerFactory)
 *
 * ПРИМЕР:
 *   auto reader = ConfigSerializerFactory::CreateJsonReader();
 *   reader->LoadFromFile("configGPU.json");
 *   std::string ver = reader->GetString("version", "1.0");
 *   for (const auto& gpu_item : reader->GetArray("gpus")) {
 *       int id = gpu_item->GetInt("id", 0);
 *       bool prof = gpu_item->GetBool("is_prof", false);
 *   }
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-04-14
 */

#include <memory>
#include <string>
#include <vector>

namespace drv_gpu_lib {

class IConfigReader;
using IConfigReaderPtr = std::shared_ptr<IConfigReader>;

/**
 * @class IConfigReader
 * @brief Интерфейс чтения конфигурации (только чтение, ISP)
 *
 * Один и тот же интерфейс работает на любом уровне иерархии:
 * - корневой объект (после LoadFromFile)
 * - объект внутри ключа (GetObject)
 * - элемент массива (GetArray[i])
 *
 * Это реализует паттерн Composite GoF — единообразный API для листа и узла.
 */
class IConfigReader {
public:
    virtual ~IConfigReader() = default;

    // ─── Загрузка (актуально только для корневого reader'а) ──────────────

    /**
     * @brief Загрузить конфиг из файла
     * @return true при успешной загрузке
     */
    virtual bool LoadFromFile(const std::string& path) = 0;

    /**
     * @brief Загрузить конфиг из строки (для тестов или сетевых данных)
     */
    virtual bool LoadFromString(const std::string& content) = 0;

    // ─── Проверка наличия ключа ──────────────────────────────────────────

    /**
     * @brief Есть ли ключ в текущем объекте (верхнего уровня текущего reader'а)
     */
    virtual bool Has(const std::string& key) const = 0;

    // ─── Скалярные геттеры с значением по умолчанию ──────────────────────

    virtual std::string GetString(const std::string& key,
                                   const std::string& default_value = "") const = 0;
    virtual int         GetInt(const std::string& key,
                                int default_value = 0) const = 0;
    virtual double      GetDouble(const std::string& key,
                                   double default_value = 0.0) const = 0;
    virtual bool        GetBool(const std::string& key,
                                 bool default_value = false) const = 0;

    // ─── Композиция (Composite): вложенные объекты и массивы ─────────────

    /**
     * @brief Получить вложенный объект как sub-reader
     * @return nullptr если ключа нет или тип не объект
     */
    virtual IConfigReaderPtr GetObject(const std::string& key) const = 0;

    /**
     * @brief Получить массив как вектор sub-reader'ов
     * @return пустой вектор если ключа нет или тип не массив
     *
     * Каждый элемент массива оборачивается в свой IConfigReader,
     * что позволяет однообразно работать с массивами объектов:
     * @code
     *   for (const auto& item : reader->GetArray("gpus")) {
     *       int id = item->GetInt("id");
     *   }
     * @endcode
     */
    virtual std::vector<IConfigReaderPtr> GetArray(const std::string& key) const = 0;
};

} // namespace drv_gpu_lib
