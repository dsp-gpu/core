#pragma once

/**
 * @file i_config_writer.hpp
 * @brief IConfigWriter — универсальный интерфейс записи конфигурации (JSON/YAML/TOML)
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Абстракция поверх конкретного сериализатора. Клиенты не знают ни про
 *   nlohmann::json::dump, ни про YAML emitter — только про Set* и Save*.
 *
 * ПАТТЕРНЫ:
 *   - SOLID: ISP (отдельный writer, не смешан с reader),
 *            OCP (новый формат = новый класс, интерфейс тот же),
 *            DIP (клиенты зависят от IConfigWriter)
 *   - GoF:   Composite (BeginObject/AppendArrayItem возвращают sub-writer),
 *            Factory Method (см. ConfigSerializerFactory)
 *
 * ПРИМЕР:
 *   auto writer = ConfigSerializerFactory::CreateJsonWriter();
 *   writer->SetString("version", "1.0");
 *   for (const auto& gpu : gpus) {
 *       auto item = writer->AppendArrayItem("gpus");
 *       item->SetInt("id", gpu.id);
 *       item->SetBool("is_prof", gpu.is_prof);
 *   }
 *   writer->SaveToFile("configGPU.json");
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-04-14
 */

#include <memory>
#include <string>

namespace drv_gpu_lib {

class IConfigWriter;
using IConfigWriterPtr = std::shared_ptr<IConfigWriter>;

/**
 * @class IConfigWriter
 * @brief Интерфейс записи конфигурации (только запись, ISP)
 *
 * Один и тот же интерфейс на всех уровнях иерархии (паттерн Composite):
 * - корневой writer (после Create*Writer)
 * - вложенный объект (BeginObject)
 * - элемент массива (AppendArrayItem)
 */
class IConfigWriter {
public:
    virtual ~IConfigWriter() = default;

    // ─── Скалярные сеттеры ───────────────────────────────────────────────

    virtual void SetString(const std::string& key, const std::string& value) = 0;
    virtual void SetInt(const std::string& key, int value) = 0;
    virtual void SetDouble(const std::string& key, double value) = 0;
    virtual void SetBool(const std::string& key, bool value) = 0;

    // ─── Композиция: вложенные объекты и массивы ─────────────────────────

    /**
     * @brief Начать вложенный объект и вернуть sub-writer на него
     *
     * Если ключ уже существует — перезаписывается. Sub-writer живёт
     * внутри родителя, его изменения сразу видны в родительском дереве.
     */
    virtual IConfigWriterPtr BeginObject(const std::string& key) = 0;

    /**
     * @brief Добавить новый объект в массив под указанным ключом
     *
     * Если массива ещё нет — создаётся. Возвращённый writer позволяет
     * заполнить новый элемент:
     * @code
     *   auto item = writer->AppendArrayItem("gpus");
     *   item->SetInt("id", 0);
     *   item->SetString("name", "Alex");
     * @endcode
     */
    virtual IConfigWriterPtr AppendArrayItem(const std::string& key) = 0;

    // ─── Сериализация (актуально только для корневого writer'а) ──────────

    /**
     * @brief Сохранить конфиг в файл
     * @return true при успехе
     */
    virtual bool SaveToFile(const std::string& path) const = 0;

    /**
     * @brief Сериализовать в строку (для тестов / сетевого обмена)
     */
    virtual std::string SaveToString() const = 0;
};

} // namespace drv_gpu_lib
