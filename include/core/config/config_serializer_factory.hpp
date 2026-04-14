#pragma once

/**
 * @file config_serializer_factory.hpp
 * @brief ConfigSerializerFactory — фабрика IConfigReader / IConfigWriter
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Единая точка создания reader/writer. Клиент не знает какой конкретно
 *   класс создаётся (JsonConfigReader, YamlConfigReader, ...) — он получает
 *   только IConfigReaderPtr / IConfigWriterPtr.
 *
 * ПАТТЕРН:
 *   - GoF Factory Method — `CreateJsonReader()` / `CreateJsonWriter()`
 *   - OCP: добавить YAML = добавить методы `CreateYamlReader/Writer`, не
 *          трогая интерфейсы и существующий код.
 *
 * ПРИМЕР:
 *   auto reader = ConfigSerializerFactory::CreateJsonReader();
 *   auto writer = ConfigSerializerFactory::CreateJsonWriter();
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-04-14
 */

#include <core/interface/i_config_reader.hpp>
#include <core/interface/i_config_writer.hpp>

namespace drv_gpu_lib {

/**
 * @class ConfigSerializerFactory
 * @brief Фабрика конкретных reader'ов/writer'ов конфигурации
 */
class ConfigSerializerFactory {
public:
    /**
     * @brief Создать JSON reader (реализация на nlohmann/json, скрыта)
     */
    static IConfigReaderPtr CreateJsonReader();

    /**
     * @brief Создать JSON writer
     */
    static IConfigWriterPtr CreateJsonWriter();

    // Будущее (OCP — добавляем новые методы, старые не трогаем):
    // static IConfigReaderPtr CreateYamlReader();
    // static IConfigWriterPtr CreateYamlWriter();
    // static IConfigReaderPtr CreateTomlReader();

    // Утилитный конструктор запрещён — фабрика статическая
    ConfigSerializerFactory() = delete;
};

} // namespace drv_gpu_lib
