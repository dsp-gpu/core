/**
 * @file json_config_serializer.cpp
 * @brief Реализация JsonConfigReader / JsonConfigWriter на nlohmann/json
 *
 * nlohmann/json виден ТОЛЬКО в этом файле. Клиенты ядра работают через
 * IConfigReader/Writer + ConfigSerializerFactory.
 *
 * @author Codo (AI Assistant)
 * @date 2026-04-14
 */

#include "json_config_serializer.hpp"
#include <core/config/config_serializer_factory.hpp>

// ═════════════════════════════════════════════════════════════════════════
// nlohmann/json — единственное место во всём ядре где виден этот заголовок.
// Клиенты ядра никогда не включают его и не зависят от него.
// ═════════════════════════════════════════════════════════════════════════
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace drv_gpu_lib {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

/// Экранирование сегмента для RFC6901 json_pointer (~ → ~0, / → ~1)
std::string EscapePointerSegment(const std::string& segment) {
    std::string out;
    out.reserve(segment.size());
    for (char c : segment) {
        if (c == '~')      out += "~0";
        else if (c == '/') out += "~1";
        else               out += c;
    }
    return out;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// JsonConfigReader::Impl — приватная реализация
// ═══════════════════════════════════════════════════════════════════════════

class JsonConfigReader::Impl {
public:
    /// Текущий подузел (для корневого reader'а — всё дерево целиком).
    /// shared_ptr чтобы sub-reader'ы могли безопасно делить дерево.
    std::shared_ptr<json> node_ = std::make_shared<json>(json::object());
};

JsonConfigReader::JsonConfigReader()
    : impl_(std::make_unique<Impl>()) {}

JsonConfigReader::JsonConfigReader(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

JsonConfigReader::~JsonConfigReader() = default;

bool JsonConfigReader::LoadFromFile(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[JsonConfigReader] ERROR: Cannot open file: " << path << "\n";
            return false;
        }
        *impl_->node_ = json::parse(file);
        return true;
    } catch (const json::parse_error& e) {
        std::cerr << "[JsonConfigReader] JSON parse error: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[JsonConfigReader] Load error: " << e.what() << "\n";
        return false;
    }
}

bool JsonConfigReader::LoadFromString(const std::string& content) {
    try {
        *impl_->node_ = json::parse(content);
        return true;
    } catch (const json::parse_error& e) {
        std::cerr << "[JsonConfigReader] JSON parse error: " << e.what() << "\n";
        return false;
    }
}

bool JsonConfigReader::Has(const std::string& key) const {
    return impl_->node_->is_object() && impl_->node_->contains(key);
}

std::string JsonConfigReader::GetString(const std::string& key, const std::string& def) const {
    if (!Has(key)) return def;
    const auto& v = (*impl_->node_)[key];
    if (!v.is_string()) return def;
    return v.get<std::string>();
}

int JsonConfigReader::GetInt(const std::string& key, int def) const {
    if (!Has(key)) return def;
    const auto& v = (*impl_->node_)[key];
    if (!v.is_number_integer()) return def;
    return v.get<int>();
}

double JsonConfigReader::GetDouble(const std::string& key, double def) const {
    if (!Has(key)) return def;
    const auto& v = (*impl_->node_)[key];
    if (!v.is_number()) return def;
    return v.get<double>();
}

bool JsonConfigReader::GetBool(const std::string& key, bool def) const {
    if (!Has(key)) return def;
    const auto& v = (*impl_->node_)[key];
    if (!v.is_boolean()) return def;
    return v.get<bool>();
}

IConfigReaderPtr JsonConfigReader::GetObject(const std::string& key) const {
    if (!Has(key) || !(*impl_->node_)[key].is_object()) return nullptr;
    auto sub_impl = std::make_unique<Impl>();
    // копия подузла: reader только читает, копия безопасна и разрывает coupling
    *sub_impl->node_ = (*impl_->node_)[key];
    return IConfigReaderPtr(new JsonConfigReader(std::move(sub_impl)));
}

std::vector<IConfigReaderPtr> JsonConfigReader::GetArray(const std::string& key) const {
    std::vector<IConfigReaderPtr> result;
    if (!Has(key) || !(*impl_->node_)[key].is_array()) return result;
    const auto& arr = (*impl_->node_)[key];
    result.reserve(arr.size());
    for (const auto& item : arr) {
        auto sub_impl = std::make_unique<Impl>();
        *sub_impl->node_ = item;  // копия элемента
        result.push_back(IConfigReaderPtr(new JsonConfigReader(std::move(sub_impl))));
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// JsonConfigWriter::Impl — приватная реализация
// ═══════════════════════════════════════════════════════════════════════════

class JsonConfigWriter::Impl {
public:
    /// Общее корневое дерево (разделяется всеми sub-writer'ами).
    std::shared_ptr<json> tree_ = std::make_shared<json>(json::object());

    /// JSON-pointer-строка (RFC6901) от корня до текущего узла.
    /// Для корневого writer'а — пустая. Для sub — "/gpus/0" и т.п.
    std::string pointer_;

    /// Разыменовать текущий узел в дереве.
    json& Resolve() const {
        if (pointer_.empty()) return *tree_;
        return tree_->operator[](json::json_pointer(pointer_));
    }
};

JsonConfigWriter::JsonConfigWriter()
    : impl_(std::make_unique<Impl>()) {}

JsonConfigWriter::JsonConfigWriter(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

JsonConfigWriter::~JsonConfigWriter() = default;

void JsonConfigWriter::SetString(const std::string& key, const std::string& value) {
    impl_->Resolve()[key] = value;
}

void JsonConfigWriter::SetInt(const std::string& key, int value) {
    impl_->Resolve()[key] = value;
}

void JsonConfigWriter::SetDouble(const std::string& key, double value) {
    impl_->Resolve()[key] = value;
}

void JsonConfigWriter::SetBool(const std::string& key, bool value) {
    impl_->Resolve()[key] = value;
}

IConfigWriterPtr JsonConfigWriter::BeginObject(const std::string& key) {
    // Создать пустой объект под ключом (или перезаписать)
    impl_->Resolve()[key] = json::object();

    // sub-writer использует ТО ЖЕ корневое дерево + удлинённый pointer
    auto sub_impl = std::make_unique<Impl>();
    sub_impl->tree_ = impl_->tree_;  // shared: изменения sub видны в корне
    sub_impl->pointer_ = impl_->pointer_ + "/" + EscapePointerSegment(key);
    return IConfigWriterPtr(new JsonConfigWriter(std::move(sub_impl)));
}

IConfigWriterPtr JsonConfigWriter::AppendArrayItem(const std::string& key) {
    json& node = impl_->Resolve();

    // Убеждаемся что по ключу лежит массив
    if (!node.contains(key) || !node[key].is_array()) {
        node[key] = json::array();
    }

    const size_t new_index = node[key].size();
    node[key].push_back(json::object());

    // sub-writer на новый элемент массива
    auto sub_impl = std::make_unique<Impl>();
    sub_impl->tree_ = impl_->tree_;
    sub_impl->pointer_ = impl_->pointer_ + "/" + EscapePointerSegment(key)
                       + "/" + std::to_string(new_index);
    return IConfigWriterPtr(new JsonConfigWriter(std::move(sub_impl)));
}

bool JsonConfigWriter::SaveToFile(const std::string& path) const {
    try {
        fs::path dir = fs::path(path).parent_path();
        if (!dir.empty() && !fs::exists(dir)) {
            fs::create_directories(dir);
        }
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[JsonConfigWriter] ERROR: Cannot create file: " << path << "\n";
            return false;
        }
        file << impl_->tree_->dump(2) << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[JsonConfigWriter] Save error: " << e.what() << "\n";
        return false;
    }
}

std::string JsonConfigWriter::SaveToString() const {
    return impl_->tree_->dump(2);
}

// ═══════════════════════════════════════════════════════════════════════════
// ConfigSerializerFactory — реализация фабрики
// ═══════════════════════════════════════════════════════════════════════════

IConfigReaderPtr ConfigSerializerFactory::CreateJsonReader() {
    return std::make_shared<JsonConfigReader>();
}

IConfigWriterPtr ConfigSerializerFactory::CreateJsonWriter() {
    return std::make_shared<JsonConfigWriter>();
}

} // namespace drv_gpu_lib
