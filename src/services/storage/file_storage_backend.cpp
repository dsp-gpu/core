/**
 * @file file_storage_backend.cpp
 * @brief File-based IStorageBackend implementation
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "file_storage_backend.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создаёт хранилище с корневым каталогом base_dir
 *
 * Каталог НЕ создаётся здесь — он создаётся лениво при первом Save().
 * @param base_dir Путь к корню хранилища; ключи маппируются относительно него
 */
FileStorageBackend::FileStorageBackend(const std::string& base_dir)
    : base_dir_(base_dir) {
}

// ════════════════════════════════════════════════════════════════════════════
// IStorageBackend interface implementation
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Сохраняет данные в файл по ключу, создавая подкаталоги при необходимости
 *
 * Ключи с '/' → автоматически создаётся иерархия директорий (create_directories).
 * Открывает файл в бинарном режиме — LF-only переносы на Windows.
 *
 * @param key  Относительный ключ (напр. "filters/lp_5000.json")
 * @param data Бинарный payload
 * @throws std::runtime_error если файл не удалось открыть на запись
 */
void FileStorageBackend::Save(const std::string& key,
                               const std::vector<uint8_t>& data) {
  std::string path = KeyToPath(key);

  // Create parent directories if needed
  fs::path p(path);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }

  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error(
        "FileStorageBackend::Save: cannot write " + path);
  }
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size()));
}

/**
 * @brief Загружает данные из файла по ключу в вектор байт
 *
 * Открываем в режиме ios::ate — позиция сразу в конце, tellg() = размер файла.
 * Один вызов вместо отдельного fs::file_size() — меньше системных вызовов.
 *
 * @param key Относительный ключ
 * @return Вектор байт содержимого файла
 * @throws std::runtime_error если файл не найден или не открывается
 */
std::vector<uint8_t> FileStorageBackend::Load(const std::string& key) const {
  std::string path = KeyToPath(key);

  // ios::ate — открываем с позицией в конце файла. tellg() сразу возвращает размер,
  // без отдельного stat() или fs::file_size() вызова. Затем seekg(0) → читаем с начала.
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    throw std::runtime_error(
        "FileStorageBackend::Load: cannot read " + path);
  }

  auto size = f.tellg();
  f.seekg(0, std::ios::beg);

  std::vector<uint8_t> data(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(data.data()),
         static_cast<std::streamsize>(size));

  return data;
}

/**
 * @brief Возвращает все ключи с заданным префиксом в лексикографическом порядке
 *
 * Рекурсивно обходит base_dir через recursive_directory_iterator.
 * Ключи — относительные пути с '/' разделителем (generic_string() — кросс-платформенно).
 * Пустой prefix → возвращает все ключи хранилища.
 * Несуществующий base_dir → пустой вектор (не бросает).
 *
 * @param prefix Фильтр-префикс ключа (напр. "filters/")
 * @return Отсортированный вектор ключей
 */
std::vector<std::string> FileStorageBackend::List(
    const std::string& prefix) const {
  std::vector<std::string> result;

  fs::path base(base_dir_);
  if (!fs::exists(base) || !fs::is_directory(base)) {
    return result;
  }

  for (const auto& entry : fs::recursive_directory_iterator(base)) {
    if (!entry.is_regular_file()) continue;

    // Get relative path as key.
    // generic_string() всегда возвращает '/' как разделитель (и на Windows),
    // что обеспечивает переносимость ключей хранилища между Windows и Linux.
    std::string rel = fs::relative(entry.path(), base).generic_string();

    // Apply prefix filter.
    // rfind(prefix, 0) ищет prefix начиная с позиции 0 — это эффективный starts_with.
    // Возвращает 0 если строка начинается с prefix, npos иначе.
    if (!prefix.empty()) {
      if (rel.rfind(prefix, 0) != 0) continue;  // doesn't start with prefix
    }

    result.push_back(rel);
  }

  std::sort(result.begin(), result.end());
  return result;
}

/**
 * @brief Проверяет существование файла по ключу
 * @param key Относительный ключ
 * @return true если соответствующий файл существует на диске
 */
bool FileStorageBackend::Exists(const std::string& key) const {
  return fs::exists(KeyToPath(key));
}

/**
 * @brief Конвертирует ключ хранилища в полный путь к файлу
 *
 * "filters/lp_5000.json" → base_dir_/filters/lp_5000.json
 * Использует fs::path оператор/ — корректная конкатенация с разделителем ОС.
 * .string() (не generic_string()) — возвращает нативный разделитель для fstream.
 *
 * @param key Относительный ключ
 * @return Полный путь к файлу (нативный разделитель ОС)
 */
std::string FileStorageBackend::KeyToPath(const std::string& key) const {
  // Use generic_string for consistent '/' separators
  return (fs::path(base_dir_) / key).string();
}

} // namespace drv_gpu_lib