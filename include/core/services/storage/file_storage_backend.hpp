#pragma once

/**
 * @file file_storage_backend.hpp
 * @brief File-based implementation of IStorageBackend
 *
 * Stores data as files in base_dir. Keys with '/' create subdirectories.
 * Uses std::filesystem (C++17).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "i_storage_backend.hpp"

#include <string>
#include <vector>
#include <cstdint>

namespace drv_gpu_lib {

// Текущая единственная реализация IStorageBackend — хранит данные как файлы на диске.
// Ключи маппируются в пути: "filters/lp_5000.json" → base_dir/filters/lp_5000.json
// Разделители '/' в ключе создают подкаталоги автоматически (fs::create_directories).
// Планируемая альтернатива: SqliteStorageBackend (для атомарных транзакций).
class FileStorageBackend : public IStorageBackend {
public:
  explicit FileStorageBackend(const std::string& base_dir);

  void Save(const std::string& key, const std::vector<uint8_t>& data) override;
  std::vector<uint8_t> Load(const std::string& key) const override;
  std::vector<std::string> List(const std::string& prefix = "") const override;
  bool Exists(const std::string& key) const override;

  /// Путь к корневой директории хранилища
  const std::string& GetBaseDir() const { return base_dir_; }

private:
  std::string base_dir_;

  // Конвертирует ключ в полный путь: key "filters/x.json" → base_dir_/filters/x.json
  std::string KeyToPath(const std::string& key) const;
};

} // namespace drv_gpu_lib