#pragma once

/**
 * @file i_storage_backend.hpp
 * @brief Abstract storage backend interface for DrvGPU services
 *
 * Different instances with different base_dir for different modules.
 * FileStorageBackend -> (future) SqliteStorageBackend.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include <string>
#include <vector>
#include <cstdint>

namespace drv_gpu_lib {

/**
 * @brief Abstract backend for data storage (files, SQLite)
 *
 * Each instance has its own base_dir (separate folders per module).
 * Key may contain '/' — interpreted as subdirectory path.
 */
struct IStorageBackend {
  virtual ~IStorageBackend() = default;

  /**
   * @brief Save data by key
   * @param key Relative key (e.g. "test/key.bin", "filters/lp_5000.json")
   * @param data Binary payload
   */
  virtual void Save(const std::string& key, const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Load data by key
   * @param key Relative key
   * @return Binary payload
   * @throws std::runtime_error if key not found
   */
  virtual std::vector<uint8_t> Load(const std::string& key) const = 0;

  /**
   * @brief List all keys with given prefix
   * @param prefix Key prefix filter (empty = all)
   * @return Vector of matching keys (relative paths)
   */
  virtual std::vector<std::string> List(const std::string& prefix = "") const = 0;

  /**
   * @brief Check if key exists
   * @param key Relative key
   * @return true if key exists in storage
   */
  virtual bool Exists(const std::string& key) const = 0;
};

} // namespace drv_gpu_lib