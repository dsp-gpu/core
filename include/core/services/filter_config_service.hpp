#pragma once

/**
 * @file filter_config_service.hpp
 * @brief Service for saving/loading filter configurations (coefficients + metadata)
 *
 * Unlike kernel cache (source+binary), filters store:
 * - FIR: type + coefficients array
 * - IIR: type + biquad sections (b0,b1,b2,a1,a2)
 *
 * Storage: JSON files via FileStorageBackend.
 * Keys: filters/{name}.json
 * Versioning: on overwrite, old -> name_00, name_01, ...
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "storage/i_storage_backend.hpp"

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cstdint>

namespace drv_gpu_lib {

/**
 * @struct FilterConfigData
 * @brief Universal filter configuration (FIR or IIR)
 */
struct FilterConfigData {
  std::string name;
  std::string type;        ///< "fir" or "iir"
  std::string comment;
  std::string created;     ///< ISO 8601

  // FIR coefficients
  std::vector<float> coefficients;

  // IIR biquad sections: [b0, b1, b2, a1, a2] per section
  std::vector<std::array<float, 5>> sections;
};

class FilterConfigService {
public:
  /**
   * @param base_dir Root directory for filter configs
   */
  explicit FilterConfigService(const std::string& base_dir);

  /**
   * @brief Save filter configuration as JSON
   * @param name Filter name (e.g. "lp_5000")
   * @param data Filter configuration
   * @param comment Optional comment override
   *
   * Saves to: base_dir/filters/{name}.json
   * On overwrite: old file -> name_00.json, name_01.json, ...
   */
  void Save(const std::string& name,
            const FilterConfigData& data,
            const std::string& comment = "");

  /**
   * @brief Load filter configuration from JSON
   * @param name Filter name
   * @return FilterConfigData populated from file
   * @throws std::runtime_error if not found
   */
  FilterConfigData Load(const std::string& name) const;

  /**
   * @brief List all saved filter names
   * @return Vector of filter names
   */
  std::vector<std::string> ListFilters() const;

  /**
   * @brief Check if filter config exists
   * @param name Filter name
   */
  bool Exists(const std::string& name) const;

private:
  std::string base_dir_;
  std::unique_ptr<IStorageBackend> storage_;

  /// Version old files on overwrite
  void VersionOldFiles(const std::string& name) const;

  /// Get ISO 8601 timestamp
  static std::string GetTimestamp();

  /// Serialize FilterConfigData to JSON string
  static std::string ToJson(const std::string& name,
                            const FilterConfigData& data,
                            const std::string& comment,
                            const std::string& timestamp);

  /// Parse FilterConfigData from JSON string
  static FilterConfigData FromJson(const std::string& json);
};

} // namespace drv_gpu_lib