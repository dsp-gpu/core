#pragma once

/**
 * @file kernel_cache_service.hpp
 * @brief On-disk cache for compiled OpenCL/ROCm kernels
 *
 * Storage-agnostic: does NOT know OpenCL. Returns {source, binary}.
 * Caller creates cl_program via clCreateProgramWithBinary or clCreateProgramWithSource.
 *
 * Features:
 * - Save: .cl source + binary + manifest.json
 * - Load: binary (fast path) or source (fallback)
 * - Versioning: old files renamed _00, _01, ...
 * - ROCm support: binary suffix _opencl.bin / _rocm.hsaco
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "common/backend_type.hpp"

#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace drv_gpu_lib {

class KernelCacheService {
public:
  /**
   * @param base_dir Root cache directory (e.g. "modules/signal_generators/kernels")
   * @param backend_type OPENCL or ROCm — determines binary file suffix
   */
  KernelCacheService(const std::string& base_dir,
                     BackendType backend_type = BackendType::OPENCL);

  /**
   * @brief Cached kernel entry (source + binary)
   */
  struct CacheEntry {
    std::string source;             ///< .cl source content
    std::vector<uint8_t> binary;    ///< Compiled binary blob

    bool has_binary() const { return !binary.empty(); }
    bool has_source() const { return !source.empty(); }
  };

  /**
   * @brief Save kernel to disk
   * @param name     Kernel name (no extension): "my_signal"
   * @param cl_source OpenCL source code
   * @param binary   Compiled binary blob
   * @param metadata Extra metadata string (e.g. params)
   * @param comment  Human-readable comment
   *
   * Creates: name.cl + bin/name_{backend}.bin + manifest.json entry
   * Versioning: if name exists, old files -> name_00, name_01, ...
   */
  void Save(const std::string& name,
            const std::string& cl_source,
            const std::vector<uint8_t>& binary,
            const std::string& metadata = "",
            const std::string& comment = "");

  /**
   * @brief Load kernel from disk
   * @param name Kernel name (no extension)
   * @return CacheEntry with source and/or binary, or nullopt if not found
   *
   * Fast path: binary exists -> return {source, binary}
   * Fallback: only source exists -> return {source, {}}
   * Cache miss: return nullopt (no throw)
   */
  std::optional<CacheEntry> Load(const std::string& name) const;

  /**
   * @brief List cached kernel names from manifest.json
   * @return Vector of kernel names
   */
  std::vector<std::string> ListKernels() const;

  /// Get cache root directory
  std::string GetCacheDir() const { return base_dir_; }

  /// Get binary subdirectory (base_dir/bin/)
  std::string GetBinDir() const;

private:
  std::string base_dir_;
  BackendType backend_type_;

  /// Returns "_opencl.bin" or "_rocm.hsaco"
  std::string GetBinarySuffix() const;

  /// Rename existing files: name -> name_00, name_01, ...
  void VersionOldFiles(const std::string& name) const;

  /// Update manifest.json with new/updated entry
  void WriteManifestEntry(const std::string& name,
                          const std::string& metadata,
                          const std::string& comment) const;

  /// Get ISO 8601 timestamp
  static std::string GetTimestamp();
};

} // namespace drv_gpu_lib