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
 * - Per-arch subdir (gfx908/gfx1100/…) — корректно для multi-GPU
 * - Atomic write: name.tmp → rename(tmp, name) (POSIX atomic)
 * - Idempotent Save: если файл того же размера уже есть — skip IO
 * - ROCm support: binary suffix _opencl.bin / _rocm.hsaco
 *
 * Multi-GPU safety (1 GPU = 1 object = 1 thread, без блокировок):
 *   При параллельном Save из нескольких потоков с одинаковым source+arch —
 *   содержимое .hsaco побайтово идентично, atomic rename не даст увидеть
 *   "полузаписанный" файл, idempotent-check избежит лишнего IO.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22  (update: 2026-04-15 — per-arch + atomic + idempotent)
 */

#include <core/common/backend_type.hpp>

#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace drv_gpu_lib {

class KernelCacheService {
public:
  /**
   * @param base_dir     Root cache directory (e.g. "<exe_dir>/kernels_cache/capon")
   * @param backend_type OPENCL or ROCm — determines binary file suffix
   * @param arch         GPU architecture (e.g. "gfx908", "gfx1100"). Пустая —
   *                     per-arch подкаталог не создаётся (legacy поведение).
   *                     При непустом arch итоговая директория = base_dir/arch/.
   */
  KernelCacheService(const std::string& base_dir,
                     BackendType backend_type = BackendType::OPENCL,
                     const std::string& arch = "");

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
  std::string base_dir_;      ///< Итоговая директория (с учётом arch если задан).
  std::string arch_;          ///< GPU arch (для логирования). Пустая = legacy.
  BackendType backend_type_;

  /// Returns "_opencl.bin" or "_rocm.hsaco"
  std::string GetBinarySuffix() const;

  /// Rename existing files: name -> name_00, name_01, ...
  /// LEGACY — больше не вызывается в Save(). Оставлено для CLI-утилит.
  void VersionOldFiles(const std::string& name) const;

  /// Atomic write: пишет в path.tmp, затем fs::rename(path.tmp → path).
  /// POSIX гарантирует атомарность rename — читатели никогда не видят
  /// полузаписанный файл. throw std::runtime_error при ошибке IO.
  static void AtomicWrite(const std::string& path,
                          const void* data, size_t bytes);

  /// Idempotent-check: файл существует и его размер == expected_size.
  /// Используется перед Save чтобы пропустить запись идентичного содержимого.
  static bool FileSizeEquals(const std::string& path, size_t expected_size);

  /// Update manifest.json with new/updated entry
  void WriteManifestEntry(const std::string& name,
                          const std::string& metadata,
                          const std::string& comment) const;

  /// Get ISO 8601 timestamp
  static std::string GetTimestamp();
};

} // namespace drv_gpu_lib