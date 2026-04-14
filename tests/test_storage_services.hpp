#pragma once

/**
 * @file test_storage_services.hpp
 * @brief Tests for FileStorageBackend, KernelCacheService, FilterConfigService
 *
 * Pure filesystem tests — no GPU required.
 * Uses temp directory, cleaned up after tests.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include <core/services/storage/file_storage_backend.hpp>
#include <core/services/kernel_cache_service.hpp>
#include <core/services/filter_config_service.hpp>

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>
#include <array>

namespace fs = std::filesystem;

namespace test_storage_services {

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Create a unique temp directory for tests
inline std::string CreateTempDir(const std::string& suffix) {
  auto p = fs::temp_directory_path() / ("gpuworklib_test_" + suffix);
  fs::create_directories(p);
  return p.string();
}

/// Remove temp directory recursively
inline void CleanupTempDir(const std::string& dir) {
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ════════════════════════════════════════════════════════════════════════════
// Test: FileStorageBackend
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFileStorageBackend() {
  std::cout << "\nTEST: FileStorageBackend\n";

  std::string dir = CreateTempDir("storage");
  bool ok = true;

  try {
    drv_gpu_lib::FileStorageBackend storage(dir);

    // 1. Save and Load
    std::vector<uint8_t> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F};  // "Hello"
    storage.Save("test/hello.bin", data);

    if (!storage.Exists("test/hello.bin")) {
      std::cout << "  [FAIL] Exists returned false after Save\n";
      ok = false;
    }

    auto loaded = storage.Load("test/hello.bin");
    if (loaded != data) {
      std::cout << "  [FAIL] Load returned different data\n";
      ok = false;
    }

    // 2. List
    storage.Save("test/world.bin", {0x57});
    storage.Save("other/file.txt", {0x41});

    auto all_keys = storage.List("");
    if (all_keys.size() < 3) {
      std::cout << "  [FAIL] List('') returned " << all_keys.size()
                << " keys, expected >= 3\n";
      ok = false;
    }

    auto test_keys = storage.List("test/");
    if (test_keys.size() != 2) {
      std::cout << "  [FAIL] List('test/') returned " << test_keys.size()
                << " keys, expected 2\n";
      ok = false;
    }

    // 3. Exists for missing key
    if (storage.Exists("nonexistent/key.bin")) {
      std::cout << "  [FAIL] Exists returned true for missing key\n";
      ok = false;
    }

    // 4. Overwrite
    std::vector<uint8_t> new_data = {0x42, 0x79, 0x65};  // "Bye"
    storage.Save("test/hello.bin", new_data);
    auto reloaded = storage.Load("test/hello.bin");
    if (reloaded != new_data) {
      std::cout << "  [FAIL] Overwrite did not update data\n";
      ok = false;
    }

  } catch (const std::exception& e) {
    std::cout << "  [FAIL] Exception: " << e.what() << "\n";
    ok = false;
  }

  CleanupTempDir(dir);
  std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " FileStorageBackend\n";
  return ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Test: KernelCacheService
// ════════════════════════════════════════════════════════════════════════════

inline bool TestKernelCacheService() {
  std::cout << "\nTEST: KernelCacheService\n";

  std::string dir = CreateTempDir("kernel_cache");
  bool ok = true;

  try {
    drv_gpu_lib::KernelCacheService cache(dir, drv_gpu_lib::BackendType::OPENCL);

    // 1. Save kernel
    std::string source = "__kernel void test(__global float* out) { out[get_global_id(0)] = 1.0f; }";
    std::vector<uint8_t> binary = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03};

    cache.Save("test_kernel", source, binary, "N=1024", "test comment");

    // 2. Load kernel
    auto entry = cache.Load("test_kernel");
    if (!entry) {
      std::cout << "  [FAIL] Load returned nullopt (cache miss after Save)\n";
      ok = false;
    } else {
      if (!entry->has_source()) {
        std::cout << "  [FAIL] Loaded entry has no source\n";
        ok = false;
      }
      if (!entry->has_binary()) {
        std::cout << "  [FAIL] Loaded entry has no binary\n";
        ok = false;
      }
      if (entry->source != source) {
        std::cout << "  [FAIL] Source mismatch\n";
        ok = false;
      }
      if (entry->binary != binary) {
        std::cout << "  [FAIL] Binary mismatch\n";
        ok = false;
      }
    }

    // 3. ListKernels
    auto names = cache.ListKernels();
    bool found = false;
    for (const auto& n : names) {
      if (n == "test_kernel") found = true;
    }
    if (!found) {
      std::cout << "  [FAIL] ListKernels does not contain 'test_kernel'\n";
      ok = false;
    }

    // 4. Versioning: save again -> old files renamed
    std::string source2 = "__kernel void test_v2(__global float* out) { out[get_global_id(0)] = 2.0f; }";
    std::vector<uint8_t> binary2 = {0xCA, 0xFE};
    cache.Save("test_kernel", source2, binary2, "N=2048", "updated");

    auto entry2 = cache.Load("test_kernel");
    if (!entry2) {
      std::cout << "  [FAIL] Load returned nullopt after re-Save\n";
      ok = false;
    } else {
      if (entry2->source != source2) {
        std::cout << "  [FAIL] Updated source mismatch\n";
        ok = false;
      }
      if (entry2->binary != binary2) {
        std::cout << "  [FAIL] Updated binary mismatch\n";
        ok = false;
      }
    }

    // Check that old versioned file exists
    bool old_exists = fs::exists(fs::path(dir) / "test_kernel_00.cl");
    if (!old_exists) {
      std::cout << "  [FAIL] Versioned file test_kernel_00.cl not found\n";
      ok = false;
    }

    // 5. Cache miss → nullopt (not found)
    auto miss = cache.Load("nonexistent_kernel");
    if (miss) {
      std::cout << "  [FAIL] Load('nonexistent_kernel') should return nullopt\n";
      ok = false;
    }

    // 6. GetBinDir / GetCacheDir (cross-platform: normalize paths for Windows)
    auto expected_bin = (fs::path(dir) / "bin").lexically_normal();
    auto actual_bin = fs::path(cache.GetBinDir()).lexically_normal();
    if (cache.GetCacheDir() != dir) {
      std::cout << "  [FAIL] GetCacheDir mismatch\n";
      ok = false;
    }
    if (actual_bin != expected_bin) {
      std::cout << "  [FAIL] GetBinDir mismatch (got: " << cache.GetBinDir()
                << ", expected: " << expected_bin.string() << ")\n";
      ok = false;
    }

  } catch (const std::exception& e) {
    std::cout << "  [FAIL] Exception: " << e.what() << "\n";
    ok = false;
  }

  CleanupTempDir(dir);
  std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " KernelCacheService\n";
  return ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Test: FilterConfigService
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFilterConfigService() {
  std::cout << "\nTEST: FilterConfigService\n";

  std::string dir = CreateTempDir("filter_config");
  bool ok = true;

  try {
    drv_gpu_lib::FilterConfigService svc(dir);

    // 1. Save FIR filter
    drv_gpu_lib::FilterConfigData fir;
    fir.name = "lp_5000";
    fir.type = "fir";
    fir.comment = "Lowpass 5kHz FIR";
    fir.coefficients = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};

    svc.Save("lp_5000", fir);

    // 2. Exists
    if (!svc.Exists("lp_5000")) {
      std::cout << "  [FAIL] Exists returned false for 'lp_5000'\n";
      ok = false;
    }

    // 3. Load FIR
    auto loaded_fir = svc.Load("lp_5000");
    if (loaded_fir.type != "fir") {
      std::cout << "  [FAIL] Loaded type != 'fir', got '" << loaded_fir.type << "'\n";
      ok = false;
    }
    if (loaded_fir.coefficients.size() != 5) {
      std::cout << "  [FAIL] Expected 5 coefficients, got "
                << loaded_fir.coefficients.size() << "\n";
      ok = false;
    }
    // Check coefficient values (within tolerance)
    for (size_t i = 0; i < fir.coefficients.size() && i < loaded_fir.coefficients.size(); ++i) {
      float diff = std::abs(fir.coefficients[i] - loaded_fir.coefficients[i]);
      if (diff > 1e-5f) {
        std::cout << "  [FAIL] Coefficient[" << i << "] mismatch: "
                  << fir.coefficients[i] << " vs " << loaded_fir.coefficients[i] << "\n";
        ok = false;
      }
    }

    // 4. Save IIR filter
    drv_gpu_lib::FilterConfigData iir;
    iir.name = "bp_1000";
    iir.type = "iir";
    iir.comment = "Bandpass 1kHz IIR";
    iir.sections = {
        {1.0f, -2.0f, 1.0f, -1.5f, 0.7f},
        {1.0f, 0.0f, -1.0f, -1.2f, 0.6f}
    };

    svc.Save("bp_1000", iir);

    // 5. Load IIR
    auto loaded_iir = svc.Load("bp_1000");
    if (loaded_iir.type != "iir") {
      std::cout << "  [FAIL] Loaded type != 'iir'\n";
      ok = false;
    }
    if (loaded_iir.sections.size() != 2) {
      std::cout << "  [FAIL] Expected 2 sections, got "
                << loaded_iir.sections.size() << "\n";
      ok = false;
    }
    // Check section values
    if (loaded_iir.sections.size() == 2) {
      float diff_b0 = std::abs(loaded_iir.sections[0][0] - 1.0f);
      float diff_a1 = std::abs(loaded_iir.sections[0][3] - (-1.5f));
      if (diff_b0 > 1e-5f || diff_a1 > 1e-5f) {
        std::cout << "  [FAIL] IIR section coefficient mismatch\n";
        ok = false;
      }
    }

    // 6. ListFilters
    auto names = svc.ListFilters();
    if (names.size() != 2) {
      std::cout << "  [FAIL] ListFilters returned " << names.size()
                << " names, expected 2\n";
      ok = false;
    }

    // 7. Versioning: overwrite FIR
    drv_gpu_lib::FilterConfigData fir2;
    fir2.name = "lp_5000";
    fir2.type = "fir";
    fir2.comment = "Updated Lowpass";
    fir2.coefficients = {0.05f, 0.15f, 0.3f, 0.3f, 0.15f, 0.05f};

    svc.Save("lp_5000", fir2, "v2 updated");

    auto loaded_fir2 = svc.Load("lp_5000");
    if (loaded_fir2.coefficients.size() != 6) {
      std::cout << "  [FAIL] Updated FIR: expected 6 coefficients, got "
                << loaded_fir2.coefficients.size() << "\n";
      ok = false;
    }

    // Check versioned file exists
    bool versioned = fs::exists(fs::path(dir) / "filters" / "lp_5000_00.json");
    if (!versioned) {
      std::cout << "  [FAIL] Versioned file lp_5000_00.json not found\n";
      ok = false;
    }

    // 8. Not found
    try {
      svc.Load("nonexistent");
      std::cout << "  [FAIL] Load('nonexistent') should have thrown\n";
      ok = false;
    } catch (const std::runtime_error&) {
      // Expected
    }

  } catch (const std::exception& e) {
    std::cout << "  [FAIL] Exception: " << e.what() << "\n";
    ok = false;
  }

  CleanupTempDir(dir);
  std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " FilterConfigService\n";
  return ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Run all
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
  std::cout << "\n****************************************************************\n";
  std::cout << "*         STORAGE SERVICES TEST SUITE                          *\n";
  std::cout << "****************************************************************\n";

  int pass = 0, fail = 0;

  if (TestFileStorageBackend()) pass++; else fail++;
  if (TestKernelCacheService())  pass++; else fail++;
  if (TestFilterConfigService()) pass++; else fail++;

  std::cout << "\n****************************************************************\n";
  std::cout << "  Storage Services: Passed: " << pass << ", Failed: " << fail << "\n";
  std::cout << "  " << (fail == 0 ? "[ALL TESTS PASSED]" : "[SOME TESTS FAILED]") << "\n";
  std::cout << "****************************************************************\n";
  return fail == 0 ? 0 : 1;
}

} // namespace test_storage_services
