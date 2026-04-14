# ── DspCore tests ──────────────────────────────────────────────────
# Все тесты через all_test.hpp — паттерн GPUWorkLib
# GPU нужен только на Linux/Debian (ROCm). На Windows только конфигурируем.

# ── HSA runtime: нужен test_core_main (через test_zero_copy.hpp)
#    и test_zerocopy_rdna4. find_library — портабельно через CMAKE_PREFIX_PATH.
# Только на Linux/ROCm — на Windows нет hsa-runtime64, тесты не линкуются с HSA.
if(UNIX)
  find_library(HSA_RUNTIME_LIB
    NAMES hsa-runtime64
    PATH_SUFFIXES lib lib64
    REQUIRED
  )
  message(STATUS "[DspCore tests] HSA runtime: ${HSA_RUNTIME_LIB}")
endif()

# ── Основной тест-раннер (all_test.hpp) ────────────────────────────
add_executable(test_core_main
  main.cpp
)
target_link_libraries(test_core_main PRIVATE
  DspCore::DspCore
  DspCore::TestUtils
  $<$<BOOL:${HSA_RUNTIME_LIB}>:${HSA_RUNTIME_LIB}>
)
target_include_directories(test_core_main PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(test_core_main PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>
)

# ── Standalone утилита: диагностика Zero-Copy на RDNA4 ─────────────
# Тестирует 4 механизма стыковки данных OpenCL → ROCm/HIP.
add_executable(test_zerocopy_rdna4
  test_zerocopy_rdna4.cpp
)
target_link_libraries(test_zerocopy_rdna4 PRIVATE
  DspCore::DspCore
  $<$<BOOL:${HSA_RUNTIME_LIB}>:${HSA_RUNTIME_LIB}>
)
