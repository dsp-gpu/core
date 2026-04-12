# ── DspCore tests ──────────────────────────────────────────────────
# Все тесты через all_test.hpp — паттерн GPUWorkLib
# GPU нужен только на Linux/Debian (ROCm). На Windows только конфигурируем.

# ── Основной тест-раннер (all_test.hpp) ────────────────────────────
add_executable(test_core_main
  main.cpp
)
target_link_libraries(test_core_main PRIVATE
  DspCore::DspCore
  DspCore::TestUtils
)
target_include_directories(test_core_main PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(test_core_main PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>
)

# ── Standalone утилиты (имеют собственный main()) ──────────────────
add_executable(test_clmem_probe
  test_clmem_gpu_va_probe.cpp
)
target_link_libraries(test_clmem_probe PRIVATE
  DspCore::DspCore
)

add_executable(test_zerocopy_rdna4
  test_zerocopy_rdna4.cpp
)
target_link_libraries(test_zerocopy_rdna4 PRIVATE
  DspCore::DspCore
)
