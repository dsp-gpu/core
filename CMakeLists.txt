cmake_minimum_required(VERSION 3.25)
project(DspCore VERSION 0.1.0
  DESCRIPTION "GPU driver, profiler and backend abstraction"
  LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_CORE_BUILD_TESTS    "Build tests"            ON)
option(DSP_CORE_BUILD_EXAMPLES "Build examples"         OFF)
option(DSP_CORE_BUILD_PYTHON   "Build Python bindings"  OFF)

# ── ROCm/HIP + OpenCL ──────────────────────────────────────────────
find_package(hip    REQUIRED)   # lowercase — Linux case-sensitive!
find_package(OpenCL REQUIRED)

# ── Основная библиотека ────────────────────────────────────────────
add_library(DspCore STATIC)
add_library(DspCore::DspCore ALIAS DspCore)

target_sources(DspCore PRIVATE
  # Core
  src/drv_gpu.cpp
  src/gpu_context.cpp
  src/gpu_kernel_op.cpp
  src/module_registry.cpp
  # Config
  src/config/gpu_config.cpp
  # Backends — OpenCL
  src/backends/opencl/opencl_backend.cpp
  src/backends/opencl/opencl_core.cpp
  src/backends/opencl/opencl_profiling.cpp
  src/backends/opencl/command_queue_pool.cpp
  # Backends — ROCm
  src/backends/rocm/rocm_backend.cpp
  src/backends/rocm/rocm_core.cpp
  src/backends/rocm/stream_pool.cpp
  src/backends/rocm/zero_copy_bridge.cpp
  # Backends — Hybrid
  src/backends/hybrid/hybrid_backend.cpp
  # Logger
  src/logger/logger.cpp
  src/logger/default_logger.cpp
  src/logger/config_logger.cpp
  # Memory
  src/memory/memory_manager.cpp
  # Services
  src/services/batch_manager.cpp
  src/services/kernel_cache_service.cpp
  src/services/filter_config_service.cpp
  src/services/storage/file_storage_backend.cpp
)

target_include_directories(DspCore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # Позволяет: #include "services/gpu_profiler.hpp" вместо "dsp/services/..."
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/dsp>
    # Сгенерированные файлы версии (version.h)
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/dsp>
  PRIVATE
    kernels/
    src
    src/config
)
target_link_libraries(DspCore PUBLIC hip::host OpenCL::OpenCL)
target_compile_features(DspCore PUBLIC cxx_std_17)

# ── TestUtils — ОТДЕЛЬНАЯ target (не протекает в публичный API) ────
add_library(DspCoreTestUtils INTERFACE)
add_library(DspCore::TestUtils ALIAS DspCoreTestUtils)

target_include_directories(DspCoreTestUtils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test_utils/include>
  $<INSTALL_INTERFACE:include/dsp/test>
)
target_link_libraries(DspCoreTestUtils INTERFACE DspCore::DspCore)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS DspCore DspCoreTestUtils
  EXPORT DspCoreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY test_utils/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dsp/test
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DspCore
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.json"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/DspCore
)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspCoreConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspCore
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(EXPORT DspCoreTargets NAMESPACE DspCore::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspCore
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspCore
)

# ── Git Version ────────────────────────────────────────────────────
find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE git)
endif()

add_custom_target(git_version_DspCore ALL
    COMMAND ${CMAKE_COMMAND}
        -D SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -D BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
        -D MODULE_PREFIX=DSPCORE
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake
    COMMENT "[DspCore] Checking git version..."
    VERBATIM
)
add_dependencies(DspCore git_version_DspCore)

# ── Конфигурационные файлы ─────────────────────────────────────────
# configGPU.json копируется в build/ — тесты ищут его рядом с exe
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/configGPU.json
  ${CMAKE_CURRENT_BINARY_DIR}/configGPU.json
  COPYONLY
)

# ── Тесты ──────────────────────────────────────────────────────────
if(DSP_CORE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ── Python bindings ────────────────────────────────────────────────
if(DSP_CORE_BUILD_PYTHON)
  add_subdirectory(python)
endif()
