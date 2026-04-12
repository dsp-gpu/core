cmake_minimum_required(VERSION 3.25)
project(DspCore VERSION 0.1.0 LANGUAGES CXX HIP)

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
  # Phase 1 skeleton — sources added in Phase 2
)

target_include_directories(DspCore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    kernels/
    src
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

# ── Тесты ──────────────────────────────────────────────────────────
if(DSP_CORE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ── Python bindings ────────────────────────────────────────────────
if(DSP_CORE_BUILD_PYTHON)
  add_subdirectory(python)
endif()
