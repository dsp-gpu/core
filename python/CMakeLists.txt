# ── DspCore Python bindings ────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(dsp_core dsp_core_module.cpp)
target_link_libraries(dsp_core PRIVATE DspCore::DspCore)
target_include_directories(dsp_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(dsp_core PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
install(TARGETS dsp_core DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
