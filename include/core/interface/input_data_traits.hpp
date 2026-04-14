#pragma once

/**
 * @file input_data_traits.hpp
 * @brief Type traits для диспетчеризации по типу входных данных
 *
 * Используется в Process/Execute для выбора CPU/GPU/SVM пути.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <type_traits>
#include <complex>
#include <vector>

namespace drv_gpu_lib {

/// CPU вектор комплексных float
template<typename T>
struct is_cpu_vector : std::false_type {};

template<>
struct is_cpu_vector<std::vector<std::complex<float>>> : std::true_type {};

template<typename T>
inline constexpr bool is_cpu_vector_v = is_cpu_vector<T>::value;

/// SVM указатель (void*)
template<typename T>
struct is_svm_pointer : std::false_type {};

template<>
struct is_svm_pointer<void*> : std::true_type {};

template<typename T>
inline constexpr bool is_svm_pointer_v = is_svm_pointer<T>::value;

/// OpenCL буфер (cl_mem) — только при наличии OpenCL заголовков
#ifdef CL_VERSION_1_0
template<typename T>
struct is_cl_mem : std::false_type {};

template<>
struct is_cl_mem<cl_mem> : std::true_type {};

template<typename T>
inline constexpr bool is_cl_mem_v = is_cl_mem<T>::value;
#endif  // CL_VERSION_1_0

}  // namespace drv_gpu_lib
