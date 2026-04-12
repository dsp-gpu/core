#pragma once

/// @file validators/numeric.hpp
/// @brief MaxRelError, AbsError, RmseError, ScalarRelError, ScalarAbsError.
/// Зеркало Python: common/validators.py (DataValidator)
/// Все вычисления в double, данные = любой T (review R2).

#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include "../test_result.hpp"

namespace gpu_test_utils {

// ── value_to_string: работает с float, double, complex (review R3) ──

namespace detail {

template<typename T>
inline std::string value_to_string(const T& v) {
  return std::to_string(v);
}

template<typename T>
inline std::string value_to_string(const std::complex<T>& v) {
  return "(" + std::to_string(v.real()) + "," + std::to_string(v.imag()) + ")";
}

} // namespace detail

// ── Шаблонные хелперы для |a - b| ──────────────────────────────

template<typename T>
inline double AbsDiff(const T& a, const T& b) {
  return static_cast<double>(std::abs(a - b));
}

template<typename T>
inline double AbsDiff(const std::complex<T>& a, const std::complex<T>& b) {
  return static_cast<double>(std::abs(a - b));
}

template<typename T>
inline double AbsVal(const T& a) {
  return static_cast<double>(std::abs(a));
}

template<typename T>
inline double AbsVal(const std::complex<T>& a) {
  return static_cast<double>(std::abs(a));
}

// ══════════════════════════════════════════════════════════════════
// FREE FUNCTIONS — основной API (1 строка вместо 4)
// ══════════════════════════════════════════════════════════════════

/// max|actual - ref| / max|ref| < tolerance
/// Основная метрика GPU vs CPU. Strict `<` (Python совместимость).
template<typename T>
inline ValidationResult MaxRelError(
    const T* actual, const T* reference, size_t count,
    double tolerance, const std::string& name = "max_rel")
{
  double max_diff = 0.0;
  double max_ref  = 0.0;
  for (size_t i = 0; i < count; ++i) {
    max_diff = std::max(max_diff, AbsDiff(actual[i], reference[i]));
    max_ref  = std::max(max_ref,  AbsVal(reference[i]));
  }
  if (max_ref < 1e-15) {
    return {max_diff < 1e-10, name, max_diff, 1e-10, "(near-zero reference)"};
  }
  double err = max_diff / max_ref;
  return {err < tolerance, name, err, tolerance, ""};
}

/// max|actual - ref| < tolerance
/// Для абсолютных величин: частоты в Гц, индексы бинов.
template<typename T>
inline ValidationResult AbsError(
    const T* actual, const T* reference, size_t count,
    double tolerance, const std::string& name = "abs")
{
  double max_diff = 0.0;
  for (size_t i = 0; i < count; ++i)
    max_diff = std::max(max_diff, AbsDiff(actual[i], reference[i]));
  return {max_diff < tolerance, name, max_diff, tolerance, ""};
}

/// rms(|actual - ref|) / rms(|ref|) < tolerance
/// Для шумных данных, фильтров.
template<typename T>
inline ValidationResult RmseError(
    const T* actual, const T* reference, size_t count,
    double tolerance, const std::string& name = "rmse")
{
  double sum_sq_diff = 0.0;
  double sum_sq_ref  = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double d = AbsDiff(actual[i], reference[i]);
    double r = AbsVal(reference[i]);
    sum_sq_diff += d * d;
    sum_sq_ref  += r * r;
  }
  double rms_diff = std::sqrt(sum_sq_diff / static_cast<double>(count));
  double rms_ref  = std::sqrt(sum_sq_ref  / static_cast<double>(count));
  if (rms_ref < 1e-15) {
    return {rms_diff < 1e-10, name, rms_diff, 1e-10, "(near-zero reference)"};
  }
  double err = rms_diff / rms_ref;
  return {err < tolerance, name, err, tolerance, ""};
}

/// Скалярная проверка: |actual - expected| / |expected| < tolerance
template<typename T>
inline ValidationResult ScalarRelError(
    T actual, T expected, double tolerance,
    const std::string& name = "scalar_rel")
{
  double diff = AbsDiff(actual, expected);
  double ref  = AbsVal(expected);
  if (ref < 1e-15) {
    return {diff < 1e-10, name, diff, 1e-10, ""};
  }
  double err = diff / ref;
  return {err < tolerance, name, err, tolerance,
          "actual=" + detail::value_to_string(actual) +
          " expected=" + detail::value_to_string(expected)};
}

/// Скалярная проверка: |actual - expected| < tolerance (абсолютная)
template<typename T>
inline ValidationResult ScalarAbsError(
    T actual, T expected, double tolerance,
    const std::string& name = "scalar_abs")
{
  double err = AbsDiff(actual, expected);
  return {err < tolerance, name, err, tolerance,
          "actual=" + detail::value_to_string(actual) +
          " expected=" + detail::value_to_string(expected)};
}

} // namespace gpu_test_utils
