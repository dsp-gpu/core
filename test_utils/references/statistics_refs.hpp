#pragma once

/// @file references/statistics_refs.hpp
/// @brief CPU-эталоны статистики: CpuMean, CpuMedian, CpuVariance, CpuStd.
/// Зеркало Python: common/references/statistics_refs.py

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace gpu_test_utils {
namespace refs {

/// Среднее (complex<float>)
template<typename T>
inline T CpuMean(const T* data, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i)
    sum += static_cast<double>(data[i]);
  return static_cast<T>(sum / static_cast<double>(n));
}

/// Специализация для complex<float>
template<>
inline std::complex<float> CpuMean(const std::complex<float>* data, size_t n) {
  double sum_r = 0.0, sum_i = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_r += data[i].real();
    sum_i += data[i].imag();
  }
  auto dn = static_cast<double>(n);
  return {static_cast<float>(sum_r / dn), static_cast<float>(sum_i / dn)};
}

/// Среднее по амплитуде |x|
inline float CpuMeanMagnitude(const std::complex<float>* data, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i)
    sum += static_cast<double>(std::abs(data[i]));
  return static_cast<float>(sum / static_cast<double>(n));
}

/// Медиана по амплитуде |x|
inline float CpuMedianMagnitude(const std::complex<float>* data, size_t n) {
  std::vector<float> mags(n);
  for (size_t i = 0; i < n; ++i)
    mags[i] = std::abs(data[i]);
  std::sort(mags.begin(), mags.end());
  if (n % 2 == 0)
    return (mags[n / 2 - 1] + mags[n / 2]) / 2.0f;
  return mags[n / 2];
}

/// Дисперсия по амплитуде (population variance, ddof=0)
inline float CpuVarianceMagnitude(const std::complex<float>* data, size_t n) {
  float mean = CpuMeanMagnitude(data, n);
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(std::abs(data[i])) - mean;
    sum_sq += diff * diff;
  }
  return static_cast<float>(sum_sq / static_cast<double>(n));
}

/// Стандартное отклонение по амплитуде
inline float CpuStdMagnitude(const std::complex<float>* data, size_t n) {
  return std::sqrt(CpuVarianceMagnitude(data, n));
}

/// Среднее для float массива
inline float CpuMeanFloat(const float* data, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i)
    sum += data[i];
  return static_cast<float>(sum / static_cast<double>(n));
}

/// Медиана для float массива
inline float CpuMedianFloat(const float* data, size_t n) {
  std::vector<float> sorted(data, data + n);
  std::sort(sorted.begin(), sorted.end());
  if (n % 2 == 0)
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0f;
  return sorted[n / 2];
}

/// Дисперсия для float массива (population, ddof=0)
inline float CpuVarianceFloat(const float* data, size_t n) {
  float mean = CpuMeanFloat(data, n);
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = static_cast<double>(data[i]) - mean;
    sum_sq += d * d;
  }
  return static_cast<float>(sum_sq / static_cast<double>(n));
}

/// Стандартное отклонение для float массива
inline float CpuStdFloat(const float* data, size_t n) {
  return std::sqrt(CpuVarianceFloat(data, n));
}

} // namespace refs
} // namespace gpu_test_utils
