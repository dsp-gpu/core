#pragma once

/// @file validators/signal.hpp
/// @brief CheckPeakFreq, CheckPeakFreqComplex, CheckPower.
/// Зеркало Python: common/validators.py (signal-related checks)

#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
#include "../test_result.hpp"

namespace gpu_test_utils {

/// Проверяет что пик FFT-спектра на ожидаемой частоте.
inline ValidationResult CheckPeakFreq(
    const float* magnitude, size_t n_bins, float fs,
    float expected_hz, float tolerance_hz,
    const std::string& name = "peak_freq_hz")
{
  size_t half = n_bins / 2;
  size_t peak_bin = 0;
  float peak_val = 0.0f;
  for (size_t i = 0; i < half; ++i) {
    if (magnitude[i] > peak_val) {
      peak_val = magnitude[i];
      peak_bin = i;
    }
  }
  float actual_hz = static_cast<float>(peak_bin) * fs / static_cast<float>(n_bins);
  float err = std::abs(actual_hz - expected_hz);
  return {err < tolerance_hz, name,
          static_cast<double>(actual_hz),
          static_cast<double>(tolerance_hz),
          "expected=" + std::to_string(expected_hz) + "Hz err=" + std::to_string(err) + "Hz"};
}

/// CheckPeakFreq из complex спектра (автоматический |FFT|).
inline ValidationResult CheckPeakFreqComplex(
    const std::complex<float>* spectrum, size_t n_bins, float fs,
    float expected_hz, float tolerance_hz,
    const std::string& name = "peak_freq_hz")
{
  std::vector<float> mag(n_bins);
  for (size_t i = 0; i < n_bins; ++i)
    mag[i] = std::abs(spectrum[i]);
  return CheckPeakFreq(mag.data(), n_bins, fs, expected_hz, tolerance_hz, name);
}

/// Проверяет мощность сигнала: |mean(|x|^2) - expected| / expected < tolerance
inline ValidationResult CheckPower(
    const std::complex<float>* data, size_t n,
    float expected_power, float tolerance = 0.05f,
    const std::string& name = "power")
{
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float m = std::abs(data[i]);
    sum_sq += static_cast<double>(m) * m;
  }
  float actual = static_cast<float>(sum_sq / static_cast<double>(n));
  float rel_err = (expected_power > 1e-10f)
      ? std::abs(actual - expected_power) / expected_power
      : std::abs(actual);
  return {rel_err < tolerance, name,
          static_cast<double>(actual),
          static_cast<double>(tolerance),
          "expected=" + std::to_string(expected_power)};
}

} // namespace gpu_test_utils
