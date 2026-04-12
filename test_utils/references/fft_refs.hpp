#pragma once

/// @file references/fft_refs.hpp
/// @brief CPU-эталоны FFT: FindPeakBin, PeakFreqHz, CpuMagnitude, FreqAxis.
/// Зеркало Python: common/references/fft_refs.py

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace gpu_test_utils {
namespace refs {

/// Найти индекс максимального бина (первая половина спектра).
inline size_t FindPeakBin(const float* magnitude, size_t n_bins,
                          size_t search_range = 0)
{
  size_t range = (search_range > 0) ? search_range : n_bins / 2;
  range = std::min(range, n_bins);
  size_t peak = 0;
  float peak_val = 0.0f;
  for (size_t i = 0; i < range; ++i) {
    if (magnitude[i] > peak_val) {
      peak_val = magnitude[i];
      peak = i;
    }
  }
  return peak;
}

/// FindPeakBin из complex спектра (автоматический |x|).
inline size_t FindPeakBinComplex(const std::complex<float>* spectrum,
                                 size_t n_bins, size_t search_range = 0)
{
  std::vector<float> mag(n_bins);
  for (size_t i = 0; i < n_bins; ++i)
    mag[i] = std::abs(spectrum[i]);
  return FindPeakBin(mag.data(), n_bins, search_range);
}

/// Частота пика спектра (Гц).
inline float PeakFreqHz(const float* magnitude, size_t n_bins, float fs,
                        size_t search_range = 0)
{
  size_t peak = FindPeakBin(magnitude, n_bins, search_range);
  return static_cast<float>(peak) * fs / static_cast<float>(n_bins);
}

/// |x| — амплитудный спектр из complex.
inline std::vector<float>
CpuMagnitude(const std::complex<float>* spectrum, size_t n_bins)
{
  std::vector<float> mag(n_bins);
  for (size_t i = 0; i < n_bins; ++i)
    mag[i] = std::abs(spectrum[i]);
  return mag;
}

/// Ось частот (Гц) — аналог np.fft.fftfreq.
inline std::vector<float> FreqAxis(size_t n_fft, float fs) {
  std::vector<float> freqs(n_fft);
  float df = fs / static_cast<float>(n_fft);
  for (size_t i = 0; i < n_fft; ++i) {
    if (i <= n_fft / 2)
      freqs[i] = static_cast<float>(i) * df;
    else
      freqs[i] = static_cast<float>(static_cast<int>(i) - static_cast<int>(n_fft)) * df;
  }
  return freqs;
}

} // namespace refs
} // namespace gpu_test_utils
