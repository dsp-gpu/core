#pragma once

/// @file references/signal_refs.hpp
/// @brief CPU-эталоны сигналов: CW, LFM, FormSignal, Noise, MultiBeam и др.
/// Зеркало Python: common/references/signal_refs.py (SignalReferences)

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <algorithm>

namespace gpu_test_utils {
namespace refs {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// CW сигнал (непрерывная синусоида).
inline std::vector<std::complex<float>>
GenerateCw(float fs, size_t n_samples, float f0,
           float amplitude = 1.0f, float phase = 0.0f)
{
  std::vector<std::complex<float>> sig(n_samples);
  for (size_t i = 0; i < n_samples; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(fs);
    double ph = 2.0 * M_PI * static_cast<double>(f0) * t + static_cast<double>(phase);
    sig[i] = std::complex<float>(
        amplitude * static_cast<float>(std::cos(ph)),
        amplitude * static_cast<float>(std::sin(ph)));
  }
  return sig;
}

/// ЛЧМ сигнал (линейная частотная модуляция).
inline std::vector<std::complex<float>>
GenerateLfm(float fs, size_t n_samples, float f_start, float f_end,
            float amplitude = 1.0f, float phase = 0.0f)
{
  std::vector<std::complex<float>> sig(n_samples);
  double duration = static_cast<double>(n_samples) / static_cast<double>(fs);
  double rate = (static_cast<double>(f_end) - f_start) / duration;
  for (size_t i = 0; i < n_samples; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(fs);
    double ph = 2.0 * M_PI * (f_start * t + 0.5 * rate * t * t) + phase;
    sig[i] = std::complex<float>(
        amplitude * static_cast<float>(std::cos(ph)),
        amplitude * static_cast<float>(std::sin(ph)));
  }
  return sig;
}

/// ЛЧМ с задержкой (для тестов гетеродина/дечирпа). Нули до delay_s.
inline std::vector<std::complex<float>>
GenerateDelayedLfm(float fs, size_t n_samples, float f_start, float f_end,
                   float delay_s, float amplitude = 1.0f)
{
  std::vector<std::complex<float>> sig(n_samples, {0.0f, 0.0f});
  double duration = static_cast<double>(n_samples) / static_cast<double>(fs);
  double rate = (static_cast<double>(f_end) - f_start) / duration;
  for (size_t i = 0; i < n_samples; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(fs);
    if (t < delay_s) continue;
    double t_local = t - delay_s;
    double ph = 2.0 * M_PI * (f_start * t_local + 0.5 * rate * t_local * t_local);
    sig[i] = std::complex<float>(
        amplitude * static_cast<float>(std::cos(ph)),
        amplitude * static_cast<float>(std::sin(ph)));
  }
  return sig;
}

/// Несколько ЛЧМ с разными задержками (массив антенн). Row-major.
inline std::vector<std::complex<float>>
GenerateMultiAntennaLfm(float fs, size_t n_samples, float f_start, float f_end,
                        const std::vector<float>& delays_s, float amplitude = 1.0f)
{
  size_t n_ant = delays_s.size();
  std::vector<std::complex<float>> result(n_ant * n_samples, {0.0f, 0.0f});
  for (size_t a = 0; a < n_ant; ++a) {
    auto row = GenerateDelayedLfm(fs, n_samples, f_start, f_end, delays_s[a], amplitude);
    std::copy(row.begin(), row.end(), result.begin() + static_cast<ptrdiff_t>(a * n_samples));
  }
  return result;
}

/// FormSignal CPU reference (getX без шума). Воспроизводит GPU FormSignalGenerator.
inline std::vector<std::complex<float>>
GenerateFormSignal(float fs, size_t points, float f0, float amplitude,
                   float phase, float fdev, float norm_val, float tau = 0.0f)
{
  double dt = 1.0 / static_cast<double>(fs);
  double ti = static_cast<double>(points) * dt;
  std::vector<std::complex<float>> result(points, {0.0f, 0.0f});
  for (size_t i = 0; i < points; ++i) {
    double t = static_cast<double>(i) * dt + static_cast<double>(tau);
    if (t < 0.0 || t > ti - dt) continue;
    double t_centered = t - ti / 2.0;
    double ph = 2.0 * M_PI * f0 * t + M_PI * fdev / ti * (t_centered * t_centered) + phase;
    float a = amplitude * norm_val;
    result[i] = std::complex<float>(a * static_cast<float>(std::cos(ph)),
                                    a * static_cast<float>(std::sin(ph)));
  }
  return result;
}

/// Синусоида — алиас GenerateCw для statistics-тестов.
inline std::vector<std::complex<float>>
GenerateSinusoid(float freq, float sample_rate, size_t n_point, float amplitude = 1.0f) {
  return GenerateCw(sample_rate, n_point, freq, amplitude);
}

/// Многоканальный сигнал (multi-beam). amp = amp_base + i * amp_step. Row-major.
inline std::vector<std::complex<float>>
GenerateMultiBeam(size_t n_beams, size_t n_point, float fs, float freq,
                  float amp_base = 1.0f, float amp_step = 0.5f)
{
  std::vector<std::complex<float>> result(n_beams * n_point);
  for (size_t b = 0; b < n_beams; ++b) {
    float amp = amp_base + static_cast<float>(b) * amp_step;
    auto beam = GenerateCw(fs, n_point, freq, amp);
    std::copy(beam.begin(), beam.end(),
              result.begin() + static_cast<ptrdiff_t>(b * n_point));
  }
  return result;
}

/// Постоянное значение (edge case тесты).
inline std::vector<std::complex<float>>
GenerateConstant(std::complex<float> value, size_t n_point) {
  return std::vector<std::complex<float>>(n_point, value);
}

/// Гауссов шум (CPU reference, воспроизводимый через seed).
inline std::vector<std::complex<float>>
GenerateNoise(size_t n_samples, float amplitude = 1.0f, uint32_t seed = 42)
{
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::complex<float>> sig(n_samples);
  float scale = amplitude / std::sqrt(2.0f);
  for (size_t i = 0; i < n_samples; ++i)
    sig[i] = std::complex<float>(dist(gen) * scale, dist(gen) * scale);
  return sig;
}

/// Дечирп: s_dc = s_rx * conj(s_ref). CPU-эталон.
inline std::vector<std::complex<float>>
CpuDechirp(const std::complex<float>* s_rx,
           const std::complex<float>* s_ref, size_t n)
{
  std::vector<std::complex<float>> result(n);
  for (size_t i = 0; i < n; ++i)
    result[i] = s_rx[i] * std::conj(s_ref[i]);
  return result;
}

/// Тестовый сигнал для фильтров (CW f_low + CW f_high).
inline std::vector<std::complex<float>>
GenerateComposite(float fs, size_t n_samples,
                  float f_low, float f_high,
                  float amp_low = 1.0f, float amp_high = 0.5f)
{
  auto s1 = GenerateCw(fs, n_samples, f_low, amp_low);
  auto s2 = GenerateCw(fs, n_samples, f_high, amp_high);
  for (size_t i = 0; i < n_samples; ++i)
    s1[i] += s2[i];
  return s1;
}

} // namespace refs
} // namespace gpu_test_utils
