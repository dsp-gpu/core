#pragma once

/// @file test_configs.hpp
/// @brief Tolerance константы + параметры сигналов/фильтров/дечирпа.
/// Зеркало Python: common/configs.py

#include <string>
#include <cstdint>
#include <cmath>

namespace gpu_test_utils {

// ══════════════════════════════════════════════════════════════════
// Централизованные Tolerance'ы
// ══════════════════════════════════════════════════════════════════

namespace tolerance {
  /// complex<float> GPU vs CPU (основной, 90% тестов)
  constexpr double kComplex32  = 1e-3;

  /// float32 statistics (mean, std, variance)
  constexpr double kStatistics = 1e-3;

  /// float64 / double precision
  constexpr double kDouble     = 1e-5;

  /// Частота пика FFT (Гц)
  constexpr double kFreqHz     = 5000.0;

  /// Мощность сигнала (относительная)
  constexpr double kPower      = 0.05;

  /// Строгое сравнение (для точных операций: copy, transpose)
  constexpr double kExact      = 1e-7;

  /// FIR/IIR фильтры (transient допуск больше)
  constexpr double kFilter     = 5e-3;
}

// ══════════════════════════════════════════════════════════════════
// SignalParams (аналог Python SignalConfig)
// ══════════════════════════════════════════════════════════════════

struct SignalParams {
  float    fs        = 12e6f;     ///< Частота дискретизации (Гц)
  size_t   n_samples = 4096;      ///< Число отсчётов
  float    f0_hz     = 2e6f;      ///< Несущая частота (Гц)
  float    fdev_hz   = 0.0f;      ///< Девиация (для ЛЧМ)
  float    amplitude = 1.0f;      ///< Амплитуда
  uint32_t seed      = 42;        ///< Seed для PRNG

  float duration_s() const { return static_cast<float>(n_samples) / fs; }
  float freq_resolution_hz(size_t nfft = 0) const {
    return fs / static_cast<float>(nfft > 0 ? nfft : n_samples);
  }
  float nyquist_hz() const { return fs / 2.0f; }
};

// ══════════════════════════════════════════════════════════════════
// FilterParams (аналог Python FilterConfig)
// ══════════════════════════════════════════════════════════════════

struct FilterParams {
  std::string filter_type = "fir";
  float       cutoff_hz   = 1e3f;
  float       fs          = 12e6f;
  int         order       = 4;
  int         n_taps      = 64;
  std::string window      = "hamming";

  float normalized_cutoff() const { return cutoff_hz / (fs / 2.0f); }
};

// ══════════════════════════════════════════════════════════════════
// DechirpParams (аналог Python HeterodyneConfig)
// ══════════════════════════════════════════════════════════════════

struct DechirpParams {
  float    fs         = 12e6f;
  float    f_start    = 0.0f;
  float    f_end      = 2e6f;
  size_t   n_samples  = 8000;
  int      n_antennas = 5;

  /// Скорость света — точное значение, double для precision (review R5)
  static constexpr double kSpeedOfLight = 299792458.0;
  double   c_light    = kSpeedOfLight;

  float bandwidth() const { return f_end - f_start; }
  float duration_s() const { return static_cast<float>(n_samples) / fs; }
  float chirp_rate() const { return bandwidth() / duration_s(); }
  double range_from_delay(double delay_s) const { return c_light * delay_s / 2.0; }
  float fbeat_from_delay(float delay_s) const { return chirp_rate() * delay_s; }
};

} // namespace gpu_test_utils
