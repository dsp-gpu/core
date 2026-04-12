#pragma once

/**
 * @file input_data.hpp
 * @brief Универсальный шаблон входных данных для всех модулей GPUWorkLib
 *
 * Используется в:
 * - fft_maxima (SpectrumMaximaFinder)
 * - fft_processor (FFTProcessor)
 * - statistics (будущий)
 * - heterodyne (будущий)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <complex>
#include <cstdint>

namespace drv_gpu_lib {

/**
 * @struct InputData
 * @brief Универсальная структура входных данных + параметры обработки
 *
 * Используется в fft_maxima, fft_processor, statistics (будущий), heterodyne (будущий).
 *
 * @tparam T Тип данных: std::vector<std::complex<float>>, cl_mem, void* (SVM)
 */
template<typename T>
struct InputData {
    // Размеры и данные
    uint32_t antenna_count = 0;     ///< Количество антенн (лучей)
    uint32_t n_point = 0;          ///< Точек на антенну
    T data{};                       ///< Данные
    size_t gpu_memory_bytes = 0;    ///< Реальный размер GPU буфера (для cl_mem)

    // Параметры обработки
    uint32_t repeat_count = 2;      ///< nFFT = nextPow2(n_point) × repeat_count
    float sample_rate = 1000.0f;   ///< Частота дискретизации (Гц)
    uint32_t search_range = 0;     ///< Диапазон поиска (0 = авто)
    float memory_limit = 0.80f;     ///< Доля GPU памяти для batch (0.0-1.0)
    size_t max_maxima_per_beam = 1000;  ///< Макс. максимумов на луч (FindAllMaxima)

    size_t TotalPoints() const { return static_cast<size_t>(antenna_count) * n_point; }
    size_t SizeBytes() const { return TotalPoints() * sizeof(std::complex<float>); }
    size_t ActualGpuMemory() const { return (gpu_memory_bytes > 0) ? gpu_memory_bytes : SizeBytes(); }
};

/**
 * @struct ProcessingParams
 * @deprecated Используйте поля в InputData<T>
 */
struct ProcessingParams {
    uint32_t repeat_count = 2;
    float sample_rate = 1000.0f;
    uint32_t search_range = 0;
    float memory_limit = 0.80f;
};

}  // namespace drv_gpu_lib
