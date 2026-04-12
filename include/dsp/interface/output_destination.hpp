#pragma once

/**
 * @file output_destination.hpp
 * @brief Куда выводить результат обработки — CPU, GPU или оба
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

namespace drv_gpu_lib {

/**
 * @enum OutputDestination
 * @brief Направление вывода результатов
 */
enum class OutputDestination {
    CPU,   ///< Результат на хосте (clEnqueueReadBuffer)
    GPU,   ///< Результат в GPU буфере (cl_mem)
    ALL    ///< Оба варианта
};

}  // namespace drv_gpu_lib
