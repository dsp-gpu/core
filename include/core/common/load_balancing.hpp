#pragma once

/**
 * @file load_balancing_strategy.hpp
 * @brief Стратегии балансировки нагрузки для Multi-GPU
 */

namespace drv_gpu_lib {

/**
 * @enum LoadBalancingStrategy
 * @brief Стратегии распределения нагрузки между GPU
 */
enum class LoadBalancingStrategy {
    ROUND_ROBIN,      ///< Round-Robin (циклический выбор)
    LEAST_LOADED,     ///< Наименее загруженная GPU
    MANUAL,           ///< Ручной выбор (пользователь указывает индекс)
    FASTEST_FIRST     ///< Сначала самая быстрая GPU (по compute units)
};

/**
 * @brief Конвертировать стратегию в строку
 */
inline const char* LoadBalancingStrategyToString(LoadBalancingStrategy strategy) {
    switch (strategy) {
        case LoadBalancingStrategy::ROUND_ROBIN:   return "Round-Robin";
        case LoadBalancingStrategy::LEAST_LOADED:  return "Least Loaded";
        case LoadBalancingStrategy::MANUAL:        return "Manual";
        case LoadBalancingStrategy::FASTEST_FIRST: return "Fastest First";
        default:                                   return "Unknown";
    }
}

} // namespace drv_gpu_lib
