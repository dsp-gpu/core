/**
 * @file batch_manager.cpp
 * @brief Реализация BatchManager — методы, зависящие от IBackend
 *
 * ============================================================================
 * РАЗДЕЛЕНИЕ:
 *   Заголовок (batch_manager.hpp): inline-методы без зависимости от IBackend
 *   Этот файл: методы, запрашивающие у IBackend информацию о памяти GPU
 *
 * Избегаем циклических #include между services/ и common/.
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <core/services/batch_manager.hpp>
#include <core/interface/i_backend.hpp>
#include <core/logger/logger.hpp>

#include <algorithm>
#include <iostream>

namespace drv_gpu_lib {

// ============================================================================
// Методы, зависящие от памяти
// ============================================================================

/**
 * @brief Запрашивает текущую свободную память GPU через IBackend
 *
 * Делегирует в backend->GetFreeMemorySize() — использует вендорные расширения:
 * - NVIDIA: CL_DEVICE_GLOBAL_FREE_MEMORY_NV
 * - AMD:    CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
 *
 * Если backend не инициализирован или расширение не поддерживается → 0.
 * Caller обрабатывает 0 как «unknown» и применяет fallback (22% total_items).
 *
 * @param backend IBackend конкретного GPU. nullptr → return 0.
 * @return Свободная память в байтах (0 если неизвестна)
 */
size_t BatchManager::GetAvailableMemory(IBackend* backend) {
    if (!backend || !backend->IsInitialized()) {
        return 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Получить РЕАЛЬНО СВОБОДНУЮ память GPU (через расширения вендоров)
    // NVIDIA: CL_DEVICE_GLOBAL_FREE_MEMORY_NV
    // AMD: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
    // Fallback: GetGlobalMemorySize() * 0.9
    // ═══════════════════════════════════════════════════════════════════════
    return backend->GetFreeMemorySize();
}

/**
 * @brief Вычисляет оптимальный размер batch по свободной VRAM GPU
 *
 * Алгоритм:
 * 1. GetAvailableMemory(backend) — запросить свободную VRAM
 * 2. Если 0 → fallback: 22% от total_items (консервативно, ≈1/4 total, чтобы не OOM)
 * 3. Вычесть external_memory_bytes (буферы вызывающего кода уже занимают VRAM)
 * 4. CalculateBatchSizeFromMemory() → batch = (available × limit) / item_bytes
 *
 * memory_limit — защитный коэффициент (обычно 0.7 = 70% VRAM).
 * Без него batch мог бы потребовать 100% VRAM → OOM при аллокации.
 *
 * @param backend               IBackend GPU (nullptr → return total_items)
 * @param total_items           Общее число элементов для обработки
 * @param item_memory_bytes     Память на один элемент (байт)
 * @param memory_limit          Доля VRAM для использования (0.7 = 70%)
 * @param external_memory_bytes Память, уже занятая вызывающим кодом (байт)
 * @return Рекомендуемый размер batch (1 ≤ result ≤ total_items)
 */
size_t BatchManager::CalculateOptimalBatchSize(
    IBackend* backend,
    size_t total_items,
    size_t item_memory_bytes,
    double memory_limit,
    size_t external_memory_bytes)
{
    if (!backend || total_items == 0 || item_memory_bytes == 0) {
        return total_items;
    }

    // Получить доступную память
    size_t available = GetAvailableMemory(backend);

    if (available == 0) {
        // Запасной вариант: 22% элементов — условное число, когда не можем спросить GPU.
        // Примерно 1/4 от total: консервативно, чтобы не получить OOM при первом запуске.
        size_t fallback = std::max(
            static_cast<size_t>(total_items * 0.22),
            static_cast<size_t>(1));
        DRVGPU_LOG_WARNING("BatchManager",
            "Cannot query GPU memory, using fallback batch size: " + std::to_string(fallback));
        return fallback;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Вычитаем память уже занятую внешними данными (напр., буфер от генератора)
    // ═══════════════════════════════════════════════════════════════════════════
    if (external_memory_bytes > 0) {
        if (external_memory_bytes >= available) {
            // Внешние данные занимают почти всю память — используем минимальный batch
            DRVGPU_LOG_WARNING("BatchManager",
                "External memory (" + std::to_string(external_memory_bytes / 1024 / 1024) +
                " MB) >= available (" + std::to_string(available / 1024 / 1024) +
                " MB), using batch=1");
            return 1;
        }
        available -= external_memory_bytes;
    }

    // Расчёт через inline-вспомогательную функцию
    size_t batch_size = CalculateBatchSizeFromMemory(
        available, total_items, item_memory_bytes, memory_limit);

    return batch_size;
}

/**
 * @brief Проверяет, уместятся ли ВСЕ элементы в одном batch
 *
 * Удобный предикат: если true — батчинг не нужен, можно обработать всё за раз.
 * Использует тот же memory_limit что и CalculateOptimalBatchSize — оценки согласованы.
 *
 * @param backend           IBackend GPU (nullptr → return true)
 * @param total_items       Число элементов
 * @param item_memory_bytes Память на один элемент (байт)
 * @param memory_limit      Доля VRAM для использования (0.7 = 70%)
 * @return true если все элементы умещаются, false если нужен batching
 */
bool BatchManager::AllItemsFit(
    IBackend* backend,
    size_t total_items,
    size_t item_memory_bytes,
    double memory_limit)
{
    if (!backend || total_items == 0) {
        return true;
    }

    size_t available = GetAvailableMemory(backend);
    size_t usable = static_cast<size_t>(
        static_cast<double>(available) * memory_limit);
    size_t required = total_items * item_memory_bytes;

    return required <= usable;
}

} // namespace drv_gpu_lib
