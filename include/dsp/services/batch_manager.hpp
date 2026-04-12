#pragma once

/**
 * @file batch_manager.hpp
 * @brief BatchManager — универсальный менеджер пакетной обработки для модулей GPU
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Централизованный расчёт размера пакетов и генерация диапазонов пакетов.
 *   Вынесено из fft_maxima для использования во ВСЕХ модулях GPU.
 *
 * ВОЗМОЖНОСТИ:
 *   - Учитывает реальную доступную память GPU (не только общий объём)
 *   - Настраиваемый % доступной памяти (по умолчанию 70%)
 *   - Умное слияние хвоста: если в последнем пакете 1–3 элемента — объединить с предыдущим
 *   - Работает с любым IBackend (не привязан к OpenCL)
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   BatchManager manager;
 *
 *   // Расчёт оптимального размера пакета
 *   size_t per_item_memory = nFFT * sizeof(complex<float>) * 2 + maxima_size;
 *   size_t batch_size = manager.CalculateOptimalBatchSize(
 *       backend, total_beams, per_item_memory, 0.7);
 *
 *   // Генерация диапазонов пакетов (с умным слиянием хвоста)
 *   auto batches = manager.CreateBatches(total_beams, batch_size, 3, true);
 *
 *   for (auto& batch : batches) {
 *       ProcessBatch(input, batch.start, batch.count);
 *   }
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <vector>
#include <cstddef>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>

// Предварительное объявление (избегаем циклического #include)
namespace drv_gpu_lib {
    class IBackend;
}

namespace drv_gpu_lib {

// ============================================================================
// BatchRange — описание одного пакета элементов для обработки
// ============================================================================

/**
 * @struct BatchRange
 * @brief Диапазон элементов для одного пакета
 *
 * Используется модулями для итерации по пакетам:
 *   for (auto& batch : batches) {
 *       process(input_data, batch.start, batch.count);
 *   }
 */
struct BatchRange {
    /// Начальный индекс (с 0)
    size_t start = 0;

    /// Количество элементов в пакете
    size_t count = 0;

    /// Индекс пакета (с 0, по порядку)
    size_t batch_idx = 0;

    /// Флаг: пакет получен слиянием с коротким хвостом
    bool is_merged = false;
};

// ============================================================================
// BatchManager — универсальный менеджер пакетной обработки
// ============================================================================

/**
 * @class BatchManager
 * @brief Вычисляет оптимальные размеры пакетов и генерирует диапазоны
 *
 * Не синглтон — можно создавать отдельный экземпляр на модуль.
 * Между вызовами внутреннее состояние не хранится (чистые вычисления).
 */
class BatchManager {
public:
    // ========================================================================
    // Расчёт размера пакета
    // ========================================================================

    /**
     * @brief Рассчитать оптимальный размер пакета по доступной памяти GPU
     *
     * @param backend Указатель на IBackend (для запросов памяти)
     * @param total_items Общее число элементов для обработки (напр., лучей)
     * @param item_memory_bytes Память на один элемент на GPU (байты)
     *        Пример: nFFT * sizeof(complex<float>) * 2 + maxima_buffer
     * @param memory_limit Доля доступной памяти (0.0 — 1.0)
     *        По умолчанию: 0.7 (70% доступной памяти GPU)
     * @param external_memory_bytes Память уже занятая внешними данными (напр., буфер от генератора)
     *        По умолчанию: 0 (нет внешних данных)
     * @return Оптимальное число элементов в пакете
     *
     * АЛГОРИТМ:
     * 1. Запрос свободной памяти GPU (GetFreeMemorySize)
     * 2. Вычитание external_memory_bytes (уже занятая память)
     * 3. available = (free - external) * memory_limit
     * 4. batch_size = available / item_memory_bytes
     * 5. Ограничение диапазоном [1, total_items]
     *
     * Если все элементы помещаются в память, возвращается total_items (пакетирование не нужно).
     */
    static size_t CalculateOptimalBatchSize(
        IBackend* backend,
        size_t total_items,
        size_t item_memory_bytes,
        double memory_limit = 0.7,
        size_t external_memory_bytes = 0);

    /**
     * @brief Рассчитать размер пакета по известной доступной памяти
     *
     * @param available_memory_bytes Доступная память GPU в байтах
     * @param total_items Общее число элементов
     * @param item_memory_bytes Память на один элемент
     * @param memory_limit Доля доступной памяти
     * @return Оптимальный размер пакета
     *
     * Использовать, когда доступная память уже известна
     * (напр., из MemoryManager::GetFreeMemory()).
     */
    static size_t CalculateBatchSizeFromMemory(
        size_t available_memory_bytes,
        size_t total_items,
        size_t item_memory_bytes,
        double memory_limit = 0.7);

    // ========================================================================
    // Генерация диапазонов пакетов
    // ========================================================================

    /**
     * @brief Создать список диапазонов пакетов с умным слиянием хвоста
     *
     * @param total_items Общее число элементов для обработки
     * @param items_per_batch Элементов в пакете (из CalculateOptimalBatchSize)
     * @param min_tail Минимум элементов в последнем пакете, чтобы оставить его отдельно
     *        Если в последнем пакете меньше — объединить с предыдущим.
     *        По умолчанию: 3 (если в последнем 1–3 элемента — слияние)
     * @param merge_small_tail Включить оптимизацию слияния хвоста
     *        По умолчанию: true
     * @return Вектор структур BatchRange
     *
     * ПРИМЕР СЛИЯНИЯ ХВОСТА:
     *   total=23, per_batch=10, min_tail=3
     *   БЕЗ слияния: [0-9], [10-19], [20-22]  (3 пакета, в последнем 3)
     *   С слиянием:  [0-9], [10-22]            (2 пакета, в последнем 13)
     *
     *   total=22, per_batch=10, min_tail=3
     *   БЕЗ слияния: [0-9], [10-19], [20-21]  (3 пакета, в последнем 2)
     *   С слиянием:  [0-9], [10-21]            (2 пакета, в последнем 12)
     *
     *   total=25, per_batch=10, min_tail=3
     *   БЕЗ слияния: [0-9], [10-19], [20-24]  (3 пакета, в последнем 5)
     *   С слиянием:  [0-9], [10-19], [20-24]  (3 пакета, без слияния — хвост >= min_tail+1)
     */
    static std::vector<BatchRange> CreateBatches(
        size_t total_items,
        size_t items_per_batch,
        size_t min_tail = 3,
        bool merge_small_tail = true);

    // ========================================================================
    // Запросы памяти
    // ========================================================================

    /**
     * @brief Получить оценку доступной памяти GPU
     * @param backend Указатель на IBackend
     * @return Доступная память в байтах (total * 0.9 — грубая оценка)
     *
     * ПРИМЕЧАНИЕ: Это оценка. OpenCL не даёт точный объём свободной памяти.
     * Используется: total_memory * 0.9 (предполагаем 10% занято ОС/драйвером).
     * Для точного контроля используйте MemoryManager::GetAllocatedSize().
     */
    static size_t GetAvailableMemory(IBackend* backend);

    /**
     * @brief Проверить, помещаются ли все элементы в память (пакетирование не нужно)
     * @param backend Указатель на IBackend
     * @param total_items Общее число элементов
     * @param item_memory_bytes Память на один элемент
     * @param memory_limit Доля используемой памяти
     * @return true если все помещаются, false если нужно пакетирование
     */
    static bool AllItemsFit(
        IBackend* backend,
        size_t total_items,
        size_t item_memory_bytes,
        double memory_limit = 0.7);

    // ========================================================================
    // Диагностика
    // ========================================================================

    // ДИАГНОСТИКА ТОЛЬКО: использует std::cout напрямую, минуя ConsoleOutput.
    // Намеренно — вызывается только при разработке/отладке, не в production multi-GPU режиме.
    // Для runtime-вывода в мультиGPU контексте используй ConsoleOutput::GetInstance().
    /**
     * @brief Вывести конфигурацию пакетов в stdout
     * @param batches Вектор диапазонов пакетов
     * @param total_items Общее число обрабатываемых элементов
     */
    static void PrintBatchInfo(
        const std::vector<BatchRange>& batches,
        size_t total_items);
};

// ============================================================================
// Inline-реализация
// ============================================================================

inline size_t BatchManager::CalculateBatchSizeFromMemory(
    size_t available_memory_bytes,
    size_t total_items,
    size_t item_memory_bytes,
    double memory_limit)
{
    if (item_memory_bytes == 0 || total_items == 0) {
        return total_items;
    }

    // Используемая память
    size_t usable = static_cast<size_t>(
        static_cast<double>(available_memory_bytes) * memory_limit);

    // Сколько элементов помещается?
    size_t fits = usable / item_memory_bytes;

    // Ограничение диапазоном [1, total_items]
    fits = std::max(fits, static_cast<size_t>(1));
    fits = std::min(fits, total_items);

    return fits;
}

inline std::vector<BatchRange> BatchManager::CreateBatches(
    size_t total_items,
    size_t items_per_batch,
    size_t min_tail,
    bool merge_small_tail)
{
    std::vector<BatchRange> batches;

    if (total_items == 0 || items_per_batch == 0) {
        return batches;
    }

    // Если все элементы помещаются в один пакет
    if (items_per_batch >= total_items) {
        BatchRange single;
        single.start = 0;
        single.count = total_items;
        single.batch_idx = 0;
        batches.push_back(single);
        return batches;
    }

    // Число полных пакетов и остаток
    size_t num_full = total_items / items_per_batch;
    size_t remainder = total_items % items_per_batch;

    // Слияние хвоста: если остаток мал (1..min_tail), объединить с предыдущим
    if (merge_small_tail && remainder > 0 && remainder <= min_tail && num_full > 0) {
        num_full--;
        remainder += items_per_batch;
    }

    // Формирование диапазонов пакетов
    size_t current = 0;
    size_t idx = 0;

    // Полные пакеты
    for (size_t i = 0; i < num_full; ++i) {
        BatchRange batch;
        batch.start = current;
        batch.count = items_per_batch;
        batch.batch_idx = idx++;
        batches.push_back(batch);
        current += items_per_batch;
    }

    // Последний пакет (остаток)
    if (remainder > 0 && current < total_items) {
        BatchRange batch;
        batch.start = current;
        batch.count = remainder;
        batch.batch_idx = idx;
        // После слияния remainder = tail + full_batch_size → всегда > items_per_batch
        batch.is_merged = (remainder > items_per_batch);  // Флаг слияния
        batches.push_back(batch);
    }

    return batches;
}

inline void BatchManager::PrintBatchInfo(
    const std::vector<BatchRange>& batches,
    size_t total_items)
{
    std::cout << "  Batch Configuration:\n";
    std::cout << "    Total items: " << total_items << "\n";
    std::cout << "    Num batches: " << batches.size() << "\n";

    for (const auto& batch : batches) {
        std::cout << "    Batch " << batch.batch_idx
                  << ": items [" << batch.start
                  << " .. " << (batch.start + batch.count - 1) << "]"
                  << " count=" << batch.count;
        if (batch.is_merged) {
            std::cout << " (merged tail)";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

} // namespace drv_gpu_lib
