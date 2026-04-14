#pragma once

/**
 * @file memory_type.hpp
 * @brief Типы памяти GPU и флаги доступа
 * 
 * Определяет типы памяти для буферов GPU:
 * - READ_ONLY - только чтение с GPU (данные не меняются kernel'ом)
 * - WRITE_ONLY - только запись с GPU (kernel пишет результаты)
 * - READ_WRITE - чтение и запись (kernel читает и пишет в буфер)
 * 
 * Эти типы соответствуют флагам cl_mem_flags в OpenCL:
 * - GPU_READ_ONLY  = CL_MEM_READ_ONLY
 * - GPU_WRITE_ONLY = CL_MEM_WRITE_ONLY
 * - GPU_READ_WRITE = CL_MEM_READ_WRITE
 * 
 * Использование:
 * - READ_ONLY: входные данные, константы, таблицы lookup
 * - WRITE_ONLY: выходные буферы для результатов
 * - READ_WRITE: промежуточные буферы, аккумуляторы
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

namespace drv_gpu_lib {

/**
 * @enum MemoryType
 * @brief Типы памяти GPU буфера (режим доступа)
 * 
 * Определяет как kernel будет обращаться к буферу:
 * - На этапе создания буфера влияет на оптимизации драйвера
 * - На этапе выполнения проверяется runtime'ом OpenCL
 * 
 * Примеры использования:
 * @code
 * // Буфер только для чтения (входные данные)
 * auto input_buffer = CreateBuffer<float>(data.size(), MemoryType::GPU_READ_ONLY);
 * 
 * // Буфер только для записи (результаты)
 * auto output_buffer = CreateBuffer<float>(result.size(), MemoryType::GPU_WRITE_ONLY);
 * 
 * // Буфер для чтения и записи (промежуточные данные)
 * auto temp_buffer = CreateBuffer<float>(temp.size(), MemoryType::GPU_READ_WRITE);
 * @endcode
 * 
 * @note Неправильное указание типа может привести к ошибкам CL_INVALID_OPERATION
 * @note Драйвер может использовать эти флаги для оптимизации размещения в памяти
 */
enum class MemoryType {
    GPU_READ_ONLY,   ///< Буфер только для чтения kernel'ом (CL_MEM_READ_ONLY)
                     ///< Используется для: входных данных, констант, lookup таблиц
                     ///< Оптимизация: драйвер может разместить в памяти хоста для быстрого чтения
    
    GPU_WRITE_ONLY,  ///< Буфер только для записи kernel'ом (CL_MEM_WRITE_ONLY)
                     ///< Используется для: выходных буферов, результатов вычислений
                     ///< Оптимизация: драйвер не читает данные, только пишет
    
    GPU_READ_WRITE   ///< Буфер для чтения и записи (CL_MEM_READ_WRITE)
                     ///< Используется для: промежуточных результатов, аккумуляторов
                     ///< Оптимизация: стандартный режим, максимальная гибкость
};

} // namespace drv_gpu_lib
