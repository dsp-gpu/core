#pragma once
/**
 * @file opencl_profiling.hpp
 * @brief Хелпер: заполнение OpenCLProfilingData из cl_event (5 параметров cl_profiling_info)
 *
 * RecordProfilingEvent — единая точка записи: clWaitForEvents + FillOpenCLProfilingData + GPUProfiler.Record
 */

#include <CL/cl.h>
#include "../../services/gpu_profiler.hpp"

namespace drv_gpu_lib {

/**
 * @brief Заполнить OpenCLProfilingData из cl_event (все 5 параметров)
 * @param event cl_event с включённым profiling (CL_QUEUE_PROFILING_ENABLE)
 * @param out Заполняемая структура
 * @return true если все 5 значений получены, false при ошибке
 */
bool FillOpenCLProfilingData(cl_event event, OpenCLProfilingData& out);

/**
 * @brief Дождаться события, заполнить OpenCLProfilingData и записать в GPUProfiler
 * @param event cl_event (может быть nullptr — тогда ничего не делаем)
 * @param gpu_id Индекс GPU
 * @param module Имя модуля (например "FFTProcessor", "SpectrumMaxima")
 * @param event_name Имя события (например "Upload", "FFT", "PostKernel")
 */
inline void RecordProfilingEvent(cl_event event, int gpu_id,
                                const char* module, const char* event_name) {
    if (!event) return;
    clWaitForEvents(1, &event);
    OpenCLProfilingData pdata{};
    if (FillOpenCLProfilingData(event, pdata)) {
        GPUProfiler::GetInstance().Record(gpu_id, module, event_name, pdata);
    }
}

} // namespace drv_gpu_lib
