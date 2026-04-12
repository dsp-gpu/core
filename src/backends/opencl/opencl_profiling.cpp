#include "opencl_profiling.hpp"

#ifndef CL_PROFILING_COMMAND_COMPLETE
#define CL_PROFILING_COMMAND_COMPLETE 0x1284
#endif

namespace drv_gpu_lib {

bool FillOpenCLProfilingData(cl_event event, OpenCLProfilingData& out) {
    if (!event) return false;
    cl_int err;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(out.queued_ns), &out.queued_ns, nullptr);
    if (err != CL_SUCCESS) return false;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(out.submit_ns), &out.submit_ns, nullptr);
    if (err != CL_SUCCESS) return false;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(out.start_ns), &out.start_ns, nullptr);
    if (err != CL_SUCCESS) return false;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(out.end_ns), &out.end_ns, nullptr);
    if (err != CL_SUCCESS) return false;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_COMPLETE, sizeof(out.complete_ns), &out.complete_ns, nullptr);
    if (err != CL_SUCCESS) out.complete_ns = out.end_ns;
    return true;
}

} // namespace drv_gpu_lib
