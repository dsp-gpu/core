/**
 * @file dsp_core_module.cpp
 * @brief pybind11 bindings for dsp::core (DrvGPU)
 *
 * Python API:
 *   import dsp_core
 *   ctx = dsp_core.ROCmGPUContext(device_index=0)
 *   print(ctx.device_name)
 *
 * Extracted from GPUWorkLib/python/gpu_worklib_bindings.cpp
 * Only DrvGPU classes: GPUContext, ROCmGPUContext, HybridGPUContext
 *
 * TODO (Фаза 3b): проверить и дополнить API биндингов
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <complex>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

// DrvGPU headers
#include "backends/opencl/opencl_backend.hpp"
#include "interface/i_backend.hpp"

#if ENABLE_ROCM
#include "backends/rocm/rocm_backend.hpp"
#include "backends/hybrid/hybrid_backend.hpp"
#endif

namespace py = pybind11;

// ============================================================================
// GPUContext — wraps OpenCL context + backend
// ============================================================================

// Лёгкая Python-обёртка над OpenCL: создаёт context/queue/backend для одного GPU-устройства.
// Lifetime GPUContext должен превышать lifetime любых объектов которые держат ссылку ctx_.
class GPUContext {
public:
    GPUContext(int device_index = 0) {
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("OpenCL: no platforms found (error " + std::to_string(err) + ")");

        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0)
            throw std::runtime_error("OpenCL: no GPU devices found");

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        if (device_index < 0 || static_cast<cl_uint>(device_index) >= num_devices)
            throw std::out_of_range("device_index " + std::to_string(device_index) +
                                    " out of range [0, " + std::to_string(num_devices) + ")");

        device_ = devices[device_index];

        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clCreateContext failed (" + std::to_string(err) + ")");

        // CL_QUEUE_PROFILING_ENABLE нужен для GPUProfiler
#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) {
            clReleaseContext(context_);
            throw std::runtime_error("OpenCL: clCreateCommandQueue failed (" + std::to_string(err) + ")");
        }

        char name[256];
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        device_name_ = name;

        backend_ = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend_->InitializeFromExternalContext(context_, device_, queue_);
    }

    ~GPUContext() {
        backend_.reset();
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    drv_gpu_lib::IBackend* backend() { return backend_.get(); }
    const std::string& device_name() const { return device_name_; }
    cl_command_queue queue() const { return queue_; }

private:
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue queue_ = nullptr;
    std::string device_name_;
    std::unique_ptr<drv_gpu_lib::OpenCLBackend> backend_;
};

// ============================================================================
// ROCmGPUContext — wraps ROCm backend (Linux + AMD GPU only)
// ============================================================================

#if ENABLE_ROCM

class ROCmGPUContext {
public:
  explicit ROCmGPUContext(int device_index = 0)
      : backend_(std::make_unique<drv_gpu_lib::ROCmBackend>()) {
    backend_->Initialize(device_index);
  }

  ~ROCmGPUContext() = default;

  ROCmGPUContext(const ROCmGPUContext&) = delete;
  ROCmGPUContext& operator=(const ROCmGPUContext&) = delete;

  drv_gpu_lib::IBackend* backend() { return backend_.get(); }
  std::string device_name() const { return backend_->GetDeviceName(); }
  int device_index() const { return backend_->GetDeviceIndex(); }

private:
  std::unique_ptr<drv_gpu_lib::ROCmBackend> backend_;
};

// ============================================================================
// HybridGPUContext — wraps HybridBackend (OpenCL + ROCm on one GPU)
// ============================================================================

class HybridGPUContext {
public:
  explicit HybridGPUContext(int device_index = 0)
      : backend_(std::make_unique<drv_gpu_lib::HybridBackend>()) {
    backend_->Initialize(device_index);
  }

  ~HybridGPUContext() = default;

  HybridGPUContext(const HybridGPUContext&) = delete;
  HybridGPUContext& operator=(const HybridGPUContext&) = delete;

  drv_gpu_lib::HybridBackend* backend() { return backend_.get(); }

  std::string opencl_device_name() const {
    auto* ocl = backend_->GetOpenCL();
    if (ocl && ocl->IsInitialized()) return ocl->GetDeviceName();
    return "Unknown";
  }

  std::string rocm_device_name() const {
    auto* rocm = backend_->GetROCm();
    if (rocm && rocm->IsInitialized()) return rocm->GetDeviceName();
    return "Unknown";
  }

  std::string device_name() const { return backend_->GetDeviceName(); }
  int device_index() const { return backend_->GetDeviceIndex(); }

  std::string zero_copy_method() const {
    auto method = backend_->GetBestZeroCopyMethod();
    return drv_gpu_lib::ZeroCopyMethodToString(method);
  }

  bool is_zero_copy_supported() const {
    return backend_->GetBestZeroCopyMethod() != drv_gpu_lib::ZeroCopyMethod::NONE;
  }

private:
  std::unique_ptr<drv_gpu_lib::HybridBackend> backend_;
};

#endif  // ENABLE_ROCM

// ============================================================================
// PYBIND11_MODULE — dsp_core
// ============================================================================

PYBIND11_MODULE(dsp_core, m) {
    m.doc() = "dsp::core — GPU backend management (DrvGPU)\n\n"
              "OpenCL classes:\n"
              "  GPUContext              - GPU device management (OpenCL)\n\n"
              "ROCm classes (Linux + AMD GPU, ENABLE_ROCM=1):\n"
              "  ROCmGPUContext          - GPU device management (ROCm/HIP)\n"
              "  HybridGPUContext        - OpenCL + ROCm on one GPU\n";

    // GPUContext (OpenCL)
    py::class_<GPUContext>(m, "GPUContext",
        "GPU context wrapping OpenCL device.\n\n"
        "Usage:\n"
        "  ctx = dsp_core.GPUContext(device_index=0)\n"
        "  print(ctx.device_name)")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_name", &GPUContext::device_name)
        .def("__repr__", [](const GPUContext& ctx) {
            return "<GPUContext device='" + ctx.device_name() + "'>";
        })
        .def("__enter__", [](GPUContext& self) -> GPUContext& { return self; },
             py::return_value_policy::reference)
        .def("__exit__", [](GPUContext&, py::object, py::object, py::object) {
            return false;
        });

#if ENABLE_ROCM
    // ROCmGPUContext
    py::class_<ROCmGPUContext>(m, "ROCmGPUContext",
        "ROCm GPU context wrapping AMD HIP device.\n\n"
        "Usage:\n"
        "  ctx = dsp_core.ROCmGPUContext(device_index=0)\n"
        "  print(ctx.device_name)")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_name", &ROCmGPUContext::device_name)
        .def_property_readonly("device_index", &ROCmGPUContext::device_index)
        .def("__repr__", [](const ROCmGPUContext& ctx) {
            return "<ROCmGPUContext device='" + ctx.device_name() + "'>";
        })
        .def("__enter__", [](ROCmGPUContext& self) -> ROCmGPUContext& { return self; },
             py::return_value_policy::reference)
        .def("__exit__", [](ROCmGPUContext&, py::object, py::object, py::object) {
            return false;
        });

    // HybridGPUContext
    py::class_<HybridGPUContext>(m, "HybridGPUContext",
        "Hybrid GPU context with OpenCL + ROCm on one GPU.\n\n"
        "Usage:\n"
        "  ctx = dsp_core.HybridGPUContext(device_index=0)\n"
        "  print(ctx.device_name)")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("opencl_device_name", &HybridGPUContext::opencl_device_name)
        .def_property_readonly("rocm_device_name",   &HybridGPUContext::rocm_device_name)
        .def_property_readonly("device_name",        &HybridGPUContext::device_name)
        .def_property_readonly("device_index",       &HybridGPUContext::device_index)
        .def_property_readonly("zero_copy_method",   &HybridGPUContext::zero_copy_method)
        .def_property_readonly("is_zero_copy_supported", &HybridGPUContext::is_zero_copy_supported)
        .def("__repr__", [](const HybridGPUContext& ctx) {
            return "<HybridGPUContext device='" + ctx.device_name() +
                   "' zero_copy='" + ctx.zero_copy_method() + "'>";
        })
        .def("__enter__", [](HybridGPUContext& self) -> HybridGPUContext& { return self; },
             py::return_value_policy::reference)
        .def("__exit__", [](HybridGPUContext&, py::object, py::object, py::object) {
            return false;
        });
#endif  // ENABLE_ROCM

    // Module-level utilities
    m.def("get_gpu_count", []() -> int {
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS) return 0;
        cl_uint count = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
        return static_cast<int>(count);
    }, "Get number of available GPU devices");

    m.def("list_gpus", []() -> py::list {
        py::list result;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS) return result;
        cl_uint count = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
        if (count == 0) return result;

        std::vector<cl_device_id> devices(count);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr);

        for (cl_uint i = 0; i < count; ++i) {
            char name[256];
            clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, nullptr);
            cl_ulong mem = 0;
            clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);

            py::dict info;
            info["index"] = static_cast<int>(i);
            info["name"] = std::string(name);
            info["memory_mb"] = static_cast<int>(mem / (1024 * 1024));
            result.append(info);
        }
        return result;
    }, "List available GPU devices with info");
}
