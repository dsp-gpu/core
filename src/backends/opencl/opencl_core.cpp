#include <core/backends/opencl/opencl_core.hpp>
#include <core/memory/svm_capabilities.hpp>
#include <core/logger/logger.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор - Per-Device Architecture (Multi-GPU)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создать OpenCLCore для конкретного устройства
 *
 * ✅ MULTI-GPU: Каждый экземпляр работает со СВОИМ устройством!
 *
 * @param device_index Индекс устройства (0, 1, 2, ...)
 * @param device_type Тип устройства: GPU или CPU
 */
OpenCLCore::OpenCLCore(int device_index, DeviceType device_type)
    : device_index_(device_index),
      device_type_(device_type),
      initialized_(false),
      platform_(nullptr),
      device_(nullptr),
      context_(nullptr) {
    DRVGPU_LOG_DEBUG_GPU(device_index, "OpenCLCore", "Created for device index " + std::to_string(device_index));
}

// ════════════════════════════════════════════════════════════════════════════
// Деструктор
// ════════════════════════════════════════════════════════════════════════════

OpenCLCore::~OpenCLCore() {
    // ✅ FIX: Не вызываем Cleanup() с lock в деструкторе,
    // чтобы избежать проблем с mutex при уничтожении
    if (initialized_) {
        ReleaseResources();
        initialized_ = false;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics - для безопасного перемещения ресурсов
// ════════════════════════════════════════════════════════════════════════════

OpenCLCore::OpenCLCore(OpenCLCore&& other) noexcept
    : device_index_(other.device_index_),
      device_type_(other.device_type_),
      initialized_(other.initialized_),
      platform_(other.platform_),
      device_(other.device_),
      context_(other.context_),
      copy_kernels_(other.copy_kernels_),
      copy_kernels_compiled_(other.copy_kernels_compiled_) {
    // Обнуляем источник
    other.initialized_ = false;
    other.platform_ = nullptr;
    other.device_ = nullptr;
    other.context_ = nullptr;
    other.copy_kernels_ = {};
    other.copy_kernels_compiled_ = false;
}

OpenCLCore& OpenCLCore::operator=(OpenCLCore&& other) noexcept {
    if (this != &other) {
        // Освобождаем свои ресурсы
        Cleanup();

        // Перемещаем
        device_index_ = other.device_index_;
        device_type_ = other.device_type_;
        initialized_ = other.initialized_;
        platform_ = other.platform_;
        device_ = other.device_;
        context_ = other.context_;
        copy_kernels_ = other.copy_kernels_;
        copy_kernels_compiled_ = other.copy_kernels_compiled_;

        // Обнуляем источник
        other.initialized_ = false;
        other.platform_ = nullptr;
        other.device_ = nullptr;
        other.context_ = nullptr;
        other.copy_kernels_ = {};
        other.copy_kernels_compiled_ = false;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация - выбор устройства по индексу
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать OpenCL контекст для КОНКРЕТНОГО устройства
 *
 * ✅ MULTI-GPU: Выбирает устройство по device_index_!
 *
 * Процесс:
 * 1. Получить все доступные устройства указанного типа
 * 2. Выбрать устройство по индексу device_index_
 * 3. Создать контекст для ЭТОГО устройства
 *
 * @throws std::runtime_error если:
 *   - Нет OpenCL платформ
 *   - Нет устройств нужного типа
 *   - device_index_ больше количества устройств
 */
void OpenCLCore::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        DRVGPU_LOG_WARNING_GPU(device_index_, "OpenCLCore", "Device " + std::to_string(device_index_) + " already initialized");
        return;
    }

    InitializeOpenCL();
    initialized_ = true;

    DRVGPU_LOG_INFO_GPU(device_index_, "OpenCLCore", "Device " + std::to_string(device_index_) + " initialized: " + GetDeviceName());
}

/**
 * @brief Внутренняя инициализация OpenCL
 */
void OpenCLCore::InitializeOpenCL() {
    cl_int err;

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 1: Получить все устройства указанного типа
    // ═══════════════════════════════════════════════════════════════════════

    auto all_devices = GetAllDevices(device_type_);

    if (all_devices.empty()) {
        throw std::runtime_error(
            "No OpenCL devices found for type: " +
            std::string(device_type_ == DeviceType::GPU ? "GPU" : "CPU"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 2: Выбрать устройство по индексу
    // ═══════════════════════════════════════════════════════════════════════

    if (device_index_ < 0 || device_index_ >= static_cast<int>(all_devices.size())) {
        throw std::runtime_error(
            "Invalid device index: " + std::to_string(device_index_) +
            ". Available devices: " + std::to_string(all_devices.size()));
    }

    platform_ = all_devices[device_index_].first;
    device_ = all_devices[device_index_].second;

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 3: Создать контекст для ЭТОГО устройства
    // ═══════════════════════════════════════════════════════════════════════

    // Свойства контекста с указанием платформы
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform_),
        0
    };

    context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext for device " + std::to_string(device_index_));

    DRVGPU_LOG_DEBUG_GPU(device_index_, "OpenCLCore", "Context created for device " + std::to_string(device_index_));
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка ресурсов
// ════════════════════════════════════════════════════════════════════════════

void OpenCLCore::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    ReleaseResources();
    initialized_ = false;
    DRVGPU_LOG_DEBUG_GPU(device_index_, "OpenCLCore", "Device " + std::to_string(device_index_) + " cleaned up");
}

void OpenCLCore::ReleaseResources() {
    // Сначала освобождаем copy kernels (зависят от context) — Ref04 БАГ-3 fix
    ReleaseCopyKernels(copy_kernels_);
    copy_kernels_compiled_ = false;

    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }

    // clReleaseDevice требуется только для sub-devices (OpenCL 1.2+)
    // Для обычных устройств это не нужно
    device_ = nullptr;
    platform_ = nullptr;
}

// ════════════════════════════════════════════════════════════════════════════
// Per-GPU кеш copy kernels (Ref04: вместо singleton GpuCopyKernelCache)
// ════════════════════════════════════════════════════════════════════════════

GpuCopyKernels* OpenCLCore::GetOrCompileCopyKernels() {
    if (copy_kernels_compiled_) {
        return &copy_kernels_;
    }

    if (!context_) {
        DRVGPU_LOG_ERROR_GPU(device_index_, "OpenCLCore", "GetOrCompileCopyKernels: context is null");
        return nullptr;
    }

    copy_kernels_ = CompileCopyKernels(context_);
    if (!copy_kernels_.program) {
        DRVGPU_LOG_ERROR_GPU(device_index_, "OpenCLCore", "GetOrCompileCopyKernels: compilation failed");
        return nullptr;
    }

    copy_kernels_compiled_ = true;
    DRVGPU_LOG_DEBUG_GPU(device_index_, "OpenCLCore", "Copy kernels compiled for this GPU context");
    return &copy_kernels_;
}

// ════════════════════════════════════════════════════════════════════════════
// СТАТИЧЕСКИЕ МЕТОДЫ - Multi-GPU Discovery
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить количество доступных устройств
 *
 * ✅ MULTI-GPU: Реальный подсчёт всех GPU/CPU на всех платформах!
 */
int OpenCLCore::GetAvailableDeviceCount(DeviceType device_type) {
    auto devices = GetAllDevices(device_type);
    return static_cast<int>(devices.size());
}

/**
 * @brief Получить все доступные устройства
 *
 * ✅ MULTI-GPU: Сканирует ВСЕ платформы и собирает ВСЕ устройства!
 *
 * @param device_type Тип устройства (GPU или CPU)
 * @return Вектор пар (platform_id, device_id)
 */
std::vector<std::pair<cl_platform_id, cl_device_id>>
OpenCLCore::GetAllDevices(DeviceType device_type) {
    std::vector<std::pair<cl_platform_id, cl_device_id>> result;

    cl_int err;
    cl_uint num_platforms = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 1: Получить все платформы
    // ═══════════════════════════════════════════════════════════════════════

    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return result;  // Нет платформ
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 2: Для каждой платформы получить устройства
    // ═══════════════════════════════════════════════════════════════════════

    cl_device_type cl_dev_type =
        (device_type == DeviceType::GPU) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    for (const auto& platform : platforms) {
        cl_uint num_devices = 0;

        // Получаем количество устройств на этой платформе
        err = clGetDeviceIDs(platform, cl_dev_type, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            continue;  // На этой платформе нет устройств нужного типа
        }

        // Получаем список устройств
        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platform, cl_dev_type, num_devices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            continue;
        }

        // Добавляем все устройства в результат
        for (const auto& device : devices) {
            result.emplace_back(platform, device);
        }
    }

    return result;
}

/**
 * @brief Получить информацию о всех устройствах (для вывода)
 */
std::string OpenCLCore::GetAllDevicesInfo(DeviceType device_type) {
    std::ostringstream oss;

    auto devices = GetAllDevices(device_type);

    oss << "\n" << std::string(70, '=') << "\n";
    oss << "Available " << (device_type == DeviceType::GPU ? "GPU" : "CPU") << " Devices\n";
    oss << std::string(70, '=') << "\n\n";

    if (devices.empty()) {
        oss << "  No devices found!\n";
    } else {
        for (size_t i = 0; i < devices.size(); ++i) {
            cl_device_id device = devices[i].second;

            // Получаем имя устройства
            char name[256] = {0};
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);

            // Получаем вендора
            char vendor[256] = {0};
            clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);

            // Получаем память
            cl_ulong global_mem = 0;
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);

            // Получаем compute units
            cl_uint compute_units = 0;
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);

            oss << "  [" << i << "] " << name << "\n";
            oss << "      Vendor: " << vendor << "\n";
            oss << "      Memory: " << std::fixed << std::setprecision(2)
                << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
            oss << "      Compute Units: " << compute_units << "\n\n";
        }
    }

    oss << std::string(70, '=') << "\n";

    return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// Информация о девайсе - приватные утилиты
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
T OpenCLCore::GetDeviceInfoValue(cl_device_info param) const {
    T value{};
    if (device_) {
        cl_int err = clGetDeviceInfo(device_, param, sizeof(T), &value, nullptr);
        if (err != CL_SUCCESS) {
            DRVGPU_LOG_WARNING_GPU(device_index_, "OpenCLCore", "Failed to get device info param " + std::to_string(param));
        }
    }
    return value;
}

std::string OpenCLCore::GetDeviceInfoString(cl_device_info param) const {
    if (!device_) {
        return "";
    }

    size_t size = 0;
    cl_int err = clGetDeviceInfo(device_, param, 0, nullptr, &size);
    if (err != CL_SUCCESS || size == 0) {
        return "";
    }

    std::vector<char> buffer(size);
    err = clGetDeviceInfo(device_, param, size, buffer.data(), nullptr);
    if (err != CL_SUCCESS) {
        return "";
    }

    return std::string(buffer.data());
}

// ════════════════════════════════════════════════════════════════════════════
// Публичные методы получения информации об устройстве
// ════════════════════════════════════════════════════════════════════════════

std::string OpenCLCore::GetDeviceName() const {
    return GetDeviceInfoString(CL_DEVICE_NAME);
}

std::string OpenCLCore::GetVendor() const {
    return GetDeviceInfoString(CL_DEVICE_VENDOR);
}

std::string OpenCLCore::GetDriverVersion() const {
    return GetDeviceInfoString(CL_DRIVER_VERSION);
}

std::string OpenCLCore::GetPlatformName() const {
    if (!platform_) return "";
    char name[256] = {0};
    clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
    return std::string(name);
}

size_t OpenCLCore::GetGlobalMemorySize() const {
    return GetDeviceInfoValue<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
}

size_t OpenCLCore::GetFreeMemorySize() const {
    if (!initialized_ || !device_) {
        return 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Попытка получить реальную свободную память через расширения вендоров
    // ═══════════════════════════════════════════════════════════════════════

    // NVIDIA: CL_DEVICE_GLOBAL_FREE_MEMORY_NV (расширение cl_nv_device_attribute_query)
    // Значение: 0x4006
    constexpr cl_device_info CL_DEVICE_GLOBAL_FREE_MEMORY_NV = 0x4006;

    cl_ulong free_mem = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_FREE_MEMORY_NV,
                                  sizeof(free_mem), &free_mem, nullptr);

    if (err == CL_SUCCESS && free_mem > 0) {
        // NVIDIA возвращает значение в KB, конвертируем в bytes
        return static_cast<size_t>(free_mem) * 1024;
    }

    // AMD: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD (расширение cl_amd_device_attribute_query)
    // Значение: 0x4039
    constexpr cl_device_info CL_DEVICE_GLOBAL_FREE_MEMORY_AMD = 0x4039;

    err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                          sizeof(free_mem), &free_mem, nullptr);

    if (err == CL_SUCCESS && free_mem > 0) {
        // AMD возвращает в KB
        return static_cast<size_t>(free_mem) * 1024;
    }

    // Fallback: эвристика — 90% от общей памяти
    size_t total = GetGlobalMemorySize();
    return static_cast<size_t>(static_cast<double>(total) * 0.9);
}

size_t OpenCLCore::GetLocalMemorySize() const {
    return GetDeviceInfoValue<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);
}

cl_uint OpenCLCore::GetComputeUnits() const {
    return GetDeviceInfoValue<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
}

size_t OpenCLCore::GetMaxWorkGroupSize() const {
    return GetDeviceInfoValue<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

std::array<size_t, 3> OpenCLCore::GetMaxWorkItemSizes() const {
    std::array<size_t, 3> sizes = {0, 0, 0};
    if (device_) {
        clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes.data(), nullptr);
    }
    return sizes;
}

// ════════════════════════════════════════════════════════════════════════════
// GetDeviceInfo - полная информация
// ════════════════════════════════════════════════════════════════════════════

std::string OpenCLCore::GetDeviceInfo() const {
    std::ostringstream oss;

    oss << "\n" << std::string(70, '=') << "\n";
    oss << "OpenCL Device [" << device_index_ << "] Information\n";
    oss << std::string(70, '=') << "\n\n";

    oss << std::left << std::setw(25) << "Device Index:" << device_index_ << "\n";
    oss << std::left << std::setw(25) << "Device Name:" << GetDeviceName() << "\n";
    oss << std::left << std::setw(25) << "Vendor:" << GetVendor() << "\n";
    oss << std::left << std::setw(25) << "Driver Version:" << GetDriverVersion() << "\n";
    oss << std::left << std::setw(25) << "Device Type:"
        << (device_type_ == DeviceType::GPU ? "GPU" : "CPU") << "\n";

    size_t global_mem = GetGlobalMemorySize();
    size_t local_mem = GetLocalMemorySize();
    oss << std::left << std::setw(25) << "Global Memory:" << std::fixed << std::setprecision(2)
        << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    oss << std::left << std::setw(25) << "Local Memory:" << (local_mem / 1024.0) << " KB\n";

    oss << std::left << std::setw(25) << "Compute Units:" << GetComputeUnits() << "\n";
    oss << std::left << std::setw(25) << "Max Work Group Size:" << GetMaxWorkGroupSize() << "\n";

    auto sizes = GetMaxWorkItemSizes();
    oss << std::left << std::setw(25) << "Max Work Item Sizes:"
        << "[" << sizes[0] << ", " << sizes[1] << ", " << sizes[2] << "]\n";

    oss << "\n" << std::string(70, '=') << "\n";

    return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// SVM (Shared Virtual Memory) методы - OpenCL 2.0+
// ════════════════════════════════════════════════════════════════════════════

cl_uint OpenCLCore::GetOpenCLVersionMajor() const {
    if (!device_) return 0;

    char version_str[256] = {0};
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version_str), version_str, nullptr);
    if (err != CL_SUCCESS) return 0;

    int major = 0, minor = 0;
    if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) >= 1) {
        return static_cast<cl_uint>(major);
    }
    return 0;
}

cl_uint OpenCLCore::GetOpenCLVersionMinor() const {
    if (!device_) return 0;

    char version_str[256] = {0};
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version_str), version_str, nullptr);
    if (err != CL_SUCCESS) return 0;

    int major = 0, minor = 0;
    if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) == 2) {
        return static_cast<cl_uint>(minor);
    }
    return 0;
}

bool OpenCLCore::IsSVMSupported() const {
    if (GetOpenCLVersionMajor() < 2) {
        return false;
    }

    cl_device_svm_capabilities svm_caps = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);

    return (err == CL_SUCCESS && svm_caps != 0);
}

SVMCapabilities OpenCLCore::GetSVMCapabilities() const {
    return SVMCapabilities::Query(device_);
}

std::string OpenCLCore::GetSVMInfo() const {
    std::ostringstream oss;

    oss << "\n" << std::string(60, '=') << "\n";
    oss << "SVM Capabilities [Device " << device_index_ << "]\n";
    oss << std::string(60, '=') << "\n\n";

    cl_uint major = GetOpenCLVersionMajor();
    cl_uint minor = GetOpenCLVersionMinor();

    oss << std::left << std::setw(25) << "OpenCL Version:" << major << "." << minor << "\n";

    if (major < 2) {
        oss << std::left << std::setw(25) << "SVM Supported:" << "NO (OpenCL < 2.0)\n";
        oss << std::string(60, '=') << "\n";
        return oss.str();
    }

    cl_device_svm_capabilities svm_caps = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);

    if (err != CL_SUCCESS || svm_caps == 0) {
        oss << std::left << std::setw(25) << "SVM Supported:" << "NO\n";
        oss << std::string(60, '=') << "\n";
        return oss.str();
    }

    oss << std::left << std::setw(25) << "SVM Supported:" << "YES\n\n";

    oss << "SVM Types:\n";
    oss << "  " << std::left << std::setw(23) << "Coarse-Grain Buffer:"
        << ((svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) ? "YES" : "NO") << "\n";
    oss << "  " << std::left << std::setw(23) << "Fine-Grain Buffer:"
        << ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? "YES" : "NO") << "\n";
    oss << "  " << std::left << std::setw(23) << "Fine-Grain System:"
        << ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) ? "YES" : "NO") << "\n";
    oss << "  " << std::left << std::setw(23) << "Atomics:"
        << ((svm_caps & CL_DEVICE_SVM_ATOMICS) ? "YES" : "NO") << "\n";

    oss << "\n" << std::string(60, '=') << "\n";

    return oss.str();
}

}  // namespace drv_gpu_lib
