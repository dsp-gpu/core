# DrvGPU OpenCL Backend

## Оглавление

1. [Обзор OpenCL бэкенда](#обзор-opencl-бэкенда)
2. [OpenCLBackend](#openclbackend)
3. [OpenCLCore](#openclcore)
4. [External Context](#external-context)
5. [Жизненный цикл](#жизненный-цикл)
6. [Примеры использования](#примеры-использования)
7. [Roadmap](#roadmap)

---

## Обзор OpenCL бэкенда

OpenCL бэкенд — основная и наиболее полно реализованная часть DrvGPU.

```
OpenCL Backend Architecture
┌─────────────────────────────────────────────────────────────┐
│                     OpenCLBackend                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   IBackend Interface                │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│       ┌──────────────────┼──────────────────┐              │
│       ▼                  ▼                  ▼              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │   Core   │      │  Memory  │      │  Queue   │         │
│  │  Utils   │      │ Manager  │      │   Pool   │         │
│  └──────────┘      └──────────┘      └──────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Ключевые классы

| Класс | Файл | Назначение |
|-------|------|------------|
| `OpenCLBackend` | opencl_backend.hpp/cpp | Основной класс бэкенда |
| `OpenCLCore` | opencl_core.hpp/cpp | Низкоуровневые операции |
| `OpenCLBackendExternal` | opencl_backend_external.hpp/cpp | External Context |
| `CommandQueuePool` | command_queue_pool.hpp/cpp | Пул очередей (см. Command.md) |

**См. также**: [Command.md](Command.md) — управление очередями

---

## OpenCLBackend

### Назначение

`OpenCLBackend` реализует интерфейс `IBackend` для OpenCL, обеспечивая единый API для работы с GPU.

### Файл

[`opencl_backend.hpp`](../../include/DrvGPU/backends/opencl/opencl_backend.hpp)

### Интерфейс класса

```cpp
class OpenCLBackend : public IBackend {
public:
    OpenCLBackend();
    ~OpenCLBackend() override;
    
    // ═══════════════════════════════════════════════
    // IBackend Lifecycle
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Инициализировать бэкенд
     * @param device_index Индекс устройства (0-based)
     * 
     * Создаёт OpenCL контекст, очередь команд и получает информацию об устройстве.
     */
    void Initialize(int device_index) override;
    
    /**
     * @brief Проверить инициализацию
     */
    bool IsInitialized() const override;
    
    /**
     * @brief Освободить ресурсы
     */
    void Cleanup() override;
    
    // ═══════════════════════════════════════════════
    // IBackend Device Info
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Получить тип бэкенда
     */
    BackendType GetType() const override { 
        return BackendType::OPENCL; 
    }
    
    /**
     * @brief Получить информацию об устройстве
     */
    GPUDeviceInfo GetDeviceInfo() const override;
    
    /**
     * @brief Получить индекс устройства
     */
    int GetDeviceIndex() const override;
    
    /**
     * @brief Получить имя устройства
     */
    std::string GetDeviceName() const override;
    
    // ═══════════════════════════════════════════════
    // IBackend Native Handles
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Получить нативный контекст OpenCL
     */
    void* GetNativeContext() const override;
    
    /**
     * @brief Получить нативное устройство OpenCL
     */
    void* GetNativeDevice() const override;
    
    /**
     * @brief Получить нативную очередь команд
     */
    void* GetNativeQueue() const override;
    
    // ═══════════════════════════════════════════════
    // IBackend Memory Operations
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Выделить память на GPU
     */
    void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
    
    /**
     * @brief Освободить память
     */
    void Free(void* ptr) override;
    
    /**
     * @brief Копировать Host → Device
     */
    void MemcpyHostToDevice(void* dst, const void* src,
                            size_t size_bytes) override;
    
    /**
     * @brief Копировать Device → Host
     */
    void MemcpyDeviceToHost(void* dst, const void* src,
                            size_t size_bytes) override;
    
    /**
     * @brief Копировать Device → Device
     */
    void MemcpyDeviceToDevice(void* dst, const void* src,
                              size_t size_bytes) override;
    
    // ═══════════════════════════════════════════════
    // IBackend Synchronization
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Синхронизировать очередь
     */
    void Synchronize() override;
    
    /**
     * @brief Flush очереди
     */
    void Flush() override;
    
    // ═══════════════════════════════════════════════
    // IBackend Capabilities
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Поддержка SVM
     */
    bool SupportsSVM() const override;
    
    /**
     * @brief Поддержка double precision
     */
    bool SupportsDoublePrecision() const override;
    
    /**
     * @brief Максимальный размер work group
     */
    size_t GetMaxWorkGroupSize() const override;
    
    /**
     * @brief Размер глобальной памяти
     */
    size_t GetGlobalMemorySize() const override;
    
    /**
     * @brief Размер локальной памяти
     */
    size_t GetLocalMemorySize() const override;

private:
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue queue_ = nullptr;
    bool initialized_ = false;
    
    /**
     * @brief Проверить ошибку OpenCL
     */
    void checkError(cl_int error, const std::string& message);
    
    /**
     * @brief Получить строку ошибки OpenCL
     */
    static std::string GetErrorString(cl_int error);
};
```

### Члены класса

| Член | Тип | Описание |
|------|-----|----------|
| `core_` | `unique_ptr<OpenCLCore>` | ✅ Per-device OpenCLCore |
| `context_` | `cl_context` | OpenCL контекст (из core_) |
| `device_` | `cl_device_id` | Идентификатор устройства (из core_) |
| `queue_` | `cl_command_queue` | Очередь команд |
| `initialized_` | `bool` | Флаг инициализации |
| `owns_resources_` | `bool` | Владеет ресурсами? |

---

## OpenCLCore

### Назначение

`OpenCLCore` — класс для управления OpenCL устройством. **Per-device architecture для Multi-GPU!**

> ⚠️ **ВАЖНО (v2.0)**: Singleton паттерн УДАЛЁН! Теперь каждый экземпляр работает со СВОИМ устройством.

### Файл

[`opencl_core.hpp`](../../include/DrvGPU/backends/opencl/opencl_core.hpp)

### Интерфейс класса

```cpp
class OpenCLCore {
public:
    // ═══════════════════════════════════════════════
    // ✅ MULTI-GPU: Per-device конструктор
    // ═══════════════════════════════════════════════

    /**
     * @brief Создать OpenCLCore для конкретного устройства
     * @param device_index Индекс устройства (0, 1, 2, ...)
     * @param device_type Тип устройства: GPU или CPU
     */
    explicit OpenCLCore(int device_index = 0, DeviceType device_type = DeviceType::GPU);

    ~OpenCLCore();

    // Move semantics (копирование запрещено)
    OpenCLCore(OpenCLCore&& other) noexcept;
    OpenCLCore& operator=(OpenCLCore&& other) noexcept;
    OpenCLCore(const OpenCLCore&) = delete;
    OpenCLCore& operator=(const OpenCLCore&) = delete;

    // ═══════════════════════════════════════════════
    // Инициализация
    // ═══════════════════════════════════════════════

    /**
     * @brief Инициализировать контекст для ЭТОГО устройства
     */
    void Initialize();

    /**
     * @brief Проверить инициализацию
     */
    bool IsInitialized() const;

    /**
     * @brief Освободить ресурсы
     */
    void Cleanup();

    // ═══════════════════════════════════════════════
    // ✅ MULTI-GPU: Статические методы обнаружения
    // ═══════════════════════════════════════════════

    /**
     * @brief Получить количество доступных устройств
     */
    static int GetAvailableDeviceCount(DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Получить все устройства (platform, device) пары
     */
    static std::vector<std::pair<cl_platform_id, cl_device_id>>
        GetAllDevices(DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Получить информацию обо всех устройствах (для вывода)
     */
    static std::string GetAllDevicesInfo(DeviceType device_type = DeviceType::GPU);
    
    // ═══════════════════════════════════════════════
    // Context
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Создать контекст для устройства
     */
    static cl_context CreateContext(cl_device_id device);
    
    /**
     * @brief Создать контекст с указанием устройств
     */
    static cl_context CreateContext(const std::vector<cl_device_id>& devices);
    
    // ═══════════════════════════════════════════════
    // Command Queue
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Создать очередь команд
     */
    static cl_command_queue CreateQueue(cl_context context, 
                                        cl_device_id device);
    
    /**
     * @brief Создать профилированную очередь
     */
    static cl_command_queue CreateProfilingQueue(cl_context context,
                                                  cl_device_id device);
    
    // ═══════════════════════════════════════════════
    // Program and Kernel
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Создать программу из исходного кода
     */
    static cl_program CreateProgram(cl_context context,
                                    const std::string& source);
    
    /**
     * @brief Создать программу из файла
     */
    static cl_program CreateProgramFromFile(cl_context context,
                                            const std::string& filename);
    
    /**
     * @brief Скомпилировать программу
     */
    static void BuildProgram(cl_program program, 
                             const std::string& options = "");
    
    /**
     * @brief Создать kernel из программы
     */
    static cl_kernel CreateKernel(cl_program program,
                                  const std::string& kernel_name);
    
    /**
     * @brief Создать kernels из программы (все)
     */
    static std::vector<cl_kernel> CreateAllKernels(cl_program program);
    
    // ═══════════════════════════════════════════════
    // Memory
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Создать буфер
     */
    static cl_mem CreateBuffer(cl_context context,
                               cl_mem_flags flags,
                               size_t size,
                               void* host_ptr = nullptr);
    
    /**
     * @brief Создать Image2D
     */
    static cl_mem CreateImage2D(cl_context context,
                                cl_mem_flags flags,
                                const cl_image_format* format,
                                size_t width,
                                size_t height,
                                size_t row_pitch = 0,
                                void* host_ptr = nullptr);
    
    // ═══════════════════════════════════════════════
    // Execution
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Запустить kernel
     */
    static void EnqueueKernel(cl_command_queue queue,
                              cl_kernel kernel,
                              cl_uint work_dim,
                              const size_t* global_work_size,
                              const size_t* local_work_size);
    
    /**
     * @brief Копировать память
     */
    static void EnqueueCopyBuffer(cl_command_queue queue,
                                  cl_mem src,
                                  cl_mem dst,
                                  size_t src_offset,
                                  size_t dst_offset,
                                  size_t size);
    
    /**
     * @brief Синхронизировать очередь
     */
    static void Finish(cl_command_queue queue);
    
    /**
     * @brief Flush очереди
     */
    static void Flush(cl_command_queue queue);
    
    // ═══════════════════════════════════════════════
    // Error Handling
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Получить строку ошибки
     */
    static std::string GetErrorString(cl_int error);
};
```

---

## External Context

### Назначение

`OpenCLBackendExternal` позволяет использовать существующий OpenCL контекст вместо создания нового.

### Файл

[`opencl_backend_external.hpp`](../../include/DrvGPU/backends/opencl/opencl_backend_external.hpp)

### Интерфейс класса

```cpp
class OpenCLBackendExternal : public OpenCLBackend {
public:
    /**
     * @brief Конструктор с внешним контекстом
     * @param external_context Внешний cl_context
     * @param device_index Индекс устройства в контексте
     */
    OpenCLBackendExternal(cl_context external_context, 
                          int device_index = 0);
    
    /**
     * @brief Конструктор с внешним устройством
     */
    OpenCLBackendExternal(cl_context external_context,
                          cl_device_id external_device);
    
    /**
     * @brief Конструктор с внешней очередью
     */
    OpenCLBackendExternal(cl_context external_context,
                          cl_device_id external_device,
                          cl_command_queue external_queue);
    
    /**
     * @brief Освободить ресурсы (переопределено)
     * 
     * Не освобождает внешние ресурсы!
     */
    void Cleanup() override;
    
    /**
     * @brief Проверить, является ли контекст внешним
     */
    bool IsExternalContext() const { return true; }
    
    /**
     * @brief Получить оригинальный контекст
     */
    cl_context GetOriginalContext() const { return external_context_; }

private:
    cl_context external_context_ = nullptr;
    bool owns_context_ = false;
};
```

### Когда использовать External Context

| Сценарий | Преимущество |
|----------|--------------|
| Интеграция с другими библиотеками | Общий контекст |
| Многобиблиотечное приложение | Один контекст на всё приложение |
| Производительность | Избежать создания нескольких контекстов |
| Совместное использование ресурсов | Общие буферы между библиотеками |

---

## Жизненный цикл

```
OpenCLBackend Lifecycle
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Constructor                                                │
│       │                                                      │
│       ▼                                                      │
│   Initialize(device_index)                                   │
│       │                                                      │
│       ├──► Get platform                                     │
│       ├──► Get device by index                              │
│       ├──► Create context                                   │
│       ├──► Create command queue                             │
│       └──► Get device info                                  │
│       │                                                      │
│       ▼                                                      │
│   IsInitialized() = true                                    │
│       │                                                      │
│       ▼                                                      │
│   Operations (Allocate, Memcpy, Execute, etc.)              │
│       │                                                      │
│       ▼                                                      │
│   Cleanup()                                                  │
│       │                                                      │
│       ├──► Release queue                                    │
│       ├──► Release context                                  │
│       └──► Reset state                                      │
│       │                                                      │
│       ▼                                                      │
│   IsInitialized() = false                                   │
│       │                                                      │
│       ▼                                                      │
│   Destructor                                                 │
│       │                                                      │
│       └──► Cleanup() (если не был вызван)                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Примеры использования

### Пример 1: Базовое использование

```cpp
#include <core/drv_gpu.hpp>

int main() {
    drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
    gpu.Initialize();
    
    auto info = gpu.GetDeviceInfo();
    printf("GPU: %s\n", info.name.c_str());
    
    auto& memory = gpu.GetMemoryManager();
    float* data = static_cast<float*>(
        memory.Allocate(1024 * sizeof(float))
    );
    
    gpu.Synchronize();
    gpu.Cleanup();
    
    return 0;
}
```

### Пример 2: Использование External Context

```cpp
#include <core/backends/opencl/opencl_backend_external.hpp>

int main() {
    // Существующий OpenCL контекст (из другой библиотеки)
    cl_context my_context = /* ... */;
    cl_device_id my_device = /* ... */;
    
    // Создать DrvGPU с внешним контекстом
    drv_gpu_lib::OpenCLBackendExternal gpu(my_context, my_device);
    gpu.Initialize();
    
    // Использовать DrvGPU API
    auto info = gpu.GetDeviceInfo();
    auto ptr = gpu.Allocate(1024);
    
    // DrvGPU использует существующий контекст
    // При Cleanup() контекст НЕ освобождается
    
    return 0;
}
```

### Пример 3: Использование OpenCLCore

```cpp
#include <core/backends/opencl/opencl_core.hpp>

void compileAndRunKernel() {
    auto devices = drv_gpu_lib::OpenCLCore::GetAllDevices();
    // devices — vector<pair<cl_platform_id, cl_device_id>>
    auto context = drv_gpu_lib::OpenCLCore::CreateContext(devices[0].second);
    auto queue = drv_gpu_lib::OpenCLCore::CreateQueue(context, devices[0].second);
    
    const char* source = R"(
        __kernel void hello(__global float* data) {
            int id = get_global_id(0);
            data[id] = id * 2.0f;
        }
    )";
    
    auto program = drv_gpu_lib::OpenCLCore::CreateProgram(context, source);
    drv_gpu_lib::OpenCLCore::BuildProgram(program);
    
    auto kernel = drv_gpu_lib::OpenCLCore::CreateKernel(program, "hello");
    
    cl_mem buffer = drv_gpu_lib::OpenCLCore::CreateBuffer(
        context, CL_MEM_READ_WRITE, 1024 * sizeof(float), nullptr
    );
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    
    size_t global_size = 256;
    drv_gpu_lib::OpenCLCore::EnqueueKernel(
        queue, kernel, 1, &global_size, nullptr
    );
    
    drv_gpu_lib::OpenCLCore::Finish(queue);
    
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
```

---

## Сводная таблица файлов

| Файл | Класс/Структура | Назначение | Сложность |
|------|-----------------|------------|-----------|
| `opencl_core.hpp` | `OpenCLCore` | Низкоуровневые операции | Средняя |
| `opencl_core.cpp` | `OpenCLCore` | Реализация | Средняя |
| `opencl_backend.hpp` | `OpenCLBackend` | Основной бэкенд | Высокая |
| `opencl_backend.cpp` | `OpenCLBackend` | Реализация | Высокая |
| `opencl_backend_external.hpp` | `OpenCLBackendExternal` | External Context | Средняя |
| `opencl_backend_external.cpp` | `OpenCLBackendExternal` | Реализация | Средняя |
| `opencl_profiling.hpp` | `FillOpenCLProfilingData`, `RecordProfilingEvent` | Хелперы профилирования cl_event | Низкая |
| `opencl_export.hpp` | `ZeroCopyMethod`, утилиты экспорта | DMA-buf / GpuVA экспорт cl_mem (Linux only) | Средняя |
| `command_queue_pool.hpp` | `CommandQueuePool` | Пул очередей | Средняя |
| `command_queue_pool.cpp` | `CommandQueuePool` | Реализация | Средняя |

---

## Multi-GPU поддержка (v2.0)

### Архитектура Multi-GPU

```
✅ НОВАЯ АРХИТЕКТУРА (Per-Device):
┌──────────────────────────────────────────────────────────┐
│                      GPUManager                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │ DrvGPU[0]  │ │ DrvGPU[1]  │ │ DrvGPU[2]  │ ...       │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘           │
│        │              │              │                   │
│  ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐           │
│  │OpenCLBack- │ │OpenCLBack- │ │OpenCLBack- │           │
│  │end[0]      │ │end[1]      │ │end[2]      │           │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘           │
│        │              │              │                   │
│  ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐           │
│  │OpenCLCore  │ │OpenCLCore  │ │OpenCLCore  │           │
│  │(GPU 0)     │ │(GPU 1)     │ │(GPU 2)     │           │
│  │context[0]  │ │context[1]  │ │context[2]  │           │
│  └────────────┘ └────────────┘ └────────────┘           │
│                                                          │
│  ✅ Каждый экземпляр имеет СВОЙ контекст!               │
└──────────────────────────────────────────────────────────┘
```

### Пример: Использование нескольких GPU

```cpp
#include <core/gpu_manager.hpp>

int main() {
    drv_gpu_lib::GPUManager manager;

    // Инициализировать ВСЕ доступные GPU
    manager.InitializeAll(drv_gpu_lib::BackendType::OPENCL);

    std::cout << "Found " << manager.GetGPUCount() << " GPUs\n";

    // Работа с разными GPU параллельно
    #pragma omp parallel for
    for (size_t i = 0; i < manager.GetGPUCount(); ++i) {
        auto& gpu = manager.GetGPU(i);
        // Каждый поток работает со СВОИМ GPU!
        processOnGPU(gpu, data[i]);
    }

    return 0;
}
```

### Пример: Прямое использование OpenCLCore

```cpp
#include <core/backends/opencl/opencl_core.hpp>

void multiGPUExample() {
    // Узнать сколько GPU доступно
    int gpu_count = drv_gpu_lib::OpenCLCore::GetAvailableDeviceCount(
        drv_gpu_lib::DeviceType::GPU
    );

    std::cout << "Available GPUs: " << gpu_count << "\n";

    // Вывести информацию
    std::cout << drv_gpu_lib::OpenCLCore::GetAllDevicesInfo(
        drv_gpu_lib::DeviceType::GPU
    );

    // Создать экземпляры для разных GPU
    std::vector<std::unique_ptr<drv_gpu_lib::OpenCLCore>> cores;
    for (int i = 0; i < gpu_count; ++i) {
        auto core = std::make_unique<drv_gpu_lib::OpenCLCore>(
            i, drv_gpu_lib::DeviceType::GPU
        );
        core->Initialize();
        cores.push_back(std::move(core));
    }

    // Теперь каждый core работает со своим GPU!
}
```

---

## Roadmap

| Версия | Компонент | Статус |
|--------|-----------|--------|
| v1.0 | OpenCLBackend | ✅ Готов |
| v1.0 | OpenCLCore (Singleton) | ❌ Устарел |
| v2.0 | OpenCLCore (Per-Device) | ✅ Готов |
| v2.0 | Multi-GPU Discovery | ✅ Готов |
| v1.0 | External Context | ✅ Готов |
| v1.0 | Command Queue Pool | ✅ Готов |
| v1.1 | opencl_export.hpp (ZeroCopy DMA/GpuVA) | ✅ Готов |
| v1.1 | opencl_profiling.hpp (FillOpenCLProfilingData) | ✅ Готов |
| v2.0 | ROCm бэкенд | ✅ Готов (см. backends/rocm/) |
| — | CUDA бэкенд | ❌ Не планируется |

---

## Зависимости

```
OpenCLBackend
    ├── OpenCLCore
    ├── IBackend
    ├── GPUDeviceInfo
    └── Logger
         └── ILogger
              └── DefaultLogger
                   └── spdlog
```

---

## Заключение

OpenCL бэкенд DrvGPU предоставляет:

- ✅ Полную реализацию `IBackend` интерфейса
- ✅ External Context поддержку для интеграции
- ✅ Command Queue Pool для оптимизации (см. Command.md)
- ✅ Низкоуровневые утилиты через `OpenCLCore`
- ✅ Thread-safe операции

**См. также**:
- [Command.md](Command.md) — управление очередями команд
- [Memory.md](Memory.md) — система памяти
- [Architecture.md](Architecture.md) — общая архитектура
