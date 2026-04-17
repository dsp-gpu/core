# DrvGPU — Полная документация

> Ядро библиотеки DSP-GPU: единая абстракция над OpenCL, ROCm/HIP и Hybrid GPU бэкендами

**Namespace**: `drv_gpu_lib`
**Каталог**: `DrvGPU/`
**Платформы**: OpenCL 3.0 (все GPU), ROCm 7.2+ (AMD, `ENABLE_ROCM=1`), Hybrid (OpenCL + ROCm одновременно)

---

## Содержание

1. [Обзор и назначение](#1-обзор-и-назначение)
2. [Архитектурные слои](#2-архитектурные-слои)
3. [Паттерны проектирования](#3-паттерны-проектирования)
4. [Бэкенды (IBackend)](#4-бэкенды)
5. [Память (MemoryManager, GPUBuffer)](#5-память)
6. [Сервисы (ConsoleOutput, GPUProfiler, ServiceManager)](#6-сервисы)
7. [Multi-GPU (GPUManager)](#7-multi-gpu)
8. [External Context Integration](#8-external-context)
9. [C4 Диаграммы](#9-c4-диаграммы)
10. [C++ примеры](#10-cpp-примеры)
11. [Python API](#11-python-api)
12. [Тесты](#12-тесты)
13. [Файловая структура](#13-файловая-структура)
14. [Ссылки](#14-ссылки)

---

## 1. Обзор и назначение

**DrvGPU** — ядро DSP-GPU. Предоставляет единый API для работы с GPU независимо от платформы. Все модули библиотеки (signal_generators, FFT, statistics, heterodyne, strategies…) принимают `IBackend*` — и больше ничего не знают о конкретном GPU.

**Принцип**: код модуля пишется один раз, работает и на OpenCL (NVIDIA/Intel/AMD), и на ROCm (AMD).

```
Приложение
    │
    ├── DrvGPU (Facade) ─────── IBackend (Bridge)
    │       │                        │
    │       ├── MemoryManager        ├── OpenCLBackend   → cl_mem / cl_command_queue
    │       ├── ModuleRegistry       ├── ROCmBackend     → hipStream_t
    │       └── ServiceManager       └── HybridBackend   → OpenCL + ROCm + ZeroCopy
    │
    └── Все модули принимают IBackend* — никакой зависимости от платформы
```

**Ключевые гарантии**:
- RAII — все ресурсы освобождаются автоматически
- Thread-safe — mutex на уровне DrvGPU
- Multi-GPU — несколько `DrvGPU` или `GPUManager`
- External Context — подключение к уже существующему OpenCL/HIP контексту

---

## 2. Архитектурные слои

```
┌─────────────────────────────────────────────────────────────────┐
│              Application Layer (Modules, Tests)                  │
│  signal_generators / fft_processor / statistics / heterodyne…   │
│  Все принимают IBackend* — не знают о платформе                  │
├─────────────────────────────────────────────────────────────────┤
│                       Core Layer                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌──────────┐   │
│  │  DrvGPU  │  │ GPUManager│  │MemoryManager │  │ Module   │   │
│  │ (Facade) │  │ (Multi)   │  │ (Allocation) │  │ Registry │   │
│  └────┬─────┘  └─────┬─────┘  └──────┬───────┘  └──────────┘   │
├───────┴───────────────┴───────────────┴─────────────────────────┤
│               Backend Abstraction Layer                          │
│                  ┌─────────────────────┐                         │
│                  │   IBackend (Bridge)  │                         │
│                  └──┬──────────┬───────┘                         │
│           ┌─────────┘          ├──────────────┐                  │
│    ┌──────▼──┐         ┌──────▼──┐    ┌──────▼──────┐           │
│    │OpenCL   │         │ ROCm    │    │  Hybrid     │           │
│    │Backend  │         │ Backend │    │ (OCL+ROCm)  │           │
│    └──────────┘         └─────────┘    └─────────────┘           │
├─────────────────────────────────────────────────────────────────┤
│      Memory Layer          │        Services Layer               │
│  GPUBuffer<T> (RAII, owns) │  ConsoleOutput  (Singleton)        │
│  HIPBuffer<T> (non-owning) │  GPUProfiler    (Singleton)        │
│  SVMBuffer    (SVM shared) │  ServiceManager (Singleton)        │
│  MemoryManager (tracking)  │  BatchManager   (large data)       │
│  ExternalCLBufferAdapter   │  KernelCacheService (JIT cache)    │
├─────────────────────────────────────────────────────────────────┤
│    Common: Logger (plog) / Config (configGPU.json) / DeviceInfo │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Паттерны проектирования

| Паттерн | Где применён | Зачем |
|---------|-------------|-------|
| **Bridge** | `IBackend` ← `OpenCLBackend` / `ROCmBackend` | Смена платформы без изменения клиентского кода |
| **Facade** | `DrvGPU` | Скрывает сложность бэкенда, памяти, модулей |
| **Singleton** | `ConsoleOutput`, `GPUProfiler`, `ServiceManager` | Один экземпляр на процесс |
| **Factory** | `DrvGPU::CreateFromExternal*()`, `GPUManager::InitializeAll()` | Создание с нужным бэкендом |
| **Strategy** | `LoadBalancingStrategy` в GPUManager | Выбор GPU (round-robin, least-loaded, fastest-first) |
| **Registry** | `ModuleRegistry` | Хранение compute-модулей |
| **Object Pool** | `CommandQueuePool` | Повторное использование cl_command_queue |
| **Template Method** | `AsyncServiceBase<T>`, `GpuBenchmarkBase` | Фиксированный алгоритм, переменные шаги |
| **Null Object** | `NullCheckpointSave` в strategies | Zero overhead в production |
| **Per-Device** | `OpenCLCore`, `ROCmCore` | Один экземпляр core per GPU (не Singleton!) |
| **RAII** | `GPUBuffer<T>`, `DrvGPU` | Автоматическое освобождение ресурсов |

> ⚠️ `OpenCLCore` с версии v2.0 — **не Singleton**! Это per-device класс для поддержки Multi-GPU.

---

## 4. Бэкенды

### IBackend — интерфейс (Bridge)

**Файл**: `DrvGPU/interface/i_backend.hpp`

Все GPU-операции проходят через `IBackend*`. Модули никогда не работают с `cl_context` или `hipStream_t` напрямую.

```
Жизненный цикл:
  Initialize(device_index) → (использование) → Cleanup()

  Ownership (для External Context):
  owns_resources_=true  → backend освободит ресурсы при Cleanup()
  owns_resources_=false → backend только обнуляет указатели, не освобождает
```

**Группы методов**:
- **Lifecycle**: `Initialize()`, `Cleanup()`, `IsInitialized()`
- **Ownership**: `SetOwnsResources(bool)`, `OwnsResources() → bool`
- **Info**: `GetType()`, `GetDeviceInfo()`, `GetDeviceIndex()`, `GetDeviceName()`
- **Native handles**: `GetNativeContext()`, `GetNativeDevice()`, `GetNativeQueue()`
- **Memory**: `Allocate(size)`, `Free(ptr)`, `MemcpyHostToDevice()`, `MemcpyDeviceToHost()`, `MemcpyDeviceToDevice()`
- **Sync**: `Synchronize()`, `Flush()`
- **Caps**: `SupportsSVM()`, `SupportsDoublePrecision()`, `GetMaxWorkGroupSize()`, `GetGlobalMemorySize()`, `GetFreeMemorySize()`, `GetLocalMemorySize()`

### OpenCLBackend

**Файлы**: `backends/opencl/opencl_backend.hpp/cpp`, `opencl_core.hpp/cpp`

- Реализация IBackend для OpenCL 3.0
- Работает на AMD/NVIDIA/Intel GPU
- `InitializeFromExternalContext(cl_context, cl_device_id, cl_command_queue)` — внешний контекст
- `CommandQueuePool` — пул очередей для параллельной работы

### ROCmBackend ✅

**Файлы**: `backends/rocm/rocm_backend.hpp/cpp`, `rocm_core.hpp/cpp`

- Реализация IBackend для ROCm/HIP (AMD GPU, `ENABLE_ROCM=1`)
- Управляет `hipStream_t` с флагом `owns_stream_`
- `InitializeFromExternalStream(device_index, hipStream_t)` — внешний stream
- `ROCmCore` — per-device HIP контекст (не Singleton)

### HybridBackend ✅

**Файлы**: `backends/hybrid/hybrid_backend.hpp/cpp`

- Комбинирует OpenCL и ROCm в одном бэкенде
- Позволяет использовать hipBLAS/hipFFT (ROCm) и cl_mem (OpenCL) одновременно
- `ZeroCopyBridge` — передача данных между cl_mem и HIP без CPU
- `InitializeFromExternalContexts(device_idx, cl_ctx, cl_dev, cl_q, hipStream_t)`

### ZeroCopyBridge ✅

**Файлы**: `backends/rocm/zero_copy_bridge.hpp/cpp`

- Мост для DMA между OpenCL и ROCm буферами (общее VRAM на AMD GPU)
- Использует DMA-buf export + GPU Virtual Address

### BackendType enum

**Файл**: `common/backend_type.hpp`

```cpp
enum class BackendType {
  OPENCL,          // OpenCL 3.0 (Windows + Linux, все GPU)
  ROCM,            // ROCm/HIP (Linux, AMD GPU, ENABLE_ROCM=1)
  OPENCL_AND_ROCM, // HybridBackend (OpenCL + ROCm одновременно)
  AUTO             // Выбрать лучший автоматически
};
```

---

## 5. Память

### GPUBuffer\<T\> — RAII, owning

**Файл**: `memory/gpu_buffer.hpp`

Типобезопасный буфер на GPU. Не копируется (move-only).

```cpp
// Создание через MemoryManager
auto buf = mem.CreateBuffer<complex<float>>(4096);
auto buf2 = mem.CreateBuffer<float>(host_vec);  // инициализация из CPU

// Запись / чтение
buf->Write(host_data, num_elements);
buf->Write(host_vector);
auto result = buf->Read();        // vector<T>
buf->Read(host_data, num_elems);

// Информация
T*     ptr   = buf->GetPtr();
size_t count = buf->GetNumElements();
size_t bytes = buf->GetSizeBytes();
bool   valid = buf->IsValid();

// RAII — при выходе из scope освобождается автоматически
```

### HIPBuffer\<T\> — non-owning (ROCm)

**Файл**: `memory/hip_buffer.hpp`

Non-owning обёртка для `void*` HIP-буфера. Не управляет памятью — только хранит указатель. Используется когда данные уже выделены через `backend->Allocate()` или hipMalloc.

### SVMBuffer — Shared Virtual Memory

**Файл**: `memory/svm_buffer.hpp`

Буфер в Shared Virtual Memory — CPU и GPU видят одни и те же адреса. Только для платформ с поддержкой SVM (проверить через `backend->SupportsSVM()`).

### MemoryManager — менеджер аллокаций

**Файл**: `memory/memory_manager.hpp`

```cpp
MemoryManager& mem = drv.GetMemoryManager();

// Создание типобезопасных буферов
auto buf = mem.CreateBuffer<float>(1024);          // пустой
auto buf = mem.CreateBuffer<float>(host_ptr, 1024); // + копия данных

// Сырые аллокации
void* ptr = mem.Allocate(size_bytes);
mem.Free(ptr);

// Статистика
size_t n     = mem.GetAllocationCount();
size_t bytes = mem.GetTotalAllocatedBytes();
std::string s = mem.GetStatistics();
mem.ResetStatistics();
```

---

## 6. Сервисы

### ConsoleOutput — мультиGPU-безопасный вывод (Singleton)

**Файл**: `services/console_output.hpp`

**Зачем**: в системе 10 GPU — без порядка вывод в std::cout перемешается. ConsoleOutput использует фоновый поток + очередь (AsyncServiceBase), гарантируя порядок строк.

**Формат вывода**: `[HH:MM:SS.mmm] [INF] [GPU_00] [ModuleName] сообщение`

```cpp
auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
con.Print("Текст\n");
con.Print("Значение: %d\n", value);   // printf-style
con.PrintWarning(gpu_id, "Module", "предупреждение");
con.PrintError(gpu_id, "Module", "ошибка");
con.PrintDebug(gpu_id, "Module", "отладка");

// ⚠️ ПРАВИЛО: ConsoleOutput — ЕДИНСТВЕННЫЙ способ вывода!
// Нельзя: std::cout << ... или printf(...)
```

### GPUProfiler — профилирование (Singleton)

**Файл**: `services/gpu_profiler.hpp`

Асинхронный сборщик статистики GPU-операций. `Record()` — неблокирующий (помещает в очередь), агрегация (min/max/avg/count) в фоновом потоке.

**⚠️ ПРАВИЛО**: Вывод профилирования — ТОЛЬКО через `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`. **ЗАПРЕЩЕНО** вручную делать `GetStats()` + цикл + `con.Print`.

**⚠️ ПРАВИЛО**: Обязательно вызывать `SetGPUInfo()` **ДО** `Start()`, иначе в отчёте «GPU -1: Unknown».

```cpp
auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();

// 1. Передать info о GPU (ДО Start!)
drv_gpu_lib::GPUReportInfo info;
info.gpu_name      = drv.GetDeviceInfo().name;
info.backend_type  = BackendType::OPENCL;
info.global_mem_mb = drv.GetDeviceInfo().global_memory_size / (1024*1024);
// info.drivers[...] = {...}  — опционально
profiler.SetGPUInfo(0, info);

// 2. Запустить
profiler.Start();

// 3. Запись событий (из модулей, неблокирующе)
profiler.Record(gpu_id, "MyModule", "KernelName", opencl_prof_data);
profiler.Record(gpu_id, "MyModule", "KernelName", rocm_prof_data);

// 4. Вывод
profiler.PrintReport();                                     // в консоль
profiler.ExportMarkdown("Results/Profiler/report.md");      // MD файл
profiler.ExportJSON("Results/Profiler/report.json");        // JSON

// 5. Остановить
profiler.Stop();
profiler.Reset();  // сбросить накопленную статистику
```

### ServiceManager — управление сервисами (Singleton)

**Файл**: `services/service_manager.hpp`

Централизованный Init/Start/Stop всех сервисов.

```cpp
auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
sm.InitializeFromConfig("configGPU.json");  // загрузить конфиг
sm.StartAll();                              // запустить ConsoleOutput + GPUProfiler
// ... работа ...
sm.StopAll();                              // остановить все сервисы
```

### AsyncServiceBase\<T\> — база для сервисов (Template Method)

**Файл**: `services/async_service_base.hpp`

Шаблон: фоновый поток + lock-free очередь + неблокирующий `Enqueue()`. ConsoleOutput и GPUProfiler наследуются от него.

### GpuBenchmarkBase — база для бенчмарков

**Файл**: `services/gpu_benchmark_base.hpp`

Template Method Pattern: warmup(5) → измерение(20) → сбор статистики через GPUProfiler.

```cpp
class MyBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
protected:
  void ExecuteKernel() override {
    // warmup (результат не записывается)
    module_.Process(input_, output_);
  }
  void ExecuteKernelTimed() override {
    cl_event ev;
    module_.Process(input_, output_, &ev);
    RecordEvent("Process", ev);  // зарегистрировать в профилировщике
  }
};

MyBenchmark bench(backend, "MyModule");
bench.Run();    // 5 warmup + 20 измерений
bench.Report(); // PrintReport() + ExportMarkdown() + ExportJSON()
```

### BatchManager — обработка больших данных

**Файл**: `services/batch_manager.hpp`

Разбивает большой входной набор данных на батчи, обрабатывает с учётом доступной VRAM и размера batch.

### KernelCacheService — кэш JIT-компиляции

**Файл**: `services/kernel_cache_service.hpp`

Кэш бинарных opencl/hiprtc программ на диске. Первый запуск: компиляция (~50 мс). Последующие: загрузка кэша (~1 мс).

---

## 7. Multi-GPU

### GPUManager

**Файл**: `DrvGPU/include/gpu_manager.hpp` (header-only)

```cpp
drv_gpu_lib::GPUManager manager;

// Инициализировать все доступные GPU
manager.InitializeAll(BackendType::OPENCL);

// Или только указанные
manager.InitializeSpecific(BackendType::OPENCL, {0, 1, 2});

// Доступ к конкретному GPU
DrvGPU& gpu0 = manager.GetGPU(0);
DrvGPU& gpu1 = manager.GetGPU(1);

// Стратегии балансировки
DrvGPU& gpu = manager.GetNextGPU();        // Round Robin
DrvGPU& gpu = manager.GetLeastLoadedGPU(); // Least Loaded

// Информация
int count = GPUManager::GetAvailableGPUCount(BackendType::OPENCL);
```

### Стратегии балансировки (LoadBalancingStrategy)

**Файл**: `common/load_balancing.hpp`

| Стратегия | Когда использовать |
|-----------|-------------------|
| `ROUND_ROBIN` | Равномерная нагрузка, все задачи одинаковые |
| `LEAST_LOADED` | Разные задачи, нужен мониторинг загрузки |
| `MANUAL` | Ручной выбор GPU для каждой задачи |
| `FASTEST_FIRST` | Всегда выбирать GPU с наибольшей производительностью |

---

## 8. External Context Integration

Позволяет использовать DrvGPU с уже существующим контекстом (hipBLAS, hipFFT, внешний OpenCL).

### DrvGPU static factory methods

```cpp
// OpenCL External
auto gpu = DrvGPU::CreateFromExternalOpenCL(
    0,                  // device_index (для логирования)
    cl_context,         // cl_context — НЕ освобождается DrvGPU
    cl_device_id,       // cl_device_id
    cl_command_queue    // cl_command_queue
);
// ⚠️ gpu.Initialize() — НЕ вызывать, уже инициализирован!

// ROCm External (ENABLE_ROCM=1)
hipStream_t stream;
hipStreamCreate(&stream);
auto gpu = DrvGPU::CreateFromExternalROCm(0, stream);

// Hybrid External (OpenCL + ROCm одновременно)
auto gpu = DrvGPU::CreateFromExternalHybrid(
    0, cl_ctx, cl_dev, cl_q, hip_stream
);
```

**Ключевой механизм ownership**: `owns_resources_=false` → при `Cleanup()` backend только обнуляет указатели, не вызывает Release/Destroy. Caller владеет и освобождает хэндлы.

### Три паттерна использования

```
1. Стандартный (DrvGPU создаёт всё сам):
   DrvGPU gpu(OPENCL, 0);
   gpu.Initialize();        → owns_resources_=true
   // Деструктор: освободит cl_context, cl_queue

2. External OpenCL (наш код встраивается в чужой контекст):
   DrvGPU gpu = DrvGPU::CreateFromExternalOpenCL(0, ctx, dev, q);
   // owns_resources_=false → не освободит чужой контекст

3. External ROCm (разделяем hipStream с hipBLAS):
   DrvGPU gpu = DrvGPU::CreateFromExternalROCm(0, hipStream_t);
   // owns_resources_=false → не вызовет hipStreamDestroy
```

---

## 9. C4 Диаграммы

**Детальные диаграммы**: [Doc_Addition/DrvGPU_Design_C4.md](../../Doc_Addition/DrvGPU_Design_C4.md)

### C1 — System Context (упрощённый)

```
┌───────────────────────────────────────────────────────┐
│                  DSP-GPU System                     │
│                                                       │
│  ┌────────────────────────────────────────────────┐   │
│  │                  DrvGPU                        │   │
│  │  Core: DrvGPU / GPUManager / ModuleRegistry   │   │
│  │  Backends: OpenCL / ROCm / Hybrid             │   │
│  │  Services: ConsoleOutput / GPUProfiler        │   │
│  │  Memory: GPUBuffer<T> / MemoryManager         │   │
│  └────────────────────────────────────────────────┘   │
│           │ IBackend*                                  │
│           ▼                                           │
│  ┌────────────────────────────────────────────────┐   │
│  │ Modules (signal_gen, fft, statistics, …)       │   │
│  └────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
        │ OpenCL API      │ HIP/ROCm API
        ▼                 ▼
  AMD / NVIDIA / Intel GPU
```

### C2 — Containers

```
DrvGPU Container
│
├── IBackend (interface)
│   ├── OpenCLBackend ──── OpenCLCore ──── cl_context / cl_device / cl_queue
│   ├── ROCmBackend  ──── ROCmCore   ──── hipDevice / hipStream_t
│   └── HybridBackend ─── OpenCLCore + ROCmCore + ZeroCopyBridge
│
├── MemoryManager
│   ├── GPUBuffer<T>  [owning, RAII]
│   ├── HIPBuffer<T>  [non-owning]
│   └── SVMBuffer     [shared virtual memory]
│
└── Services
    ├── ConsoleOutput  ← AsyncServiceBase<ConsoleMessage>
    ├── GPUProfiler    ← AsyncServiceBase<ProfilingMessage>
    ├── ServiceManager ← управляет ConsoleOutput + GPUProfiler
    ├── BatchManager   ← обработка больших данных
    └── KernelCacheService ← кэш JIT-бинарей
```

### C3 — Components (ключевые зависимости)

```
Модуль (пример: heterodyne)
    │ принимает IBackend*
    ▼
IBackend* ──────────────────────────────────────────────
    │                                                   │
    │ Allocate()          Memory                        │ GetNativeQueue()
    │ Free()              Layer                         │ нативный API
    ▼                                                   ▼
MemoryManager                                  cl_command_queue / hipStream_t
    │
    ▼
GPUBuffer<T> (RAII)
```

---

## 10. C++ примеры

### Минимальный старт — один GPU, OpenCL

```cpp
#include <core/drv_gpu.hpp>
#include <core/services/service_manager.hpp>
#include <core/services/console_output.hpp>

// 1. Инициализация сервисов (один раз в main)
auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
sm.InitializeFromConfig("configGPU.json");
sm.StartAll();

// 2. Создать GPU
drv_gpu_lib::DrvGPU drv(drv_gpu_lib::BackendType::OPENCL, 0);
drv.Initialize();

// 3. Консоль (никакого std::cout!)
auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
con.Print("GPU: %s\n", drv.GetDeviceName().c_str());

// 4. Память
auto& mem = drv.GetMemoryManager();
auto buf = mem.CreateBuffer<float>(1024);
buf->Write(host_data, 1024);

// 5. IBackend — передать в модуль
drv_gpu_lib::IBackend* backend = &drv.GetBackend();
MyModule module(backend);
module.Process(buf->GetPtr(), 1024);

// 6. Синхронизация
drv.Synchronize();

// 7. Завершение (RAII — buf освобождается здесь, drv — при выходе из scope)
sm.StopAll();
```

### Multi-GPU

```cpp
drv_gpu_lib::GPUManager manager;
manager.InitializeAll(drv_gpu_lib::BackendType::OPENCL);

int n = drv_gpu_lib::GPUManager::GetAvailableGPUCount(BackendType::OPENCL);
for (int i = 0; i < n; ++i) {
    auto& gpu = manager.GetGPU(i);
    MyModule mod(&gpu.GetBackend());
    // ... запуск модулей параллельно на разных GPU
}
```

### ROCm GPU

```cpp
#if ENABLE_ROCM
drv_gpu_lib::DrvGPU drv(drv_gpu_lib::BackendType::ROCM, 0);
drv.Initialize();
// Всё то же самое — IBackend* работает одинаково
#endif
```

### Профилирование — полный паттерн

```cpp
#include <core/services/gpu_profiler.hpp>

auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();

// 1. SetGPUInfo ДО Start!
drv_gpu_lib::GPUReportInfo info;
info.gpu_name      = drv.GetDeviceInfo().name;
info.backend_type  = drv_gpu_lib::BackendType::OPENCL;
info.global_mem_mb = drv.GetDeviceInfo().global_memory_size / (1024*1024);

std::map<std::string, std::string> drv_info;
drv_info["driver_type"]    = "OpenCL";
drv_info["version"]        = drv.GetDeviceInfo().opencl_version;
drv_info["driver_version"] = drv.GetDeviceInfo().driver_version;
drv_info["vendor"]         = drv.GetDeviceInfo().vendor;
info.drivers.push_back(drv_info);

profiler.SetGPUInfo(0, info);  // ← ДО Start!

// 2. Запустить + выполнить операции
profiler.Start();
my_module.ProcessWithProfiling(input, output, &profiler);
profiler.Stop();

// 3. Экспорт (ТОЛЬКО через эти методы!)
profiler.PrintReport();
profiler.ExportMarkdown("Results/Profiler/2026-03-09.md");
profiler.ExportJSON("Results/Profiler/2026-03-09.json");
```

### External Context (ROCm + hipBLAS)

```cpp
// Создать hipStream извне (например, из hipBLAS handle)
hipStream_t stream;
hipStreamCreate(&stream);

// DrvGPU берёт stream, но не освобождает его
auto gpu = drv_gpu_lib::DrvGPU::CreateFromExternalROCm(0, stream);
// gpu.Initialize() — НЕ вызывать!

strategies::AntennaProcessor_v1 proc(&gpu.GetBackend(), cfg);
proc.process(d_S, d_W);

// Освободить stream — caller's responsibility
hipStreamDestroy(stream);
```

### Kernel Loader

**Файл**: `include/kernel_loader.hpp`

```cpp
#include <core/services/kernel_cache_service.hpp>

// Загрузить .cl файл из KERNELS_DIR (задаётся через CMake)
std::string src = drv_gpu_lib::LoadKernelFile("my_kernel.cl");

// Или загрузить несколько и склеить (для PRNG + kernel)
std::string combined = drv_gpu_lib::LoadKernelFile("prng.cl")
                     + drv_gpu_lib::LoadKernelFile("my_kernel.cl");

cl_program prog = clCreateProgramWithSource(..., combined.c_str(), ...);
```

---

## 11. Python API

Python-биндинги через pybind11. Основной класс — `GPUContext`.

```python
import dsp_core

# Создание контекста (device_index=0, backend="opencl")
ctx = dsp_core.GPUContext(0)

# Информация
print(ctx.device_name)          # "AMD Radeon RX 9070"
print(ctx.global_memory_mb)     # 16384

# Передаётся в конструкторы модулей
gen  = dsp_core.FormSignalGenerator(ctx)
fft  = dsp_core.FFTProcessor(ctx)
stat = dsp_core.StatisticsProcessor(ctx)
```

---

## 12. Тесты

### C++ тесты DrvGPU

**Каталог**: `DrvGPU/tests/`
**Точка входа**: `all_test.hpp`

| Файл | Что тестирует | Входные данные | Что проверяет |
|------|--------------|----------------|--------------|
| `single_gpu.hpp` | Базовая инициализация, выделение памяти, H2D/D2H | Буфер 1024 float | Initialize не бросает, Allocate/Free работают, MemcpyH2D+D2H даёт исходные данные с точностью 1e-6 |
| `test_services.hpp` | ConsoleOutput + GPUProfiler + ServiceManager | Несколько потоков пишут одновременно | Нет дедлока, нет перемешанных строк, профилировщик агрегирует корректно |
| `test_gpu_profiler.hpp` | GPUProfiler: Record, GetStats, ExportJSON | 100 событий по 10 мс каждое | avg≈10 мс, count=100, JSON валиден |
| `test_storage_services.hpp` | KernelCacheService: запись/чтение | OpenCL kernel source | После SaveKernel → LoadKernel содержимое идентично |
| `test_rocm_backend.hpp` | ROCmBackend: Initialize, Allocate, Memcpy | 4096 complex float | D2H после H2D совпадает, нет утечек |
| `test_zero_copy.hpp` | ZeroCopyBridge: OpenCL → ROCm без CPU | 1M float | D2H после ZeroCopy совпадает с оригиналом |
| `test_hybrid_backend.hpp` | HybridBackend: OpenCL + ROCm в одном DrvGPU | Простые операции | Оба backend'а инициализированы, ZeroCopyBridge работает |
| `test_rocm_external_context.hpp` | ROCmBackend::InitializeFromExternalStream | Внешний hipStream_t | `OwnsResources()==false`, не вызывает hipStreamDestroy |
| `test_hybrid_external_context.hpp` | HybridBackend::InitializeFromExternalContexts | Внешние cl_ctx + hipStream | Оба флага `owns_resources_=false` |
| `test_drv_gpu_external.hpp` | DrvGPU::CreateFromExternal*() factory | Внешние хэндлы | `IsInitialized()==true`, нет double-free при деструкции |
| `test_clmem_gpu_va_probe.cpp` | cl_mem ↔ GPU VA зондирование (DMA-buf export) | AMD GPU с RDNA4 | GPU Virtual Address cl_mem совпадает с ожидаемым диапазоном VRAM |
| `test_zerocopy_rdna4.cpp` | ZeroCopy RDNA4: cl_mem → hipStream_t без CPU copy | 1M float на RDNA4 gfx1201 | D2H после zero-copy совпадает с оригиналом, CPU не задействован |

#### Детали ключевых тестов

**single_gpu.hpp** — почему именно 1024 float:
Минимальный размер для проверки выравнивания (1024×4=4 КБ, типичная страница GPU). Выбран int-кратный для CL_MEM_ALLOC_HOST_PTR. Порог 1e-6: float32 precision, нет вычислений — только Copy, должно быть точным.

**test_rocm_external_context.hpp** — почему важен этот тест:
Ловит баг double-free: если `owns_stream_=true` по ошибке → `hipStreamDestroy` вызывается дважды (в DrvGPU и в caller) → segfault. Тест принудительно разрушает DrvGPU, потом явно вызывает `hipStreamDestroy` — без краша = OK.

**test_gpu_profiler.hpp** — почему avg≈10 мс:
Записывает синтетические события с фиксированным duration. Проверяет что агрегация корректна: sum/count = expected_avg. Порог ±0.1 мс: clock resolution фонового потока.

---

## 13. Файловая структура

```
DrvGPU/
├── include/                         # Публичные заголовки
│   ├── drv_gpu.hpp                  # DrvGPU (main facade, НЕ Singleton)
│   ├── gpu_manager.hpp              # GPUManager (multi-GPU, header-only)
│   └── module_registry.hpp          # ModuleRegistry
│
├── interface/                       # Интерфейсы
│   ├── i_backend.hpp                # IBackend (Bridge pattern)
│   ├── i_compute_module.hpp         # IComputeModule
│   ├── i_memory_buffer.hpp          # IMemoryBuffer
│   ├── i_logger.hpp                 # ILogger
│   ├── i_data_sink.hpp              # IDataSink
│   ├── input_data.hpp               # InputData<T>
│   ├── input_data_traits.hpp        # is_cpu_vector_v и др.
│   └── output_destination.hpp       # Цели вывода
│
├── common/                          # Общие типы
│   ├── backend_type.hpp             # BackendType enum (OPENCL/ROCM/HYBRID/AUTO)
│   ├── gpu_device_info.hpp          # GPUDeviceInfo
│   └── load_balancing.hpp           # LoadBalancingStrategy
│
├── backends/
│   ├── opencl/
│   │   ├── opencl_core.hpp/cpp      # Per-device OpenCL (НЕ Singleton v2.0)
│   │   ├── opencl_backend.hpp/cpp   # OpenCLBackend — реализация IBackend
│   │   ├── opencl_profiling.hpp     # FillOpenCLProfilingData helper
│   │   ├── opencl_export.hpp        # ZeroCopy DMA export
│   │   └── command_queue_pool.hpp/cpp # Пул cl_command_queue
│   ├── rocm/                        # ENABLE_ROCM=1
│   │   ├── rocm_core.hpp/cpp        # Per-device HIP (owns_stream_)
│   │   ├── rocm_backend.hpp/cpp     # ROCmBackend — реализация IBackend
│   │   └── zero_copy_bridge.hpp/cpp # OpenCL ↔ ROCm ZeroCopy (DMA-buf)
│   └── hybrid/
│       └── hybrid_backend.hpp/cpp   # HybridBackend = OpenCL + ROCm
│
├── memory/
│   ├── memory_type.hpp
│   ├── i_memory_buffer.hpp          # Интерфейс буфера
│   ├── gpu_buffer.hpp               # GPUBuffer<T> (RAII, owning, cl_mem)
│   ├── hip_buffer.hpp               # HIPBuffer<T> (non-owning, ROCm)
│   ├── svm_buffer.hpp               # SVMBuffer (Shared Virtual Memory)
│   ├── svm_capabilities.hpp
│   ├── memory_manager.hpp/cpp       # MemoryManager (allocation tracking)
│   └── external_cl_buffer_adapter.hpp # Адаптер внешних cl_mem
│
├── services/
│   ├── async_service_base.hpp       # AsyncServiceBase<T> (Template Method)
│   ├── console_output.hpp           # ConsoleOutput (Singleton)
│   ├── gpu_profiler.hpp             # GPUProfiler (Singleton)
│   ├── service_manager.hpp          # ServiceManager (Singleton)
│   ├── gpu_benchmark_base.hpp       # GpuBenchmarkBase (Template Method)
│   ├── profiling_types.hpp          # ProfilingMessage, OpenCLProfilingData
│   ├── profiling_stats.hpp          # EventStats, ModuleStats, GPUReportInfo
│   ├── batch_manager.hpp/cpp        # BatchManager
│   ├── filter_config_service.hpp/cpp # FilterConfigService
│   ├── kernel_cache_service.hpp/cpp  # KernelCacheService (JIT cache)
│   └── storage/
│       ├── i_storage_backend.hpp
│       └── file_storage_backend.hpp/cpp
│
├── config/
│   ├── gpu_config.hpp/cpp           # GPUConfig (configGPU.json)
│   └── config_types.hpp             # GPUConfigEntry и др.
│
├── logger/
│   ├── logger.hpp/cpp               # Logger (per-GPU фасад)
│   ├── default_logger.hpp/cpp       # DefaultLogger (plog)
│   └── config_logger.hpp/cpp        # ConfigLogger (пути, Singleton)
│
└── tests/
    ├── all_test.hpp                  # Точка входа тестов
    ├── single_gpu.hpp
    ├── test_services.hpp
    ├── test_gpu_profiler.hpp
    ├── test_storage_services.hpp
    ├── test_rocm_backend.hpp
    ├── test_zero_copy.hpp
    ├── test_hybrid_backend.hpp
    ├── test_rocm_external_context.hpp
    ├── test_hybrid_external_context.hpp
    ├── test_drv_gpu_external.hpp
    ├── test_clmem_gpu_va_probe.cpp      # RDNA4: cl_mem GPU VA probe
    ├── test_zerocopy_rdna4.cpp          # RDNA4: ZeroCopy без CPU round-trip
    └── example_external_context_usage.hpp

include/
└── kernel_loader.hpp                 # LoadKernelFile() — загрузка .cl файлов

configGPU.json                        # Конфигурация GPU (device indexes, log paths)
```

---

## 14. Ссылки

- [Quick.md](Quick.md) — шпаргалка и быстрый старт
- [API.md](API.md) — полный API reference
- [Architecture.md](Architecture.md) — слои и зависимости (предыдущий формат)
- [Memory.md](Memory.md) — детали системы памяти
- [OpenCL.md](OpenCL.md) — детали OpenCL бэкенда
- [Command.md](Command.md) — CommandQueuePool
- [Classes.md](Classes.md) — полный справочник классов
- [Doc_Addition/DrvGPU_Design_C4.md](../../Doc_Addition/DrvGPU_Design_C4.md) — C4 Model диаграммы
- [Examples/GPUProfiler_SetGPUInfo.md](../../Examples/GPUProfiler_SetGPUInfo.md) — паттерн SetGPUInfo
- [Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md](../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) — HIP/ROCm оптимизации

---

*Создано: 2026-03-09 | Обновлено: 2026-03-28 | Автор: Кодо*
