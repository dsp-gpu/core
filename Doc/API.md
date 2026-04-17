# DrvGPU — API Reference

> Полный справочник публичных классов и методов модуля `DrvGPU/`

**Namespace**: `drv_gpu_lib`

---

## Содержание

1. [DrvGPU](#1-drvgpu)
2. [IBackend](#2-ibackend)
3. [GPUManager](#3-gpumanager)
4. [MemoryManager](#4-memorymanager)
5. [GPUBuffer\<T\>](#5-gpubuffert)
6. [HIPBuffer\<T\>](#6-hipbuffert)
7. [ConsoleOutput](#7-consoleoutput)
8. [GPUProfiler](#8-gpuprofiler)
9. [ServiceManager](#9-servicemanager)
10. [GpuBenchmarkBase](#10-gpubenchmarkbase)
11. [BackendType / GPUDeviceInfo](#11-backendtype--gpudeviceinfo)
12. [Цепочки вызовов](#12-цепочки-вызовов)

---

## 1. DrvGPU

**Файл**: `DrvGPU/include/drv_gpu.hpp`

Главный фасад. НЕ Singleton — один экземпляр на GPU.

```cpp
namespace drv_gpu_lib {

class DrvGPU {
public:
  // ─── Конструкторы ─────────────────────────────────────────────────────────

  /** Создать DrvGPU для устройства device_index с указанным бэкендом */
  explicit DrvGPU(BackendType backend_type, int device_index = 0);

  ~DrvGPU();  // RAII: автоматически Cleanup() + освобождение backend

  // Move-only: копирование запрещено
  DrvGPU(DrvGPU&&) noexcept;
  DrvGPU& operator=(DrvGPU&&) noexcept;
  DrvGPU(const DrvGPU&) = delete;
  DrvGPU& operator=(const DrvGPU&) = delete;

  // ─── Static factory: External Context ─────────────────────────────────────

  /**
   * Создать из внешнего OpenCL контекста (НЕ освобождает хэндлы!)
   * Initialize() НЕ вызывать — уже инициализирован.
   */
  static DrvGPU CreateFromExternalOpenCL(
      int device_index,
      cl_context context,
      cl_device_id device,
      cl_command_queue queue);

  /** Создать из внешнего HIP stream (ENABLE_ROCM=1) */
  static DrvGPU CreateFromExternalROCm(int device_index, hipStream_t stream);

  /** Создать из внешних OpenCL + HIP ресурсов (HybridBackend) */
  static DrvGPU CreateFromExternalHybrid(
      int device_index,
      cl_context context,
      cl_device_id device,
      cl_command_queue queue,
      hipStream_t stream);

  // ─── Lifecycle ─────────────────────────────────────────────────────────────

  void Initialize();                ///< Инициализировать GPU (бросает runtime_error)
  void Cleanup();                   ///< Освободить ресурсы (вызывается в деструкторе)
  bool IsInitialized() const;       ///< true если Initialize() успешно выполнен

  // ─── Доступ к подсистемам ──────────────────────────────────────────────────

  IBackend&             GetBackend();           ///< GPU бэкенд (OpenCL/ROCm/Hybrid)
  const IBackend&       GetBackend() const;
  MemoryManager&        GetMemoryManager();     ///< Менеджер памяти
  const MemoryManager&  GetMemoryManager() const;
  ModuleRegistry&       GetModuleRegistry();    ///< Регистр compute модулей
  const ModuleRegistry& GetModuleRegistry() const;

  // ─── Информация ────────────────────────────────────────────────────────────

  GPUDeviceInfo GetDeviceInfo() const;           ///< Полная информация о GPU
  int           GetDeviceIndex() const;          ///< 0-based индекс устройства
  BackendType   GetBackendType() const;          ///< OPENCL / ROCM / HYBRID / AUTO
  std::string   GetDeviceName() const;           ///< "AMD Radeon RX 9070"
  void          PrintDeviceInfo() const;         ///< Вывод через ConsoleOutput

  // ─── Синхронизация ─────────────────────────────────────────────────────────

  void Synchronize();  ///< Ждать завершения всех GPU-операций
  void Flush();        ///< Сброс буфера команд без ожидания

  // ─── Статистика памяти ─────────────────────────────────────────────────────

  void        PrintStatistics() const;   ///< Вывод статистики аллокаций
  std::string GetStatistics() const;     ///< Строка со статистикой
  void        ResetStatistics();         ///< Сбросить счётчики
};

}  // namespace drv_gpu_lib
```

---

## 2. IBackend

**Файл**: `DrvGPU/interface/i_backend.hpp`

Абстрактный интерфейс. Bridge Pattern — все модули работают только через него.

```cpp
namespace drv_gpu_lib {

class IBackend {
public:
  virtual ~IBackend() = default;

  // ─── Lifecycle ─────────────────────────────────────────────────────────────
  virtual void Initialize(int device_index) = 0;
  virtual bool IsInitialized() const = 0;
  virtual void Cleanup() = 0;

  // ─── Ownership (для External Context) ─────────────────────────────────────
  /** true = backend владеет ресурсами и освободит при Cleanup()
   *  false = ресурсы внешние, backend только обнуляет указатели */
  virtual void SetOwnsResources(bool owns) = 0;
  virtual bool OwnsResources() const = 0;

  // ─── Информация об устройстве ──────────────────────────────────────────────
  virtual BackendType   GetType() const = 0;
  virtual GPUDeviceInfo GetDeviceInfo() const = 0;
  virtual int           GetDeviceIndex() const = 0;
  virtual std::string   GetDeviceName() const = 0;

  // ─── Нативные хэндлы ───────────────────────────────────────────────────────
  virtual void* GetNativeContext() const = 0;  ///< cl_context / hipCtx_t
  virtual void* GetNativeDevice() const = 0;   ///< cl_device_id / hipDevice_t
  virtual void* GetNativeQueue() const = 0;    ///< cl_command_queue / hipStream_t
  virtual MemoryManager* GetMemoryManager() { return nullptr; }

  // ─── Память ────────────────────────────────────────────────────────────────
  virtual void* Allocate(size_t size_bytes, unsigned int flags = 0) = 0;
  virtual void  Free(void* ptr) = 0;
  virtual void  MemcpyHostToDevice(void* dst, const void* src, size_t bytes) = 0;
  virtual void  MemcpyDeviceToHost(void* dst, const void* src, size_t bytes) = 0;
  virtual void  MemcpyDeviceToDevice(void* dst, const void* src, size_t bytes) = 0;

  // ─── Синхронизация ─────────────────────────────────────────────────────────
  virtual void Synchronize() = 0;
  virtual void Flush() = 0;

  // ─── Возможности ───────────────────────────────────────────────────────────
  virtual bool   SupportsSVM() const = 0;
  virtual bool   SupportsDoublePrecision() const = 0;
  virtual size_t GetMaxWorkGroupSize() const = 0;
  virtual size_t GetGlobalMemorySize() const = 0;  ///< bytes
  virtual size_t GetFreeMemorySize() const = 0;    ///< bytes (эвристика для AMD)
  virtual size_t GetLocalMemorySize() const = 0;   ///< bytes (LDS/shared mem)
};

}
```

---

## 3. GPUManager

**Файл**: `DrvGPU/include/gpu_manager.hpp` (header-only)

```cpp
namespace drv_gpu_lib {

class GPUManager {
public:
  GPUManager() = default;
  ~GPUManager() = default;

  // ─── Инициализация ─────────────────────────────────────────────────────────

  /** Инициализировать все доступные GPU */
  void InitializeAll(BackendType type);

  /** Инициализировать конкретные GPU по индексам */
  void InitializeSpecific(BackendType type, const std::vector<int>& indices);

  // ─── Доступ ────────────────────────────────────────────────────────────────

  DrvGPU& GetGPU(size_t index);               ///< По абсолютному индексу
  DrvGPU& GetNextGPU();                        ///< Round Robin
  DrvGPU& GetLeastLoadedGPU();                 ///< Least Loaded (по статистике)

  size_t GetGPUCount() const;                  ///< Число инициализированных GPU

  // ─── Static утилиты ────────────────────────────────────────────────────────

  /** Число доступных GPU (до инициализации) */
  static int GetAvailableGPUCount(BackendType type);

  /** Информация о всех GPU в виде строки */
  static std::string GetAllDevicesInfo(BackendType type);
};

}
```

**Стратегии балансировки** (`common/load_balancing.hpp`):

```cpp
enum class LoadBalancingStrategy {
  ROUND_ROBIN,    // Циклический выбор
  LEAST_LOADED,   // Наименее загруженный
  MANUAL,         // Ручной выбор через GetGPU(index)
  FASTEST_FIRST   // Сначала самый быстрый
};
```

---

## 4. MemoryManager

**Файл**: `DrvGPU/memory/memory_manager.hpp`

```cpp
namespace drv_gpu_lib {

class MemoryManager {
public:
  explicit MemoryManager(IBackend* backend);

  // ─── Типобезопасные буферы (предпочтительный способ) ──────────────────────

  /** Создать пустой буфер */
  template<typename T>
  std::shared_ptr<GPUBuffer<T>> CreateBuffer(size_t num_elements);

  /** Создать буфер и скопировать данные с CPU */
  template<typename T>
  std::shared_ptr<GPUBuffer<T>> CreateBuffer(const T* host_data, size_t num_elements);

  template<typename T>
  std::shared_ptr<GPUBuffer<T>> CreateBuffer(const std::vector<T>& host_data);

  // ─── Сырые аллокации ───────────────────────────────────────────────────────

  void* Allocate(size_t size_bytes);    ///< Выделить (backend->Allocate внутри)
  void  Free(void* ptr);                ///< Освободить

  // ─── Статистика ────────────────────────────────────────────────────────────

  size_t      GetAllocationCount() const;
  size_t      GetTotalAllocatedBytes() const;
  std::string GetStatistics() const;
  void        ResetStatistics();
};

}
```

---

## 5. GPUBuffer\<T\>

**Файл**: `DrvGPU/memory/gpu_buffer.hpp`

```cpp
namespace drv_gpu_lib {

template<typename T>
class GPUBuffer {
public:
  // Move-only (копирование запрещено)
  GPUBuffer(GPUBuffer&&) noexcept;
  GPUBuffer& operator=(GPUBuffer&&) noexcept;
  GPUBuffer(const GPUBuffer&) = delete;

  // ─── Запись / чтение ───────────────────────────────────────────────────────

  void Write(const T* host_data, size_t num_elements);
  void Write(const std::vector<T>& host_data);

  void            Read(T* host_data, size_t num_elements) const;
  std::vector<T>  Read() const;

  // ─── Копирование GPU→GPU ───────────────────────────────────────────────────

  void CopyFrom(const GPUBuffer<T>& other);

  // ─── Информация ────────────────────────────────────────────────────────────

  T*     GetPtr();             ///< Указатель на GPU-память (cl_mem* / void*)
  size_t GetNumElements() const;
  size_t GetSizeBytes() const;
  bool   IsValid() const;      ///< true если память выделена
};

}
```

---

## 6. HIPBuffer\<T\>

**Файл**: `DrvGPU/memory/hip_buffer.hpp` (ENABLE_ROCM=1)

Non-owning wrapper над `void*` HIP-памятью.

```cpp
namespace drv_gpu_lib {

template<typename T>
class HIPBuffer {
public:
  explicit HIPBuffer(void* ptr, size_t num_elements);
  // НЕ освобождает память!

  T*     GetPtr() const;
  size_t GetNumElements() const;
  size_t GetSizeBytes() const;
};

}
```

---

## 7. ConsoleOutput

**Файл**: `DrvGPU/services/console_output.hpp` (Singleton)

```cpp
namespace drv_gpu_lib {

class ConsoleOutput : public AsyncServiceBase<ConsoleMessage> {
public:
  static ConsoleOutput& GetInstance();

  // ─── Вывод ─────────────────────────────────────────────────────────────────

  /** printf-style, без gpu_id и module (быстрый способ) */
  void Print(const char* fmt, ...);

  /** С явным gpu_id и именем модуля */
  void Print(int gpu_id, const std::string& module, const std::string& message);
  void PrintWarning(int gpu_id, const std::string& module, const std::string& message);
  void PrintError  (int gpu_id, const std::string& module, const std::string& message);
  void PrintDebug  (int gpu_id, const std::string& module, const std::string& message);

  // ─── Управление ────────────────────────────────────────────────────────────

  void SetGPUEnabled(int gpu_id, bool enabled);  ///< Фильтр по GPU
  void Start();  ///< Запустить рабочий поток (вызывается ServiceManager)
  void Stop();   ///< Остановить поток
};

}
```

**Формат строки**: `[HH:MM:SS.mmm] [INF] [GPU_00] [ModuleName] сообщение`

**Уровни**: `INF` (Print), `WRN` (PrintWarning), `ERR` (PrintError), `DBG` (PrintDebug)

---

## 8. GPUProfiler

**Файл**: `DrvGPU/services/gpu_profiler.hpp` (Singleton)

```cpp
namespace drv_gpu_lib {

struct GPUReportInfo {
  std::string  gpu_name;        // "AMD Radeon RX 9070"
  BackendType  backend_type;    // OPENCL / ROCM / HYBRID
  size_t       global_mem_mb;   // 16384
  std::vector<std::map<std::string, std::string>> drivers;  // driver info
};

class GPUProfiler : public AsyncServiceBase<ProfilingMessage> {
public:
  static GPUProfiler& GetInstance();

  // ─── Setup (ДО Start!) ─────────────────────────────────────────────────────

  /** Передать информацию о GPU для заголовка отчёта.
   *  ОБЯЗАТЕЛЬНО вызвать ДО Start(), иначе в отчёте «GPU -1: Unknown» */
  void SetGPUInfo(int gpu_id, const GPUReportInfo& info);

  // ─── Lifecycle ─────────────────────────────────────────────────────────────

  void Start();   ///< Запустить фоновый поток агрегации
  void Stop();    ///< Остановить поток (завершить обработку очереди)
  void Reset();   ///< Сбросить накопленную статистику

  // ─── Запись событий (неблокирующая!) ──────────────────────────────────────

  /** OpenCL: 5 cl_profiling_info значений */
  void Record(int gpu_id, const std::string& module,
              const std::string& event, const OpenCLProfilingData& data);

  /** ROCm/HIP: hipEvent_t пара start/stop */
  void Record(int gpu_id, const std::string& module,
              const std::string& event, const ROCmProfilingData& data);

  // ─── Получение статистики ──────────────────────────────────────────────────

  /** Статистика по конкретному GPU.
   *  ⚠️ НЕ выводить вручную — использовать PrintReport()/ExportMarkdown() */
  std::map<std::string, ModuleStats> GetStats(int gpu_id) const;
  std::map<int, std::map<std::string, ModuleStats>> GetAllStats() const;

  // ─── Вывод (ТОЛЬКО эти методы!) ───────────────────────────────────────────

  void PrintReport() const;                           ///< В ConsoleOutput
  void PrintSummary() const;                          ///< Краткий вывод
  bool ExportMarkdown(const std::string& path) const; ///< .md файл
  bool ExportJSON(const std::string& path) const;     ///< .json файл

  // ─── Управление ────────────────────────────────────────────────────────────

  void SetEnabled(bool enabled);
  void SetGPUEnabled(int gpu_id, bool enabled);
};

// EventStats — агрегированная статистика одного события
struct EventStats {
  size_t count    = 0;
  double min_ms   = 0.0;
  double max_ms   = 0.0;
  double avg_ms   = 0.0;
  double total_ms = 0.0;
};

// ModuleStats — статистика модуля (map event→EventStats)
using ModuleStats = std::map<std::string, EventStats>;

}
```

---

## 9. ServiceManager

**Файл**: `DrvGPU/services/service_manager.hpp` (Singleton)

```cpp
namespace drv_gpu_lib {

class ServiceManager {
public:
  static ServiceManager& GetInstance();

  /** Загрузить конфиг (пути логов, gpu_ids, настройки профилировщика) */
  void InitializeFromConfig(const std::string& config_path);

  /** Запустить ConsoleOutput + GPUProfiler */
  void StartAll();

  /** Остановить все сервисы (ждать опустошения очередей) */
  void StopAll();
};

}
```

---

## 10. GpuBenchmarkBase

**Файл**: `DrvGPU/services/gpu_benchmark_base.hpp`

```cpp
namespace drv_gpu_lib {

class GpuBenchmarkBase {
public:
  explicit GpuBenchmarkBase(IBackend* backend, const std::string& module_name);
  virtual ~GpuBenchmarkBase() = default;

  /** Запустить бенчмарк: warmup_count раз + measure_count раз */
  void Run(int warmup_count = 5, int measure_count = 20);

  /** Вывести результаты (PrintReport + ExportMarkdown + ExportJSON) */
  void Report(const std::string& output_path = "Results/Profiler/");

protected:
  // Template Method — переопределить в наследнике:

  /** Warmup: запустить kernel без записи в профилировщик */
  virtual void ExecuteKernel() = 0;

  /** Measurement: запустить kernel + RecordEvent() */
  virtual void ExecuteKernelTimed() = 0;

  /** Зарегистрировать OpenCL событие */
  void RecordEvent(const std::string& event_name, cl_event event);

  /** Зарегистрировать ROCm события (start/stop) */
  void RecordEvent(const std::string& event_name,
                   hipEvent_t start, hipEvent_t stop);

  IBackend*   backend_;
  std::string module_name_;
};

}
```

---

## 11. BackendType / GPUDeviceInfo

**Файл**: `DrvGPU/common/backend_type.hpp`

```cpp
namespace drv_gpu_lib {

enum class BackendType : uint8_t {
  OPENCL          = 0,  // OpenCL 3.0 (все GPU, Windows + Linux)
  ROCM            = 1,  // ROCm/HIP (AMD GPU, Linux, ENABLE_ROCM=1)
  OPENCL_AND_ROCM = 2,  // HybridBackend
  AUTO            = 3   // Выбрать лучший автоматически
};

}
```

**Файл**: `DrvGPU/common/gpu_device_info.hpp`

```cpp
namespace drv_gpu_lib {

struct GPUDeviceInfo {
  std::string name;               // "AMD Radeon RX 9070"
  std::string vendor;             // "Advanced Micro Devices, Inc."
  std::string driver_version;     // "3593.0 (HSA1.1,LC)"
  std::string opencl_version;     // "OpenCL 3.0"
  size_t      global_memory_size; // bytes (16 * 1024 * 1024 * 1024 = 16 ГБ)
  size_t      local_memory_size;  // bytes (LDS размер)
  size_t      max_work_group_size;// 1024 / 256 и т.д.
  bool        supports_svm;
  bool        supports_double;
  int         device_index;
  BackendType backend_type;
};

}
```

---

## 12. Цепочки вызовов

### Стандартный старт + работа модуля

```cpp
// ── main.cpp ──────────────────────────────────────────────────────────────
#include <core/drv_gpu.hpp>
#include <core/services/service_manager.hpp>
#include <core/services/console_output.hpp>
#include <core/services/gpu_profiler.hpp>
using namespace drv_gpu_lib;

int main() {
  // 1. Сервисы
  auto& sm = ServiceManager::GetInstance();
  sm.InitializeFromConfig("configGPU.json");
  sm.StartAll();

  // 2. GPU
  DrvGPU drv(BackendType::OPENCL, 0);
  drv.Initialize();

  auto& con = ConsoleOutput::GetInstance();
  con.Print("[main] GPU: %s, %.1f ГБ\n",
            drv.GetDeviceName().c_str(),
            drv.GetDeviceInfo().global_memory_size / 1e9);

  // 3. Профилировщик
  auto& profiler = GPUProfiler::GetInstance();
  GPUReportInfo gpu_info;
  gpu_info.gpu_name      = drv.GetDeviceInfo().name;
  gpu_info.backend_type  = BackendType::OPENCL;
  gpu_info.global_mem_mb = drv.GetDeviceInfo().global_memory_size / (1024*1024);
  profiler.SetGPUInfo(0, gpu_info);  // ← ДО Start!
  profiler.Start();

  // 4. Буферы
  auto& mem = drv.GetMemoryManager();
  auto input  = mem.CreateBuffer<float>(n);
  auto output = mem.CreateBuffer<float>(n);
  input->Write(host_data, n);

  // 5. Модуль
  IBackend* backend = &drv.GetBackend();
  MyModule module(backend);
  module.Process(input->GetPtr(), output->GetPtr(), n);
  drv.Synchronize();

  auto result = output->Read();

  // 6. Результаты профилировщика
  profiler.Stop();
  profiler.PrintReport();
  profiler.ExportJSON("Results/Profiler/run.json");

  // 7. Завершение
  sm.StopAll();
  return 0;
}
```

### Multi-GPU round-robin

```cpp
GPUManager manager;
manager.InitializeAll(BackendType::OPENCL);

std::vector<std::unique_ptr<MyModule>> modules;
for (size_t i = 0; i < manager.GetGPUCount(); ++i) {
  modules.push_back(std::make_unique<MyModule>(&manager.GetGPU(i).GetBackend()));
}

// Распределить задачи
for (auto& task : tasks) {
  DrvGPU& gpu = manager.GetNextGPU();  // round-robin
  MyModule& mod = *modules[gpu.GetDeviceIndex()];
  mod.Process(task.input, task.output, task.size);
}
```

### External ROCm context (hipBLAS + DrvGPU)

```cpp
// hipBLAS уже создал stream
hipblasHandle_t blas_handle;
hipblasCreate(&blas_handle);
hipStream_t stream;
hipStreamCreate(&stream);
hipblasSetStream(blas_handle, stream);

// DrvGPU берёт stream но НЕ освобождает
auto gpu = DrvGPU::CreateFromExternalROCm(0, stream);
// IsInitialized() == true, Initialize() НЕ вызывать!

strategies::AntennaProcessorConfig cfg;
cfg.n_ant = 256; cfg.n_samples = 1'200'000;
strategies::AntennaProcessor_v1 proc(&gpu.GetBackend(), cfg);
auto result = proc.process(d_S, d_W);

// Caller освобождает сам:
hipStreamDestroy(stream);
hipblasDestroy(blas_handle);
```

### GpuBenchmarkBase

```cpp
class FFTBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  FFTBenchmark(IBackend* b) : GpuBenchmarkBase(b, "FFTProcessor"), proc_(b) {}

protected:
  void ExecuteKernel() override {
    proc_.Process(d_in_, d_out_, n_);   // warmup
  }
  void ExecuteKernelTimed() override {
    cl_event ev;
    proc_.Process(d_in_, d_out_, n_, &ev);
    RecordEvent("FFT_Execute", ev);
  }
private:
  FFTProcessorOpenCL proc_;
  void* d_in_;  void* d_out_;  size_t n_;
};

FFTBenchmark bench(&drv.GetBackend());
bench.Run(5, 20);                       // 5 warmup + 20 измерений
bench.Report("Results/Profiler/fft/");  // PrintReport + ExportJSON + ExportMarkdown
```

### Python API

```python
import dsp_core

# Создание контекста
ctx = dsp_core.GPUContext(0)                # device_index=0, OPENCL по умолчанию
ctx = dsp_core.GPUContext(0, backend="rocm") # ROCm

# Информация
print(ctx.device_name)          # "AMD Radeon RX 9070"
print(ctx.global_memory_mb)     # 16384

# Передаётся в конструкторы модулей
gen  = dsp_core.FormSignalGenerator(ctx)
fft  = dsp_core.FFTProcessor(ctx)
stat = dsp_core.StatisticsProcessor(ctx)

gen.set_params(fs=12e6, f0=2e6, antennas=8, points=4096)
data = gen.generate()  # np.ndarray [8, 4096] complex64
```

---

## См. также

- [Full.md](Full.md) — полная документация с C4 диаграммами
- [Quick.md](Quick.md) — концепция и быстрый старт
- [Architecture.md](Architecture.md) — слои и зависимости
- [Examples/GPUProfiler_SetGPUInfo.md](../../Examples/GPUProfiler_SetGPUInfo.md) — SetGPUInfo паттерн

---

*Обновлено: 2026-03-28 | Автор: Кодо*
