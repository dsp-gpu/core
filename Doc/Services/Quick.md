# DrvGPU Services — Краткий справочник

> ConsoleOutput, GPUProfiler, ServiceManager, KernelCacheService, FilterConfigService

---

## Сервисы

| Сервис | Назначение |
|--------|------------|
| **ServiceManager** | Init/Start/Stop всех сервисов (из configGPU.json) |
| **ConsoleOutput** | Потокобезопасный stdout для multi-GPU |
| **GPUProfiler** | Сбор статистики, экспорт JSON/MD |
| **KernelCacheService** | On-disk кэш скомпилированных kernel (.cl + binary + manifest) |
| **FilterConfigService** | Сохранение/загрузка конфигов фильтров (FIR/IIR коэффициенты) |
| **IStorageBackend** | Абстракция хранения (FileStorageBackend → SQLite) |

---

## Быстрый старт

### ServiceManager

```cpp
#include <core/services/service_manager.hpp>

auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
sm.InitializeFromConfig("configGPU.json");  // или InitializeDefaults()
sm.StartAll();
// ... работа GPU ...
sm.StopAll();
```

### ConsoleOutput

```cpp
#include <core/services/console_output.hpp>

auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
con.Print(0, "AntennaFFT", "Processing...");       // INFO
con.PrintWarning(1, "Memory", "High usage: 90%");  // WARNING
con.PrintError(0, "Backend", "hipMalloc failed");  // ERROR → stderr
con.PrintSystem("Main", "App started");            // SYSTEM (без GPU)
```

### GPUProfiler

```cpp
#include <core/services/gpu_profiler.hpp>

auto& prof = drv_gpu_lib::GPUProfiler::GetInstance();
// SetGPUInfo ДО StartAll():
drv_gpu_lib::GPUReportInfo info;
info.gpu_name = "Radeon RX 9070 XT";
info.global_mem_mb = 16384;
prof.SetGPUInfo(0, info);

// После StartAll() из модулей:
prof.Record(0, "AntennaFFT", "FFT_Execute", opencl_data);

// Вывод:
prof.PrintReport();
prof.ExportJSON("Results/Profiler/2026-03-02.json");
```

---

### KernelCacheService

```cpp
#include <core/services/kernel_cache_service.hpp>

drv_gpu_lib::KernelCacheService cache("modules/signal_generators/kernels");

// Save
cache.Save("my_kernel", cl_source, binary, "params", "comment");

// Load (binary fast path или source fallback)
auto entry = cache.Load("my_kernel");
if (entry.has_binary())
  LoadFromBinary(entry.binary);
else
  LoadFromSource(entry.source);

// List
auto names = cache.ListKernels();
```

### FilterConfigService

```cpp
#include <core/services/filter_config_service.hpp>

drv_gpu_lib::FilterConfigService svc("Results/FilterConfigs");

drv_gpu_lib::FilterConfigData data;
data.type = "fir";
data.coefficients = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};

svc.Save("lp_5000", data);
auto loaded = svc.Load("lp_5000");
```

---

## Структура на диске

```
base_dir/
├── name.cl              # KernelCacheService: source
├── bin/
│   └── name_opencl.bin  # binary (или _rocm.hsaco)
├── manifest.json        # индекс кернелов
└── filters/            # FilterConfigService
    └── lp_5000.json
```

---

## Ссылки

- [Full](Full.md) — полное описание, API, тесты

---

*Обновлено: 2026-03-02*
