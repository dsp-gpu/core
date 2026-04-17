# DrvGPU — Краткий справочник

> Ядро DSP-GPU: единая точка управления GPU (OpenCL / ROCm / Hybrid)

**Namespace**: `drv_gpu_lib` | **Каталог**: `DrvGPU/`

---

## Концепция — зачем и что это такое

**Зачем нужен DrvGPU?**
Все модули библиотеки (FFT, фильтры, гетеродин, статистика…) должны работать одновременно на AMD (ROCm), NVIDIA/Intel (OpenCL) и в тестах — на любом GPU. DrvGPU — единственная точка, которая знает о платформе. Модули принимают только `IBackend*` и больше ничего не знают.

**Аналогия**: DrvGPU — это «системная шина» GPU. Все вычислительные модули — «платы расширения», которые вставляются в эту шину не зная, какой GPU стоит внутри.

---

### DrvGPU — один GPU

**Что делает**: создаёт бэкенд (OpenCL/ROCm/Hybrid), управляет памятью, предоставляет `IBackend*` для модулей. RAII — сам освобождает ресурсы.

**Когда брать**: всегда — это главный класс. Не Singleton, можно создать несколько (один на GPU).

```cpp
drv_gpu_lib::DrvGPU drv(drv_gpu_lib::BackendType::OPENCL, 0);
drv.Initialize();
IBackend* backend = &drv.GetBackend();  // передать в модуль
```

---

### GPUManager — несколько GPU

**Что делает**: создаёт и хранит несколько `DrvGPU`, выдаёт GPU по стратегии балансировки.

**Когда брать**: если в системе >1 GPU и нужно распределять нагрузку.

```cpp
drv_gpu_lib::GPUManager mgr;
mgr.InitializeAll(BackendType::OPENCL);
DrvGPU& gpu = mgr.GetNextGPU();  // round-robin
```

---

### IBackend — абстракция платформы

**Что делает**: интерфейс для всех GPU-операций (выделение памяти, копирование, синхронизация, нативные хэндлы).

**Когда брать**: передать указатель в конструктор модуля. Не создавать напрямую — получать из `DrvGPU`.

---

### GPUBuffer\<T\> — типобезопасный RAII буфер

**Что делает**: обёртка над GPU-памятью (cl_mem или hipMalloc). Освобождает память при выходе из scope.

**Когда брать**: для долгоживущих буферов (входные данные, результаты). Не для временных внутри kernels.

```cpp
auto buf = mem.CreateBuffer<complex<float>>(4096);
buf->Write(host_data, 4096);
auto result = buf->Read();
```

---

### HIPBuffer\<T\> — non-owning ROCm буфер

**Что делает**: не-владеющая обёртка над `void*` HIP-указателем. Не освобождает память.

**Когда брать**: когда память уже выделена через `backend->Allocate()` и управляется вручную.

---

### ConsoleOutput — мультиGPU-безопасный вывод

**Что делает**: Singleton. Принимает строки из любого потока, выводит в порядке поступления. Формат: `[HH:MM:SS.mmm] [INF] [GPU_00] сообщение`.

**Когда брать**: всегда вместо `std::cout` / `printf`. 10 GPU → без него вывод перемешается.

```cpp
auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
con.Print("Готово: %d\n", result);
```

---

### GPUProfiler — профилировщик GPU-операций

**Что делает**: Singleton. Собирает тайминги GPU-операций (неблокирующе), агрегирует (min/max/avg), экспортирует в Markdown/JSON.

**Когда брать**: при разработке и бенчмаркинге. В production — не включать (небольшой overhead на Record).

**⚠️ Два железных правила**:
1. `SetGPUInfo()` — ОБЯЗАТЕЛЬНО ДО `Start()`, иначе «GPU -1: Unknown»
2. Вывод — ТОЛЬКО через `PrintReport()` / `ExportMarkdown()` / `ExportJSON()`

---

### ServiceManager — жизненный цикл сервисов

**Что делает**: Init/Start/Stop для ConsoleOutput и GPUProfiler одним вызовом.

**Когда брать**: вызвать в `main()` один раз в начале и конце.

---

### GpuBenchmarkBase — база для бенчмарков

**Что делает**: шаблон (Template Method): warmup(5) → измерение(20) → GPUProfiler.

**Когда брать**: наследоваться при написании бенчмарков модулей. Не изобретать велосипед.

---

### External Context — встраивание в чужой контекст

**Что делает**: static factory методы `CreateFromExternalOpenCL/ROCm/Hybrid()`. DrvGPU использует чужие хэндлы, но НЕ освобождает их.

**Когда брать**: когда DrvGPU должен работать в уже существующем OpenCL/HIP контексте (hipBLAS, hipFFT, внешний OpenCL).

---

## Правила, которые нельзя нарушать

| ❌ НЕЛЬЗЯ | ✅ НУЖНО |
|-----------|----------|
| `std::cout`, `printf` | `ConsoleOutput::GetInstance().Print(...)` |
| `profiler.GetStats()` + цикл + Print | `profiler.PrintReport()` / `ExportMarkdown()` / `ExportJSON()` |
| `profiler.Start()` без `SetGPUInfo()` | `SetGPUInfo(gpu_id, info)` ДО `Start()` |
| Прямо работать с `cl_context` в модулях | Только через `IBackend*` |
| `gpu.Initialize()` после `CreateFromExternal*()` | `CreateFromExternal*()` уже возвращает готовый DrvGPU |

---

## Выбор бэкенда

| Ситуация | BackendType |
|----------|------------|
| AMD/NVIDIA/Intel, Windows | `OPENCL` |
| AMD GPU, нужен hipBLAS/hipFFT | `ROCM` (ENABLE_ROCM=1, Linux) |
| AMD GPU, смешать cl_mem + hipStream | `OPENCL_AND_ROCM` (HybridBackend) |
| Не знаю, выбери сам | `AUTO` |

---

## Полный старт за 10 строк

```cpp
#include <core/drv_gpu.hpp>
#include <core/services/service_manager.hpp>
#include <core/services/console_output.hpp>
using namespace drv_gpu_lib;

auto& sm = ServiceManager::GetInstance();
sm.InitializeFromConfig("configGPU.json");
sm.StartAll();

DrvGPU drv(BackendType::OPENCL, 0);
drv.Initialize();

auto& con = ConsoleOutput::GetInstance();
con.Print("GPU готов: %s\n", drv.GetDeviceName().c_str());

auto buf = drv.GetMemoryManager().CreateBuffer<float>(4096);
buf->Write(host_data, 4096);
// ... передать &drv.GetBackend() в модуль ...
drv.Synchronize();
sm.StopAll();
```

---

## Важные нюансы

| # | Нюанс |
|---|-------|
| ⚠️ | `DrvGPU` — **НЕ Singleton**! Один экземпляр на GPU. Для multi-GPU → `GPUManager` |
| ⚠️ | `OpenCLCore` v2.0 — **НЕ Singleton**! Per-device (важно для Multi-GPU) |
| ⚠️ | `GPUBuffer<T>` — запрещено копирование, только `std::move` |
| ⚠️ | `HIPBuffer<T>` — non-owning, не освобождает память — помни освободить вручную |
| ⚠️ | Логи в `Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log` (per-GPU через plog) |
| ⚠️ | `configGPU.json` — пути логов, индексы устройств, настройки профилировщика |

---

## Ссылки

- [Full.md](Full.md) — полная документация с C4 диаграммами и таблицей тестов
- [API.md](API.md) — все сигнатуры с примерами вызовов
- [Architecture.md](Architecture.md) — слои и зависимости
- [Memory.md](Memory.md) — GPUBuffer, SVMBuffer, MemoryManager
- [OpenCL.md](OpenCL.md) — OpenCL бэкенд
- [Classes.md](Classes.md) — полный справочник классов
- [Examples/GPUProfiler_SetGPUInfo.md](../../Examples/GPUProfiler_SetGPUInfo.md) — паттерн SetGPUInfo

---

*Обновлено: 2026-03-28*
