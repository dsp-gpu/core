# DrvGPU Services — Полная документация

> ConsoleOutput, GPUProfiler, ServiceManager, AsyncServiceBase, KernelCacheService, FilterConfigService, IStorageBackend

**Namespace**: `drv_gpu_lib`
**Каталог**: `DrvGPU/services/`
**Зависимости**: std::filesystem (C++17), nlohmann_json (FilterConfigService)

---

## Содержание

1. [Обзор](#1-обзор)
2. [AsyncServiceBase](#2-asyncservicebase)
3. [ConsoleOutput](#3-consoleoutput)
4. [GPUProfiler](#4-gpuprofiler)
5. [ServiceManager](#5-servicemanager)
6. [IStorageBackend и FileStorageBackend](#6-istoragebackend-и-filestoragebackend)
7. [KernelCacheService](#7-kernelcacheservice)
8. [FilterConfigService](#8-filterconfigservice)
9. [Потребители](#9-потребители)
10. [Структура файлов](#10-структура-файлов)
11. [Тесты](#11-тесты)
12. [Ссылки](#12-ссылки)

---

## 1. Обзор

DrvGPU Services — инфраструктура I/O и сервисов модулей:

| Сервис | Назначение | Паттерн |
|--------|------------|---------|
| **AsyncServiceBase\<T\>** | Базовый шаблон: фоновый поток + очередь | Template |
| **ConsoleOutput** | Потокобезопасный stdout для multi-GPU | Singleton |
| **GPUProfiler** | Сбор статистики, экспорт JSON/MD | Singleton |
| **ServiceManager** | Init/Start/Stop всех сервисов | Singleton + Facade |
| **KernelCacheService** | Кэш скомпилированных kernels на диск | Service |
| **FilterConfigService** | Хранение конфигов FIR/IIR фильтров | Service |
| **FileStorageBackend** | key-value хранилище (файловая система) | Strategy |

**Принцип:** Все runtime-сервисы (ConsoleOutput, GPUProfiler) — Singleton + AsyncServiceBase.
Все storage-сервисы (KernelCache, FilterConfig) — отдельные объекты на base_dir.

---

## 2. AsyncServiceBase\<T\>

**Файл**: `DrvGPU/services/async_service_base.hpp`

### Назначение

Базовый шаблон для сервисов с фоновым рабочим потоком. Производный класс реализует `ProcessMessage()`.

### Архитектура

```
Любой поток GPU --> Enqueue(msg) --> [lock-free очередь] --> Worker Thread --> ProcessMessage()
```

### API

```cpp
template<typename T>
class AsyncServiceBase {
public:
    void Start();           // Запустить фоновый поток
    void Stop();            // Дождаться опустения очереди + join
    void Enqueue(T&& msg);  // Неблокирующая постановка
    void WaitEmpty();       // Дождаться обработки всех сообщений
    bool IsRunning() const;
    size_t GetProcessedCount() const;
    size_t GetQueueSize() const;

protected:
    virtual void ProcessMessage(const T& msg) = 0;
    virtual std::string GetServiceName() const = 0;
};
```

> ⚠️ **Важно**: Производный класс ОБЯЗАН вызывать `Stop()` в своём деструкторе ДО сброса vtable.

---

## 3. ConsoleOutput

**Файл**: `DrvGPU/services/console_output.hpp`

### Назначение

Потокобезопасный stdout для одновременного вывода с нескольких GPU.

### Архитектура

```
GPU Thread 0 --> Print(0, "FFT", "Done") --> Enqueue() --+
GPU Thread 1 --> Print(1, "FFT", "Done") --> Enqueue() --+--> [Очередь] --> Worker --> stdout
GPU Thread N --> Print(N, "FFT", "Done") --> Enqueue() --+
```

### API

```cpp
// Singleton
ConsoleOutput& ConsoleOutput::GetInstance();

// Методы вывода (неблокирующие)
void Print(int gpu_id, const std::string& module, const std::string& message);
void PrintWarning(int gpu_id, const std::string& module, const std::string& message);
void PrintError(int gpu_id, const std::string& module, const std::string& message);
void PrintDebug(int gpu_id, const std::string& module, const std::string& message);
void PrintSystem(const std::string& module, const std::string& message);  // gpu_id = -1

// Управление
void SetEnabled(bool enabled);
bool IsEnabled() const;
void SetGPUEnabled(int gpu_id, bool enabled);  // из configGPU.json is_console
bool IsGPUEnabled(int gpu_id) const;
```

### Формат вывода

```
[HH:MM:SS.mmm] [INF] [GPU_00] [ModuleName] сообщение
[HH:MM:SS.mmm] [WRN] [GPU_01] [FFT] Предупреждение
[HH:MM:SS.mmm] [ERR] [GPU_00] [Memory] Ошибка  →  stderr
[HH:MM:SS.mmm] [INF] [SYSTEM] [ServiceManager] All services started
```

### Пример

```cpp
#include <core/services/console_output.hpp>

auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
con.Print(0, "AntennaFFT", "Processing 1024 beams...");
con.PrintWarning(1, "Memory", "High usage: 90%");
con.PrintError(0, "Backend", "hipMalloc failed");
con.PrintSystem("Main", "Application started");
```

---

## 4. GPUProfiler

**Файл**: `DrvGPU/services/gpu_profiler.hpp`

### Назначение

Асинхронный сбор и агрегация статистики GPU операций. Поддерживает OpenCL и ROCm события.

### Архитектура

```
Module --> Record(gpu_id, "FFT", "Execute", opencl_data) --> Enqueue()
                                                                  |
                                                           [Worker Thread]
                                                                  |
                                                    Агрегация: min/max/avg/total
                                                    Хранение: stats_[gpu_id][module][event]
```

### API

```cpp
// Singleton
GPUProfiler& GPUProfiler::GetInstance();

// Запись событий
void Record(int gpu_id, const std::string& module, const std::string& event,
            const OpenCLProfilingData& data);
void Record(int gpu_id, const std::string& module, const std::string& event,
            const ROCmProfilingData& data);

// GPU Info (для шапки отчёта) — вызывать ДО Start()!
void SetGPUInfo(int gpu_id, const GPUReportInfo& info);
GPUReportInfo GetGPUInfo(int gpu_id) const;

// Статистика
std::map<std::string, ModuleStats> GetStats(int gpu_id) const;
std::map<int, std::map<std::string, ModuleStats>> GetAllStats() const;

// Вывод (ТОЛЬКО через эти методы!)
void PrintReport() const;       // Полный отчёт в stdout
void PrintSummary() const;      // Краткая сводка
bool ExportJSON(const std::string& path) const;
bool ExportMarkdown(const std::string& path) const;

// Управление
void Reset();
void SetEnabled(bool enabled);
bool IsEnabled() const;
void SetGPUEnabled(int gpu_id, bool enabled);
```

> ⚠️ **Вызывать `SetGPUInfo()` ПЕРЕД `Start()`**, иначе в отчёте «Unknown» и «нет информации о драйверах».
> ⚠️ **Вывод данных — ТОЛЬКО через `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`.**
> ❌ Запрещено: `GetStats()` + цикл + `con.Print` или `std::cout`.

### Структуры данных

```cpp
// profiling_types.hpp
struct ProfilingDataBase {
    uint64_t queued, submitted, started, ended, completed;
};
struct OpenCLProfilingData : ProfilingDataBase { /* cl_event поля */ };
struct ROCmProfilingData { /* HIP activity поля */ };

struct GPUReportInfo {
    std::string gpu_name;
    uint64_t global_mem_mb;
    std::vector<std::map<std::string, std::string>> drivers;
};

// profiling_stats.hpp
struct DetailedTimingStats { double GetAvgMs() const; /* sum, count */ };
struct EventStats {
    size_t total_calls;
    double total_time_ms, min_time_ms, max_time_ms;
    DetailedTimingStats queue_delay, submit_delay, exec_time, complete_delay;
    bool has_rocm_data;
    double GetAvgTimeMs() const;
};
struct ModuleStats {
    std::string module_name;
    std::map<std::string, EventStats> events;
    size_t GetRunCount() const;
    double GetAvgRunTimeMs() const;
};
```

---

## 5. ServiceManager

**Файл**: `DrvGPU/services/service_manager.hpp`

### Назначение

Единая точка управления жизненным циклом всех сервисов. Читает `configGPU.json`.

### Жизненный цикл

```
main():
  1. GPUManager::InitializeAll()        # создание GPU
  2. ServiceManager::InitializeFromConfig()  # настройка из JSON
  3. ServiceManager::StartAll()          # запуск фоновых потоков
  4. ... работа GPU ...
  5. ServiceManager::StopAll()           # остановка (деструктор тоже вызывает)
```

### API

```cpp
// Singleton
ServiceManager& ServiceManager::GetInstance();

// Инициализация
bool InitializeFromConfig(const std::string& config_file);  // из configGPU.json
void InitializeDefaults();  // без файла

// Жизненный цикл
void StartAll();   // Запустить ConsoleOutput + GPUProfiler
void StopAll();    // Остановить все (GPUProfiler → ConsoleOutput)
bool IsRunning() const;
bool IsInitialized() const;

// Удобный API
bool ExportProfiling(const std::string& file_path) const;  // → ExportJSON
void PrintProfilingSummary() const;
void PrintConfig() const;
std::string GetStatus() const;
```

### configGPU.json — поля для сервисов

| Поле | Тип | Сервис | Описание |
|------|-----|--------|----------|
| `is_console` | bool | ConsoleOutput | Включить вывод в консоль для GPU |
| `is_prof` | bool | GPUProfiler | Включить профилирование для GPU |
| `is_logger` | bool | Logger | Включить файловые логи для GPU |
| `log_level` | string | Logger | Уровень: "debug", "info", "warning" |

---

---

## 6. IStorageBackend и FileStorageBackend

### Интерфейс

```cpp
struct IStorageBackend {
  virtual void Save(const std::string& key, const std::vector<uint8_t>& data) = 0;
  virtual std::vector<uint8_t> Load(const std::string& key) const = 0;
  virtual std::vector<std::string> List(const std::string& prefix = "") const = 0;
  virtual bool Exists(const std::string& key) const = 0;
};
```

**Ключ** может содержать `/` — интерпретируется как поддиректория: `filters/lp_5000.json` → `base_dir/filters/lp_5000.json`.

### FileStorageBackend

- **Конструктор:** `FileStorageBackend(const std::string& base_dir)`
- **Save:** создаёт поддиректории при необходимости
- **Load:** выбрасывает `std::runtime_error` если ключ не найден
- Использует `std::filesystem` (C++17)

**Будущее:** SqliteStorageBackend с тем же интерфейсом.

---

## 7. KernelCacheService

### Назначение

On-disk кэш скомпилированных kernel. **Storage-agnostic** — не знает OpenCL, возвращает `{source, binary}`. Вызывающий создаёт `cl_program` через `clCreateProgramWithBinary` или `clCreateProgramWithSource`.

### API

```cpp
KernelCacheService(const std::string& base_dir,
                   BackendType backend_type = BackendType::OPENCL);

struct CacheEntry {
  std::string source;
  std::vector<uint8_t> binary;
  bool has_binary() const;
  bool has_source() const;
};

void Save(const std::string& name,
          const std::string& cl_source,
          const std::vector<uint8_t>& binary,
          const std::string& metadata = "",
          const std::string& comment = "");

CacheEntry Load(const std::string& name) const;  // throws if not found
std::vector<std::string> ListKernels() const;
std::string GetCacheDir() const;
std::string GetBinDir() const;
```

### Логика Save

1. `VersionOldFiles(name)` — если существуют `name.cl` и `name_*opencl.bin`, переименовать в `name_00.cl`, `name_opencl_00.bin`
2. Записать `base_dir/name.cl`
3. Записать `base_dir/bin/name_opencl.bin` (или `name_rocm.hsaco`)
4. Обновить `manifest.json`

### Логика Load

1. **Fast path:** `bin/name_opencl.bin` существует → прочитать binary + source из `name.cl`
2. **Fallback:** только `name.cl` → вернуть `{source, {}}` (вызывающий скомпилирует и вызовет Save)
3. Ничего нет → `std::runtime_error`

### ROCm

При `BackendType::ROCm` суффикс `_rocm.hsaco`. Логика идентична (байты в файл).

---

## 8. FilterConfigService

### Назначение

Сохранение/загрузка конфигураций фильтров (тип + коэффициенты). В отличие от kernel cache — хранит **данные фильтра**, не binary.

### FilterConfigData

```cpp
struct FilterConfigData {
  std::string name;
  std::string type;        // "fir" или "iir"
  std::string comment;
  std::string created;     // ISO 8601

  std::vector<float> coefficients;           // FIR: h[k]
  std::vector<std::array<float, 5>> sections; // IIR: [b0,b1,b2,a1,a2] per section
};
```

### API

```cpp
FilterConfigService(const std::string& base_dir);

void Save(const std::string& name, const FilterConfigData& data,
          const std::string& comment = "");
FilterConfigData Load(const std::string& name) const;
std::vector<std::string> ListFilters() const;
bool Exists(const std::string& name) const;
```

### Формат JSON

**FIR:**
```json
{
  "name": "lp_5000",
  "type": "fir",
  "comment": "Lowpass 5kHz",
  "created": "2026-02-21T12:00:00",
  "coefficients": [0.01, 0.02, ...]
}
```

**IIR:**
```json
{
  "name": "bp_1000",
  "type": "iir",
  "sections": [[b0,b1,b2,a1,a2], ...]
}
```

### Версионирование

При перезаписи того же имени → старый файл `lp_5000_00.json`, `lp_5000_01.json`, …

---

## 9. Потребители

| Модуль | Сервис | base_dir |
|--------|--------|----------|
| **FormScriptGenerator** | KernelCacheService | `modules/signal_generators/kernels` |
| **FirFilter** | KernelCacheService | `modules/filters/kernels` |
| **IirFilter** | KernelCacheService | `modules/filters/kernels` |
| **FirFilter / IirFilter** | FilterConfigService | configurable |

> Примечание: FilterConfigService использует FileStorageBackend как backend для хранения JSON.

---

## 10. Структура файлов

```
DrvGPU/services/
├── async_service_base.hpp        # Базовый шаблон сервисов
├── console_output.hpp            # ConsoleOutput (Singleton)
├── gpu_profiler.hpp              # GPUProfiler (Singleton)
├── service_manager.hpp           # ServiceManager (Singleton)
├── gpu_benchmark_base.hpp        # GpuBenchmarkBase
├── profiling_types.hpp           # ProfilingDataBase, OpenCLProfilingData, ROCmProfilingData, GPUReportInfo
├── profiling_stats.hpp           # ProfilingMessage, DetailedTimingStats, EventStats, ModuleStats
├── batch_manager.hpp/cpp         # BatchManager
├── kernel_cache_service.hpp/cpp
├── filter_config_service.hpp/cpp
└── storage/
    ├── i_storage_backend.hpp
    ├── file_storage_backend.hpp
    └── file_storage_backend.cpp
```

---

## 11. Тесты

**Файлы**: `DrvGPU/tests/`
**Вызов**: через `drvgpu_all_test::run()` в `main.cpp` (раскомментировать нужные).
**GPU не требуется**: `test_storage_services.hpp` — pure filesystem, `test_services.hpp` / `test_gpu_profiler.hpp` — CPU threads only.

---

### 11.1 `test_services.hpp` — многопоточные сервисы

---

**TestConsoleOutput**

*Что делает*: 8 потоков параллельно пишут по 50 сообщений через `ConsoleOutput::Print()` (итого 400 сообщений). Ожидает опустения очереди, проверяет атомарный счётчик `total == 400`.

*Почему важно*: `ConsoleOutput` — это `AsyncServiceBase<>` с одним worker-потоком и lock-free очередью. Если синхронизация очереди содержит data race, при 8 параллельных потоках часть сообщений будет потеряна или программа упадёт. Счётчик `std::atomic<int> total` инкрементируется в каждом потоке до постановки в очередь — поэтому `total == 400` означает «все Enqueue вызовы вернулись без потерь».

*Что НЕ проверяет*: порядок вывода в stdout (он недетерминирован при параллельной записи — это нормально).

---

**TestStressAsyncService**

*Что делает*: Создаёт `StressService` — производный от `AsyncServiceBase<StressMsg>`. 8 потоков параллельно отправляют по 1000 сообщений (итого 8000). Каждое сообщение содержит `time_point` создания. Worker-поток измеряет латентность `now - ts` для каждого сообщения и накапливает через `compare_exchange_weak`. Выводит: обработано/ожидалось, среднюю латентность (мкс), пропускную способность (msg/s).

*Почему важно*: Это stress-тест самого механизма `AsyncServiceBase<T>` — не конкретного сервиса. Проверяет:
1. Очередь не теряет сообщения при 8×1000 параллельных `Enqueue`.
2. `Stop()` корректно дожидается обработки всех сообщений перед join.
3. `compare_exchange_weak` для атомарного накопления латентности — без lock, но корректно.

*Числа*: Типичный результат на современном CPU — латентность ~5-50 мкс, пропускная способность ~200 000–800 000 msg/s.

---

**TestServiceManager**

*Что делает*: Полный жизненный цикл через `ServiceManager`:
`InitializeDefaults()` → `StartAll()` → `GetStatus()` → 40 записей в `GPUProfiler` (4 GPU × 10 событий) → `PrintProfilingSummary()` → `StopAll()`.

*Почему важно*: `ServiceManager::StopAll()` должен остановить сервисы в правильном порядке: сначала `GPUProfiler`, потом `ConsoleOutput` — иначе при завершении `GPUProfiler` попытается писать в уже остановленный `ConsoleOutput`. Тест проверяет что shutdown не вызывает deadlock и все записи профилировщика попадают в отчёт.

---

### 11.2 `test_gpu_profiler.hpp` — GPUProfiler

---

**TestGPUProfilerMultithread**

*Что делает*: 8 потоков × 50 записей в `GPUProfiler::Record()` (итого 400). Все пишут в модуль `"FFT"`, событие `"Execute"`, с данными `MakeOpenCLFromDurationMs(0.5 + i*0.1)` — синтетические длительности от 0.5 до 5.4 мс. После `join()` ждёт пустой очереди, считает агрегированные события через `GetAllStats()`.

*Почему важно*: `GPUProfiler::Record()` — неблокирующий `Enqueue`. Worker-поток агрегирует: `min/max/avg/total`. Если агрегация не thread-safe (например, worker читает из `stats_` без lock пока другой поток пишет), данные перемешаются. Проверка: `aggregated == 400` означает что ни одна запись не потеряна и не задвоена при агрегации.

---

**TestGPUProfilerLibraryDemo**

*Что делает*: Имитация "библиотеки" — несколько модулей и событий на 2 GPU:
- GPU 0, модуль `"AntennaFFT"`: 2× `SingleBatchFFT` (5 мс), 1× `Copy` (0.8 мс)
- GPU 1, модуль `"AntennaFFT"`: 1× `Copy` (0.8 мс)
- GPU 1, модуль `"ROCmKernels"`: 1× `vector_add` через `ROCmProfilingData` (1.2 мс)
- GPU 0, модуль `"TestModule"`: 2× `Upload` через `MakeOpenCLFromDurationMs` (0.25 и 0.31 мс)

Итого: ожидается 7 событий. Выводит `PrintSummary()`.

*Почему важно*: Это demo-тест, он показывает что `GPUProfiler` корректно работает с **двумя** бэкендами одновременно (`OpenCLProfilingData` и `ROCmProfilingData`) и раздельно агрегирует по GPU и модулю.

---

**TestGPUProfilerPrintReport**

*Что делает*: Самый реалистичный тест. Создаёт `GPUManager`, получает **реальный** `GPUReportInfo` с реального OpenCL устройства через `manager.GetGPUReportInfo(0)` и передаёт в `profiler.SetGPUInfo()`. Добавляет 320 событий с реалистичными временными метками в наносекундах:
- 100× `FFT_Execute` (11–15 мс каждое, с queue/submit delay)
- 100× `Padding_Kernel` (0.7–1.0 мс)
- 50× `HostToDevice` + 50× `DeviceToHost` (0.25 мс)
- 20× ROCm `MatrixMul` (2–2.5 мс, с `bytes` и `kernel_name`)

Вызывает `PrintReport()` и `ExportMarkdown("Results/Profiler/test_report.md")`.

*Почему важно*: Проверяет корректность форматирования полного отчёта (заголовок с реальными данными GPU и драйверов), агрегацию min/max/avg через DetailedTimingStats, и что `ExportMarkdown` создаёт файл без ошибок. 320 событий — достаточно чтобы проверить точность агрегации при большом числе записей.

*Важный нюанс*: `SetGPUInfo()` вызывается **до** первого `Record()` — иначе в шапке отчёта будет «Unknown GPU». В тесте это соблюдено явно.

---

### 11.3 `test_storage_services.hpp` — хранилища (без GPU)

---

**TestFileStorageBackend**

*Что делает*: Создаёт `FileStorageBackend` в temp-директории. Последовательно:
1. `Save("test/hello.bin", {0x48,0x65,0x6C,0x6C,0x6F})` → `Exists` → `Load` → byte-сравнение
2. Сохраняет ещё 2 файла в разных поддиректориях → `List("")` (≥3 ключа) → `List("test/")` (ровно 2)
3. `Exists("nonexistent/key.bin")` → должен вернуть false
4. `Save` того же ключа новыми данными → `Load` → проверка что данные обновились

*Почему именно такие данные*: Байты `{0x48,0x65,...}` = "Hello" в ASCII — легко читать в отладчике. `{0x42,0x79,0x65}` = "Bye" — явно отличается при перезаписи. Поддиректории (`test/`, `other/`) проверяют что ключ с `/` создаёт вложенные папки через `std::filesystem::create_directories`.

*Cleanup*: `fs::remove_all(temp_dir)` — тест не оставляет следов в файловой системе.

---

**TestKernelCacheService**

*Что делает*:
1. Сохраняет kernel: OpenCL source + бинарный blob `{0xDE,0xAD,0xBE,0xEF,...}`
2. Загружает обратно — проверяет `has_source()`, `has_binary()`, точное совпадение байт
3. `ListKernels()` — kernel присутствует в списке
4. **Версионирование**: повторный `Save` того же имени → старые файлы переименовываются в `test_kernel_00.cl`, `test_kernel_opencl_00.bin` → `Load` возвращает новую версию → `fs::exists("test_kernel_00.cl")` == true
5. `GetCacheDir()` / `GetBinDir()` — пути совпадают с ожидаемыми (кросс-платформенное сравнение через `lexically_normal()`)

*Почему магические байты*: `{0xDE,0xAD,0xBE,0xEF}` — известный паттерн, сразу видно в hex-дампе если Load вернул неверные данные. Source — минимальный валидный OpenCL kernel.

*Что проверяет версионирование*: В реальном использовании ядро пересобирается при изменении кода. Старый `.hsaco` должен сохраняться, чтобы не потерять рабочую версию. Тест подтверждает что `VersionOldFiles()` переименовывает, а не удаляет.

---

**TestFilterConfigService**

*Что делает*:
1. **FIR**: сохраняет `lp_5000` с коэффициентами `{0.1, 0.2, 0.4, 0.2, 0.1}` → `Exists` → `Load` → проверка типа, числа коэффициентов и значений (порог `1e-5f` — float32 round-trip через JSON)
2. **IIR**: сохраняет `bp_1000` с 2 секциями `[b0,b1,b2,a1,a2]` → `Load` → проверка числа секций и конкретных значений `b0=1.0`, `a1=-1.5`
3. `ListFilters()` → ровно 2 записи
4. **Версионирование**: перезаписывает FIR с 6 коэффициентами → `Load` возвращает новую версию → `fs::exists("filters/lp_5000_00.json")` == true
5. **Негативный тест**: `Load("nonexistent")` → обязан выбросить `std::runtime_error` (не вернуть пустой объект!)

*Почему float-порог `1e-5f`*: Коэффициенты сериализуются в JSON как decimal (например `0.1` → `"0.1"`). При парсинге обратно в float32 возникает ошибка представления ~`1e-7`. Порог `1e-5` достаточно строгий чтобы поймать ошибку порядка, но не ложно-отрицательный для float round-trip.

*Зачем негативный тест*: Убеждаемся что API сигнализирует об ошибке исключением, а не тихим возвратом пустого `FilterConfigData`. Код вызывающей стороны должен знать что `Load` may throw.

---

## 12. Ссылки

### Референсы

| Источник | Применение |
|----------|------------|
| [Intel COMPILER_CACHE](https://github.com/intel/compute-runtime/blob/master/programmers-guide/COMPILER_CACHE.md) | Hash = source+options+device, eviction LRU |
| [gnieto/cl-cache](https://github.com/gnieto/cl-cache) | Backend abstraction |
| Khronos clCreateProgramWithBinary | Binary device-specific |

---

*Обновлено: 2026-03-02*
