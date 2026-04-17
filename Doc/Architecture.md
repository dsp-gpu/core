# DrvGPU Архитектура проекта

## Оглавление

1. [Обзор архитектуры](#обзор-архитектуры)
2. [Структура директорий](#структура-директорий)
3. [Архитектурные слои](#архитектурные-слои)
4. [Паттерны проектирования](#паттерны-проектирования)
5. [Зависимости между компонентами](#зависимости-между-компонентами)
6. [Документация по разделам](#документация-по-разделам)

---

## Обзор архитектуры

DrvGPU — модульная библиотека для работы с GPU, предоставляющая единый интерфейс для различных бэкендов (OpenCL, ROCm, CUDA).

```
┌─────────────────────────────────────────────────────────────────┐
│                         DrvGPU Library                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Core Layer                             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │ DrvGPU   │ │ GPUManager│ │ Module   │ │ Memory   │    │  │
│  │  │          │ │          │ │ Registry │ │ Manager  │    │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘    │  │
│  └───────┼────────────┼────────────┼────────────┼───────────┘  │
│          │            │            │            │               │
│          └────────────┴─────┬──────┴────────────┘               │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Backend Abstraction Layer                    │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │                   IBackend Interface                │  │  │
│  │  └───────────────────────┬─────────────────────────────┘  │  │
│  │                          │                                 │
│  │    ┌─────────────────────┼─────────────────────┐          │
│  │    ▼                     ▼                     ▼          │
│  │  ┌──────────┐       ┌──────────┐       ┌──────────┐      │
│  │  │ OpenCL   │       │  ROCm    │       │  Hybrid  │      │
│  │  │ Backend  │       │  Backend │       │  Backend │      │
│  │  │ (ГОТОВ)  │       │ (ГОТОВ)  │       │(OpenCL+  │      │
│  │  └──────────┘       └──────────┘       │ ROCm)    │      │
│  │                                        └──────────┘      │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Common Services                         │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │ Console  │ │   GPU    │ │ Service  │ │  Batch   │    │  │
│  │  │ Output   │ │ Profiler │ │ Manager  │ │ Manager  │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │ Logger   │ │ Config   │ │ GPUDevice│ │ Load     │    │  │
│  │  │          │ │ Logger   │ │ Info     │ │ Balancing│    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Модули, использующие DrvGPU

Все модули библиотеки DSP-GPU работают через `IBackend*` из DrvGPU:

```
┌────────────────────────────────────────────────────────────────┐
│                     Modules Layer                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   │
│  │  Signal       │ │ FFT          │ │ FFT Maxima           │   │
│  │  Generators   │ │ Processor    │ │ (SpectrumMaxima      │   │
│  │  (signal_gen) │ │(fft_processor│ │  Finder)             │   │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   │
│  │  Statistics   │ │  Heterodyne  │ │  Filters             │   │
│  │  (statistics) │ │ (heterodyne) │ │  (filters)           │   │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘   │
│         │                │                     │               │
│         └────────────────┼─────────────────────┘               │
│                          │  IBackend*                          │
│                    ┌─────▼─────┐                               │
│                    │  DrvGPU   │                               │
│                    └───────────┘                               │
└────────────────────────────────────────────────────────────────┘
```

Подробнее о модулях: [`../Modules/`](../Modules/)

---

## Структура директорий

```
DSP-GPU/
├── DrvGPU/                       # Ядро библиотеки
│   ├── include/                      # Публичные заголовки (симлинки)
│   │   ├── drv_gpu.hpp
│   │   ├── gpu_manager.hpp
│   │   └── module_registry.hpp
│   │
│   ├── src/                          # Реализации (.cpp)
│   │   ├── drv_gpu.cpp
│   │   └── module_registry.cpp
│   │
│   ├── interface/                    # Интерфейсы
│   │   ├── i_backend.hpp             # Интерфейс бэкенда
│   │   ├── i_compute_module.hpp      # Интерфейс модуля
│   │   ├── i_memory_buffer.hpp       # Интерфейс буфера
│   │   ├── i_logger.hpp              # Интерфейс логера
│   │   ├── i_data_sink.hpp           # Sink для данных
│   │   ├── input_data.hpp            # InputData<T>
│   │   ├── input_data_traits.hpp     # is_cpu_vector_v и др.
│   │   └── output_destination.hpp    # Цели вывода
│   │
│   ├── common/                       # Общие типы и утилиты
│   │   ├── backend_type.hpp          # BackendType enum
│   │   ├── gpu_device_info.hpp       # GPUDeviceInfo
│   │   └── load_balancing.hpp        # LoadBalancingStrategy
│   │
│   ├── backends/                     # Бэкенды (см. OpenCL.md)
│   │   ├── opencl/
│   │   │   ├── opencl_core.cpp/hpp       # Низкоуровневые операции
│   │   │   ├── opencl_backend.cpp/hpp    # OpenCL реализация IBackend
│   │   │   ├── opencl_profiling.hpp      # FillOpenCLProfilingData
│   │   │   ├── opencl_export.hpp         # ZeroCopy экспорт (DMA, GpuVA)
│   │   │   └── command_queue_pool.cpp/hpp
│   │   ├── rocm/
│   │   │   ├── rocm_backend.cpp/hpp      # ROCm реализация IBackend ✅
│   │   │   ├── rocm_core.cpp/hpp         # Per-device HIP контекст ✅
│   │   │   └── zero_copy_bridge.cpp/hpp  # OpenCL↔ROCm ZeroCopy ✅
│   │   └── hybrid/
│   │       └── hybrid_backend.cpp/hpp    # Гибридный OpenCL+ROCm бэкенд ✅
│   │
│   ├── memory/                       # Управление памятью (см. Memory.md)
│   │   ├── memory_type.hpp
│   │   ├── i_memory_buffer.hpp
│   │   ├── gpu_buffer.hpp
│   │   ├── hip_buffer.hpp            # HIPBuffer (non-owning, ROCm only)
│   │   ├── memory_manager.cpp/hpp
│   │   ├── svm_buffer.hpp
│   │   ├── svm_capabilities.hpp
│   │   └── external_cl_buffer_adapter.hpp
│   │
│   ├── services/                     # Сервисы (см. Services/)
│   │   ├── async_service_base.hpp    # Базовый шаблон сервисов
│   │   ├── console_output.hpp        # ConsoleOutput (singleton)
│   │   ├── gpu_profiler.hpp          # GPUProfiler (singleton)
│   │   ├── service_manager.hpp       # ServiceManager (singleton)
│   │   ├── gpu_benchmark_base.hpp    # Базовый класс бенчмарков
│   │   ├── profiling_types.hpp       # ProfilingDataBase, OpenCLProfilingData и др.
│   │   ├── profiling_stats.hpp       # ProfilingMessage, EventStats, ModuleStats
│   │   ├── batch_manager.cpp/hpp     # BatchManager
│   │   ├── filter_config_service.cpp/hpp
│   │   ├── kernel_cache_service.cpp/hpp
│   │   └── storage/
│   │       ├── i_storage_backend.hpp
│   │       └── file_storage_backend.cpp/hpp
│   │
│   ├── config/                       # Конфигурация
│   │   ├── gpu_config.cpp/hpp        # GPUConfig (configGPU.json)
│   │   └── config_types.hpp          # GPUConfigEntry и др.
│   │
│   ├── logger/                       # Логирование (plog)
│   │   ├── logger.cpp/hpp            # Logger фасад (per-GPU)
│   │   ├── default_logger.cpp/hpp    # DefaultLogger на plog
│   │   └── config_logger.cpp/hpp     # ConfigLogger (пути, Singleton)
│   │
│   └── tests/                        # Тесты DrvGPU
│       ├── all_test.hpp
│       ├── single_gpu.hpp
│       ├── test_services.hpp
│       ├── test_gpu_profiler.hpp
│       ├── test_storage_services.hpp
│       ├── test_rocm_backend.hpp
│       ├── test_zero_copy.hpp
│       ├── test_hybrid_backend.hpp
│       └── example_external_context_usage.hpp
│
├── modules/                      # Модули обработки (см. Doc/Modules/)
│   ├── signal_generators/
│   ├── fft_processor/
│   ├── fft_maxima/
│   ├── statistics/
│   ├── heterodyne/
│   └── filters/
│
├── python/                       # Python bindings
│   └── gpu_worklib_bindings.cpp
│
└── Doc/
    ├── DrvGPU/                       # Документация ядра
    │   ├── Architecture.md           # Этот файл
    │   ├── Memory.md                 # Система памяти
    │   ├── OpenCL.md                 # OpenCL бэкенд
    │   ├── Command.md                # Command Queue
    │   ├── Classes.md                # Все классы по категориям
    │   └── Services/                 # Документация сервисов
    └── Modules/                      # Документация модулей
```

---

## Архитектурные слои

### 1. Core Layer (Основной слой)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `DrvGPU` | drv_gpu.hpp/cpp | Фасад библиотеки |
| `GPUManager` | gpu_manager.hpp | Управление несколькими GPU |
| `ModuleRegistry` | module_registry.hpp/cpp | Регистр compute модулей |

**См. также**: [Classes.md](Classes.md)

### 2. Backend Layer (Слой бэкендов)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `IBackend` | interface/i_backend.hpp | Интерфейс бэкенда |
| `OpenCLBackend` | backends/opencl/opencl_backend.hpp/cpp | OpenCL реализация |
| `OpenCLCore` | backends/opencl/opencl_core.hpp/cpp | Низкоуровневые операции |
| `OpenCLBackendExternal` | backends/opencl/opencl_backend_external.hpp/cpp | External Context |
| `ROCmBackend` | backends/rocm/rocm_backend.hpp/cpp | ✅ ROCm/HIP реализация |
| `ROCmCore` | backends/rocm/rocm_core.hpp/cpp | ✅ Per-device HIP контекст |
| `ZeroCopyBridge` | backends/rocm/zero_copy_bridge.hpp/cpp | ✅ OpenCL↔ROCm ZeroCopy |
| `HybridBackend` | backends/hybrid/hybrid_backend.hpp/cpp | ✅ OpenCL + ROCm вместе |

**См. также**: [OpenCL.md](OpenCL.md)

### 3. Command Layer (Слой команд)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `CommandQueuePool` | opencl/command_queue_pool.hpp/cpp | Пул очередей команд |

**См. также**: [Command.md](Command.md)

### 4. Memory Layer (Слой памяти)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `IMemoryBuffer` | memory/i_memory_buffer.hpp | Интерфейс буфера |
| `GPUBuffer<T>` | memory/gpu_buffer.hpp | Owning RAII буфер (cl_mem) |
| `SVMBuffer` | memory/svm_buffer.hpp | SVM буфер |
| `HIPBuffer<T>` | memory/hip_buffer.hpp | ✅ Non-owning HIP буфер (ROCm) |
| `MemoryManager` | memory/memory_manager.hpp | Менеджер памяти |

**См. также**: [Memory.md](Memory.md)

### 5. Services Layer (Сервисы)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `AsyncServiceBase<T>` | services/async_service_base.hpp | Базовый шаблон: фоновый поток + очередь |
| `ConsoleOutput` | services/console_output.hpp | ✅ Singleton: мультиGPU-безопасный stdout |
| `GPUProfiler` | services/gpu_profiler.hpp | ✅ Singleton: сбор статистики, экспорт JSON/MD |
| `ServiceManager` | services/service_manager.hpp | ✅ Singleton: Init/Start/Stop всех сервисов |
| `GpuBenchmarkBase` | services/gpu_benchmark_base.hpp | Базовый класс бенчмарков |
| `BatchManager` | services/batch_manager.hpp/cpp | Обработка больших данных по частям |
| `KernelCacheService` | services/kernel_cache_service.hpp | Кэш скомпилированных kernels |
| `FilterConfigService` | services/filter_config_service.hpp | Хранение конфигов фильтров |

### 6. Common (Логирование и утилиты)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `Logger` | logger/logger.hpp/cpp | Фасад логирования (per-GPU) |
| `ILogger` | interface/i_logger.hpp | Интерфейс логера |
| `DefaultLogger` | logger/default_logger.hpp/cpp | Реализация на plog |
| `ConfigLogger` | logger/config_logger.hpp/cpp | Конфигурация путей логов |
| `GPUDeviceInfo` | common/gpu_device_info.hpp | Структура: информация о GPU |
| `GPUConfig` | config/gpu_config.hpp | Загрузка configGPU.json |

---

## Паттерны проектирования

| Паттерн | Применение |
|---------|------------|
| **Bridge** | Разделение абстракции и реализации бэкендов |
| **Facade** | DrvGPU как упрощённый интерфейс |
| **Singleton** | Logger, ConfigLogger, DefaultLogger |
| **Factory** | Создание бэкендов, GPUManager |
| **Strategy** | LoadBalancingStrategy |
| **Registry** | ModuleRegistry |
| **Object Pool** | CommandQueuePool |
| **Per-Device** | OpenCLCore (v2.0) - каждый экземпляр для своего GPU |

> ⚠️ **Примечание (v2.0)**: OpenCLCore больше НЕ использует Singleton! Теперь это per-device класс для поддержки Multi-GPU.

**См. также**: [Classes.md](Classes.md) - подробности о паттернах

---

## Зависимости между компонентами

```
                    ┌─────────────────┐
                    │     DrvGPU      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  GPUManager   │   │ MemoryManager │   │ ModuleRegistry│
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                    │
        │                   ▼                    │
        │          ┌───────────────┐             │
        │          │  IBackend     │◄────────────┘
        │          └───────┬───────┘
        │                  │
        │    ┌─────────────┼─────────────┐
        │    │             │             │
        ▼    ▼             ▼             ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│  OpenCL   │   │   ROCm    │   │   CUDA    │
│  Backend  │   │  Backend  │   │  Backend  │
└───────────┘   └───────────┘   └───────────┘
```

---

## Документация по разделам

### DrvGPU (ядро)

| Раздел | Файл | Описание |
|--------|------|----------|
| **Общая архитектура** | [Architecture.md](Architecture.md) | Этот файл |
| **Система памяти** | [Memory.md](Memory.md) | IMemoryBuffer, GPUBuffer, SVMBuffer, MemoryManager |
| **OpenCL бэкенд** | [OpenCL.md](OpenCL.md) | OpenCLBackend, OpenCLCore, OpenCLBackendExternal |
| **Command Queue** | [Command.md](Command.md) | CommandQueuePool, управление очередями |
| **Все классы** | [Classes.md](Classes.md) | Полный справочник классов |

### Модули

| Модуль | Документация | Описание |
|--------|-------------|----------|
| **Signal Generators** | [../Modules/signal_generators/](../Modules/signal_generators/) | CW, LFM, Noise, Script генераторы |
| **FFT Processor** | [../Modules/fft_processor/](../Modules/fft_processor/) | GPU FFT (hipFFT/clFFT) |
| **FFT Maxima** | [../Modules/fft_maxima/](../Modules/fft_maxima/) | Поиск максимумов спектра |
| **Statistics** | [../Modules/statistics/](../Modules/statistics/) | Welford, medians, radix sort |
| **Heterodyne** | [../Modules/heterodyne/](../Modules/heterodyne/) | LFM Dechirp, NCO |
| **Filters** | [../Modules/filters/](../Modules/filters/) | FIR, IIR фильтры |

---

## Связь с LID-Architecture

Документация DrvGPU соответствует концепции LID (Library Interface Definition):

- **LID-Core**: `DrvGPU`, `GPUManager`, `ModuleRegistry`
- **LID-Backend**: `IBackend`, `OpenCLBackend`, `ROCmBackend`, `HybridBackend`
- **LID-Services**: `ConsoleOutput`, `GPUProfiler`, `ServiceManager`, `BatchManager`
- **LID-Utils**: `Logger`, `GPUDeviceInfo`, `LoadBalancing`, `GPUConfig`
- **LID-Memory**: `IMemoryBuffer`, `GPUBuffer<T>`, `HIPBuffer<T>`, `MemoryManager`

---

*Последнее обновление: 2026-03-02*
