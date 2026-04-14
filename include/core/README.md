# DrvGPU/include — Публичный API

Главные классы DrvGPU — точка входа для приложений.

## Классы

| Файл | Класс | Описание |
|------|-------|----------|
| `drv_gpu.hpp` | `DrvGPU` | Абстракция GPU (НЕ Singleton). Multi-instance. `Initialize()`, `GetMemoryManager()`, `GetDeviceInfo()`, `PrintDeviceInfo()` |
| `gpu_manager.hpp` | `GPUManager` | Multi-GPU координатор. `InitializeAll()`, `GetGPU(id)`, `GetGPUCount()`, load balancing |
| `module_registry.hpp` | `ModuleRegistry` | Реестр compute-модулей. `Register()`, `Get()` |

## Использование

```cpp
// Single GPU
DrvGPU gpu(BackendType::OPENCL, 0);
gpu.Initialize();
auto& mem = gpu.GetMemoryManager();

// Multi-GPU
GPUManager manager;
manager.InitializeAll(BackendType::OPENCL);
auto gpu0 = manager.GetGPU(0);
```

## Как тестировать

- **DrvGPU**: `tests/single_gpu.hpp` — `example_drv_gpu_singl::run()` — инициализация, память, device info
- **GPUManager**: `tests/multi_gpu.hpp` — multi-GPU сценарии
- **ModuleRegistry**: косвенно через модули (FFT, SignalGen и т.д.)
