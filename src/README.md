# DrvGPU/src — Реализации

Исходники основных классов. Заголовки — в `include/`.

## Файлы

| Файл | Класс | Описание |
|------|-------|----------|
| `drv_gpu.cpp` | `DrvGPU` | Инициализация, MemoryManager, ModuleRegistry |
| `module_registry.cpp` | `ModuleRegistry` | Регистрация и доступ к compute-модулям |

## Как тестировать

Через публичный API: `tests/single_gpu.hpp` — `example_drv_gpu_singl::run()` (тестирует DrvGPU).
