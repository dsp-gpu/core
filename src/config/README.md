# DrvGPU/config — Конфигурация GPU

Конфигурация GPU из `configGPU.json` (per-GPU флаги: is_prof, is_logger, is_console).

## Классы и файлы

| Файл | Описание |
|------|----------|
| `gpu_config.hpp/cpp` | `GPUConfig` — Singleton. `Load()`, `LoadOrCreate()`, `GetConfig(gpu_id)`, `GetAllConfigs()` |
| `config_types.hpp` | `GPUConfigEntry` — структура (id, name, is_prof, is_logger, is_console) |
| `configGPU.json` | JSON-конфиг (версия, массив gpus) |

## Формат configGPU.json

```json
{
  "version": "1.0",
  "gpus": [
    { "id": 0, "name": "GPU0", "is_prof": true, "is_logger": true, "is_console": true },
    { "id": 1, "name": "GPU1" }
  ]
}
```

## Использование

```cpp
// Загрузить
GPUConfig::GetInstance().Load("./configGPU.json");

// Получить конфиг для GPU
auto& cfg = GPUConfig::GetInstance().GetConfig(0);
if (cfg.is_prof) { /* профилирование включено */ }
```

## Как тестировать

- **GPUConfig**: `tests/test_services.hpp` — `test_services::run()` — ServiceManager использует GPUConfig при инициализации
- **Проверка загрузки**: создать тестовый JSON, вызвать `Load()`, проверить `GetConfig()`
