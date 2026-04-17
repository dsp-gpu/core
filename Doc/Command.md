# DrvGPU Command Queue System

## Оглавление

1. [Обзор системы команд](#обзор-системы-команд)
2. [CommandQueuePool](#commandqueuepool)
3. [Интеграция с бэкендом](#интеграция-с-бэкендом)
4. [Примеры использования](#примеры-использования)
5. [Thread Safety](#thread-safety)
6. [Оптимизация производительности](#оптимизация-производительности)

---

## Обзор системы команд

Система команд DrvGPU обеспечивает эффективное управление очередями команд GPU через паттерн Object Pool.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Command Queue Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   OpenCLBackend                             │   │
│   │  ┌─────────────────────────────────────────────────────┐   │   │
│   │  │            CommandQueuePool (main queue)            │   │   │
│   │  └───────────────────────┬─────────────────────────────┘   │   │
│   │                          │                                     │
│   │       ┌──────────────────┼──────────────────┐                │
│   │       ▼                  ▼                  ▼                │
│   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│   │  │   Queue 0   │   │   Queue 1   │   │   Queue N   │        │
│   │  │  (active)   │   │   (idle)    │   │   (idle)    │        │
│   │  └─────────────┘   └─────────────┘   └─────────────┘        │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Ключевые классы

| Класс | Файл | Назначение |
|-------|------|------------|
| `CommandQueuePool` | command_queue_pool.hpp/cpp | Пул очередей команд |
| `OpenCLBackend` | opencl_backend.hpp/cpp | Бэкенд с интегрированным пулом |

---

## CommandQueuePool

### Назначение

`CommandQueuePool` реализует паттерн Object Pool для переиспользования очередей команд OpenCL, что значительно снижает накладные расходы на создание/удаление очередей.

### Файл

[`opencl/command_queue_pool.hpp`](../../include/DrvGPU/backends/opencl/command_queue_pool.hpp)

### Интерфейс класса

```cpp
class CommandQueuePool {
public:
    /**
     * @brief Создать пул очередей
     * @param context OpenCL контекст
     * @param device Устройство
     * @param initial_size Начальный размер пула
     */
    explicit CommandQueuePool(cl_context context, 
                               cl_device_id device,
                               size_t initial_size = 4);
    
    ~CommandQueuePool();
    
    // ═══════════════════════════════════════════════
    // Pool Management
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Получить очередь из пула
     * @return cl_command_queue
     * 
     * Если пул пуст, создаёт новую очередь автоматически.
     * Внимание: полученная очередь считается "в использовании"
     * и должна быть возвращена через ReleaseQueue().
     */
    cl_command_queue AcquireQueue();
    
    /**
     * @brief Вернуть очередь в пул
     * @param queue Очередь для возврата
     * 
     * Очередь должна быть создана этим пулом или совместима с ним.
     */
    void ReleaseQueue(cl_command_queue queue);
    
    /**
     * @brief Установить максимальный размер пула
     * @param max_size Максимальное количество очередей
     * 
     * При превышении лимита новые очереди не создаются.
     */
    void SetMaxSize(size_t max_size);
    
    /**
     * @brief Получить текущий размер пула
     * @return Количество очередей в пуле
     */
    size_t GetPoolSize() const;
    
    /**
     * @brief Получить количество доступных очередей
     * @return Количество свободных очередей
     */
    size_t GetAvailableCount() const;
    
    // ═══════════════════════════════════════════════
    // Operations
    // ═══════════════════════════════════════════════
    
    /**
     * @brief Синхронизировать все очереди в пуле
     * 
     * Вызывает clFinish для всех доступных очередей.
     * Используется для полной синхронизации перед очисткой.
     */
    void SynchronizeAll();
    
    /**
     * @brief Очистить пул (освободить все очереди)
     * 
     * Освобождает все очереди и очищает внутренние списки.
     */
    void Clear();
    
    /**
     * @brief Проверить, пуст ли пул
     */
    bool IsEmpty() const;
    
    /**
     * @brief Проверить, полон ли пул
     */
    bool IsFull() const;

private:
    cl_context context_;
    cl_device_id device_;
    size_t max_size_;
    
    std::vector<cl_command_queue> available_;
    std::vector<cl_command_queue> in_use_;
    
    mutable std::mutex mutex_;
};
```

### Внутренняя структура

```
CommandQueuePool Internal State
┌─────────────────────────────────────────┐
│           Pool Manager                  │
├─────────────────────────────────────────┤
│  context_: cl_context                   │
│  device_: cl_device_id                  │
│  max_size_: size_t (default: SIZE_MAX)  │
├─────────────────────────────────────────┤
│  available_: vector<cl_command_queue>   │
│     [Queue 0] - idle, ready to use      │
│     [Queue 1] - idle, ready to use      │
│     ...                                 │
├─────────────────────────────────────────┤
│  in_use_: vector<cl_command_queue>      │
│     [Queue A] - acquired, in use        │
│     [Queue B] - acquired, in use        │
│     ...                                 │
├─────────────────────────────────────────┤
│  mutex_: std::mutex (thread safety)     │
└─────────────────────────────────────────┘
```

### Жизненный цикл очереди

```
AcquireQueue()                    ReleaseQueue()
     │                                │
     ▼                                │
┌─────────┐     ┌─────────┐     ┌─────────┐
│ available│ ──► │  in_use │ ──► │available│
│   [0]    │     │   [A]   │     │   [0]   │
└─────────┘     └─────────┘     └─────────┘
```

---

## Интеграция с бэкендом

### OpenCLBackend с CommandQueuePool

```cpp
class OpenCLBackend : public IBackend {
public:
    OpenCLBackend();
    ~OpenCLBackend() override;
    
    // ... другие методы ...
    
    /**
     * @brief Получить пул очередей
     */
    CommandQueuePool& GetCommandQueuePool() {
        return command_queue_pool_;
    }
    
    /**
     * @brief Получить очередь для асинхронной операции
     */
    cl_command_queue AcquireCommandQueue() {
        return command_queue_pool_.AcquireQueue();
    }
    
    /**
     * @brief Вернуть очередь
     */
    void ReleaseCommandQueue(cl_command_queue queue) {
        command_queue_pool_.ReleaseQueue(queue);
    }
    
private:
    // ... другие члены ...
    CommandQueuePool command_queue_pool_;
};
```

---

## Примеры использования

### Пример 1: Базовое использование пула

```cpp
#include <core/backends/opencl/command_queue_pool.hpp>

void processWithPool(cl_context context, cl_device_id device) {
    // Создать пул с 4 очередями
    drv_gpu_lib::CommandQueuePool pool(context, device, 4);
    
    // Получить очередь из пула
    cl_command_queue queue1 = pool.AcquireQueue();
    cl_command_queue queue2 = pool.AcquireQueue();
    
    // Использовать очереди для операций
    clEnqueueNDRangeKernel(queue1, kernel1, 1, nullptr, 
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(queue2, kernel2, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    
    // Вернуть очереди в пул
    pool.ReleaseQueue(queue1);
    pool.ReleaseQueue(queue2);
}
```

### Пример 2: Асинхронные операции с несколькими очередями

```cpp
void asyncProcessing(cl_context context, cl_device_id device,
                     cl_kernel kernels[], int count) {
    drv_gpu_lib::CommandQueuePool pool(context, device, count);
    
    std::vector<cl_command_queue> queues;
    std::vector<cl_event> events;
    
    // Запустить все kernels параллельно
    for (int i = 0; i < count; i++) {
        cl_command_queue queue = pool.AcquireQueue();
        queues.push_back(queue);
        
        cl_event event;
        clEnqueueNDRangeKernel(queue, kernels[i], 1, nullptr,
                               &global_size, &local_size, 
                               i > 0 ? &events.back() : 0,
                               nullptr, &event);
        events.push_back(event);
    }
    
    // Дождаться завершения всех операций
    clWaitForEvents(events.size(), events.data());
    
    // Освободить события
    for (auto& event : events) {
        clReleaseEvent(event);
    }
    
    // Вернуть очереди в пул
    for (auto& queue : queues) {
        pool.ReleaseQueue(queue);
    }
}
```

### Пример 3: Интеграция с DrvGPU

```cpp
#include <core/drv_gpu.hpp>

void processWithDrvGPU() {
    drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
    gpu.Initialize();
    
    // Получить пул очередей через бэкенд
    auto& backend = gpu.GetBackend();
    
    // OpenCLBackend предоставляет доступ к пулу
    cl_command_queue queue = backend.AcquireCommandQueue();
    
    // Использовать очередь
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    
    // Вернуть очередь
    backend.ReleaseCommandQueue(queue);
    
    gpu.Cleanup();
}
```

---

## Thread Safety

`CommandQueuePool` является thread-safe благодаря использованию `std::mutex`.

### Гарантии потокобезопасности

| Операция | Thread Safety | Notes |
|----------|---------------|-------|
| `AcquireQueue()` | Да | Блокирует при отсутствии свободных очередей |
| `ReleaseQueue()` | Да | Безопасно возвращать любую очередь |
| `GetPoolSize()` | Да | Константная операция |
| `SynchronizeAll()` | Да | Блокирует доступ к пулу |
| `Clear()` | Да | Требует синхронизации всех очередей |

### Пример многопоточного использования

```cpp
void parallelProcessing(cl_context context, cl_device_id device, int num_threads) {
    drv_gpu_lib::CommandQueuePool pool(context, device, num_threads);
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&pool, i]() {
            // Каждый поток безопасно получает свою очередь
            cl_command_queue queue = pool.AcquireQueue();
            
            // ... операции с очередью ...
            
            // Возврат очереди
            pool.ReleaseQueue(queue);
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}
```

---

## Оптимизация производительности

### Когда использовать Command Queue Pool

| Сценарий | Преимущество |
|----------|--------------|
| Многопоточные приложения | Избежать блокировок при создании очередей |
| Асинхронные операции | Параллельное выполнение kernel |
| Высокая нагрузка | Уменьшение overhead создания очередей |
| Pipeline обработка | Конвейеризация операций |
| Real-time системы | Предсказуемое время отклика |

### Рекомендации по размеру пула

```cpp
// Для однопоточных приложений: 1-2 очереди
CommandQueuePool pool(context, device, 1);

// Для многопоточных: количество потоков или больше
int num_threads = std::thread::hardware_concurrency();
CommandQueuePool pool(context, device, num_threads * 2);

// Для высокой нагрузки: динамическое расширение
CommandQueuePool pool(context, device, 8);
pool.SetMaxSize(32);  // Не более 32 очередей
```

### Сравнение производительности

| Метод | Время создания | Overhead на операцию |
|-------|----------------|---------------------|
| Без пула (clCreateCommandQueue) | ~100-500 мкс | - |
| CommandQueuePool::AcquireQueue | ~1-10 мкс | Минимальный |
| Повторное использование | - | ~0 мкс |

---

## Сводная таблица файлов

| Файл | Класс | Назначение | Сложность |
|------|-------|------------|-----------|
| `command_queue_pool.hpp` | `CommandQueuePool` | Интерфейс пула | Средняя |
| `command_queue_pool.cpp` | `CommandQueuePool` | Реализация | Средняя |
| `opencl_backend.hpp` | `OpenCLBackend` | Бэкенд с интеграцией | Высокая |
| `opencl_backend.cpp` | `OpenCLBackend` | Реализация | Высокая |

---

## Зависимости

```
CommandQueuePool
    ├── CL/cl.h (OpenCL)
    └── std::mutex, std::vector (C++ STL)

OpenCLBackend
    ├── CommandQueuePool
    ├── OpenCLCore
    ├── IBackend
    └── Logger
```

---

## Roadmap

| Версия | Изменение | Статус |
|--------|-----------|--------|
| v1.0 | Базовый CommandQueuePool | ✅ Готов |
| v1.1 | Динамическое расширение пула | 📋 План |
| v1.2 | Profiling queue support | 📋 План |
| v2.0 | Асинхронные события | 📋 План |

---

## Заключение

Система Command Queue в DrvGPU предоставляет:

- ✅ **Object Pool** паттерн для эффективного переиспользования очередей
- ✅ **Thread-safe** операции для многопоточных приложений
- ✅ **Интеграция** с OpenCLBackend для прозрачного использования
- ✅ **Гибкость** настройки размера и лимитов пула

**См. также**:
- [OpenCL.md](OpenCL.md) — OpenCL бэкенд
- [Architecture.md](Architecture.md) — общая архитектура
