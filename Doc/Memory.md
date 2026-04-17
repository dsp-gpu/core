# DrvGPU Система памяти

## Оглавление

1. [Обзор системы памяти](#обзор-системы-памяти)
2. [Типы памяти](#типы-памяти)
3. [Интерфейс буфера](#интерфейс-буфера)
4. [Реализации буферов](#реализации-буферов)
5. [Менеджер памяти](#менеджер-памяти)
6. [Shared Virtual Memory](#shared-virtual-memory)
7. [External Buffer Adapter](#external-buffer-adapter)
8. [Примеры использования](#примеры-использования)
9. [Сводная таблица файлов](#сводная-таблица-файлов)

---

## Обзор системы памяти

DrvGPU предоставляет абстракцию над различными типами памяти GPU с единым интерфейсом.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Memory Abstraction Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    IMemoryBuffer                            │   │
│   │                 (Абстрактный интерфейс)                     │   │
│   └─────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│           ┌─────────────────┼─────────────────┐                     │
│           │                 │                 │                     │
│           ▼                 ▼                 ▼                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │   GPUBuffer  │  │  SVMBuffer   │  │ CustomBuffer │             │
│   │  (Standard)  │  │    (SVM)     │  │   (User)     │             │
│   └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Backend (OpenCL/ROCm)                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │
│   │ Device Mem  │  │  Host Mem   │  │         SVM                 │ │
│   └─────────────┘  └─────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Ключевые компоненты

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `IMemoryBuffer` | i_memory_buffer.hpp | Интерфейс буфера |
| `GPUBuffer` | gpu_buffer.hpp | Стандартный буфер |
| `SVMBuffer` | svm_buffer.hpp | Shared Virtual Memory |
| `MemoryManager` | memory_manager.hpp | Менеджер памяти |
| `ExternalCLBufferAdapter` | external_cl_buffer_adapter.hpp | Адаптер внешних буферов |

---

## Типы памяти

### MemoryType

**Файл**: [`memory_type.hpp`](../../include/DrvGPU/memory/memory_type.hpp)

```cpp
enum class MemoryType {
    /// Память только на устройстве (GPU)
    Device,
    
    /// Память только на хосте (CPU)
    Host,
    
    /// Shared память (OpenCL pinned memory)
    Shared,
    
    /// Shared Virtual Memory (SVM)
    SVM,
    
    /// Unified Memory (CUDA managed)
    Unified
};
```

### Сравнение типов памяти

| Тип | Доступ с хоста | Доступ с GPU | Производительность | SVM |
|-----|----------------|--------------|-------------------|-----|
| Device | ❌ Через копирование | ✅ Быстрый | Высокая | ❌ |
| Host | ✅ Быстрый | ❌ Через копирование | Средняя | ❌ |
| Shared | ✅ | ✅ | Средняя | ❌ |
| SVM | ✅ | ✅ | Высокая | ✅ |
| Unified | ✅ | ✅ | Переменная | ✅ |

### Когда использовать

| Тип | Когда использовать |
|-----|-------------------|
| Device | Буферы, используемые только GPU |
| Host | Данные, обрабатываемые только CPU |
| Shared | Частые обмены между CPU и GPU |
| SVM | Максимальная производительность обмена |
| Unified | Автоматическое управление памятью |

---

## Интерфейс буфера

### IMemoryBuffer

**Файл**: [`i_memory_buffer.hpp`](../../include/DrvGPU/memory/i_memory_buffer.hpp)

```cpp
class IMemoryBuffer {
public:
    virtual ~IMemoryBuffer() = default;

    // ═══════════════════════════════════════════════
    // Basic Information
    // ═══════════════════════════════════════════════
    
    virtual size_t GetSize() const = 0;
    virtual MemoryType GetType() const = 0;
    virtual void* GetDevicePointer() = 0;
    virtual const void* GetDevicePointer() const = 0;

    // ═══════════════════════════════════════════════
    // Data Transfer
    // ═══════════════════════════════════════════════
    
    virtual bool CopyFromHost(const void* data, 
                              size_t size,
                              size_t offset = 0) = 0;
    
    virtual bool CopyToHost(void* data, 
                            size_t size,
                            size_t offset = 0) = 0;
    
    virtual bool CopyToDevice(IMemoryBuffer& dst,
                              size_t size,
                              size_t src_offset = 0,
                              size_t dst_offset = 0) = 0;

    // ═══════════════════════════════════════════════
    // Mapping (для SVM и Shared memory)
    // ═══════════════════════════════════════════════
    
    virtual bool Map() = 0;
    virtual void Unmap() = 0;
    virtual void* GetMappedPointer() = 0;

    // ═══════════════════════════════════════════════
    // Synchronization
    // ═══════════════════════════════════════════════
    
    virtual void Wait() = 0;
    virtual bool IsReady() const = 0;
};
```

---

## Реализации буферов

### GPUBuffer

**Файл**: [`gpu_buffer.hpp`](../../include/DrvGPU/memory/gpu_buffer.hpp)

Стандартный буфер памяти GPU.

```cpp
class GPUBuffer : public IMemoryBuffer {
public:
    GPUBuffer(IBackend* backend, size_t size, MemoryType type = MemoryType::Device);
    ~GPUBuffer() override;

    // IMemoryBuffer implementation
    size_t GetSize() const override;
    MemoryType GetType() const override;
    void* GetDevicePointer() override;
    const void* GetDevicePointer() const override;
    
    bool CopyFromHost(const void* data, size_t size, size_t offset = 0) override;
    bool CopyToHost(void* data, size_t size, size_t offset = 0) override;
    bool CopyToDevice(IMemoryBuffer& dst, size_t size,
                      size_t src_offset = 0, size_t dst_offset = 0) override;
    
    bool Map() override;
    void Unmap() override;
    void* GetMappedPointer() override;
    
    void Wait() override;
    bool IsReady() const override;

    // GPUBuffer specific
    cl_mem GetCLMem() const;
    static std::shared_ptr<GPUBuffer> FromCLMem(IBackend* backend, cl_mem cl_mem_object);

private:
    IBackend* backend_;
    size_t size_;
    MemoryType type_;
    void* device_ptr_ = nullptr;
    void* host_ptr_ = nullptr;
    bool mapped_ = false;
};
```

---

### SVMBuffer

**Файл**: [`svm_buffer.hpp`](../../include/DrvGPU/memory/svm_buffer.hpp)

Буфер Shared Virtual Memory.

```cpp
class SVMBuffer : public IMemoryBuffer {
public:
    SVMBuffer(IBackend* backend, size_t size, unsigned int flags = 0);
    ~SVMBuffer() override;

    // IMemoryBuffer implementation
    size_t GetSize() const override;
    MemoryType GetType() const override;
    void* GetDevicePointer() override;
    const void* GetDevicePointer() const override;
    
    bool CopyFromHost(const void* data, size_t size, size_t offset = 0) override;
    bool CopyToHost(void* data, size_t size, size_t offset = 0) override;
    bool CopyToDevice(IMemoryBuffer& dst, size_t size,
                      size_t src_offset = 0, size_t dst_offset = 0) override;
    
    bool Map() override;      // Для SVM не требуется
    void Unmap() override;    // Для SVM не требуется
    void* GetMappedPointer() override;
    
    void Wait() override;
    bool IsReady() const override;

    // SVMBuffer specific
    static bool IsSupported(IBackend* backend);
    SVMCapabilities::Type GetSVMType() const;
};
```

**Преимущества SVM**:
```cpp
// SVMBuffer - один указатель для хоста и GPU
auto svm = std::make_shared<SVMBuffer>(&backend, 1024);

// Запись с хоста
float* host_ptr = static_cast<float*>(svm->GetDevicePointer());
for (int i = 0; i < 256; i++) {
    host_ptr[i] = static_cast<float>(i);
}

// GPU kernel использует тот же указатель
// Чтение с хоста без копирования
for (int i = 0; i < 256; i++) {
    printf("%f ", host_ptr[i]);
}
```

---

## Менеджер памяти

### MemoryManager

**Файл**: [`memory_manager.hpp`](../../include/DrvGPU/memory/memory_manager.hpp)

```cpp
class MemoryManager {
public:
    explicit MemoryManager(IBackend* backend);
    ~MemoryManager();

    // ═══════════════════════════════════════════════
    // Memory Operations
    // ═══════════════════════════════════════════════
    
    void* Allocate(size_t size_bytes, unsigned int flags = 0);
    void Free(void* ptr);
    
    std::shared_ptr<IMemoryBuffer> AllocateBuffer(size_t size, 
                                                   MemoryType type = MemoryType::Device);
    
    std::shared_ptr<SVMBuffer> AllocateSVM(size_t size);

    // ═══════════════════════════════════════════════
    // Statistics
    // ═══════════════════════════════════════════════
    
    size_t GetAllocationCount() const;
    size_t GetTotalAllocatedBytes() const;
    size_t GetPeakMemoryUsage() const;
    size_t GetCurrentMemoryUsage() const;
    
    void PrintStatistics() const;
    std::string GetStatistics() const;
    void ResetStatistics();

private:
    IBackend* backend_;
    
    std::atomic<size_t> total_allocations_;
    std::atomic<size_t> total_frees_;
    std::atomic<size_t> current_allocations_;
    std::atomic<size_t> total_bytes_allocated_;
    std::atomic<size_t> peak_bytes_allocated_;
    
    mutable std::mutex mutex_;
    
    void TrackAllocation(size_t size);
    void TrackFree(size_t size);
};
```

---

## Shared Virtual Memory

### SVMCapabilities

**Файл**: [`svm_capabilities.hpp`](../../include/DrvGPU/memory/svm_capabilities.hpp)

```cpp
struct SVMCapabilities {
    bool supports_svm = false;
    bool supports_coarse_grain = false;
    bool supports_fine_grain = false;
    bool supports_fine_grain_system = false;
    bool supports_atomic = false;
    
    static SVMCapabilities Get(IBackend* backend);
    bool HasSVM() const { return supports_svm; }
    SVMType GetBestSVMType() const;
};

enum class SVMType {
    None,           // SVM не поддерживается
    CoarseGrain,    // Coarse-grain SVM
    FineGrain,      // Fine-grain SVM
    FineGrainSystem // Fine-grain System SVM
};
```

---

## External Buffer Adapter

### ExternalCLBufferAdapter

**Файл**: [`external_cl_buffer_adapter.hpp`](../../include/DrvGPU/memory/external_cl_buffer_adapter.hpp)

```cpp
class ExternalCLBufferAdapter : public IMemoryBuffer {
public:
    ExternalCLBufferAdapter(IBackend* backend,
                            cl_mem external_buffer,
                            size_t size,
                            bool take_ownership = false);
    
    ~ExternalCLBufferAdapter() override;

    // IMemoryBuffer implementation
    size_t GetSize() const override;
    MemoryType GetType() const override;
    void* GetDevicePointer() override;
    const void* GetDevicePointer() const override;
    
    bool CopyFromHost(const void* data, size_t size, size_t offset = 0) override;
    bool CopyToHost(void* data, size_t size, size_t offset = 0) override;
    bool CopyToDevice(IMemoryBuffer& dst, size_t size,
                      size_t src_offset = 0, size_t dst_offset = 0) override;
    
    bool Map() override;
    void Unmap() override;
    void* GetMappedPointer() override;
    
    void Wait() override;
    bool IsReady() const override;

    // ExternalCLBufferAdapter specific
    cl_mem GetCLMem() const;
    bool IsValid() const;
};
```

---

## Примеры использования

### Пример 1: Базовое выделение памяти

```cpp
#include <core/drv_gpu.hpp>

int main() {
    drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
    gpu.Initialize();
    
    auto& memory = gpu.GetMemoryManager();
    
    float* data = static_cast<float*>(
        memory.Allocate(1024 * sizeof(float))
    );
    
    gpu.GetBackend().MemcpyHostToDevice(data, original_data, 1024 * sizeof(float));
    gpu.Synchronize();
    
    memory.Free(data);
    
    return 0;
}
```

### Пример 2: Использование буферов

```cpp
auto buffer = memory.AllocateBuffer(1024, drv_gpu_lib::MemoryType::Shared);

float* host_ptr = static_cast<float*>(buffer->GetDevicePointer());
for (int i = 0; i < 256; i++) {
    host_ptr[i] = static_cast<float>(i);
}

// GPU kernel может использовать буфер
// Результат доступен сразу на хосте
```

### Пример 3: SVM буфер

```cpp
if (!drv_gpu_lib::SVMBuffer::IsSupported(&gpu.GetBackend())) {
    std::cerr << "SVM not supported!\n";
    return 1;
}

auto svm = memory.AllocateSVM(1024);
float* ptr = static_cast<float*>(svm->GetDevicePointer());

ptr[0] = 42.0f;  // Запись с хоста
// GPU kernel использует тот же указатель
printf("%f\n", ptr[0]);  // Чтение без копирования
```

### Пример 4: External Buffer

```cpp
cl_mem external_buffer = /* из другой библиотеки */;
size_t buffer_size = 1024;

auto adapter = std::make_shared<ExternalCLBufferAdapter>(
    &gpu.GetBackend(),
    external_buffer,
    buffer_size,
    false  // Не забираем владение
);

adapter->CopyFromHost(data, buffer_size);
adapter->Wait();
```

---

## Сводная таблица файлов

| Файл | Класс | Назначение | Сложность |
|------|-------|------------|-----------|
| `memory_type.hpp` | `MemoryType` | Перечисление типов | Низкая |
| `i_memory_buffer.hpp` | `IMemoryBuffer` | Интерфейс буфера | Средняя |
| `gpu_buffer.hpp` | `GPUBuffer` | Стандартный буфер | Средняя |
| `memory_manager.hpp` | `MemoryManager` | Менеджер памяти | Средняя |
| `svm_buffer.hpp` | `SVMBuffer` | SVM буфер | Высокая |
| `svm_capabilities.hpp` | `SVMCapabilities` | Возможности SVM | Низкая |
| `external_cl_buffer_adapter.hpp` | `ExternalCLBufferAdapter` | Адаптер | Средняя |

---

## Зависимости

```
IMemoryBuffer
    ├── MemoryType
    └── IBackend (для операций)

GPUBuffer
    └── IMemoryBuffer
        └── IBackend

SVMBuffer
    └── IMemoryBuffer
        └── IBackend
            └── SVMCapabilities

MemoryManager
    ├── IBackend
    └── IMemoryBuffer

ExternalCLBufferAdapter
    └── IMemoryBuffer
        └── IBackend
```

---

## Заключение

Система памяти DrvGPU предоставляет:

- ✅ Единый интерфейс `IMemoryBuffer`
- ✅ Поддержку различных типов памяти
- ✅ SVM с высокой производительностью
- ✅ Статистику использования памяти
- ✅ External Buffer адаптер
- ✅ Thread-safe операции

**См. также**:
- [Architecture.md](Architecture.md) — общая архитектура
- [OpenCL.md](OpenCL.md) — OpenCL бэкенд
- [Classes.md](Classes.md) — справочник классов

Выбор типа памяти:
1. **Device** - максимальная производительность GPU
2. **Shared** - частые обмены CPU-GPU
3. **SVM** - лучшая производительность для обменов
4. **Unified** - автоматическое управление
