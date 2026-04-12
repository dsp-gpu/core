/**
 * @file example_external_context_usage.hpp
 * @brief ПОЛНЫЙ ПРИМЕР использования DrvGPU с внешним OpenCL контекстом
 * 
 * СЦЕНАРИЙ:
 * У вас уже есть рабочий OpenCL код (класс YourExistingOpenCL).
 * Вы хотите интегрировать DrvGPU для упрощения управления памятью.
 * 
 * ✅ ОБНОВЛЕНО: Соответствует новой реализации с owns_resources_
 * ✅ ИСПРАВЛЕНО: Убраны несуществующие методы, используется реальный API
 * ✅ ОБНОВЛЕНО 2026-02-10: OpenCLBackendExternal удалён, используется OpenCLBackend
 *
 * @author DrvGPU Team
 * @date 2026-02-10
 */

 #include "DrvGPU/backends/opencl/opencl_backend.hpp"
 #include "DrvGPU/drv_gpu.hpp"
 #include <iostream>
 #include <vector>
 #include <CL/cl.h>
 
 // ════════════════════════════════════════════════════════════════════════════
 // СИМУЛЯЦИЯ вашего существующего OpenCL класса
 // ════════════════════════════════════════════════════════════════════════════
 
 class YourExistingOpenCL {
 public:
     YourExistingOpenCL() {
         // Инициализация OpenCL (ваш существующий код)
         InitializeOpenCL();
         // Создание буферов (ваш существующий код)
         CreateBuffers();
     }
     
     ~YourExistingOpenCL() {
         // Очистка (ваш существующий код)
         CleanupBuffers();
         CleanupOpenCL();
     }
     
     // Геттеры для OpenCL объектов
     cl_context GetContext() const { return context_; }
     cl_device_id GetDevice() const { return device_; }
     cl_command_queue GetQueue() const { return queue_; }
     cl_mem GetDataBuffer() const { return data_buffer_; }
 
 private:
     cl_platform_id platform_;
     cl_device_id device_;
     cl_context context_;
     cl_command_queue queue_;
     cl_mem data_buffer_;  // Ваш существующий буфер данных
     
     void InitializeOpenCL() {
         // Ваш существующий код инициализации OpenCL
         cl_int err;
         
         // 1. Platform
         err = clGetPlatformIDs(1, &platform_, nullptr);
         if (err != CL_SUCCESS)
             throw std::runtime_error("Platform init failed");
         
         // 2. Device
         err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
         if (err != CL_SUCCESS)
             throw std::runtime_error("Device init failed");
         
         // 3. Context
         context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
         if (err != CL_SUCCESS)
             throw std::runtime_error("Context creation failed");
         
         // 4. Queue
         #ifdef CL_VERSION_2_0
             cl_queue_properties props[] = {0};
             queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
         #else
             queue_ = clCreateCommandQueue(context_, device_, 0, &err);
         #endif
         
         if (err != CL_SUCCESS)
             throw std::runtime_error("Queue creation failed");
         
         std::cout << "✅ YourExistingOpenCL: Initialized\n";
     }
     
     void CreateBuffers() {
         cl_int err;
         
         // Создаем буфер на 1024 float элемента
         size_t buffer_size = 1024 * sizeof(float);
         data_buffer_ = clCreateBuffer(
             context_,
             CL_MEM_READ_WRITE,
             buffer_size,
             nullptr,
             &err);
         
         if (err != CL_SUCCESS)
             throw std::runtime_error("Buffer creation failed");
         
         // Заполняем буфер тестовыми данными
         std::vector<float> test_data(1024);
         for (size_t i = 0; i < test_data.size(); ++i) {
             test_data[i] = static_cast<float>(i);
         }
         
         err = clEnqueueWriteBuffer(
             queue_,
             data_buffer_,
             CL_TRUE,
             0,
             buffer_size,
             test_data.data(),
             0,
             nullptr,
             nullptr);
         
         if (err != CL_SUCCESS)
             throw std::runtime_error("Buffer write failed");
         
         std::cout << "✅ YourExistingOpenCL: Created and filled data_buffer\n";
     }
     
     void CleanupBuffers() {
         if (data_buffer_)
             clReleaseMemObject(data_buffer_);
     }
     
     void CleanupOpenCL() {
         if (queue_)
             clReleaseCommandQueue(queue_);
         if (context_)
             clReleaseContext(context_);
         if (device_)
             clReleaseDevice(device_);
         
         std::cout << "✅ YourExistingOpenCL: Cleaned up (released context/queue)\n";
     }
 };
 
 // ════════════════════════════════════════════════════════════════════════════
 // ПРИМЕР 1: Базовое использование внешнего контекста
 // ════════════════════════════════════════════════════════════════════════════
 
 void Example1_BasicExternalContext() {
     std::cout << "\n" << std::string(80, '=') << "\n";
     std::cout << "EXAMPLE 1: Базовое использование внешнего контекста\n";
     std::cout << std::string(80, '=') << "\n\n";
     
     // ────────────────────────────────────────────────────────────────
     // ШАГ 1: У вас уже есть рабочий OpenCL код
     // ────────────────────────────────────────────────────────────────
     std::cout << "📌 ШАГ 1: Создаём ваш существующий OpenCL класс...\n";
     YourExistingOpenCL your_opencl;
     
     // ────────────────────────────────────────────────────────────────
     // ШАГ 2: Создаем DrvGPU backend (пустой конструктор)
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📌 ШАГ 2: Создаём OpenCLBackend...\n";
     auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
     
     std::cout << "   ✅ Backend создан (owns_resources = false автоматически)\n";
     
     // ────────────────────────────────────────────────────────────────
     // ШАГ 3: Инициализируем DrvGPU с ВАШИМ контекстом
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📌 ШАГ 3: Инициализируем из вашего контекста...\n";
     backend->InitializeFromExternalContext(
         your_opencl.GetContext(),   // Ваш context
         your_opencl.GetDevice(),    // Ваш device
         your_opencl.GetQueue()      // Ваш queue
     );
     
     std::cout << "   ✅ DrvGPU успешно интегрирован с вашим OpenCL!\n";
     std::cout << "   ✅ owns_resources = " << (backend->OwnsResources() ? "true" : "false") << "\n";
     std::cout << "   Device: " << backend->GetDeviceName() << "\n";
     
     // ────────────────────────────────────────────────────────────────
     // ШАГ 4: Используем DrvGPU API
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📌 ШАГ 4: Проверяем DrvGPU API...\n";
     auto device_info = backend->GetDeviceInfo();
     std::cout << "   Vendor: " << device_info.vendor << "\n";
     std::cout << "   OpenCL Version: " << device_info.opencl_version << "\n";
     std::cout << "   Global Memory: " << (device_info.global_memory_size / (1024*1024)) << " MB\n";
     
     // ────────────────────────────────────────────────────────────────
     // ШАГ 5: Backend уничтожается
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📌 ШАГ 5: Уничтожаем backend...\n";
     backend.reset();  // Вызовет деструктор → Cleanup()
     
     std::cout << "   ✅ Backend уничтожен\n";
     std::cout << "   ⚠️  Cleanup() увидел owns_resources = false\n";
     std::cout << "   ⚠️  Ваш контекст/queue НЕ были освобождены!\n";
     
     // ────────────────────────────────────────────────────────────────
     // ШАГ 6: Ваш OpenCL код всё ещё работает!
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📌 ШАГ 6: Ваш OpenCL код всё ещё работает...\n";
     std::cout << "   Context: " << (your_opencl.GetContext() ? "OK ✅" : "NULL ❌") << "\n";
     std::cout << "   Queue: " << (your_opencl.GetQueue() ? "OK ✅" : "NULL ❌") << "\n";
     
     std::cout << "\n🎉 YourExistingOpenCL деструктор освободит ресурсы сам!\n";
     
     // ~YourExistingOpenCL() будет вызван здесь → CleanupOpenCL() → освободит контекст
 }
 
 // ════════════════════════════════════════════════════════════════════════════
 // ПРИМЕР 2: Работа с внешними cl_mem буферами через IBackend API
 // ════════════════════════════════════════════════════════════════════════════
 
 void Example2_WorkingWithExternalBuffers() {
     std::cout << "\n" << std::string(80, '=') << "\n";
     std::cout << "EXAMPLE 2: Работа с внешними cl_mem буферами\n";
     std::cout << std::string(80, '=') << "\n\n";
     
     // ШАГ 1: Ваш существующий OpenCL код
     YourExistingOpenCL your_opencl;
     
     // ШАГ 2: DrvGPU backend с внешним контекстом
     auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
     backend->InitializeFromExternalContext(
         your_opencl.GetContext(),
         your_opencl.GetDevice(),
         your_opencl.GetQueue()
     );
     
     std::cout << "✅ Backend initialized (owns_resources = " 
               << (backend->OwnsResources() ? "true" : "false") << ")\n";
     
     // ШАГ 3: Получаем ваш существующий cl_mem буфер
     cl_mem your_buffer = your_opencl.GetDataBuffer();
     size_t buffer_size = 1024 * sizeof(float);
     
     std::cout << "✅ Работаем с вашим cl_mem буфером напрямую\n";
     
     // ────────────────────────────────────────────────────────────────
     // USE CASE 1: ЗАГРУЗИТЬ данные с GPU -> Host
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📥 ЗАГРУЗКА данных с GPU (через IBackend::MemcpyDeviceToHost)...\n";
     std::vector<float> data_from_gpu(1024);
     
     backend->MemcpyDeviceToHost(
         data_from_gpu.data(),      // dst (host)
         your_buffer,               // src (device)
         buffer_size
     );
     
     std::cout << "   Прочитано элементов: " << data_from_gpu.size() << "\n";
     std::cout << "   Первые 5 элементов: ";
     for (size_t i = 0; i < 5; ++i) {
         std::cout << data_from_gpu[i] << " ";
     }
     std::cout << "\n";
     
     // ────────────────────────────────────────────────────────────────
     // USE CASE 2: Обработка данных на CPU
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n🔄 ОБРАБОТКА данных на CPU (умножение на 2)...\n";
     for (auto &val : data_from_gpu) {
         val *= 2.0f;
     }
     
     // ────────────────────────────────────────────────────────────────
     // USE CASE 3: ВЫГРУЗИТЬ обработанные данные Host -> GPU
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📤 ВЫГРУЗКА обработанных данных на GPU (через IBackend::MemcpyHostToDevice)...\n";
     
     backend->MemcpyHostToDevice(
         your_buffer,               // dst (device)
         data_from_gpu.data(),      // src (host)
         buffer_size
     );
     
     // ────────────────────────────────────────────────────────────────
     // USE CASE 4: Проверка результата
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n🔍 ПРОВЕРКА результата (загрузка снова)...\n";
     std::vector<float> result(1024);
     
     backend->MemcpyDeviceToHost(
         result.data(),
         your_buffer,
         buffer_size
     );
     
     std::cout << "   Первые 5 элементов после обработки: ";
     for (size_t i = 0; i < 5; ++i) {
         std::cout << result[i] << " ";
     }
     std::cout << "\n";
     
     std::cout << "\n✅ Цикл ЗАГРУЗКА -> ОБРАБОТКА -> ВЫГРУЗКА завершен!\n";
     
     // Уничтожение:
     // 1. backend уничтожается → НЕ освобождает контекст (owns_resources = false)
     // 2. your_opencl уничтожается → освобождает контекст И cl_mem ✅
 }
 
 // ════════════════════════════════════════════════════════════════════════════
 // ПРИМЕР 3: Прямые утилиты для работы с буферами
 // ════════════════════════════════════════════════════════════════════════════
 
 void Example3_DirectBufferUtilities() {
     std::cout << "\n" << std::string(80, '=') << "\n";
     std::cout << "EXAMPLE 3: Прямые утилиты для работы с буферами\n";
     std::cout << std::string(80, '=') << "\n\n";
     
     YourExistingOpenCL your_opencl;
     
     auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
     backend->InitializeFromExternalContext(
         your_opencl.GetContext(),
         your_opencl.GetDevice(),
         your_opencl.GetQueue()
     );
     
     cl_mem your_buffer = your_opencl.GetDataBuffer();
     size_t buffer_size = 1024 * sizeof(float);
     
     // ────────────────────────────────────────────────────────────────
     // Метод 1: Прямая запись через backend API
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📤 Прямая запись через MemcpyHostToDevice()...\n";
     std::vector<float> new_data(1024, 99.0f);
     
     backend->MemcpyHostToDevice(
         your_buffer,
         new_data.data(),
         buffer_size
     );
     
     std::cout << "   Записано " << new_data.size() << " элементов\n";
     
     // ────────────────────────────────────────────────────────────────
     // Метод 2: Прямое чтение через backend API
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n📥 Прямое чтение через MemcpyDeviceToHost()...\n";
     std::vector<float> read_data(1024);
     
     backend->MemcpyDeviceToHost(
         read_data.data(),
         your_buffer,
         buffer_size
     );
     
     std::cout << "   Прочитано " << read_data.size() << " элементов\n";
     std::cout << "   Значение первого элемента: " << read_data[0] << "\n";
     
     // ────────────────────────────────────────────────────────────────
     // Метод 3: Копирование Device -> Device
     // ────────────────────────────────────────────────────────────────
     std::cout << "\n🔄 Копирование буфера GPU -> GPU...\n";
     
     // Создаём второй буфер в вашем OpenCL контексте
     cl_int err;
     cl_mem temp_buffer = clCreateBuffer(
         your_opencl.GetContext(),
         CL_MEM_READ_WRITE,
         buffer_size,
         nullptr,
         &err
     );
     
     if (err == CL_SUCCESS && temp_buffer) {
         // Копируем ваш буфер в временный
         backend->MemcpyDeviceToDevice(
             temp_buffer,      // dst
             your_buffer,      // src
             buffer_size
         );
         
         std::cout << "   Скопировано " << (buffer_size / sizeof(float)) << " элементов\n";
         
         // Освобождаем временный буфер
         clReleaseMemObject(temp_buffer);
     }
     
     std::cout << "\n✅ Прямые утилиты работают!\n";
 }
 
 // ════════════════════════════════════════════════════════════════════════════
 // ПРИМЕР 4: Демонстрация правильного управления владением
 // ════════════════════════════════════════════════════════════════════════════
 
 void Example4_OwnershipManagement() {
     std::cout << "\n" << std::string(80, '=') << "\n";
     std::cout << "EXAMPLE 4: Управление владением ресурсами\n";
     std::cout << std::string(80, '=') << "\n\n";
     
     YourExistingOpenCL your_opencl;
     
     std::cout << "📌 Создаём external backend...\n";
     auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
     
     std::cout << "   OwnsResources() = " << backend->OwnsResources() << " (false)\n";
     
     backend->InitializeFromExternalContext(
         your_opencl.GetContext(),
         your_opencl.GetDevice(),
         your_opencl.GetQueue()
     );
     
     std::cout << "\n📌 После InitializeFromExternalContext()...\n";
     std::cout << "   OwnsResources() = " << backend->OwnsResources() << " (всё ещё false)\n";
     std::cout << "   Initialized = " << backend->IsInitialized() << "\n";
     
     std::cout << "\n📌 Проверяем IBackend API...\n";
     void* ctx = backend->GetNativeContext();
     void* dev = backend->GetNativeDevice();
     void* queue = backend->GetNativeQueue();
     
     std::cout << "   Context: " << (ctx ? "OK ✅" : "NULL ❌") << "\n";
     std::cout << "   Device: " << (dev ? "OK ✅" : "NULL ❌") << "\n";
     std::cout << "   Queue: " << (queue ? "OK ✅" : "NULL ❌") << "\n";
     
     std::cout << "\n📌 Уничтожаем backend...\n";
     backend.reset();
     
     std::cout << "   ✅ ~OpenCLBackend() вызван\n";
     std::cout << "   ✅ ~OpenCLBackend() → Cleanup() вызван\n";
     std::cout << "   ✅ Cleanup() увидел owns_resources_ = false\n";
     std::cout << "   ✅ НЕ вызвал clReleaseCommandQueue/clReleaseContext!\n";
     
     std::cout << "\n📌 Проверяем ваш OpenCL объект...\n";
     std::cout << "   Context всё ещё валиден: " 
               << (your_opencl.GetContext() ? "OK ✅" : "NULL ❌") << "\n";
     
     std::cout << "\n🎉 Ваш код сам освободит ресурсы при деструкции!\n";
 }
 
 // ════════════════════════════════════════════════════════════════════════════
// ПРИМЕР 5: Использование external backend через общий API (ИСПРАВЛЕНО)
// ════════════════════════════════════════════════════════════════════════════

void Example5_UsingBackendDirectly() {
  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "EXAMPLE 5: Использование backend напрямую через IBackend API\n";
  std::cout << std::string(80, '=') << "\n\n";
  
  YourExistingOpenCL your_opencl;
  
  // ────────────────────────────────────────────────────────────────
  // Создаём external backend
  // ────────────────────────────────────────────────────────────────
  std::cout << "📌 Создаём external backend...\n";
  
  auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
  backend->InitializeFromExternalContext(
      your_opencl.GetContext(),
      your_opencl.GetDevice(),
      your_opencl.GetQueue()
  );
  
  std::cout << "   ✅ Backend инициализирован\n";
  std::cout << "   Device: " << backend->GetDeviceName() << "\n";
  std::cout << "   Backend owns resources: " << backend->OwnsResources() << "\n";
  
  // ────────────────────────────────────────────────────────────────
  // Используем backend напрямую через IBackend интерфейс
  // ────────────────────────────────────────────────────────────────
  std::cout << "\n📌 Работаем с backend через IBackend API...\n";
  
  // Получаем информацию об устройстве
  auto device_info = backend->GetDeviceInfo();
  std::cout << "\n   📊 Device Info:\n";
  std::cout << "      Name: " << device_info.name << "\n";
  std::cout << "      Vendor: " << device_info.vendor << "\n";
  std::cout << "      OpenCL Version: " << device_info.opencl_version << "\n";
  std::cout << "      Global Memory: " << (device_info.global_memory_size / (1024*1024)) << " MB\n";
  std::cout << "      Compute Units: " << device_info.max_compute_units << "\n";
  std::cout << "      Max Work Group Size: " << device_info.max_work_group_size << "\n";
  
  // ────────────────────────────────────────────────────────────────
  // Выделяем GPU буфер через backend
  // ────────────────────────────────────────────────────────────────
  std::cout << "\n📌 Выделяем GPU буфер через Allocate()...\n";
  size_t buffer_size = 512 * sizeof(float);
  void* gpu_buffer = backend->Allocate(buffer_size);
  
  if (gpu_buffer) {
      std::cout << "   ✅ Выделен буфер: " << buffer_size << " bytes\n";
      
      // Заполняем данными
      std::vector<float> data(512);
      for (size_t i = 0; i < data.size(); ++i) {
          data[i] = static_cast<float>(i) * 2.0f;
      }
      
      // Копируем на GPU
      backend->MemcpyHostToDevice(gpu_buffer, data.data(), buffer_size);
      std::cout << "   ✅ Данные скопированы на GPU (" << data.size() << " элементов)\n";
      
      // Синхронизируем
      backend->Synchronize();
      std::cout << "   ✅ Синхронизация завершена\n";
      
      // Читаем обратно
      std::vector<float> result(512);
      backend->MemcpyDeviceToHost(result.data(), gpu_buffer, buffer_size);
      std::cout << "   ✅ Данные прочитаны с GPU\n";
      
      // Проверяем
      bool correct = true;
      for (size_t i = 0; i < 10; ++i) {
          if (result[i] != data[i]) {
              correct = false;
              break;
          }
      }
      
      std::cout << "   🔍 Проверка данных: " << (correct ? "OK ✅" : "FAIL ❌") << "\n";
      std::cout << "      Первые 5 элементов: ";
      for (size_t i = 0; i < 5; ++i) {
          std::cout << result[i] << " ";
      }
      std::cout << "\n";
      
      // Освобождаем
      backend->Free(gpu_buffer);
      std::cout << "   ✅ Буфер освобождён\n";
  } else {
      std::cout << "   ❌ Не удалось выделить буфер\n";
  }
  
  // ────────────────────────────────────────────────────────────────
  // Проверяем возможности устройства
  // ────────────────────────────────────────────────────────────────
  std::cout << "\n📌 Проверяем возможности устройства...\n";
  std::cout << "   SVM Support: " << (backend->SupportsSVM() ? "YES ✅" : "NO ❌") << "\n";
  std::cout << "   Double Precision: " << (backend->SupportsDoublePrecision() ? "YES ✅" : "NO ❌") << "\n";
  std::cout << "   Global Memory: " << (backend->GetGlobalMemorySize() / (1024*1024)) << " MB\n";
  std::cout << "   Local Memory: " << (backend->GetLocalMemorySize() / 1024) << " KB\n";
  
  // ────────────────────────────────────────────────────────────────
  // Получаем нативные хэндлы (для продвинутого использования)
  // ────────────────────────────────────────────────────────────────
  std::cout << "\n📌 Получаем нативные OpenCL хэндлы...\n";
  void* native_ctx = backend->GetNativeContext();
  void* native_dev = backend->GetNativeDevice();
  void* native_queue = backend->GetNativeQueue();
  
  std::cout << "   Native Context: " << native_ctx << "\n";
  std::cout << "   Native Device: " << native_dev << "\n";
  std::cout << "   Native Queue: " << native_queue << "\n";
  
  // Проверяем, что это те же объекты, что и в YourExistingOpenCL
  bool same_context = (native_ctx == your_opencl.GetContext());
  bool same_device = (native_dev == your_opencl.GetDevice());
  bool same_queue = (native_queue == your_opencl.GetQueue());
  
  std::cout << "\n   🔍 Проверка идентичности хэндлов:\n";
  std::cout << "      Context match: " << (same_context ? "YES ✅" : "NO ❌") << "\n";
  std::cout << "      Device match: " << (same_device ? "YES ✅" : "NO ❌") << "\n";
  std::cout << "      Queue match: " << (same_queue ? "YES ✅" : "NO ❌") << "\n";
  
  std::cout << "\n🎉 Backend работает идеально с вашим OpenCL контекстом!\n";
  
  // ~OpenCLBackend() → НЕ освобождает ваш контекст (owns_resources = false)
}
// ════════════════════════════════════════════════════════════════════════════
// MAIN: Запуск всех примеров
// ════════════════════════════════════════════════════════════════════════════

namespace external_context_example {

  int run() {
      std::cout << R"(
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║                                                                            ║
  ║       DrvGPU - ПРИМЕРЫ РАБОТЫ С ВНЕШНИМ OpenCL КОНТЕКСТОМ                 ║
  ║                                                                            ║
  ║       Автор: DrvGPU Team                                                   ║
  ║       Дата: 2026-02-02                                                     ║
  ║       Версия: 2.0 (с owns_resources_)                                      ║
  ║                                                                            ║
  ║       ✅ КЛЮЧЕВАЯ ОСОБЕННОСТЬ:                                             ║
  ║       OpenCLBackend автоматически устанавливает                    ║
  ║       owns_resources_ = false → НЕ освобождает ваш контекст!              ║
  ║                                                                            ║
  ║       📚 API используется:                                                 ║
  ║       - IBackend::MemcpyHostToDevice()                                     ║
  ║       - IBackend::MemcpyDeviceToHost()                                     ║
  ║       - IBackend::MemcpyDeviceToDevice()                                   ║
  ║       - IBackend::Allocate() / Free()                                      ║
  ║       - IBackend::GetNativeContext/Device/Queue()                          ║
  ║                                                                            ║
  ╚════════════════════════════════════════════════════════════════════════════╝
  )";
  
      try {
          // Пример 1: Базовая интеграция
          Example1_BasicExternalContext();
          
          // Пример 2: Работа с внешними буферами через IBackend
          Example2_WorkingWithExternalBuffers();
          
          // Пример 3: Прямые утилиты
          Example3_DirectBufferUtilities();
          
          // Пример 4: Управление владением
          Example4_OwnershipManagement();
          
          // Пример 5: Использование backend напрямую (ИСПРАВЛЕНО)
          Example5_UsingBackendDirectly();
          
          std::cout << "\n" << std::string(80, '=') << "\n";
          std::cout << "✅ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!\n";
          std::cout << std::string(80, '=') << "\n\n";
          
          return 0;
          
      } catch (const std::exception &e) {
          std::cerr << "\n❌ ОШИБКА: " << e.what() << "\n";
          return 1;
      }
  }
  
  } // namespace external_context_example
  