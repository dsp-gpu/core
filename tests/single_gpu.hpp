/**
 * @file example_single_gpu.cpp
 * @brief Пример использования DrvGPU для Single GPU
 *
 * Демонстрирует базовое использование DrvGPU с одной GPU.
 */

#include <core/drv_gpu.hpp>
#include <core/common/backend_type.hpp>
#include <iostream>
#include <vector>

using namespace drv_gpu_lib;

namespace example_drv_gpu_singl
{
  int run()
  {
    try
    {
      std::cout << "=== DrvGPU Single GPU Example ===\n\n";

      // ═══════════════════════════════════════════════════════════════
      // 1. Создать и инициализировать DrvGPU
      // ═══════════════════════════════════════════════════════════════

      std::cout << "Initializing DrvGPU with ROCm backend...\n";
      DrvGPU gpu(BackendType::ROCm, 0); // GPU #0
      gpu.Initialize();

      // ═══════════════════════════════════════════════════════════════
      // 2. Вывести информацию об устройстве
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\nDevice Information:\n";
      gpu.PrintDeviceInfo();

      auto device_info = gpu.GetDeviceInfo();
      std::cout << "\nDevice: " << device_info.name << "\n";
      std::cout << "Memory: " << device_info.GetGlobalMemoryGB() << " GB\n";
      std::cout << "Compute Units: " << device_info.max_compute_units << "\n";
      std::cout << "SVM Support: " << (device_info.supports_svm ? "Yes" : "No") << "\n";

      // ═══════════════════════════════════════════════════════════════
      // 3. Работа с памятью - создать буфер
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n--- Memory Management ---\n";

      MemoryManager &mem_mgr = gpu.GetMemoryManager();

      // Создать буфер для 1024 float элементов
      const size_t N = 1024;
      auto buffer = mem_mgr.CreateBuffer<float>(N);

      std::cout << "Created buffer: " << N << " elements\n";

      // ═══════════════════════════════════════════════════════════════
      // 4. Записать данные Host -> Device
      // ═══════════════════════════════════════════════════════════════

      std::vector<float> host_data(N);
      for (size_t i = 0; i < N; ++i)
      {
        host_data[i] = static_cast<float>(i);
      }

      buffer->Write(host_data);
      std::cout << "Written " << N << " elements to GPU\n";

      // ═══════════════════════════════════════════════════════════════
      // 5. Прочитать данные Device -> Host
      // ═══════════════════════════════════════════════════════════════

      auto result = buffer->Read();
      std::cout << "Read " << result.size() << " elements from GPU\n";

      // Проверить первые и последние элементы
      std::cout << "First element [0]: " << result[0] << "\n";
      std::cout << "-  element [1]: " << result[1] << "\n";
      std::cout << "-  element [2]: " << result[2] << "\n";
      std::cout << "-  element [3]: " << result[3] << "\n";
      std::cout << "-  element [4]: " << result[4] << "\n";
      std::cout << "Last element: " << result[N - 1] << "\n";

      // ═══════════════════════════════════════════════════════════════
      // 6. Статистика памяти
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n--- Memory Statistics ---\n";
      mem_mgr.PrintStatistics();

      // ═══════════════════════════════════════════════════════════════
      // 7. Синхронизация
      // ═══════════════════════════════════════════════════════════════

      gpu.Synchronize();
      std::cout << "\nGPU synchronized\n";

      // ═══════════════════════════════════════════════════════════════
      // 8. Очистка (автоматически в деструкторе)
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n=== Example completed successfully! ===\n";
    }
    catch (const std::exception &e)
    {
      std::cerr << "ERROR: " << e.what() << "\n";
      return 1;
    }

    return 0;
  }
}