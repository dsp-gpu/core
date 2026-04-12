/**
 * @file test_clmem_gpu_va_probe.cpp
 * @brief Поиск GPU VA внутри cl_mem handle (ROCm 7.2, CLR internals)
 *
 * Идея: cl_mem — это указатель на C++ объект amd::Memory.
 * Внутри него есть поле deviceMemory_ (void*) — GPU virtual address.
 * Мы сканируем структуру, ищем значения похожие на GPU VA,
 * проверяем через hsa_amd_pointer_info.
 *
 * Компиляция:
 *   g++ test_clmem_gpu_va_probe.cpp -o test_clmem_gpu_va_probe \
 *       -D__HIP_PLATFORM_AMD__ -DCL_TARGET_OPENCL_VERSION=220 \
 *       -I/opt/rocm/include -L/opt/rocm/lib \
 *       -lOpenCL -lamdhip64 -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib -std=c++17
 */

#include <CL/cl.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
  printf("═══════════════════════════════════════════════════════\n");
  printf("  cl_mem GPU VA Probe (ROCm 7.2, gfx1201)\n");
  printf("═══════════════════════════════════════════════════════\n\n");

  // --- Init OpenCL ---
  cl_platform_id platform;
  cl_device_id device;
  cl_int err;
  clGetPlatformIDs(1, &platform, nullptr);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
  cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);

  // --- Init HSA ---
  hsa_init();

  // --- Создаём cl_mem с известными данными ---
  const size_t N = 1024;
  const size_t sz = N * sizeof(float);

  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sz, nullptr, &err);
  printf("cl_mem handle: %p\n", (void*)buf);

  // Записываем паттерн: 1.0, 2.0, 3.0, 4.0 ...
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = 1.0f + (float)i;
  clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, sz, input.data(), 0, nullptr, nullptr);
  clFinish(queue);

  // --- Сканируем cl_mem handle как массив void* ---
  // cl_mem — указатель на объект ~500-2000 байт.
  // Ищем поля которые выглядят как GPU VA (проверяем через HSA).

  printf("\nСканирование cl_mem объекта (каждые 8 байт, первые 2048 байт):\n");
  printf("%-8s  %-18s  %-6s  %-18s  %-18s  %s\n",
         "Offset", "Value", "Type", "AgentBase", "HostBase", "Size");
  printf("─────────────────────────────────────────────────────────────────────────────\n");

  uint8_t* raw = (uint8_t*)buf;
  int found_count = 0;
  int gpu_va_offset = -1;

  for (int offset = 0; offset < 2048; offset += 8) {
    void* val = *(void**)(raw + offset);

    // Пропускаем null, маленькие значения и очевидно невалидные
    if (!val || (uintptr_t)val < 0x1000) continue;

    // Спрашиваем HSA: знает ли он этот указатель?
    hsa_amd_pointer_info_t info = {};
    info.size = sizeof(info);
    hsa_status_t hsa_err = hsa_amd_pointer_info(val, &info, nullptr, nullptr, nullptr);

    if (hsa_err == HSA_STATUS_SUCCESS && info.type != 0) {
      // HSA знает этот указатель!
      const char* type_str = "?";
      switch (info.type) {
        case 1: type_str = "HSA"; break;
        case 2: type_str = "LOCKED"; break;
        case 3: type_str = "GFX"; break;
        case 4: type_str = "IPC"; break;
        default: type_str = "OTHER"; break;
      }

      printf("  +%-4d   0x%-16lx  %-6s  0x%-16lx  0x%-16lx  %zu\n",
             offset, (uintptr_t)val, type_str,
             (uintptr_t)info.agentBaseAddress,
             (uintptr_t)info.hostBaseAddress,
             info.sizeInBytes);

      // Если размер совпадает с нашим буфером и тип HSA — это кандидат!
      if (info.sizeInBytes == sz && info.type == 1) {
        printf("    ^^^ КАНДИДАТ: размер %zu совпадает с буфером! ^^^\n", sz);
        gpu_va_offset = offset;
      }

      found_count++;
    }
  }

  printf("\nНайдено HSA-указателей: %d\n", found_count);

  // --- Если нашли GPU VA — проверяем данные через HIP ---
  if (gpu_va_offset >= 0) {
    void* gpu_va = *(void**)(raw + gpu_va_offset);
    printf("\n══════════════════════════════════════════════════\n");
    printf("  ПРОВЕРКА GPU VA (offset +%d = %p)\n", gpu_va_offset, gpu_va);
    printf("══════════════════════════════════════════════════\n");

    // Читаем через hipMemcpy
    std::vector<float> result(N, 0.0f);
    hipError_t herr = hipMemcpy(result.data(), gpu_va, sz, hipMemcpyDeviceToHost);
    printf("  hipMemcpy: %s (err=%d)\n",
           herr == hipSuccess ? "OK" : hipGetErrorString(herr), herr);

    if (herr == hipSuccess) {
      // Проверяем данные
      float max_err = 0;
      for (size_t i = 0; i < N; i++) {
        float diff = fabsf(result[i] - input[i]);
        if (diff > max_err) max_err = diff;
      }
      printf("  Данные: [%.0f, %.0f, %.0f, %.0f, ...]\n",
             result[0], result[1], result[2], result[3]);
      printf("  Max error: %.2e\n", max_err);

      if (max_err < 1e-6f) {
        printf("\n  ✅ GPU VA НАЙДЕН! offset=%d внутри cl_mem\n", gpu_va_offset);
        printf("  ✅ cl_mem данные читаются через HIP — TRUE ZERO-COPY!\n");

        // Пробуем dma-buf export
        int fd = -1;
        uint64_t dmabuf_offset = 0;
        hsa_amd_pointer_info_t info = {};
        info.size = sizeof(info);
        hsa_amd_pointer_info(gpu_va, &info, nullptr, nullptr, nullptr);

        hsa_status_t hs = hsa_amd_portable_export_dmabuf(
            info.agentBaseAddress, info.sizeInBytes, &fd, &dmabuf_offset);
        printf("  dma-buf export: status=%d, fd=%d\n", hs, fd);
        if (fd >= 0) {
          printf("  ✅ DMA-BUF export тоже работает!\n");
          hsa_amd_portable_close_dmabuf(fd);
        }
      } else {
        printf("  ❌ Данные не совпадают — это не тот указатель\n");
      }
    }
  } else {
    printf("\n❌ GPU VA с совпадающим размером не найден.\n");
    printf("   Попробуем искать по ЛЮБОМУ HSA типу...\n");
  }

  // --- Cleanup ---
  clReleaseMemObject(buf);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
  hsa_shut_down();

  return 0;
}
