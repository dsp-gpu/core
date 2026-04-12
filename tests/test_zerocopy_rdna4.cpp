/**
 * @file test_zerocopy_rdna4.cpp
 * @brief Диагностика: какие методы ZeroCopy работают на RDNA4 (gfx1201)
 *
 * Компиляция:
 *   g++ test_zerocopy_rdna4.cpp -o test_zerocopy_rdna4 \
 *       -I/opt/rocm/include -L/opt/rocm/lib \
 *       -lOpenCL -lamdhip64 -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib
 *
 * Запуск:
 *   ./test_zerocopy_rdna4
 *
 * Тестирует 4 подхода:
 *   1. Coarse-grain SVM → HIP напрямую (VRAM, true zero-copy)
 *   2. Fine-grain SVM → HIP напрямую (system RAM, работает но медленно)
 *   3. HSA dma-buf export (SVM ptr → dma-buf fd → hipImportExternalMemory)
 *   4. cl_mem → hsa_amd_pointer_info (попытка получить GPU VA из cl_mem)
 */

#include <CL/cl.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

static cl_context     g_ctx    = nullptr;
static cl_device_id   g_dev    = nullptr;
static cl_command_queue g_queue = nullptr;

static bool init_opencl() {
  cl_platform_id platform;
  cl_int err;
  clGetPlatformIDs(1, &platform, nullptr);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_dev, nullptr);

  char name[256] = {};
  clGetDeviceInfo(g_dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  printf("OpenCL device: %s\n", name);

  g_ctx = clCreateContext(nullptr, 1, &g_dev, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) { printf("clCreateContext err=%d\n", err); return false; }

  g_queue = clCreateCommandQueueWithProperties(g_ctx, g_dev, nullptr, &err);
  if (err != CL_SUCCESS) { printf("clCreateCommandQueue err=%d\n", err); return false; }

  return true;
}

static void cleanup_opencl() {
  if (g_queue) clReleaseCommandQueue(g_queue);
  if (g_ctx) clReleaseContext(g_ctx);
}

static bool verify_data(const float* data, size_t N, float start, float step) {
  float max_err = 0;
  for (size_t i = 0; i < N; i++) {
    float expected = start + step * i;
    float diff = fabsf(data[i] - expected);
    if (diff > max_err) max_err = diff;
  }
  printf("    max error: %.2e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");
  return max_err < 1e-5f;
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Coarse-grain SVM → HIP (VRAM на dGPU)
// ═══════════════════════════════════════════════════════════════════════════

static bool test_coarse_grain_svm_to_hip() {
  printf("\n[TEST 1] Coarse-grain SVM -> HIP (VRAM, true zero-copy)\n");

  const size_t N = 4096;
  const size_t sz = N * sizeof(float);

  // Coarse-grain SVM: CL_MEM_READ_WRITE без CL_MEM_SVM_FINE_GRAIN_BUFFER
  // На dGPU → аллоцирует в VRAM
  void* svm = clSVMAlloc(g_ctx, CL_MEM_READ_WRITE, sz, 0);
  if (!svm) {
    printf("  clSVMAlloc (coarse-grain) failed — device may not support coarse-grain SVM\n");
    return false;
  }
  printf("  clSVMAlloc (coarse-grain): %p\n", svm);

  // Записать данные через map (coarse-grain требует map/unmap)
  cl_int err = clEnqueueSVMMap(g_queue, CL_TRUE, CL_MAP_WRITE, svm, sz, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    printf("  clEnqueueSVMMap err=%d\n", err);
    clSVMFree(g_ctx, svm);
    return false;
  }

  float* fdata = (float*)svm;
  for (size_t i = 0; i < N; i++) fdata[i] = 1.0f + 0.5f * i;

  err = clEnqueueSVMUnmap(g_queue, svm, 0, nullptr, nullptr);
  clFinish(g_queue);

  // HIP: прочитать из SVM указателя напрямую
  std::vector<float> result(N, 0.0f);
  hipError_t herr = hipMemcpy(result.data(), svm, sz, hipMemcpyDeviceToHost);
  printf("  hipMemcpy from coarse-grain SVM: %s (err=%d)\n",
         herr == hipSuccess ? "OK" : hipGetErrorString(herr), herr);

  bool ok = false;
  if (herr == hipSuccess) {
    ok = verify_data(result.data(), 4, 1.0f, 0.5f);
  }

  clSVMFree(g_ctx, svm);
  printf("  [TEST 1] %s\n", ok ? "PASSED ✅" : "FAILED ❌");
  return ok;
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Fine-grain SVM → HIP (system RAM)
// ═══════════════════════════════════════════════════════════════════════════

static bool test_fine_grain_svm_to_hip() {
  printf("\n[TEST 2] Fine-grain SVM -> HIP (system RAM)\n");

  const size_t N = 4096;
  const size_t sz = N * sizeof(float);

  void* svm = clSVMAlloc(g_ctx, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE, sz, 0);
  if (!svm) {
    printf("  clSVMAlloc (fine-grain) failed\n");
    return false;
  }
  printf("  clSVMAlloc (fine-grain): %p\n", svm);

  // Fine-grain: прямой доступ CPU без map
  float* fdata = (float*)svm;
  for (size_t i = 0; i < N; i++) fdata[i] = 2.0f + 0.25f * i;

  // HIP: прочитать
  std::vector<float> result(N, 0.0f);
  hipError_t herr = hipMemcpy(result.data(), svm, sz, hipMemcpyDeviceToHost);
  printf("  hipMemcpy from fine-grain SVM: %s (err=%d)\n",
         herr == hipSuccess ? "OK" : hipGetErrorString(herr), herr);

  bool ok = false;
  if (herr == hipSuccess) {
    ok = verify_data(result.data(), 4, 2.0f, 0.25f);
  }

  clSVMFree(g_ctx, svm);
  printf("  [TEST 2] %s\n", ok ? "PASSED ✅" : "FAILED ❌");
  return ok;
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: HSA dma-buf export (SVM → dma-buf fd → hipImportExternalMemory)
// ═══════════════════════════════════════════════════════════════════════════

static bool test_hsa_dmabuf_export() {
  printf("\n[TEST 3] HSA dma-buf export (SVM -> dma-buf -> HIP)\n");

  // Инициализация HSA
  hsa_status_t hsa_err = hsa_init();
  if (hsa_err != HSA_STATUS_SUCCESS) {
    printf("  hsa_init failed: %d\n", hsa_err);
    return false;
  }

  const size_t N = 4096;
  const size_t sz = N * sizeof(float);

  // Аллоцируем coarse-grain SVM
  void* svm = clSVMAlloc(g_ctx, CL_MEM_READ_WRITE, sz, 0);
  if (!svm) {
    printf("  clSVMAlloc failed\n");
    hsa_shut_down();
    return false;
  }
  printf("  SVM ptr: %p\n", svm);

  // Записать данные
  cl_int err = clEnqueueSVMMap(g_queue, CL_TRUE, CL_MAP_WRITE, svm, sz, 0, nullptr, nullptr);
  float* fdata = (float*)svm;
  for (size_t i = 0; i < N; i++) fdata[i] = 3.0f + 0.1f * i;
  clEnqueueSVMUnmap(g_queue, svm, 0, nullptr, nullptr);
  clFinish(g_queue);

  // Запросить HSA pointer info
  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(info);
  hsa_err = hsa_amd_pointer_info(svm, &info, nullptr, nullptr, nullptr);
  printf("  hsa_amd_pointer_info: status=%d, type=%d\n", hsa_err, info.type);
  if (hsa_err == HSA_STATUS_SUCCESS) {
    printf("    agentBaseAddress: %p\n", info.agentBaseAddress);
    printf("    hostBaseAddress:  %p\n", info.hostBaseAddress);
    printf("    sizeInBytes:      %zu\n", info.sizeInBytes);
  }

  // Экспорт как dma-buf
  int dmabuf_fd = -1;
  uint64_t offset = 0;
  void* export_ptr = info.agentBaseAddress ? info.agentBaseAddress : svm;
  size_t export_size = info.sizeInBytes > 0 ? info.sizeInBytes : sz;

  hsa_err = hsa_amd_portable_export_dmabuf(export_ptr, export_size, &dmabuf_fd, &offset);
  printf("  hsa_amd_portable_export_dmabuf: status=%d, fd=%d, offset=%lu\n",
         hsa_err, dmabuf_fd, offset);

  bool ok = false;
  if (hsa_err == HSA_STATUS_SUCCESS && dmabuf_fd >= 0) {
    // Импорт в HIP
    hipExternalMemoryHandleDesc ext_desc = {};
    ext_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    ext_desc.handle.fd = dmabuf_fd;
    ext_desc.size = export_size;

    hipExternalMemory_t ext_mem = nullptr;
    hipError_t herr = hipImportExternalMemory(&ext_mem, &ext_desc);
    printf("  hipImportExternalMemory: %s (err=%d)\n",
           herr == hipSuccess ? "OK" : hipGetErrorString(herr), herr);

    if (herr == hipSuccess) {
      hipExternalMemoryBufferDesc buf_desc = {};
      buf_desc.offset = offset;
      buf_desc.size = sz;
      void* hip_ptr = nullptr;
      herr = hipExternalMemoryGetMappedBuffer(&hip_ptr, ext_mem, &buf_desc);
      printf("  hipExternalMemoryGetMappedBuffer: %s, hip_ptr=%p\n",
             herr == hipSuccess ? "OK" : hipGetErrorString(herr), hip_ptr);

      if (herr == hipSuccess && hip_ptr) {
        std::vector<float> result(N, 0.0f);
        herr = hipMemcpy(result.data(), hip_ptr, sz, hipMemcpyDeviceToHost);
        if (herr == hipSuccess) {
          ok = verify_data(result.data(), 4, 3.0f, 0.1f);
        }
      }

      if (ext_mem) hipDestroyExternalMemory(ext_mem);
    }

    hsa_amd_portable_close_dmabuf(dmabuf_fd);
  }

  clSVMFree(g_ctx, svm);
  hsa_shut_down();
  printf("  [TEST 3] %s\n", ok ? "PASSED ✅" : "FAILED ❌");
  return ok;
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: cl_mem → hsa_amd_pointer_info (попытка получить GPU VA)
// ═══════════════════════════════════════════════════════════════════════════

static bool test_clmem_hsa_pointer_info() {
  printf("\n[TEST 4] cl_mem -> hsa_amd_pointer_info (can we get GPU VA?)\n");

  hsa_status_t hsa_err = hsa_init();
  if (hsa_err != HSA_STATUS_SUCCESS) {
    printf("  hsa_init failed\n");
    return false;
  }

  const size_t N = 4096;
  const size_t sz = N * sizeof(float);

  // Обычный cl_mem (clCreateBuffer)
  cl_int cl_err;
  cl_mem buf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, sz, nullptr, &cl_err);
  if (cl_err != CL_SUCCESS) {
    printf("  clCreateBuffer failed: %d\n", cl_err);
    hsa_shut_down();
    return false;
  }

  // Записать данные
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = 4.0f + 0.3f * i;
  clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, sz, input.data(), 0, nullptr, nullptr);

  // Попытка 1: hsa_amd_pointer_info на cl_mem handle (маловероятно)
  printf("  Attempt 1: hsa_amd_pointer_info on cl_mem handle (%p)\n", (void*)buf);
  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(info);
  hsa_err = hsa_amd_pointer_info((void*)buf, &info, nullptr, nullptr, nullptr);
  printf("    status=%d, type=%d, agentBase=%p, hostBase=%p\n",
         hsa_err, info.type, info.agentBaseAddress, info.hostBaseAddress);

  // Попытка 2: map cl_mem → hsa_amd_pointer_info на mapped ptr
  void* mapped = clEnqueueMapBuffer(g_queue, buf, CL_TRUE, CL_MAP_READ, 0, sz,
                                     0, nullptr, nullptr, &cl_err);
  printf("  Attempt 2: hsa_amd_pointer_info on mapped ptr (%p)\n", mapped);

  if (mapped) {
    memset(&info, 0, sizeof(info));
    info.size = sizeof(info);
    hsa_err = hsa_amd_pointer_info(mapped, &info, nullptr, nullptr, nullptr);
    printf("    status=%d, type=%d, agentBase=%p, hostBase=%p, size=%zu\n",
           hsa_err, info.type, info.agentBaseAddress, info.hostBaseAddress, info.sizeInBytes);

    // Если тип HSA — попробуем export dma-buf
    if (hsa_err == HSA_STATUS_SUCCESS && info.type != 0) {  // type != UNKNOWN
      void* export_ptr = info.agentBaseAddress ? info.agentBaseAddress : mapped;
      size_t export_size = info.sizeInBytes > 0 ? info.sizeInBytes : sz;

      int fd = -1;
      uint64_t offset = 0;
      hsa_err = hsa_amd_portable_export_dmabuf(export_ptr, export_size, &fd, &offset);
      printf("    dma-buf export: status=%d, fd=%d, offset=%lu\n", hsa_err, fd, offset);

      if (hsa_err == HSA_STATUS_SUCCESS && fd >= 0) {
        // Импорт в HIP
        hipExternalMemoryHandleDesc ext_desc = {};
        ext_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        ext_desc.handle.fd = fd;
        ext_desc.size = export_size;

        hipExternalMemory_t ext_mem = nullptr;
        hipError_t herr = hipImportExternalMemory(&ext_mem, &ext_desc);
        printf("    hipImportExternalMemory: %s\n",
               herr == hipSuccess ? "OK" : hipGetErrorString(herr));

        if (herr == hipSuccess) {
          hipExternalMemoryBufferDesc buf_desc = {};
          buf_desc.offset = offset;
          buf_desc.size = sz;
          void* hip_ptr = nullptr;
          herr = hipExternalMemoryGetMappedBuffer(&hip_ptr, ext_mem, &buf_desc);
          printf("    hip_ptr: %p\n", hip_ptr);

          if (herr == hipSuccess && hip_ptr) {
            std::vector<float> result(N, 0.0f);
            herr = hipMemcpy(result.data(), hip_ptr, sz, hipMemcpyDeviceToHost);
            if (herr == hipSuccess) {
              verify_data(result.data(), 4, 4.0f, 0.3f);
            }
          }
          if (ext_mem) hipDestroyExternalMemory(ext_mem);
        }

        hsa_amd_portable_close_dmabuf(fd);
      }
    }

    clEnqueueUnmapMemObject(g_queue, buf, mapped, 0, nullptr, nullptr);
    clFinish(g_queue);
  }

  // Попытка 3: hipMemcpy напрямую от mapped ptr (SVM через HSA)
  mapped = clEnqueueMapBuffer(g_queue, buf, CL_TRUE, CL_MAP_READ, 0, sz,
                               0, nullptr, nullptr, &cl_err);
  if (mapped) {
    printf("  Attempt 3: hipMemcpy directly from mapped ptr\n");
    std::vector<float> result(N, 0.0f);
    hipError_t herr = hipMemcpy(result.data(), mapped, sz, hipMemcpyDefault);
    printf("    hipMemcpy (hipMemcpyDefault): %s\n",
           herr == hipSuccess ? "OK" : hipGetErrorString(herr));
    if (herr == hipSuccess) {
      verify_data(result.data(), 4, 4.0f, 0.3f);
    }

    clEnqueueUnmapMemObject(g_queue, buf, mapped, 0, nullptr, nullptr);
    clFinish(g_queue);
  }

  clReleaseMemObject(buf);
  hsa_shut_down();
  printf("  [TEST 4] diagnostic only — see results above\n");
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
  printf("═══════════════════════════════════════════════════════\n");
  printf("  ZeroCopy RDNA4 Diagnostic (gfx1201)\n");
  printf("═══════════════════════════════════════════════════════\n");

  if (!init_opencl()) {
    printf("OpenCL init failed!\n");
    return 1;
  }

  // SVM capabilities
  cl_device_svm_capabilities svm_caps = 0;
  clGetDeviceInfo(g_dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
  printf("\nSVM capabilities: 0x%x\n", (unsigned)svm_caps);
  printf("  Coarse-grain buffer: %s\n", (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) ? "YES" : "NO");
  printf("  Fine-grain buffer:   %s\n", (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? "YES" : "NO");
  printf("  Fine-grain system:   %s\n", (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) ? "YES" : "NO");
  printf("  Atomics:             %s\n", (svm_caps & CL_DEVICE_SVM_ATOMICS) ? "YES" : "NO");

  int passed = 0, total = 0;

  total++; if (test_coarse_grain_svm_to_hip())  passed++;
  total++; if (test_fine_grain_svm_to_hip())     passed++;
  total++; if (test_hsa_dmabuf_export())         passed++;
  test_clmem_hsa_pointer_info();  // diagnostic

  printf("\n═══════════════════════════════════════════════════════\n");
  printf("  Results: %d/%d PASSED\n", passed, total);
  printf("═══════════════════════════════════════════════════════\n");

  cleanup_opencl();
  return (passed == total) ? 0 : 1;
}
