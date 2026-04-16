/**
 * @file hsa_interop.cpp
 * @brief Реализация ProbeGpuVA и ExportGpuVAasDmaBuf
 *
 * Вынесено из hsa_interop.hpp для сокращения header bloat.
 * Логика не изменена — чистый рефакторинг.
 *
 * @see include/core/backends/rocm/hsa_interop.hpp
 * @author Кодо (AI Assistant)
 * @date 2026-04-16
 */

#if ENABLE_ROCM

#include <core/backends/rocm/hsa_interop.hpp>

namespace drv_gpu_lib {

HsaProbeResult ProbeGpuVA(cl_mem cl_buffer, size_t expected_size) {
  HsaProbeResult result;

  if (!cl_buffer || expected_size == 0) return result;
  if (!EnsureHsaInitialized()) return result;

  // Кеш offset: -1 = не определён, -2 = probe не нашёл (не пробовать снова)
  static std::atomic<int> cached_offset{-1};
  static std::mutex probe_mutex;

  int off = cached_offset.load(std::memory_order_acquire);

  // Быстрый путь: offset уже известен
  if (off >= 0) {
    auto* raw = reinterpret_cast<uint8_t*>(cl_buffer);
    void* val = *reinterpret_cast<void**>(raw + off);
    if (val && reinterpret_cast<uintptr_t>(val) % 4096 == 0) {
      hsa_amd_pointer_info_t info = {};
      info.size = sizeof(info);
      if (hsa_amd_pointer_info(val, &info, nullptr, nullptr, nullptr) == HSA_STATUS_SUCCESS
          && info.type == HSA_EXT_POINTER_TYPE_HSA
          && info.sizeInBytes >= expected_size
          && info.agentBaseAddress == val) {
        result.gpu_va = val;
        result.alloc_size = info.sizeInBytes;
        result.offset = off;
        result.valid = true;
        return result;
      }
    }
    // Cached offset не подошёл для ЭТОГО буфера — НЕ сбрасываем кеш!
    // Другой поток может успешно использовать этот offset для другого буфера.
    // Просто переходим к полному скану ниже (под мьютексом).
  }

  // Если probe ранее провалился (-2), всё равно пробуем снова —
  // другой буфер может иметь другой размер/alignment


  // Полное сканирование (однократно)
  std::lock_guard<std::mutex> lock(probe_mutex);

  // Double-check после захвата мьютекса (с полной HSA валидацией)
  off = cached_offset.load(std::memory_order_acquire);
  if (off >= 0) {
    auto* raw = reinterpret_cast<uint8_t*>(cl_buffer);
    void* val = *reinterpret_cast<void**>(raw + off);
    if (val && reinterpret_cast<uintptr_t>(val) % 4096 == 0) {
      hsa_amd_pointer_info_t info = {};
      info.size = sizeof(info);
      if (hsa_amd_pointer_info(val, &info, nullptr, nullptr, nullptr) == HSA_STATUS_SUCCESS
          && info.type == HSA_EXT_POINTER_TYPE_HSA
          && info.sizeInBytes >= expected_size
          && info.agentBaseAddress == val) {
        result.gpu_va = val;
        result.alloc_size = info.sizeInBytes;
        result.offset = off;
        result.valid = true;
        return result;
      }
    }
  }

  // Сканируем cl_mem объект — собираем HSA-managed GPU VA кандидатов.
  auto* raw = reinterpret_cast<uint8_t*>(cl_buffer);
  int max_scan = kMaxProbeBytes;

  // Фаза 1: собрать все подходящие HSA-указатели
  struct Candidate { int offset; void* va; size_t size; };
  Candidate candidates[16];
  int n_candidates = 0;

  for (int probe_off = 0; probe_off < max_scan && n_candidates < 16;
       probe_off += kProbeStep) {
    void* val = *reinterpret_cast<void**>(raw + probe_off);

    if (!val || reinterpret_cast<uintptr_t>(val) < 0x10000) continue;

    // GPU VRAM аллокации page-aligned (4K minimum)
    if (reinterpret_cast<uintptr_t>(val) % 4096 != 0) continue;

    hsa_amd_pointer_info_t info = {};
    info.size = sizeof(info);
    hsa_status_t st = hsa_amd_pointer_info(val, &info, nullptr, nullptr, nullptr);

    if (st != HSA_STATUS_SUCCESS || info.type != HSA_EXT_POINTER_TYPE_HSA) continue;
    if (info.sizeInBytes < expected_size) continue;

    // Ищем начало аллокации (не offset внутри чужой)
    if (info.agentBaseAddress && info.agentBaseAddress != val) continue;

    // Дедупликация: один и тот же VA может быть в нескольких полях
    bool dup = false;
    for (int i = 0; i < n_candidates; i++) {
      if (candidates[i].va == val) { dup = true; break; }
    }
    if (dup) continue;

    candidates[n_candidates++] = {probe_off, val, info.sizeInBytes};
  }

  // Фаза 2: если один кандидат — берём его. Если несколько — выбираем по совпадению размера.
  if (n_candidates == 1) {
    cached_offset.store(candidates[0].offset, std::memory_order_release);
    result.gpu_va = candidates[0].va;
    result.alloc_size = candidates[0].size;
    result.offset = candidates[0].offset;
    result.valid = true;
    return result;
  }

  if (n_candidates > 1) {
    // Предпочитаем кандидата с точным размером
    for (int i = 0; i < n_candidates; i++) {
      if (candidates[i].size == expected_size) {
        cached_offset.store(candidates[i].offset, std::memory_order_release);
        result.gpu_va = candidates[i].va;
        result.alloc_size = candidates[i].size;
        result.offset = candidates[i].offset;
        result.valid = true;
        return result;
      }
    }

    // Нет exact match — берём с ближайшим размером
    size_t best_diff = SIZE_MAX;
    int best_idx = 0;
    for (int i = 0; i < n_candidates; i++) {
      size_t diff = candidates[i].size - expected_size;
      if (diff < best_diff) { best_diff = diff; best_idx = i; }
    }
    cached_offset.store(candidates[best_idx].offset, std::memory_order_release);
    result.gpu_va = candidates[best_idx].va;
    result.alloc_size = candidates[best_idx].size;
    result.offset = candidates[best_idx].offset;
    result.valid = true;
    return result;
  }

  return result;
}

bool ExportGpuVAasDmaBuf(void* gpu_va, size_t size,
                          int* fd, uint64_t* offset) {
  if (!gpu_va || !fd || !offset) return false;
  if (!EnsureHsaInitialized()) return false;

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(gpu_va, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS) {
    return false;
  }

  void* base = info.agentBaseAddress ? info.agentBaseAddress : gpu_va;
  size_t alloc_size = info.sizeInBytes > 0 ? info.sizeInBytes : size;

  hsa_status_t st = hsa_amd_portable_export_dmabuf(base, alloc_size, fd, offset);
  return (st == HSA_STATUS_SUCCESS && *fd >= 0);
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
