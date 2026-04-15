#pragma once

/**
 * @file cache_dir_resolver.hpp
 * @brief Резолвер пути для дискового кеша kernels (portable, exe-relative).
 *
 * Цель: все модули DSP-GPU получают cache_dir по единому правилу без
 * захардкоженных путей. Работает на Linux (main target).
 *
 * Chain fallback (первое непустое):
 *   1. $DSP_CACHE_DIR/<module>/            — env override (CI, Docker, tests)
 *   2. <exe_dir>/kernels_cache/<module>/   — production default
 *   3. $HOME/.cache/dsp-gpu/<module>/      — sandbox/read-only exe
 *   4. ""                                   — disable disk cache
 *
 * Per-arch subdir (gfx908 / gfx1100 / …) добавляет KernelCacheService внутри —
 * здесь возвращается путь на уровень выше arch.
 *
 * Usage:
 *   #include <core/services/cache_dir_resolver.hpp>
 *   GpuContext ctx(backend, "Capon", drv_gpu_lib::ResolveCacheDir("capon"));
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-15
 */

#include <string>

namespace drv_gpu_lib {

/**
 * @brief Возвращает портируемый путь к кешу kernels для заданного модуля.
 * @param module_name Имя модуля (lowercase snake_case): "capon", "vector_algebra", "fft", …
 * @return Абсолютный путь к директории кеша (в рантайме — относительно exe).
 *         Пустая строка → disk cache отключён (не удалось создать ни один путь).
 *
 * Побочный эффект: создаёт директорию через fs::create_directories если
 * её нет. При ошибке создания (read-only, permission denied) переходит к
 * следующему звену цепочки.
 *
 * Thread-safety: safe для вызова из одного потока на один модуль. При
 * параллельном вызове из нескольких потоков с ОДНИМ module_name — безопасно
 * (create_directories idempotent на POSIX).
 */
std::string ResolveCacheDir(const char* module_name);

}  // namespace drv_gpu_lib
