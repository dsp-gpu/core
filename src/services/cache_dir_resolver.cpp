/**
 * @file cache_dir_resolver.cpp
 * @brief Резолвер cache_dir (Linux): ENV → exe-relative → $HOME → disable.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-15
 */

#include <core/services/cache_dir_resolver.hpp>
#include <core/services/console_output.hpp>

#include <filesystem>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
  #include <unistd.h>
  #include <linux/limits.h>
#endif

namespace fs = std::filesystem;

namespace drv_gpu_lib {

namespace {

/**
 * @brief Пытается создать директорию. true — если существует или создана.
 * create_directories возвращает false если директория уже есть — это НЕ ошибка,
 * поэтому дополнительно проверяем fs::exists.
 */
bool EnsureDir(const fs::path& p) {
  std::error_code ec;
  fs::create_directories(p, ec);
  if (ec) {
    // Не удалось создать (read-only FS, permission, etc)
    return false;
  }
  return fs::exists(p, ec) && fs::is_directory(p, ec);
}

/**
 * @brief Читает путь к исполняемому файлу через /proc/self/exe (Linux only).
 * @return Абсолютный путь или пустая строка при ошибке.
 */
std::string GetExePath() {
#ifdef __linux__
  char buf[PATH_MAX] = {0};
  ssize_t len = readlink("/proc/self/exe", buf, PATH_MAX - 1);
  if (len > 0) {
    buf[len] = '\0';
    return std::string(buf);
  }
#endif
  return {};
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// ResolveCacheDir
// ════════════════════════════════════════════════════════════════════════════

std::string ResolveCacheDir(const char* module_name) {
  if (!module_name || !*module_name) {
    return {};
  }

  const std::string module(module_name);
  auto& con = ConsoleOutput::GetInstance();

  // ── 1. ENV override: $DSP_CACHE_DIR ──────────────────────────────────────
  //
  // Используется в CI, Docker, custom deployments. Если переменная задана —
  // путь будет <env>/<module>/. Родительская директория $DSP_CACHE_DIR должна
  // существовать (мы создадим только подкаталог модуля).
  if (const char* env = std::getenv("DSP_CACHE_DIR"); env && *env) {
    fs::path cache = fs::path(env) / module;
    if (EnsureDir(cache)) {
      con.Print(0, module.c_str(),
                "disk cache (ENV): " + cache.string());
      return cache.string();
    }
    // ENV задан, но создать не удалось — НЕ переходим к fallback, чтобы
    // ошибка конфигурации была видна в логах, а не замаскирована.
    con.Print(0, module.c_str(),
              "[!] DSP_CACHE_DIR set but directory creation failed: " + cache.string());
    // Всё-таки переходим к fallback, чтобы не уронить работу.
  }

  // ── 2. Возле исполняемого файла (production default) ─────────────────────
  //
  // Стандартный сценарий: exe в <prefix>/bin/app → кеш в <prefix>/bin/kernels_cache/<module>/.
  // При переносе на сервер вся папка bin/ копируется вместе с кешем.
  std::string exe = GetExePath();
  if (!exe.empty()) {
    fs::path cache = fs::path(exe).parent_path() / "kernels_cache" / module;
    if (EnsureDir(cache)) {
      con.Print(0, module.c_str(),
                "disk cache (exe-rel): " + cache.string());
      return cache.string();
    }
    // Папка exe read-only (например /opt/... или /usr/local/...) — fallback.
  }

  // ── 3. $HOME/.cache/dsp-gpu/<module>/ (user-level fallback) ──────────────
  //
  // Используется если exe лежит в read-only месте (например после apt install).
  // Работает на всех пользовательских установках — $HOME всегда writable.
  if (const char* home = std::getenv("HOME"); home && *home) {
    fs::path cache = fs::path(home) / ".cache" / "dsp-gpu" / module;
    if (EnsureDir(cache)) {
      con.Print(0, module.c_str(),
                "disk cache (home): " + cache.string());
      return cache.string();
    }
  }

  // ── 4. Disable disk cache (возвращаем пустую строку) ─────────────────────
  //
  // Все варианты провалились — KernelCacheService не создастся в GpuContext,
  // компиляция через hiprtc без кеша (медленный cold start, но всё работает).
  con.Print(0, module.c_str(),
            "[!] disk cache disabled (no writable location)");
  return {};
}

}  // namespace drv_gpu_lib
