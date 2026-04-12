/**
 * @file kernel_cache_service.cpp
 * @brief On-disk kernel cache implementation
 *
 * Extracted from FormScriptGenerator (signal_generators).
 * Generic, storage-agnostic — works with filesystem directly.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "kernel_cache_service.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <cstdio>

namespace fs = std::filesystem;

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создаёт сервис кеширования кернелов для указанного каталога и бэкенда
 *
 * @param base_dir     Корневая директория кеша (напр. "modules/signal_generators/kernels").
 *                     Подкаталог bin/ создаётся автоматически при первом Save().
 * @param backend_type OPENCL → суффикс "_opencl.bin"; ROCm → "_rocm.hsaco"
 */
KernelCacheService::KernelCacheService(const std::string& base_dir,
                                       BackendType backend_type)
    : base_dir_(base_dir), backend_type_(backend_type) {
}

// ════════════════════════════════════════════════════════════════════════════
// Save
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Сохраняет кернел на диск: .cl источник + бинарь + запись в manifest.json
 *
 * Порядок операций важен:
 * 1. GetBinDir() + create_directories — гарантируем существование bin/
 * 2. VersionOldFiles(name) — переименуем старые файлы (_00, _01...) ДО записи новых
 * 3. Сохраняем {name}.cl в base_dir_/
 * 4. Сохраняем {name}{suffix} в base_dir_/bin/
 * 5. WriteManifestEntry() — UPSERT в manifest.json
 *
 * @param name      Имя кернела (без расширения), напр. "my_signal"
 * @param cl_source OpenCL/HIP исходный код
 * @param binary    Скомпилированный бинарь (clGetProgramInfo → CL_PROGRAM_BINARIES)
 * @param metadata  Строка-метаданные (параметры компиляции, версия)
 * @param comment   Человекочитаемый комментарий для manifest.json
 */
void KernelCacheService::Save(const std::string& name,
                               const std::string& cl_source,
                               const std::vector<uint8_t>& binary,
                               const std::string& metadata,
                               const std::string& comment) {
  if (name.empty()) {
    throw std::runtime_error(
        "KernelCacheService::Save: name cannot be empty");
  }

  std::string bin_dir = GetBinDir();
  fs::create_directories(bin_dir);

  // Version old files if they exist
  VersionOldFiles(name);

  // Save .cl source
  std::string cl_path = base_dir_ + "/" + name + ".cl";
  {
    std::ofstream f(cl_path);
    if (!f.is_open()) {
      throw std::runtime_error(
          "KernelCacheService::Save: cannot write " + cl_path);
    }
    f << cl_source;
  }

  // Save binary
  std::string bin_path = bin_dir + "/" + name + GetBinarySuffix();
  {
    std::ofstream f(bin_path, std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error(
          "KernelCacheService::Save: cannot write " + bin_path);
    }
    f.write(reinterpret_cast<const char*>(binary.data()),
            static_cast<std::streamsize>(binary.size()));
  }

  // Update manifest
  WriteManifestEntry(name, metadata, comment);
}

// ════════════════════════════════════════════════════════════════════════════
// Load
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Загружает кернел с диска: бинарь (fast path) или исходник (fallback)
 *
 * Приоритет бинаря: clCreateProgramWithBinary пропускает JIT-компиляцию →
 * быстрее старта, стабильнее результат. Если бинарь есть — возвращаем {source+binary}.
 * Если только исходник — {source, {}}: caller компилирует и может сохранить через Save().
 * Если ни source ни binary нет — возвращаем nullopt (cache miss, без исключения).
 *
 * @param name Имя кернела (без расширения)
 * @return CacheEntry с source и/или binary, или nullopt при cache miss
 */
std::optional<KernelCacheService::CacheEntry>
KernelCacheService::Load(const std::string& name) const {
  CacheEntry entry;

  std::string cl_path = base_dir_ + "/" + name + ".cl";
  std::string bin_path = GetBinDir() + "/" + name + GetBinarySuffix();

  // Бинарный путь приоритетнее: clCreateProgramWithBinary не требует JIT-компиляции.
  // Если бинарь есть → возвращаем {source + binary}, caller сам выбирает что использовать.
  // Если бинаря нет → только source; caller компилирует и может сохранить бинарь через Save().
  if (fs::exists(bin_path)) {
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (f.is_open()) {
      auto size = f.tellg();
      f.seekg(0, std::ios::beg);
      entry.binary.resize(static_cast<size_t>(size));
      f.read(reinterpret_cast<char*>(entry.binary.data()),
             static_cast<std::streamsize>(size));
    }
  }

  // Try source
  if (fs::exists(cl_path)) {
    std::ifstream f(cl_path);
    if (f.is_open()) {
      std::ostringstream ss;
      ss << f.rdbuf();
      entry.source = ss.str();
    }
  }

  // Neither found → cache miss
  if (!entry.has_binary() && !entry.has_source()) {
    return std::nullopt;
  }

  return entry;
}

// ════════════════════════════════════════════════════════════════════════════
// ListKernels
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Возвращает имена всех кернелов из manifest.json
 *
 * Читает manifest.json и извлекает все поля "name" ручным парсингом.
 * Не зависит от внешних JSON-библиотек — достаточно для нашего простого формата.
 * Если manifest.json отсутствует → возвращает пустой вектор (не бросает).
 *
 * @return Вектор имён кернелов в порядке записи в manifest.json
 */
std::vector<std::string> KernelCacheService::ListKernels() const {
  std::vector<std::string> names;
  std::string manifest_path = base_dir_ + "/manifest.json";

  if (!fs::exists(manifest_path)) return names;

  std::ifstream f(manifest_path);
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());

  // Ручной парсинг JSON: не зависим от внешних библиотек (nlohmann/rapidjson).
  // Ищем ВСЕ вхождения "name": "..." в документе — достаточно для нашего манифеста.
  // Ограничение: значение "name" не должно содержать экранированные кавычки (\").
  // Simple JSON parsing: find all "name": "value" pairs
  std::string search = "\"name\"";
  size_t pos = 0;
  while ((pos = content.find(search, pos)) != std::string::npos) {
    pos += search.size();
    // Skip whitespace and colon
    while (pos < content.size() &&
           (content[pos] == ' ' || content[pos] == ':' || content[pos] == '"'))
      ++pos;
    size_t end = content.find('"', pos);
    if (end != std::string::npos) {
      names.push_back(content.substr(pos, end - pos));
      pos = end + 1;
    }
  }

  return names;
}

// ════════════════════════════════════════════════════════════════════════════
// GetBinDir
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Возвращает путь к подкаталогу бинарей (base_dir_/bin)
 * Каталог создаётся в Save(), здесь только формируется строка пути.
 */
std::string KernelCacheService::GetBinDir() const {
  return base_dir_ + "/bin";
}

// ════════════════════════════════════════════════════════════════════════════
// GetBinarySuffix
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Возвращает суффикс имени бинарного файла по типу бэкенда
 *
 * Разные суффиксы гарантируют отсутствие коллизий при смешанном кеше:
 * - OpenCL: "_opencl.bin"  → платформо-зависимый бинарь (SPIR / native ISA)
 * - ROCm:   "_rocm.hsaco"  → AMD GPU shader compiled object (RDNA, GCN)
 * default → OpenCL для безопасности.
 */
std::string KernelCacheService::GetBinarySuffix() const {
  switch (backend_type_) {
    case BackendType::ROCm:
      return "_rocm.hsaco";
    case BackendType::OPENCL:
    default:
      return "_opencl.bin";
  }
}

// ════════════════════════════════════════════════════════════════════════════
// VersionOldFiles
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Переименовывает существующие файлы кернела в версионированные (_00, _01, ...)
 *
 * Вызывается ДО записи новых файлов в Save() — чтобы старая версия не потерялась.
 * fs::rename() атомарно (POSIX) — нет window в котором файл исчезает.
 *
 * Суффикс версии вставляется ПЕРЕД расширением: "_opencl.bin" → "_opencl_00.bin".
 * Перебирает _00.._99; если все заняты — функция молча возвращается (не бросает).
 *
 * @param name Имя кернела (без расширения)
 */
void KernelCacheService::VersionOldFiles(const std::string& name) const {
  std::string cl_path = base_dir_ + "/" + name + ".cl";
  std::string bin_dir = GetBinDir();
  std::string bin_path = bin_dir + "/" + name + GetBinarySuffix();

  bool cl_exists = fs::exists(cl_path);
  bool bin_exists = fs::exists(bin_path);

  if (!cl_exists && !bin_exists) return;

  // Find next free suffix: _00, _01, ...
  int suffix = 0;
  while (suffix <= 99) {
    char buf[8];
    snprintf(buf, sizeof(buf), "_%02d", suffix);
    std::string s(buf);

    std::string old_cl = base_dir_ + "/" + name + s + ".cl";
    // Binary suffix: e.g. name_opencl_00.bin
    std::string suffix_str = GetBinarySuffix();
    // Суффикс версии вставляется ПЕРЕД расширением файла:
    // "_opencl.bin" → rfind('.') = 7 → "_opencl" + "_00" + ".bin" = "_opencl_00.bin"
    auto dot_pos = suffix_str.rfind('.');
    std::string versioned_suffix = suffix_str.substr(0, dot_pos)
                                   + s + suffix_str.substr(dot_pos);
    std::string old_bin = bin_dir + "/" + name + versioned_suffix;

    if (!fs::exists(old_cl) && !fs::exists(old_bin)) {
      if (cl_exists)  fs::rename(cl_path, old_cl);
      if (bin_exists) fs::rename(bin_path, old_bin);
      return;
    }
    ++suffix;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// WriteManifestEntry
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief UPSERT-запись в manifest.json: обновляет или добавляет запись по имени кернела
 *
 * UPSERT логика: читаем существующий manifest → фильтруем запись с тем же name →
 * добавляем новую в конец → перезаписываем весь файл.
 * Позволяет обновить comment/metadata без дублирования строк.
 *
 * Бинарный режим записи (ios::binary) — LF-only переносы строк на Windows,
 * что обеспечивает совместимость с git (no CRLF).
 *
 * @param name     Имя кернела (ключ UPSERT)
 * @param metadata Строка метаданных (параметры, версия)
 * @param comment  Человекочитаемый комментарий
 */
void KernelCacheService::WriteManifestEntry(
    const std::string& name,
    const std::string& metadata,
    const std::string& comment) const {

  std::string manifest_path = base_dir_ + "/manifest.json";
  std::string timestamp = GetTimestamp();

  // Read existing manifest or start fresh
  std::string content;
  if (fs::exists(manifest_path)) {
    std::ifstream f(manifest_path);
    content.assign((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
  }

  // Build new entry JSON
  std::ostringstream entry;
  entry << "    {\n";
  entry << "      \"name\": \"" << name << "\",\n";
  entry << "      \"comment\": \"" << comment << "\",\n";
  entry << "      \"created\": \"" << timestamp << "\",\n";
  entry << "      \"params\": \"" << metadata << "\",\n";

  // Backend string
  const char* backend_str = "opencl";
  if (backend_type_ == BackendType::ROCm) backend_str = "rocm";
  entry << "      \"backend\": \"" << backend_str << "\"\n";
  entry << "    }";

  // UPSERT: читаем старый манифест, отфильтровываем запись с тем же именем,
  // добавляем новую. Это позволяет обновлять comment/params без дублирования.
  // Новая запись всегда добавляется в конец массива.
  std::vector<std::string> entries;
  if (!content.empty()) {
    size_t arr_start = content.find('[');
    size_t arr_end = content.rfind(']');
    if (arr_start != std::string::npos && arr_end != std::string::npos) {
      std::string arr = content.substr(arr_start + 1, arr_end - arr_start - 1);

      size_t pos = 0;
      while (true) {
        size_t obj_start = arr.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end = arr.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj = arr.substr(obj_start, obj_end - obj_start + 1);

        // Check if this is the same name
        bool same_name = false;
        std::string name_check = "\"name\": \"" + name + "\"";
        if (obj.find(name_check) != std::string::npos) same_name = true;
        name_check = "\"name\":\"" + name + "\"";
        if (obj.find(name_check) != std::string::npos) same_name = true;

        if (!same_name) {
          entries.push_back("    " + obj);
        }

        pos = obj_end + 1;
      }
    }
  }

  // Add new entry
  entries.push_back(entry.str());

  // Write manifest (binary mode — LF only)
  std::ofstream f(manifest_path, std::ios::binary);
  f << "{\n";
  f << "  \"version\": 1,\n";
  f << "  \"kernels\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    f << entries[i];
    if (i + 1 < entries.size()) f << ",";
    f << "\n";
  }
  f << "  ]\n";
  f << "}\n";
}

// ════════════════════════════════════════════════════════════════════════════
// GetTimestamp
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Возвращает текущее время в ISO 8601: "2026-02-22T14:35:00"
 *
 * Кросс-платформенный: localtime_s (MSVC/Win32) / localtime_r (POSIX).
 * Используется для поля "created" в manifest.json.
 */
std::string KernelCacheService::GetTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
#ifdef _WIN32
  localtime_s(&tm_buf, &t);
#else
  localtime_r(&t, &tm_buf);
#endif

  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
  return std::string(buf);
}

} // namespace drv_gpu_lib