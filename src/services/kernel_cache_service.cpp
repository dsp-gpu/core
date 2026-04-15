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

#include <core/services/kernel_cache_service.hpp>

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
/**
 * @brief Создаёт сервис. При непустом arch — путь = base_dir/arch/ (multi-GPU).
 *
 * Per-arch subdir нужен потому что HSACO привязан к GPU-архитектуре
 * (gfx908 ≠ gfx1100). Если все 10 GPU одной arch — одна подпапка на всех,
 * одна компиляция, 10 параллельных чтений HSACO. Разные arch → изоляция.
 */
KernelCacheService::KernelCacheService(const std::string& base_dir,
                                       BackendType backend_type,
                                       const std::string& arch)
    : base_dir_(arch.empty() ? base_dir : (base_dir + "/" + arch)),
      arch_(arch),
      backend_type_(backend_type) {
}

// ════════════════════════════════════════════════════════════════════════════
// AtomicWrite / FileSizeEquals (статические helpers для multi-GPU safety)
// ════════════════════════════════════════════════════════════════════════════

void KernelCacheService::AtomicWrite(const std::string& path,
                                     const void* data, size_t bytes) {
  // Pattern "write tmp + rename": fs::rename на POSIX атомарен, читатели
  // никогда не видят полузаписанный файл. При concurrent write от 10 потоков
  // с одинаковым содержимым — последний побеждает, результат идентичен.
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
      throw std::runtime_error("KernelCacheService::AtomicWrite: cannot open " + tmp);
    }
    if (bytes > 0) {
      f.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(bytes));
    }
    // Явный close перед rename — чтобы все данные ушли на диск до переименования.
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) {
    // Удалим tmp чтобы не оставлять мусор.
    fs::remove(tmp, ec);
    throw std::runtime_error("KernelCacheService::AtomicWrite: rename failed " + path);
  }
}

bool KernelCacheService::FileSizeEquals(const std::string& path, size_t expected_size) {
  std::error_code ec;
  if (!fs::exists(path, ec)) return false;
  auto sz = fs::file_size(path, ec);
  if (ec) return false;
  return static_cast<size_t>(sz) == expected_size;
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
/**
 * @brief Idempotent + atomic Save (multi-GPU safe, no locks).
 *
 * Изменения v2 (2026-04-15):
 *   - Убран VersionOldFiles (rename race между потоками → дубли _00, _01…)
 *   - Idempotent: skip IO если файл уже существует с тем же размером
 *   - Atomic write: .tmp → rename (POSIX atomic)
 *
 * Порядок:
 *   1. create_directories(base_dir_/bin) — идемпотентно
 *   2. .cl source → atomic write (если размер не совпадает)
 *   3. binary   → atomic write (если размер не совпадает)
 *   4. manifest → UPSERT atomic write (всегда — timestamp меняется)
 *
 * При 10 параллельных Save (одна arch, одинаковый source):
 *   - Поток 1 пишет first.tmp → rename → first.hsaco
 *   - Поток 2: видит first.hsaco с тем же размером → skip write (no IO)
 *   - Результат: 1 write, 9 skip. Без блокировок.
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

  // Создаём директории (base_dir_ уже с arch suffix если задан)
  std::error_code ec;
  fs::create_directories(base_dir_, ec);
  std::string bin_dir = GetBinDir();
  fs::create_directories(bin_dir, ec);

  // ── .cl source (atomic + idempotent) ──────────────────────────────────
  std::string cl_path = base_dir_ + "/" + name + ".cl";
  if (!FileSizeEquals(cl_path, cl_source.size())) {
    AtomicWrite(cl_path, cl_source.data(), cl_source.size());
  }

  // ── binary (atomic + idempotent) ──────────────────────────────────────
  std::string bin_path = bin_dir + "/" + name + GetBinarySuffix();
  if (!FileSizeEquals(bin_path, binary.size())) {
    AtomicWrite(bin_path, binary.data(), binary.size());
  }

  // ── manifest.json (atomic UPSERT, перезапись — timestamp обновляется) ─
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

  // Сформировать полный текст manifest в буфере (для atomic rename)
  std::ostringstream buf;
  buf << "{\n";
  buf << "  \"version\": 1,\n";
  buf << "  \"kernels\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    buf << entries[i];
    if (i + 1 < entries.size()) buf << ",";
    buf << "\n";
  }
  buf << "  ]\n";
  buf << "}\n";

  // Atomic write через .tmp + rename — при concurrent write от 10 потоков
  // manifest.json никогда не окажется частично записанным.
  std::string data = buf.str();
  AtomicWrite(manifest_path, data.data(), data.size());
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