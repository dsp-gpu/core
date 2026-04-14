#pragma once

/**
 * @file gpu_benchmark_base.hpp
 * @brief GpuBenchmarkBase — Template Method Pattern для GPU бенчмарков
 *
 * Базовый класс для стандартного профилирования GPU-модулей.
 * Обеспечивает:
 *  - Инициализацию GPUProfiler (SetGPUInfo + Start)
 *  - Прогрев GPU (n_warmup запусков)
 *  - Накопление измерений (n_runs запусков → GPUProfiler)
 *  - Экспорт результатов (PrintReport + JSON + Markdown)
 *
 * Принцип: Production-класс модуля — ЧИСТЫЙ (ноль кода профилирования).
 * Весь код профилирования — только здесь и в наследниках (тест-файлы).
 *
 * Использование:
 * @code
 *   class MyBenchmark : public GpuBenchmarkBase {
 *   protected:
 *     void ExecuteKernel() override {
 *       my_module_.Process(input_, output_);  // warmup — без timing
 *     }
 *     void ExecuteKernelTimed() override {
 *       cl_event ev;
 *       my_module_.Process(input_, output_, &ev);  // measure — с cl_event
 *       RecordEvent("Process", ev);                // → GPUProfiler
 *     }
 *   };
 *
 *   MyBenchmark bench(backend, "MyModule");
 *   bench.Run();     // warmup(5) + measure(20)
 *   bench.Report();  // profiler.PrintReport() + Export
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see specs: MemoryBank/specs/Profil_GPU.md
 */

#include "gpu_profiler.hpp"
#include "profiling_types.hpp"
#include "../interface/i_backend.hpp"
#include <core/config/gpu_config.hpp>
#include "../backends/opencl/opencl_profiling.hpp"

#include <string>
#include <filesystem>

namespace drv_gpu_lib {

struct BenchmarkConfig {
  int         n_warmup   = 5;                  ///< Прогревочных запусков (без замеров)
  int         n_runs     = 20;                 ///< Измерений для статистики (>= 20)
  std::string output_dir = "Results/Profiler"; ///< Каталог для JSON/MD отчётов
};

class GpuBenchmarkBase {
public:
  using Config = BenchmarkConfig;

  // ═══════════════════════════════════════════════════════════════════════
  // Конструктор / Деструктор
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Конструктор
   * @param backend Указатель на IBackend (не владеет)
   * @param module_name Имя модуля для GPUProfiler (например "FFTProcessor")
   * @param cfg Параметры бенчмарка
   *
   * Читает is_prof из configGPU.json. Если is_prof=false — Run() и Report()
   * становятся no-op (нулевой overhead в production).
   */
  GpuBenchmarkBase(IBackend* backend,
                   std::string module_name,
                   Config cfg = {})
    : backend_(backend),
      module_name_(std::move(module_name)),
      cfg_(std::move(cfg))
  {
    if (backend_) {
      gpu_id_ = backend_->GetDeviceIndex();
      // InitializeFromExternalContext устанавливает device_index_=-1 (неизвестен).
      // Предполагаем GPU 0 чтобы корректно читать конфиг и писать в профайлер.
      if (gpu_id_ < 0) gpu_id_ = 0;

      // Конфиг мог не загружаться (тест без DrvGPU). Загружаем если нужно.
      if (!GPUConfig::GetInstance().IsLoaded()) {
        GPUConfig::GetInstance().LoadOrCreate("configGPU.json");
      }

      is_prof_ = GPUConfig::GetInstance().IsProfilingEnabled(gpu_id_);
    }
  }

  virtual ~GpuBenchmarkBase() = default;

  // Запрет копирования
  GpuBenchmarkBase(const GpuBenchmarkBase&) = delete;
  GpuBenchmarkBase& operator=(const GpuBenchmarkBase&) = delete;

  // ═══════════════════════════════════════════════════════════════════════
  // Публичный API
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Запустить бенчмарк: InitProfiler → Warmup → Measure
   *
   * Поток:
   *  1. InitProfiler()  — SetGPUInfo + profiler.Start()
   *  2. Warmup           — n_warmup раз ExecuteKernel() (без timing)
   *  3. profiler.Reset() — очистить данные warmup
   *  4. Measure          — n_runs раз ExecuteKernelTimed() (с timing → GPUProfiler)
   *
   * ExecuteKernelTimed() вызывает RecordEvent() → profiler.Record().
   * GPUProfiler копит все N вызовов → min/max/avg считается автоматически.
   *
   * no-op если is_prof=false в configGPU.json
   */
  void Run() {
    if (!is_prof_ || !backend_) return;

    InitProfiler();

    // Warmup — прогреть GPU (JIT, clock ramp-up, shader cache)
    // ExecuteKernel() — БЕЗ timing, просто запуск
    for (int i = 0; i < cfg_.n_warmup; ++i) {
      ExecuteKernel();
    }

    // Сбросить данные warmup (если что-то попало в profiler)
    GPUProfiler::GetInstance().Reset();

    // Measure — собрать n_runs замеров
    // ExecuteKernelTimed() → RecordEvent() → profiler.Record()
    for (int i = 0; i < cfg_.n_runs; ++i) {
      ExecuteKernelTimed();
    }
  }

  /**
   * @brief Вывести результаты через GPUProfiler
   *
   * Вызывает ТОЛЬКО:
   *  - profiler.PrintReport()
   *  - profiler.ExportJSON(output_dir/...)
   *  - profiler.ExportMarkdown(output_dir/...)
   *  - profiler.Stop()
   *
   * Ничего своего НЕ печатает — весь вывод через GPUProfiler.
   * no-op если is_prof=false
   */
  void Report() {
    if (!is_prof_) return;

    auto& profiler = GPUProfiler::GetInstance();

    // Дождаться обработки всех Record()-сообщений из фоновой очереди.
    // Без этого последнее сообщение может не успеть обработаться → N будет на 1 меньше.
    profiler.WaitEmpty();

    // Печать в консоль — стандартный формат GPUProfiler
    profiler.PrintReport();

    // Создать каталог для экспорта если нужно
    if (!cfg_.output_dir.empty()) {
      std::filesystem::create_directories(cfg_.output_dir);

      // Формируем имя файла с датой
      std::string timestamp = profiler.GetCurrentDateTimeString();
      for (char& c : timestamp) {
        if (c == ' ') c = '_';
        if (c == ':') c = '-';
      }

      std::string base_name = cfg_.output_dir + "/" + module_name_
                             + "_benchmark_" + timestamp;
      profiler.ExportJSON(base_name + ".json");
      profiler.ExportMarkdown(base_name + ".md");
    }

    profiler.Stop();
  }

  /// Проверить, включено ли профилирование для данного GPU
  bool IsProfEnabled() const { return is_prof_; }

protected:
  // ═══════════════════════════════════════════════════════════════════════
  // Виртуальные методы — переопределяет каждый модуль
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Выполнить один запуск кернела БЕЗ timing (для warmup)
   *
   * Просто вызвать основную функцию модуля:
   *   proc_.ProcessComplex(input_, params_);
   *
   * Никаких cl_event, никаких Record — просто прогрев GPU.
   */
  virtual void ExecuteKernel() = 0;

  /**
   * @brief Выполнить один запуск кернела С timing (для замеров)
   *
   * Вызвать основную функцию модуля с cl_event* и записать в GPUProfiler:
   *   cl_event ev;
   *   proc_.ProcessComplex(input_, params_, &ev);
   *   RecordEvent("FFT_Execute", ev);
   *
   * RecordEvent() — helper в базовом классе.
   */
  virtual void ExecuteKernelTimed() = 0;

  // ═══════════════════════════════════════════════════════════════════════
  // Helpers для наследников
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Записать cl_event в GPUProfiler (OpenCL)
   *
   * Ждёт завершения → извлекает timing → profiler.Record()
   * Освобождает cl_event после записи.
   *
   * @param event_name Имя события (напр. "FFT_Execute", "Upload")
   * @param ev cl_event из OpenCL операции
   */
  void RecordEvent(const char* event_name, cl_event ev) {
    if (!ev) return;
    clWaitForEvents(1, &ev);
    OpenCLProfilingData pdata{};
    if (FillOpenCLProfilingData(ev, pdata)) {
      GPUProfiler::GetInstance().Record(gpu_id_, module_name_, event_name, pdata);
    }
    clReleaseEvent(ev);
  }

#ifdef ENABLE_ROCM
  /**
   * @brief Записать ROCm timing в GPUProfiler
   * @param event_name Имя события
   * @param data ROCm profiling данные
   */
  void RecordROCmEvent(const char* event_name, const ROCmProfilingData& data) {
    GPUProfiler::GetInstance().Record(gpu_id_, module_name_, event_name, data);
  }
#endif

  // ═══════════════════════════════════════════════════════════════════════
  // Доступ для наследников
  // ═══════════════════════════════════════════════════════════════════════

  IBackend* backend_ = nullptr;
  int       gpu_id_  = 0;

private:
  // ═══════════════════════════════════════════════════════════════════════
  // Инициализация GPUProfiler
  // ═══════════════════════════════════════════════════════════════════════

  void InitProfiler() {
    auto& profiler = GPUProfiler::GetInstance();

    // Сброс предыдущих данных
    profiler.Reset();
    profiler.SetEnabled(true);
    profiler.SetGPUEnabled(gpu_id_, true);

    // Заполнить информацию о GPU для шапки отчёта
    auto device_info = backend_->GetDeviceInfo();
    GPUReportInfo gpu_info;
    gpu_info.gpu_name = device_info.name;
    gpu_info.global_mem_mb = backend_->GetGlobalMemorySize() / (1024 * 1024);

    // Определить тип backend и заполнить драйвер
    auto backend_type = backend_->GetType();
    gpu_info.backend_type = backend_type;

    std::map<std::string, std::string> driver;
    if (backend_type == BackendType::OPENCL ||
        backend_type == BackendType::OPENCLandROCm) {
      driver["driver_type"]    = "OpenCL";
      driver["version"]        = device_info.opencl_version;
      driver["driver_version"] = device_info.driver_version;
      driver["vendor"]         = device_info.vendor;
      gpu_info.drivers.push_back(driver);
    }
#ifdef ENABLE_ROCM
    if (backend_type == BackendType::ROCm ||
        backend_type == BackendType::OPENCLandROCm) {
      std::map<std::string, std::string> rocm_driver;
      rocm_driver["driver_type"]    = "ROCm";
      rocm_driver["version"]        = device_info.driver_version;
      rocm_driver["driver_version"] = device_info.driver_version;
      gpu_info.drivers.push_back(rocm_driver);
    }
#endif

    // ⚠️ ВАЖНО: SetGPUInfo ПЕРЕД Start()
    profiler.SetGPUInfo(gpu_id_, gpu_info);
    profiler.Start();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Поля
  // ═══════════════════════════════════════════════════════════════════════

  std::string module_name_;
  Config      cfg_;
  bool        is_prof_ = false;    // из configGPU.json
};

}  // namespace drv_gpu_lib
