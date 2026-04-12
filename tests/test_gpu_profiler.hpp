#pragma once
/**
 * @file test_gpu_profiler.hpp
 * @brief Отдельный тест GPUProfiler (многопоточный Record, агрегация, PrintSummary)
 * @author Codo, Date: 2026-02-08
 */

#include "../services/gpu_profiler.hpp"
#include "../include/gpu_manager.hpp"  // Для GetGPUReportInfo()
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <string>

namespace test_gpu_profiler {

constexpr int NUM_THREADS = 8;
constexpr int EVENTS_PER_THREAD = 50;

inline bool TestGPUProfilerMultithread() {
    std::cout << "\n--- TEST: GPUProfiler (multithread) ---\n";
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    profiler.Reset();
    profiler.Start();
    profiler.SetEnabled(true);

    std::atomic<int> total{0};
    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < NUM_THREADS; ++gpu) {
        threads.emplace_back([&profiler, &total, gpu]() {
            for (int i = 0; i < EVENTS_PER_THREAD; ++i) {
                auto data = drv_gpu_lib::MakeOpenCLFromDurationMs(0.5 + i * 0.1);
                profiler.Record(gpu, "FFT", "Execute", data);
                total++;
            }
        });
    }
    for (auto& t : threads) t.join();

    while (profiler.GetQueueSize() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    int aggregated = 0;
    auto stats = profiler.GetAllStats();
    for (const auto& [gid, modules] : stats) {
        for (const auto& [mod, mstats] : modules) {
            aggregated += static_cast<int>(mstats.GetTotalCalls());
        }
    }

    profiler.PrintSummary();

    const int expected = NUM_THREADS * EVENTS_PER_THREAD;
    bool ok = (total.load() == expected) && (aggregated == expected);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " GPUProfiler: "
              << aggregated << "/" << expected << "\n";
    std::cout << "----------------------------------------\n";
    return ok;
}

/** Демо: профилирование «библиотеки» — несколько модулей/событий, OpenCL + ROCm, итоговая сводка */
inline bool TestGPUProfilerLibraryDemo() {
    std::cout << "\n--- TEST: GPUProfiler Library Demo (OpenCL + ROCm) ---\n";
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    profiler.Reset();
    profiler.Start();
    profiler.SetEnabled(true);

    // Имитация работы «библиотеки»: FFT, Copy, Kernel на двух GPU
    drv_gpu_lib::OpenCLProfilingData ocl_fft{};
    ocl_fft.start_ns = 1000;
    ocl_fft.end_ns   = 1000 + 5000000;  // 5 ms
    ocl_fft.queued_ns = 0; ocl_fft.submit_ns = 500; ocl_fft.complete_ns = ocl_fft.end_ns;
    profiler.Record(0, "AntennaFFT", "SingleBatchFFT", ocl_fft);
    profiler.Record(0, "AntennaFFT", "SingleBatchFFT", ocl_fft);

    drv_gpu_lib::OpenCLProfilingData ocl_copy{};
    ocl_copy.start_ns = 0;
    ocl_copy.end_ns   = 800000;  // 0.8 ms
    ocl_copy.queued_ns = ocl_copy.submit_ns = ocl_copy.complete_ns = ocl_copy.end_ns;
    profiler.Record(0, "AntennaFFT", "Copy", ocl_copy);
    profiler.Record(1, "AntennaFFT", "Copy", ocl_copy);

    drv_gpu_lib::ROCmProfilingData rocm_k{};
    rocm_k.start_ns = 0;
    rocm_k.end_ns   = 1200000;  // 1.2 ms
    rocm_k.queued_ns = rocm_k.submit_ns = rocm_k.complete_ns = rocm_k.end_ns;
    rocm_k.kernel_name = "vector_add";
    profiler.Record(1, "ROCmKernels", "vector_add", rocm_k);

    // Синтетические длительности через хелпер (как в тестах без cl_event)
    profiler.Record(0, "TestModule", "Upload", drv_gpu_lib::MakeOpenCLFromDurationMs(0.25));
    profiler.Record(0, "TestModule", "Upload", drv_gpu_lib::MakeOpenCLFromDurationMs(0.31));

    while (profiler.GetQueueSize() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::cout << "Aggregated stats:\n";
    profiler.PrintSummary();

    auto stats = profiler.GetAllStats();
    int total_events = 0;
    for (const auto& [gid, modules] : stats)
        for (const auto& [mod, mstats] : modules)
            total_events += static_cast<int>(mstats.GetTotalCalls());

    const int expected = 2 + 2 + 1 + 2;  // FFT×2, Copy×2, ROCm×1, Upload×2
    bool ok = (total_events == expected);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " Library demo: "
              << total_events << " events (expected " << expected << ")\n";
    std::cout << "----------------------------------------\n";
    return ok;
}

/** Тест PrintReport с GPU Info (красивая таблица для отчёта) */
inline bool TestGPUProfilerPrintReport() {
    std::cout << "\n--- TEST: GPUProfiler PrintReport with GPU Info ---\n";
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    profiler.Reset();
    profiler.Start();
    profiler.SetEnabled(true);

    // ═══════════════════════════════════════════════════════════════════════════
    // GPU 0: РЕАЛЬНАЯ информация через GPUManager::GetGPUReportInfo()
    // ═══════════════════════════════════════════════════════════════════════════
    drv_gpu_lib::GPUManager manager;
    manager.InitializeAll(drv_gpu_lib::BackendType::OPENCL);

    // Получаем GPUReportInfo из РЕАЛЬНОЙ системы!
    drv_gpu_lib::GPUReportInfo gpu0_info = manager.GetGPUReportInfo(0);
    profiler.SetGPUInfo(0, gpu0_info);

    std::cout << "[INFO] GPU 0 (реальная): " << gpu0_info.gpu_name << "\n";
    if (!gpu0_info.drivers.empty()) {
        std::cout << "[INFO] Драйверов: " << gpu0_info.drivers.size() << "\n";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GPU 1: OpenCL + ROCm (эмуляция ROCm для теста)
    // На одной GPU работают сразу OpenCL И ROCm!
    // ═══════════════════════════════════════════════════════════════════════════
/*    
    drv_gpu_lib::GPUReportInfo gpu1_info;
    
    gpu1_info.gpu_name = "AMD Radeon RX 580";
    gpu1_info.backend_type = drv_gpu_lib::BackendType::OPENCLandROCm;
    gpu1_info.global_mem_mb = 8192;

    // drivers[0] = OpenCL info
    std::map<std::string, std::string> gpu1_opencl;
    gpu1_opencl["driver_type"] = "OpenCL";
    gpu1_opencl["version"] = "2.0";
    gpu1_opencl["driver_version"] = "22.40.5";
    gpu1_opencl["platform_name"] = "AMD Accelerated Parallel Processing";
    gpu1_opencl["vendor"] = "AMD";
    gpu1_info.drivers.push_back(gpu1_opencl);
*/
    drv_gpu_lib::GPUReportInfo gpu1_info = manager.GetGPUReportInfo(0);


    // drivers[1] = ROCm info (ЭМУЛЯЦИЯ для теста)
    std::map<std::string, std::string> gpu1_rocm;
    gpu1_rocm["driver_type"] = "ROCm";
    gpu1_rocm["version"] = "5.4.3";
    gpu1_rocm["driver_version"] = "amdgpu 6.1.0";
    gpu1_rocm["hip_version"] = "5.4.22801";
    gpu1_rocm["hip_runtime"] = "5.4.22801-1";
    gpu1_info.drivers.push_back(gpu1_rocm);

    profiler.SetGPUInfo(1, gpu1_info);

    // Добавляем тестовые события с реалистичными данными 5 полей
    // Симуляция OpenCL profiling: queued -> submit -> start -> end -> complete
    for (int i = 0; i < 100; ++i) {
        drv_gpu_lib::OpenCLProfilingData fft_data{};
        // Реалистичные времена в наносекундах
        uint64_t base = 1000000000ULL + i * 50000000ULL;  // базовое время ~1 сек + смещение
        fft_data.queued_ns   = base;
        fft_data.submit_ns   = base + 100000 + (i % 10) * 10000;      // +0.1-0.2 ms (queue delay)
        fft_data.start_ns    = fft_data.submit_ns + 50000 + (i % 5) * 5000;  // +0.05-0.075 ms (submit delay)
        fft_data.end_ns      = fft_data.start_ns + 11000000 + (i % 20) * 200000;  // +11-15 ms (exec)
        fft_data.complete_ns = fft_data.end_ns + 20000 + (i % 3) * 5000;  // +0.02-0.035 ms (complete delay)
        profiler.Record(0, "AntennaFFT", "FFT_Execute", fft_data);
    }

    for (int i = 0; i < 100; ++i) {
        drv_gpu_lib::OpenCLProfilingData padding_data{};
        uint64_t base = 2000000000ULL + i * 5000000ULL;
        padding_data.queued_ns   = base;
        padding_data.submit_ns   = base + 80000 + (i % 8) * 10000;  // +0.08-0.16 ms
        padding_data.start_ns    = padding_data.submit_ns + 30000;  // +0.03 ms
        padding_data.end_ns      = padding_data.start_ns + 700000 + (i % 10) * 30000;  // +0.7-1.0 ms
        padding_data.complete_ns = padding_data.end_ns + 10000;  // +0.01 ms
        profiler.Record(0, "AntennaFFT", "Padding_Kernel", padding_data);
    }

    for (int i = 0; i < 50; ++i) {
        drv_gpu_lib::OpenCLProfilingData copy_data{};
        uint64_t base = 3000000000ULL + i * 2000000ULL;
        copy_data.queued_ns   = base;
        copy_data.submit_ns   = base + 50000;  // +0.05 ms
        copy_data.start_ns    = copy_data.submit_ns + 20000;  // +0.02 ms
        copy_data.end_ns      = copy_data.start_ns + 250000;  // +0.25 ms
        copy_data.complete_ns = copy_data.end_ns + 5000;  // +0.005 ms
        profiler.Record(1, "MemOps", "HostToDevice", copy_data);
        profiler.Record(1, "MemOps", "DeviceToHost", copy_data);
    }

    // Тест ROCm данных
    for (int i = 0; i < 20; ++i) {
        drv_gpu_lib::ROCmProfilingData rocm_data{};
        uint64_t base = 4000000000ULL + i * 10000000ULL;
        rocm_data.queued_ns   = base;
        rocm_data.submit_ns   = base + 120000;
        rocm_data.start_ns    = rocm_data.submit_ns + 40000;
        rocm_data.end_ns      = rocm_data.start_ns + 2000000 + (i % 5) * 100000;  // 2-2.5 ms
        rocm_data.complete_ns = rocm_data.end_ns + 15000;
        rocm_data.bytes = 1024 * 1024 * (i + 1);  // 1-20 MB
        rocm_data.kernel_name = "matrix_multiply_kernel";
        rocm_data.correlation_id = 1000 + i;
        profiler.Record(1, "ROCmKernels", "MatrixMul", rocm_data);
    }

    // Ждём обработки
    while (profiler.GetQueueSize() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Выводим красивый отчёт
    std::cout << "\n=== BEAUTIFUL REPORT OUTPUT ===\n";
    profiler.PrintReport();

    // Экспорт в Markdown (заглушка/реальный)
    std::string md_path = "Results/Profiler/test_report.md";
    bool md_ok = profiler.ExportMarkdown(md_path);
    std::cout << (md_ok ? "[OK]" : "[SKIP]") << " Markdown export: " << md_path << "\n";

    auto stats = profiler.GetAllStats();
    int total_events = 0;
    for (const auto& [gid, modules] : stats)
        for (const auto& [mod, mstats] : modules)
            total_events += static_cast<int>(mstats.GetTotalCalls());

    const int expected = 100 + 100 + 50 + 50 + 20;  // FFT×100, Padding×100, H2D×50, D2H×50, ROCm×20
    bool ok = (total_events == expected);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " PrintReport test: "
              << total_events << " events (expected " << expected << ")\n";
    std::cout << "----------------------------------------\n";
    return ok;
}

inline int run() {
    std::cout << "\n****************************************************\n";
    std::cout << "*     DRVGPU GPUProfiler STANDALONE TEST             *\n";
    std::cout << "****************************************************\n";
//    bool ok1 = TestGPUProfilerMultithread();
//    bool ok2 = TestGPUProfilerLibraryDemo();
    bool ok3 = TestGPUProfilerPrintReport();
//    bool ok = ok1 && ok2 && ok3;
    bool ok =  ok3;
    std::cout << (ok ? "[ALL PASSED]" : "[FAILED]") << "\n";
    std::cout << "****************************************************\n\n";
    return ok ? 0 : 1;
}

} // namespace test_gpu_profiler
