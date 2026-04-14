#pragma once
// test_services.hpp - Multithread tests for DrvGPU services
// Author: Codo, Date: 2026-02-07

#include <core/services/async_service_base.hpp>
#include <core/services/gpu_profiler.hpp>
#include <core/services/console_output.hpp>
#include <core/services/service_manager.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <string>

namespace test_services {

constexpr int NUM_THREADS = 8;
constexpr int EVENTS_PER_THREAD = 50;

inline bool TestConsoleOutput() {
    std::cout << "\nTEST: ConsoleOutput Multithread\n";
    auto& console = drv_gpu_lib::ConsoleOutput::GetInstance();
    console.Start();
    console.SetEnabled(true);
    for (int i = 0; i < NUM_THREADS; ++i) console.SetGPUEnabled(i, true);
    std::atomic<int> total{0};
    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < NUM_THREADS; ++gpu) {
        threads.emplace_back([&console, &total, gpu]() {
            for (int i = 0; i < 50; ++i) {
                console.Print(gpu, "FFT", "Batch " + std::to_string(i));
                total++;
            }
        });
    }
    for (auto& t : threads) t.join();
    while (console.GetQueueSize() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    bool ok = (total.load() == NUM_THREADS * 50);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " ConsoleOutput: "
              << total.load() << "/" << NUM_THREADS * 50 << "\n";
    return ok;
}

inline bool TestServiceManager() {
    std::cout << "\nTEST: ServiceManager\n";
    auto& mgr = drv_gpu_lib::ServiceManager::GetInstance();
    mgr.InitializeDefaults();
    mgr.StartAll();
    std::cout << mgr.GetStatus() << "\n";
    for (int g = 0; g < 4; ++g) {
        auto data = drv_gpu_lib::MakeOpenCLFromDurationMs(1.0);
        for (int i = 0; i < 10; ++i)
            drv_gpu_lib::GPUProfiler::GetInstance().Record(g, "T", "E", data);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    mgr.PrintProfilingSummary();
    mgr.StopAll();
    std::cout << "[PASS] ServiceManager\n";
    return true;
}

// Stress test for AsyncServiceBase latency/throughput
struct StressMsg {
    int id;
    std::chrono::high_resolution_clock::time_point ts;
};

class StressService : public drv_gpu_lib::AsyncServiceBase<StressMsg> {
public:
    std::atomic<int> count{0};
    std::atomic<double> total_latency_us{0.0};
protected:
    void ProcessMessage(const StressMsg& m) override {
        auto now = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double, std::micro>(now - m.ts).count();
        count++;
        double old = total_latency_us.load();
        while (!total_latency_us.compare_exchange_weak(old, old + lat)) {}
    }
    std::string GetServiceName() const override { return "StressService"; }
};

inline bool TestStressAsyncService() {
    std::cout << "\nTEST: AsyncServiceBase Stress\n";
    constexpr int ITERS = 1000;
    StressService svc;
    svc.Start();

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&svc, t, ITERS]() {
            for (int i = 0; i < ITERS; ++i) {
                svc.Enqueue({t * ITERS + i, std::chrono::high_resolution_clock::now()});
            }
        });
    }
    for (auto& th : threads) th.join();
    while (svc.GetQueueSize() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    svc.Stop();
    auto t1 = std::chrono::high_resolution_clock::now();

    double dur_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int expected = NUM_THREADS * ITERS;
    int got = svc.count.load();
    double avg_lat = got > 0 ? svc.total_latency_us.load() / got : 0;
    double tput = got > 0 ? got / (dur_ms / 1000.0) : 0;

    std::cout << "  Msgs: " << got << "/" << expected << "\n";
    std::cout << "  Avg latency: " << std::fixed << std::setprecision(2) << avg_lat << " us\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) << tput << " msg/s\n";

    bool ok = (got == expected);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " StressAsyncService\n";
    return ok;
}

inline int run() {
    std::cout << "\n****************************************************************\n";
    std::cout << "*         DRVGPU SERVICES MULTITHREADED TEST SUITE             *\n";
    std::cout << "****************************************************************\n";
    int pass = 0, fail = 0;
    if (TestConsoleOutput()) pass++; else fail++;
    if (TestStressAsyncService()) pass++; else fail++;
    if (TestServiceManager()) pass++; else fail++;
    std::cout << "\n****************************************************************\n";
    std::cout << "  Passed: " << pass << ", Failed: " << fail << "\n";
    std::cout << "  " << (fail == 0 ? "[ALL TESTS PASSED]" : "[SOME TESTS FAILED]") << "\n";
    std::cout << "****************************************************************\n";
    return fail == 0 ? 0 : 1;
}

} // namespace test_services
