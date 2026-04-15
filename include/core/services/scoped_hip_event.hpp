#pragma once

/**
 * @file scoped_hip_event.hpp
 * @brief ScopedHipEvent — RAII-обёртка над hipEvent_t (exception-safe)
 *
 * Зачем:
 *   В hot-path GPU-модулей (FFT, filters, lch_farrow, heterodyne, stats,
 *   strategies и др.) событиями профилируются 3-6 стадий подряд.
 *   Если между hipEventCreate и последним hipEventDestroy что-то
 *   бросает исключение (hipfftExecC2C, kernel launch, runtime_error),
 *   ранее созданные события утекают.
 *
 *   ScopedHipEvent гарантирует hipEventDestroy в деструкторе через RAII.
 *
 * Использование:
 *   drv_gpu_lib::ScopedHipEvent ev_up_s, ev_up_e;
 *   ev_up_s.Create(); ev_up_e.Create();
 *   hipEventRecord(ev_up_s.get(), stream);
 *   UploadData(...);
 *   hipEventRecord(ev_up_e.get(), stream);
 *   // При исключении — события корректно освобождаются
 *
 * Move-only: копирование запрещено, перемещение передаёт владение.
 *
 * История:
 *   - 2026-04-14: создан в spectrum (namespace fft_processor) сестрёнкой
 *   - 2026-04-15: перенесён в core/services (namespace drv_gpu_lib),
 *     доступен всем репо DSP-GPU — generic утилита для GPU-профилирования
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-14 (v1 в spectrum), 2026-04-15 (v2 перенос в core)
 */

#if ENABLE_ROCM

#include <hip/hip_runtime.h>

namespace drv_gpu_lib {

class ScopedHipEvent {
public:
  ScopedHipEvent() = default;

  ~ScopedHipEvent() {
    if (event_) {
      hipEventDestroy(event_);
    }
  }

  ScopedHipEvent(const ScopedHipEvent&) = delete;
  ScopedHipEvent& operator=(const ScopedHipEvent&) = delete;

  ScopedHipEvent(ScopedHipEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  ScopedHipEvent& operator=(ScopedHipEvent&& other) noexcept {
    if (this != &other) {
      if (event_) hipEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  /// Создать hipEvent_t. Повторный вызов — сначала уничтожит старое.
  /// @return hipError_t из hipEventCreate
  hipError_t Create() {
    if (event_) {
      hipEventDestroy(event_);
      event_ = nullptr;
    }
    return hipEventCreate(&event_);
  }

  /// Создать hipEvent_t с флагами (hipEventDisableTiming, hipEventBlockingSync, ...).
  /// Используется для sync-событий между streams через hipStreamWaitEvent
  /// (не для тайминга — flags=hipEventDisableTiming снижает overhead).
  /// @param flags флаги hipEventCreateWithFlags
  /// @return hipError_t из hipEventCreateWithFlags
  hipError_t CreateWithFlags(unsigned int flags) {
    if (event_) {
      hipEventDestroy(event_);
      event_ = nullptr;
    }
    return hipEventCreateWithFlags(&event_, flags);
  }

  hipEvent_t get() const { return event_; }
  bool valid() const { return event_ != nullptr; }

private:
  hipEvent_t event_ = nullptr;
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
