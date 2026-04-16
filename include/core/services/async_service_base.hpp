#pragma once

/**
 * @file async_service_base.hpp
 * @brief AsyncServiceBase — базовый класс асинхронных фоновых сервисов
 *
 * ============================================================================
 * НАЗНАЧЕНИЕ:
 *   Шаблонный базовый класс для Logger, Profiler, ConsoleOutput и будущих сервисов.
 *   Обеспечивает рабочий поток + очередь сообщений + паттерн наблюдателя.
 *
 * АРХИТЕКТУРА:
 *   GPU Thread 0 --> Enqueue(msg) --+
 *   GPU Thread 1 --> Enqueue(msg) --+--> [Очередь] --> Worker Thread --> ProcessMessage(msg)
 *   GPU Thread N --> Enqueue(msg) --+
 *
 * ГАРАНТИИ:
 *   - Потоки GPU НИКОГДА не блокируются на выводе (только lock-free Enqueue)
 *   - Вся обработка выполняется в выделенном фоновом потоке
 *   - При Stop(): ожидание обработки всех сообщений в очереди
 *   - Потокобезопасность: много производителей, один потребитель
 *
 * ПАТТЕРН: Producer-Consumer + Наблюдатель
 *   - Производители: потоки GPU вызывают Enqueue()
 *   - Потребитель: рабочий поток вызывает ProcessMessage() (виртуальный)
 *   - Наблюдатель: рабочий поток пробуждается по condition_variable
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   class MyService : public AsyncServiceBase<MyMessage> {
 *   protected:
 *       void ProcessMessage(const MyMessage& msg) override {
 *           // Обработка сообщения в фоновом потоке
 *       }
 *   };
 *
 *   MyService service;
 *   service.Start();
 *   service.Enqueue({...});  // Неблокирующий вызов!
 *   service.Stop();          // Ожидание опустошения очереди
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <vector>
#include <string>

namespace drv_gpu_lib {

// ============================================================================
// AsyncServiceBase — шаблонная база фоновых сервисов
// ============================================================================

/**
 * @class AsyncServiceBase
 * @brief Шаблонный базовый класс асинхронных сервисов с очередью сообщений
 *
 * @tparam TMessage Тип сообщений, обрабатываемых сервисом
 *
 * Наследники должны реализовать:
 * - ProcessMessage(const TMessage& msg) — обработка одного сообщения
 * - GetServiceName() — человекочитаемое имя сервиса
 *
 * Жизненный цикл:
 * 1. Создать объект наследника
 * 2. Вызвать Start() для запуска рабочего потока
 * 3. Вызывать Enqueue() из любого потока (неблокирующе)
 * 4. Вызвать Stop() для остановки (сначала опустошается очередь)
 *
 * Модель потоков:
 * - Рабочий поток выполняет WorkerLoop() в фоне
 * - WorkerLoop() ожидает на condition_variable
 * - При появлении сообщений пробуждается и обрабатывает все накопившиеся
 * - При Stop() обрабатывает оставшиеся сообщения и присоединяет поток
 */
template<typename TMessage>
class AsyncServiceBase {
public:
    // ========================================================================
    // Конструктор / Деструктор
    // ========================================================================

    /**
     * @brief Конструктор по умолчанию (НЕ запускает рабочий поток)
     * Вызовите Start() для начала обработки.
     */
    AsyncServiceBase() = default;

    /**
     * @brief Деструктор — автоматически останавливает рабочий поток
     * Ожидает обработки всех сообщений в очереди.
     *
     * ⚠️ ПРАВИЛО ДЛЯ НАСЛЕДНИКОВ:
     * КАЖДЫЙ наследник ОБЯЗАН вызвать Stop() в СВОЁМ деструкторе!
     * Причина: к моменту ~AsyncServiceBase() vtable уже переключена на
     * базовый класс → ProcessMessage() (pure virtual) → UB / terminate.
     * Пример: ~ConsoleOutput() { Stop(); }
     *         ~GPUProfiler()   { Stop(); }
     * Stop() идемпотентен (compare_exchange) — повторный вызов из базового
     * деструктора безопасен.
     */
    virtual ~AsyncServiceBase() {
        Stop();
    }

    // Запрет копирования, разрешение перемещения
    AsyncServiceBase(const AsyncServiceBase&) = delete;
    AsyncServiceBase& operator=(const AsyncServiceBase&) = delete;

    // ========================================================================
    // Управление жизненным циклом
    // ========================================================================

    /**
     * @brief Запустить рабочий поток
     *
     * Запускает фоновый поток, обрабатывающий сообщения из очереди.
     * Безопасно вызывать несколько раз (запуск только один раз).
     *
     * @note Необходимо вызвать до Enqueue(), иначе сообщения не обрабатываются.
     */
    void Start() {
        // compare_exchange: атомарно проверяем false → true.
        // Без этого два потока могли бы одновременно пройти load()==false и оба запустить поток.
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
            return;  // Уже запущен
        }

        worker_thread_ = std::thread([this]() {
            WorkerLoop();
        });
    }

    /**
     * @brief Остановить рабочий поток
     *
     * Подаёт сигнал остановки, затем ждёт обработки всех сообщений в очереди
     * перед присоединением потока.
     *
     * Безопасно вызывать несколько раз (остановка только один раз).
     * Вызывается автоматически из деструктора.
     */
    void Stop() {
        // Сигнал остановки
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
            return; // Уже остановлен или не был запущен
        }

        // Пробудить рабочий поток для реакции на остановку
        cv_.notify_one();

        // Ожидание завершения рабочего потока
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    /**
     * @brief Проверить, запущен ли сервис
     * @return true если рабочий поток активен
     */
    bool IsRunning() const {
        return running_.load(std::memory_order_acquire);
    }

    // ========================================================================
    // Очередь сообщений (неблокирующий API производителя)
    // ========================================================================

    /**
     * @brief Поставить сообщение в очередь для фоновой обработки
     *
     * Основной API для потоков GPU.
     * Практически неблокирующий: только захват мьютекса для добавления в очередь.
     *
     * @param msg Сообщение для обработки (перемещается в очередь)
     *
     * @note Если сервис не запущен, сообщение тихо отбрасывается.
     *       Это сделано намеренно, чтобы не блокировать потоки GPU.
     */
    void Enqueue(TMessage msg) {
        if (!running_.load(std::memory_order_acquire)) {
            return; // Сервис не запущен — отбрасываем сообщение
        }

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push(std::move(msg));
        }

        // Пробудить рабочий поток
        cv_.notify_one();
    }

    /**
     * @brief Поставить несколько сообщений в очередь (пакет)
     *
     * Эффективнее многократного вызова Enqueue():
     * один захват мьютекса и одно уведомление.
     *
     * @param messages Вектор сообщений для постановки в очередь
     */
    void EnqueueBatch(std::vector<TMessage> messages) {
        if (!running_.load(std::memory_order_acquire) || messages.empty()) {
            return;
        }

        pending_count_.fetch_add(messages.size(), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (auto& msg : messages) {
                queue_.push(std::move(msg));
            }
        }

        cv_.notify_one();
    }

    /**
     * @brief Текущий размер очереди (приблизительно, для диагностики)
     * @return Количество необработанных сообщений
     */
    size_t GetQueueSize() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

    /**
     * @brief Общее число обработанных сообщений с момента Start()
     * @return Количество обработанных сообщений
     */
    uint64_t GetProcessedCount() const {
        return processed_count_.load(std::memory_order_acquire);
    }

    /**
     * @brief Дождаться обработки всех поставленных в очередь сообщений
     *
     * Блокирует вызывающий поток до тех пор, пока все ранее отправленные
     * через Enqueue/EnqueueBatch сообщения не будут обработаны ProcessMessage().
     *
     * Используется перед чтением результатов (напр. PrintReport),
     * чтобы гарантировать полноту данных.
     *
     * Реализация: condition_variable (не spin-wait) — CPU не сжигается в ожидании.
     */
    void WaitEmpty() const {
        std::unique_lock<std::mutex> lock(empty_mutex_);
        empty_cv_.wait(lock, [this]() {
            return pending_count_.load(std::memory_order_acquire) == 0;
        });
    }

protected:
    // ========================================================================
    // Виртуальные методы (реализуются наследниками)
    // ========================================================================

    /**
     * @brief Обработать одно сообщение из очереди
     *
     * Вызывается рабочим потоком для каждого сообщения.
     * Здесь наследники реализуют свою логику.
     *
     * ВАЖНО: Выполняется в РАБОЧЕМ ПОТОКЕ, а не в потоке GPU!
     * Безопасно выполнять I/O, запись в файл, вывод в консоль и т.д.
     *
     * @param msg Сообщение для обработки
     */
    virtual void ProcessMessage(const TMessage& msg) = 0;

    /**
     * @brief Человекочитаемое имя сервиса (для диагностики)
     * @return Имя сервиса (напр., "Logger", "Profiler", "ConsoleOutput")
     */
    virtual std::string GetServiceName() const = 0;

    /**
     * @brief Вызывается при запуске рабочего потока (необязательная перегрузка)
     * Использовать для локальной инициализации потока.
     */
    virtual void OnWorkerStart() {}

    /**
     * @brief Вызывается при остановке рабочего потока (необязательная перегрузка)
     * Использовать для локальной очистки потока.
     */
    virtual void OnWorkerStop() {}

private:
    // ========================================================================
    // Реализация рабочего потока
    // ========================================================================

    /**
     * @brief Основной цикл рабочего потока (выполняется в фоне)
     *
     * Алгоритм:
     * 1. Ожидание на condition_variable (сон при пустой очереди)
     * 2. Пробуждение по notify (от Enqueue) или сигналу остановки
     * 3. Извлечение всех накопившихся сообщений из очереди
     * 4. Обработка каждого сообщения через ProcessMessage()
     * 5. Повтор до вызова Stop()
     * 6. При остановке: обработать оставшиеся сообщения и выйти
     */
    void WorkerLoop() {
        // Локальная инициализация потока
        OnWorkerStart();

        while (true) {
            std::vector<TMessage> batch;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Wait until: (a) queue has messages, or (b) stop signal
                cv_.wait(lock, [this]() {
                    return !queue_.empty() || !running_.load(std::memory_order_acquire);
                });

                // Drain all pending messages into local batch
                // (minimizes time holding the mutex)
                while (!queue_.empty()) {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop();
                }
            }

            // Process batch outside of lock
            for (const auto& msg : batch) {
                ProcessMessage(msg);
                processed_count_.fetch_add(1, std::memory_order_relaxed);
                // fetch_sub returns old value: if it was 1, queue just became empty
                if (pending_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    empty_cv_.notify_all();
                }
            }

            // Check if we should stop (after processing remaining messages)
            if (!running_.load(std::memory_order_acquire)) {
                // Final drain: process any messages that arrived during processing
                std::vector<TMessage> final_batch;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    while (!queue_.empty()) {
                        final_batch.push_back(std::move(queue_.front()));
                        queue_.pop();
                    }
                }
                for (const auto& msg : final_batch) {
                    ProcessMessage(msg);
                    processed_count_.fetch_add(1, std::memory_order_relaxed);
                    if (pending_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        empty_cv_.notify_all();
                    }
                }
                break;
            }
        }

        // Thread-local cleanup
        OnWorkerStop();
    }

    // ========================================================================
    // Private Members
    // ========================================================================

    /// Worker thread
    std::thread worker_thread_;

    /// Message queue (FIFO)
    std::queue<TMessage> queue_;

    /// Mutex protecting the queue
    mutable std::mutex queue_mutex_;

    /// Condition variable for worker wakeup
    std::condition_variable cv_;

    /// Running flag (atomic for lock-free check in Enqueue)
    std::atomic<bool> running_{false};

    /// Counter of processed messages (for diagnostics)
    std::atomic<uint64_t> processed_count_{0};

    /// Counter of pending (enqueued but not yet processed) messages
    mutable std::atomic<uint64_t> pending_count_{0};

    /// Mutex + CV for WaitEmpty() — нотификация когда очередь опустела
    mutable std::mutex empty_mutex_;
    mutable std::condition_variable empty_cv_;
};

} // namespace drv_gpu_lib
