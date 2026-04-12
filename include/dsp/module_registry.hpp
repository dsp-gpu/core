#pragma once

/**
 * @file module_registry.hpp
 * @brief Регистр compute модулей для DrvGPU
 * 
 * ModuleRegistry управляет compute модулями (FFT, Matrix, etc.)
 * и предоставляет централизованный доступ к ним.
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "../interface/i_compute_module.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <stdexcept>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: ModuleRegistry - Регистр compute модулей
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ModuleRegistry
 * @brief Централизованный реестр compute модулей
 * 
 * ModuleRegistry хранит экземпляры compute модулей и предоставляет
 * доступ к ним по имени. Каждый модуль реализует IComputeModule интерфейс.
 * 
 * Примеры модулей:
 * - FFTModule (быстрое преобразование Фурье)
 * - MatrixModule (операции с матрицами)
 * - ConvolutionModule (свёртка)
 * - SortModule (сортировка на GPU)
 * 
 * Использование:
 * @code
 * ModuleRegistry& registry = gpu.GetModuleRegistry();
 * 
 * // Зарегистрировать модуль
 * auto fft_module = std::make_shared<FFTModule>(backend);
 * registry.RegisterModule("FFT", fft_module);
 * 
 * // Получить модуль
 * auto fft = registry.GetModule("FFT");
 * fft->Initialize();
 * fft->Execute(params);
 * @endcode
 * 
 * Паттерн: Registry (хранилище объектов по ключу)
 */
class ModuleRegistry {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать ModuleRegistry для указанного GPU (логи в DRVGPU_XX)
     * @param gpu_id Индекс GPU (0, 1, ...) для привязки логов
     */
    explicit ModuleRegistry(int gpu_id = 0);
    
    /**
     * @brief Деструктор (очистит все модули)
     */
    ~ModuleRegistry();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    ModuleRegistry(const ModuleRegistry&) = delete;
    ModuleRegistry& operator=(const ModuleRegistry&) = delete;
    
    ModuleRegistry(ModuleRegistry&& other) noexcept;
    ModuleRegistry& operator=(ModuleRegistry&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Регистрация модулей
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Зарегистрировать compute модуль
     * @param name Имя модуля (уникальное)
     * @param module Shared pointer на модуль
     * @throws std::runtime_error если модуль с таким именем уже существует
     */
    void RegisterModule(const std::string& name, 
                       std::shared_ptr<IComputeModule> module);
    
    /**
     * @brief Удалить модуль из реестра
     * @param name Имя модуля
     * @return true если модуль был удалён
     */
    bool UnregisterModule(const std::string& name);
    
    /**
     * @brief Проверить наличие модуля
     */
    bool HasModule(const std::string& name) const;
    
    // ═══════════════════════════════════════════════════════════════
    // Доступ к модулям
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить модуль по имени
     * @param name Имя модуля
     * @throws std::runtime_error если модуль не найден
     */
    std::shared_ptr<IComputeModule> GetModule(const std::string& name);
    
    /**
     * @brief Получить модуль по имени (const версия)
     */
    std::shared_ptr<const IComputeModule> GetModule(const std::string& name) const;
    
    /**
     * @brief Получить типизированный модуль
     * @tparam T Тип модуля (наследник IComputeModule)
     */
    template<typename T>
    std::shared_ptr<T> GetModule(const std::string& name);
    
    // ═══════════════════════════════════════════════════════════════
    // Информация о реестре
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество зарегистрированных модулей
     */
    size_t GetModuleCount() const;
    
    /**
     * @brief Получить список имён всех модулей
     */
    std::vector<std::string> GetModuleNames() const;
    
    /**
     * @brief Вывести список модулей
     */
    void PrintModules() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Очистка
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Очистить все модули
     */
    void Clear();

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    int gpu_id_;  ///< Индекс GPU для логов (DRVGPU_XX)
    std::unordered_map<std::string, std::shared_ptr<IComputeModule>> modules_;
    mutable std::mutex mutex_;
};

// ════════════════════════════════════════════════════════════════════════════
// Template реализация
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
std::shared_ptr<T> ModuleRegistry::GetModule(const std::string& name) {
    static_assert(std::is_base_of<IComputeModule, T>::value,
                  "T должен быть наследником IComputeModule");
    
    auto module = GetModule(name);
    auto typed_module = std::dynamic_pointer_cast<T>(module);
    
    if (!typed_module) {
        throw std::runtime_error(
            "ModuleRegistry::GetModule: модуль '" + name +
            "' не совпадает с запрошенным типом");
    }
    
    return typed_module;
}

} // namespace drv_gpu_lib
