#include "module_registry.hpp"
#include "../logger/logger.hpp"
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// ModuleRegistry Implementation - Регистр вычислительных модулей
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор ModuleRegistry
 * @param gpu_id Индекс GPU для привязки логов (DRVGPU_XX)
 */
ModuleRegistry::ModuleRegistry(int gpu_id) : gpu_id_(gpu_id) {
}

/**
 * @brief Деструктор ModuleRegistry
 * 
 * Вызывает Clear() для освобождения всех модулей.
 */
ModuleRegistry::~ModuleRegistry() {
    Clear();
}

// ════════════════════════════════════════════════════════════════════════════
// Move конструктор и оператор присваивания
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Move конструктор
 * @param other Перемещаемый объект
 * 
 * Переносит все модули из other в новый объект.
 * Thread-safe: блокирует mutex other.
 */
ModuleRegistry::ModuleRegistry(ModuleRegistry&& other) noexcept
    : gpu_id_(other.gpu_id_) {
    std::lock_guard<std::mutex> lock(other.mutex_);
    modules_ = std::move(other.modules_);
}

/**
 * @brief Move оператор присваивания
 * @param other Перемещаемый объект
 * @return Ссылка на this
 */
ModuleRegistry& ModuleRegistry::operator=(ModuleRegistry&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock_this(mutex_);
        std::lock_guard<std::mutex> lock_other(other.mutex_);
        gpu_id_ = other.gpu_id_;
        modules_ = std::move(other.modules_);
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Регистрация модулей
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Зарегистрировать compute модуль
 * @param name Уникальное имя модуля
 * @param module Shared pointer на модуль
 * 
 * Добавляет модуль в реестр по уникальному имени.
 * 
 * @throws std::runtime_error если модуль с таким именем уже существует
 * 
 * Пример:
 * @code
 * auto fft_module = std::make_shared<FFTModule>(backend);
 * registry.RegisterModule("FFT", fft_module);
 * @endcode
 */
void ModuleRegistry::RegisterModule(const std::string& name, 
                                     std::shared_ptr<IComputeModule> module) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (modules_.find(name) != modules_.end()) {
        throw std::runtime_error("ModuleRegistry: module '" + name + "' already registered");
    }
    
    modules_[name] = module;
}

/**
 * @brief Удалить модуль из реестра
 * @param name Имя модуля для удаления
 * @return true если модуль был найден и удалён
 * 
 * Thread-safe метод для удаления модуля.
 */
bool ModuleRegistry::UnregisterModule(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.erase(name) > 0;
}

/**
 * @brief Проверить наличие модуля в реестре
 * @param name Имя модуля
 * @return true если модуль существует
 */
bool ModuleRegistry::HasModule(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.find(name) != modules_.end();
}

// ════════════════════════════════════════════════════════════════════════════
// Доступ к модулям
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить модуль по имени (не-const версия)
 * @param name Имя модуля
 * @return Shared pointer на модуль
 * 
 * @throws std::runtime_error если модуль не найден
 */
std::shared_ptr<IComputeModule> ModuleRegistry::GetModule(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        throw std::runtime_error("ModuleRegistry: module '" + name + "' not found");
    }
    
    return it->second;
}

/**
 * @brief Получить модуль по имени (const версия)
 * @param name Имя модуля
 * @return Const shared pointer на модуль
 */
std::shared_ptr<const IComputeModule> ModuleRegistry::GetModule(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        throw std::runtime_error("ModuleRegistry: module '" + name + "' not found");
    }
    
    return it->second;
}

// ════════════════════════════════════════════════════════════════════════════
// Информация о реестре
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить количество зарегистрированных модулей
 * @return Количество модулей в реестре
 */
size_t ModuleRegistry::GetModuleCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.size();
}

/**
 * @brief Получить список имён всех зарегистрированных модулей
 * @return Вектор строк с именами модулей
 */
std::vector<std::string> ModuleRegistry::GetModuleNames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> names;
    names.reserve(modules_.size());
    
    for (const auto& pair : modules_) {
        names.push_back(pair.first);
    }
    
    return names;
}

/**
 * @brief Вывести список зарегистрированных модулей в лог
 * 
 * Логирует все зарегистрированные модули с помощью DRVGPU_LOG_DEBUG.
 * Если реестр пуст, логирует соответствующее сообщение.
 */
void ModuleRegistry::PrintModules() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    DRVGPU_LOG_DEBUG_GPU(gpu_id_, "ModuleRegistry", "Printing registered modules");
    
    if (modules_.empty()) {
        DRVGPU_LOG_DEBUG_GPU(gpu_id_, "ModuleRegistry", "No modules registered");
    } else {
        for (const auto& pair : modules_) {
            DRVGPU_LOG_DEBUG_GPU(gpu_id_, "ModuleRegistry", "  - " + pair.first);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Очистить все модули из реестра
 * 
 * Удаляет все модули и освобождает память.
 * Thread-safe через mutex.
 */
void ModuleRegistry::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    modules_.clear();
}

} // namespace drv_gpu_lib
