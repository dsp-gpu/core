/**
 * @file main.cpp
 * @brief Точка входа для тестов DspCore
 *
 * Запускает все тесты через all_test.hpp (паттерн GPUWorkLib).
 * Требует GPU на Linux/ROCm. На Windows — только конфигурация.
 */

#include "all_test.hpp"

int main() {
    drvgpu_all_test::run();
    return 0;
}
