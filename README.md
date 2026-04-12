# DspCore

> Part of [dsp-gpu](https://github.com/dsp-gpu) organization.

GPU driver core library (DrvGPU). Provides unified ROCm/HIP + OpenCL backend, profiling, logging, and test utilities.

## Dependencies

- ROCm/HIP (hip, OpenCL)

## Build

```bash
cmake -S . -B build --preset local-dev
cmake --build build
```

## Structure

```
core/
├── include/dsp/          # Public headers
├── kernels/rocm/         # PRIVATE: ROCm .hip kernels
├── src/                  # Implementation
├── test_utils/           # DspCore::TestUtils (INTERFACE target)
└── tests/                # C++ tests
```
