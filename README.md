# vs-boxblur
Provides AVX2-vectorized routines for uint16 and float32 inputs.

For int16 input, it favors architectures with fast cross lane shuffle  (e.g. haswell or later architectures of intel) or slow integer division (e.g. pre-zen3 architectures of amd).

## Usage
Prototype:

`box.Blur(vnode clip[, int[] planes, int hradius = 1, int hpasses = 1, int vradius = 1, int vpasses = 1])`

## Building
```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
```
