// Stub replacement for ExLlamaV3 torch/ATen host headers.
//
// ExLlamaV3 ships kernels alongside host-side launcher functions whose
// signatures take `at::Tensor` and use `TORCH_CHECK_*` macros. When we
// generate PTX via `nvcc --ptx`, the host launchers are unreachable
// (they're called from Rust through cudarc, not from these files), but
// nvcc still parses every translation unit fully and refuses to compile
// if `at::Tensor` / `c10::optional` / `TORCH_CHECK*` are undefined.
//
// Two options exist: surgically strip every host launcher, or stub the
// torch namespaces just enough that vendored sources parse cleanly. We
// chose the stub. It contains only minimal declarations — they are
// never invoked at runtime because the host launcher functions are
// not exported as PTX symbols.
//
// MIT-License (ExLlamaV3) — see crates/core/LICENSE-THIRD-PARTY.

#pragma once

#include <cstdint>
#include <cstddef>
#include <tuple>

// ─── ATen / c10 minimal stubs ─────────────────────────────────────────────

namespace at {

// Scalar dtype enum tags. Real ATen uses an enum class; here a few
// constants are enough for "x.dtype() == at::kHalf" comparisons to
// compile.
struct ScalarType { int v; constexpr bool operator==(ScalarType o) const { return v == o.v; } };
constexpr ScalarType kHalf{1};
constexpr ScalarType kFloat{2};
constexpr ScalarType kInt{3};
constexpr ScalarType kShort{4};
constexpr ScalarType kByte{5};
constexpr ScalarType kLong{6};
constexpr ScalarType kBFloat16{7};
constexpr ScalarType kBool{8};

struct Device {};

// Sentinel sizes container that supports == comparison.
struct IntArrayRef {
    constexpr bool operator==(IntArrayRef) const { return true; }
};

class Tensor {
public:
    void* data_ptr() const { return nullptr; }
    int64_t size(int) const { return 0; }
    IntArrayRef sizes() const { return IntArrayRef{}; }
    int dim() const { return 0; }
    int64_t numel() const { return 0; }
    ScalarType dtype() const { return kHalf; }
    Device device() const { return Device{}; }
    Tensor contiguous() const { return *this; }
};

namespace cuda {
struct OptionalCUDAGuard { explicit OptionalCUDAGuard(Device) {} };
struct CUDAStream { cudaStream_t stream() const { return nullptr; } };
inline CUDAStream getCurrentCUDAStream() { return CUDAStream{}; }
}  // namespace at::cuda
}  // namespace at

namespace c10 {
template <typename T> class optional {
public:
    optional() = default;
    optional(const T&) {}
    bool has_value() const { return false; }
    const T& value() const { static T x; return x; }
    const T& operator*() const { return value(); }
};
namespace cuda {
struct OptionalCUDAGuard { explicit OptionalCUDAGuard(at::Device) {} };
}  // namespace c10::cuda
}  // namespace c10

// ─── TORCH_CHECK* macros: no-ops ──────────────────────────────────────────

#define TORCH_CHECK(cond, ...) if (false) {}
#define TORCH_CHECK_DTYPE(x, dt) if (false) {}
#define TORCH_CHECK_DTYPE_OPT(x, dt) if (false) {}
#define TORCH_CHECK_FLOAT_HALF(x) if (false) {}
#define TORCH_CHECK_SHAPES(x, i, y, j, s) if (false) {}
#define TORCH_CHECK_SHAPES_OPT(x, i, y, j, s) if (false) {}
#define TORCH_CHECK_SHAPES_FULL(x, y) if (false) {}
#define TORCH_CHECK_NUMEL(x, y) if (false) {}
#define TORCH_CHECK_DIV(x, i, d) if (false) {}
#define TORCH_CHECK_DIM(x, d) if (false) {}
#define TORCH_CHECK_DIM_OPT(x, d) if (false) {}
#define TORCH_CHECK_SIZE(x, i, s) if (false) {}
#define OPTPTR(x) (nullptr)

// ─── Util helpers (subset from exllamav3_ext/util.h) ─────────────────────

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// `cuda_check` is called from host launchers (which we strip); make it a
// no-op so any straggling reference still compiles.
inline void cuda_check(cudaError_t) {}
