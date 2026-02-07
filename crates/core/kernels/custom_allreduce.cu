// Custom all-reduce CUDA kernels for multi-GPU tensor parallelism.
// Ported from vLLM's custom_all_reduce.cuh with adaptations for
// standalone PTX compilation (no C++ class, no torch dependency).
//
// For small tensors (< 8MB), a custom P2P kernel using direct GPU memory
// access is 2-3x faster than NCCL by avoiding NCCL's protocol overhead
// (ring/tree topology negotiation, multi-stage buffering).
//
// Two algorithms:
// 1. One-stage (one-shot): Each GPU reads all peers' data and reduces locally.
//    Best for small tensors (< 256KB per GPU) where the read amplification
//    is offset by a single barrier pair.
// 2. Two-stage (reduce-scatter + all-gather): Each GPU reduces its partition,
//    then all GPUs gather the reduced partitions. Better for medium tensors
//    where reducing read amplification outweighs the extra barrier.
//
// Memory layout:
//   - Each GPU's signal buffer: [Signal struct | tmp workspace (for 2-stage)]
//   - Signal struct contains start/end barrier flags for each block
//   - RankData struct holds pointers to each GPU's input buffer
//
// Requires: sm_80+ (Ampere) for native bf16 arithmetic and
//           acquire/release memory ordering via PTX instructions.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================

// Maximum number of thread blocks in all-reduce kernels.
// 36 blocks is optimal across A100/A30/L40S based on vLLM benchmarks:
// too few blocks underutilizes NVLink bandwidth, too many causes
// contention on the interconnect.
static constexpr int kMaxBlocks = 36;

// Maximum supported GPU count per all-reduce group.
static constexpr int kMaxGpus = 8;

// Counter type for barrier synchronization. Unsigned overflow is
// well-defined in C/CUDA, so counters can wrap around safely.
using FlagType = uint32_t;

// ============================================================================
// Data structures
//
// These must be allocated in IPC-accessible GPU memory (cudaMalloc +
// cudaIpcGetMemHandle) so all GPUs in the group can access each
// other's signals and data through P2P.
// ============================================================================

// Synchronization signals for inter-GPU barriers.
//
// Two separate flag arrays (start/end) prevent a race where a fast GPU
// reaches the second barrier while a slow GPU is still at the first.
// Each block has its own flag slot to avoid cross-block contention.
struct Signal {
    alignas(128) FlagType start[kMaxBlocks][kMaxGpus];
    alignas(128) FlagType end[kMaxBlocks][kMaxGpus];
    alignas(128) FlagType _flag[kMaxBlocks];
};

// Pointers to each rank's input data, stored in device memory so
// kernels can dereference peer GPU pointers directly via P2P/NVLink.
struct __align__(16) RankData {
    const void* ptrs[kMaxGpus];
};

// Pointers to each rank's Signal structure.
struct __align__(16) RankSignals {
    Signal* signals[kMaxGpus];
};

// ============================================================================
// Packed types for 128-bit memory transactions
//
// Using 128-bit loads/stores (ld.128/st.128) maximizes NVLink bandwidth
// utilization. For bf16, each packed load fetches 8 elements; for f32,
// 4 elements. Reduction accumulates in f32 to avoid precision loss.
// ============================================================================

template <typename T, int N>
struct __align__(sizeof(T) * N) PackedArray {
    T data[N];
};

// BF16: 8 elements per 128-bit transaction, accumulate in f32
struct PackedBf16 {
    using LoadType = PackedArray<__nv_bfloat16, 8>;
    using AccType = PackedArray<float, 8>;
    static constexpr int kPackSize = 8;
};

// F32: 4 elements per 128-bit transaction, accumulate in f32
struct PackedF32 {
    using LoadType = PackedArray<float, 4>;
    using AccType = PackedArray<float, 4>;
    static constexpr int kPackSize = 4;
};

// ============================================================================
// Memory ordering primitives (PTX inline assembly)
//
// These provide the minimum necessary ordering for inter-GPU synchronization
// via P2P memory. We use:
// - st.release / ld.acquire for barriers where subsequent reads must
//   see prior writes from peer GPUs (the "end" barrier between stages)
// - st.volatile / ld.volatile for barriers where we only need to ensure
//   the flag write is visible (the "start" barrier and final barrier)
// ============================================================================

__device__ __forceinline__ void st_flag_release(FlagType* addr, FlagType val) {
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(val), "l"(addr));
}

__device__ __forceinline__ FlagType ld_flag_acquire(FlagType* addr) {
    FlagType val;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(val) : "l"(addr));
    return val;
}

__device__ __forceinline__ void st_flag_volatile(FlagType* addr, FlagType val) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(val), "l"(addr));
}

__device__ __forceinline__ FlagType ld_flag_volatile(FlagType* addr) {
    FlagType val;
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(val) : "l"(addr));
    return val;
}

// ============================================================================
// Barrier implementations
//
// Each GPU block writes its "I'm here" flag to all peer GPUs, then
// spins until it sees the expected flag from all peers. The flag value
// increments monotonically (with wrap-around) to distinguish successive
// barrier instances.
//
// Only threads 0..world_size-1 participate in flag exchange. All
// threads sync via __syncthreads() before/after.
// ============================================================================

// Start barrier: used before reading peer data. Peers need only see the
// flag write (volatile), not any prior data writes.
__device__ void barrier_start(
    const RankSignals& sg,
    Signal* self_sg,
    int rank,
    int world_size
) {
    FlagType flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < world_size) {
        FlagType* peer_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
        FlagType* self_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
        st_flag_volatile(peer_ptr, flag);
        while (ld_flag_volatile(self_ptr) != flag);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self_sg->_flag[blockIdx.x] = flag;
    }
}

// End barrier: used after writing reduced data. Uses release/acquire
// semantics so the reading GPU sees the writes that preceded this barrier.
// When final_sync is true (last barrier in kernel), volatile ordering
// suffices since no subsequent reads depend on this data within the kernel.
__device__ void barrier_end(
    const RankSignals& sg,
    Signal* self_sg,
    int rank,
    int world_size,
    bool final_sync
) {
    __syncthreads();
    FlagType flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < world_size) {
        FlagType* peer_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
        FlagType* self_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
        if (final_sync) {
            st_flag_volatile(peer_ptr, flag);
            while (ld_flag_volatile(self_ptr) != flag);
        } else {
            st_flag_release(peer_ptr, flag);
            while (ld_flag_acquire(self_ptr) != flag);
        }
    }
    if (!final_sync) {
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        self_sg->_flag[blockIdx.x] = flag;
    }
}

// ============================================================================
// Packed reduction helpers
// ============================================================================

__device__ __forceinline__ PackedBf16::AccType
upcast_bf16(PackedBf16::LoadType val) {
    PackedBf16::AccType out;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out.data[i] = __bfloat162float(val.data[i]);
    }
    return out;
}

__device__ __forceinline__ PackedBf16::LoadType
downcast_bf16(PackedBf16::AccType val) {
    PackedBf16::LoadType out;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out.data[i] = __float2bfloat16(val.data[i]);
    }
    return out;
}

__device__ __forceinline__ void
acc_add_bf16(PackedBf16::AccType& acc, PackedBf16::LoadType val) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc.data[i] += __bfloat162float(val.data[i]);
    }
}

__device__ __forceinline__ void
acc_add_f32(PackedF32::AccType& acc, PackedF32::LoadType val) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc.data[i] += val.data[i];
    }
}

// ============================================================================
// Helper: get tmp buffer pointer (stored after Signal struct)
// ============================================================================

template <typename P>
__device__ __forceinline__ P* get_tmp_buf(Signal* sg) {
    return (P*)(((Signal*)sg) + 1);
}

// ============================================================================
// One-stage all-reduce (BF16)
//
// Algorithm: barrier -> each thread reads the same element from all
// GPUs' input buffers and sums -> write result -> barrier.
//
// This is optimal when the tensor is small enough that the read
// amplification (world_size reads per element) is cheaper than the
// extra barrier of the two-stage approach.
//
// Grid: (min(kMaxBlocks, ceil(packed_size / 512)),)
// Block: (512,)
// ============================================================================

extern "C" __global__ void __launch_bounds__(512, 1)
custom_allreduce_1stage_bf16(
    RankData* __restrict__ rank_data,
    RankSignals rank_signals,
    Signal* __restrict__ self_signal,
    __nv_bfloat16* __restrict__ result,
    const int rank,
    const int world_size,
    const int packed_size            // N / PackedBf16::kPackSize
) {
    using P = PackedBf16::LoadType;
    using A = PackedBf16::AccType;

    RankData dp = *rank_data;
    barrier_start(rank_signals, self_signal, rank, world_size);

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < packed_size;
         idx += stride) {
        // Accumulate in f32 for precision
        A acc = upcast_bf16(((const P*)dp.ptrs[0])[idx]);
        for (int g = 1; g < world_size; g++) {
            acc_add_bf16(acc, ((const P*)dp.ptrs[g])[idx]);
        }
        ((P*)result)[idx] = downcast_bf16(acc);
    }

    barrier_end(rank_signals, self_signal, rank, world_size, true);
}

// ============================================================================
// Two-stage all-reduce (BF16)
//
// Stage 1 (reduce-scatter): Each GPU reduces its assigned partition
//   from all peers and writes the partial result to its tmp buffer.
// Stage 2 (all-gather): Each GPU reads all peers' tmp buffers to
//   assemble the full reduced output.
//
// This halves the total bytes read compared to one-stage at the cost
// of an extra barrier between stages.
//
// Grid: (min(kMaxBlocks, ceil(packed_size / 512)),)
// Block: (512,)
// ============================================================================

extern "C" __global__ void __launch_bounds__(512, 1)
custom_allreduce_2stage_bf16(
    RankData* __restrict__ rank_data,
    RankSignals rank_signals,
    Signal* __restrict__ self_signal,
    __nv_bfloat16* __restrict__ result,
    const int rank,
    const int world_size,
    const int packed_size
) {
    using P = PackedBf16::LoadType;
    using A = PackedBf16::AccType;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int part = packed_size / world_size;
    int start = rank * part;
    int end = (rank == world_size - 1) ? packed_size : start + part;
    int largest_part = part + packed_size % world_size;

    // Collect pointers with rotation to balance NVLink traffic
    const P* ptrs[kMaxGpus];
    P* tmps[kMaxGpus];
    for (int i = 0; i < world_size; i++) {
        int target = (rank + i) % world_size;
        ptrs[i] = (const P*)rank_data->ptrs[target];
        tmps[i] = get_tmp_buf<P>(rank_signals.signals[target]);
    }
    P* tmp_out = tmps[0];

    barrier_start(rank_signals, self_signal, rank, world_size);

    // Stage 1: reduce-scatter — each GPU reduces its assigned partition
    for (int idx = start + tid; idx < end; idx += stride) {
        A acc = upcast_bf16(ptrs[0][idx]);
        for (int g = 1; g < world_size; g++) {
            acc_add_bf16(acc, ptrs[g][idx]);
        }
        tmp_out[idx - start] = downcast_bf16(acc);
    }

    barrier_end(rank_signals, self_signal, rank, world_size, false);

    // Stage 2: all-gather — each GPU reads all reduced partitions.
    // Thread identity must match between stages: thread i that reduced
    // element X in stage 1 also gathers element X in stage 2, because
    // inter-GPU memory visibility is only guaranteed between matching tids.
    for (int idx = tid; idx < largest_part; idx += stride) {
        for (int i = 0; i < world_size; i++) {
            int gather_from = (rank + i) % world_size;
            if (gather_from == world_size - 1 || idx < part) {
                int dst_idx = gather_from * part + idx;
                ((P*)result)[dst_idx] = tmps[i][idx];
            }
        }
    }

    barrier_end(rank_signals, self_signal, rank, world_size, true);
}

// ============================================================================
// One-stage all-reduce (F32)
// ============================================================================

extern "C" __global__ void __launch_bounds__(512, 1)
custom_allreduce_1stage_f32(
    RankData* __restrict__ rank_data,
    RankSignals rank_signals,
    Signal* __restrict__ self_signal,
    float* __restrict__ result,
    const int rank,
    const int world_size,
    const int packed_size            // N / PackedF32::kPackSize
) {
    using P = PackedF32::LoadType;
    using A = PackedF32::AccType;

    RankData dp = *rank_data;
    barrier_start(rank_signals, self_signal, rank, world_size);

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < packed_size;
         idx += stride) {
        A acc;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc.data[j] = ((const P*)dp.ptrs[0])[idx].data[j];
        }
        for (int g = 1; g < world_size; g++) {
            acc_add_f32(acc, ((const P*)dp.ptrs[g])[idx]);
        }
        ((P*)result)[idx] = acc;
    }

    barrier_end(rank_signals, self_signal, rank, world_size, true);
}

// ============================================================================
// Two-stage all-reduce (F32)
// ============================================================================

extern "C" __global__ void __launch_bounds__(512, 1)
custom_allreduce_2stage_f32(
    RankData* __restrict__ rank_data,
    RankSignals rank_signals,
    Signal* __restrict__ self_signal,
    float* __restrict__ result,
    const int rank,
    const int world_size,
    const int packed_size
) {
    using P = PackedF32::LoadType;
    using A = PackedF32::AccType;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int part = packed_size / world_size;
    int start = rank * part;
    int end = (rank == world_size - 1) ? packed_size : start + part;
    int largest_part = part + packed_size % world_size;

    const P* ptrs[kMaxGpus];
    P* tmps[kMaxGpus];
    for (int i = 0; i < world_size; i++) {
        int target = (rank + i) % world_size;
        ptrs[i] = (const P*)rank_data->ptrs[target];
        tmps[i] = get_tmp_buf<P>(rank_signals.signals[target]);
    }
    P* tmp_out = tmps[0];

    barrier_start(rank_signals, self_signal, rank, world_size);

    // Stage 1: reduce-scatter
    for (int idx = start + tid; idx < end; idx += stride) {
        A acc;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc.data[j] = ptrs[0][idx].data[j];
        }
        for (int g = 1; g < world_size; g++) {
            acc_add_f32(acc, ptrs[g][idx]);
        }
        tmp_out[idx - start] = acc;
    }

    barrier_end(rank_signals, self_signal, rank, world_size, false);

    // Stage 2: all-gather
    for (int idx = tid; idx < largest_part; idx += stride) {
        for (int i = 0; i < world_size; i++) {
            int gather_from = (rank + i) % world_size;
            if (gather_from == world_size - 1 || idx < part) {
                int dst_idx = gather_from * part + idx;
                ((P*)result)[dst_idx] = tmps[i][idx];
            }
        }
    }

    barrier_end(rank_signals, self_signal, rank, world_size, true);
}

// ============================================================================
// Standalone barrier kernel
//
// Useful for explicit synchronization between GPUs outside of all-reduce,
// e.g. before/after P2P memory copies or pipeline stage boundaries.
//
// Grid: (1,)
// Block: (max(world_size, 32),)  — at least one warp
// ============================================================================

extern "C" __global__ void custom_allreduce_barrier(
    RankSignals rank_signals,
    Signal* __restrict__ self_signal,
    const int rank,
    const int world_size
) {
    barrier_start(rank_signals, self_signal, rank, world_size);
    barrier_end(rank_signals, self_signal, rank, world_size, true);
}
