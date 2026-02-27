// CUDA sequential scan kernel for the Mamba2 SSD recurrence.
//
// Recurrence per time step t, for each (batch b, head h, position p, state n):
//
//   decay          = exp(dt[b, t, h] * a[h])
//   h[b,h,p,n]    = decay * h[b,h,p,n] + dt[b,t,h] * x[b,t,h,p] * b[b,t,g,n]
//   y[b,t,h,p]    = sum_n(h[b,h,p,n] * c[b,t,g,n]) + d[h] * x[b,t,h,p]
//
// where g = h / heads_per_group.
//
// Parallelism:
//   Grid:  (batch * num_heads, 1, 1)  — one block per (b, h) pair
//   Block: (head_dim, 1, 1)           — one thread per p position
//
// Shared memory per block: 2 * d_state * sizeof(float)
//   b_sm[0..d_state]: b projection vector for the current time step
//   c_sm[0..d_state]: c projection vector for the current time step
//
//   b and c are loaded cooperatively by all threads in the block once per
//   time step, amortising the global-memory bandwidth over all P threads.
//
// Register state: each thread maintains float h_state[MAX_D_STATE] for its
//   p-strip.  h_state stays in registers for small d_state; the compiler may
//   spill to L1-backed local memory when d_state approaches MAX_D_STATE, which
//   is transparent to correctness.
//
// Packed output layout: out[0..y_elems] = y, out[y_elems..] = hn.
//   The Rust wrapper splits the flat buffer into (y, hn) tensors via narrow().
//
// Constraints enforced by the Rust wrapper before launching:
//   head_dim <= 1024  (CUDA block-size limit)
//   d_state  <= MAX_D_STATE (register array bound)

#include <cuda_runtime.h>
#include <math.h>

// Compile-time upper bound on d_state.
// Increase if larger state dimensions are required.
#define MAX_D_STATE 256

extern "C" __global__ void ssd_scan_f32(
    float* __restrict__ out,         // [y_elems + hn_elems]: y then hn (flat)
    const float* __restrict__ x,     // [B, L, H, P]
    const float* __restrict__ dt,    // [B, L, H]
    const float* __restrict__ a,     // [H]
    const float* __restrict__ b,     // [B, L, G, N]  (grouped)
    const float* __restrict__ c,     // [B, L, G, N]  (grouped)
    const float* __restrict__ d_in,  // [H]
    const float* __restrict__ h0,    // [B, H, P, N]
    int y_elems,                     // B * L * H * P (offset to hn section)
    int seq_len,
    int num_heads,
    int head_dim,
    int n_groups,
    int d_state,
    int heads_per_group
) {
    // Split packed output into y and hn sub-arrays.
    float* y  = out;
    float* hn = out + y_elems;

    const int h_idx = blockIdx.x % num_heads;
    const int b_idx = blockIdx.x / num_heads;
    const int p     = threadIdx.x;

    // Shared memory: b_sm[d_state] | c_sm[d_state]
    extern __shared__ float shmem[];
    float* b_sm = shmem;
    float* c_sm = shmem + d_state;

    // Per-head constants.
    const float a_h = a[h_idx];
    const float d_h = d_in[h_idx];
    const int   g_idx = h_idx / heads_per_group;

    // Initialise per-thread state from h0[b_idx, h_idx, p, *].
    float h_state[MAX_D_STATE];
    {
        const int base = ((b_idx * num_heads + h_idx) * head_dim + p) * d_state;
        for (int n = 0; n < d_state; ++n) {
            h_state[n] = h0[base + n];
        }
    }

    // Sequential scan over time steps.
    for (int t = 0; t < seq_len; ++t) {

        // ── Cooperative load of b_sm and c_sm ─────────────────────────────
        // b[b_idx, t, g_idx, *] and c[b_idx, t, g_idx, *] are shared across
        // all P threads in this block; load them once into shared memory.
        //
        // Each thread loads indices n = threadIdx.x, n + blockDim.x, ...
        // so the full d_state range is covered regardless of block size.
        {
            const int base = ((b_idx * seq_len + t) * n_groups + g_idx) * d_state;
            for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
                b_sm[n] = b[base + n];
                c_sm[n] = c[base + n];
            }
        }
        __syncthreads();  // ensure b_sm/c_sm are visible to all threads

        // ── Per-thread state update and output ────────────────────────────
        {
            const float dt_t  = dt[(b_idx * seq_len + t) * num_heads + h_idx];
            const float decay = expf(dt_t * a_h);
            const float x_p   = x[((b_idx * seq_len + t) * num_heads + h_idx) * head_dim + p];

            float y_acc = 0.0f;
            for (int n = 0; n < d_state; ++n) {
                const float new_h = decay * h_state[n] + dt_t * x_p * b_sm[n];
                h_state[n] = new_h;
                y_acc += new_h * c_sm[n];
            }
            y[((b_idx * seq_len + t) * num_heads + h_idx) * head_dim + p] =
                y_acc + d_h * x_p;
        }

        // Protect b_sm/c_sm from next iteration's cooperative load until
        // all threads finish reading the current values.
        __syncthreads();
    }

    // Write final state to hn[b_idx, h_idx, p, *].
    {
        const int base = ((b_idx * num_heads + h_idx) * head_dim + p) * d_state;
        for (int n = 0; n < d_state; ++n) {
            hn[base + n] = h_state[n];
        }
    }
}
