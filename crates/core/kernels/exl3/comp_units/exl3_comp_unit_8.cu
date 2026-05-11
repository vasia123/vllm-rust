#include <cuda_fp16.h>
#include "../exl3_torch_stub.h"
#include "../exl3_torch_stub.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../exl3_torch_stub.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "../exl3_gemm_kernel.cuh"
#include "../exl3_gemm_decode_kernel.cuh"
#include "exl3_comp_unit_8.cuh"

ALL_EXL3_KERNEL_INSTANCES(8)
