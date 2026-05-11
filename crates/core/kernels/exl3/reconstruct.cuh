#pragma once

#include "exl3_torch_stub.h"

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);
