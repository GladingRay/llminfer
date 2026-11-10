inline int4 macPackedI8x16(int4 acc, uint4 a, uint4 w) {
  acc += multiplyI8x4(as_type<char4>(a.x), as_type<char4>(w.x));
  acc += multiplyI8x4(as_type<char4>(a.y), as_type<char4>(w.y));
  acc += multiplyI8x4(as_type<char4>(a.z), as_type<char4>(w.z));
  acc += multiplyI8x4(as_type<char4>(a.w), as_type<char4>(w.w));
  return acc;
}

template <uint Lanes, typename T>
inline T reduceW8Column(T acc, uint tid, uint simdWidth,
                        threadgroup T *partials) {
  if (simdWidth % Lanes == 0) {
    for (ushort mask = Lanes / 2; mask != 0; mask >>= 1)
      acc += simd_shuffle_xor(acc, mask);
    return acc;
  }
  // A logical lane group can straddle SIMD groups on a narrower device.
  partials[tid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % Lanes == 0) {
    for (uint i = 1; i < Lanes; ++i)
      acc += partials[tid + i];
  }
  return acc;
}

kernel void linearW8A8Gemv(device const uint4 *activation [[buffer(0)]],
                           device const float *activationScales [[buffer(1)]],
                           device const uint4 *weight [[buffer(2)]],
                           device const half *weightScales [[buffer(3)]],
                           device half *output [[buffer(4)]],
                           constant LinearW8Params &p [[buffer(5)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint simdWidth [[threads_per_simdgroup]]) {
  uint col = group * linearW8GemvColumns + tid / linearW8GemvLanes;
  uint lane = tid % linearW8GemvLanes;
  float acc = 0.0f;
  if (col < p.columns) {
    for (uint g = lane; g < p.groups; g += linearW8GemvLanes) {
      uint w = col * p.groups + g;
      int4 products = 0;
#pragma unroll
      for (uint i = 0; i < 2; ++i)
        products =
            macPackedI8x16(products, activation[g * 2 + i], weight[w * 2 + i]);
      acc += float(sumInt4(products)) * activationScales[g] *
             float(weightScales[w]);
    }
  }
  threadgroup float partials[linearW8WorkgroupSize];
  acc = reduceW8Column<linearW8GemvLanes>(acc, tid, simdWidth, partials);
  if (lane == 0 && col < p.columns)
    output[p.firstRow + col] = half(acc);
}

kernel void linearW8A8Gemm(device const uint4 *activation [[buffer(0)]],
                           device const float *activationScales [[buffer(1)]],
                           device const uint4 *weight [[buffer(2)]],
                           device const half *weightScales [[buffer(3)]],
                           device half *output [[buffer(4)]],
                           constant LinearW8Params &p [[buffer(5)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint simdWidth [[threads_per_simdgroup]]) {
  uint col = group.x * linearW8GemmColumns + tid / linearW8GemmLanes;
  uint row = group.y * linearW8GemmRows;
  uint lane = tid % linearW8GemmLanes;
  float4 acc = 0.0f;
  if (col < p.columns) {
    for (uint g = lane; g < p.groups; g += linearW8GemmLanes) {
      uint w = col * p.groups + g;
      int4 products[linearW8GemmRows] = {};
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        // Reuse each packed weight vector across the rows of this tile.
        uint4 weights = weight[w * 2 + i];
#pragma unroll
        for (uint r = 0; r < linearW8GemmRows; ++r) {
          if (row + r < p.rows) {
            uint a = (row + r) * p.groups + g;
            products[r] =
                macPackedI8x16(products[r], activation[a * 2 + i], weights);
          }
        }
      }
      float ws = float(weightScales[w]);
#pragma unroll
      for (uint r = 0; r < linearW8GemmRows; ++r) {
        if (row + r < p.rows)
          acc[r] += float(sumInt4(products[r])) *
                    activationScales[(row + r) * p.groups + g] * ws;
      }
    }
  }
  threadgroup float4 partials[linearW8WorkgroupSize];
  acc = reduceW8Column<linearW8GemmLanes>(acc, tid, simdWidth, partials);
  if (lane == 0 && col < p.columns) {
#pragma unroll
    for (uint r = 0; r < linearW8GemmRows; ++r)
      if (row + r < p.rows)
        output[(row + r) * p.outputWidth + p.firstRow + col] = half(acc[r]);
  }
}
