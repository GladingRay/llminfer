kernel void benchInt8Mac(device uint *output [[buffer(0)]],
                         constant Int8BenchParams &p [[buffer(1)]],
                         uint threadId [[thread_position_in_grid]]) {
  if (threadId >= p.threads)
    return;

  uint accum[int8BenchChains];
  char4 weights[int8BenchChains];
#pragma unroll
  for (uint chain = 0; chain < int8BenchChains; ++chain) {
    accum[chain] = int8BenchSeed(threadId, chain);
    weights[chain] = as_type<char4>(p.weights[chain]);
  }

  // Feed the low bytes of each accumulator back into its next dot product.
  // The independent chains let the compiler overlap their instructions.
  for (uint i = 0; i < p.iterations; ++i) {
#pragma unroll
    for (uint chain = 0; chain < int8BenchChains; ++chain)
      accum[chain] += uint(dotI8x4(as_type<char4>(accum[chain]), weights[chain]));
  }

#pragma unroll
  for (uint chain = 0; chain < int8BenchChains; ++chain)
    output[threadId * int8BenchChains + chain] = accum[chain];
}
