kernel void gqaFlashAttention(device const half4 *query [[buffer(0)]],
                              device const half4 *keys [[buffer(1)]],
                              device const half4 *values [[buffer(2)]],
                              device half4 *output [[buffer(3)]],
                              constant GqaParams &p [[buffer(4)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simdIndex [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint maxVectors = flashGqaMaxHeadDim / flashGqaVectorWidth;
  threadgroup half4 keyTile[flashGqaKeyTile * maxVectors];
  threadgroup half4 valueTile[flashGqaKeyTile * maxVectors];

  // Each SIMD group owns one query, including its online softmax and output.
  uint firstQuery = group.x * flashGqaQueryTile;
  uint token = firstQuery + simdIndex;
  uint visible = token < p.tokens ? p.past + token + 1 : 0;
  uint lastVisible = p.past + min(firstQuery + flashGqaQueryTile, p.tokens);
  uint kvHead = group.y / (p.queryHeads / p.kvHeads);
  uint vectors = p.headDim / flashGqaVectorWidth;
  uint dim0 = lane, dim1 = lane + flashGqaSimdWidth;
  uint queryOffset = (token * p.queryHeads + group.y) * vectors;
  float4 q0 =
      visible && dim0 < vectors ? float4(query[queryOffset + dim0]) : 0.0f;
  float4 q1 =
      visible && dim1 < vectors ? float4(query[queryOffset + dim1]) : 0.0f;
  float4 out0 = 0.0f, out1 = 0.0f;
  float maxScore = -INFINITY, normalizer = 0.0f;
  float scoreScale = rsqrt(float(p.headDim));

  for (uint base = 0; base < lastVisible; base += flashGqaKeyTile) {
    uint tileKeys = min(uint(flashGqaKeyTile), lastVisible - base);
    for (uint i = tid; i < tileKeys * vectors; i += flashGqaThreads) {
      uint src =
          ((base + i / vectors) * p.kvHeads + kvHead) * vectors + i % vectors;
      keyTile[i] = keys[src];
      valueTile[i] = values[src];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All groups participate in the tile barriers, including padded queries.
    if (base < visible) {
      uint validKeys = min(tileKeys, visible - base);
      float score = -INFINITY;
      for (uint k = 0; k < validKeys; ++k) {
        uint offset = k * vectors;
        float partial = 0.0f;
        if (dim0 < vectors)
          partial = dot(q0, float4(keyTile[offset + dim0]));
        if (dim1 < vectors)
          partial += dot(q1, float4(keyTile[offset + dim1]));
        float dotProduct = simd_sum(partial);
        if (lane == k)
          score = dotProduct * scoreScale;
      }

      float nextMax = max(maxScore, simd_max(score));
      float rescale = exp(maxScore - nextMax);
      float probability = exp(score - nextMax);
      normalizer = normalizer * rescale + simd_sum(probability);
      maxScore = nextMax;
      out0 *= rescale;
      out1 *= rescale;

      for (uint k = 0; k < validKeys; ++k) {
        uint offset = k * vectors;
        float4 weight = float4(simd_shuffle(probability, ushort(k)));
        if (dim0 < vectors)
          out0 = fma(weight, float4(valueTile[offset + dim0]), out0);
        if (dim1 < vectors)
          out1 = fma(weight, float4(valueTile[offset + dim1]), out1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (visible) {
    if (dim0 < vectors)
      output[queryOffset + dim0] = half4(out0 / normalizer);
    if (dim1 < vectors)
      output[queryOffset + dim1] = half4(out1 / normalizer);
  }
}
