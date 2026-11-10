kernel void copyHalfView(device const half *source [[buffer(0)]],
                         device half *output [[buffer(1)]],
                         constant HalfViewParams &p [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= p.count)
    return;
  uint linear = i;
  uint offset = p.offset;
  for (int d = 2; d >= 0; --d) {
    offset += (linear % p.dims[d]) * p.strides[d];
    linear /= p.dims[d];
  }
  output[i] = source[offset];
}

kernel void embeddingW8(device const uint *ids [[buffer(0)]],
                        device const char *weight [[buffer(1)]],
                        device const half *scales [[buffer(2)]],
                        device half *output [[buffer(3)]],
                        constant GroupedEmbedParams &p [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= p.tokens * p.hidden)
    return;
  uint id = ids[i / p.hidden];
  if (id < p.firstRow || id - p.firstRow >= p.rowCount)
    return;
  uint row = id - p.firstRow;
  uint col = i % p.hidden;
  output[i] = half(float(weight[row * p.groups * 32 + col]) *
                   float(scales[row * p.groups + col / 32]));
}

kernel void quantizeActivation(device const half *input [[buffer(0)]],
                               device char *quantized [[buffer(1)]],
                               device float *scales [[buffer(2)]],
                               device atomic_uint *invalid [[buffer(3)]],
                               constant QuantizeParams &p [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.rows * p.groups)
    return;
  uint row = i / p.groups;
  uint begin = (i % p.groups) * 32;
  uint length = min(32u, p.width - begin);
  device const half *group = input + row * p.width + begin;
  float maxAbs = 0.0f;
  bool finite = true;
  for (uint j = 0; j < length; ++j) {
    float value = float(group[j]);
    finite = finite && isfinite(value);
    maxAbs = max(maxAbs, abs(value));
  }
  if (!finite)
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  float scale = finite && maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
  scales[i] = scale;
  for (uint j = 0; j < 32; ++j) {
    int rounded = 0;
    if (finite && j < length) {
      float x = clamp(float(group[j]) / scale, -127.0f, 127.0f);
      float lower = floor(x);
      float fraction = x - lower;
      rounded = int(lower);
      if (fraction > 0.5f || (fraction == 0.5f && (rounded % 2 != 0)))
        ++rounded;
    }
    quantized[i * 32 + j] = char(rounded);
  }
}

kernel void siluF16(device const half *input [[buffer(0)]],
                    device half *output [[buffer(1)]],
                    constant ElementParams &p [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  if (i < p.count)
    output[i] = half(silu(float(input[i])));
}

kernel void sigmoidMul(device const half *gate [[buffer(0)]],
                       device const half *input [[buffer(1)]],
                       device half *output [[buffer(2)]],
                       constant ElementParams &p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < p.count)
    output[i] = half(float(half(logistic(float(gate[i])))) * float(input[i]));
}

kernel void multiplyF16(device const half *lhs [[buffer(0)]],
                        device const half *rhs [[buffer(1)]],
                        device half *output [[buffer(2)]],
                        constant ElementParams &p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < p.count)
    output[i] = half(float(lhs[i]) * float(rhs[i]));
}

kernel void l2NormF32(device const half *input [[buffer(0)]],
                      device float *output [[buffer(1)]],
                      constant L2Params &p [[buffer(2)]],
                      uint row [[thread_position_in_grid]]) {
  if (row >= p.rows)
    return;
  uint offset = row * p.width;
  float sum = 0.0f;
  for (uint i = 0; i < p.width; ++i)
    sum = fma(float(input[offset + i]), float(input[offset + i]), sum);
  float factor = rsqrt(sum + p.epsilon) * p.scale;
  for (uint i = 0; i < p.width; ++i)
    output[offset + i] = float(input[offset + i]) * factor;
}

kernel void partialRope(device const half *input [[buffer(0)]],
                        device half *output [[buffer(1)]],
                        constant PartialRopeParams &p [[buffer(2)]],
                        uint row [[thread_position_in_grid]]) {
  if (row >= p.rows)
    return;
  uint offset = row * p.width;
  float position = float(p.past + row / p.heads);
  uint rotaryHalf = p.rotaryWidth / 2;
  for (uint i = 0; i < rotaryHalf; ++i) {
    uint j = i + rotaryHalf;
    half2 rotated = rotatePair(input[offset + i], input[offset + j], position,
                               p.theta, i, p.rotaryWidth);
    output[offset + i] = rotated.x;
    output[offset + j] = rotated.y;
  }
  for (uint i = p.rotaryWidth; i < p.width; ++i)
    output[offset + i] = input[offset + i];
}

kernel void deltaGates(device const half *a [[buffer(0)]],
                       device const half *b [[buffer(1)]],
                       device const float *aLog [[buffer(2)]],
                       device const float *dtBias [[buffer(3)]],
                       device float *beta [[buffer(4)]],
                       device float *decay [[buffer(5)]],
                       constant RowParams &p [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= p.rows * p.width)
    return;
  uint head = i % p.width;
  beta[i] = float(half(logistic(float(b[i]))));
  decay[i] = exp(-exp(aLog[head]) * stableSoftplus(float(a[i]) + dtBias[head]));
}

kernel void graphDeltaUpdate(device const float *q [[buffer(0)]],
                             device const float *k [[buffer(1)]],
                             device const half *v [[buffer(2)]],
                             device const float *beta [[buffer(3)]],
                             device const float *decay [[buffer(4)]],
                             device float *state [[buffer(5)]],
                             device half *output [[buffer(6)]],
                             constant GraphDeltaParams &p [[buffer(7)]],
                             uint column [[thread_position_in_grid]]) {
  uint valueWidth = p.valueHeads * p.valueDim;
  if (column >= valueWidth)
    return;
  uint valueHead = column / p.valueDim;
  uint keyHead = valueHead / (p.valueHeads / p.keyHeads);
  device float *stateColumn =
      state + valueHead * p.keyDim * p.valueDim + column % p.valueDim;
  for (uint t = 0; t < p.tokens; ++t) {
    uint input = (t * p.keyHeads + keyHead) * p.keyDim;
    uint gate = t * p.valueHeads + valueHead;
    uint index = t * valueWidth + column;
    output[index] =
        updateDeltaColumn(q + input, k + input, float(v[index]), beta[gate],
                          decay[gate], stateColumn, p.keyDim, p.valueDim);
  }
}

kernel void gqaSoftmax(device half *scores [[buffer(0)]],
                       constant GqaParams &p [[buffer(1)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint simdIndex [[simdgroup_index_in_threadgroup]],
                       uint simdLane [[thread_index_in_simdgroup]],
                       uint simdWidth [[threads_per_simdgroup]]) {
  uint count = p.past + row / p.queryHeads + 1;
  device half *logits = scores + row * (p.past + p.tokens);
  uint simdGroups = (gqaSoftmaxWorkgroupSize + simdWidth - 1) / simdWidth;
  threadgroup float maxima[gqaSoftmaxWorkgroupSize];
  threadgroup float sums[gqaSoftmaxWorkgroupSize];

  float maximum = -INFINITY;
  for (uint i = lane; i < count; i += gqaSoftmaxWorkgroupSize)
    maximum = max(maximum, float(logits[i]));
  maximum = simd_max(maximum);
  if (simdLane == 0)
    maxima[simdIndex] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxLogit = maxima[0];
  for (uint i = 1; i < simdGroups; ++i)
    maxLogit = max(maxLogit, maxima[i]);

  float sum = 0.0f;
  for (uint i = lane; i < count; i += gqaSoftmaxWorkgroupSize)
    sum += exp(float(logits[i]) - maxLogit);
  sum = simd_sum(sum);
  if (simdLane == 0)
    sums[simdIndex] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float normalizer = 0.0f;
  for (uint i = 0; i < simdGroups; ++i)
    normalizer += sums[i];

  // Keep the FP16 probability rounding used by the reference attention path.
  for (uint i = lane; i < count; i += gqaSoftmaxWorkgroupSize)
    logits[i] = half(exp(float(logits[i]) - maxLogit) / normalizer);
}

kernel void gqaValue(device const half *probabilities [[buffer(0)]],
                     device const half *values [[buffer(1)]],
                     device half *output [[buffer(2)]],
                     constant GqaParams &p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  uint row = i / p.headDim;
  if (row >= p.tokens * p.queryHeads)
    return;
  uint dim = i % p.headDim;
  uint count = p.past + row / p.queryHeads + 1;
  uint kvHead = (row % p.queryHeads) / (p.queryHeads / p.kvHeads);
  device const half *prob = probabilities + row * (p.past + p.tokens);
  device const half *value = values + kvHead * p.headDim + dim;
  uint stride = p.kvHeads * p.headDim;
  float result = 0.0f;
  for (uint k = 0; k < count; ++k)
    result = fma(float(prob[k]), float(value[k * stride]), result);
  output[i] = half(result);
}
