#include <metal_stdlib>
using namespace metal;

static float logistic(float x) { return 1.0f / (1.0f + exp(-x)); }

static float silu(float x) { return x * logistic(x); }

static float stableSoftplus(float x) {
  float e = exp(-abs(x));
  float tail = e < 1e-4f ? e * (1.0f - 0.5f * e) : log(1.0f + e);
  return max(x, 0.0f) + tail;
}

kernel void embedding(device const uint *ids [[buffer(0)]],
                      device const half *weight [[buffer(1)]],
                      device half *output [[buffer(2)]],
                      constant EmbedParams &p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p.tokens * p.hidden)
    return;
  uint id = ids[i / p.hidden];
  if (id >= p.firstRow && id - p.firstRow < p.rowCount)
    output[i] = weight[(id - p.firstRow) * p.hidden + i % p.hidden];
}

kernel void linearF16(device const half *input [[buffer(0)]],
                      device const half *weight [[buffer(1)]],
                      device half *output [[buffer(2)]],
                      constant LinearParams &p [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint2 local [[thread_position_in_threadgroup]]) {
  constexpr uint tile = 16;
  threadgroup half inputTile[tile][tile];
  threadgroup half weightTile[tile][tile];
  uint row = group.y * tile + local.y;
  uint col = group.x * tile + local.x;
  float acc = 0.0f;

  for (uint base = 0; base < p.inputWidth; base += tile) {
    uint inputCol = base + local.x;
    uint weightCol = base + local.y;
    inputTile[local.y][local.x] = row < p.tokens && inputCol < p.inputWidth
                                      ? input[row * p.inputWidth + inputCol]
                                      : half(0);
    weightTile[local.y][local.x] = col < p.rowCount && weightCol < p.inputWidth
                                       ? weight[col * p.inputWidth + weightCol]
                                       : half(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < tile; ++k)
      acc =
          fma(float(inputTile[local.y][k]), float(weightTile[k][local.x]), acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < p.tokens && col < p.rowCount)
    output[row * p.outputWidth + p.firstRow + col] = half(acc);
}

static float rmsScale(device const half *x, uint width, float epsilon) {
  float sum = 0.0f;
  for (uint i = 0; i < width; ++i)
    sum = fma(float(x[i]), float(x[i]), sum);
  return rsqrt(sum / float(width) + epsilon);
}

kernel void rmsNorm(device const half *input [[buffer(0)]],
                    device const float *weight [[buffer(1)]],
                    device half *output [[buffer(2)]],
                    constant RowParams &p [[buffer(3)]],
                    uint row [[thread_position_in_grid]]) {
  if (row >= p.rows)
    return;
  uint offset = row * p.width;
  float scale = rmsScale(input + offset, p.width, p.epsilon);
  for (uint i = 0; i < p.width; ++i)
    output[offset + i] =
        half(float(input[offset + i]) * scale * (1.0f + weight[i]));
}

kernel void residualAdd(device const half *lhs [[buffer(0)]],
                        device const half *rhs [[buffer(1)]],
                        device half *output [[buffer(2)]],
                        constant ElementParams &p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < p.count)
    output[i] = half(float(lhs[i]) + float(rhs[i]));
}

kernel void swiGlu(device const half *gate [[buffer(0)]],
                   device const half *up [[buffer(1)]],
                   device half *output [[buffer(2)]],
                   constant ElementParams &p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
  if (i < p.count)
    output[i] = half(float(half(silu(float(gate[i])))) * float(up[i]));
}

kernel void causalConv(device const half *input [[buffer(0)]],
                       device const half *weight [[buffer(1)]],
                       device half *history [[buffer(2)]],
                       device half *output [[buffer(3)]],
                       constant ConvParams &p [[buffer(4)]],
                       uint channel [[thread_position_in_grid]]) {
  if (channel >= p.channels)
    return;
  uint historySize = p.kernelSize - 1;
  device half *recent = history + channel * historySize;
  device const half *filter = weight + channel * p.kernelSize;
  for (uint t = 0; t < p.tokens; ++t) {
    uint i = t * p.channels + channel;
    half current = input[i];
    float acc = float(current) * float(filter[historySize]);
    for (uint j = 0; j < historySize; ++j)
      acc = fma(float(recent[j]), float(filter[j]), acc);
    output[i] = p.applySilu ? half(silu(float(half(acc)))) : half(acc);
    if (historySize != 0) {
      for (uint j = 1; j < historySize; ++j)
        recent[j - 1] = recent[j];
      recent[historySize - 1] = current;
    }
  }
}

kernel void deltaQkNorm(device const half *qkv [[buffer(0)]],
                        device float *normalized [[buffer(1)]],
                        constant DeltaNormParams &p [[buffer(2)]],
                        uint row [[thread_position_in_grid]]) {
  if (row >= p.tokens * 2 * p.keyHeads)
    return;
  uint head = row % (2 * p.keyHeads);
  uint token = row / (2 * p.keyHeads);
  device const half *input = qkv + token * p.qkvWidth + head * p.keyDim;
  float sum = 0.0f;
  for (uint i = 0; i < p.keyDim; ++i)
    sum = fma(float(input[i]), float(input[i]), sum);
  float scale = rsqrt(sum + 1e-6f);
  if (head < p.keyHeads)
    scale *= rsqrt(float(p.keyDim));
  for (uint i = 0; i < p.keyDim; ++i)
    normalized[row * p.keyDim + i] = float(input[i]) * scale;
}

static half updateDeltaColumn(device const float *q, device const float *k,
                              float v, float beta, float decay,
                              device float *column, uint keys, uint stride) {
  float prediction = 0.0f;
  for (uint i = 0; i < keys; ++i)
    prediction = fma(column[i * stride] * decay, k[i], prediction);
  float delta = (v - prediction) * beta;
  float y = 0.0f;
  for (uint i = 0; i < keys; ++i) {
    uint index = i * stride;
    float next = fma(k[i], delta, column[index] * decay);
    column[index] = next;
    y = fma(next, q[i], y);
  }
  return half(y);
}

kernel void deltaUpdate(
    device const half *qkv [[buffer(0)]], device const float *qk [[buffer(1)]],
    device const half *a [[buffer(2)]], device const half *b [[buffer(3)]],
    device const float *aLog [[buffer(4)]],
    device const float *dtBias [[buffer(5)]], device float *state [[buffer(6)]],
    device half *output [[buffer(7)]], constant DeltaParams &p [[buffer(8)]],
    uint column [[thread_position_in_grid]]) {
  uint valueWidth = p.valueHeads * p.valueDim;
  if (column >= valueWidth)
    return;
  uint valueHead = column / p.valueDim;
  uint keyHead = valueHead / (p.valueHeads / p.keyHeads);
  device float *stateColumn =
      state + valueHead * p.keyDim * p.valueDim + column % p.valueDim;
  uint keyWidth = p.keyHeads * p.keyDim;
  for (uint t = 0; t < p.tokens; ++t) {
    uint gateIndex = t * p.valueHeads + valueHead;
    float x = float(a[gateIndex]) + dtBias[valueHead];
    float softplus = stableSoftplus(x);
    float decay = exp(-exp(aLog[valueHead]) * softplus);
    float beta = float(half(logistic(float(b[gateIndex]))));
    device const float *q = qk + 2 * t * keyWidth + keyHead * p.keyDim;
    device const float *k = q + keyWidth;
    float v = float(qkv[t * p.qkvWidth + 2 * keyWidth + column]);
    output[t * valueWidth + column] = updateDeltaColumn(
        q, k, v, beta, decay, stateColumn, p.keyDim, p.valueDim);
  }
}

kernel void gatedRmsNorm(device const half *input [[buffer(0)]],
                         device const half *gate [[buffer(1)]],
                         device const float *weight [[buffer(2)]],
                         device half *output [[buffer(3)]],
                         constant RowParams &p [[buffer(4)]],
                         uint row [[thread_position_in_grid]]) {
  if (row >= p.rows)
    return;
  uint offset = row * p.width;
  float scale = rmsScale(input + offset, p.width, p.epsilon);
  for (uint i = 0; i < p.width; ++i) {
    float norm = float(half(float(input[offset + i]) * scale)) * weight[i];
    output[offset + i] = half(norm * silu(float(gate[offset + i])));
  }
}

static half2 rotatePair(half inputX, half inputY, float position, float theta,
                        uint pair, uint rotaryWidth) {
  float phase = position / pow(theta, 2.0f * float(pair) / float(rotaryWidth));
  float cosine = float(half(cos(phase)));
  float sine = float(half(sin(phase)));
  float x = float(inputX), y = float(inputY);
  return half2(half(float(half(x * cosine)) - float(half(y * sine))),
               half(float(half(y * cosine)) + float(half(x * sine))));
}

kernel void normRope(device const half *input [[buffer(0)]],
                     device const float *weight [[buffer(1)]],
                     device half *output [[buffer(2)]],
                     constant RopeParams &p [[buffer(3)]],
                     uint row [[thread_position_in_grid]]) {
  if (row >= p.tokens * p.heads)
    return;
  device const half *head = input + row * p.headStride;
  device half *out = output + (p.outputStart * p.heads + row) * p.headDim;
  float scale = rmsScale(head, p.headDim, p.epsilon);
  uint rotaryHalf = p.rotaryDim / 2;
  float position = float(p.past + row / p.heads);
  for (uint i = 0; i < rotaryHalf; ++i) {
    uint j = i + rotaryHalf;
    half x = half(float(head[i]) * scale * (1.0f + weight[i]));
    half y = half(float(head[j]) * scale * (1.0f + weight[j]));
    half2 rotated = rotatePair(x, y, position, p.theta, i, p.rotaryDim);
    out[i] = rotated.x;
    out[j] = rotated.y;
  }
  for (uint i = p.rotaryDim; i < p.headDim; ++i)
    out[i] = half(float(head[i]) * scale * (1.0f + weight[i]));
}

kernel void cacheAppend(device const half *input [[buffer(0)]],
                        device half *output [[buffer(1)]],
                        constant CacheParams &p [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < p.count)
    output[p.offset + i] = input[i];
}

kernel void gqaScores(device const half *query [[buffer(0)]],
                      device const half *keys [[buffer(1)]],
                      device half *scores [[buffer(2)]],
                      constant GqaParams &p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  uint stride = p.past + p.tokens;
  if (i >= p.tokens * p.queryHeads * stride)
    return;
  uint row = i / stride;
  uint token = row / p.queryHeads;
  uint keyToken = i % stride;
  if (keyToken > p.past + token)
    return;
  uint kvHead = (row % p.queryHeads) / (p.queryHeads / p.kvHeads);
  device const half *q = query + row * p.headDim;
  device const half *k = keys + (keyToken * p.kvHeads + kvHead) * p.headDim;
  float dotProduct = 0.0f;
  for (uint d = 0; d < p.headDim; ++d)
    dotProduct = fma(float(q[d]), float(k[d]), dotProduct);
  scores[i] = half(float(half(dotProduct)) * rsqrt(float(p.headDim)));
}

static void applyGqa(device const half *scores, device const half *values,
                     device const half *projectedQ, device half *output,
                     constant GqaParams &p, uint row) {
  if (row >= p.tokens * p.queryHeads)
    return;
  uint count = p.past + row / p.queryHeads + 1;
  uint stride = p.past + p.tokens;
  device const half *logits = scores + row * stride;
  float maxLogit = float(logits[0]);
  for (uint i = 1; i < count; ++i)
    maxLogit = max(maxLogit, float(logits[i]));
  float sum = 0.0f;
  for (uint i = 0; i < count; ++i)
    sum += exp(float(logits[i]) - maxLogit);
  uint kvHead = (row % p.queryHeads) / (p.queryHeads / p.kvHeads);

  for (uint d = 0; d < p.headDim; ++d) {
    float result = 0.0f;
    for (uint i = 0; i < count; ++i) {
      half prob = half(exp(float(logits[i]) - maxLogit) / sum);
      result =
          fma(float(prob),
              float(values[(i * p.kvHeads + kvHead) * p.headDim + d]), result);
    }
    half value = half(result);
    if (projectedQ) {
      half gate = projectedQ[(2 * row + 1) * p.headDim + d];
      value = half(float(value) * float(half(logistic(float(gate)))));
    }
    output[row * p.headDim + d] = value;
  }
}

kernel void gqaApply(device const half *scores [[buffer(0)]],
                     device const half *values [[buffer(1)]],
                     device const half *projectedQ [[buffer(2)]],
                     device half *output [[buffer(3)]],
                     constant GqaParams &p [[buffer(4)]],
                     uint row [[thread_position_in_grid]]) {
  applyGqa(scores, values, projectedQ, output, p, row);
}
