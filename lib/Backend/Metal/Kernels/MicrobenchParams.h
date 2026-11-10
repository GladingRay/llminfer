#ifndef LLMINFER_BACKEND_METAL_MICROBENCHPARAMS_H
#define LLMINFER_BACKEND_METAL_MICROBENCHPARAMS_H

enum : unsigned { int8BenchChains = 8 };

struct Int8BenchParams {
  unsigned threads, iterations;
  unsigned weights[int8BenchChains];
};

inline unsigned int8BenchSeed(unsigned threadId, unsigned chain) {
  return threadId * 747796405u + (chain + 1) * 2891336453u;
}

#endif
