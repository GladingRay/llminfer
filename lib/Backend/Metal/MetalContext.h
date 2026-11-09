#ifndef LLMINFER_BACKEND_METAL_METALCONTEXT_H
#define LLMINFER_BACKEND_METAL_METALCONTEXT_H

#include "llminfer/Support/Result.h"
#import <Metal/Metal.h>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>

namespace llminfer::metal {

using Buffer = id<MTLBuffer>;

class MetalContext {
public:
  static Result<MetalContext> create();

  Result<Buffer> makeBuffer(uint64_t bytes) const;
  uint64_t maxBufferLength() const;
  Result<id<MTLComputePipelineState>> getPipeline(const char *name);
  id<MTLCommandBuffer> makeCommandBuffer() const;

private:
  MetalContext(id<MTLDevice> device, id<MTLCommandQueue> queue,
               id<MTLLibrary> library)
      : device(device), queue(queue), library(library) {}

  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  std::map<std::string, id<MTLComputePipelineState>, std::less<>> pipelines;
};

/// Records one graph step. A failed encoding is discarded before submission.
class Command {
public:
  explicit Command(MetalContext &context);
  Buffer allocate(uint64_t elements, unsigned elementSize);
  void reject(Error error);

  template <typename Params>
  void dispatch(const char *name, std::initializer_list<Buffer> buffers,
                const Params &params, uint64_t threads) {
    constexpr unsigned width = 64;
    encode(name, buffers, &params, sizeof(params),
           MTLSizeMake((threads + width - 1) / width, 1, 1),
           MTLSizeMake(width, 1, 1));
  }

  template <typename Params>
  void dispatchMatrix(const char *name, std::initializer_list<Buffer> buffers,
                      const Params &params, unsigned rows, unsigned columns) {
    constexpr uint64_t tile = 16;
    encode(
        name, buffers, &params, sizeof(params),
        MTLSizeMake((columns + tile - 1) / tile, (rows + tile - 1) / tile, 1),
        MTLSizeMake(tile, tile, 1));
  }

  /// Sets submitted if the GPU may have mutated persistent state.
  Error finish(bool &submitted);

private:
  void encode(const char *name, std::initializer_list<Buffer> buffers,
              const void *params, size_t paramsSize, MTLSize groups,
              MTLSize threads);

  MetalContext &context;
  id<MTLCommandBuffer> commandBuffer;
  Error failure;
};

} // namespace llminfer::metal

#endif
