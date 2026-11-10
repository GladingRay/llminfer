#include "MetalContext.h"
#include "KernelSource.h"
#include <algorithm>

using namespace llminfer;
using namespace llminfer::metal;

namespace {
Error metalError(std::string message, NSError *error) {
  if (const char *detail = error.localizedDescription.UTF8String)
    message += ": " + std::string(detail);
  return Error(std::move(message));
}
} // namespace

Result<MetalContext> MetalContext::create() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device)
    return Error("No Metal device available");
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue)
    return Error("Cannot create a Metal command queue");

  auto *options = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  NSString *source = [NSString stringWithUTF8String:metalKernelSource];
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source
                                                options:options
                                                  error:&error];
  if (!library)
    return metalError("Compiling Metal kernels", error);
  return MetalContext(device, queue, library);
}

Result<Buffer> MetalContext::makeBuffer(uint64_t bytes) const {
  if (bytes > maxBufferLength())
    return Error("Metal buffer exceeds maxBufferLength");
  Buffer buffer =
      [device newBufferWithLength:std::max<uint64_t>(4, bytes)
                          options:MTLResourceStorageModeShared |
                                  MTLResourceHazardTrackingModeTracked];
  if (!buffer)
    return Error("Cannot allocate Metal buffer (" + std::to_string(bytes) +
                 " bytes)");
  return buffer;
}

uint64_t MetalContext::maxBufferLength() const {
  return device.maxBufferLength;
}

Result<id<MTLComputePipelineState>>
MetalContext::getPipeline(const char *name) {
  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;
  id<MTLFunction> function =
      [library newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!function)
    return Error("Missing Metal kernel: " + std::string(name));
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline)
    return metalError("Creating Metal pipeline " + std::string(name), error);
  pipelines.emplace(name, pipeline);
  return pipeline;
}

id<MTLCommandBuffer> MetalContext::makeCommandBuffer() const {
  return [queue commandBuffer];
}

Command::Command(MetalContext &context)
    : context(context), commandBuffer(context.makeCommandBuffer()) {
  if (!commandBuffer)
    reject(Error("Cannot create Metal command buffer"));
}

void Command::reject(Error error) {
  if (!failure)
    failure = std::move(error);
}

Buffer Command::allocate(uint64_t elements, unsigned elementSize) {
  if (failure)
    return nil;
  if (elements > UINT32_MAX) {
    reject(Error("Reference tensor exceeds Metal's 32-bit index range"));
    return nil;
  }
  auto buffer = context.makeBuffer(elements * elementSize);
  if (!buffer) {
    reject(buffer.getError());
    return nil;
  }
  return *buffer;
}

void Command::encode(const char *name, const Buffer *buffers,
                     size_t bufferCount, const void *params, size_t paramsSize,
                     MTLSize groups, MTLSize threads) {
  if (failure)
    return;
  auto pipeline = context.getPipeline(name);
  if (!pipeline) {
    reject(pipeline.getError());
    return;
  }
  if (threads.width * threads.height >
          (*pipeline).maxTotalThreadsPerThreadgroup ||
      groups.width == 0 || groups.height == 0 ||
      std::find(buffers, buffers + bufferCount, nil) != buffers + bufferCount) {
    reject(Error("Invalid Metal dispatch for " + std::string(name)));
    return;
  }
  // Separate encoders let Metal track dependencies between dispatches.
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder) {
    reject(Error("Cannot create Metal compute encoder"));
    return;
  }
  encoder.label = [NSString stringWithUTF8String:name];
  [encoder setComputePipelineState:*pipeline];
  unsigned index = 0;
  for (; index < bufferCount; ++index)
    [encoder setBuffer:buffers[index] offset:0 atIndex:index];
  [encoder setBytes:params length:paramsSize atIndex:index];
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  [encoder endEncoding];
}

Error Command::finish(bool &submitted) {
  submitted = false;
  if (failure)
    return failure;
  [commandBuffer commit];
  submitted = true;
  [commandBuffer waitUntilCompleted];
  if (commandBuffer.status != MTLCommandBufferStatusCompleted)
    return metalError("Running Metal reference graph", commandBuffer.error);
  return {};
}
