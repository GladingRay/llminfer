# llminfer

面向 Qwen3.5-4B 的 Metal 推理引擎项目。当前已实现文本权重转换、`.llmi` 加载和检查工具。

## 构建与测试

需要 C++17 编译器、CMake 3.24 或更新版本以及 zlib。CMake 优先查找已安装的 nlohmann/json，否则会下载固定版本。运行测试需要 Python 3。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## 转换和检查模型

输入目录应包含 Hugging Face 的 `config.json`、`model.safetensors.index.json` 和索引引用的全部分片。

```sh
build/llmi convert /path/to/qwen3.5-4b build/qwen3.5-4b.llmi
build/llmi inspect build/qwen3.5-4b.llmi
build/llmi inspect build/qwen3.5-4b.llmi --tensor lm_head.weight --json
```

转换默认保存每 32 个元素一组的 INT8 投影权重和 FP16 scale。加 `--precision f16` 可生成对拍用的未量化版本。输出头和 embedding 共用一份权重。`inspect` 会验证文件结构和 CRC32，也可以报告指定张量的数据及 scale 区间。

## Qwen3.5 GraphIR

`llmi graph` 从已转换的 FP16 或 W8A8 模型构建完整文本主干。`--tokens` 指定本次输入长度，`--past` 指定已处理长度。默认 cache 容量为两者之和，可用 `--capacity` 指定。`--logits` 支持 `last`（默认）、`all` 和 `none`。

```sh
build/llmi graph build/qwen3.5-4b.llmi --tokens 16 --past 64 --capacity 4096
build/llmi graph build/qwen3.5-4b.llmi --tokens 1 --past 80 --json
```

摘要列出各类算子的节点数、计算量和逻辑读写量。浮点计算量以 GFLOPs 表示，INT8 以等效 GFLOPs 表示，两者都按 `2 × MAC / 10⁹` 换算。逻辑读写量以 GB 表示，`1 GB = 10⁹` 字节。`--json` 提供 `op_counts`、SSA 值、算子属性和状态读写信息，并在总成本和逐算子成本中同时给出原始 MAC、字节数及换算值。

## 执行 GraphIR

macOS 上用 `llmi run` 执行 FP16 或 W8A8 模型。每个逗号分隔的 token ID 列表是一段 chunk。所有 chunk 共用同一会话和缓存。

```sh
build/llmi run build/qwen3.5-4b.llmi 1,15392 42 --capacity 128 --plan
```

每个 chunk 输出一行 JSON，包含累计已处理 token 数 `past`、词表 logits、dispatch 数和临时缓冲区字节数。`--logits last` 是默认值，`all` 返回每个输入 token 的 logits，`none` 用于仅推进状态。`--plan` 还输出每个 dispatch 的 `kernel` 和 `op_index`，后者对应相同输入形状下 `llmi graph --json` 的 `ops` 下标。

## FP16 Metal 参考图

macOS 构建提供 `llmi-reference`。它从 `.llmi` 载入首个 DeltaNet 层和首个 GQA 层，以共享 embedding 和输出头连接两层。每个 token 参数代表一个 chunk，命令在同一个会话内依次执行，逐 chunk 输出包含中间结果和 cache 的 JSON 行。

```sh
build/llmi convert /path/to/qwen3.5-4b build/qwen3.5-4b-f16.llmi --precision f16
build/llmi-reference build/qwen3.5-4b-f16.llmi 1,15392 248319 42,7
```

缩小模型对拍通过 CTest 运行。需要可访问的 Metal GPU 以及 `tests/requirements-reference.txt` 中固定版本的 Python 依赖。用 `-DPython3_EXECUTABLE=/path/to/python` 指定环境。也可以直接运行同一个脚本，对拍真实模型的两层。

```sh
ctest --test-dir build -R metal_reference --output-on-failure
python tests/MetalReferenceTest.py build/llmi build/llmi-reference \
  --hf-model /path/to/qwen3.5-4b --fp16-model build/qwen3.5-4b-f16.llmi
```

格式和执行设计见 [架构文档](docs/architecture.md)。
