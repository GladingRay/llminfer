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
