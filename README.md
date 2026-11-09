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

格式和执行设计见 [架构文档](docs/architecture.md)。
