# Repository Guidelines

## Project Structure

`llminfer` is an inference engine project targeting Qwen3.5 4B. The root `README.md` describes the project, `LICENSE` contains the MIT license, and `.gitignore` covers generated build files and caches. Source code, tests, and model assets have yet to be added. When introducing them, document their locations and any required model files in the README.

## Build, Test, and Development Commands

Run `git diff --check` to catch whitespace errors in unstaged changes or `git diff --cached --check` for staged changes. Build, test, and local run commands will be available once the implementation and build configuration land. Document those commands and their prerequisites in the same change that introduces them. Keep generated output in an ignored directory such as `build/`.

## Coding Style and Naming

Match the surrounding code when editing an existing file. For the first source files in a language, document the indentation and naming conventions and commit any formatter or linter configuration used to enforce them. Use descriptive names for modules, functions, and files.

## Testing Guidelines

The first test suite should document its framework, test file naming, and how to run it. For inference changes, cover the affected behavior with small, deterministic cases. If validation requires model weights or specific hardware, record the prerequisites and the steps to reproduce the check.

## Commits and Pull Requests

The sole commit is `Initial commit`. Use short, imperative subjects for new commits, such as `Add tokenizer tests`. In pull requests, summarize what changed and why, list the validation performed, and link related issues when available.
## 目标
目标：实现一个以metal为后端运行的 qwen3.5 4B 大模型的推理引擎。
## instructions
1. 在实现过程中如果功能点或者bugfix完成可以自动commit。
2. commit message与代码中的注释需要使用英文。
3. 文档使用中文书写。
4. 代码风格、命名规范与注释格式与MLIR保持一致。
5. 项目主体使用C++实现，可辅以部分python与shell脚本。
6. 不同的模块要放在不同的文件中，不同的大组件要放到不同的目录中。
7. qwen3.5 4B 的路径在 `/Users/guorui/Work/models/qwen3.5-4b` 。
8. python环境：`/Users/guorui/Work/models/venv314`。