import argparse
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{args[0]} exited with {result.returncode}: {result.stderr}")
    return result.stdout


def make_tiny_checkpoint(path):
    path.mkdir()
    text = {
        "model_type": "qwen3_5_text",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attn_output_gate": True,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "mlp_only_layers": [],
        "hidden_size": 33,
        "intermediate_size": 19,
        "vocab_size": 37,
        "num_hidden_layers": 2,
        "head_dim": 12,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 7,
        "linear_value_head_dim": 5,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 1],
            "rope_theta": 1e7,
        },
        "layer_types": ["linear_attention", "full_attention"],
    }
    config = Qwen3_5TextConfig(**text)
    generator = torch.Generator().manual_seed(1597)

    def random(shape, scale):
        return torch.randn(shape, generator=generator) * scale

    weights = {
        "embed_tokens.weight": random((text["vocab_size"], text["hidden_size"]), 0.25)
        .to(torch.bfloat16),
        "norm.weight": random((text["hidden_size"],), 0.1).to(torch.bfloat16),
    }
    for index in range(2):
        layer = Qwen3_5DecoderLayer(config, index)
        for name, tensor in layer.state_dict().items():
            scale = 0.12 if tensor.ndim > 1 else 0.1
            value = random(tensor.shape, scale)
            if name == "linear_attn.norm.weight":
                value = 1 + value
            elif name != "linear_attn.A_log":
                value = value.to(torch.bfloat16)
            weights[f"layers.{index}.{name}"] = value

    prefix = "model.language_model."
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for shard in range(2):
        name = f"model-{shard + 1:05d}-of-00002.safetensors"
        values = {prefix + key: weights[key] for key in list(weights)[shard::2]}
        save_file(values, path / name)
        index["weight_map"].update(dict.fromkeys(values, name))
        index["metadata"]["total_size"] += sum(t.nbytes for t in values.values())
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "tie_word_embeddings": True, "text_config": text
    }))


def quantized_linear(input, weight):
    data, weight_scales, width = weight
    rows = input.reshape(-1, width).float()
    groups = data.size(1)
    grouped = torch.nn.functional.pad(rows, (0, 32 * groups - width))
    grouped = grouped.reshape(rows.size(0), groups, 32)
    maximum = grouped.abs().amax(dim=-1)
    scales = torch.where(maximum > 0, maximum / 127, 1.0)
    quantized = torch.round(grouped / scales[..., None]).clamp(-127, 127).int()
    dots = (quantized[:, None] * data[None]).sum(dim=-1, dtype=torch.int32)
    result = (dots.float() * scales[:, None] * weight_scales[None]).sum(dim=-1)
    return result.half().reshape(*input.shape[:-1], data.size(0))


class TransformersReference:
    def __init__(self, path):
        text = json.loads((path / "config.json").read_text())["text_config"]
        self.config = Qwen3_5TextConfig(**text)
        self.config._attn_implementation = "eager"
        self.delta_index = text["layer_types"].index("linear_attention")
        self.gqa_index = text["layer_types"].index("full_attention")
        weight_map = json.loads(
            (path / "model.safetensors.index.json").read_text()
        )["weight_map"]

        def read_weight(key):
            key = "model.language_model." + key
            with safe_open(path / weight_map[key], framework="pt", device="cpu") as f:
                value = f.get_tensor(key)
            # Match the converter's precision policy for the FP16 payload.
            return value.to(torch.float16 if value.ndim > 1 else torch.float32)

        def load_layer(index):
            with torch.device("meta"):
                layer = Qwen3_5DecoderLayer(self.config, index)
            layer.load_state_dict(
                {
                    name: read_weight(f"layers.{index}.{name}")
                    for name in layer.state_dict()
                },
                assign=True,
            )
            return layer.eval()

        self.delta = load_layer(self.delta_index)
        self.gqa = load_layer(self.gqa_index)
        self.embedding = read_weight("embed_tokens.weight")
        self.norm = Qwen3_5RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.norm.load_state_dict({"weight": read_weight("norm.weight")})
        self.rope = Qwen3_5TextRotaryEmbedding(self.config)
        self.lm_head = lambda x: torch.nn.functional.linear(x, self.embedding)

    def use_w8_weights(self, cli, model):
        weights = {}
        metadata = json.loads(command([cli, "inspect", model, "--json"]))
        with model.open("rb") as file:
            for entry in metadata["tensors"]:
                if entry["dtype"] != "I8":
                    continue
                rows, width = entry["shape"]
                groups = entry["storage_shape"][1] // 32
                file.seek(entry["data"]["offset"])
                data = torch.frombuffer(
                    bytearray(file.read(entry["data"]["size"])), dtype=torch.int8
                ).int().reshape(rows, groups, 32)
                file.seek(entry["scales"]["offset"])
                scales = torch.frombuffer(
                    bytearray(file.read(entry["scales"]["size"])), dtype=torch.float16
                ).float().reshape(rows, groups)
                weights[entry["id"]] = (data, scales, width)

        data, scales, width = weights["embed_tokens.weight"]
        self.embedding = (data.float() * scales[..., None]).reshape(data.size(0), -1)
        self.embedding = self.embedding[:, :width].half()
        self.lm_head = partial(quantized_linear, weight=weights["embed_tokens.weight"])
        for index, layer in ((self.delta_index, self.delta), (self.gqa_index, self.gqa)):
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module.forward = partial(
                        quantized_linear, weight=weights[f"layers.{index}.{name}.weight"]
                    )

    def run(self, chunks):
        def flat(value):
            return value.reshape(-1).float().clone()

        def block(layer, x, **kwargs):
            trace = {}
            hooks = [
                module.register_forward_pre_hook(
                    lambda _, args, name=name: trace.__setitem__(name, flat(args[0]))
                )
                for module, name in (
                    (layer.post_attention_layernorm, "attention_residual"),
                    (layer.mlp.down_proj, "swiglu"),
                )
            ]
            try:
                result = layer(x, **kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
            trace["output"] = flat(result)
            return result, trace

        past = 0
        cache = DynamicCache(config=self.config)
        with torch.inference_mode():
            for tokens in chunks:
                end = past + len(tokens)
                positions = torch.arange(past, end)
                hidden = self.embedding[torch.tensor(tokens)][None]
                delta, delta_trace = block(
                    self.delta, hidden, position_embeddings=None, past_key_values=cache
                )
                visible = torch.arange(end)[None, :] <= positions[:, None]
                mask = torch.zeros(visible.shape, dtype=torch.float16)
                mask.masked_fill_(~visible, float("-inf"))
                gqa, gqa_trace = block(
                    self.gqa,
                    delta,
                    position_embeddings=self.rope(delta, positions[None, None]),
                    position_ids=positions[None],
                    attention_mask=mask[None, None],
                    past_key_values=cache,
                )
                conv = cache.layers[self.delta_index].conv_states[0]
                history = self.config.linear_conv_kernel_dim - 1
                state = cache.layers[self.delta_index].recurrent_states[0]
                keys = cache.layers[self.gqa_index].keys.transpose(1, 2)
                yield {
                    "embedding": flat(hidden),
                    "delta": delta_trace,
                    "gqa": gqa_trace,
                    "conv_history": flat(conv[..., conv.size(-1) - history:]),
                    "recurrent_state": flat(state),
                    "rotated_keys": flat(keys),
                    "logits": flat(self.lm_head(self.norm(gqa))),
                }
                past = end


def flatten(trace, prefix=""):
    for name, value in trace.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from flatten(value, key)
        else:
            yield key, value


def run_case(reference_cli, reference, model, chunks, real=False):
    args = [reference_cli, model, *(",".join(map(str, chunk)) for chunk in chunks)]
    outputs = [dict(flatten(json.loads(line))) for line in command(args).splitlines()]
    assert len(outputs) == len(chunks)
    traces = []
    errors = {}
    for step, (actual, expected) in enumerate(zip(outputs, reference.run(chunks))):
        expected = dict(flatten(expected))
        assert actual.keys() == expected.keys()
        trace = {}
        for name, values in actual.items():
            got = torch.tensor(values, dtype=torch.float32)
            want = expected[name]
            if name == "embedding":
                atol, rtol = 0, 0
            elif real:
                atol, rtol = (0.03, 0.01) if name == "logits" else (0.012, 0.01)
                if name == "recurrent_state":
                    atol, rtol = 0.001, 0.015
            else:
                atol, rtol = (0.0015, 0.003) if name == "logits" else (0.0006, 0.003)
                if name == "recurrent_state":
                    atol, rtol = 0.00005, 0.002
            torch.testing.assert_close(
                got, want, atol=atol, rtol=rtol,
                msg=lambda msg: f"Step {step}, {name}: {msg}",
            )
            errors[name] = max(errors.get(name, 0), (got - want).abs().max().item())
            trace[name] = got
        traces.append(trace)
    print(f"Chunks {[len(chunk) for chunk in chunks]}: "
          + ", ".join(f"{name}={error:.6g}" for name, error in errors.items()))
    return traces


def compare_partitions(full, chunked):
    states = {"conv_history", "recurrent_state", "rotated_keys"}
    for name in full:
        joined = chunked[-1][name] if name in states else torch.cat(
            [part[name] for part in chunked]
        )
        torch.testing.assert_close(full[name], joined, atol=1e-5, rtol=1e-5)


def run_graph(cli, model, chunks, mode, *, plan=False):
    args = [cli, model, "--logits", mode]
    if plan:
        args.append("--plan")
    args.extend(",".join(map(str, chunk)) for chunk in chunks)
    steps = [json.loads(line) for line in command(args).splitlines()]
    assert len(steps) == len(chunks)
    past = 0
    for step, chunk in zip(steps, chunks):
        past += len(chunk)
        assert step["past"] == past and step["scratch_bytes"] > 0
        assert step["dispatch_count"] > 0
        rows = {"none": 0, "last": 1, "all": len(chunk)}[mode]
        assert step["logit_rows"] == rows
        if not rows:
            assert not step["logits"]
    return steps


def test_graph_lowering(cli, run_cli, model, reference, tokens, quantized=False):
    chunks = [tokens[:2], tokens[2:3], tokens[3:]]
    steps = run_graph(run_cli, model, chunks, "all")
    pieces = []
    for step, expected in zip(steps, reference.run(chunks)):
        logits = torch.tensor(step["logits"])
        torch.testing.assert_close(
            logits, expected["logits"], atol=0.003, rtol=0.008
        )
        pieces.append(logits)
    joined = torch.cat(pieces)

    full = run_graph(run_cli, model, [tokens], "all", plan=True)[0]
    torch.testing.assert_close(torch.tensor(full["logits"]), joined, atol=1e-5, rtol=1e-5)
    lowered = full["dispatches"]
    assert len(lowered) == full["dispatch_count"]
    model_graph = json.loads(command([
        cli, "graph", model, "--tokens", str(len(tokens)), "--logits", "all", "--json"
    ]))
    indices = [dispatch["op_index"] for dispatch in lowered]
    assert indices == sorted(indices)
    assert set(indices) == {
        i for i, op in enumerate(model_graph["ops"]) if op["kind"] not in ("Reshape", "Slice")
    }
    assert any(dispatch["kernel"] == "copyHalfView" for dispatch in lowered)
    linear = "linearW8A8" if quantized else "linearF16"
    assert lowered[-1]["kernel"] == linear

    last = run_graph(run_cli, model, [tokens], "last")[0]
    torch.testing.assert_close(
        torch.tensor(last["logits"]), joined[-reference.config.vocab_size:],
        atol=1e-5, rtol=1e-5,
    )
    skipped = run_graph(run_cli, model, chunks, "none")[-1]
    assert skipped["dispatch_count"] < steps[-1]["dispatch_count"]


def test_text_generation(args, checkpoint, model):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer_path = model.with_name("tokenizer.json")
    vocab = {"[UNK]": 0, "你好": 1, "🌍": 2, "继续": 4, "完成": 7}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tokenizer_path))
    candidates = sorted(vocab.values())
    chunks = [[1, 2]]
    generated = []
    for _ in range(3):
        logits = run_graph(args.run_cli, model, chunks, "last")[-1]["logits"]
        token = max(candidates, key=logits.__getitem__)
        chunks.append([token])
        generated.append(token)

    options = ["--tokenizer", tokenizer_path, "--prompt", "你好 🌍", "--max-new-tokens", "3"]
    assert command([args.run_cli, model, *options]) == tokenizer.decode(generated) + "\n"

    config_path = checkpoint / "config.json"
    original = config_path.read_text()
    config = json.loads(original)
    config["text_config"]["eos_token_id"] = generated[0]
    config_path.write_text(json.dumps(config))
    eos_model = model.with_name("tiny-eos.llmi")
    command([args.converter, "convert", checkpoint, eos_model, "--precision", "f16"])
    config_path.write_text(original)
    assert command([args.run_cli, eos_model, *options]) == "\n"

    template_path = model.with_name("chat_template.jinja")
    tokenizer_config = model.with_name("tokenizer_config.json")
    template = (
        "{%- for message in messages -%}"
        "{{ message.content | trim }} "
        "{%- endfor %}"
        "{% if add_generation_prompt and enable_thinking is false %} 🌍{% endif %}"
    )
    template_path.write_text(template, encoding="utf-8")
    chat_prompt = " 你好 "
    chat_options = ["--tokenizer", tokenizer_path, "--prompt", chat_prompt,
                    "--max-new-tokens", "1", "--chat"]
    chat_logits = run_graph(args.run_cli, model, [[1, 2]], "last")[0]["logits"]
    chat_token = max(candidates, key=chat_logits.__getitem__)
    expected = tokenizer.decode([chat_token]) + "\n"
    assert command([args.run_cli, model, *chat_options]) == expected
    assert command([args.run_cli, model, *chat_options, "--thinking", "off"]) == expected

    thinking_logits = run_graph(args.run_cli, model, [[1]], "last")[0]["logits"]
    thinking_token = max(candidates, key=thinking_logits.__getitem__)
    assert command([args.run_cli, model, *chat_options, "--thinking", "on",
                    "--capacity", "1"]) == tokenizer.decode([thinking_token]) + "\n"

    tokenizer_config.write_text(json.dumps({"chat_template": template}), encoding="utf-8")
    template_path.unlink()
    assert command([args.run_cli, model, *chat_options]) == expected

    tokenizer_config.write_text(json.dumps({
        "chat_template": "{{ raise_exception('Wrong template') }}",
        "eos_token": tokenizer.id_to_token(chat_token),
    }), encoding="utf-8")
    override = model.with_name("override.jinja")
    override.write_text(template, encoding="utf-8")
    assert command([args.run_cli, model, *chat_options,
                    "--chat-template", override]) == "\n"


def test_model_benchmark(cli, model, prefill, decode):
    lines = command([cli, model, "--prefill-tokens", str(prefill),
                     "--decode-tokens", str(decode)]).splitlines()
    expected = [(phase, count) for phase, count in (("Prefill", prefill), ("Decode", decode))
                if count]
    assert len(lines) == len(expected)
    for line, (phase, count) in zip(lines, expected):
        match = re.fullmatch(rf"{phase}: {count} tokens in ([0-9.]+) s, ([0-9.]+) tok/s", line)
        assert match and float(match[1]) > 0 and float(match[2]) > 0, line


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("converter", type=Path)
    parser.add_argument("reference_cli", type=Path)
    parser.add_argument("--run-cli", type=Path)
    parser.add_argument("--benchmark-cli", type=Path)
    parser.add_argument("--allow-skip", action="store_true")
    parser.add_argument("--hf-model", type=Path)
    parser.add_argument("--fp16-model", type=Path)
    args = parser.parse_args()
    if bool(args.hf_model) != bool(args.fp16_model):
        parser.error("--hf-model and --fp16-model must be provided together")
    if args.run_cli is None:
        args.run_cli = args.converter.with_name("llmi-run")
    return args


def check_prerequisites(reference_cli):
    for line in Path(__file__).with_name("requirements-reference.txt").read_text().splitlines():
        name, required = line.split("==")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != required:
            return f"Reference dependency {name}=={required} required; found {actual}"

    probe = subprocess.run([reference_cli, "--probe"], check=False)
    if probe.returncode == 77:
        return "No Metal device available"
    probe.check_returncode()
    return None


def main(args):
    torch.set_num_threads(4)
    if args.hf_model:
        reference = TransformersReference(args.hf_model)
        run_case(args.reference_cli, reference, args.fp16_model,
                 [[1, 15392], [248319], [42, 7]], real=True)
    else:
        with tempfile.TemporaryDirectory(prefix="llmi-metal-test-") as directory:
            checkpoint = Path(directory) / "checkpoint"
            model = Path(directory) / "tiny.llmi"
            make_tiny_checkpoint(checkpoint)
            command([args.converter, "convert", checkpoint, model, "--precision", "f16"])
            reference = TransformersReference(checkpoint)
            tokens = [(i * 13) % 37 for i in range(19)]
            full = run_case(args.reference_cli, reference, model, [tokens])[0]
            chunked = run_case(args.reference_cli, reference, model,
                               [tokens[:2], tokens[2:3], tokens[3:8], tokens[8:]])
            compare_partitions(full, chunked)
            run_case(args.reference_cli, reference, model, [[36], [1]])
            test_graph_lowering(args.converter, args.run_cli, model, reference, tokens)
            test_text_generation(args, checkpoint, model)

            w8_model = Path(directory) / "tiny-w8.llmi"
            command([args.converter, "convert", checkpoint, w8_model])
            reference.use_w8_weights(args.converter, w8_model)
            test_graph_lowering(
                args.converter, args.run_cli, w8_model, reference, tokens, quantized=True
            )
            if args.benchmark_cli:
                test_model_benchmark(args.benchmark_cli, model, 3, 2)
                test_model_benchmark(args.benchmark_cli, w8_model, 4, 3)
                test_model_benchmark(args.benchmark_cli, w8_model, 2, 0)
                test_model_benchmark(args.benchmark_cli, w8_model, 0, 1)


if __name__ == "__main__":
    args = parse_args()
    skip = check_prerequisites(args.reference_cli)
    if skip:
        print(skip, file=sys.stderr)
        sys.exit(77 if args.allow_skip else 1)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
    )

    main(args)
