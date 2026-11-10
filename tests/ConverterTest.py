from collections import Counter
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


CLI = Path(sys.argv[1])
PREFIX = "model.language_model."


def run(*args, error=None):
    result = subprocess.run(
        [CLI, *args], text=True, capture_output=True, check=False
    )
    if error is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 1 and error in result.stderr, result
    return result.stdout


def write_checkpoint(directory, *, invalid_weight=False):
    config = {
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_5_text",
            "tie_word_embeddings": True,
            "attention_bias": False,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "mlp_only_layers": [],
            "hidden_size": 33,
            "intermediate_size": 9,
            "vocab_size": 5,
            "num_hidden_layers": 2,
            "head_dim": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
                "rope_theta": 1e7,
            },
            "layer_types": ["linear_attention", "full_attention"],
        },
    }
    (directory / "config.json").write_text(json.dumps(config))

    embedding = [0.0] * (5 * 33)
    embedding[:5] = [127.0, -127.0, 2.5, -1.5, 0.5]
    embedding[32] = -1.0
    embedding[4 * 33] = 2.0
    if invalid_weight:
        embedding[-1] = float("nan")
    entries = [(PREFIX + "embed_tokens.weight", [5, 33], "BF16", embedding)]
    entries.append((PREFIX + "norm.weight", [33], "BF16", [0.5] * 33))

    def add(layer, name, shape, dtype="BF16", value=0.0):
        entries.append(
            (f"{PREFIX}layers.{layer}.{name}", shape, dtype, [value] * math.prod(shape))
        )

    for layer in range(2):
        add(layer, "input_layernorm.weight", [33])
        add(layer, "post_attention_layernorm.weight", [33])
        add(layer, "mlp.gate_proj.weight", [9, 33])
        add(layer, "mlp.up_proj.weight", [9, 33])
        add(layer, "mlp.down_proj.weight", [33, 9])
    for name, shape in (
        ("in_proj_qkv.weight", [16, 33]),
        ("in_proj_z.weight", [8, 33]),
        ("in_proj_a.weight", [2, 33]),
        ("in_proj_b.weight", [2, 33]),
        ("out_proj.weight", [33, 8]),
        ("conv1d.weight", [16, 1, 4]),
        ("dt_bias", [2]),
    ):
        add(0, f"linear_attn.{name}", shape)
    add(0, "linear_attn.A_log", [2], "F32", -0.125)
    add(0, "linear_attn.norm.weight", [4], "F32", 1.0)
    for name, shape in (
        ("q_proj.weight", [32, 33]),
        ("k_proj.weight", [8, 33]),
        ("v_proj.weight", [8, 33]),
        ("o_proj.weight", [33, 16]),
        ("q_norm.weight", [8]),
        ("k_norm.weight", [8]),
    ):
        add(1, f"self_attn.{name}", shape)
    entries += [
        ("model.visual.pos_embed.weight", [1], "BF16", [0.0]),
        ("mtp.fc.weight", [1], "BF16", [0.0]),
    ]

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for i in range(2):
        shard = f"model-{i + 1:05d}-of-00002.safetensors"
        header = {"__metadata__": {"format": "pt"}}
        payload = bytearray()
        for name, shape, dtype, values in entries[i::2]:
            start = len(payload)
            for value in values:
                encoded = struct.pack("<f", value)
                payload.extend(encoded[2:] if dtype == "BF16" else encoded)
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, len(payload)],
            }
            index["weight_map"][name] = shard
        index["metadata"]["total_size"] += len(payload)
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        (directory / shard).write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + payload
        )
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


def test_graph(w8_model, dense_model):
    args = ("graph", w8_model, "--tokens", "3", "--past", "2", "--capacity", "8")
    graph = json.loads(run(*args, "--json"))
    assert graph["step"] == {"tokens": 3, "past": 2, "capacity": 8}
    assert len(graph["states"]) == 4
    assert graph["values"][graph["outputs"][0]]["shape"] == [1, 5]
    ops = graph["ops"]
    counts = Counter(op["kind"] for op in ops)
    assert graph["op_counts"] == counts
    summary = run(*args).splitlines()
    for label, field, raw_field, multiplier in (
        ("INT8 compute (equivalent GFLOPs)", "int8_equivalent_gflops", "int8_macs", 2),
        ("Floating-point compute (GFLOPs)", "fp_gflops", "fp_macs", 2),
        ("Logical read (GB)", "logical_read_gb", "logical_read_bytes", 1),
        ("Logical write (GB)", "logical_write_gb", "logical_write_bytes", 1),
    ):
        cost = graph["cost"]
        assert math.isclose(cost[field], multiplier * cost[raw_field] / 1e9)
        assert f"{label}: {cost[field]:.9f}" in summary
    assert summary[summary.index("Operation counts:") + 1:] == [
        f"  {name}: {count}" for name, count in sorted(counts.items())
    ]
    assert {op["layer"] for op in ops if "layer" in op} == {0, 1}
    embed = ops[0]
    head = ops[-1]
    assert embed["kind"] == "EmbeddingGather" and head["kind"] == "LinearW8A8"
    assert embed["inputs"][1] == head["inputs"][2]

    delta_quant = next(op for op in ops if op["kind"] == "QuantizeActivation")
    projections = [
        op for op in ops if op["kind"] == "LinearW8A8" and
        op["inputs"][:2] == delta_quant["outputs"]
    ]
    assert len(projections) == 4
    assert delta_quant["cost"]["logical_write_bytes"] == 123

    kv = next(op for op in ops if op["kind"] == "KvAppend")
    gqa = next(op for op in ops if op["kind"] == "CausalGqa")
    delta = next(op for op in ops if op["kind"] == "DeltaUpdate")
    assert kv["cost"]["logical_write_bytes"] == 96
    assert gqa["cost"]["logical_read_bytes"] == 256
    assert math.isclose(kv["cost"]["logical_write_gb"], 96 / 1e9)
    assert math.isclose(gqa["cost"]["logical_read_gb"], 256 / 1e9)
    assert gqa["cost"]["fp_macs"] == 384
    assert math.isclose(gqa["cost"]["fp_gflops"], 768 / 1e9)
    assert delta["cost"]["fp_macs"] == 288
    assert all(
        e["begin"] == 2 and e["length"] == 3 and e["version_after"] == 1
        for e in kv["effects"]
    )
    assert all(
        e["begin"] == 0 and e["length"] == 5 and e["version_before"] == 1
        for e in gqa["effects"]
    )

    dense = json.loads(run("graph", dense_model, "--tokens", "3", "--logits", "all", "--json"))
    assert dense["values"][dense["outputs"][0]]["shape"] == [3, 5]
    assert dense["cost"]["int8_macs"] == 0 and dense["cost"]["fp_macs"] > 0
    assert dense["op_counts"] == Counter(op["kind"] for op in dense["ops"])
    assert "QuantizeActivation" not in dense["op_counts"]

    no_logits = json.loads(run("graph", w8_model, "--logits", "none", "--json"))
    assert not no_logits["outputs"]
    assert len(no_logits["states"]) == 4
    run("graph", w8_model, "--tokens", "0", error="Invalid graph step")
    run("graph", w8_model, "--capacity", "129", error="exceeds the model context")


def test_conversion(directory):
    write_checkpoint(directory)
    first = directory / "first.llmi"
    second = directory / "second.llmi"
    run("convert", directory, first)
    run("convert", directory, second)
    data = first.read_bytes()
    assert data == second.read_bytes()

    def inspect_tensor(model, tensor):
        return json.loads(run("inspect", model, "--tensor", tensor, "--json"))

    summary = inspect_tensor(first, "embed_tokens.weight")
    tensor = summary["tensor"]
    assert tensor == inspect_tensor(first, "lm_head.weight")["tensor"]
    assert tensor["shape"] == [5, 33] and tensor["storage_shape"] == [5, 64]
    assert tensor["layout"] == "RowMajorI8Group32" and tensor["dtype"] == "I8"
    assert summary["tensor_count"] == 27
    assert summary["manifest"]["weight_group_size"] == 32
    assert summary["manifest"]["skipped_tensors"] == [
        "model.visual.pos_embed.weight",
        "mtp.fc.weight",
    ]

    offset = tensor["data"]["offset"]
    quantized = struct.unpack_from("<320b", data, offset)
    assert quantized[:5] == (127, -127, 2, -2, 0)
    assert quantized[32] == -127 and not any(quantized[33:256])
    assert quantized[256] == 127 and not any(quantized[257:])
    scale = struct.unpack_from("<10e", data, tensor["scales"]["offset"])
    assert scale[0] == 1 and scale[1] == struct.unpack("<e", struct.pack("<e", 1 / 127))[0]
    assert scale[2:8] == (1,) * 6 and scale[-1] == 1
    run("inspect", first, "--tensor", "missing", error="Unknown tensor")
    run("inspect", directory / "missing.llmi", error="Opening")

    norm = inspect_tensor(first, "layers.0.linear_attn.A_log")["tensor"]
    assert norm["dtype"] == "F32"
    assert struct.unpack_from("<2f", data, norm["data"]["offset"]) == (-0.125, -0.125)
    conv = inspect_tensor(first, "layers.0.linear_attn.conv1d.weight")["tensor"]
    assert conv["dtype"] == "F16"

    dense = directory / "dense.llmi"
    run("convert", directory, dense, "--precision", "f16")
    dense_tensor = inspect_tensor(dense, "lm_head.weight")["tensor"]
    assert dense_tensor["dtype"] == "F16" and dense_tensor["shape"] == [5, 33]
    assert dense_tensor["scales"]["size"] == 0
    assert struct.unpack_from("<5e", dense.read_bytes(), dense_tensor["data"]["offset"]) == (
        127.0, -127.0, 2.5, -1.5, 0.5
    )

    test_graph(first, dense)

    bad = directory / "bad.llmi"
    bad.write_bytes(data[:-1])
    run("inspect", bad, error="file length")
    bad.write_bytes(b"BAD!" + data[4:])
    run("inspect", bad, error="magic")

    directory_offset = struct.unpack_from("<Q", data, 32)[0]
    changed = bytearray(data)
    struct.pack_into("<Q", changed, directory_offset + 160, 0)
    bad.write_bytes(changed)
    run("inspect", bad, error="directory entry")
    changed = bytearray(data)
    changed[offset] ^= 1
    bad.write_bytes(changed)
    run("inspect", bad, error="checksum")
    manifest_offset = struct.unpack_from("<Q", data, 16)[0]
    changed = bytearray(data)
    changed[manifest_offset] = ord("!")
    bad.write_bytes(changed)
    run("inspect", bad, error="Invalid JSON metadata")

    config_path = directory / "config.json"
    valid_config = config_path.read_text()
    config = json.loads(valid_config)
    config["text_config"]["hidden_size"] = "33"
    config_path.write_text(json.dumps(config))
    run("convert", directory, first, error="Invalid model dimension: hidden_size")
    config_path.write_text('{"model_type":"qwen3_5",' + valid_config[1:])
    run("convert", directory, first, error="Duplicate JSON key: model_type")
    config_path.write_text("[" * 65 + "]" * 65)
    run("convert", directory, first, error="JSON nesting exceeds limit")
    assert first.read_bytes() == data
    config_path.write_text(valid_config)

    target_directory = directory / "output.llmi"
    target_directory.mkdir()
    run("convert", directory, target_directory, error="Publishing model")
    assert target_directory.is_dir()
    assert not list(directory.glob("*.tmp.*"))

    write_checkpoint(directory, invalid_weight=True)
    run("convert", directory, first, error="Non-finite weight")
    assert first.read_bytes() == data
    assert not list(directory.glob("*.tmp.*"))

    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"][PREFIX + "embed_tokens.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))
    run("convert", directory, first, error="shard filename")
    assert first.read_bytes() == data


with tempfile.TemporaryDirectory(prefix="llmi-test-") as path:
    test_conversion(Path(path))
