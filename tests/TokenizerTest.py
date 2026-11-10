import json
from pathlib import Path
import subprocess
import sys
import tempfile


def write_tokenizer(path, vocab):
    path.write_text(json.dumps({
        "version": "1.0",
        "added_tokens": [],
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }), encoding="utf-8")


def test_byte_level_stream(path, stream_test):
    direct = {*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)}
    extra = iter(range(0x100, 0x200))
    alphabet = [chr(byte if byte in direct else next(extra)) for byte in range(256)]
    tokens = [b"A", b" \xe4", b"\xbd", b"\xa0\xf0", b"\x9f\x8c", b"\x8d!",
              b"\xef\xbf\xbd", b"\xe4\xbd", b"Q", b"\xe0\x80", b"\xc3", b"\xa9"]
    vocab = {"".join(alphabet[byte] for byte in token): id for id, token in enumerate(tokens)}
    vocab["é🌍"] = len(vocab)
    path.write_text(json.dumps({
        "version": "1.0",
        "added_tokens": [],
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }), encoding="utf-8")
    subprocess.run([stream_test, path], check=True)


def main(cli, stream_test):
    with tempfile.TemporaryDirectory(prefix="llmi-tokenizer-test-") as directory:
        tokenizer = Path(directory) / "tokenizer.json"
        write_tokenizer(tokenizer, {"[UNK]": 0, "你好": 2, "🌍": 5})

        def run(mode, value):
            return subprocess.run([cli, mode, tokenizer, value], capture_output=True,
                                  encoding="utf-8", check=False)

        encoded = run("encode", "你好 🌍")
        assert encoded.returncode == 0, encoded.stderr
        assert json.loads(encoded.stdout) == [2, 5]

        decoded = run("decode", "2,5")
        assert decoded.returncode == 0, decoded.stderr
        assert decoded.stdout == "你好 🌍\n"

        invalid_id = run("decode", "3")
        assert invalid_id.returncode == 1 and "outside" in invalid_id.stderr

        tokenizer.write_text("{", encoding="utf-8")
        invalid_json = run("encode", "你好")
        assert invalid_json.returncode == 1 and "JSON" in invalid_json.stderr

        test_byte_level_stream(tokenizer, stream_test)


if __name__ == "__main__":
    main(*sys.argv[1:])
