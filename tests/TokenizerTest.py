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


def main(cli):
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


if __name__ == "__main__":
    main(sys.argv[1])
