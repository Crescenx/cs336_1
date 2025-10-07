from __future__ import annotations

import argparse
import pickle  # 导入 pickle 模块
from pathlib import Path

from bpe_impl.bpe_train import train_bpe
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "owt_train.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "bpe_impl" / "artifacts"
DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"

# 以下辅助函数不再需要，因为 pickle 可以直接处理 bytes
# def encode_bytes(value: bytes) -> str:
# def serialize_vocab(vocab: dict[int, bytes]) -> dict[str, str]:
# def serialize_merges(merges: list[tuple[bytes, bytes]]) -> list[list[str]]:


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE on TinyStories and dump artifacts.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the TinyStories training corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the serialized tokenizer artifacts.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size, including special tokens.",
    )
    parser.add_argument(
        "--extra-special-token",
        action="append",
        dest="extra_special_tokens",
        default=[],
        help="Additional special tokens to include without splitting.",
    )
    args = parser.parse_args()

    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = [DEFAULT_SPECIAL_TOKEN]
    for token in args.extra_special_tokens:
        if token not in special_tokens:
            special_tokens.append(token)

    start_time = time.time()
    # 训练 BPE，获取词汇表和合并规则
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Training completed in {minutes}m {seconds:.2f}s.")

    # 定义输出文件路径，使用 .pkl 后缀
    vocab_path = output_dir / "owt_vocab.pkl"
    merges_path = output_dir / "owt_merges.pkl"

    # 使用 pickle 将 vocab 对象序列化到文件
    print(f"Saving vocabulary to {vocab_path}...")
    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved.")

    # 使用 pickle 将 merges 对象序列化到文件
    print(f"Saving merges to {merges_path}...")
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)
    print("Merges saved.")


if __name__ == "__main__":
    main()