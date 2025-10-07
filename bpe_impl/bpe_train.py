import os
import regex as re
from collections import Counter
import functools
from functools import lru_cache
from cs336_basics.pretokenization_example import find_chunk_boundaries

try:
    from accelerator_rust import train_bpe as rust_train_bpe
except ImportError:  # pragma: no cover - fallback to Python implementation
    rust_train_bpe = None

# 1) 预编译 GPT-2 模式，并用占有量词减少回溯（语义等价）
#    解释：X+ 变为 X++（占有量词），避免“回头看”导致的多余回溯；匹配集合不变。
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s+(?!\S)|\s+"""
GPT2_RE = re.compile(GPT2_PAT)

# 2) 缓存 special_tokens 的拆分正则，避免每块重复 escape + join + compile
@lru_cache(maxsize=64)
def _compile_special_re(tokens_tuple: tuple[str, ...] | None):
    if not tokens_tuple:
        return None
    # 仍保留“按长度降序”以维持你原有的匹配优先级与语义
    pats = [re.escape(t) for t in sorted(tokens_tuple, key=len, reverse=True)]
    return re.compile("|".join(pats))

def pre_tokenize_chunk(
    start_end: tuple[int, int],
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> dict[tuple[int, ...], int]:
    # Read the chunk
    start, end = start_end
    with open(input_path, "rb") as f_chunk:
        f_chunk.seek(start)
        chunk = f_chunk.read(end - start).decode("utf-8", errors="ignore")
    # For windows line endings
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    # 3) 复用/缓存的拆分模式；不改变“有就 split，没有就直接全串 find”的逻辑
    SPECIAL_RE = _compile_special_re(tuple(special_tokens) if special_tokens else None)

    word_counts = Counter()
    if SPECIAL_RE is not None:
        # 4) 用编译对象的方法（更快），并用 findall 直接拿字符串，少一次 m.group(0)
        for piece in SPECIAL_RE.split(chunk):
            if not piece:
                continue
            word_counts.update(GPT2_RE.findall(piece))
    else:
        word_counts.update(GPT2_RE.findall(chunk))

    # 保持你原始的“转 tuple[int] 作为 key”的逻辑不变
    sequences: dict[tuple[int, ...], int] = {}
    for k, v in word_counts.items():
        token_ids = tuple(b for b in k.encode("utf-8"))
        sequences[token_ids] = v
    return sequences


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Pre-tokenization
    task = functools.partial(
        pre_tokenize_chunk,
        input_path=input_path,
        special_tokens=special_tokens,
    )
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count() or 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunk_boundaries = list(zip(boundaries[:-1], boundaries[1:]))
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(task, chunk_boundaries))

    sequences = Counter()
    for result in results:
        sequences.update(result)

    num_merges = max(0, vocab_size - len(special_tokens) - 256)
    merges: list[tuple[bytes, bytes]]
    if rust_train_bpe is not None and num_merges > 0:
        rust_vocab, rust_merges = rust_train_bpe(sequences, int(num_merges))
        vocab = {int(token): bytes(value) for token, value in rust_vocab.items()}
        merges = [(bytes(left), bytes(right)) for left, right in rust_merges]
        current_available_index = max(vocab.keys(), default=-1) + 1
    else:
        raise NotImplementedError("Rust extension for BPE training is not available.")
    # Add special tokens to vocab
    for token in special_tokens:
        vocab[current_available_index] = token.encode('utf-8')
        current_available_index += 1

    return vocab, merges