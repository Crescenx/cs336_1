from collections import deque
import os
import pickle

from typing import Iterable, Iterator
import regex as re

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        token_merges = []
        for a, b in merges:
            merged_token = a + b
            if a in self.reverse_vocab and b in self.reverse_vocab and merged_token in self.reverse_vocab:
                token_merges.append((self.reverse_vocab[a], self.reverse_vocab[b], self.reverse_vocab[merged_token]))
        
        self.token_merges = token_merges
        self.merges_rank = { (a, b): i for i, (a, b, _) in enumerate(token_merges) }

        self._compile_patterns()
        
    def _compile_patterns(self):
        special_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self._special_alt = "|".join(re.escape(t) for t in special_sorted) if special_sorted else None

        if self._special_alt:
            pat = rf"({self._special_alt})|({GPT2_PAT})"
        else:
            pat = rf"({GPT2_PAT})"

        self._pretoken_re = re.compile(pat)

    @classmethod
    def from_files(cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_path, "rb") as f_vocab:
            vocab = pickle.load(f_vocab)
        with open(merges_path, "rb") as f_merges:
            merges = pickle.load(f_merges)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    class LinkedSeq:
        __slots__ = ("val", "left", "right", "alive", "_head")

        def __init__(self, tokens: Iterable[int]):
            vals = list(tokens)
            n = len(vals)
            self.val = vals
            self.left = [-1] + [i for i in range(n - 1)]
            self.right = [i+1 for i in range(n - 1)] + [-1]
            self.alive = [True] * n
            self._head = 0 if n > 0 else -1

        def is_valid(self, idx: int) -> bool:
            return 0 <= idx < len(self.val) and self.alive[idx]
        def get_head(self) -> int:
            return self._head
        def left_of(self, idx: int) -> int:
            return self.left[idx] if self.is_valid(idx) else -1
        def right_of(self, idx: int) -> int:
            return self.right[idx] if self.is_valid(idx) else -1
        def get(self, idx: int) -> int:
            return self.val[idx]
        def set(self, idx: int, value: int) -> None:
            self.val[idx] = value

        def remove(self, idx: int) -> tuple[int, int]:
            if not self.is_valid(idx):
                return -1, -1
            left_idx = self.left[idx]
            right_idx = self.right[idx]

            if left_idx != -1:
                self.right[left_idx] = right_idx
            if right_idx != -1:
                self.left[right_idx] = left_idx
            if idx == self._head:
                self._head = right_idx

            self.alive[idx] = False
            self.left[idx] = -1
            self.right[idx] = -1
            return left_idx, right_idx
    
        def pair_at(self, idx: int) -> tuple[int, int, int, int] | None:
            if not self.is_valid(idx):
                return None
            right_idx = self.right_of(idx)
            if right_idx == -1 or not self.is_valid(right_idx):
                return None
            return (idx, right_idx, self.get(idx), self.get(right_idx))
    
        def __iter__(self):
            idx = self.get_head()
            while idx != -1:
                yield idx, self.get(idx)
                idx = self.right_of(idx)
        
        def to_list(self) -> list[int]:
            return [val for _, val in self]
        
    def _get_block_from_match(self, m: re.Match) -> list[int] | None:
        if m.start() == m.end():
            return None

        if self._special_alt:
            special_str = m.group(1)
            piece_str = m.group(2)
        else:
            special_str = None
            piece_str = m.group(1)

        if special_str is not None:
            special_bytes = special_str.encode("utf-8")
            if special_bytes in self.reverse_vocab:
                return [self.reverse_vocab[special_bytes]]
            else:
                return None
        elif piece_str is not None:
            piece_bytes = piece_str.encode("utf-8")
            block = [
                self.reverse_vocab[bytes([b])]
                for b in piece_bytes
                if bytes([b]) in self.reverse_vocab
            ]
        return None
        
    def _pre_tokenize(self, text: str) -> list[list[int]]:
        blocks = []

        for m in self._pretoken_re.finditer(text):
            block = self._get_block_from_match(m)
            if block:
                blocks.append(block)
        return blocks
    

    def _merge_block(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) < 2 or not self.token_merges:
            return token_ids

        seq = self.LinkedSeq(token_ids)

        RMAX = len(self.token_merges) - 1
        buckets = [deque() for _ in range(RMAX + 1)]

        def try_push(i: int):
            if not seq.is_valid(i):
                return
            
            pair = seq.pair_at(i)
            if pair is None:
                return
            
            a_idx, b_idx, a_val, b_val = pair
            if not seq.is_valid(a_idx) or not seq.is_valid(b_idx):
                return
            
            rank = self.merges_rank.get((a_val, b_val))
            if rank is not None:
                buckets[rank].append(a_idx)

        # Initialize buckets
        for i, _ in seq:
            try_push(i)

        # Clean the buckets and perform merges
        for rank, q in enumerate(buckets):
            a_rule, b_rule, after = self.token_merges[rank]
            while q:
                i = q.popleft()

                pair = seq.pair_at(i)
                if pair is None:
                    continue
                a_idx, b_idx, a_val, b_val = pair
                if a_val != a_rule or b_val != b_rule:
                    continue
                
                seq.set(a_idx, after)  # (la,a,b,rb) -> (la,x,b,rb)
                x_idx, rb_idx = seq.remove(b_idx) # (la,x,b,rb) -> (la,x,rb)

                # Re-evaluate pairs around the merged token
                la_idx = seq.left_of(x_idx)
                if seq.is_valid(la_idx):
                    try_push(la_idx) # (la,x)
                try_push(x_idx)    # (x,rb)
            
        return seq.to_list()
    
    def encode(self, text: str) -> list[int]:
        pretokenized_blocks = self._pre_tokenize(text)
        output = []
        for block in pretokenized_blocks:
            merged = self._merge_block(block)
            output.extend(merged)

        return output
    
    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        tail = ""
        max_special_len = max((len(s) for s in self.special_tokens), default=0)
        window = max(1, max_special_len) 

        for chunk in text:
            if not chunk:
                continue

            buf = tail + chunk
            cutoff = max(0, len(buf) - window)
            last_processed_end = 0

            for m in self._pretoken_re.finditer(buf):
                if m.end() > cutoff:
                    # This match extends into the unsafe region. Stop processing here
                    # and keep the rest of the buffer for the next chunk.
                    tail = buf[last_processed_end:]
                    break

                if (block := self._get_block_from_match(m)) is not None:
                    yield from self._merge_block(block)
                
                last_processed_end = m.end()
            else:
                # All matches were safely processed. The new tail is what's left.
                tail = buf[last_processed_end:]
                
        # After the loop, process any remaining text in the tail.
        if tail:
            for m in self._pretoken_re.finditer(tail):
                if (block := self._get_block_from_match(m)) is not None:
                    yield from self._merge_block(block)

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for i in ids:
            bs = self.vocab.get(i)
            if bs is not None:
                buf.extend(bs)
        return buf.decode("utf-8", errors="ignore")
