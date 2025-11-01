from collections import deque
import os
import pickle
import ahocorasick
import heapq

from typing import Iterable, Iterator
import regex as re

from cs336_basics.tokenizer.linkedsq import LinkedSeq

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

        # Precompile the regex for pre-tokenization
        self.gpt2_re = re.compile(GPT2_PAT)

        # Create the Aho-Corasick automaton for special tokens
        if self.special_tokens:
            self._spec_automaton = ahocorasick.Automaton()
            for token in self.special_tokens:
                self._spec_automaton.add_word(token, token)
            self._spec_automaton.make_automaton()
        
        special_token_ids = [self.reverse_vocab[token.encode("utf-8")] for token in self.special_tokens if token.encode("utf-8") in self.reverse_vocab]
        self.special_token_ids = set(special_token_ids)

    @classmethod
    def from_files(cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_path, "rb") as f_vocab:
            vocab = pickle.load(f_vocab)
        with open(merges_path, "rb") as f_merges:
            merges = pickle.load(f_merges)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    

        
    def split(self, text: str) -> Iterator[tuple[str, bool]]:
        last_idx = 0

        if self.special_tokens:
            for end_idx, spec_token in self._spec_automaton.iter_long(text):
                start_idx = end_idx - len(spec_token) + 1
                if start_idx > last_idx:
                    normal_piece = text[last_idx:start_idx]
                    for m in self.gpt2_re.finditer(normal_piece):
                        yield m.group(0), False

                yield spec_token, True
                last_idx = end_idx + 1
        
        if last_idx < len(text):
            normal_piece = text[last_idx:]
            for m in self.gpt2_re.findall(normal_piece):
                yield m, False

    def _get_piece(self, piece: str, is_special: bool) -> list[int]:
        if is_special:
            token_id = self.reverse_vocab.get(piece.encode("utf-8"))
            if token_id is not None:
                return [token_id]
            else:
                return []
        else:
            piece_bytes = piece.encode("utf-8")
            token_ids = []
            for byte in piece_bytes:
                token_id = self.reverse_vocab.get(bytes([byte]))
                if token_id is not None:
                    token_ids.append(token_id)
            return token_ids

    def _merge_piece(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) < 2 or not self.token_merges:
            return token_ids

        seq = LinkedSeq(token_ids)

        queues: dict[int, deque[int]] = {}
        active_ranks: list[int] = []
        in_heap: set[int] = set()

        def get_queue(rank: int) -> deque[int]:
            q = queues.get(rank)
            if q is None:
                q = deque()
                queues[rank] = q
            return q
        
        def activate_rank(rank: int):
            if rank not in in_heap:
                in_heap.add(rank)
                heapq.heappush(active_ranks, rank)

        def try_push(i: int, curr_rank: int | None = None):
            if not seq.is_valid(i):
                return
            
            pair = seq.pair_at(i)
            if pair is None:
                return
            
            a_idx, b_idx, a_val, b_val = pair
            if not seq.is_valid(a_idx) or not seq.is_valid(b_idx):
                return
            
            rank = self.merges_rank.get((a_val, b_val))
            if rank is None:
                return
            
            if curr_rank is not None and rank == curr_rank:
                queues[rank].append(i) # Re-add to the same queue for immediate processing
            else:
                get_queue(rank).append(i)
                activate_rank(rank)

        # Initialize buckets
        for i, _ in seq:
            try_push(i)

        # Clean the buckets and perform merges
        while active_ranks:
            curr_rank = heapq.heappop(active_ranks)
            in_heap.discard(curr_rank)
            q = get_queue(curr_rank)
            if not q:
                continue

            a_rule, b_rule, after = self.token_merges[curr_rank]
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
                    try_push(la_idx, curr_rank) # (la,x)
                try_push(x_idx, curr_rank)    # (x,rb)
            
            del queues[curr_rank]
        return seq.to_list()
    
    def _tid_generator_from_splits(self, splt:tuple[list[int],bool]) -> Iterator[int]:
        piece, is_special = splt
        token_ids = self._get_piece(piece, is_special)
        merged_ids = self._merge_piece(token_ids)
        for tid in merged_ids:
            yield tid
    
    def encode(self, text: str) -> list[int]:
        all_ids = []
        for splt in self.split(text):
            for tid in self._tid_generator_from_splits(splt):
                all_ids.append(tid)
        return all_ids
        
        
    
    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        buf = ""
        for chunk in text:
            buf += chunk
            
            splts = list(self.split(buf))
            if len(splts) < 2:
                continue

            safe_splits = splts[:-1]
            for splt in safe_splits:
                yield from self._tid_generator_from_splits(splt)

            buf = splts[-1][0] if splts else ""
        if buf:
            splts = list(self.split(buf))
            for splt in splts:
                yield from self._tid_generator_from_splits(splt)



    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for i in ids:
            bs = self.vocab.get(i)
            if bs is not None:
                buf.extend(bs)
        return buf.decode("utf-8", errors="ignore")
    
