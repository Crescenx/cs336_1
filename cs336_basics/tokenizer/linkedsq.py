from typing import Iterable


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