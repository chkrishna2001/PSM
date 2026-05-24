from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3

SPECIAL_TOKENS = {
    "<pad>": PAD_ID,
    "<unk>": UNK_ID,
    "<cls>": CLS_ID,
    "<sep>": SEP_ID,
}


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)


@dataclass
class HashTokenizer:
    vocab_size: int
    max_length: int

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        tokens = ["<cls>", *tokenize(text), "<sep>"]
        ids = [token_to_id(token, self.vocab_size) for token in tokens[: self.max_length]]
        attention = [1] * len(ids)
        if len(ids) < self.max_length:
            pad = self.max_length - len(ids)
            ids.extend([PAD_ID] * pad)
            attention.extend([0] * pad)
        return ids, attention

    def to_json(self) -> dict[str, int]:
        return {"type": "hash", "vocab_size": self.vocab_size, "max_length": self.max_length}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def token_to_id(token: str, vocab_size: int) -> int:
    if token in SPECIAL_TOKENS:
        return SPECIAL_TOKENS[token]
    if vocab_size <= len(SPECIAL_TOKENS):
        return UNK_ID
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return len(SPECIAL_TOKENS) + (value % (vocab_size - len(SPECIAL_TOKENS)))

