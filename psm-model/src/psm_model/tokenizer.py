from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|end|>"


@dataclass(frozen=True)
class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer for pipeline bring-up.

    This is not the final production tokenizer. It exists so the generative
    training and inference code can move before BPE/SentencePiece is selected.
    """

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.byte_offset + byte for byte in text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_pieces(self, pieces: list[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in pieces:
            ids.extend(self.encode(piece))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        output = bytearray()
        for token_id in ids:
            if token_id in {self.pad_id, self.bos_id, self.eos_id}:
                if skip_special:
                    continue
                output.extend(_special_token_text(token_id, self).encode("utf-8"))
                continue
            if token_id < self.byte_offset or token_id >= self.vocab_size:
                raise ValueError(f"token id out of range: {token_id}")
            output.append(token_id - self.byte_offset)
        return output.decode("utf-8")

    def save(self, path: Path) -> None:
        payload = {
            "type": "byte",
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "byte_offset": self.byte_offset,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "byte":
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')}")
        return cls(
            pad_id=int(payload["pad_id"]),
            bos_id=int(payload["bos_id"]),
            eos_id=int(payload["eos_id"]),
            byte_offset=int(payload["byte_offset"]),
        )


@dataclass(frozen=True)
class BpeTokenizer:
    merges: tuple[tuple[int, int], ...]
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.byte_offset + byte for byte in text.encode("utf-8")]
        next_id = self.byte_offset + 256
        for merge in self.merges:
            ids = _apply_merge(ids, merge, next_id)
            next_id += 1
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_pieces(self, pieces: list[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in pieces:
            ids.extend(self.encode(piece))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        id_to_bytes = self._id_to_bytes()
        output = bytearray()
        for token_id in ids:
            if token_id in {self.pad_id, self.bos_id, self.eos_id}:
                if skip_special:
                    continue
                output.extend(_special_token_text(token_id, self).encode("utf-8"))
                continue
            try:
                output.extend(id_to_bytes[token_id])
            except KeyError as exc:
                raise ValueError(f"token id out of range: {token_id}") from exc
        return output.decode("utf-8")

    def save(self, path: Path) -> None:
        payload = {
            "type": "byte_bpe",
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "byte_offset": self.byte_offset,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BpeTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "byte_bpe":
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')}")
        return cls(
            merges=tuple((int(pair[0]), int(pair[1])) for pair in payload["merges"]),
            pad_id=int(payload["pad_id"]),
            bos_id=int(payload["bos_id"]),
            eos_id=int(payload["eos_id"]),
            byte_offset=int(payload["byte_offset"]),
        )

    def _id_to_bytes(self) -> dict[int, bytes]:
        id_to_bytes = {self.byte_offset + byte: bytes([byte]) for byte in range(256)}
        next_id = self.byte_offset + 256
        for left, right in self.merges:
            id_to_bytes[next_id] = id_to_bytes[left] + id_to_bytes[right]
            next_id += 1
        return id_to_bytes


def train_bpe_tokenizer(texts: list[str], *, vocab_size: int = 512, min_pair_count: int = 2) -> BpeTokenizer:
    if vocab_size < ByteTokenizer().vocab_size:
        raise ValueError(f"vocab_size must be at least {ByteTokenizer().vocab_size}")
    sequences = [[ByteTokenizer().byte_offset + byte for byte in text.encode("utf-8")] for text in texts if text]
    merges: list[tuple[int, int]] = []
    next_id = ByteTokenizer().byte_offset + 256
    target_merges = vocab_size - next_id

    for _ in range(target_merges):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for sequence in sequences:
            pair_counts.update(zip(sequence, sequence[1:]))
        if not pair_counts:
            break
        pair, count = pair_counts.most_common(1)[0]
        if count < min_pair_count:
            break
        sequences = [_apply_merge(sequence, pair, next_id) for sequence in sequences]
        merges.append(pair)
        next_id += 1

    return BpeTokenizer(merges=tuple(merges))


def load_tokenizer(path: Path) -> ByteTokenizer | BpeTokenizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "byte":
        return ByteTokenizer.load(path)
    if payload.get("type") == "byte_bpe":
        return BpeTokenizer.load(path)
    if payload.get("type") == "pattern":
        return PatternTokenizer.load(path)
    if payload.get("type") == "dsl":
        return DslTokenizer.load(path)
    raise ValueError(f"unsupported tokenizer type: {payload.get('type')}")


DEFAULT_DSL_PIECES = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "A:",
    "M:-",
    "T:",
    "C:",
    "Q:",
    "G:",
    "TE:",
    "RT:",
    "F:",
    "R:",
    "END",
    "ignore",
    "store_episodic",
    "promote_semantic",
    "update_existing",
    "flag_conflict",
    "flag_and_store",
    "episodic",
    "semantic",
    "explicit",
)


@dataclass(frozen=True)
class DslTokenizer:
    pieces: tuple[str, ...] = DEFAULT_DSL_PIECES
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3

    @property
    def piece_offset(self) -> int:
        return self.byte_offset + 256

    @property
    def vocab_size(self) -> int:
        return self.piece_offset + len(self.pieces)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        piece_to_id = {piece: self.piece_offset + index for index, piece in enumerate(self.pieces)}
        sorted_pieces = sorted(self.pieces, key=len, reverse=True)
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        index = 0
        byte_buffer: list[str] = []

        def flush_bytes() -> None:
            if byte_buffer:
                chunk = "".join(byte_buffer)
                ids.extend(self.byte_offset + byte for byte in chunk.encode("utf-8"))
                byte_buffer.clear()

        while index < len(text):
            matched = None
            for piece in sorted_pieces:
                if text.startswith(piece, index):
                    matched = piece
                    break
            if matched is None:
                byte_buffer.append(text[index])
                index += 1
            else:
                flush_bytes()
                ids.append(piece_to_id[matched])
                index += len(matched)
        flush_bytes()
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_pieces(self, pieces: list[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in pieces:
            ids.extend(self.encode(piece))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id in {self.pad_id, self.bos_id, self.eos_id}:
                flush_bytes()
                if not skip_special:
                    output.append(_special_token_text(token_id, self))
                continue
            if self.byte_offset <= token_id < self.piece_offset:
                byte_buffer.append(token_id - self.byte_offset)
                continue
            piece_index = token_id - self.piece_offset
            if 0 <= piece_index < len(self.pieces):
                flush_bytes()
                output.append(self.pieces[piece_index])
                continue
            raise ValueError(f"token id out of range: {token_id}")
        flush_bytes()
        return "".join(output)

    def save(self, path: Path) -> None:
        payload = {
            "type": "dsl",
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "byte_offset": self.byte_offset,
            "pieces": list(self.pieces),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DslTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "dsl":
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')}")
        return cls(
            pieces=tuple(str(piece) for piece in payload["pieces"]),
            pad_id=int(payload["pad_id"]),
            bos_id=int(payload["bos_id"]),
            eos_id=int(payload["eos_id"]),
            byte_offset=int(payload["byte_offset"]),
        )


_PIECE_RE = re.compile(
    r"\r\n|\n|\s+|<\|[a-z_]+\|>|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[:|,{}()[\]\"'./\\;-]|[^\sA-Za-z0-9_]"
)


@dataclass(frozen=True)
class PatternTokenizer:
    pieces: tuple[str, ...]
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3

    @property
    def piece_offset(self) -> int:
        return self.byte_offset + 256

    @property
    def vocab_size(self) -> int:
        return self.piece_offset + len(self.pieces)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        piece_to_id = {piece: self.piece_offset + index for index, piece in enumerate(self.pieces)}
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in _pieces(text):
            token_id = piece_to_id.get(piece)
            if token_id is not None:
                ids.append(token_id)
            else:
                ids.extend(self.byte_offset + byte for byte in piece.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_pieces(self, pieces: list[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in pieces:
            ids.extend(self.encode(piece))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id in {self.pad_id, self.bos_id, self.eos_id}:
                flush_bytes()
                if not skip_special:
                    output.append(_special_token_text(token_id, self))
                continue
            if self.byte_offset <= token_id < self.piece_offset:
                byte_buffer.append(token_id - self.byte_offset)
                continue
            piece_index = token_id - self.piece_offset
            if 0 <= piece_index < len(self.pieces):
                flush_bytes()
                output.append(self.pieces[piece_index])
                continue
            raise ValueError(f"token id out of range: {token_id}")
        flush_bytes()
        return "".join(output)

    def save(self, path: Path) -> None:
        payload = {
            "type": "pattern",
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "byte_offset": self.byte_offset,
            "pieces": list(self.pieces),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PatternTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "pattern":
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')}")
        return cls(
            pieces=tuple(str(piece) for piece in payload["pieces"]),
            pad_id=int(payload["pad_id"]),
            bos_id=int(payload["bos_id"]),
            eos_id=int(payload["eos_id"]),
            byte_offset=int(payload["byte_offset"]),
        )


def train_pattern_tokenizer(texts: list[str], *, vocab_size: int = 2048, min_count: int = 1) -> PatternTokenizer:
    base = ByteTokenizer().vocab_size
    if vocab_size < base:
        raise ValueError(f"vocab_size must be at least {base}")
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_pieces(text))
    max_pieces = vocab_size - base
    pieces = [
        piece
        for piece, count in sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        if count >= min_count
    ][:max_pieces]
    return PatternTokenizer(pieces=tuple(pieces))


def _pieces(text: str) -> list[str]:
    pieces: list[str] = []
    index = 0
    for match in _PIECE_RE.finditer(text):
        if match.start() > index:
            pieces.append(text[index : match.start()])
        pieces.append(match.group(0))
        index = match.end()
    if index < len(text):
        pieces.append(text[index:])
    return [piece for piece in pieces if piece]


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    if len(ids) < 2:
        return ids
    merged: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and ids[index] == pair[0] and ids[index + 1] == pair[1]:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


def _special_token_text(token_id: int, tokenizer: ByteTokenizer) -> str:
    if token_id == tokenizer.pad_id:
        return PAD_TOKEN
    if token_id == tokenizer.bos_id:
        return BOS_TOKEN
    if token_id == tokenizer.eos_id:
        return EOS_TOKEN
    raise ValueError(f"not a special token id: {token_id}")
