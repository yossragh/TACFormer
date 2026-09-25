from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from utils import read_json, write_json


# Atom-aware SMILES tokenizer.
# Important:
# - bracket atoms like [C@@H] are kept as one token
# - two-character atoms like Br and Cl are matched before single letters
# - ring closures like %10 are supported
# - backslash bond token is matched explicitly
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]"          # bracket expressions, e.g. [C@@H], [nH], [NH+]
    r"|Br|Cl|Si|Se|Na|Ca|Li|Mg|Al"
    r"|%\d{2}"              # ring numbers like %10
    r"|@@?|\\|/"            # stereochemistry / bond direction
    r"|\(|\)|\.|=|#|-|\+|:|~|@|\?|\*|\$|>"
    r"|[A-Za-z]"            # single-letter atoms/aromatic symbols
    r"|\d)"                 # ring digits 0-9
)


class SmilesTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, token_to_id: dict[str, int], min_freq: int = 1) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.min_freq = min_freq

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def tokenize(cls, smiles: str) -> list[str]:
        tokens = SMILES_REGEX.findall(smiles)

        if "".join(tokens) != smiles:
            raise ValueError(f"Tokenizer could not fully parse SMILES: {smiles}")

        return tokens

    @classmethod
    def build_from_smiles(
        cls,
        smiles_list: Iterable[str],
        min_freq: int = 1,
    ) -> "SmilesTokenizer":
        counter: Counter[str] = Counter()

        for smiles in smiles_list:
            counter.update(cls.tokenize(smiles))

        specials = [cls.PAD, cls.BOS, cls.EOS, cls.UNK]
        token_to_id = {
            tok: idx
            for idx, tok in enumerate(specials)
        }

        for tok, freq in sorted(counter.items()):
            if freq >= min_freq and tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

        return cls(
            token_to_id=token_to_id,
            min_freq=min_freq,
        )

    def encode(
        self,
        smiles: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []

        if add_bos:
            ids.append(self.bos_id)

        for tok in self.tokenize(smiles):
            ids.append(self.token_to_id.get(tok, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(
        self,
        ids: list[int],
        stop_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        toks: list[str] = []

        for idx in ids:
            tok = self.id_to_token.get(int(idx), self.UNK)

            if stop_at_eos and tok == self.EOS:
                break

            # Do not skip UNK here.
            # Keeping <unk> visible prevents generated sequences such as
            # CC<unk>O from being silently converted into different valid
            # molecules such as CCO during evaluation.
            if skip_special_tokens and tok in {self.PAD, self.BOS, self.EOS}:
                continue

            toks.append(tok)

        return "".join(toks)

    def to_dict(self) -> dict:
        return {
            "token_to_id": self.token_to_id,
            "min_freq": self.min_freq,
        }

    def save(self, path: str | Path) -> None:
        write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        obj = read_json(path)

        return cls(
            token_to_id=obj["token_to_id"],
            min_freq=obj.get("min_freq", 1),
        )