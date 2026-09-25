from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from chemistry import mol_from_smiles, randomize_smiles_from_mol
from tokenizer import SmilesTokenizer
from utils import read_jsonl


class SmilesDataset(Dataset):
    """
    Dataset for unconditional SMILES language modeling.

    Expected input format:
        one JSON object per line, for example:
        {
            "smiles": "...",
            "target_uniprot": "...",
            "poi_norm": "...",
            "e3_norm": "..."
        }

    This dataset:
        - reads a JSONL file
        - takes the SMILES field
        - optionally randomizes SMILES on the fly
        - tokenizes it
        - adds BOS/EOS
        - truncates to max_seq_len if needed

    Randomized SMILES augmentation should normally be enabled only for
    the training split, not validation.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: SmilesTokenizer,
        smiles_field: str = "smiles",
        max_seq_len: int = 320,
        randomize_smiles: bool = False,
        randomize_probability: float = 1.0,
        randomize_max_attempts: int = 10,
    ) -> None:
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2 for causal LM shifting.")

        if not 0.0 <= randomize_probability <= 1.0:
            raise ValueError("randomize_probability must be between 0.0 and 1.0.")

        if randomize_max_attempts < 1:
            raise ValueError("randomize_max_attempts must be at least 1.")

        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.smiles_field = smiles_field
        self.max_seq_len = max_seq_len
        self.randomize_smiles = randomize_smiles
        self.randomize_probability = randomize_probability
        self.randomize_max_attempts = randomize_max_attempts

        self.mols = None

        if self.randomize_smiles:
            self.mols = []

            for record in self.records:
                if self.smiles_field not in record:
                    raise KeyError(f"Record is missing SMILES field: {self.smiles_field}")

                self.mols.append(mol_from_smiles(record[self.smiles_field]))

    def __len__(self) -> int:
        return len(self.records)

    def _get_training_smiles(self, idx: int, original_smiles: str) -> str:
        """
        Return either the original SMILES or an on-the-fly randomized SMILES.
        """
        if not self.randomize_smiles:
            return original_smiles

        if random.random() > self.randomize_probability:
            return original_smiles

        mol = self.mols[idx]

        randomized = randomize_smiles_from_mol(
            mol,
            max_attempts=self.randomize_max_attempts,
        )

        if randomized is None:
            return original_smiles

        return randomized

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        if self.smiles_field not in record:
            raise KeyError(f"Record is missing SMILES field: {self.smiles_field}")

        original_smiles = record[self.smiles_field]
        training_smiles = self._get_training_smiles(
            idx=idx,
            original_smiles=original_smiles,
        )

        ids = self.tokenizer.encode(
            training_smiles,
            add_bos=True,
            add_eos=True,
        )

        ids = ids[: self.max_seq_len]

        if len(ids) == self.max_seq_len and ids[-1] != self.tokenizer.eos_id:
            ids[-1] = self.tokenizer.eos_id

        meta = {
            k: v
            for k, v in record.items()
            if k != self.smiles_field
        }

        if self.randomize_smiles:
            meta["original_smiles"] = original_smiles
            meta["randomized_smiles_used"] = training_smiles != original_smiles

        return {
            "ids": ids,
            "smiles": training_smiles,
            "meta": meta,
        }


class CausalLMCollator:
    """
    Collator for decoder-only language modeling.

    Given token ids:
        [BOS, t1, t2, t3, EOS]

    create:
        input_ids = [BOS, t1, t2, t3]
        labels    = [t1,  t2, t3, EOS]

    Padding:
        - input_ids padded with pad_id
        - labels padded with -100 so loss ignores them
        - attention_mask padded with 0
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(batch) == 0:
            raise ValueError("Received empty batch in CausalLMCollator.")

        max_len = max(len(item["ids"]) for item in batch)

        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for item in batch:
            ids = item["ids"]

            x = ids[:-1]
            y = ids[1:]

            pad_len = (max_len - 1) - len(x)

            input_ids.append(x + [self.pad_id] * pad_len)
            labels.append(y + [-100] * pad_len)
            attention_mask.append([1] * len(x) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }