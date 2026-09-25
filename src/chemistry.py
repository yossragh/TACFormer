from __future__ import annotations

import random
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from rdkit.Contrib.SA_Score import sascorer as _sascorer

    _SA_AVAILABLE = True
except ImportError:
    _SA_AVAILABLE = False


# ---------------------------------------------------------------------
# RDKit logging
# ---------------------------------------------------------------------


def suppress_rdkit_warnings() -> None:
    RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------
# SMILES parsing and canonicalization
# ---------------------------------------------------------------------


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    return Chem.MolFromSmiles(smiles)


def canonicalize_from_mol(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

def randomize_smiles_from_mol(
    mol: Chem.Mol | None,
    isomeric_smiles: bool = True,
    max_attempts: int = 10,
) -> str | None:
    if mol is None:
        return None

    for _ in range(max_attempts):
        try:
            randomized = Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=isomeric_smiles,
            )
        except Exception:
            continue

        if randomized:
            return randomized

    return None

# ---------------------------------------------------------------------
# Murcko scaffolds
# ---------------------------------------------------------------------


def murcko_scaffold_from_mol(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None

    scaffold_smiles = Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=True,
    )

    return scaffold_smiles if scaffold_smiles else None


# ---------------------------------------------------------------------
# Morgan fingerprints
# ---------------------------------------------------------------------


def get_morgan_fingerprints_from_mols(
    mols: list[Chem.Mol | None],
    radius: int = 2,
    n_bits: int = 2048,
) -> list[Any]:
    """
    Compute Morgan fingerprints for a list of RDKit Mol objects.

    None values are skipped.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    return [
        generator.GetFingerprint(mol)
        for mol in mols
        if mol is not None
    ]


# ---------------------------------------------------------------------
# Per-molecule properties
# ---------------------------------------------------------------------


def compute_mol_properties(mol: Chem.Mol | None) -> dict[str, float | int] | None:
    """
    Compute basic molecule-level RDKit properties.

    Returns None if mol is None.
    """
    if mol is None:
        return None

    props: dict[str, float | int] = {
        "mw": float(Descriptors.MolWt(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "n_rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "n_rings": int(mol.GetRingInfo().NumRings()),
        "n_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "n_non_aromatic_rings": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "n_hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "n_hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
    }

    if _SA_AVAILABLE:
        props["sa_score"] = float(_sascorer.calculateScore(mol))

    return props


# ---------------------------------------------------------------------
# Chemical diversity
# ---------------------------------------------------------------------


def internal_diversity_from_mols(
    mols: list[Chem.Mol | None],
    radius: int = 2,
    n_bits: int = 2048,
    max_samples: int = 1000,
    seed: int = 42,
) -> float | None:
    """
    Compute internal diversity from already-parsed RDKit Mol objects.

    Internal diversity is the mean pairwise Tanimoto distance:

        distance = 1 - Tanimoto similarity

    Molecules are optionally subsampled before pairwise comparison to keep
    the calculation tractable for large generated sets.
    """
    valid_mols = [
        mol
        for mol in mols
        if mol is not None
    ]

    if len(valid_mols) < 2:
        return None

    if len(valid_mols) > max_samples:
        rng = random.Random(seed)
        valid_mols = rng.sample(valid_mols, max_samples)

    fingerprints = get_morgan_fingerprints_from_mols(
        mols=valid_mols,
        radius=radius,
        n_bits=n_bits,
    )

    if len(fingerprints) < 2:
        return None

    total_distance = 0.0
    num_pairs = 0

    for i in range(len(fingerprints) - 1):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i],
            fingerprints[i + 1:],
        )

        total_distance += sum(1.0 - sim for sim in similarities)
        num_pairs += len(similarities)

    if num_pairs == 0:
        return None

    return float(total_distance / num_pairs)


def cap_mapped_dummy_with_hydrogen(
    mol: Chem.Mol | None,
    atom_map_num: int = 1,
) -> Chem.Mol | None:
    if mol is None:
        return None

    rw_mol = Chem.RWMol(mol)

    dummy_indices = [
        atom.GetIdx()
        for atom in rw_mol.GetAtoms()
        if (
            atom.GetAtomicNum() == 0
            and atom.GetAtomMapNum() == atom_map_num
        )
    ]

    if not dummy_indices:
        result = Chem.Mol(mol)

        try:
            Chem.SanitizeMol(result)
        except Exception:
            return None

        return result

    for atom_idx in dummy_indices:
        atom = rw_mol.GetAtomWithIdx(atom_idx)

        if atom.GetDegree() != 1:
            return None

        bonds = list(atom.GetBonds())

        if len(bonds) != 1:
            return None

        if bonds[0].GetBondType() != Chem.BondType.SINGLE:
            return None

    for atom_idx in dummy_indices:
        hydrogen = Chem.Atom(1)
        hydrogen.SetNoImplicit(True)

        rw_mol.ReplaceAtom(
            atom_idx,
            hydrogen,
            preserveProps=False,
        )

    capped = rw_mol.GetMol()

    try:
        Chem.SanitizeMol(capped)
    except Exception:
        return None

    try:
        parent = Chem.RemoveHs(capped)
        Chem.SanitizeMol(parent)
    except Exception:
        return None

    if any(
        atom.GetAtomicNum() == 0
        for atom in parent.GetAtoms()
    ):
        return None

    return parent