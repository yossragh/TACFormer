from __future__ import annotations

from typing import Iterable

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


def draw_molecule_grid(
    smiles_list: list[str],
    legends: list[str] | None = None,
    mols_per_row: int = 5,
    sub_img_size: tuple[int, int] = (300, 300),
):
    mols: list[Chem.Mol] = []
    filtered_legends: list[str] = []

    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        if legends is not None and i < len(legends):
            filtered_legends.append(legends[i])

    if not mols:
        return None

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=filtered_legends if legends is not None else None,
        useSVG=False,
    )


def draw_scaffold_grid(
    smiles_list: list[str],
    legends: list[str] | None = None,
    mols_per_row: int = 5,
    sub_img_size: tuple[int, int] = (300, 300),
):
    scaffold_mols: list[Chem.Mol] = []
    scaffold_legends: list[str] = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            continue

        scaffold_mols.append(scaffold)

        if legends is not None and i < len(legends):
            scaffold_legends.append(legends[i])

    if not scaffold_mols:
        return None

    return Draw.MolsToGridImage(
        scaffold_mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=scaffold_legends if legends is not None else None,
        useSVG=False,
    )


def draw_highlighted_scaffold_grid(
    smiles_list: list[str],
    legends: list[str] | None = None,
    mols_per_row: int = 5,
    sub_img_size: tuple[int, int] = (300, 300),
):
    mols: list[Chem.Mol] = []
    highlight_atom_lists: list[list[int]] = []
    highlight_bond_lists: list[list[int]] = []
    kept_legends: list[str] = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            continue

        match = mol.GetSubstructMatch(scaffold)
        if not match:
            continue

        highlight_atoms = list(match)

        highlight_bonds = []
        match_set = set(match)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in match_set and a2 in match_set:
                highlight_bonds.append(bond.GetIdx())

        mols.append(mol)
        highlight_atom_lists.append(highlight_atoms)
        highlight_bond_lists.append(highlight_bonds)

        if legends is not None and i < len(legends):
            kept_legends.append(legends[i])

    if not mols:
        return None

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=kept_legends if legends is not None else None,
        highlightAtomLists=highlight_atom_lists,
        highlightBondLists=highlight_bond_lists,
        useSVG=False,
    )