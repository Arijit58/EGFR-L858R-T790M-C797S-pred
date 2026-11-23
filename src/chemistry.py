"""
Chemical standardization, featurization, SMILES enumeration, and atom-level
pharmacophore attribution utilities for EGFR C797S NCE screening.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()

# Modern RDKit MorganGenerator with chirality enabled (critical for EGFR C797S enantiomer cliffs)
try:
    from rdkit.Chem import rdFingerprintGenerator
    _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
except Exception:
    _MORGAN_GEN = None


@dataclass
class MoleculeRecord:
    mol: Chem.Mol
    canonical_smiles: str
    inchikey: str
    murcko_smiles: str


def standardize_smiles(smiles: str, uncharge: bool = False) -> Optional[MoleculeRecord]:
    """
    Parse, sanitize, keep largest organic fragment, and return canonical
    ISOMERIC SMILES (preserving @/@@ tetrahedral and E/Z stereochemistry).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        mol = _LARGEST_FRAG.choose(mol)
        if uncharge:
            mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
        can_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        mol_can = Chem.MolFromSmiles(can_smi)
        if mol_can is None:
            return None
        inchikey = Chem.MolToInchiKey(mol_can)
        try:
            scaff = MurckoScaffold.GetScaffoldForMol(mol_can)
            murcko_smi = Chem.MolToSmiles(scaff, canonical=True, isomericSmiles=False)
        except Exception:
            murcko_smi = ""
        return MoleculeRecord(
            mol=mol_can,
            canonical_smiles=can_smi,
            inchikey=inchikey,
            murcko_smiles=murcko_smi,
        )
    except Exception:
        return None


def enumerate_smiles(smiles: str, n_variants: int = 5, seed: int = 42) -> List[str]:
    """
    Generate canonical SMILES + up to (n_variants - 1) distinct randomized
    isomeric SMILES strings for LLM sequence augmentation / Test-Time Augmentation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if n_variants <= 1:
        return [can]

    rng = np.random.RandomState(seed)
    variants = [can]
    seen = {can}
    for _ in range(n_variants * 6):
        if len(variants) >= n_variants:
            break
        atom_order = list(range(mol.GetNumAtoms()))
        rng.shuffle(atom_order)
        renum = Chem.RenumberAtoms(mol, atom_order)
        rnd_smi = Chem.MolToSmiles(renum, canonical=False, isomericSmiles=True)
        if rnd_smi not in seen:
            seen.add(rnd_smi)
            variants.append(rnd_smi)
    return variants


def compute_chiral_ecfp4_bitvect(mol: Chem.Mol):
    """Return RDKit ExplicitBitVect for Tanimoto similarity."""
    if _MORGAN_GEN is not None:
        return _MORGAN_GEN.GetFingerprint(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)


def compute_chiral_ecfp4_array(mol: Chem.Mol) -> np.ndarray:
    """Return 2048-d float32 numpy vector for chiral ECFP4."""
    bv = compute_chiral_ecfp4_bitvect(mol)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


DESCRIPTOR_NAMES = [
    "MW",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "ArRings",
    "HeteroCycles",
    "Fsp3",
    "QED",
    "HeavyAtoms",
    "ChiralCenters",
]


def compute_physchem_vector(mol: Chem.Mol) -> np.ndarray:
    """Compute 12-d medicinal chemistry descriptor vector."""
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    vals = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        QED.qed(mol),
        float(mol.GetNumHeavyAtoms()),
        float(chiral_centers),
    ]
    return np.array(vals, dtype=np.float32)


def compute_nce_profile(mol: Chem.Mol) -> Dict[str, object]:
    """Full medicinal chemistry & drug-likeness profile for the NCE web app."""
    mw = round(Descriptors.MolWt(mol), 2)
    clogp = round(Crippen.MolLogP(mol), 2)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))
    ar_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    het_cycles = int(rdMolDescriptors.CalcNumHeterocycles(mol))
    fsp3 = round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
    qed_val = round(QED.qed(mol), 3)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    lipinski_violations = sum([mw > 500, clogp > 5, hbd > 5, hba > 10])
    veber_pass = (rotb <= 10) and (tpsa <= 140)

    # Detect key EGFR C797S relevant structural alerts/pharmacophores
    smarts_motifs = {
        "Phosphine oxide [P(=O)(C)C] (Brigatinib-like HBA)": "[P](=O)([CH3])[CH3]",
        "2-Aminopyrimidine hinge core": "c1nc(N)ncc1",
        "Sulfonamide / Sulfonyl group": "[S](=O)(=O)",
        "Acrylamide covalent warhead (Cys797-dependent)": "[NH]C(=O)C=C",
        "Basic piperazine / piperidine solubilizing tail": "N1CCNCC1",
        "Aminothiazole / Thiazole heterocycle": "c1cscn1",
    }
    detected_motifs = []
    for label, sma in smarts_motifs.items():
        patt = Chem.MolFromSmarts(sma)
        if patt is not None and mol.HasSubstructMatch(patt):
            detected_motifs.append(label)

    return {
        "formula": formula,
        "MW": mw,
        "cLogP": clogp,
        "TPSA": tpsa,
        "HBD": hbd,
        "HBA": hba,
        "RotBonds": rotb,
        "ArRings": ar_rings,
        "HeteroCycles": het_cycles,
        "Fsp3": fsp3,
        "QED": qed_val,
        "lipinski_violations": int(lipinski_violations),
        "lipinski_pass": lipinski_violations <= 1,
        "veber_pass": bool(veber_pass),
        "detected_motifs": detected_motifs,
    }


def render_mol_svg(
    mol: Chem.Mol,
    atom_weights: Optional[np.ndarray] = None,
    width: int = 440,
    height: int = 300,
) -> str:
    """
    Render clean 2D structure SVG, optionally shaded with atom-level attribution
    weights (green = pro-active against EGFR C797S, coral/red = pro-inactive).
    """
    mol_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_draw)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.0
    opts.padding = 0.08

    if atom_weights is not None and len(atom_weights) == mol_draw.GetNumAtoms():
        max_abs = float(np.max(np.abs(atom_weights)))
        highlight_atoms = []
        atom_colors = {}
        radii = {}
        if max_abs > 1e-6:
            norm_w = atom_weights / max_abs
            for idx, w in enumerate(norm_w):
                intensity = float(min(abs(w), 1.0))
                if intensity < 0.15:
                    continue
                highlight_atoms.append(idx)
                radii[idx] = 0.28 + 0.18 * intensity
                if w > 0:
                    # Emerald green tint for positive contribution to C797S activity
                    atom_colors[idx] = (0.16, 0.78, 0.45, 0.35 + 0.45 * intensity)
                else:
                    # Coral red tint for negative contribution (detrimental motif)
                    atom_colors[idx] = (0.94, 0.33, 0.31, 0.35 + 0.45 * intensity)
        drawer.DrawMolecule(
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=[],
            highlightAtomRadii=radii,
        )
    else:
        drawer.DrawMolecule(mol_draw)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_mol_png_bytes(
    mol: Chem.Mol,
    atom_weights: Optional[np.ndarray] = None,
    width: int = 430,
    height: int = 275,
) -> bytes:
    """
    Render crisp high-DPI 2D structure PNG bytes via RDKit MolDraw2DCairo
    for the standalone native Windows Desktop GUI application.
    """
    mol_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_draw)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.1
    opts.padding = 0.08

    if atom_weights is not None and len(atom_weights) == mol_draw.GetNumAtoms():
        max_abs = float(np.max(np.abs(atom_weights)))
        highlight_atoms = []
        atom_colors = {}
        radii = {}
        if max_abs > 1e-6:
            norm_w = atom_weights / max_abs
            for idx, w in enumerate(norm_w):
                intensity = float(min(abs(w), 1.0))
                if intensity < 0.15:
                    continue
                highlight_atoms.append(idx)
                radii[idx] = 0.28 + 0.18 * intensity
                if w > 0:
                    atom_colors[idx] = (0.16, 0.78, 0.45, 0.35 + 0.45 * intensity)
                else:
                    atom_colors[idx] = (0.94, 0.33, 0.31, 0.35 + 0.45 * intensity)
        drawer.DrawMolecule(
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=[],
            highlightAtomRadii=radii,
        )
    else:
        drawer.DrawMolecule(mol_draw)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()
