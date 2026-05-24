"""
Convert SMILES strings into heterogeneous graph objects for inference.
"""

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import HeteroData


BOND_NAMES = {
    BT.SINGLE: "single",
    BT.DOUBLE: "double",
    BT.TRIPLE: "triple",
    BT.AROMATIC: "aromatic",
}


def smiles_to_hetero_data(smiles: str, use_H: bool = True) -> HeteroData:
    """
    Build a HeteroData graph from a SMILES string using the same
    atom and bond featurization as the TadfPL dataset.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if use_H:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    N = mol.GetNumAtoms()

    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    formal_charge = []
    degrees = []
    total_degrees = []
    explicit_valence = []
    implicit_valence = []
    is_in_ring = []
    atomic_number = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        formal_charge.append(atom.GetFormalCharge())
        degrees.append(atom.GetDegree())
        explicit_valence.append(atom.GetValence(which=Chem.ValenceType.EXPLICIT))
        if use_H:
            implicit_valence.append(0)
            total_degrees.append(0)
            num_hs.append(0)
        else:
            implicit_valence.append(atom.GetValence(which=Chem.ValenceType.IMPLICIT))
            total_degrees.append(atom.GetTotalDegree())
            num_hs.append(atom.GetTotalNumHs())
        is_in_ring.append(atom.IsInRing())

    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

    hetero_bonds = {
        BT.SINGLE: [],
        BT.DOUBLE: [],
        BT.TRIPLE: [],
        BT.AROMATIC: [],
    }
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        hetero_bonds[bond.GetBondType()].extend(((start, end), (end, start)))

    x = torch.tensor(
        [
            atomic_number,
            aromatic,
            sp,
            sp2,
            sp3,
            num_hs,
            formal_charge,
            degrees,
            total_degrees,
            explicit_valence,
            implicit_valence,
            is_in_ring,
        ],
        dtype=torch.float,
    ).t().contiguous()

    graph_dict = {
        ("atom", BOND_NAMES[bond_type], "atom"): {
            "edge_index": torch.tensor(bond_idxs, dtype=torch.long).t().contiguous()
            if len(bond_idxs) > 0
            else torch.empty(2, 0, dtype=torch.long)
        }
        for bond_type, bond_idxs in hetero_bonds.items()
    }
    graph_dict["atom"] = {"x": x, "z": z}

    return HeteroData(
        graph_dict,
        smiles=smiles,
        mol_size=sum(aromatic),
    )
