import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter
from torch_geometric.transforms import ToUndirected

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


class TadfPL(InMemoryDataset):
    """
    .. note::

        We also provide a pre-processed version of the dataset in case
        :class:`rdkit` is not installed. The pre-processed version matches with
        the manually processed version as outlined in :meth:`process`.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
    """  # noqa: E501


    def __init__(
        self,
        root: str,
        use_H: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.use_H = use_H
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['TadfPL_rdkit_one_conf.sdf', 'TadfPL.csv']
        except ImportError:
            raise NotImplementedError(
                "The dataset is not available in the pre-processed format. ")

    @property
    def processed_file_names(self) -> str:
        return 'data_one_conf.pt'

    # local dataset no need to download
    # def download(self) -> None:
    #     try:
    #         import rdkit  # noqa
    #         file_path = download_url(self.raw_url, self.raw_dir)
    #         extract_zip(file_path, self.raw_dir)
    #         os.unlink(file_path)

    #         file_path = download_url(self.raw_url2, self.raw_dir)
    #         os.rename(osp.join(self.raw_dir, '3195404'),
    #                   osp.join(self.raw_dir, 'uncharacterized.txt'))
    #     except ImportError:
    #         path = download_url(self.processed_url, self.raw_dir)
    #         extract_zip(path, self.raw_dir)
    #         os.unlink(path)

    def process(self) -> None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')  # type: ignore
            WITH_RDKIT = True

        except ImportError:
            raise NotImplementedError(
                "The dataset is not available in the pre-processed format. "
                "Please install rdkit to process the raw data.")

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        bond_names = {BT.SINGLE: 'single', BT.DOUBLE: 'double',
                      BT.TRIPLE: 'triple', BT.AROMATIC: 'aromatic'}
        types = {'B': 1, 'Br': 11, 'C': 2, 'Cl': 9, 'F': 5, 'Ge': 10, 'H': 0, 'N': 3, 'O': 4, 'P': 7, 'S': 8, 'Si':6}
        solvent_types = {
            'CS(C)=O': 0,
            'Cc1ccccc1': 1,
            'CC1CCCCC1': 2,
            'C1CCOC1': 3,
            'CCCCCC': 4,
            'C1CCCCC1': 5,
            'CC#N': 6,
            'CC1CCCO1': 7,
            'CO': 8,
            'ClCCl': 9,
            'ClC(Cl)Cl': 10,
            'CN(C)C=O': 11,
        }
        solvent = []
        target = []
        with open(self.raw_paths[1]) as f:
            for line in f.read().split('\n')[1:-1]:
                solvent.append(line.split(",")[2])
                target.append(float(line.split(",")[1]))
        y = torch.tensor(target, dtype=torch.float) #y is sorted in the order of tadf_id

        # with open(self.raw_paths[2]) as f:
        #     skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)): #  i is the order or the molecule in the sdf file
            
            if not self.use_H:
                mol = Chem.RemoveHs(mol)

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
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
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                formal_charge.append(atom.GetFormalCharge())
                degrees.append(atom.GetDegree())
                explicit_valence.append(atom.GetExplicitValence())
                if self.use_H:
                    implicit_valence.append(0)
                    total_degrees.append(0)
                    num_hs.append(0)
                else:
                    implicit_valence.append(atom.GetImplicitValence())
                    total_degrees.append(atom.GetTotalDegree())
                    num_hs.append(atom.GetTotalNumHs())
                is_in_ring.append(atom.IsInRing())

            z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
            # edges are fully connected
            # bonds are only between chemically bonded atoms
            rows, cols, bond_types = [], [], []
            hetero_bonds = {
                BT.SINGLE: [],
                BT.DOUBLE: [],
                BT.TRIPLE: [],
                BT.AROMATIC: [],
            }
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                bond_types += 2 * [bonds[bond.GetBondType()]]
                hetero_bonds[bond.GetBondType()].extend(((start, end), (end, start)))
                

            bond_index = torch.tensor([rows, cols], dtype=torch.long)
            bond_types = torch.tensor(bond_types, dtype=torch.long)
            bond_attr = one_hot(bond_types, num_classes=len(bonds))
            
            perm = (bond_index[0] * N + bond_index[1]).argsort()
            bond_index = bond_index[:, perm]
            bond_types = bond_types[perm]
            bond_attr = bond_attr[perm]
            
            rows, cols = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
            edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)
            
            solvent_idx = [solvent_types[solvent[i]]]
            solvent_vector = one_hot(torch.tensor(solvent_idx), num_classes=len(solvent_types))
            
            mol_size = sum(aromatic)

            x = torch.tensor([
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
                dtype=torch.float).t().contiguous()

            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            graph_dict = {('atom', bond_names[bond_t], 'atom'):
                    {'edge_index': torch.tensor(bond_idxs, dtype=torch.long).t().contiguous() if len(bond_idxs) > 0 else torch.empty(2, 0, dtype=torch.long)}
                    for bond_t, bond_idxs in hetero_bonds.items()}
            graph_dict['atom'] = {'x': x, 'z': z, 'pos': pos}
            data = HeteroData(
                graph_dict,
                smiles=smiles,
                pl=y[i].unsqueeze(0),
                name=name,
                idx=i,
                solvent=solvent_vector,
                mol_size=mol_size,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
        
if __name__ == "__main__":
    ds = TadfPL(root='tadf_pl', use_H=False, force_reload=True)
    print(ds[642])
