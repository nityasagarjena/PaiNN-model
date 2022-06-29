from ase.db import connect
from ase.io import read, write, Trajectory
import torch
from typing import List
import asap3
import numpy as np

def ase_properties(atoms):
    """Guess dataset format from an ASE atoms"""
    atoms_prop = {
        'elems': {'dtype':  'int32', 'shape': [None]},
        'coord': {'dtype':  'float', 'shape': [None, 3]}}

    if atoms.pbc.any():
        atoms_prop['cell'] = {'dtype': 'float', 'shape': [3, 3]}

    try:
        atoms.get_potential_energy()
        atoms_prop['energy'] = {'dtype': 'float', 'shape': []}
    except:
        pass

    try:
        atoms.get_forces()
        atoms_prop['forces'] = {'dtype': 'float', 'shape': [None, 3]}
    except:
        pass

    return atoms_prop

def ase_data_reader(atoms, atoms_prop):
    atoms_data = {
        'num_atoms': torch.tensor(atoms.get_global_number_of_atoms()),
        'elems': torch.tensor(atoms.numbers),
        'coord': torch.tensor(atoms.positions, dtype=torch.float),
    }
    if 'cell' in atoms_prop:
        atoms_data['cell'] = torch.tensor(atoms.cell[:], dtype=torch.float)

    if 'energy' in atoms_prop:
        atoms_data['energy'] = torch.tensor(atoms.get_potential_energy(), dtype=torch.float)

    if 'forces' in atoms_prop:
        atoms_data['forces'] = torch.tensor(atoms.get_forces(), dtype=torch.float)
    
    return atoms_data

class AseDataset(torch.utils.data.Dataset):
    def __init__(self, ase_db, cutoff, **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(ase_db, str):
            self.db = Trajectory(ase_db)
        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_prop = ase_properties(self.db[0])
        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        atoms = self.db[idx]    # ase database indexing from 1 
        atoms_data = ase_data_reader(atoms, self.atoms_prop)
        nl = asap3.FullNeighborList(self.cutoff, atoms)
        pair_i_idx = []
        pair_j_idx = []
        n_diff = []
        for i in range(len(atoms)):
            indices, diff, _ = nl.get_neighbors(i)
            pair_i_idx += [i] * len(indices)               # local index of pair i
            pair_j_idx.append(indices)   # local index of pair j
            n_diff.append(diff)
        
        pair_j_idx = np.concatenate(pair_j_idx)
        pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
        atoms_data['pairs'] = torch.from_numpy(pairs)
        atoms_data['n_diff'] = torch.from_numpy(np.concatenate(n_diff)).float()
        atoms_data['num_pairs'] = torch.tensor(pairs.shape[0])
        
        return atoms_data

def cat_tensors(tensors: List[torch.Tensor]):
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)

def collate_atomsdata(atoms_data: List[dict], pin_memory=True):
    # convert from list of dicts to dict of lists
    dict_of_lists = {k: [dic[k] for dic in atoms_data] for k in atoms_data[0]}
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x
        
    collated = {k: cat_tensors(v) for k, v in dict_of_lists.items()}
    return collated
