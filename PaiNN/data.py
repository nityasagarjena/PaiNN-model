from ase.io import read, write, Trajectory
import torch
from typing import List
import asap3
import numpy as np
from scipy.spatial import distance_matrix

# def ase_properties(atoms):
#     """Guess dataset format from an ASE atoms"""
#     atoms_prop = []
# 
#     if atoms.pbc.any():
#         atoms_prop.append('cell')
# 
#     try:
#         atoms.get_potential_energy()
#         atoms_prop.append('energy')
#     except:
#         pass
# 
#     try:
#         atoms.get_forces()
#         atoms_prop.append('forces')
#     except:
#         pass
# 
#     return atoms_prop

class AseDataReader:
    def __init__(self, cutoff=5.0):            
        self.cutoff = cutoff
        
    def __call__(self, atoms):
        atoms_data = {
            'num_atoms': torch.tensor([atoms.get_global_number_of_atoms()]),
            'elems': torch.tensor(atoms.numbers),
            'coord': torch.tensor(atoms.positions, dtype=torch.float),
        }
        
        if atoms.pbc.any():
            pairs, n_diff = self.get_neighborlist(atoms)
            atoms_data['cell'] = torch.tensor(atoms.cell[:], dtype=torch.float)
        else:
            pairs, n_diff = self.get_neighborlist_simple(atoms)
            
        atoms_data['pairs'] = torch.from_numpy(pairs)
        atoms_data['n_diff'] = torch.from_numpy(n_diff).float()
        atoms_data['num_pairs'] = torch.tensor([pairs.shape[0]])
        
        try:
            energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)
            atoms_data['energy'] = energy
        except (AttributeError, RuntimeError):
            pass
        
        try: 
            forces = torch.tensor(atoms.get_forces(apply_constraint=False), dtype=torch.float)
            atoms_data['forces'] = forces
        except (AttributeError, RuntimeError):
            pass
        
        return atoms_data
            
    
    def get_neighborlist(self, atoms):        
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
        n_diff = np.concatenate(n_diff)
        
        return pairs, n_diff
    
    def get_neighborlist_simple(self, atoms):
        pos = atoms.get_positions()
        dist_mat = distance_matrix(pos, pos)
        mask = dist_mat < self.cutoff
        np.fill_diagonal(mask, False)        
        pairs = np.argwhere(mask)
        n_diff = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        
        return pairs, n_diff

class AseDataset(torch.utils.data.Dataset):
    def __init__(self, ase_db, cutoff=5.0, **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(ase_db, str):
            self.db = Trajectory(ase_db)
        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_reader = AseDataReader(cutoff)
        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        atoms = self.db[idx]
        atoms_data = self.atoms_reader(atoms)
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
