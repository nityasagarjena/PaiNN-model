from ase.md.langevin import Langevin
from ase.calculators.plumed import Plumed
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory

import numpy as np
import torch
import sys
import glob

from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel_predict
from PaiNN.calculator import MLCalculator
from ase.constraints import FixAtoms

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_pth = glob.glob('/home/energy/xinyang/work/Au_MD/graphnn/ads_images/ensembles/*_layer/runs/model_outputs/best_model.pth')
# # models = []
# for each in model_pth:
#     node_size = int(each.split('/')[-4].split('_')[0])
#     num_inter = int(each.split('/')[-4].split('_')[2])
#     model = PainnModel(num_interactions=num_inter, hidden_state_size=node_size, cutoff=5.0)
#     model.to(device)
#     state_dict = torch.load(each)
#     model.load_state_dict(state_dict["model"])
#     models.append(model)
# 
# encalc = EnsembleCalculator(models)

# set md parameters
#dataset="/home/energy/xinyang/work/Au_MD/training_loop/Au_larger/dataset_selector/dataset_repository/corrected_ads_images.traj"
#images = read(dataset, ':')
#indices = [i for i in range(len(images)) if images[i].info['system'] == '1OH']
#atoms = images[np.random.choice(indices)]
#atoms = read('MD.traj', -1)
#cons = FixAtoms(mask=atoms.positions[:, 2] < 5.9)
#atoms.set_constraint(cons)

model = PainnModel_predict(num_interactions=3, hidden_state_size=128, cutoff=5.0)
model.to(device)
state_dict = torch.load('/home/energy/xinyang/work/Au_MD/graphnn/pure_water/runs/model_outputs/best_model.pth')
new_names = ["linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"]
old_names = ["readout_mlp.0.weight", "readout_mlp.0.bias", "readout_mlp.2.weight", "readout_mlp.2.bias"]
for old, new in zip(old_names, new_names):
    state_dict['model'][new] = state_dict['model'].pop(old)

state_dict["model"]["U_in_0"] = torch.randn(128, 500) / 500 ** 0.5
state_dict["model"]["U_out_1"] = torch.randn(128, 500) / 500 ** 0.5
state_dict["model"]["U_in_1"] = torch.randn(128, 500) / 500 ** 0.5
model.load_state_dict(state_dict["model"])
mlcalc = MLCalculator(model)

atoms = read('water_O2.cif')
atoms.calc = mlcalc
atoms.get_potential_energy()

collect_traj = Trajectory('bad_struct.traj', 'a')
steps = 0
def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()
    ensemble = a.calc.results['ensemble']
    energy_var = ensemble['energy_var']
    forces_var = np.mean(ensemble['forces_var'])
    forces_l2_var = np.mean(ensemble['forces_l2_var'])
    global steps
    steps += 1
    
    if (forces_l2_var > 0.0):
        with open('ensemble.log', 'a') as f:
            f.write(
            f"Steps={steps:12.3f} Epot={epot:12.3f} Ekin={ekin:12.3f} temperature={temp:8.2f} energy_var={energy_var:10.6f} forces_var={forces_var:10.6f} forces_l2_var={forces_l2_var:10.6f}\n")
    
    if (forces_l2_var > 0.01):
        collect_traj.write(a)
        with open('bad_ensemble.log', 'a') as f:
            f.write(
            f"Steps={steps:12.3f} Epot={epot:12.3f} Ekin={ekin:12.3f} temperature={temp:8.2f} energy_var={energy_var:10.6f} forces_var={forces_var:10.6f} forces_l2_var={forces_l2_var:10.6f}\n")
    
    if len(collect_traj) > 1000:
        sys.exit()

#atoms.calc = encalc
MaxwellBoltzmannDistribution(atoms, temperature_K=350)
dyn = Langevin(atoms, 0.25 * units.fs, temperature_K=350, friction=0.1)
dyn.attach(printenergy, interval=1)

traj = Trajectory('MD.traj', 'w', atoms)
dyn.attach(traj.write, interval=400)
dyn.run(10000000)
