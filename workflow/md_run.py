from ase.md.langevin import Langevin
from ase.calculators.plumed import Plumed
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory

import numpy as np
import torch
import sys
import glob
import toml
import argparse
from pathlib import Path
import logging

from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel
from PaiNN.calculator import MLCalculator, EnsembleCalculator
from ase.constraints import FixAtoms

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="MD simulations drive by graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--init_traj",
        type=str,
        help="Path to start configurations",
    )
    parser.add_argument(
        "--start_indice",
        type=int,
        help="Indice of the start configuration",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Where to find the models",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.5,
        help="Time step of MD simulation",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000000,
        help="Maximum steps of MD",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=100000,
        help="Minimum steps of MD, raise error if not reached",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=350.0,
        help="Maximum time steps of MD",
    )
    parser.add_argument(
        "--fix_under",
        type=float,
        default=5.9,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--dump_step",
        type=int,
        default=100,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=1,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--num_uncertain",
        type=int,
        default=1000,
        help="Stop MD when too many structures with large uncertainty are collected",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Set which device to use for running MD e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        ns.__dict__[k] = v

class CallsCounter:
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.func(*args, **kwargs)

def main():
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)

    setup_seed(args.random_seed)

    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler('md.log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler('error.log', mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))

    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)

    # load model
    model_pth = Path(args.load_model).rglob('*best_model.pth')
    models = []
    for each in model_pth:
        state_dict = torch.load(each) 
        model = PainnModel(
            num_interactions=state_dict["num_layer"], 
            hidden_state_size=state_dict["node_size"], 
            cutoff=state_dict["cutoff"],
        )
        model.to(args.device)
        model.load_state_dict(state_dict["model"])    
        models.append(model)

    encalc = EnsembleCalculator(models)

    # set up md start configuration
    images = read(args.init_traj, ':')
    start_indice = np.random.choice(len(images)) if args.start_indice == None else args.start_indice
    logger.debug(f'MD starts from No.{start_indice} configuration in {args.init_traj}')
    atoms = images[start_indice]
    atoms.wrap()
    cons = FixAtoms(mask=atoms.positions[:, 2] < args.fix_under) if args.fix_under else []
    atoms.set_constraint(cons)
    atoms.calc = encalc
    atoms.get_potential_energy()

    collect_traj = Trajectory('warning_struct.traj', 'w')
    @CallsCounter
    def printenergy(a=atoms):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()
        ensemble = a.calc.results['ensemble']
        energy_var = ensemble['energy_var']
        forces_var = np.mean(ensemble['forces_var'])
        forces_sd = np.mean(np.sqrt(ensemble['forces_var']))
        forces_l2_var = np.mean(ensemble['forces_l2_var'])

        if forces_sd > 0.2:
            logger.error("Too large uncertainty!")
            if logger.info.calls + logger.warning.calls > args.min_steps:
                sys.exit(0)
            else:
                sys.exit("Too large uncertainty!")
        elif forces_sd > 0.05:
            collect_traj.write(a)
            logger.warning("Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} energy_var={:10.6f} forces_var={:10.6f} forces_sd={:10.6f} forces_l2_var={:10.6f}".format(
                printenergy.calls * args.print_step,
                epot,
                ekin,
                temp,
                energy_var,
                forces_var,
                forces_sd,
                forces_l2_var,
            ))
            if logger.warning.calls > args.num_uncertain:
                logger.error(f"More than {args.num_uncertain} uncertain structures are collected!")
                if logger.info.calls + logger.warning.calls > args.min_steps:
                    sys.exit(0)
                else:
                    sys.exit(f"More than {args.num_uncertain} uncertain structures are collected!")
        else:
            logger.info("Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} energy_var={:10.6f} forces_var={:10.6f} forces_sd={:10.6f} forces_l2_var={:10.6f}".format(
                printenergy.calls * args.print_step,
                epot,
                ekin,
                temp,
                energy_var,
                forces_var,
                forces_sd,
                forces_l2_var,
            ))

    #atoms.calc = encalc
    if not np.any(atoms.get_momenta()):        
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    dyn = Langevin(atoms, args.time_step * units.fs, temperature_K=args.temperature, friction=0.1)
    dyn.attach(printenergy, interval=args.print_step)

    traj = Trajectory('MD.traj', 'w', atoms)
    dyn.attach(traj.write, interval=args.dump_step)
    dyn.run(args.max_steps)

if __name__ == "__main__":
    main()