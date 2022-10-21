from ase.calculators.vasp import Vasp
from ase.io import read, write, Trajectory
from shutil import copy
import os, subprocess
import numpy as np
import argparse
import json
import toml
from pathlib import Path

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--label_set",
        type=str,
        help="Path to trajectory to be labeled by DFT",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        help="Path to existing training data set",
    )
    parser.add_argument(
        "--pool_set", 
        type=str, 
        help="Path to MD trajectory obtained from machine learning potential",
    )
    parser.add_argument(
        "--al_info", 
        type=str, 
        help="Path to json file that stores indices selected in pool set",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        help="Number of DFT jobs",
    )
    parser.add_argument(
        "--job_order",
        type=int,
        help="Split DFT jobs to several different parts",
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
        if not isinstance(v, dict):
            ns.__dict__[k] = v

def main():
    # set environment variables
    os.putenv('ASE_VASP_VDW', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
    os.putenv('VASP_PP_PATH', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
    os.putenv('ASE_VASP_COMMAND', 'mpirun vasp_std')

    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)

    # get images and set parameters
    if args.label_set:
        images = read(args.label_set, index = ':')
    elif args.pool_set:
        if isinstance(args.pool_set, list):
            pool_traj = []
            for pool_path  in args.pool_set:
                if Path(pool_path).stat().st_size > 0:
                    pool_traj += read(pool_path, ':')
        else:
            pool_traj = Trajectory(args.pool_set)
        with open(args.al_info) as f:
            indices = json.load(f)["selected"]
        if args.num_jobs:
            split_idx = np.array_split(indices, args.num_jobs)
            indices = split_idx[args.job_order]
        images = [pool_traj[i] for i in indices]        
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')

    vasp_params = params['VASP']
    calc = Vasp(**vasp_params)
    traj = Trajectory('dft_structures.traj', mode = 'a')
    check_result = False
    unconverged = Trajectory('unconverged.traj', mode = 'a')
    unconverged_idx = []
    for i, atoms in enumerate(images):
        atoms.set_pbc([True,True,True])
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        steps = int(subprocess.getoutput('grep LOOP OUTCAR | wc -l'))
        if steps <= vasp_params['nelm']:
            traj.write(atoms)
        else:
            check_result = True
            unconverged.write(atoms)
            unconverged_idx.append(i)
        copy('OSZICAR', 'OSZICAR_{}'.format(i))

    traj.close()
    # write to training set
    if check_result:
        raise RuntimeError(f"DFT calculations of {unconverged_idx} are not converged!")

    if args.train_set:
        train_traj = Trajectory(args.train_set, mode = 'a')
        images = read('dft_structures.traj', ':')
        for atoms in images:
            atoms.info['system'] = args.system
            atoms.info['path'] = str(Path('dft_structures.traj').resolve())
            train_traj.write(atoms)

    os.remove('WAVECAR')

if __name__ == "__main__":
    main()
