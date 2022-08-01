import numpy as np
import math
import json, os, sys
import argparse
import logging
import itertools
import torch

from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Atomic interaction cutoff distance [�~E]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/qm9.db", help="Path to ASE trajectory",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(1e6),
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of molecules per minibatch",
    )

    parser.add_argument(
        "--initial_lr", type=float, default=0.0001, help="Initial learning rate",
    )
    parser.add_argument(
        "--forces_weight",
        type=float,
        default=0.99,
        help="Tradeoff between training on forces (weight=1) and energy (weight=0)",
    )
    
    return parser.parse_args(arg_list)

def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.10))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

    # Save split file
    with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
        json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits

def forces_criterion(predicted, target, reduction="mean"):
    # predicted, target are (bs, max_nodes, 3) tensors
    # node_count is (bs) tensor
    diff = predicted - target
    total_squared_norm = torch.linalg.norm(diff, dim=1)  # bs
    if reduction == "mean":
        scalar = torch.mean(total_squared_norm)
    elif reduction == "sum":
        scalar = torch.sum(total_squared_norm)
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")
    return scalar

def eval_model(model, dataloader, device, forces_weight):
    energy_running_ae = 0
    energy_running_se = 0

    forces_running_l2_ae = 0
    forces_running_l2_se = 0
    forces_running_c_ae = 0
    forces_running_c_se = 0
    forces_running_loss = 0

    running_loss = 0
    count = 0
    forces_count = 0
    criterion = torch.nn.MSELoss()

    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        out = model(device_batch)

        # counts
        count += batch["energy"].shape[0]
        forces_count += batch['forces'].shape[0]
        
        # use mean square loss here
        forces_loss = forces_criterion(out["forces"], device_batch["forces"]).item()
        energy_loss = criterion(out["energy"], device_batch["energy"]).item()  #problem here
        total_loss = forces_weight * forces_loss + (1 - forces_weight) * energy_loss
        running_loss += total_loss * batch["energy"].shape[0]
        
        # energy errors
        outputs = {key: val.detach().cpu().numpy() for key, val in out.items()}
        energy_targets = batch["energy"].detach().cpu().numpy()
        energy_running_ae += np.sum(np.abs(energy_targets - outputs["energy"]), axis=0)
        energy_running_se += np.sum(
            np.square(energy_targets - outputs["energy"]), axis=0
        )

        # force errors
        forces_targets = batch["forces"].detach().cpu().numpy()
        forces_diff = forces_targets - outputs["forces"]
        forces_l2_norm = np.sqrt(np.sum(np.square(forces_diff), axis=1))

        forces_running_c_ae += np.sum(np.abs(forces_diff))
        forces_running_c_se += np.sum(np.square(forces_diff))

        forces_running_l2_ae += np.sum(np.abs(forces_l2_norm))
        forces_running_l2_se += np.sum(np.square(forces_l2_norm))

    energy_mae = energy_running_ae / count
    energy_rmse = np.sqrt(energy_running_se / count)

    forces_l2_mae = forces_running_l2_ae / forces_count
    forces_l2_rmse = np.sqrt(forces_running_l2_se / forces_count)

    forces_c_mae = forces_running_c_ae / (forces_count * 3)
    forces_c_rmse = np.sqrt(forces_running_c_se / (forces_count * 3))

    total_loss = running_loss / count

    evaluation = {
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "forces_l2_mae": forces_l2_mae,
        "forces_l2_rmse": forces_l2_rmse,
        "forces_mae": forces_c_mae,
        "forces_rmse": forces_c_rmse,
        "sqrt(total_loss)": np.sqrt(total_loss),
    }

    return evaluation

def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Create device
    device = torch.device(args.device)
    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Setup dataset and loader
    logging.info("loading data %s", args.dataset)
    dataset = AseDataset(
        args.dataset,
        cutoff = args.cutoff,
    )
    
    datasplits = split_data(dataset, args)

    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        args.batch_size,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=collate_atomsdata,
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"], 
        args.batch_size, 
        collate_fn=collate_atomsdata,
    )
    
    net = PainnModel(
        num_interactions=args.num_interactions, 
        hidden_state_size=args.node_size, 
        cutoff=args.cutoff,
    )
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr)
    criterion = torch.nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

    log_interval = 10000
    running_loss = 0
    running_loss_count = 0
    best_val_loss = np.inf
    step = 0    

    if args.load_model:
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        step = state_dict["step"]
        best_val_loss = state_dict["best_val_loss"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
    
    mem_allo_1, mem_allo_2, mem_allo_3, mem_cache = [], [], [], []
    for epoch in itertools.count():
        for batch_host in train_loader:
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            mem_allo_1.append(torch.cuda.memory_allocated())
            # Reset gradient
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = net(
                batch, compute_forces=bool(args.forces_weight)
            )
            energy_loss = criterion(outputs["energy"], batch["energy"])
            if args.forces_weight:
                forces_loss = forces_criterion(outputs['forces'], batch['forces'])
            else:
                forces_loss = 0.0
            total_loss = (
                args.forces_weight * forces_loss
                + (1 - args.forces_weight) * energy_loss
            )
            mem_allo_2.append(torch.cuda.memory_allocated())
            total_loss.backward()
            optimizer.step()
            mem_allo_3.append(torch.cuda.memory_allocated())
            mem_cache.append(torch.cuda.memory_reserved())
            running_loss += total_loss.item() * batch["energy"].shape[0]
            running_loss_count += batch["energy"].shape[0]
            
            mem_check = np.vstack((mem_allo_1, mem_allo_2, mem_allo_3, mem_cache)).T
            np.save('mem_stat.npy', mem_check)
            # print(step, loss_value)
            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                train_loss = running_loss / running_loss_count
                running_loss = running_loss_count = 0

                eval_dict = eval_model(net, val_loader, device, args.forces_weight)
                eval_formatted = ", ".join(
                    ["%s=%g" % (k, v) for (k, v) in eval_dict.items()]
                )

                logging.info(
                    "step=%d, %s, sqrt(train_loss)=%g, max memory used=%g",
                    step,
                    eval_formatted,
                    math.sqrt(train_loss),
                    torch.cuda.max_memory_allocated() / 2**20,
                )
               
                # Save checkpoint
                if eval_dict["sqrt(total_loss)"] < best_val_loss:
                    best_val_loss = eval_dict["sqrt(total_loss)"]
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
            step += 1

            scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                torch.save(
                    {
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(args.output_dir, "exit_model.pth"),
                )
                sys.exit(0)

if __name__ == "__main__":
    main()
