# <div align="center">PaiNN-model introduction</div>
This is a simple implementation of [PaiNN](https://arxiv.org/abs/2102.03150) model and active learning workflow for fitting interatomic potentials.  
The learned features or gradients in the model are used for active learning. Several selection methods are implemented.  
All the active learning codes are to be tested.
## <div align="center">Documentation</div>
No documentation yet.

## <div align="center">Quick Start</div>
<details open>
<summary>How to install</summary>

This code is only tested on [**Python>=3.8.0**](https://www.python.org/) and [**PyTorch>=1.10**](https://pytorch.org/get-started/locally/).  
Requirements: [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)(if you want to use active learning),
[toml](https://toml.io/en/), [myqueue](https://myqueue.readthedocs.io/en/latest/installation.html)(if you want to submit jobs automatically).

```bash
$ conda install pytorch-scatter -c pyg
$ conda install -c conda-forge toml
$ python3 -m pip install myqueue
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
$ git clone https://github.com/Yangxinsix/PaiNN-model.git
$ cd PaiNN-model
$ python -m pip install -U .
```

</details>

<details open>
<summary>How to use</summary>

* See `train.py` in `scripts` for training, and `md_run.py` for running MD simulations by using ASE.
* See `al_select.py` for active learning.
* See `flow.py` for distributing and submitting active learning jobs.

</details>
