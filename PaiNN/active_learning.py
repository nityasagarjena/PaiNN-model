import torch
from torch import nn
from collections import defaultdict
from torch_scatter import scatter_mean
from typing import List, Dict, Tuple, Optional
from PaiNN.data import collate_atomsdata
from PaiNN.select import *
from PaiNN.kernel import *

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._features = []
        self._grads = []
        self.hooks = []
        for name, layer in self.model.named_modules():
            if 'readout_mlp' in name and isinstance(layer, nn.Linear):
                self.hooks.append(layer.register_forward_pre_hook(self.save_feats_hook))
                self.hooks.append(layer.register_backward_hook(self.save_grads_hook))
        
    def save_feats_hook(self, _, in_feat):
        new_feat = torch.cat((in_feat[0].detach().clone(), torch.ones_like(in_feat[0][:, 0:1])), dim=-1)
        self._features.append(new_feat)
    
    def save_grads_hook(self, _, __, grad_output):
        self._grads.append(grad_output[0].detach().clone())
    
    def unhook(self):
        for hook in self.hooks:
            hook.remove()
    
    def forward(self, model_inputs: Dict[str, torch.Tensor]):
        self._features = []
        self._grads = []
        _ = self.model(model_inputs)                            
        return self._features, self._grads[::-1]

class RandomProjections:
    """Store parameters of random projections"""
    def __init__(self, model: nn.Module, num_features: int):
        self.num_features = num_features
        if self.num_features > 0:
            self.in_feat_proj = [
                torch.randn(l.in_features +1, num_features, device=next(model.parameters()).device)
                for l in model.readout_mlp.children() if isinstance(l, nn.Linear)
            ]
            self.out_grad_proj = [
                torch.randn(l.out_features, num_features, device=next(model.parameters()).device)
                for l in model.readout_mlp.children() if isinstance(l, nn.Linear)
            ]
    
class FeatureStatistics:
    """
    Generate features by giving models, pool, and training dataset
    """
    def __init__(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        random_projections: List[RandomProjections],
        batch_size: int=8,
    ):
        self.models = models
        self.batch_size = batch_size
        self.dataset = dataset
        self.random_projections = random_projections
        self.device = next(models[0].parameters()).device
        self.g = None
        self.ens_stats = None
        self.Fisher = None
        self.F_reg_inv = None
    
    def _compute_ens_stats(self, model_inputs: Dict[str, torch.Tensor], labeled_data: bool=False) -> Dict[str, torch.Tensor]:
        """Compute energy variance, forces variance, energy absolute error, and forces absolute error"""
        ens_stats = defaultdict(list)
        predictions = defaultdict(list)
        for model in self.models:
            model_results = model(model_inputs)
            predictions['energy'].append(model_results["energy"].detach())
            predictions['forces'].append(model_results["forces"].detach())
        
        predictions = {k: torch.stack(v) for k, v in predictions.items()}
        
        image_idx = torch.arange(
            model_inputs['num_atoms'].shape[0],
            device=model_inputs['num_atoms'].device,                                   
        )
        image_idx = torch.repeat_interleave(image_idx, model_inputs['num_atoms'])

        if len(self.models) > 1:
            E_var = torch.var(predictions['energy'], dim=0)
            F_var = torch.var(predictions['forces'], dim=0)
            F_var = scatter_mean(torch.mean(F_var, dim=-1), image_idx, dim=0)          
            ens_stats['Energy-Var'] = E_var
            ens_stats['Forces-Var'] = F_var

        if labeled_data:
            E_AE = torch.abs(model_inputs['energy'] - torch.mean(predictions['energy'], dim=0))
            F_AE = torch.abs(model_inputs['forces'] - torch.mean(predictions['forces'], dim=0))
            F_AE = scatter_mean(torch.mean(F_AE, dim=-1), image_idx, dim=0)
            
            ens_stats['Energy-AE'] = E_AE
            ens_stats['Forces-AE'] = F_AE
        
        return ens_stats
                
    def _compute_features(
        self, 
        feature_extractor: FeatureExtractor, 
        model_inputs: torch.tensor,
        random_projection: RandomProjections,
        kernel: str='ll-gradient',
    ) -> torch.Tensor:
        """
        Implementing features calculation and kernel transformation. 
        Available features are:
        ll-gradient: last layer gradient feature, obtained from neural networks.
        full-gradient: All gradient information from NN, must use random projections kernel transformation.
        gnn: Features learned by message passing layers
        symmetry-function: Behler Parrinello symmetry function, can only be used for CUR. To be implemented.
        """
        image_idx = torch.arange(
            model_inputs['num_atoms'].shape[0],
            device=model_inputs['num_atoms'].device,                                   
        )
        image_idx = torch.repeat_interleave(image_idx, model_inputs['num_atoms'])
        
        if kernel == 'full-gradient':
            assert random_projection.num_features != 0, "Error! Random projections must be provided!"
            feats, grads = feature_extractor(model_inputs)
            atomic_g = torch.zeros((image_idx.shape[0], random_projection.num_features))
            for feat, grad, in_proj, out_proj in zip(
                feats, 
                grads, 
                random_projection.in_feat_proj,
                random_projection.out_grad_proj,
            ):
                atomic_g = (feat @ in_proj) * (grad @ out_proj)

            g = torch.zeros(
                (model_inputs['num_atoms'].shape[0], atomic_g.shape[1]),
                dtype = atomic_g.dtype,
                device = atomic_g.device,
            ).index_add(0, image_idx, atomic_g)
        elif kernel == 'local_full-g':
            assert random_projection.num_features != 0, "Error! Random projections must be provided!"
            feats, grads = feature_extractor(model_inputs)
            atomic_g = torch.zeros((image_idx.shape[0], random_projection.num_features))
            for feat, grad, in_proj, out_proj in zip(
                feats, 
                grads, 
                random_projection.in_feat_proj,
                random_projection.out_grad_proj,
            ):
                atomic_g = (feat @ in_proj) * (grad @ out_proj)
            g = atomic_g
        
        elif kernel == 'll-gradient':
            feats, grads = feature_extractor(model_inputs)
            if random_projection.num_features != 0:
                atomic_g = (feats[-1] @ random_projection.in_feat_proj[-1]) *\
                           (grads[-1] @ random_projection.out_grad_proj[-1])
            else:
                atomic_g = feats[-1][:, :-1]

            g = torch.zeros(
                (model_inputs['num_atoms'].shape[0], atomic_g.shape[1]),
                dtype = atomic_g.dtype,
                device = atomic_g.device,
            ).index_add(0, image_idx, atomic_g)

        elif kernel == 'local_ll-g':
            feats, grads = feature_extractor(model_inputs)
            if random_projection.num_features != 0:
                atomic_g = (feats[-1] @ random_projection.in_feat_proj[-1]) *\
                           (grads[-1] @ random_projection.out_grad_proj[-1])
            else:
                atomic_g = feats[-1][:, :-1]
            g = atomic_g
        
        elif kernel == 'gnn':
            feats, grads = feature_extractor(model_inputs)
            if random_projection.num_features != 0:
                atomic_g = (feats[0] @ random_projection.in_feat_proj[0]) *\
                           (grads[0] @ random_projection.out_grad_proj[0])
            else:
                atomic_g = feats[0][:, :-1]
                
            g = torch.zeros(
                (model_inputs['num_atoms'].shape[0], atomic_g.shape[1]),
                dtype = atomic_g.dtype,
                device = atomic_g.device,
            ).index_add(0, image_idx, atomic_g)

        elif kernel == 'local_gnn':
            feats, grads = feature_extractor(model_inputs)
            if random_projection.num_features != 0:
                atomic_g = (feats[0] @ random_projection.in_feat_proj[0]) *\
                           (grads[0] @ random_projection.out_grad_proj[0])
            else:
                atomic_g = feats[0][:, :-1]
            g = atomic_g
        
        return g
    
    def _compute_fisher(self, g: torch.Tensor) -> torch.Tensor:
        return torch.einsum('mci, mcj -> mij', g, g)
                                                                                               
    def get_features(
        self, 
        dataset: Optional[torch.utils.data.Dataset]=None, 
        kernel: str='full-gradient',
    ) -> torch.Tensor:
        """
        :return: Feature vector of ``shape=(n_models, n_structures, n_features)``.
        """
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self.g = None

        if self.g == None:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=collate_atomsdata,
            )
            global_g = []
            for i, model in enumerate(self.models):
                feat_extract = FeatureExtractor(model)
                model_g = []
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    model_g.append(self._compute_features(
                        feat_extract, 
                        batch, 
                        kernel=kernel, 
                        random_projection=self.random_projections[i],
                    ))
                feat_extract.unhook()
                model_g = torch.cat(model_g)
                # Normalization
                model_g = (model_g - torch.mean(model_g, dim=0)) / torch.var(model_g, dim=0)
                global_g.append(model_g)
#                global_g.append(torch.cat(model_g))
                
            self.g = torch.stack(global_g)
                
        return self.g

    def get_num_atoms(
        self,
        dataset: Optional[torch.utils.data.Dataset]=None,
    ):
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
        num_atoms = []
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=collate_atomsdata,
        )
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            num_atoms.append(batch['num_atoms'])
        
        return torch.cat(num_atoms)
            
    def get_ens_stats(self, dataset: Optional[torch.utils.data.Dataset]=None) -> Dict[str, torch.Tensor]:
        """
        :return: Dict of energy statistics
        """
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self.ens_stats = None
            
        if self.ens_stats == None:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=collate_atomsdata,
            )
            ens_stats = []
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labeled_data = True if 'energy' in batch.keys() else False
                ens_stats.append(self._compute_ens_stats(batch, labeled_data))

            self.ens_stats = {k: torch.cat([ens[k] for ens in ens_stats]) for k in ens_stats[0].keys()}
            
        return self.ens_stats
    
    def get_fisher(self) -> torch.Tensor:
        if self.Fisher is None:
            self.Fisher = self._compute_fisher(self.get_features())
        return self.Fisher

    def get_F_inv(self) -> torch.Tensor:
        """
        :return: Regularized inverse of Fisher matrix of "shape=(n_models, n_features, n_features)".
        """
        if self.F_reg_inv is None:
            F = self.get_features()
            g = self.get_g()
            # empirical regularisation
            lam = torch.linalg.trace(F) / (g.shape[1] * g.shape[2])
            self.F_train_reg_inv = torch.linalg.inv(F + lam * torch.eye(F.shape[1]))
        return self.F_train_reg_inv

class GeneralActiveLearning:
    """Provides methods for selecting batches during active learning.

    :param kernel: Name of the kernel, e.g. "full-g", "ll-g", "full-F_inv", "ll-F_inv", "qbc-energy", "qbc-force".
                   "random" produces random selection and "ae-energy" and "ae-force" select by absolute errors
                   on the pool data, which is only possible if the pool data is already labeled.
    :param selection: Selection method, one of "max_dist_greedy", "deterministic_CUR", "lcmd_greedy", "max_det_greedy" or "max_diag".
    :param n_random_features: If "n_random_features = 0", do not use random projections.
                              Otherwise, use random projections of all linear-layer gradients.
    """
    def __init__(
        self,
        kernel = 'full-g',
        selection = 'max_diag',
        n_random_features = 0,
    ):
        self.kernel = kernel
        self.selection = selection
        self.n_random_features = n_random_features
    
    def select(
        self, 
        models: List[nn.Module], 
        datasets: Dict[str, torch.utils.data.Dataset], 
        batch_size: int = 8, 
        al_batch_size: int = 100,
    ):
        """
        models: pytorch models,
        dataset: a dictionary containing pool, train, and validation dataset,
        batch_size: batch size for extracting features,
        al_batch_size: active learning selection batch size
        """        
        if (self.kernel == 'qbc-energy' or self.kernel == 'qbc-force' or self.kernel == 'ae-energy' or
            self.kernel == 'ae-force' or self.kernel == 'random') and self.selection != 'max_diag':
            raise RuntimeError(f'{self.kernel} kernel can only be used with max_diag selection method,'
                               f' not with {self.selection}!')
        random_projections = [RandomProjections(model, self.n_random_features) for model in models]
        
        stats = {
            key: FeatureStatistics(models, ds, random_projections, batch_size)
            for key, ds in datasets.items()
        }
        
        if self.selection == 'max_dist_greedy':
            matrix = self._get_kernel_matrix(stats['pool'], stats['train'])
            idxs = max_dist_greedy(matrix=matrix, batch_size=al_batch_size, n_train=len(datasets['train']))
        elif self.selection == 'max_diag':
            matrix = self._get_kernel_matrix(stats['pool'])
            idxs = max_diag(matrix=matrix, batch_size=al_batch_size)
        elif self.selection == 'max_det_greedy':
            matrix = self._get_kernel_matrix(stats['pool'])
            idxs = max_det_greedy(matrix=matrix, batch_size=al_batch_size)
        elif self.selection == 'lcmd_greedy':
            matrix = self._get_kernel_matrix(stats['pool'], stats['train'])
            idxs = lcmd_greedy(matrix=matrix, batch_size=al_batch_size, n_train=len(datasets['train']))
        elif self.selection == 'max_det_greedy_local':
            matrix, num_atoms = self._get_kernel_matrix(stats['pool'])
            idxs = max_det_greedy_local(matrix=matrix, batch_size=al_batch_size, num_atoms=num_atoms)
        else:
            raise NotImplementedError(f"Unknown selection method '{self.selection}' for active learning!")
            
        return idxs.cpu().tolist()

    
    def _get_kernel_matrix(self, pool_stats: FeatureStatistics, train_stats: Optional[FeatureStatistics]=None) -> KernelMatrix:
        stats_list = [pool_stats] if train_stats == None else [pool_stats, train_stats]
        
        if self.kernel == 'full-g':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='full-gradient') for s in stats_list], dim=1))
        elif self.kernel == 'll-g':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='ll-gradient') for s in stats_list], dim=1))
        elif self.kernel == 'gnn':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='gnn') for s in stats_list], dim=1))
        elif self.kernel == 'local_full-g':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_full-g') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms
        elif self.kernel == 'local_ll-g':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_ll-g') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms 
        elif self.kernel == 'local_gnn':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_gnn') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms 
        elif self.kernel == 'full-F_inv':
            return FeatureCovKernelMatrix(torch.cat([s.get_features(kernel='full-gradient') for s in stats_list], dim=1),
                                          train_stats.get_F_reg_inv())
        elif self.kernel == 'll-F_inv':
            return FeatureCovKernelMatrix(torch.cat([s.get_features(kernel='ll-gradient') for s in stats_list], dim=1),
                                          train_stats.get_F_reg_inv())
        elif self.kernel == 'qbc-energy':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Energy-Var'])
        elif self.kernel == 'qbc-force':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Forces-Var'])
        elif self.kernel == 'ae-energy':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Energy-AE'])
        elif self.kernel == 'ae-force':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Forces-AE'])
        elif self.kernel == 'random':
            return DiagonalKernelMatrix(torch.rand([sum([len(s.dataset) for s in stats_list])]))
        else:
            raise RuntimeError(f"Unknown active learning kernel {self.kernel}!")