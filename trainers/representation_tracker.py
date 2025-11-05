import torch
import torch.nn 
from typing import Optional, Dict, Literal
from trainers.distances import (
    mse_potential,
    wasserstein_distance,
    mmd_distance
)

use_cuda = torch.cuda.is_available()

class RepresentationTracker:
    """
    Tracks representation statistics for inter-cycle repulsion using configurable distance metrics.
    Supports multiple distance metrics for repulsion between cycle representations, including:
        - Mean Squared Error (MSE)
        - Wasserstein distance
        - Maximum Mean Discrepancy (MMD)
    The distance metric can be selected via the `distance` parameter.
    """
    
    def __init__(
        self,
        device: torch.device,
        num_ref_samples: int = 128,
        regularization_strength: float = 1e-6,
        batch_size: int = 1,
        distance: Literal['mse', 'wasserstein', 'mmd'] = 'mse'
    ):
        self.device = device
        self.num_ref_samples = num_ref_samples
        self.regularization_strength = regularization_strength
        self.reference_samples = None
        self.batch_size = batch_size
        self.cycle_representations = {}
        self.ordered_cycle_keys = []
        self.repulse_n_cycles = 1  # Number of past cycles to repulse from
        self.distance = distance
        
    def initialize_reference_samples(self, data_loader):
        """Initialize reference samples from training data."""
        print("==> Initializing reference samples for representation tracking...")
        ref_samples = []
        
        for batch in data_loader:
            if isinstance(batch, dict):
                x = batch["img"]
            else:
                x = batch[0]
                
            ref_samples.append(x)
            if len(torch.cat(ref_samples, dim=0)) >= self.num_ref_samples:
                break
        
        self.reference_samples = torch.cat(ref_samples, dim=0)[:self.num_ref_samples].to(torch.half)
            
        if use_cuda:
            self.reference_samples = self.reference_samples.cuda()
        
        print(f"Reference samples initialized: {self.reference_samples.shape}")

    def extract_representation(self, net: torch.nn.Module) -> Optional[torch.Tensor]:
        """Extract representation using reference samples and registered hooks."""
        if self.reference_samples is None:
            return None
        return net(self.reference_samples, return_text_features=True)[1]

    def update_cycle_representation(self, net: torch.nn.Module, cycle_num: int):
        """Update the representation for the current cycle."""
        current_repr = self.extract_representation(net)
        if current_repr is not None:
            self.cycle_representations[cycle_num] = current_repr.clone()
            self.ordered_cycle_keys.append(cycle_num)
            print(f"Updated representation for cycle {cycle_num}: {current_repr.shape}")

    def compute_repulsion_gradients(
        self,
        net: torch.nn.Module,
        current_cycle: int,
        repulsion_strength: float
    ) -> Dict[torch.nn.parameter.Parameter, torch.Tensor]:
        """Compute repulsive gradients using probability metrics."""
        if repulsion_strength==0 or current_cycle < 0 or self.reference_samples is None: 
            return {}
        if current_cycle > 0 and not self.cycle_representations:
            raise ValueError(f"No past cycle representations to compute repulsion for cycle {current_cycle}, cycle_representations: {self.cycle_representations.keys()}")
        
        # Get most recent past cycle
        most_recent_cycle = self.ordered_cycle_keys[-1] # get latest cycle key 
        past_repr = self.cycle_representations[most_recent_cycle].detach()  # Detach past representation
        
        current_repr = self.extract_representation(net)
        if current_repr is None:
            return {}

        if self.batch_size < len(self.reference_samples) and self.batch_size != -1:
            # Use all reference samples if batch_size == -1
            randperm = torch.randperm(current_repr.size(0))
            current_repr = current_repr[randperm][:self.batch_size]
            past_repr = past_repr[randperm][:self.batch_size]
        force_potential = self.compute_repulsion_matrix(current_repr, past_repr, repulsion_strength)
        
        grad_params = [p for p in net.parameters() if p.requires_grad]
        
        param_grads = torch.autograd.grad(
            outputs=force_potential,
            inputs=grad_params,
            grad_outputs=None,
            allow_unused=True,
            retain_graph=False
        )
        
        grad_dict = {}
        for param, grad in zip(grad_params, param_grads):
            if grad is None:
                print(f"Warning: Computed gradient is None for a parameter. See: {force_potential=}, {grad_params=}")
                continue

            if torch.isnan(grad).any():
                print(f"Warning: Computed gradient contains NaNs for a parameter. See: {force_potential=}")
                for p in grad_params:
                    if torch.isnan(p).any():
                        print("Got at least one parameter with NaNs.")
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print("Got at least one parameter with NanGrad.")
                continue

            if torch.isinf(grad).any():
                print(f"Warning: Computed gradient contains Infs for a parameter. See: {force_potential=}")

                for p in grad_params:
                    if torch.isinf(p).any():
                        print("Got at least one parameter with Infs.")
                    if p.grad is not None and torch.isinf(p.grad).any():
                        print("Got at least one parameter with InfGrad.")

                continue

            grad_dict[param] = grad
        return grad_dict
            
    def compute_repulsion_matrix(self, current_repr, past_repr, repulsion_strength):
        """
        Compute the scalar repulsion potential between two representations using the selected distance metric.
        Args:
            current_repr (torch.Tensor): The current representation tensor, typically of shape [batch_size, feature_dim].
            past_repr (torch.Tensor): The past representation tensor, typically of shape [batch_size, feature_dim].
            repulsion_strength (float): Scalar factor to scale the repulsion potential.
        Behavior:
            The method selects the distance metric based on self.distance ('mse', 'wasserstein', or 'mmd').
            It computes the reciprocal of the distance (with a small epsilon for stability) as the repulsion potential.
        Returns:
            torch.Tensor: A scalar tensor representing the repulsion potential between the two representations.
        """
        eps = 1e-6

        print(current_repr.shape, past_repr.shape)
        
        if self.distance == 'mse':
            dist = mse_potential(current_repr, past_repr).mean()
        elif self.distance == 'wasserstein':
            dist = wasserstein_distance(current_repr, past_repr)
        elif self.distance == 'mmd':
            dist = mmd_distance(current_repr, past_repr)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        potential = torch.reciprocal(dist + eps)

        # Simple: force proportional to difference (like springs)
        return repulsion_strength * potential  # Pushes away proportionally
    
        # if current_repr.shape != past_repr.shape:
        #     min_samples = min(current_repr.shape[0], past_repr.shape[0])
        #     min_features = min(current_repr.shape[1], past_repr.shape[1])
        #     current_repr = current_repr[:min_samples, :min_features]
        #     past_repr = past_repr[:min_samples, :min_features]
        
        # # Detached SVD computation for stability
        # with torch.no_grad():
        #     # Center matrices
        #     current_centered = (current_repr - current_repr.mean(dim=0, keepdim=True)).detach()
        #     past_centered = (past_repr - past_repr.mean(dim=0, keepdim=True)).detach()
            
        #     # Normalize
        #     current_norm = torch.norm(current_centered, 'fro')
        #     past_norm = torch.norm(past_centered, 'fro')
            
        #     if current_norm > 1e-6 and past_norm > 1e-6:
        #         current_normalized = current_centered / current_norm
        #         past_normalized = past_centered / past_norm
        #     else:
        #         raise ValueError("Degenerate matrices")
            
        #     # # Cross-covariance with regularization
        #     # H = current_normalized.T @ past_normalized
        #     # H_reg = H + self.regularization_strength * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            
        #     # H_reg_float = H_reg.float()
        #     # U_float, S_float, Vt_float = torch.linalg.svd(H_reg_float)
        #     # R_float = Vt_float.T @ U_float.T

        #     # R = R_float.to(H.dtype)
            
        #     # if torch.det(R.float()) < 0: 
        #     #     Vt = Vt_float.to(H.dtype) 
        #     #     Vt[-1, :] *= -1
        #     #     R = (Vt.T @ U_float.to(H.dtype).T)
                        
        # current_centered_grad = current_repr - current_repr.mean(dim=0, keepdim=True)
        # past_centered_grad = past_repr - past_repr.mean(dim=0, keepdim=True)
        
        # current_normalized_grad = current_centered_grad
        # past_normalized_grad = past_centered_grad
        
        # # Apply rotation and compute force
        # #R_detached = R.detach()
        # current_aligned = current_normalized_grad # @ R_detached
        # diff_matrix = current_aligned - past_normalized_grad.detach()
        
        # dist_sq = torch.sum(diff_matrix.pow(2))
        # force_magnitude = repulsion_strength / (dist_sq + 1e-8)
        
        # aligned_force = force_magnitude * diff_matrix
        # force_matrix = aligned_force @ R_detached.T
        
        # return force_matrix
    
  