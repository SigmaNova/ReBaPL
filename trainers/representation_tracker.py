import torch 

use_cuda = torch.cuda.is_available()

class RepresentationTracker:
    """Tracks representation statistics for inter-cycle repulsion using Procrustes distance."""
    
    def __init__(self, device, num_ref_samples=128, regularization_strength=1e-6):
        self.device = device
        self.num_ref_samples = num_ref_samples
        self.regularization_strength = regularization_strength
        self.reference_samples = None
        
        # Track representation statistics per cycle
        self.cycle_representations = {}
        
        # Hook storage for feature extraction
        self.feature_hooks = {}
        self.extracted_features = {}
        self.hooks_registered = False
        
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
        
        self.reference_samples = torch.cat(ref_samples, dim=0)[:self.num_ref_samples]
            
        if use_cuda:
            self.reference_samples = self.reference_samples.cuda()
        
        print(f"Reference samples initialized: {self.reference_samples.shape}")
    
    def register_hooks(self, net):
        """Register forward hooks to extract intermediate representations."""
        if self.hooks_registered:
            return
        
        self.extracted_features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Flatten and store features
                    flat_output = output.reshape(output.size(0), -1)
                    self.extracted_features[name] = flat_output
            return hook
        
        # Fix 2: Use more generic layer names that work with CLIP
        target_layers = ['text_encoder']
        hooks_registered = 0
        
        for name, module in net.named_modules():
            if name in target_layers:
                try:
                    handle = module.register_forward_hook(hook_fn(name))
                    self.feature_hooks[name] = handle
                    hooks_registered += 1
                    print(f"Registered hook for layer: {name}")
                except Exception as e:
                    print(f"Failed to register hook for {name}: {e}")
        
        if hooks_registered == 0:
            print("Warning: No hooks registered. Using fallback method.")
        
        self.hooks_registered = True
    
    def extract_representation(self, net):
        """Extract representation using reference samples and registered hooks."""
        if self.reference_samples is None:
            return None
        
        # ... (code for preparing reference_samples)
        
        self.register_hooks(net)
        self.extracted_features = {}
        
        net.eval()
        try:
            dummy_labels = torch.zeros(self.reference_samples.shape[0], dtype=torch.long, device=self.reference_samples.device)
            _ = net(self.reference_samples, dummy_labels) # This triggers the hook
            
            # **FIX: Select the specific feature we want instead of concatenating**
            if 'text_encoder' in self.extracted_features:
                representation = self.extracted_features['text_encoder']
            else:
                # Fallback to direct image encoding if text_encoder hook failed
                with torch.no_grad():
                    representation = net.image_encoder(self.reference_samples)

        except Exception as e:
            print(f"Error in representation extraction: {e}")
            # Fallback representation
            representation = torch.randn(self.reference_samples.shape[0], 512, device=self.device)
        
        net.train()
        return representation
    
    def update_cycle_representation(self, net, cycle_num):
        """Update the representation for the current cycle."""
        try:
            current_repr = self.extract_representation(net)
            if current_repr is not None:
                self.cycle_representations[cycle_num] = current_repr.clone()
                print(f"Updated representation for cycle {cycle_num}: {current_repr.shape}")
        except Exception as e:
            print(f"Failed to update cycle representation: {e}")
    
    def compute_procrustes_repulsion_gradients(self, net, current_cycle, repulsion_strength):
        """Compute repulsive gradients using Procrustes distance."""
        if repulsion_strength <= 0 or current_cycle == 0 or not self.cycle_representations:
            return {}
        
        # Get most recent past cycle
        past_cycles = [c for c in self.cycle_representations.keys() if c <= current_cycle]
        if not past_cycles:
            return {}
        
        most_recent_cycle = max(past_cycles)
        past_repr = self.cycle_representations[most_recent_cycle]
        
        # Get current representation with gradients enabled
        self.register_hooks(net)
        self.extracted_features = {}
        
        current_repr = self.extract_representation(net)
        # # Forward pass with gradients
        # if hasattr(net, 'image_encoder'):
        #     current_repr = net.image_encoder(self.reference_samples)
        # else:
        #     current_repr = net(self.reference_samples)
        
        # if self.extracted_features:
        #     all_features = [self.extracted_features[name] 
        #                   for name in sorted(self.extracted_features.keys())]
        #     if all_features:
        #         current_repr = torch.cat(all_features, dim=1)
        
        # Compute Procrustes-based repulsive force
        force_matrix = self._compute_procrustes_force(current_repr, past_repr, repulsion_strength)
        mean_force = force_matrix.mean(dim=0)
        
        # Clamp force magnitude to prevent instability
        force_norm = torch.norm(mean_force)
        if force_norm > 10.0:
            mean_force = mean_force * (10.0 / force_norm)
        
        # Compute gradients w.r.t. network parameters
        current_mean = current_repr.mean(dim=0)
        
        # Fix 4: Filter parameters that require gradients
        grad_params = [p for p in net.parameters() if p.requires_grad]
        
        param_grads = torch.autograd.grad(
            outputs=current_mean,
            inputs=grad_params,
            grad_outputs=mean_force,
            allow_unused=True,
            retain_graph=False
        )
        
        # Create gradient dictionary
        grad_dict = {}
        for param, grad in zip(grad_params, param_grads):
            if grad is not None and not (torch.isnan(grad).any() or torch.isinf(grad).any()):
                grad_dict[param] = grad
        return grad_dict
            
    
    
    def _compute_procrustes_force(self, current_repr, past_repr, repulsion_strength):
        """Compute repulsive force based on Procrustes distance."""
        # Ensure same shape
        if current_repr.shape != past_repr.shape:
            min_samples = min(current_repr.shape[0], past_repr.shape[0])
            min_features = min(current_repr.shape[1], past_repr.shape[1])
            current_repr = current_repr[:min_samples, :min_features]
            past_repr = past_repr[:min_samples, :min_features]
        
        try:
            # Detached SVD computation for stability
            with torch.no_grad():
                # Center matrices
                current_centered = (current_repr - current_repr.mean(dim=0, keepdim=True)).detach()
                past_centered = (past_repr - past_repr.mean(dim=0, keepdim=True)).detach()
                
                # Normalize
                current_norm = torch.norm(current_centered, 'fro')
                past_norm = torch.norm(past_centered, 'fro')
                
                if current_norm > 1e-6 and past_norm > 1e-6:
                    current_normalized = current_centered / current_norm
                    past_normalized = past_centered / past_norm
                else:
                    raise ValueError("Degenerate matrices")
                
                # Cross-covariance with regularization
                H = current_normalized.T @ past_normalized
                H_reg = H + self.regularization_strength * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
                
                H_reg_float = H_reg.float()
                U_float, S_float, Vt_float = torch.linalg.svd(H_reg_float)
                R_float = Vt_float.T @ U_float.T

                R = R_float.to(H.dtype)
                
                if torch.det(R.float()) < 0: 
                    Vt = Vt_float.to(H.dtype) 
                    Vt[-1, :] *= -1
                    R = (Vt.T @ U_float.to(H.dtype).T)
                            
            # Force computation with gradients
            current_centered_grad = current_repr - current_repr.mean(dim=0, keepdim=True)
            past_centered_grad = past_repr - past_repr.mean(dim=0, keepdim=True)
            
            current_norm_grad = torch.norm(current_centered_grad, 'fro')
            past_norm_grad = torch.norm(past_centered_grad, 'fro')
            
            if current_norm_grad > 1e-8 and past_norm_grad > 1e-8:
                current_normalized_grad = current_centered_grad / current_norm_grad
                past_normalized_grad = past_centered_grad / past_norm_grad
            else:
                current_normalized_grad = current_centered_grad
                past_normalized_grad = past_centered_grad
            
            # Apply rotation and compute force
            R_detached = R.detach()
            current_aligned = current_normalized_grad @ R_detached
            diff_matrix = current_aligned - past_normalized_grad.detach()
            
            dist_sq = torch.sum(diff_matrix.pow(2))
            force_magnitude = torch.clamp(repulsion_strength / (dist_sq + 1e-8), max=10.0)
            
            # Transform force back
            aligned_force = force_magnitude * diff_matrix
            force_matrix = aligned_force @ R_detached.T
            
            if current_norm_grad > 1e-8:
                force_matrix = force_matrix / current_norm_grad
            
            return force_matrix
            
        except Exception as e:
            print(f"Procrustes computation failed: {e}. Using Euclidean fallback.")
            # Fallback to simple Euclidean distance
            diff_matrix = current_repr - past_repr
            dist_sq = torch.sum(diff_matrix.pow(2))
            force_magnitude = torch.clamp(repulsion_strength / (dist_sq + 1e-8), max=1.0)
            return force_magnitude * diff_matrix
    
    def cleanup_hooks(self):
        """Remove all registered hooks."""
        for handle in self.feature_hooks.values():
            handle.remove()
        self.feature_hooks = {}
        self.hooks_registered = False
