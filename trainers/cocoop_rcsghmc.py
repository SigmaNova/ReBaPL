import os.path as osp
import os
import math
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from scipy.linalg import orthogonal_procrustes


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

use_cuda = torch.cuda.is_available()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoCoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

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
                print("Warning: 'text_encoder' features not found. Falling back.")
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
        
        try:
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
            
        except Exception as e:
            print(f"Warning: Procrustes gradient computation failed: {e}")
            return {}
    
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits

@TRAINER_REGISTRY.register()
class CoCoOp_rcSGHMC(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]

        self.model.to(self.device)
        
        # Create real PyTorch optimizer but override its step method
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        # Store scheduler parameters for handmade scheduling
        self.base_lr = cfg.OPTIM.LR
        self.max_epochs = cfg.OPTIM.MAX_EPOCH
        self.warmup_epochs = cfg.OPTIM.WARMUP_EPOCH
        self.warmup_lr = cfg.OPTIM.WARMUP_CONS_LR
        self.lr_scheduler_type = cfg.OPTIM.LR_SCHEDULER
        
        # Store trainable params and momentum buffers for handmade SGD
        self.trainable_params = trainable_params
        self.momentum_buffers = [torch.zeros_like(p) for p in self.trainable_params]
        
        # Initialize RepresentationTracker for inter-cycle repulsion
        self.representation_tracker = RepresentationTracker(
            device=self.device,
            num_ref_samples=getattr(cfg.TRAINER.REPULSION, 'REF_SAMPLES', 128),
            regularization_strength=getattr(cfg.TRAINER.REPULSION, 'REG_STRENGTH', 1e-6)
        )
        
        # Inter-cycle repulsion parameters
        self.repulsion_strength = getattr(cfg.TRAINER.REPULSION, 'REPULSION_STRENGTH', 0.01)
        self.current_cycle = 0
        self.cycle_length = getattr(cfg.OPTIM, 'CYCLE_LENGTH', 20)
        
        self.original_optim_step = self.optim.step
        self.original_sched_step = self.sched.step
        self.optim.step = self.handmade_sgd_step

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpts={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def run_epoch(self):
        """Override run_epoch to initialize reference samples and load previous cycles."""
        if self.epoch == 0 and self.representation_tracker.reference_samples is None:
            
            rng_state = torch.get_rng_state()

            train_dataset = self.train_loader_x.dataset
            ref_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True, # This is now safe to use
                num_workers=self.train_loader_x.num_workers,
                pin_memory=self.train_loader_x.pin_memory
            )

            self.representation_tracker.initialize_reference_samples(ref_loader)

            torch.set_rng_state(rng_state)

        if self.epoch % self.cycle_length == 0 and self.epoch > 0:
            self.current_cycle = self.epoch // self.cycle_length
            self._load_previous_cycle_representations()
            
        super().run_epoch()


    def _load_previous_cycle_representations(self):
        """Load representations from all available previous cycle checkpoints."""
        model_dir = glob.glob(osp.join(self.cfg.OUTPUT_DIR, "Samples/"))
        checkpoint_paths = glob.glob(osp.join(model_dir, "cycle_ep*.pth.tar"))
        
        if not checkpoint_paths:
            print("No previous cycle checkpoints found.")
            return
        
        print(f"Loading representations from {len(checkpoint_paths)} previous cycles...")
        
        for checkpoint_path in checkpoint_paths:
            try:
                # Extract cycle number from filename
                filename = osp.basename(checkpoint_path)
                cycle_epoch = int(filename.split('ep')[1].split('.')[0])
                cycle_num = cycle_epoch // self.cycle_length
                
                if cycle_num >= self.current_cycle:
                    continue  # Skip current or future cycles
                
                # Load checkpoint
                checkpoint = load_checkpoint(checkpoint_path)
                state_dict = checkpoint["state_dict"]
                
                # Handle DataParallel prefix if necessary
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                
                # Ignore fixed token vectors
                if "prompt_learner.token_prefix" in state_dict:
                    del state_dict["prompt_learner.token_prefix"]
                if "prompt_learner.token_suffix" in state_dict:
                    del state_dict["prompt_learner.token_suffix"]
                
                # Temporarily load the model state
                original_state = self.model.state_dict()
                self.model.load_state_dict(state_dict, strict=False)
                
                # Extract representation for this cycle
                self.representation_tracker.update_cycle_representation(self.model, cycle_num)
                
                # Restore original model state
                self.model.load_state_dict(original_state)
                
                print(f"Loaded representation for cycle {cycle_num} from epoch {cycle_epoch}")
                
            except Exception as e:
                print(f"Failed to load cycle representation from {checkpoint_path}: {e}")

    def handmade_sgd_step(self, closure=None):
        """Our handmade SGD step using current LR from the real optimizer"""
        current_lr = self.optim.param_groups[0]['lr']
        weight_decay = self.optim.param_groups[0]['weight_decay']
        momentum = self.optim.param_groups[0]['momentum']

        
        for i, param in enumerate(self.trainable_params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Apply weight decay
            if weight_decay != 0:
                grad = grad.add(param.data, alpha=weight_decay)
            
            # Apply momentum
            buf = self.momentum_buffers[i]
            buf.mul_(momentum).add_(grad)
            
            #sghmc noise
            if self.epoch%20+1>15  :
                temperature = 1
                noise_std = math.sqrt(2 * weight_decay * current_lr)
                noise = torch.randn_like(param) * noise_std * temperature
                buf.add_(noise)

            # Update parameters
            param.data.add_(buf, alpha=-current_lr)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            
            if self.repulsion_strength > 0 and self.current_cycle > 0:
                self._add_repulsion_gradients()
            
            scaler.step(optim)  
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            
            if self.repulsion_strength > 0 and self.current_cycle > 0:
                self._add_repulsion_gradients()
            
            optim.step()  

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # Save checkpoint every cfg.OPTIM.CYCLE_LENGTH epochs
        if self.lr_scheduler_type == "cosine_restart":
            cycle_length = self.cfg.OPTIM.CYCLE_LENGTH

            if cycle_length > 0 and (self.epoch) % cycle_length == 0 and (self.batch_idx + 1) == self.num_batches and self.epoch > 0 or (self.epoch + 1) == self.max_epochs and (self.batch_idx + 1) == self.num_batches:
                print(f"Saving checkpoint at epoch {self.epoch} due to cosine restart")
                # Update current cycle representation before saving
                self.representation_tracker.update_cycle_representation(self.model, self.current_cycle)
                model_name = f"cycle_ep{self.epoch}.pth.tar"
                self.save_model(self.epoch, "./output/", is_best=False, model_name=model_name)

        return loss_summary

    def _add_repulsion_gradients(self):
        """Add Procrustes-based repulsion gradients to current gradients."""
        try:
            # Get repulsion gradients from representation tracker
            repulsion_grads = self.representation_tracker.compute_procrustes_repulsion_gradients(
                net=self.model,
                current_cycle=self.current_cycle,
                repulsion_strength=self.repulsion_strength
            )
            
            if repulsion_grads:
                # Add repulsion gradients to existing gradients
                for param in self.model.parameters():
                    if param in repulsion_grads and param.grad is not None:
                        param.grad.data.add_(repulsion_grads[param])
                        
        except Exception as e:
            print(f"Warning: Failed to add repulsion gradients: {e}")



    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    @torch.no_grad()
    def test(self, split=None):
        """
        Perform ensemble evaluation by averaging predictions from all saved cycle checkpoints.
        """
        # Find all saved model checkpoints from the cycles
        model_dir = glob.glob(osp.join(self.cfg.OUTPUT_DIR, "Samples/"))
        checkpoint_paths = glob.glob(osp.join(model_dir, "cycle_ep*.pth.tar"))
        
        if not checkpoint_paths:
            print("No cycle checkpoints found. Falling back to standard evaluation.")
            return super().test(split)

        print(f"Found {len(checkpoint_paths)} cycle checkpoints for ensemble evaluation.")

        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        
        data_loader = self.test_loader
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        
        print(f"Evaluating on the *{split}* set using an ensemble of {len(checkpoint_paths)} models...")

        for batch in tqdm(data_loader, desc="Ensemble Evaluation"):
            input, label = self.parse_batch_test(batch)
            
            # Accumulate logits from all models in the ensemble
            ensembled_logits = 0
            
            for checkpoint_path in checkpoint_paths:
                # Load state dict from a checkpoint
                checkpoint = load_checkpoint(checkpoint_path)
                state_dict = checkpoint["state_dict"]
                
                # Handle DataParallel prefix if necessary
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}

                #ignore fixed token vectors
                if "prompt_learner.token_prefix" in state_dict:
                    del state_dict["prompt_learner.token_prefix"]
                if "prompt_learner.token_suffix" in state_dict:
                    del state_dict["prompt_learner.token_suffix"]

                # Load the weights into the model
                self.model.load_state_dict(state_dict, strict=False)

                # Get logits for the current model
                with torch.no_grad():
                    logits = self.model_inference(input)
                
                ensembled_logits += logits
            
            # Average the logits
            averaged_logits = ensembled_logits / len(checkpoint_paths)
            
            # Process the averaged predictions
            self.evaluator.process(averaged_logits, label)

        results = self.evaluator.evaluate()

        print(f"Ensemble evaluation results on the *{split}* set:")
        for k, v in results.items():
            print(f"- {k}: {v:.2f}")

        return list(results.values())[0]
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        """
        Override save_model to handle unpicklable handmade methods.
        """
        # Temporarily restore original methods to allow parent save_model to work
        original_optim_step = self.optim.step
        original_sched_step = self.sched.step
        self.optim.step = self.original_optim_step  # Restore original picklable method
        self.sched.step = self.original_sched_step  # Restore original picklable method

        try:
            super().save_model(epoch, directory, is_best=is_best, val_result=val_result, model_name=model_name)
        
        finally:
            self.optim.step = original_optim_step
            self.sched.step = original_sched_step