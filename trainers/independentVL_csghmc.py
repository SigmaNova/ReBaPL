import os.path as osp
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class cSGHMC_Optimizer:
    """Contour Stochastic Gradient Hamiltonian Monte Carlo optimizer
    
    Based on the original cSGHMC paper implementation with proper cosine annealing
    and temperature-controlled noise injection during sampling phase.
    """
    def __init__(self, params, lr=0.5, alpha=0.9, temperature=1./50000, weight_decay=5e-4, 
                 total_epochs=200, datasize=100, M=2, cycle_length=None, 
                 warmup_epochs=1, warmup_lr=1e-5): # Add warmup parameters
        self.params = list(params)
        self.lr_0 = lr  # initial learning rate
        self.alpha = alpha  # friction coefficient (1.0 = SGLD, <1.0 = SGHMC)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.total_epochs = total_epochs
        self.datasize = datasize
        
        # Cosine annealing parameters - configurable
        self.M = M  # number of cycles
        self.cycle_length = cycle_length if cycle_length is not None else total_epochs // self.M
        self.current_epoch = 0
        self.current_batch = 0
        self.batches_per_epoch = 3  # will be updated based on actual data

        # Warmup parameters
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        
        # Initialize momentum buffers (called 'buf' in original implementation)
        self.momentum_buffer = []
        for param in self.params:
            self.momentum_buffer.append(torch.zeros_like(param.data))
            
        # Add param_groups for compatibility with Dassl
        self.param_groups = [{'lr': lr, 'params': self.params}]
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def update_epoch(self, epoch, batches_per_epoch=3):
        """Update epoch information for cosine annealing schedule"""
        self.current_epoch = epoch
        self.batches_per_epoch = batches_per_epoch
        self.current_batch = 0
    
    def get_current_lr(self):
        """
        Compute current learning rate with warmup at the start of each cycle.
        This now updates at each batch for more fine-grained control.
        """
        # Calculate total batches processed so far
        total_batches_so_far = self.current_epoch * self.batches_per_epoch + self.current_batch
        
        # Calculate which cycle we're in based on total batches
        total_batches_per_cycle = self.cycle_length * self.batches_per_epoch
        current_cycle = total_batches_so_far // total_batches_per_cycle
        batch_in_cycle = total_batches_so_far % total_batches_per_cycle
        
        # First batch of each cycle is warmup
        if batch_in_cycle == 0:
            return self.warmup_lr
        
        # For other batches in the cycle, apply cosine annealing
        annealing_batches_per_cycle = total_batches_per_cycle - 1
        if annealing_batches_per_cycle <= 0:
            return self.warmup_lr
        
        # Calculate how far we are into the annealing phase (BATCH-LEVEL)
        batches_into_annealing = batch_in_cycle - 1
        
        # Cosine annealing formula applied at the BATCH level
        cos_inner = math.pi * batches_into_annealing / annealing_batches_per_cycle
        cos_out = math.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr_0
        
        return lr
    
    def is_sampling_phase(self):
        """Check if we're in the sampling phase (last epoch of each cycle)
        
        Generic implementation that works for any cycle length:
        - Sample at the last epoch of each cycle
        """
        epoch_in_cycle = (self.current_epoch % self.cycle_length) + 1
        return epoch_in_cycle == self.cycle_length  # Sample only at the last epoch of each cycle
    
    def step(self):
        """One step of cSGHMC update following the original implementation"""
        lr = self.get_current_lr()
        is_sampling = self.is_sampling_phase()
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            momentum = self.momentum_buffer[i]
            # momentum = 0
            
            # # Add weight decay to gradient
            grad = grad + self.weight_decay * param.data
            
            # Update momentum: buf_new = (1-alpha)*buf - lr*grad
            momentum_new = (1 - self.alpha) * momentum - lr * grad
            
            # Add temperature-controlled noise during sampling phase
            if is_sampling:
                noise_std = math.sqrt(2.0 * lr * self.alpha * self.temperature / self.datasize)
                noise = torch.randn_like(param.data) * noise_std
                momentum_new = momentum_new + noise * self.temperature

            # Update parameters: param = param + momentum_new
            param.data.add_(momentum_new)
            
            # Store updated momentum
            self.momentum_buffer[i] = momentum_new
        
        # Update batch counter
        self.current_batch += 1
        
        # Update learning rate in param_groups for compatibility
        self.param_groups[0]['lr'] = lr
    
    def state_dict(self):
        """Return state dictionary for checkpointing"""
        return {
            'momentum_buffer': self.momentum_buffer,
            'param_groups': self.param_groups,
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary from checkpoint"""
        self.momentum_buffer = state_dict['momentum_buffer']
        self.param_groups = state_dict['param_groups']
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.current_batch = state_dict.get('current_batch', 0)


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
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.IVLP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.IVLP.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


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


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

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

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class IVLP_cSGHMC(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]

        sghmc_beta = getattr(cfg.TRAINER, 'CSGHMC', {}).get('BETA', 0.9)  # momentum parameter
        sghmc_alpha = getattr(cfg.TRAINER, 'CSGHMC', {}).get('ALPHA', 0.01)  # friction coefficient
        temp_start = getattr(cfg.TRAINER, 'CSGHMC', {}).get('TEMP_START', 1.0)  # starting temperature
        temp_end = getattr(cfg.TRAINER, 'CSGHMC', {}).get('TEMP_END', 0.1)  # ending temperature
        cycle_length = getattr(cfg.TRAINER, 'CSGHMC', {}).get('CYCLE_LENGTH', 5)  # epochs per cycle
        M = getattr(cfg.TRAINER, 'CSGHMC', {}).get('M', 2)  # number of cycles
        max_samples = getattr(cfg.TRAINER, 'CSGHMC', {}).get('MAX_SAMPLES', 20)  # max samples in memory
        sample_at_end = getattr(cfg.TRAINER, 'CSGHMC', {}).get('SAMPLE_AT_END', True)  # sampling strategy
        csghmc_weight_decay = getattr(cfg.TRAINER, 'CSGHMC', {}).get('WEIGHT_DECAY', 5e-4)  # cSGHMC weight decay
        
        total_epochs = cfg.OPTIM.MAX_EPOCH
        dataset_size = 100  # approximated for few-shot learning (will be updated)

        warmup_epochs = cfg.OPTIM.WARMUP_EPOCH
        warmup_lr = cfg.OPTIM.WARMUP_CONS_LR
        
        print(f"Using cSGHMC with beta={sghmc_beta}, alpha={sghmc_alpha}, temp_start={temp_start}, temp_end={temp_end}, cycle_length={cycle_length}, M={M}, max_samples={max_samples}")
        self.sghmc_optim = cSGHMC_Optimizer(
            trainable_params,
            lr=cfg.OPTIM.LR,  # Use the standard learning rate from config
            alpha=sghmc_alpha,
            temperature=temp_start,  # Start with initial temperature
            weight_decay=csghmc_weight_decay,  # Use configurable weight decay
            total_epochs=total_epochs,
            datasize=dataset_size,
            M=M,  # Pass configurable number of cycles
            cycle_length=cycle_length,  # Pass configurable cycle length
            warmup_epochs=warmup_epochs,  # Pass warmup params
            warmup_lr=warmup_lr  # Pass warmup params
        )

        print("--- Custom cSGHMC Optimizer ---")
        for i, group in enumerate(self.sghmc_optim.param_groups):
            print(f"  Group {i}:")
            print(f"    LR: {group['lr']}")
            print(f"    Weight Decay: {self.sghmc_optim.weight_decay}") # You stored it separately
            print(f"    Num Params: {len(group['params'])}")

        self.register_model("VLPromptLearner", self.model, self.sghmc_optim, None)
        # Keep track of samples for Bayesian averaging and cycle parameters
        self.samples = []
        self.sample_count = 0
        self.cycle_length = cycle_length
        self.total_cycles = M
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.max_samples = max_samples
        self.sample_at_end = sample_at_end
        self.last_sampled_epoch = -1  # Track to avoid multiple samples per epoch

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def before_epoch(self):
        """Hook executed before each epoch."""
        super().before_epoch()
        # Update optimizer with current epoch info (for cosine annealing)
        # This should be done once per epoch, not per batch.
        self.sghmc_optim.update_epoch(self.epoch, batches_per_epoch=len(self.train_loader_x))

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        # Update optimizer with current epoch info (for cosine annealing)
        # self.sghmc_optim.update_epoch(self.epoch, batches_per_epoch=len(self.train_loader_x))
        
        model = self.model
        optim = self.sghmc_optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.IVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
    
        # Collect samples based on configurable strategy
        epoch_in_cycle = (self.epoch % self.cycle_length) + 1
        
        if self.sample_at_end:
            # Sample only at the last epoch of each cycle
            should_sample = epoch_in_cycle == self.cycle_length
        else:
            # Sample during the last portion of each cycle (like original cSGHMC)
            last_portion = max(1, self.cycle_length // 5)  # Last 20% of cycle, minimum 1 epoch
            should_sample = epoch_in_cycle > (self.cycle_length - last_portion)
        
        if should_sample and self.epoch != self.last_sampled_epoch:
            sample_type = "end-of-cycle" if self.sample_at_end else "sampling-phase"
            print(f"🎯 Collecting IVLP sample at epoch {self.epoch} (epoch {epoch_in_cycle}/{self.cycle_length} in cycle, {sample_type})")
            self.collect_sample()
            self.last_sampled_epoch = self.epoch

        # Compute accuracy for monitoring
        with torch.no_grad():
            # Temporarily set to eval mode to get logits instead of loss
            model.eval()
            output = model(image)  # Get logits for accuracy computation
            model.train()  # Set back to training mode
            acc = compute_accuracy(output, label)[0].item()

        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
            "n_samples": len(self.samples),
            "current_lr": self.sghmc_optim.get_current_lr(),
            "sampling_phase": self.sghmc_optim.is_sampling_phase(),
            "epoch_in_cycle": (self.epoch % self.cycle_length) + 1,
            "cycle_length": self.cycle_length,
            "should_sample": (self.epoch % self.cycle_length) + 1 == self.cycle_length
        }

        return loss_summary
    
    def collect_sample(self):
        """Collect current parameters as a sample for Bayesian averaging
        
        Collects samples at the end of each cycle based on configured cycle_length
        """
        sample = {}
        for name, param in self.model.prompt_learner.named_parameters():
            sample[name] = param.data.clone()
        
        # Add metadata about when this sample was collected
        epoch_in_cycle = (self.epoch % self.cycle_length) + 1
        cycle = (self.epoch // self.cycle_length) + 1
        
        sample_info = {
            "epoch": self.epoch,
            "cycle": cycle,
            "epoch_in_cycle": epoch_in_cycle,
            "cycle_length": self.cycle_length,
            "parameters": sample
        }
        
        self.samples.append(sample_info)
        self.sample_count += 1
        
        print(f" Collected IVLP sample #{self.sample_count} from epoch {self.epoch} (cycle {cycle}, epoch {epoch_in_cycle}/{self.cycle_length})")
        
        # Keep only recent samples to manage memory (configurable)
        if len(self.samples) > self.max_samples:
            oldest = self.samples.pop(0)
            print(f"🗑️  Removed oldest sample from epoch {oldest['epoch']} to manage memory (max_samples={self.max_samples})")
    
    
    
    def load_samples_from_files(self, output_dir):
        """Load cSGHMC samples from individual epoch files"""
        import glob
        
        # Look for csghmc_sample_epoch_*.pth files
        sample_pattern = osp.join(output_dir, "csghmc_sample_epoch_*.pth")
        sample_files = glob.glob(sample_pattern)
        
        if not sample_files:
            print(f"⚠️  No sample files found in {output_dir}")
            return []
        
        # Sort files by epoch number
        sample_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"🔍 Found {len(sample_files)} IVLP sample files:")
        loaded_samples = []
        
        for sample_file in sample_files:
            try:
                # Extract epoch number from filename
                epoch_num = int(osp.basename(sample_file).split('_')[-1].split('.')[0])
                
                # Load the parameters
                sample_params = torch.load(sample_file, weights_only=True)
                
                # Create sample info structure (use cycle_length from config or default)
                cycle_length = self.cycle_length if hasattr(self, 'cycle_length') else 50  # Fallback to 50 for old files
                sample_info = {
                    "epoch": epoch_num,
                    "cycle": (epoch_num // cycle_length) + 1,
                    "epoch_in_cycle": (epoch_num % cycle_length) + 1,
                    "cycle_length": cycle_length,
                    "parameters": sample_params
                }
                
                loaded_samples.append(sample_info)
                print(f"  ✅ Loaded epoch {epoch_num} from {osp.basename(sample_file)}")
                
            except Exception as e:
                print(f"  ❌ Failed to load {sample_file}: {e}")
                
        print(f"🎯 Successfully loaded {len(loaded_samples)} IVLP samples for ensemble")
        return loaded_samples

    @torch.no_grad()
    def test(self, split=None):
        """Override test to use Bayesian averaged parameters"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        # Auto-load samples from files if not already loaded
        if len(self.samples) == 0:
            print("🔄 No samples in memory, searching for sample files...")
            output_dir = getattr(self.cfg, 'OUTPUT_DIR', './output')
            self.samples = self.load_samples_from_files(output_dir)
        
        if len(self.samples) > 0:
            print(f"Using IVLP Bayesian ensemble averaging over {len(self.samples)} samples")
            epochs = [s['epoch'] for s in self.samples]
            print(f"Sample epochs: {epochs}")
            
            # Evaluate individual samples first
            print(f"\n{'='*60}")
            print("📊 INDIVIDUAL SAMPLE PERFORMANCE:")
            print(f"{'='*60}")
            
            individual_results = []
            
            for i, sample_info in enumerate(self.samples):
                # Reset evaluator for this sample
                self.evaluator.reset()
                
                # Load this sample's parameters into the model
                for name, param in self.model.prompt_learner.named_parameters():
                    if name in sample_info["parameters"]:
                        param.data.copy_(sample_info["parameters"][name])
                
                # Evaluate this sample on the entire test set
                for batch_idx, batch in enumerate(data_loader):
                    input, label = self.parse_batch_test(batch)
                    output = self.model(input)
                    self.evaluator.process(output, label)
                
                # Get results for this sample
                sample_results = self.evaluator.evaluate()
                sample_acc = list(sample_results.values())[0]  # Get the first metric (usually accuracy)
                individual_results.append(sample_acc)
                
                # Print individual sample performance
                cycle = sample_info.get('cycle', 'N/A')
                epoch_in_cycle = sample_info.get('epoch_in_cycle', 'N/A')
                print(f"  Sample {i+1:2d} (Epoch {sample_info['epoch']:2d}, Cycle {cycle}, Epoch {epoch_in_cycle}/{sample_info['cycle_length']}): {sample_acc:.4f}%")
            
            # Print summary statistics
            print(f"\n📈 INDIVIDUAL SAMPLE STATISTICS:")
            print(f"  Mean:   {sum(individual_results)/len(individual_results):.4f}%")
            print(f"  Std:    {(sum([(x - sum(individual_results)/len(individual_results))**2 for x in individual_results]) / len(individual_results))**0.5:.4f}%")
            print(f"  Min:    {min(individual_results):.4f}%")
            print(f"  Max:    {max(individual_results):.4f}%")
            print(f"  Range:  {max(individual_results) - min(individual_results):.4f}%")
            
            # Now evaluate ensemble
            print(f"\n🎯 ENSEMBLE EVALUATION:")
            print(f"{'='*30}")
            
            # Reset evaluator for ensemble evaluation
            self.evaluator.reset()
            
            for batch_idx, batch in enumerate(data_loader):
                input, label = self.parse_batch_test(batch)
                
                # True ensemble averaging: compute predictions from all samples
                ensemble_probs = None
                
                for i, sample_info in enumerate(self.samples):
                    # Load this sample's parameters into the model
                    for name, param in self.model.prompt_learner.named_parameters():
                        if name in sample_info["parameters"]:
                            param.data.copy_(sample_info["parameters"][name])
                    
                    # Get logits from this sample
                    logits = self.model(input)
                    probs = logits
                    
                    # Accumulate probabilities
                    if ensemble_probs is None:
                        ensemble_probs = probs
                    else:
                        ensemble_probs += probs
                
                # Average the probabilities
                ensemble_probs = ensemble_probs / len(self.samples)
                output = ensemble_probs
                
                self.evaluator.process(output, label)
        
        else:
            print("⚠️  No samples available - using single model inference")
            
            # Single model inference (fallback)
            for batch_idx, batch in enumerate(data_loader):
                input, label = self.parse_batch_test(batch)
                output = self.model(input)
                self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        # Print final ensemble results
        if len(self.samples) > 0:
            ensemble_acc = list(results.values())[0]
            mean_individual = sum(individual_results) / len(individual_results)
            improvement = ensemble_acc - mean_individual
            
            print(f"\n🏆 FINAL COMPARISON:")
            print(f"  Ensemble Accuracy:     {ensemble_acc:.4f}%")
            print(f"  Mean Individual:       {mean_individual:.4f}%")
            print(f"  Ensemble Improvement:  {improvement:+.4f}%")
            print(f"  Best Individual:       {max(individual_results):.4f}%")
            print(f"  Ensemble vs Best:      {ensemble_acc - max(individual_results):+.4f}%")

        return list(results.values())[0]


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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

        # Auto-load samples from individual files or csghmc_samples.pth
        samples_path = osp.join(directory, "csghmc_samples.pth")
        if osp.exists(samples_path):
            samples_data = torch.load(samples_path)
            self.samples = samples_data.get("samples", [])
            self.sample_count = samples_data.get("sample_count", 0)
            print(f"✅ Loaded {len(self.samples)} IVLP cSGHMC samples from csghmc_samples.pth")
            if self.samples:
                epochs = [s['epoch'] for s in self.samples]
                print(f"📊 Sample epochs: {epochs}")
        else:
            # Try to load from individual sample files
            print("🔄 csghmc_samples.pth not found, searching for individual sample files...")
            self.samples = self.load_samples_from_files(directory)
            self.sample_count = len(self.samples)
    
    def evaluate_ensemble_from_directory(self, checkpoint_dir):
        """Evaluate Bayesian ensemble from a checkpoint directory
        
        This method loads samples and evaluates both ensemble and single model performance
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING IVLP BAYESIAN ENSEMBLE FROM: {checkpoint_dir}")
        print(f"{'='*80}")
        
        # Load samples directly (skip standard model checkpoint loading for cSGHMC)
        self.load_samples_from_files(checkpoint_dir)
        
        if len(self.samples) == 0:
            print("❌ No samples found! Cannot perform ensemble evaluation.")
            print("   Make sure the checkpoint directory contains csghmc_sample_epoch_*.pth files")
            return None
            
        # Run ensemble evaluation
        print("\n🎯 Running IVLP Ensemble Evaluation...")
        ensemble_result = self.test(split="test")
        
        # Run single model evaluation for comparison
        print("\n📈 Running Single IVLP Model Evaluation...")
        backup_samples = self.samples.copy()  # Backup samples
        self.samples = []  # Temporarily clear samples
        single_result = self.test(split="test")
        self.samples = backup_samples  # Restore samples
        
        # Print results
        print(f"\n{'='*60}")
        print("📋 IVLP FINAL RESULTS:")
        print(f"   Ensemble Accuracy: {ensemble_result:.4f}%")
        print(f"   Single Model Accuracy: {single_result:.4f}%")
        print(f"   Improvement: {ensemble_result - single_result:.4f}%")
        print(f"   Number of Samples: {len(backup_samples)}")
        
        return {
            "ensemble_accuracy": ensemble_result,
            "single_accuracy": single_result, 
            "improvement": ensemble_result - single_result,
            "n_samples": len(backup_samples),
            "sample_epochs": [s['epoch'] for s in backup_samples]
        }
    
    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        """Override save_model to also save cSGHMC samples"""
        super().save_model(epoch, directory, is_best, val_result, model_name)
        
        # Save samples for Bayesian averaging
        if len(self.samples) > 0:
            samples_path = osp.join(directory, "csghmc_samples.pth")
            samples_data = {
                "samples": self.samples,
                "sample_count": self.sample_count,
                "cycle_length": self.cycle_length,
                "total_cycles": self.total_cycles,
                "collection_rule": f"epoch % {self.cycle_length} == {self.cycle_length - 1} (last epoch of each {self.cycle_length}-epoch cycle)"
            }
            torch.save(samples_data, samples_path)
            print(f"💾 Saved {len(self.samples)} IVLP cSGHMC samples to {samples_path}")
            
            # Also save individual sample files for inspection
            for i, sample_info in enumerate(self.samples):
                sample_path = osp.join(directory, f"csghmc_sample_epoch_{sample_info['epoch']}.pth")
                torch.save(sample_info["parameters"], sample_path)
            print(f"Saved individual IVLP sample files for epochs: {[s['epoch'] for s in self.samples]}")