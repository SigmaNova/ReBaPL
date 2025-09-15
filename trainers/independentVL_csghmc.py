import os.path as osp
import os
import math
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


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
        
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]

        self.model.to(self.device)
        
        # Create real PyTorch optimizer but override its step method
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        
        # Store scheduler parameters for handmade scheduling
        self.base_lr = cfg.OPTIM.LR
        self.max_epochs = cfg.OPTIM.MAX_EPOCH
        self.warmup_epochs = cfg.OPTIM.WARMUP_EPOCH
        self.warmup_lr = cfg.OPTIM.WARMUP_CONS_LR
        self.lr_scheduler_type = cfg.OPTIM.LR_SCHEDULER
        
        # Store trainable params and momentum buffers for handmade SGD
        self.trainable_params = trainable_params
        self.momentum_buffers = [torch.zeros_like(p) for p in self.trainable_params]
        
        self.original_optim_step = self.optim.step
        self.original_sched_step = self.sched.step
        self.optim.step = self.handmade_sgd_step
        # self.sched.step = self.handmade_scheduler_step  # Override scheduler step

        print("--- Handmade IVLP Optimizer with Dassl Scheduler ---")
        print(f"  Base LR: {cfg.OPTIM.LR}")
        print(f"  Weight Decay: {getattr(cfg.OPTIM, 'WEIGHT_DECAY', 0.0)}")
        print(f"  Momentum: {getattr(cfg.OPTIM, 'MOMENTUM', 0.9)}")
        print(f"  Trainable params: {len(trainable_params)}")


        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpts={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

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
                temperature = 1.0
                noise_std = math.sqrt(2 * weight_decay * current_lr)
                noise = torch.randn_like(param) * noise_std / temperature
                buf.add_(noise)

            # Update parameters
            param.data.add_(buf, alpha=-current_lr)

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
            
    def handmade_scheduler_step(self):
        """Our handmade scheduler step that updates the LR in optimizer"""
        current_epoch = self.epoch + 1  # epoch is 0-indexed
        
        # if current_epoch <= self.warmup_epochs:
        #     new_lr = self.warmup_lr
        # else:
        if self.lr_scheduler_type == "cosine":
            # Cosine annealing with minimum LR
            eta_min = self.base_lr * 0.01  # 1% of base LR as minimum
            progress = (current_epoch - self.warmup_epochs) / (self.max_epochs)
            cos_val = math.cos(math.pi * progress)
            new_lr = eta_min + (self.base_lr - eta_min) * 0.5 * (1 + cos_val)
            new_lr = max(new_lr, eta_min)  # Ensure minimum
        else:
            new_lr = self.base_lr
        
        # Update the learning rate in optimizer's param_groups
        for param_group in self.optim.param_groups:
            param_group['lr'] = new_lr
            
        print(f"Epoch {current_epoch}: Learning rate updated to {new_lr:.6f}")

    # Remove custom zero_grad and sgd_step methods
    # def zero_grad(self):
    # def sgd_step(self):

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.IVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)  # This will call our handmade_sgd_step
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()  # This will call our handmade_sgd_step

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # Save checkpoint every cfg.OPTIM.CYCLE_LENGTH epochs
        if self.lr_scheduler_type == "cosine_restart":
            cycle_length = self.cfg.OPTIM.CYCLE_LENGTH

            if cycle_length > 0 and (self.epoch) % cycle_length == 0 and (self.batch_idx + 1) == self.num_batches and self.epoch > 0 or (self.epoch + 1) == self.max_epochs and (self.batch_idx + 1) == self.num_batches:
                print(f"Saving checkpoint at epoch {self.epoch} due to cosine restart")
                model_name = f"cycle_ep{self.epoch}.pth.tar"
                self.save_model(self.epoch, "./output/",is_best=False, model_name=model_name)

        return loss_summary

    # def update_lr(self):
    #     """Update learning rate at the end of each epoch"""
    #     current_epoch = self.epoch + 1  # epoch is 0-indexed
    #     self.current_lr = self.get_current_lr()
    #     print(f"Epoch {current_epoch}: Learning rate updated to {self.current_lr:.6f}")

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
        model_dir = "/home/ubuntu/omar/promptsrc/PromptSRC/output/VLPromptLearner/"
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
            # Call the parent's save_model method.
            # This saves the standard model checkpoint (e.g., model-best.pth.tar).
            super().save_model(epoch, directory, is_best=is_best, val_result=val_result, model_name=model_name)
        
        finally:
            # Always restore the handmade methods so training can continue.
            self.optim.step = original_optim_step
            self.sched.step = original_sched_step