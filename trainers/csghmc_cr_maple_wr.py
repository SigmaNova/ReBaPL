import os.path as osp
import torch
import math
from torch.utils.data import DataLoader

from dassl.engine import TRAINER_REGISTRY
from .schedulers import build_lr_scheduler
from .optimizers import build_optimizer
# from dassl.optim import build_optimizer
from dassl.utils import load_checkpoint

import copy
# wrapper of any trainer, to add our optimization algorithm on top of it
# set the name of the original trainer
from .maple import MaPLe  # 
from .independentVL import IVLP
from .cocoop import CoCoOp
from .coop import CoOp


from pathlib import Path
import glob 
#from torch.amp import autocast
from torch.cuda.amp.autocast_mode import autocast


@TRAINER_REGISTRY.register()
class CSGHMC_CR_MAPLE_WR(MaPLe):
    def build_model(self):
        super().build_model()
        # self.optim = build_optimizer(self.model, self.cfg.OPTIM)
        self.cycle_length = self.cfg.CSGHMC.CYCLE_LENGTH
        self.optim.cycle_length = self.cfg.CSGHMC.CYCLE_LENGTH
        self.optim.noise_last_epochs = self.cfg.CSGHMC.NOISE_LAST_EPOCHS
        self.optim.noise_temperature = self.cfg.CSGHMC.NOISE_TEMPERATURE
        self.optim.dataset_size = len(self.train_loader_x.dataset)  # for noise calculation
        self.initial_model = copy.deepcopy(self.model)
        self.initial_optim = copy.deepcopy(self.optim)
        self.rng_state = torch.get_rng_state()

        if self.cfg.CSGHMC.CHAINS == "parallel":
            save_dir = Path(self.cfg.OUTPUT_DIR) / f"initial_checkpoints"
            model_name = "model-best.pth.tar"
            self.save_model(self.epoch, save_dir, is_best=False, model_name=model_name)
            self.initial_checkpoint = save_dir
            lr_scheduler = "cosine"
        else: 
            lr_scheduler = "cosine_restart"

        self.lr_scheduler_type = lr_scheduler
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM, lr_scheduler, cycle_length=self.cycle_length, max_epoch=self.cycle_length)
        self._scheds["MultiModalPromptLearner"] = self.sched
        self.models = []
        
        #### Weight-based repulsion between cycles ####
        # Store prompt weights from previous cycles for repulsion
        self.previous_cycle_prompts = []
        self.repulsion_strength = self.cfg.CSGHMC.REPULSION.REPULSION_STRENGTH
        self.current_cycle = 0
        self.noise_last_epochs = self.cfg.CSGHMC.NOISE_LAST_EPOCHS
        self.samples_per_cycle = self.cfg.CSGHMC.SAMPLES_PER_CYCLE
        self.last_cycle_samples = []

    def _get_prompt_parameters(self, model=None):
        """Extract all learnable prompt parameters from the model."""
        if model is None:
            model = self.model
        
        prompt_params = {}
        prompt_learner = model.prompt_learner
        
        # Main context vectors (shallow prompts)
        prompt_params['ctx'] = prompt_learner.ctx.data.clone()
        
        # Compound prompts (deeper layers)
        compound_prompts = []
        for param in prompt_learner.compound_prompts_text:
            compound_prompts.append(param.data.clone())
        prompt_params['compound_prompts'] = compound_prompts
        
        return prompt_params

    def _compute_weight_repulsion_gradients(self):
        """Compute repulsion gradients based on prompt weight distances."""
        if not self.previous_cycle_prompts or self.repulsion_strength <= 0:
            return {}
        
        current_prompts = self._get_prompt_parameters()
        repulsion_grads = {}
        
        # Get current prompt learner
        prompt_learner = self.model.prompt_learner
        
        # Compute repulsion for main context vectors
        ctx_grad = torch.zeros_like(prompt_learner.ctx.data)
        for prev_prompts in self.previous_cycle_prompts:
            # Compute distance and repulsion gradient
            diff = current_prompts['ctx'] - prev_prompts['ctx']
            distance = torch.norm(diff)
            if distance > 1e-8:  # Avoid division by zero
                # Repulsion force: push away from previous cycle prompts
                repulsion_force = diff / (distance + 1e-8)
                ctx_grad += self.repulsion_strength * repulsion_force
        
        repulsion_grads[prompt_learner.ctx] = ctx_grad
        
        # Compute repulsion for compound prompts
        for i, param in enumerate(prompt_learner.compound_prompts_text):
            compound_grad = torch.zeros_like(param.data)
            for prev_prompts in self.previous_cycle_prompts:
                if i < len(prev_prompts['compound_prompts']):
                    diff = current_prompts['compound_prompts'][i] - prev_prompts['compound_prompts'][i]
                    distance = torch.norm(diff)
                    if distance > 1e-8:
                        repulsion_force = diff / (distance + 1e-8)
                        compound_grad += self.repulsion_strength * repulsion_force
            
            repulsion_grads[param] = compound_grad
        
        return repulsion_grads

    def run_epoch(self):
        """Override run_epoch to handle cycle transitions."""
        cycle_length = self.cfg.CSGHMC.CYCLE_LENGTH
        
        super().run_epoch()
        print(f"c {self.epoch}, cycle_length: {cycle_length}, current_cycle: {self.current_cycle}")
        
        if cycle_length > 0 and (self.epoch + 1) % cycle_length == 0 and self.epoch > 0 or (self.epoch + 1) == self.cfg.OPTIM.MAX_EPOCH:
            print(f'self.epoch: {self.epoch}, cycle_length: {cycle_length}, current_cycle: {self.current_cycle}, max_epoch: {self.cfg.OPTIM.MAX_EPOCH}')
            
            # Store prompt weights from current cycle for future repulsion
            cycle_prompts = []
            for i, weights in enumerate(self.last_cycle_samples):
                # Load weights to copy of the model
                model = copy.deepcopy(self.model)
                model.load_state_dict(weights)
                model.eval()
                
                # Extract and store prompt parameters
                prompt_params = self._get_prompt_parameters(model)
                cycle_prompts.append(prompt_params)
                
                print(f"Saving checkpoint at epoch {self.epoch} due to cosine restart")           
                save_dir = Path(self.cfg.OUTPUT_DIR) / f"cycle_epochs_ep{self.epoch}_sample{i}"
                model_name = "model-best.pth.tar"
                self.save_model(self.epoch, save_dir, is_best=False, model_name=model_name)
            
            # Add current cycle prompts to history for repulsion
            self.previous_cycle_prompts.extend(cycle_prompts)
            
            self.current_cycle += 1
            # Reset samples for the new cycle
            self.last_cycle_samples = []

            ######## Parallel Chains ########
            if self.cfg.CSGHMC.CHAINS == "parallel":
                super().build_model(first_build=False)
                
                self.optim = build_optimizer(self.model, self.cfg.OPTIM)
                self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM, "cosine", cycle_length=None, max_epoch=self.cycle_length)
                self.model.train()
                
                self.optim.cycle_length = self.cfg.CSGHMC.CYCLE_LENGTH
                self.optim.noise_last_epochs = self.cfg.CSGHMC.NOISE_LAST_EPOCHS
                self.optim.noise_temperature = self.cfg.CSGHMC.NOISE_TEMPERATURE
                self.optim.dataset_size = len(self.train_loader_x.dataset)  # for noise calculation
                
                self._models["MultiModalPromptLearner"] = self.model
                self._optims["MultiModalPromptLearner"] = self.optim
                self._scheds["MultiModalPromptLearner"] = self.sched
                print(f"[DEBUG]: Param CTX mean after reinit: {self.model.prompt_learner.ctx.data.mean()}, std: {self.model.prompt_learner.ctx.data.std()}")
                for param in self.model.prompt_learner.compound_prompts_text:
                    print(f"[DEBUG]: Param mean after reinit: {param.data.mean()}, std: {param.data.std()}")

    def model_inference(self, input): # return average logits over all models
        logits = 0
        for model in self.models:
            # model = model.to(self.device)
            with torch.inference_mode():
                logits += model(input)
            # model = model.to("cpu")
        return logits / len(self.models)
    
    def load_model(self, directory, epoch=None):
        # get checkpoint paths 
        # get all folders in the directory that start with "cycle_epochs_ep"
        checkpoint_paths = glob.glob(osp.join(directory, "cycle_epochs_ep*"))
        print(f"Found {len(checkpoint_paths)} checkpoints in {directory}")
        checkpoint_paths = sorted(zip([p.split("/")[-1].replace("cycle_epochs_ep", "") for p in checkpoint_paths], checkpoint_paths), key=lambda x: x[0])  # sort by epoch
        checkpoint_paths = [p[1] for p in checkpoint_paths]
        print(f"Checkpoints: {checkpoint_paths}")
        checkpoint_paths = checkpoint_paths[:]  # load first cycle only for now
        self.models = []
        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {directory}")
        for checkpoint in checkpoint_paths: 
            print(f"Loading checkpoint from {checkpoint}")
            super().load_model(checkpoint, epoch=None) # Important to keep it None
            # clone the model and add to the list
            model = copy.deepcopy(self.model)
            model.eval()  # Ensure each model in ensemble is in eval mode
            self.models.append(model)

    def after_train(self): 
        # Load all checkpoints again to form the final ensemble
        self.load_model(self.cfg.OUTPUT_DIR)
        super().after_train()

    
    def after_epoch(self): 
        super().after_epoch()
        self.optim.set_epoch(self.epoch)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()

            if self.repulsion_strength > 0 and self.current_cycle > 0:
                self._add_weight_repulsion_gradients()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            if self.repulsion_strength > 0 and self.current_cycle > 0:
                self._add_weight_repulsion_gradients()
            optim.step()

        loss_summary = {"loss": loss.item()}
        # collect samples at the end of each cycle
        cycle_epoch = (self.epoch + 1) % self.cycle_length
        if cycle_epoch == 0 and self.batch_idx == 0:
            print(f"DEBUG: cycle_epoch: {cycle_epoch}, self.epoch: {self.epoch}, batch_idx: {self.batch_idx}, num_batches: {self.num_batches}")
            print(f"loss: {loss.item()}")

        if self.noise_last_epochs == 0 or self.samples_per_cycle == 1: 
            # if last batch of last epoch in cycle, collect sample
            if cycle_epoch == 0 and (self.batch_idx + 1) == self.num_batches: 
                print(f"Collecting sample at epoch {self.epoch}, batch {self.batch_idx}")
                self.last_cycle_samples.append(copy.deepcopy(self.model.state_dict()))
        else:
            if cycle_epoch >= self.cycle_length - self.noise_last_epochs: #  
                # Compute remaining number of steps in this cycle
                steps_in_cycle = self.noise_last_epochs * self.num_batches
                current_step_in_cycle = (cycle_epoch * self.num_batches) + self.batch_idx + 1
                
                # collect uniformly spaced samples
                if self.samples_per_cycle > 0 and len(self.last_cycle_samples) < self.samples_per_cycle:
                    interval = steps_in_cycle // self.samples_per_cycle
                    if (current_step_in_cycle - (self.cycle_length - self.noise_last_epochs) * self.num_batches) % interval == 0:
                        print(f"Collecting sample at epoch {self.epoch}, batch {self.batch_idx}")
                        self.last_cycle_samples.append(copy.deepcopy(self.model.state_dict()))
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def _add_weight_repulsion_gradients(self):
        """Add weight-based repulsion gradients to current gradients."""
        repulsion_grads = self._compute_weight_repulsion_gradients()

        average_grad_norm = 0.0
        if repulsion_grads:
            # Add repulsion gradients to existing gradients
            for param in self.model.parameters():
                if param in repulsion_grads and param.grad is not None:
                    param.grad.data.add_(repulsion_grads[param])
                    average_grad_norm += repulsion_grads[param].norm().item()
            average_grad_norm /= len(repulsion_grads)
            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                print(f"Avg repulsion grad norm: {average_grad_norm:.4f}")
        else: 
            if self.current_cycle > 0:  # Only warn if we expect repulsion but don't get it
                print("Warning: No repulsion gradients computed despite current_cycle > 0")
