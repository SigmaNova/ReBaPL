import os.path as osp
import torch
import math

from dassl.engine import TRAINER_REGISTRY
from .schedulers import build_lr_scheduler
# from .optimizers import build_optimizer
from dassl.optim import build_optimizer
from dassl.utils import load_checkpoint

import copy
# wrapper of any trainer, to add our optimization algorithm on top of it
# set the name of the original trainer
from .maple import MaPLe  # 
from .independentVL import IVLP
from .cocoop import CoCoOp

from pathlib import Path
import glob 
from torch.amp import autocast

from .representation_tracker import RepresentationTracker


@TRAINER_REGISTRY.register()
class CSGHMC_CR(CoCoOp):
    def build_model(self):
        super().build_model()
        self.cycle_length = self.cfg.CSGHMC.CYCLE_LENGTH
        self.optim.cycle_length = self.cfg.CSGHMC.CYCLE_LENGTH
        self.optim.noise_last_epochs = self.cfg.CSGHMC.NOISE_LAST_EPOCHS
        self.optim.noise_temperature = self.cfg.CSGHMC.NOISE_TEMPERATURE
        self.optim.dataset_size = len(self.train_loader_x.dataset)  # for noise calculation

        lr_scheduler = "cosine" # "cosine_restart" if self.cfg.CSGHMC.CYCLE_LENGTH > 0 else "cosine"
        self.lr_scheduler_type = lr_scheduler
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM, lr_scheduler, cycle_length=None, max_epoch=self.cycle_length)
        self.models = []
        self.cycles_state_dict = {}
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())


        #### Repulsion between cycles ####
        # Initialize RepresentationTracker for inter-cycle repulsion
        self.representation_tracker = RepresentationTracker(
            device=self.device,
            num_ref_samples=self.cfg.CSGHMC.REPULSION.REF_SAMPLES,
            regularization_strength=self.cfg.CSGHMC.REPULSION.REG_STRENGTH,
            batch_size=self.cfg.CSGHMC.REPULSION.BATCH_SIZE
        )
        # Inter-cycle repulsion parameters
        self.repulsion_strength = self.cfg.CSGHMC.REPULSION.REPULSION_STRENGTH
        self.current_cycle = 0

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

        if self.epoch % self.cycle_length == 0:
            if self.epoch > 0:
                self.current_cycle = self.epoch // self.cycle_length
                model = copy.deepcopy(self.model)
                # Update representation for this cycle
                self.representation_tracker.update_cycle_representation(model, self.current_cycle)
                #reintialize model weights to first cycle weights
                # first_cycle_ckpt = load_checkpoint(Path(self.cfg.OUTPUT_DIR) / "first_cycle" / "prompt_learner" / "first_cycle.tar")
                # state_dict = first_cycle_ckpt["state_dict"]

                # if "prompt_learner.token_prefix" in state_dict:
                #     del state_dict["prompt_learner.token_prefix"]

                # if "prompt_learner.token_suffix" in state_dict:
                #     del state_dict["prompt_learner.token_suffix"]

                # re-build scheduler
                self.model.load_state_dict(self.initial_state_dict)
                self.model.train()
                self.optim.state.clear()  # Clear all accumulated state

                self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM, "cosine", cycle_length=None, max_epoch=self.cycle_length)

            # else:
            #     #save the first cycle too
            #     model_name = "first_cycle.tar"
            #     save_dir = Path(self.cfg.OUTPUT_DIR) / "first_cycle"
            #     self.save_model(self.epoch, save_dir, is_best=False, model_name=model_name)
        super().run_epoch()
        
        if self.cycle_length > 0:
            cycle_length = self.cfg.CSGHMC.CYCLE_LENGTH
            print(f'self.epoch: {self.epoch}, cycle_length: {cycle_length}, max_epoch: {self.cfg.OPTIM.MAX_EPOCH}')
            if cycle_length > 0 and (self.epoch + 1) % cycle_length == 0 and self.epoch > 0 or (self.epoch + 1) == self.cfg.OPTIM.MAX_EPOCH:
                print(f"Saving checkpoint at epoch {self.epoch} due to cosine restart")
                model_name = "model-best.pth.tar"
                save_dir = Path(self.cfg.OUTPUT_DIR) / f"cycle_epochs_ep{self.epoch}"
                self.save_model(self.epoch, save_dir, is_best=False, model_name=model_name)
                self.cycles_state_dict[self.epoch] = copy.deepcopy(self.model.state_dict())


    def model_inference(self, input): # return average logits over all models
        logits = 0
        for model in self.models:
            with torch.inference_mode():
                logits += model(input)
        return logits / len(self.models)
    
    def load_model(self, directory, epoch=None):
        # get checkpoint paths 
        # get all folders in the directory that start with "cycle_epochs_ep"
        checkpoint_paths = glob.glob(osp.join(directory, "cycle_epochs_ep*"))
        print(f"Found {len(checkpoint_paths)} checkpoints in {directory}")
        checkpoint_paths = sorted(zip([int(p.split("/")[-1].replace("cycle_epochs_ep", "")) for p in checkpoint_paths], checkpoint_paths), key=lambda x: x[0])  # sort by epoch
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

        return loss_summary

    def _add_repulsion_gradients(self):
        """Add Procrustes-based repulsion gradients to current gradients."""
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
                    
