import os
from typing import List, Tuple, Mapping, Dict
from functools import partial
from pathlib import Path

import wandb

import h5py
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from mattstools.mattstools.torch_utils import get_sched, ema_param_sync
from mattstools.mattstools.modules import IterativeNormLayer, DenseNetwork
from mattstools.mattstools.transformers import (
    FullTransformerEncoder,
    FullTransformerVectorEncoder,
)

# from pcjedi.src.models.pc_jedi import TransformerDiffusionGenerator
from src.models.pcdroid import TransformerDiffusionGenerator

from src.models.modules import MultiplicityFlow
from src.plotting.physics import (
    plot_pointclouds_for_batch,
    plot_marginals,
    plot_marginals_2D,
    plot_masses,
    plot_momenta,
    plot_rapidities,
)
from src.metrics import compute_ks
from src.physics import compute_observables
from src.data.utils import inv_log_squash, pad_with_zeros
from src.data.utils import get_particle_masks, get_particle_mult


class PIPPIN(pl.LightningModule):
    """Particles Into Particles with Permutation Invariant Networks

    Generalised method for taking an input particle cloud with some
    multiplicity and feature dimension and returning another particle cloud
    with a different multiplicity and dimension.
    - Transformer encoder with class attention pooling and normalizing flows
      learns target multiplicity N
    - Generates N many output nodes using random noise
    - Transforms random noise to target point cloud using transformer decoder
      which also takes in the encoded tokens from the input
    - Iteratively applies the previous step as a diffusion process

    Both the multiplicity and the output particle cloud are learned during
    training.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        mult_dim: int,
        pre_enc_config: Mapping,
        pres_head_config: Mapping,
        mult_enc_config: Mapping,
        mult_pred_config: Mapping,
        post_enc_config: Mapping,
        pip_jedi_config: Mapping,
        part_tkn_config: Mapping,
        pres_tkn_config: Mapping,
        normaliser_config: Mapping,
        optimizer: partial,
        sched_config: Mapping,
        max_n: List[int] = [0, 0, 16],
        bias_n: List[int] = [0, 0, 6],
        use_input_n: bool = True,
        loss_delay_n: int = 0,
        loss_weights: Dict[str, float] = {"pc": 1., "n": 1., "p": 1.},
        decay_channel: str = "allhad",
        pres_mode: str = "continuous",
        do_test_loop: bool = True,
        plot_by_channel: bool = True,
        path_plots: str = ".",
        path_hdf5: str = ".",
        job_id: int | None = None,
    ) -> None:
        """Init method for PIPPIN

        Args:
            inpt_dim:
                The feature size of the input point cloud
            outp_dim:
                The feature size of the output point cloud
            mult_dim:
                The feature size of the multiplicity
            pre_enc_config:
                Dict for Transformer Pre-Encoder
            pres_head_config:
                Dict for Transformer Presence Prediction Head
            mult_enc_config:
                Dict for Transformer Multiplicity Encoder
            mult_pred_config:
                Dict for Flow Multiplicity Predictor
            post_enc_config:
                Dict for Transformer Post-Encoder
            pip_jedi_config:
                Dict for Transformer Diffusion Decoder
            part_tkn_config:
                Dict for Particle Tokens Network
            pres_tkn_config:
                Dict for Presence Tokens Network
            normaliser_config:
                Dict for input normaliser
            optimizer:
                Partially configured optimizer
            sched_config:
                Dict for how lightning should implement the scheduler

        Kwargs:
            max_n:
                List of maximum number of nodes that this model will try and
                generate per particle type
            bias_n:
                List of biases to add to the output of the multiplicity network
            use_input_n:
                If the input cardinality is used as extra context info
            loss_delay_n:
                Number of epochs to wait before starting to compute the
                multiplicity loss
            loss_weights:
                Dict of weights for the different loss terms
            decay_channel:
                The decay channel of the dataset
            pres_mode:
                The mode for the presence tokens
            do_test_loop:
                Whether to perform test steps or jump to end method.
                First step always done.
            plot_by_channel:
                Whether to plot the observables by channel
            path_plots:
                Path to save the plots to
            path_hdf5:
                Path to save the HDF5 files to
            job_id:
                The job ID for the current run. Used to separate saved files
        """

        super().__init__()

        self.save_hyperparameters(logger=False)

        # The normaliser that normalises the input point cloud
        self.normaliser_input = IterativeNormLayer(**normaliser_config)

        # The pre-encoder that pre-embeds the input point cloud
        self.pre_encoder = FullTransformerEncoder(**pre_enc_config)

        # The presence head that predicts the presence of partons in output
        self.presence_head = DenseNetwork(**pres_head_config)

        # The cross-attention encoder that condition the multiplicity flow
        self.mult_encoder = FullTransformerVectorEncoder(**mult_enc_config)

        # The multiplicity predictor flow network
        self.mult_predictor = MultiplicityFlow(**mult_pred_config)

        # The post-encoder that further embeds the input point cloud
        self.post_encoder = FullTransformerEncoder(**post_enc_config)

        # The PIP-JeDi decoder conditional diffusion network
        self.pip_jedi = TransformerDiffusionGenerator(**pip_jedi_config)

        # The particle-tokens network that generates tokens for particle types
        self.part_tokens = nn.Embedding(**part_tkn_config)

        # The presence-tokens network that generates tokens for presence
        if pres_mode == "discrete":
            self.pres_tokens = nn.Embedding(**pres_tkn_config[pres_mode])
        elif pres_mode == "continuous":
            self.pres_tokens = nn.Linear(**pres_tkn_config[pres_mode])


    ## NETWORK METHODS ########################################################
    def forward(
        self,
        pc_in: T.Tensor,
        mask_in: T.BoolTensor,
        channel: T.ByteTensor,
        true_p: T.BoolTensor,
        mask_target: T.BoolTensor | None = None,
    ) -> Tuple[T.Tensor, T.BoolTensor, T.ByteTensor, T.Tensor]:
        """
        Common step for training, validation, and testing

        Args:
            pc_in:
                The input point cloud
            mask_in:
                The mask for the input point cloud
            channel:
                The decay channel of the event
            true_p:
                The true presence of the particles in the output

        Kwargs:
            mask_target:
                The mask for the target point cloud. (Default: None)

        Returns:
            pc_enc:
                The encoded input point cloud
            pred_p:
                The predicted presence of the particles in the output
            pred_n:
                The predicted multiplicity
            n_nll:
                The negative log-likelihood of the true multiplicity
        """

        # Normalise the input point cloud
        pc_in = self.normaliser_input(pc_in, mask_in)

        # Pre-encode the input point cloud
        pc_enc = self.pre_encoder(pc_in, mask_in)

        # Get the number of nodes in the output point cloud
        pc_enc, pred_p, pred_n, n_nll = self._mult_step(
            pc_enc.clone(),
            mask_in,
            channel,
            true_p,
            mask_target=mask_target,
        )

        # Further encode the input point cloud
        pc_enc = self.post_encoder(pc_enc, mask_in)

        return pc_enc, pred_p, pred_n, n_nll

    def _mult_step(
        self,
        pc_enc: T.Tensor,
        mask_in: T.BoolTensor,
        channel: T.ByteTensor,
        true_p: T.BoolTensor,
        mask_target: T.BoolTensor | None = None,
    ) -> Tuple[T.BoolTensor, T.ByteTensor, T.Tensor]:
        """
        Gets the output multiplicity or the negative log-likelihood,
        depending on the training stage.

        If in training/validation modes and if the delay has passed,
            -> get the negative log-likelihood
        If in validation/testing modes and if the delay has passed,
            -> predict the multiplicity

        If in training mode (always) OR for validation (before the delay),
        uses the true multiplicity values for generation.

        NB: After the delay, multiplicity will still contribute to the loss,
        but using the real value for generation stabilises the training.

        Args:
            pc_enc:
                The encoded input point cloud
            mask_in:
                The mask for the input point cloud
            channel:
                The decay channel of the event
            true_p:
                The true presence of the particles in the output

        Kwargs:
            mask_target:
                The mask for the target point cloud. (Default: None)

        Returns:
            pred_p:
                The predicted presence of the particles in the output
            pred_n:
                The predicted multiplicity
            n_nll:
                The negative log-likelihood of the true multiplicity
        """

        # Initialise the outputs to None
        # Some of them will be overwritten depending on the stage
        pred_p = None
        pred_n = None
        n_nll = None

        # Compute the outputs only if the delay has passed
        if self.current_epoch >= self.hparams.loss_delay_n:

            # Freeze the CDFDequantisation layer after one epoch
            if self.current_epoch == self.hparams.loss_delay_n + 1:
                if self.mult_predictor.dequant_mode == "cdf":
                    self.mult_predictor.dequantiser.freeze()

            # Extract the input multiplicity if used for conditioning
            input_n = None
            if self.hparams.use_input_n:
                n_lep, n_nu, n_quarks = get_particle_mult(
                    mask=mask_in,
                    channel=channel,
                    level="part",
                )
                input_n = T.cat([n_lep, n_nu, n_quarks], dim=-1)

            # Get the presence of the input particles in the output
            pred_p = self.presence_head(pc_enc, ctxt=input_n)
            pred_p = pred_p * mask_in.unsqueeze(-1)
            pred_p = pred_p.squeeze()

            # Add the presence tokens to the encoded point cloud
            if self.hparams.pres_mode == "discrete":
                pres_tkn = self._get_presence_tokens(pred_p > 0.5)
            elif self.hparams.pres_mode == "continuous":
                pres_tkn = self.pres_tokens(pred_p.unsqueeze(-1))

            pc_enc = pc_enc + pres_tkn

            # Encode the input point cloud via cross-attention
            pc_global = self.mult_encoder(pc_enc, ctxt=input_n)

            # Get the negative log-likelihood of the true multiplicity
            if self.stage in ["train", "valid"]:

                # Get the target multiplicity
                n_lep, n_met, n_jet = get_particle_mult(
                    mask=mask_target,
                    channel=channel,
                    level="reco",
                    max_n=self.hparams.max_n,
                    dataset_name=self.hparams.decay_channel,
                )

                if self.mult_predictor.dequant_mode == "cdf":
                    target_n = T.cat([
                        n_lep,
                        n_met,
                        n_jet,
                    ], dim=-1).long()
                else:
                    target_n = T.cat([
                        n_lep - self.hparams.bias_n[0],
                        n_met - self.hparams.bias_n[1],
                        n_jet - self.hparams.bias_n[2],
                    ], dim=-1)

                # Get the negative log-likelihood of the true multiplicity
                n_nll = self.mult_predictor.get_nll(
                    inpt=target_n,
                    ctxt=pc_global,
                )

            # Predict the output multiplicity
            if self.stage in ["valid", "test"]:
                pred_n = self.mult_predictor.generate(ctxt=pc_global)

                if self.mult_predictor.dequant_mode != "cdf":
                    pred_n[..., 0] += self.hparams.bias_n[0]
                    pred_n[..., 1] += self.hparams.bias_n[1]
                    pred_n[..., 2] += self.hparams.bias_n[2]

                pred_n[..., 0] = pred_n[..., 0].clamp(0, self.hparams.max_n[0])
                pred_n[..., 1] = pred_n[..., 1].clamp(0, self.hparams.max_n[1])
                pred_n[..., 2] = pred_n[..., 2].clamp(0, self.hparams.max_n[2])

                pred_n = pred_n.to(T.uint8)

        # Otherwise add the true presence tokens to the encoded point cloud
        else:
            if self.hparams.pres_mode == "discrete":
                pres_tkn = self._get_presence_tokens(true_p)
            elif self.hparams.pres_mode == "continuous":
                pres_tkn = self.pres_tokens(true_p.unsqueeze(-1).float())

            pc_enc = pc_enc + pres_tkn

        return pc_enc, pred_p, pred_n, n_nll

    def _get_particle_tokens(
        self,
        mask: T.BoolTensor,
        channel: T.ByteTensor,
    ) -> T.Tensor:
        """
        Generate the output particle type tokens given the mask and channel.

        Token indices correspond to the following particle types:
            - 0: padding (i.e. fake particle)
            - 1: lepton
            - 2: MET
            - 3: jet

        Args:
            mask:
                The mask for the point cloud
            channel:
                The decay channel of the event

        Returns:
            part_tkn:
                The particle type tokens
        """

        masks = get_particle_masks(
            mask=mask,
            channel=channel,
            level="reco",
            max_n=self.hparams.max_n,
            dataset_name=self.hparams.decay_channel,
        )
        mask_lep, mask_met, mask_jet = masks

        token_idx = T.zeros_like(mask, dtype=T.int)
        token_idx[mask_lep] = 1
        token_idx[mask_met] = 2
        token_idx[mask_jet] = 3

        return self.part_tokens(token_idx)

    def _get_presence_tokens(
        self,
        presence: T.BoolTensor,
    ) -> T.Tensor:
        """
        Generate the input presence tokens given the presence mask.

        Token indices correspond to the following:
            - 0: parton not present
            - 1: parton present

        Args:
            presence:
                The presence mask for the point cloud

        Returns:
            pres_tkn:
                The particle presence tokens
        """

        token_idx = T.zeros_like(presence, dtype=T.int)
        token_idx[presence] = 1

        return self.pres_tokens(token_idx)

    def _generate_output(
        self,
        pc_enc: T.Tensor,
        mask_in: T.BoolTensor,
        channel: T.ByteTensor,
        gen_n: T.Tensor,
    ) -> Tuple[T.Tensor, T.BoolTensor]:
        """
        Generate the output point cloud with a given predicted multiplicity and
        conditioned on a given encoded input point cloud.

        Args:
            pc_enc:
                The encoded input point cloud used as condition
            mask_in:
                The mask for the input point cloud
            channel:
                The decay channel of the event
            gen_n:
                The output multiplicity to use for generation

        Returns:
            pc_out:
                The generated output point cloud
            mask_out:
                The mask for the output point cloud
        """

        gen_n = gen_n.unsqueeze(-1)

        # Generate the output mask from the multiplicity
        mask_lep = T.arange(
            self.hparams.max_n[0],
            device=self.device,
        ) < gen_n[:, 0]

        mask_met = T.arange(
            self.hparams.max_n[1],
            device=self.device,
        ) < gen_n[:, 1]

        mask_jet = T.arange(
            self.hparams.max_n[2],
            device=self.device,
        ) < gen_n[:, 2]

        mask_out = T.cat([mask_lep, mask_met, mask_jet], dim=-1)

        part_tkn = self._get_particle_tokens(mask_out, channel)

        # Pass through the decoder and get the final output
        pc_out = self.pip_jedi.full_generation(
            mask=mask_out,
            ctxt=None,
            initial_noise=None,
            kv_seq=pc_enc,
            kv_mask=mask_in,
            add_seq=part_tkn,
            mode="CA",
        )

        return pc_out, mask_out

    def _compute_loss(
        self,
        pc_target: T.Tensor,
        mask_target: T.BoolTensor,
        pc_enc: T.Tensor,
        mask_in: T.BoolTensor,
        channel: T.ByteTensor,
        pred_p: T.BoolTensor | None = None,
        true_p: T.BoolTensor | None = None,
        n_nll: T.Tensor | None = None,
    ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        """
        Compute the loss for a given batch

        Args:
            pc_target:
                The target point cloud
            mask_target:
                The mask for the target point cloud
            pc_enc:
                The encoded input point cloud used as condition
            mask_in:
                The mask for the input point cloud
            channel:
                The decay channel of the event

        Kwargs:
            pred_p:
                The predicted presence of the particles in the output
            true_p:
                The true presence of the particles in the output
            n_nll:
                The negative log-likelihood of the true multiplicity

        Returns:
            p_loss:
                The presence prediction loss
            n_loss:
                The multiplicity loss
            pc_loss:
                The point cloud diffusion loss
        """

        # Calculate the multiplicity loss
        n_loss = n_nll if n_nll is not None else T.zeros(1, device=self.device)

        # Calculate the presence prediction loss
        if pred_p is not None and true_p is not None:
            p_loss = F.binary_cross_entropy(pred_p, true_p.float())
        else:
            p_loss = T.zeros(1, device=self.device)

        # Get the particle type tokens
        part_tkn = self._get_particle_tokens(mask_target, channel)

        # Calculate the point cloud diffusion loss
        pc_loss = self.pip_jedi._shared_step(
            sample=(
                pc_target,
                mask_target,
                pc_enc,
                mask_in,
                part_tkn,
                None,
                None,
            ),
            mode="CA",
        )

        return p_loss, n_loss, pc_loss


    ## TRAINING LOOP ##########################################################
    def on_fit_start(self) -> None:
        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            # Training losses
            wandb.define_metric("train/p_loss", summary="min")
            wandb.define_metric("train/n_loss", summary="min")
            wandb.define_metric("train/pc_loss", summary="min")
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("train/scaled_loss", summary="min")

            # Validation losses
            wandb.define_metric("valid/p_loss", summary="min")
            wandb.define_metric("valid/n_loss", summary="min")
            wandb.define_metric("valid/pc_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")
            wandb.define_metric("valid/scaled_loss", summary="min")

            # Validation metrics
            wandb.define_metric("valid/ks_lep_pt", summary="min")
            wandb.define_metric("valid/ks_lep_eta", summary="min")
            wandb.define_metric("valid/ks_lep_phi", summary="min")
            wandb.define_metric("valid/ks_lep_energy", summary="min")

            wandb.define_metric("valid/ks_met_pt", summary="min")
            wandb.define_metric("valid/ks_met_eta", summary="min")
            wandb.define_metric("valid/ks_met_phi", summary="min")
            wandb.define_metric("valid/ks_met_energy", summary="min")

            wandb.define_metric("valid/ks_jet_pt", summary="min")
            wandb.define_metric("valid/ks_jet_eta", summary="min")
            wandb.define_metric("valid/ks_jet_phi", summary="min")
            wandb.define_metric("valid/ks_jet_energy", summary="min")

            wandb.define_metric("valid/ks_m_w1", summary="min")
            wandb.define_metric("valid/ks_m_w2", summary="min")
            wandb.define_metric("valid/ks_m_t1", summary="min")
            wandb.define_metric("valid/ks_m_t2", summary="min")
            wandb.define_metric("valid/ks_m_tt", summary="min")

            wandb.define_metric("valid/ks_pt_w1", summary="min")
            wandb.define_metric("valid/ks_pt_w2", summary="min")
            wandb.define_metric("valid/ks_pt_t1", summary="min")
            wandb.define_metric("valid/ks_pt_t2", summary="min")
            wandb.define_metric("valid/ks_pt_tt", summary="min")

            wandb.define_metric("valid/ks_eta_w1", summary="min")
            wandb.define_metric("valid/ks_eta_w2", summary="min")
            wandb.define_metric("valid/ks_eta_t1", summary="min")
            wandb.define_metric("valid/ks_eta_t2", summary="min")
            wandb.define_metric("valid/ks_eta_tt", summary="min")

        # Initialise a loss scale to rescale the total loss when the
        # multiplicity and presence losses are added after the delay
        self.loss_scale = None

    def training_step(self, batch: Tuple, batch_idx: int) -> T.Tensor:
        """Called by Trainer class, single batch pass in the training loop"""
        self.stage = "train"

        # Break up the batch tuple into its component parts
        pc_in, mask_in, pc_target, mask_target, channel, true_p = batch

        # Pass through the shared prediction step
        pc_enc, pred_p, pred_n, n_nll = self(
            pc_in,
            mask_in,
            channel,
            true_p,
            mask_target=mask_target,
        )

        # Compute the losses
        p_loss, n_loss, pc_loss = self._compute_loss(
            pc_target,
            mask_target,
            pc_enc,
            mask_in,
            channel,
            pred_p=pred_p,
            true_p=true_p,
            n_nll=n_nll,
        )

        # Combine the losses with the appropriate weights
        if self.current_epoch >= self.hparams.loss_delay_n:
            w_pc = self.hparams.loss_weights["pc"]
            w_n = self.hparams.loss_weights["n"]
            w_p = self.hparams.loss_weights["p"]
        else:
            w_pc = 1.
            w_n = 0.
            w_p = 0.

        total_loss = w_pc * pc_loss + w_n * n_loss + w_p * p_loss

        # Set the loss scale after the delay has passed
        if self.current_epoch >= self.hparams.loss_delay_n:
            if self.loss_scale is None:
                self.loss_scale = pc_loss.detach() / total_loss.detach()
            scaled_loss = total_loss * self.loss_scale
        else:
            scaled_loss = total_loss

        # Log the losses
        self.log("train/p_loss", p_loss)
        self.log("train/n_loss", n_loss)
        self.log("train/pc_loss", pc_loss)
        self.log("train/total_loss", total_loss)
        self.log("train/scaled_loss", scaled_loss)

        # Log the particle token values
        for i in range(4):
            self.log(f"model/part_tkn_pad_{i}", self.part_tokens.weight[0, i])
            self.log(f"model/part_tkn_lep_{i}", self.part_tokens.weight[1, i])
            self.log(f"model/part_tkn_met_{i}", self.part_tokens.weight[2, i])
            self.log(f"model/part_tkn_jet_{i}", self.part_tokens.weight[3, i])

        for i, tkn in enumerate(["pad", "lep", "met", "jet"]):
            self.log(f"model/part_tkn_{tkn}_min", self.part_tokens.weight[i].min())
            self.log(f"model/part_tkn_{tkn}_max", self.part_tokens.weight[i].max())
            self.log(f"model/part_tkn_{tkn}_mean", self.part_tokens.weight[i].mean())
            self.log(f"model/part_tkn_{tkn}_mstd", self.part_tokens.weight[i].mean()+self.part_tokens.weight[i].std())

        # Synchronise the EMA model
        ema_param_sync(
            self.pip_jedi.net,
            self.pip_jedi.ema_net,
            self.pip_jedi.ema_sync,
        )

        return total_loss


    ## VALIDATION LOOP ########################################################
    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        self._init_wandb_tables()

    def validation_step(self, batch: Tuple, batch_idx: int) -> dict:
        """Called by the Trainer class, single batch pass in the validation loop"""
        self.stage = "valid"

        # Break up the batch tuple into its component parts
        pc_in, mask_in, pc_target, mask_target, channel, true_p = batch

        # Pass through the shared prediction step
        pc_enc, pred_p, pred_n, n_nll = self(
            pc_in,
            mask_in,
            channel,
            true_p,
            mask_target=mask_target,
        )

        # Compute the losses
        p_loss, n_loss, pc_loss = self._compute_loss(
            pc_target,
            mask_target,
            pc_enc,
            mask_in,
            channel,
            pred_p=pred_p,
            true_p=true_p,
            n_nll=n_nll,
        )

        # Combine the losses with the appropriate weights
        if self.current_epoch >= self.hparams.loss_delay_n:
            w_pc = self.hparams.loss_weights["pc"]
            w_n = self.hparams.loss_weights["n"]
            w_p = self.hparams.loss_weights["p"]
        else:
            w_pc = 1.
            w_n = 1.
            w_p = 1.

        total_loss = w_pc * pc_loss + w_n * n_loss + w_p * p_loss

        # Set the scaled total loss after the delay has passed
        if self.current_epoch >= self.hparams.loss_delay_n and self.loss_scale:
            scaled_loss = total_loss * self.loss_scale
        else:
            scaled_loss = total_loss

        # Log the losses
        self.log("valid/p_loss", p_loss)
        self.log("valid/n_loss", n_loss)
        self.log("valid/pc_loss", pc_loss)
        self.log("valid/total_loss", total_loss)
        self.log("valid/scaled_loss", scaled_loss)

        # For the first batch only
        if batch_idx == 0:

            # Use the true multiplicity for generation before the delay
            if self.current_epoch >= self.hparams.loss_delay_n:
                gen_n = pred_n
            else:
                n_lep, n_met, n_jet = get_particle_mult(
                        mask=mask_target,
                        channel=channel,
                        level="reco",
                        max_n=self.hparams.max_n,
                        dataset_name=self.hparams.decay_channel,
                    )
                gen_n = T.cat([n_lep, n_met, n_jet], dim=-1)

            # Generate a the output point cloud
            pc_out, mask_out = self._generate_output(
                pc_enc,
                mask_in,
                channel,
                gen_n,
            )

            # Undo the pre-processing
            pc_in = self._undo_preprocessing(pc_in)
            pc_out = self._undo_preprocessing(pc_out)
            pc_target = self._undo_preprocessing(pc_target)

            # Binarise the presence
            if pred_p is not None:
                pred_p = pred_p > 0.5

            # Compute the observables
            masses_in, momenta_in, rapidities_in, _, _, _ = compute_observables(
                pc=pc_in,
                mask=mask_in,
                is_parton=True,
            )
            masses_out, momenta_out, rapidities_out, check_out, percent_out, match_out = compute_observables(
                pc=pc_out,
                mask=mask_out,
                pc_ref=pc_in,
                mask_ref=mask_in,
                channel=channel,
            )
            masses_target, momenta_target, rapidities_target, check_target, percent_target, match_target = compute_observables(
                pc=pc_target,
                mask=mask_target,
                pc_ref=pc_in,
                mask_ref=mask_in,
                channel=channel,
            )

            # Compute and log some metrics
            metric_ks = compute_ks(
                pc_x=pc_out,
                mask_x_part=get_particle_masks(mask_out, channel),
                masses_x=masses_out[check_out],
                momenta_x=momenta_out[check_out],
                rapidities_x=rapidities_out[check_out],
                pc_y=pc_target,
                mask_y_part=get_particle_masks(mask_target, channel),
                masses_y=masses_target[check_target],
                momenta_y=momenta_target[check_target],
                rapidities_y=rapidities_target[check_target],
            )

            self._log_metric_ks(metric_ks, stage="valid")

            # Record the reconstructions using plots
            self._make_plots(
                pc_out=pc_out,
                mask_out=mask_out,
                pc_target=pc_target,
                mask_target=mask_target,
                channel=channel,
                pred_p=pred_p,
                true_p=true_p,
                masses_in=masses_in,
                masses_out=masses_out,
                masses_target=masses_target,
                momenta_in=momenta_in,
                momenta_out=momenta_out,
                momenta_target=momenta_target,
                rapidities_in=rapidities_in,
                rapidities_out=rapidities_out,
                rapidities_target=rapidities_target,
                percent_out=percent_out,
                percent_target=percent_target,
                check_out=check_out,
                check_target=check_target,
                match_out=match_out,
                match_target=match_target,
                by_channel=True,
                stage="valid",
            )

            # Record the reconstructions using scatter plots
            plot_list = plot_pointclouds_for_batch(
                pc_in[:10],
                mask_in[:10],
                pc_out[:10],
                mask_out[:10],
                pc_target[:10],
                mask_target[:10],
                path=Path(self.hparams.path_plots)/"valid/pointclouds",
            )

            # Log the scatter plots
            for idx, img in enumerate(plot_list):
                self.table_pointclouds.add_data(
                    f"{batch_idx}-{idx}",
                    wandb.Image(img[1]),
                    wandb.Image(img[2]),
                    wandb.Image(img[3]),
                )

    def on_validation_end(self) -> None:
        """Called at the end of validation."""
        self._log_wandb_tables(stage="valid")


    ## TESTING LOOP ###########################################################
    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self._init_wandb_tables()

        # Initialise the lists for storing the test loop outputs
        self.outputs_pointclouds = []
        self.outputs_masks = []
        self.outputs_channels = []
        self.outputs_presences = []
        self.outputs_checks = []
        self.outputs_percents = []
        self.outputs_matches = []
        self.outputs_masses = []
        self.outputs_momenta = []
        self.outputs_rapidities = []

    def test_step(self, batch: Tuple, batch_idx: int):
        """Called by the Trainer class, single batch pass in the test loop"""
        self.stage = "test"

        if self.hparams.do_test_loop or batch_idx == 0:

            # Break up the batch tuple into its component parts
            pc_in, mask_in, pc_target, mask_target, channel, true_p = batch

            # Pass through the shared prediction step
            pc_enc, pred_p, pred_n, n_nll = self(
                pc_in,
                mask_in,
                channel,
                true_p,
                mask_target=mask_target,
            )

            # Generate a the output point cloud
            pc_out, mask_out = self._generate_output(
                pc_enc,
                mask_in,
                channel,
                pred_n,
            )

            # Undo the pre-processing
            pc_in = self._undo_preprocessing(pc_in)
            pc_out = self._undo_preprocessing(pc_out)
            pc_target = self._undo_preprocessing(pc_target)

            # Binarise the presence
            if pred_p is not None:
                pred_p = pred_p > 0.5

            # Compute the observables
            masses_in, momenta_in, rapidities_in, _, _, _ = compute_observables(
                pc=pc_in,
                mask=mask_in,
                is_parton=True,
            )
            masses_out, momenta_out, rapidities_out, check_out, percent_out, match_out = compute_observables(
                pc=pc_out,
                mask=mask_out,
                pc_ref=pc_in,
                mask_ref=mask_in,
                channel=channel,
            )
            masses_target, momenta_target, rapidities_target, check_target, percent_target, match_target = compute_observables(
                pc=pc_target,
                mask=mask_target,
                pc_ref=pc_in,
                mask_ref=mask_in,
                channel=channel,
            )

            # Stack the outputs in single tensors
            pointclouds = T.stack(
                [
                    pad_with_zeros(pc_in, max_n=pc_out.shape[1]),
                    pc_out,
                    pc_target,
                ],
                dim=-1,
            )
            masks = T.stack(
                [
                    pad_with_zeros(mask_in, max_n=mask_out.shape[1]),
                    mask_out,
                    mask_target,
                ],
                dim=-1,
            )
            presences = T.stack([pred_p, true_p], dim=-1)
            checks = T.stack([check_out, check_target], dim=-1)
            percents = T.stack([percent_out, percent_target], dim=-1)
            matches = T.stack([match_out, match_target], dim=-1)
            masses = T.stack([masses_in, masses_out, masses_target], dim=-1)
            momenta = T.stack([momenta_in, momenta_out, momenta_target], dim=-1)
            rapidities = T.stack([rapidities_in, rapidities_out, rapidities_target], dim=-1)

            # Store the batch outputs for combination at the end of the epoch
            # Moving to CPU to save GPU memory
            self.outputs_pointclouds.append(pointclouds.cpu())
            self.outputs_masks.append(masks.cpu())
            self.outputs_channels.append(channel.cpu())
            self.outputs_presences.append(presences.cpu())
            self.outputs_checks.append(checks.cpu())
            self.outputs_percents.append(percents.cpu())
            self.outputs_matches.append(matches.cpu())
            self.outputs_masses.append(masses.cpu())
            self.outputs_momenta.append(momenta.cpu())
            self.outputs_rapidities.append(rapidities.cpu())

            # For the first batch only
            if batch_idx == 0:

                # Record the reconstructions using scatter plots
                path = Path(self.hparams.path_plots)/"test"
                if self.hparams.job_id: path = path/f"{self.hparams.job_id}"
                plot_list = plot_pointclouds_for_batch(
                    pc_in[:10],
                    mask_in[:10],
                    pc_out[:10],
                    mask_out[:10],
                    pc_target[:10],
                    mask_target[:10],
                    path=path/"pointclouds",
                )

                # Log the scatter plots
                for idx, img in enumerate(plot_list):
                    self.table_pointclouds.add_data(
                        f"{batch_idx}-{idx}",
                        wandb.Image(img[1]),
                        wandb.Image(img[2]),
                        wandb.Image(img[3]),
                    )

    def on_test_epoch_end(self) -> None:
        """Called at the end of the testing epoch."""

        if self.hparams.do_test_loop:

            # Combine the outputs in single tensors
            pointclouds = T.cat(self.outputs_pointclouds, dim=0)
            masks = T.cat(self.outputs_masks, dim=0)
            channel = T.cat(self.outputs_channels, dim=0)
            presences = T.cat(self.outputs_presences, dim=0)
            checks = T.cat(self.outputs_checks, dim=0)
            percents = T.cat(self.outputs_percents, dim=0)
            matches = T.cat(self.outputs_matches, dim=0)
            masses = T.cat(self.outputs_masses, dim=0)
            momenta = T.cat(self.outputs_momenta, dim=0)
            rapidities = T.cat(self.outputs_rapidities, dim=0)

            # Save the outputs to an HDF5 file
            path = Path(self.hparams.path_hdf5)/f"{self.stage}"
            if self.hparams.job_id: path = path/f"{self.hparams.job_id}"
            filename = Path("outputs.h5")
            os.makedirs(path, exist_ok=True)

            with h5py.File(path/filename, "w") as f:
                pippin = f.create_group("pippin")
                pippin.create_dataset("pointclouds", data=pointclouds.numpy())
                pippin.create_dataset("masks", data=masks.numpy())
                pippin.create_dataset("channel", data=channel.numpy())
                pippin.create_dataset("presences", data=presences.numpy())
                pippin.create_dataset("checks", data=checks.numpy())
                pippin.create_dataset("percents", data=percents.numpy())
                pippin.create_dataset("matchabilities", data=matches.numpy())
                pippin.create_dataset("masses", data=masses.numpy())
                pippin.create_dataset("momenta", data=momenta.numpy())
                pippin.create_dataset("rapidities", data=rapidities.numpy())

        else:

            # Load the outputs from the HDF5 file
            path = Path(self.hparams.path_hdf5)/f"{self.stage}"
            path = path/f"{self.hparams.job_id}"
            filename = Path("outputs.h5")

            with h5py.File(path/filename, "r") as f:
                pippin = f["pippin"]
                pointclouds = T.tensor(pippin["pointclouds"][...])
                masks = T.tensor(pippin["masks"][...])
                channel = T.tensor(pippin["channel"][...])
                presences = T.tensor(pippin["presences"][...])
                checks = T.tensor(pippin["checks"][...])
                percents = T.tensor(pippin["percents"][...])
                matches = T.tensor(pippin["matchabilities"][...])
                masses = T.tensor(pippin["masses"][...])
                momenta = T.tensor(pippin["momenta"][...])
                rapidities = T.tensor(pippin["rapidities"][...])

        # Extract the output and target point clouds
        pc_in = pointclouds[..., 0]
        pc_out = pointclouds[..., 1]
        pc_target = pointclouds[..., 2]

        # Extract the output and target masks
        mask_in = masks[..., 0]
        mask_out = masks[..., 1]
        mask_target = masks[..., 2]

        # Extract the predicted and true presences
        pred_p = presences[..., 0]
        true_p = presences[..., 1]

        # Extract the output and target checks
        check_out = checks[..., 0]
        check_target = checks[..., 1]

        # Extract the output and target percentages
        percent_out = percents[..., 0]
        percent_target = percents[..., 1]

        # Extract the output and target matchabilities
        match_out = matches[..., 0]
        match_target = matches[..., 1]

        # Extract the input, output and target masses
        masses_in = masses[..., 0]
        masses_out = masses[..., 1]
        masses_target = masses[..., 2]

        # Extract the input, output and target momenta
        momenta_in = momenta[..., 0]
        momenta_out = momenta[..., 1]
        momenta_target = momenta[..., 2]

        # Extract the input, output and target rapidities
        rapidities_in = rapidities[..., 0]
        rapidities_out = rapidities[..., 1]
        rapidities_target = rapidities[..., 2]

        # Compute and log some metrics
        metric_ks = compute_ks(
            pc_x=pc_out,
            mask_x_part=get_particle_masks(mask_out, channel),
            masses_x=masses_out[check_out],
            momenta_x=momenta_out[check_out],
            rapidities_x=rapidities_out[check_out],
            pc_y=pc_target,
            mask_y_part=get_particle_masks(mask_target, channel),
            masses_y=masses_target[check_target],
            momenta_y=momenta_target[check_target],
            rapidities_y=rapidities_target[check_target],
        )

        self._log_metric_ks(metric_ks, stage="test")

        # Record the reconstructions using plots
        stage = "test"
        if self.hparams.job_id: stage += f"/{self.hparams.job_id}"
        self._make_plots(
            pc_out=pc_out,
            mask_out=mask_out,
            pc_target=pc_target,
            mask_target=mask_target,
            channel=channel,
            pred_p=pred_p,
            true_p=true_p,
            masses_in=masses_in,
            masses_out=masses_out,
            masses_target=masses_target,
            momenta_in=momenta_in,
            momenta_out=momenta_out,
            momenta_target=momenta_target,
            rapidities_in=rapidities_in,
            rapidities_out=rapidities_out,
            rapidities_target=rapidities_target,
            percent_out=percent_out,
            percent_target=percent_target,
            check_out=check_out,
            check_target=check_target,
            match_out=match_out,
            match_target=match_target,
            by_channel=True,
            stage=stage,
        )

        # Free memory
        self.outputs_pointclouds.clear()
        self.outputs_masks.clear()
        self.outputs_channels.clear()
        self.outputs_presences.clear()
        self.outputs_checks.clear()
        self.outputs_percents.clear()
        self.outputs_matches.clear()
        self.outputs_masses.clear()
        self.outputs_momenta.clear()
        self.outputs_rapidities.clear()

    def on_test_end(self) -> None:
        """Called at the end of testing."""
        self._log_wandb_tables(stage="test")


    ## INTERNAL METHODS #######################################################
    def _init_wandb_tables(self) -> None:
        """
        Initialise the tables for logging to wandb.
        Called at the beginning of the validation and testing.
        """

        # The pointclouds table contains 2D coloured point clouds
        self.table_pointclouds = wandb.Table(
            columns=["idx", "input", "output", "truth"],
        )

        # The marginals table contains 1D histograms of
        # the point cloud features
        columns = [
            "lep_pt", "lep_eta", "lep_phi", "lep_energy",
            "met_pt", "met_eta", "met_phi", "met_energy",
            "jet_pt", "jet_eta", "jet_phi", "jet_energy",
            "lep_n", "met_n", "jet_n",
            "match_inter", "match_final",
        ]
        self.table_marginals = wandb.Table(columns=columns)

        # The marginals_2D table contains 2D histograms of
        # the point cloud features
        columns = columns[:-2] + ["match_raw"]
        self.table_marginals_2D = wandb.Table(columns=columns)

        # The masses table contains 1D histograms of
        # the underlying invariant masses
        self.table_masses = wandb.Table(
            columns=["w1", "w2", "top1", "top2", "ttbar"],
        )

        # The momenta table contains 1D histograms of
        # the underlying transverse momenta
        self.table_momenta = wandb.Table(
            columns=["w1", "w2", "top1", "top2", "ttbar"],
        )

        # The rapidities table contains 1D histograms of
        # the underlying pseudo-rapidities
        self.table_rapidities = wandb.Table(
            columns=["w1", "w2", "top1", "top2", "ttbar"],
        )

    def _log_wandb_tables(self, stage: str = "unknown") -> None:
        """
        Manually pushes the tables and images to wandb.
        Called at the end of the validation and testing.
        """
        if wandb.run is not None:
            wandb.log(
            {
                f"{stage}/pointclouds": self.table_pointclouds,
                f"{stage}/marginals": self.table_marginals,
                f"{stage}/marginals_2D": self.table_marginals_2D,
                f"{stage}/masses": self.table_masses,
                f"{stage}/momenta": self.table_momenta,
                f"{stage}/rapidities": self.table_rapidities,
            },
            commit=False,
        )

    def _undo_preprocessing(
        self,
        pc: T.Tensor,
        # mask: T.BoolTensor,
    ) -> T.Tensor:
        """
        Undo the pre-processing on the point cloud using the inverse log-squash
        function.

        Args:
            pc:
                The point cloud to be un-preprocessed.

        Returns:
            pc:
                The un-preprocessed point cloud.
        """
        pc[..., 0] = inv_log_squash(pc[..., 0])
        pc[..., 3] = inv_log_squash(pc[..., 3])
        return pc

    def _log_metric_ks(
        self,
        metric_ks: T.Tensor,
        stage: str = "unknown",
    ):
        """
        Log the computed metrics to wandb.

        Args:
            metric_ks:
                The computed metrics.

        Kwargs:
            stage:
                The stage of the trainer ('valid' or 'test').
        """

        # Extract the metrics
        ks_lep = metric_ks[..., 0]
        ks_met = metric_ks[..., 1]
        ks_jet = metric_ks[..., 2]
        ks_masses = metric_ks[..., 3]
        ks_momenta = metric_ks[..., 4]
        ks_rapidities = metric_ks[..., 5]

        # Log the metrics
        self.log(f"{stage}/ks_lep_pt", ks_lep[0])
        self.log(f"{stage}/ks_lep_eta", ks_lep[1])
        self.log(f"{stage}/ks_lep_phi", ks_lep[2])
        self.log(f"{stage}/ks_lep_energy", ks_lep[3])

        self.log(f"{stage}/ks_met_pt", ks_met[0])
        self.log(f"{stage}/ks_met_eta", ks_met[1])
        self.log(f"{stage}/ks_met_phi", ks_met[2])
        self.log(f"{stage}/ks_met_energy", ks_met[3])

        self.log(f"{stage}/ks_jet_pt", ks_jet[0])
        self.log(f"{stage}/ks_jet_eta", ks_jet[1])
        self.log(f"{stage}/ks_jet_phi", ks_jet[2])
        self.log(f"{stage}/ks_jet_energy", ks_jet[3])

        self.log(f"{stage}/ks_m_w1", ks_masses[0])
        self.log(f"{stage}/ks_m_w2", ks_masses[1])
        self.log(f"{stage}/ks_m_t1", ks_masses[2])
        self.log(f"{stage}/ks_m_t2", ks_masses[3])
        self.log(f"{stage}/ks_m_tt", ks_masses[4])

        self.log(f"{stage}/ks_pt_w1", ks_momenta[0])
        self.log(f"{stage}/ks_pt_w2", ks_momenta[1])
        self.log(f"{stage}/ks_pt_t1", ks_momenta[2])
        self.log(f"{stage}/ks_pt_t2", ks_momenta[3])
        self.log(f"{stage}/ks_pt_tt", ks_momenta[4])

        self.log(f"{stage}/ks_eta_w1", ks_rapidities[0])
        self.log(f"{stage}/ks_eta_w2", ks_rapidities[1])
        self.log(f"{stage}/ks_eta_t1", ks_rapidities[2])
        self.log(f"{stage}/ks_eta_t2", ks_rapidities[3])
        self.log(f"{stage}/ks_eta_tt", ks_rapidities[4])

    def _make_plots(
        self,
        pc_out: T.Tensor,
        mask_out: T.BoolTensor,
        pc_target: T.Tensor,
        mask_target: T.BoolTensor,
        channel: T.ByteTensor,
        pred_p: T.Tensor,
        true_p: T.Tensor,
        masses_in: T.Tensor,
        masses_out: T.Tensor,
        masses_target: T.Tensor,
        momenta_in: T.Tensor,
        momenta_out: T.Tensor,
        momenta_target: T.Tensor,
        rapidities_in: T.Tensor,
        rapidities_out: T.Tensor,
        rapidities_target: T.Tensor,
        percent_out: T.Tensor,
        percent_target: T.Tensor,
        check_out: T.BoolTensor,
        check_target: T.BoolTensor,
        match_out: T.ByteTensor,
        match_target: T.ByteTensor,
        by_channel: bool = False,
        do_log: bool = True,
        stage: str = "unknown",
        file_suffix: str = "",
        path_suffix: str = "",
    ) -> None:
        """
        Create and add the plots to wandb tables.

        Args:
            pc_out:
                The output point cloud.
            mask_out:
                The output mask.
            pc_target:
                The target point cloud.
            mask_target:
                The target mask.
            channel:
                The channel of the event.
            pred_p:
                The predicted presence.
            true_p:
                The true presence.
            masses_in:
                The input masses.
            masses_out:
                The output masses.
            masses_target:
                The target masses.
            momenta_in:
                The input momenta.
            momenta_out:
                The output momenta.
            momenta_target:
                The target momenta.
            rapidities_in:
                The input rapidities.
            rapidities_out:
                The output rapidities.
            rapidities_target:
                The target rapidities.
            percent_out:
                The output percentage of fully matched events.
            percent_target:
                The target percentage of fully matched events.
            check_out:
                The output check of fully matched events.
            check_target:
                The target check of fully matched events.
            match_out:
                The output computed matchability.
            match_target:
                The target computed matchability.

        Kwargs:
            by_channel:
                If True, the plots are also split by channel.
            do_log:
                If True, the plots are logged to wandb.
            stage:
                The stage of the trainer ('valid' or 'test').
            file_suffix:
                The suffix to append to the file name.
            path_suffix:
                The suffix to append to the path.
        """

        path_plots = Path(self.hparams.path_plots)/stage

        # Compute additional quantities
        mult_out = get_particle_mult(
            mask=mask_out,
            channel=channel,
            level="reco",
            max_n=self.hparams.max_n,
            dataset_name=self.hparams.decay_channel,
        )
        mult_target = get_particle_mult(
            mask=mask_target,
            channel=channel,
            level="reco",
            max_n=self.hparams.max_n,
            dataset_name=self.hparams.decay_channel,
        )
        mask_out_part = get_particle_masks(
            mask=mask_out,
            channel=channel,
            level="reco",
            max_n=self.hparams.max_n,
            dataset_name=self.hparams.decay_channel,
        )
        mask_target_part = get_particle_masks(
            mask=mask_target,
            channel=channel,
            level="reco",
            max_n=self.hparams.max_n,
            dataset_name=self.hparams.decay_channel,
        )

        # Record the reconstructions using marginal histograms
        plot_list = plot_marginals(
            pc_out=pc_out,
            mult_out=mult_out,
            mask_out=mask_out,
            mask_out_part=mask_out_part,
            pc_target=pc_target,
            mult_target=mult_target,
            mask_target=mask_target,
            mask_target_part=mask_target_part,
            channel=channel,
            pred_p=pred_p,
            true_p=true_p,
            match_out=match_out,
            match_target=match_target,
            suffix=file_suffix,
            path=path_plots/"marginals"/path_suffix,
        )
        if do_log:
            self.table_marginals.add_data(
                *[wandb.Image(p) for p in plot_list],
            )

        # Record the reconstructions using 2D marginal histograms
        plot_list = plot_marginals_2D(
            pc_out=pc_out,
            mult_out=mult_out,
            mask_out=mask_out,
            mask_out_part=mask_out_part,
            pc_target=pc_target,
            mult_target=mult_target,
            mask_target=mask_target,
            mask_target_part=mask_target_part,
            channel=channel,
            pred_p=pred_p,
            true_p=true_p,
            suffix=file_suffix,
            path=path_plots/"marginals_2D"/path_suffix,
        )
        if do_log:
            self.table_marginals_2D.add_data(
                *[wandb.Image(p) for p in plot_list],
            )

        # Record the invariant masses using histograms
        plot_list = plot_masses(
            masses_in=masses_in,
            masses_out=masses_out[check_out],
            masses_target=masses_target[check_target],
            percent_out=percent_out,
            percent_target=percent_target,
            suffix=file_suffix,
            path=path_plots/"masses"/path_suffix,
        )
        if do_log:
            self.table_masses.add_data(
                *[wandb.Image(p) for p in plot_list],
            )

        # Record the transverse momentum using histograms
        plot_list = plot_momenta(
            momenta_in=momenta_in,
            momenta_out=momenta_out[check_out],
            momenta_target=momenta_target[check_target],
            percent_out=percent_out,
            percent_target=percent_target,
            suffix=file_suffix,
            path=path_plots/"momenta"/path_suffix,
        )
        if do_log:
            self.table_momenta.add_data(
                *[wandb.Image(p) for p in plot_list],
            )

        # Record the pseudo-rapidities using histograms
        plot_list = plot_rapidities(
            rapidities_in=rapidities_in,
            rapidities_out=rapidities_out[check_out],
            rapidities_target=rapidities_target[check_target],
            percent_out=percent_out,
            percent_target=percent_target,
            suffix=file_suffix,
            path=path_plots/"rapidities"/path_suffix,
        )
        if do_log:
            self.table_rapidities.add_data(
                *[wandb.Image(p) for p in plot_list],
            )

        # If specified, record all plots split by channel as well
        if by_channel and self.hparams.plot_by_channel:
            for c in [0b00, 0b01, 0b10, 0b11]:
                self._make_plots(
                    pc_out=pc_out[channel==c],
                    mask_out=mask_out[channel==c],
                    pc_target=pc_target[channel==c],
                    mask_target=mask_target[channel==c],
                    channel=channel[channel==c],
                    pred_p=pred_p[channel==c] if pred_p is not None else None,
                    true_p=true_p[channel==c],
                    masses_in=masses_in[channel==c],
                    masses_out=masses_out[channel==c],
                    masses_target=masses_target[channel==c],
                    momenta_in=momenta_in[channel==c],
                    momenta_out=momenta_out[channel==c],
                    momenta_target=momenta_target[channel==c],
                    rapidities_in=rapidities_in[channel==c],
                    rapidities_out=rapidities_out[channel==c],
                    rapidities_target=rapidities_target[channel==c],
                    percent_out=percent_out*(channel==c).float().mean(),
                    percent_target=percent_target*(channel==c).float().mean(),
                    check_out=check_out[channel==c],
                    check_target=check_target[channel==c],
                    match_out=match_out[channel==c],
                    match_target=match_target[channel==c],
                    by_channel=False,
                    do_log=False,
                    stage=stage,
                    file_suffix=f"_c{c:02b}",
                    path_suffix="by_channel",
                )

    ## CONFIGURATION ##########################################################
    def configure_optimizers(self) -> Dict:
        """
        Configure the optimisers and learning rate sheduler for this model.
        This method is called by the Lightning Trainer.

        Returns:
            Dict:
                A dictionary containing the optimiser and scheduler.
        """

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.sched_config.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                **self.hparams.sched_config.lightning,
            },
        }
