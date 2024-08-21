import copy
from functools import partial
from typing import Callable, Mapping, Optional

import lightning.pytorch as pl
import torch as T

from mattstools.mattstools.k_diffusion import append_dims
from mattstools.mattstools.modules import CosineEncodingLayer, IterativeNormLayer
from mattstools.mattstools.torch_utils import get_loss_fn


class TransformerDiffusionGenerator(pl.LightningModule):
    """A generative model which uses the diffusion process on a point cloud."""

    def __init__(
        self,
        *,
        data_dims: tuple,
        cosine_config: Mapping,
        normaliser_config: Mapping,
        architecture: partial,
        loss_name: partial,
        min_sigma: float = 1e-5,
        max_sigma: float = 80.0,
        ema_sync: float = 0.999,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sampler_function: Callable | None = None,
        sigma_function: Callable | None = None,
    ) -> None:
        """
        Parameters
        ----------
        data_dims : tuple
            A tuple with three integers representing the point cloud dimensions,
            the context dimensions, and the number of nodes, respectively.
        cosine_config : Mapping
            A dictionary with the configuration options for the CosineEncodingLayer object.
        normaliser_config : Mapping
            A dictionary with the configuration options for the IterativeNormLayer object.
        architecture : partial
            A function to initialise the seq-to-seq neural network
        loss_name : str, optional
            The name of the loss function to use. Default is 'mse'.
        min_sigma : float, optional
            The minimum value for the diffusion sigma during generation.
            Default is 1e-5.
        max_sigma : float, optional
            The maximum value for the diffusion sigma. Default is 80.0.
        ema_sync : float, optional
            The exponential moving average synchronization factor. Default is 0.999.
        p_mean : float, optional
            The mean of the log normal distribution used to sample sigmas when training.
            Default is -1.2.
        p_std : float, optional
            The standard deviation of the log normal distribution used to sample the
            sigmas during training. Default is 1.2.
        sampler_function : Callable | None, optional
            A function to sample the points on the point cloud during the
            validation/testing loop. Default is None.
        sigma_function : Callable | None, optional
            A function to compute the diffusion coefficient sigmas for the diffusion
            process. Default is None.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.pc_dim = data_dims[0]
        self.ctxt_dim = data_dims[1]
        self.n_nodes = data_dims[2]
        self.loss_fn = get_loss_fn(loss_name)
        self.ema_sync = ema_sync
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.p_mean = p_mean
        self.p_std = p_std

        # The encoder and scheduler needed for diffusion
        self.sigma_encoder = CosineEncodingLayer(
            min_value=min_sigma, max_value=max_sigma, **cosine_config
        )

        # The layer which normalises the input point cloud data
        self.normaliser = IterativeNormLayer(self.pc_dim, **normaliser_config)
        if self.ctxt_dim:
            self.ctxt_normaliser = IterativeNormLayer(
                self.ctxt_dim, **normaliser_config
            )

        # The denoising neural network (transformer / perceiver / CAE / Epic)
        self.net = architecture(
            inpt_dim=self.pc_dim,
            outp_dim=self.pc_dim,
            ctxt_dim=self.ctxt_dim + self.sigma_encoder.outp_dim,
        )

        # A copy of the network which will sync with an exponential moving average
        self.ema_net = copy.deepcopy(self.net)
        self.ema_net.requires_grad_(False)

        # Sampler to run in the validation/testing loop
        self.val_step_outs = []
        self.sampler_function = sampler_function
        self.sigma_function = sigma_function

    def get_c_values(self, sigmas: T.Tensor) -> tuple:
        """Calculate the Karras C values needed to modify the inputs, outputs,
        and skip connection for the neural network."""

        # we use cos encoding so we dont need c_noise
        c_in = 1 / (1 + sigmas**2).sqrt()
        c_out = sigmas / (1 + sigmas**2).sqrt()
        c_skip = 1 / (1 + sigmas**2)

        return c_in, c_out, c_skip

    def forward(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
        mask: T.BoolTensor,
        ctxt: Optional[T.Tensor] = None,
        kv_seq: Optional[T.Tensor] = None,
        kv_mask: Optional[T.BoolTensor] = None,
        add_seq: Optional[T.Tensor] = None,
        use_ema: bool = False,
    ) -> T.Tensor:
        """Return the denoised data given the current sigma values."""

        # Get the c values for the data scaling
        c_in, c_out, c_skip = self.get_c_values(append_dims(sigmas, noisy_data.dim()))

        # Scale the inputs and pass through the network
        outputs = self.get_outputs(
            c_in * noisy_data,
            sigmas,
            mask,
            ctxt,
            kv_seq,
            kv_mask,
            add_seq,
            use_ema,
        )

        # Get the denoised output by passing the scaled input through the network
        return c_skip * noisy_data + c_out * outputs

    def get_outputs(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
        mask: T.BoolTensor,
        ctxt: Optional[T.Tensor] = None,
        kv_seq: Optional[T.Tensor] = None,
        kv_mask: Optional[T.BoolTensor] = None,
        add_seq: Optional[T.Tensor] = None,
        use_ema: bool = False,
    ) -> T.Tensor:
        """Pass through the model, corresponds to F_theta in the Karras
        paper."""

        # Use the appropriate network for training or validation
        if self.training and not use_ema:
            network = self.net
        else:
            network = self.ema_net

        # Encode the sigmas and combine with existing context info
        context = self.sigma_encoder(sigmas)
        if self.ctxt_dim:
            context = T.cat([context, ctxt], dim=-1)

        # Use the selected network to esitmate the noise present in the data.
        # If we are using cross-attnetion conditioning (e.g. for PIPPIN),
        # we need to pass the kv_seq and kv_mask to the network.
        # Else we pass the noisy data and mask to the selt-attention network.
        if kv_seq is not None:
            return network(
                q_seq=noisy_data,
                kv_seq=kv_seq,
                q_mask=mask,
                kv_mask=kv_mask,
                ctxt=context,
                add_seq=add_seq,
            )
        else:
            return network(noisy_data, mask=mask, ctxt=context)

    def _shared_step(self, sample: tuple, mode: str = "SA") -> T.Tensor:
        """Shared step used in both training and validaiton."""

        # Unpack the sample tuple
        if mode == "SA":
            nodes, mask, ctxt, pt = sample
            kv_seq, kv_mask = None, None
        elif mode == "CA":
            nodes, mask, kv_seq, kv_mask, add_seq, ctxt, pt = sample

        # Pass through the normalisers
        nodes = self.normaliser(nodes, mask)
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)

        # Sample sigmas using the Karras method of a log normal distribution
        sigmas = T.zeros(size=(nodes.shape[0], 1), device=self.device)
        sigmas.add_(self.p_mean + self.p_std * T.randn_like(sigmas))
        sigmas.exp_().clamp_(self.min_sigma, self.max_sigma)

        # Get the c values for the data scaling
        c_in, c_out, c_skip = self.get_c_values(append_dims(sigmas, nodes.dim()))

        # Sample from N(0, sigma**2)
        noises = T.randn_like(nodes) * append_dims(sigmas, nodes.dim())

        # Make the noisy samples by mixing with the real data
        noisy_nodes = nodes + noises

        # Pass through the just the base network (manually scale with c values)
        output = self.get_outputs(
                c_in * noisy_nodes,
                sigmas,
                mask,
                ctxt,
                kv_seq,
                kv_mask,
                add_seq,
                use_ema=False,
            )

        # Calculate the effective training target
        target = (nodes - c_skip * noisy_nodes) / c_out

        # Return the denoising loss (only non masked samples)
        mask = mask.unsqueeze(-1)
        return self.loss_fn(output*mask, target*mask).mean()

    @T.no_grad()
    def full_generation(
        self,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        initial_noise: Optional[T.Tensor] = None,
        kv_seq: Optional[T.Tensor] = None,
        kv_mask: Optional[T.BoolTensor] = None,
        add_seq: Optional[T.Tensor] = None,
        mode: str = "SA",
    ) -> T.Tensor:
        """Fully generate a batch of data from noise, given context information
        and a mask."""

        # Either a mask or initial noise must be defined or we dont know how
        # many samples to generate and with what cardinality
        if mask is None and initial_noise is None:
            raise ValueError("Please provide either a mask or noise to generate from")
        if mask is None:
            mask = T.full(initial_noise.shape[:-1], True, device=self.device)
        if initial_noise is None:
            initial_noise = (
                T.randn((*mask.shape, self.pc_dim), device=self.device) * self.max_sigma
            )

        # Normalise the context
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            assert len(ctxt) == len(initial_noise)

        # Generate the sigma values
        sigmas = self.sigma_function(self.min_sigma, self.max_sigma).to(self.device)

        # Run the sampler
        extra_args = {"ctxt": ctxt, "mask": mask}
        if mode == "CA":
            extra_args.update(
                {"kv_seq": kv_seq, "kv_mask": kv_mask, "add_seq": add_seq}
            )

        outputs = self.sampler_function(
            model=self,
            x=initial_noise,
            sigmas=sigmas,
            extra_args=extra_args,
        )

        # My samplers return 2 vars, k-diffusion returns 1
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        # Ensure that the output adheres to the mask
        outputs[~mask] = 0

        # Return the normalisation of the generated point cloud
        return self.normaliser.reverse(outputs, mask=mask)
