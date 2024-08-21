from typing import Mapping
from functools import partial
import torch as T
import lightning.pytorch as pl

from nflows import flows, distributions

from mattstools.mattstools.modules import IterativeNormLayer

from src.models.utils import masked_dequantize, masked_round
from src.models.utils import masked_logit, masked_expit
from src.models.utils import normal_cdf, normal_quantile


class MultiplicityFlow(pl.LightningModule):
    """Neural network to estimate the multplicity for the generator."""

    def __init__(
        self,
        inpt_dim: int,
        int_dims: list | None,
        ctxt_dim: int | None,
        invertible_net: partial,
        ctxt_net: partial,
        dequant_mode: str = "noise",
        dequant_scale: float = 1.0,
        dequant_distribution: str = "normal",
        do_logit: bool = False,
        logit_eps: float = 1e-2,
        cdf_eps: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        inpt_dim : int
            The dimension of the input data
        int_dims : list
            A list of bools which shows which inputs are intergers
        ctxt_dim : int
            The dimension of the context data
        invertible_net : partial
            The configuration for creating the invertible neural network
        ctxt_net : partial
            For setting up the shared context extractor
        dequant_mode : str (default: 'noise')
            The mode for dequantisation. Options are 'noise' or 'cdf'.
        dequant_scale : float (default: 1.0)
            The scale of the dequantisation noise
        dequant_distribution : str (default: 'normal')
            The distribution of the dequantisation noise
        do_logit : bool (default: False)
            Whether to apply the logit function to the data
        logit_eps : float (default: 1e-2)
            The epsilon threshold value for the logit function
        cdf_eps : float (default: 1e-6)
            The epsilon threshold value for the CDF dequantisation
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.inpt_dim = inpt_dim
        self.int_dims = int_dims
        self.ctxt_dim = ctxt_dim
        self.dequant_mode = dequant_mode
        self.dequant_scale = dequant_scale
        self.dequant_distribution = dequant_distribution
        self.do_logit = do_logit
        self.logit_eps = logit_eps
        self.cdf_eps = cdf_eps

        # The dequantisation layer for pre-processing
        if self.dequant_mode == "cdf":
            self.dequantiser = CDFDequantization(
                size=[4, 3, 18],
                eps=self.cdf_eps,
            )
        else:
            self.dequantiser = None

        # The normalisation layer for pre-processing
        self.inpt_normaliser = IterativeNormLayer(self.inpt_dim)
        if self.ctxt_dim:
            self.ctxt_normaliser = IterativeNormLayer(self.ctxt_dim)
            self.ctxt_net = ctxt_net(self.ctxt_dim)

        # The flow itself
        self.flow = flows.Flow(
            invertible_net(
                xz_dim=self.inpt_dim,
                ctxt_dim=self.ctxt_net.outp_dim if self.ctxt_dim else 0,
            ),
            distributions.StandardNormal([self.inpt_dim]),
        )

    def get_nll(self, inpt: T.Tensor, ctxt: T.Tensor = None) -> T.Tensor:
        """
        Compute the negative log liklihood of the sample under the flow.

        Parameters
        ----------
        inpt : Tensor
            The input data
        ctxt : Tensor (default: None)
            The context data

        Returns
        -------
        nll : Tensor
            The negative log liklihood of the sample under the flow
        """

        if self.dequantiser is None:
            # Add noise to the input to dequantise
            inpt = masked_dequantize(
                inpt,
                mask=self.int_dims,
                scale=self.dequant_scale,
                distribution=self.dequant_distribution,
            )

        # Normalise the scale (preprocess)
        if self.dequantiser is None:
            inpt = self.inpt_normaliser(inpt)

        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            ctxt = self.ctxt_net(ctxt)

        if self.dequantiser is None:
            # Apply the logit function
            if self.do_logit:
                inpt = masked_logit(
                    inpt,
                    mask=self.int_dims,
                    eps=self.logit_eps,
                )
        else:
            # Dequantize the input (using the CDFDequantization algorithm)
            inpt = self.dequantiser(inpt)

        # Calculate the negative log liklihood
        nll = -self.flow.log_prob(inpt, ctxt).mean()

        return nll

    def generate(self, n_points: int = 1, ctxt: T.Tensor | None = None) -> T.Tensor:
        """
        Sample from the flow, undo the scaling and the dequantisation.

        Parameters
        ----------
        n_points : int (default: 1)
            The number of points to generate
        ctxt : Tensor (default: None)
            The context data

        Returns
        -------
        gen : Tensor
            The generated data
        """

        # Normalise the context
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            ctxt = self.ctxt_net(ctxt)

        # Sample from the flow
        gen = self.flow.sample(n_points, ctxt)

        if self.dequantiser is None:
            # Apply the expit function (inverse of logit)
            if self.do_logit:
                gen = masked_expit(gen, mask=self.int_dims)

            # Unnormalise the scale
            gen = self.inpt_normaliser.reverse(gen)

        # Reshape to the correct shape
        if ctxt is not None:
            gen = gen.squeeze(1)

        if self.dequantiser is None:
            # Round to the nearest integer
            gen = masked_round(gen, mask=self.int_dims)

        else:
            # Quantise the generated data
            gen = self.dequantiser.inverse(gen)

        return gen


# CDFDequantization from https://arxiv.org/abs/2403.15782
# See https://github.com/simonschnake/CaloPointFlow
class CDFDequantization(pl.LightningModule):
    """
    Dequantize the data.
    Adds uniform noise to each element of the tensor.
    (See https://arxiv.org/abs/2403.15782)
    """
    def __init__(
        self,
        size: T.Size | tuple[int, ...] | int,
        eps: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        size : tuple
            The size of the input tensor
        eps : float (default: 1e-6)
            The epsilon threshold value for the CDF dequantisation
        """

        super().__init__()

        self.size = size
        self.eps = eps

        for i, s in enumerate(self.size):
            self.register_buffer(f'pdf_{i}', T.zeros(s))
            self.register_buffer(f'cdf_{i}', T.zeros(s))
        self.register_buffer("count", T.tensor(0, dtype=T.long))

        self.register_buffer('_frozen', T.tensor(False))

    def forward(self, x: T.LongTensor) -> T.FloatTensor:
        """
        Normalize the input tensor.

        Args:
            x (Tensor): The input tensor to normalize.

        Returns:
            y (Tensor): The normalized input tensor.
        """
        if not self.frozen:
            # Update the mean and variance with the input tensor
            self._update(x)

        y = T.empty(x.size(), dtype=T.float32, device=x.device)

        # Apply the CDF dequantization
        for i in range(len(self.size)):
            y[:, i] = self.cdf[i][x[:, i]]
            y[:, i] += self.pdf[i][x[:, i]] * T.rand_like(y[:, i])

        # Transform to a normal distribution (apply the normal quantile)
        y = normal_quantile(y, eps=self.eps)

        return y

    def inverse(self, y : T.FloatTensor) -> T.LongTensor:
        """
        Inverse normalize the input tensor.

        Args:
            y (Tensor): The input tensor to inverse normalize.

        Returns:
            x (Tensor): The inverse normalized input tensor.
        """
        # Transform to a uniform distribution (apply the normal CDF)
        y = normal_cdf(y, eps=self.eps)

        x = T.empty(y.size(), dtype=T.long, device=y.device)

        # Apply the inverse CDF dequantization
        for i in range(len(self.size)):
            x[:, i] = T.searchsorted(self.cdf[i], y[:, i].contiguous()) - 1

        # There is a extremely small chance that y is less than 0
        T.clamp_min_(x, 0)

        return x

    @property
    def pdf(self):
        return [getattr(self, f"pdf_{i}") for i in range(len(self.size))]

    @property
    def cdf(self):
        return [getattr(self, f"cdf_{i}") for i in range(len(self.size))]

    def _update(self, x: T.LongTensor) -> None:
        """
        Update the pdf and cdf.
        """
        self._update_pdf(x)
        self._calculate_cdf()

    def _update_pdf(self, x: T.LongTensor) -> None:
        """
        Update the pdf.
        """
        new_count = x.size(0)
        for i, s in enumerate(self.size):
            hist = T.bincount(x[:, i], minlength=s)
            self.pdf[i] *= (self.count / (self.count + new_count))
            self.pdf[i] += hist / (new_count + self.count)

        self.count += new_count

    def _calculate_cdf(self) -> None:
        """
        Calculate the cdf.
        """
        for i in range(len(self.size)):
            cspdf = T.cumsum(self.pdf[i], dim=0)
            self.cdf[i][1:] = cspdf[:-1]

    @property
    def frozen(self) -> bool:
        return self._frozen.item()

    def freeze(self) -> None:
        self._frozen = T.tensor(True)
