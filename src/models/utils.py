import torch as T

# from pcjedi.src.models.utils import plot_mpgan_marginals  # Backward compatibility


def masked_dequantize(
    inputs: T.Tensor,
    mask: list,
    scale: float = 1.0,
    distribution: str = "normal",
) -> T.Tensor:
    """
    Add noise to the final dimension of a tensor only where the mask is 'True'.

    Arguments:
        inputs (Tensor): The input tensor.
        mask (list): The mask that determines where to add noise.
        scale (float, optional): The scale of the noise. Default is 1.0.
        distribution (str, optional): The distribution of the noise. Default is "normal".

    Returns:
        out (Tensor): The dequantized tensor.
    """
    inputs = inputs.clone()
    if distribution == "normal":
        noise = T.randn_like(inputs[..., mask])  # Tested and 1 KDE looks good!
    elif distribution == "uniform":
        noise = T.rand_like(inputs[..., mask]) * 2 - 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    inputs[..., mask] = inputs[..., mask] + noise * scale
    return inputs


def masked_round(
    inputs: T.Tensor,
    mask: list,
) -> T.Tensor:
    """
    Round to int the final dimension of a tensor only where the mask is 'True'.

    Arguments:
        inputs (Tensor): The input tensor.
        mask (list): The mask that determines where to round.

    Returns:
        out (Tensor): The rounded tensor
    """
    inputs = inputs.clone()
    inputs[..., mask] = T.round(inputs[..., mask])
    return inputs


def masked_logit(
    inputs: T.Tensor,
    mask: list,
    eps: float = 1e-2,
) -> T.Tensor:
    """
    Apply the logit function to the final dimension of a tensor only where the mask is 'True'.

    Arguments:
        inputs (Tensor): The input tensor.
        mask (list): The mask that determines where to apply the logit function.
        eps (float, optional): A small value to avoid division by zero. Default is 1e-2.

    Returns:
        out (Tensor): The logit-transformed tensor.
    """
    inputs = inputs.clone()
    inputs[..., mask] = T.special.logit(inputs[..., mask], eps=eps)
    return inputs


def masked_expit(
    inputs: T.Tensor,
    mask: list,
) -> T.Tensor:
    """
    Apply the expit (sigmoid) function to the final dimension of a tensor only where the mask is 'True'.

    Arguments:
        inputs (Tensor): The input tensor.
        mask (list): The mask that determines where to apply the expit function.

    Returns:
        out (Tensor): The expit-transformed tensor.
    """
    inputs = inputs.clone()
    inputs[..., mask] = T.special.expit(inputs[..., mask])
    return inputs


def shrink(x: T.Tensor, eps: float = 1e-6):
    """
    Shrinks the range of the input tensor to [eps, 1-eps].

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        out (Tensor): The shrunk tensor.
    """
    T.clamp_(x, min=0, max=1)
    return x * (1 - 2 * eps) + eps


def expand(x: T.Tensor, eps: float = 1e-6):
    """
    Expands the range of the input tensor to [0, 1].

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        out (Tensor): The expanded tensor
    """
    x = (x - eps) / (1 - 2 * eps)
    return T.clamp_(x, min=0, max=1)


def normal_quantile(x: T.Tensor, eps=1e-6):
    """
    Applies the quantile function of the normal distribution (mu=0, sigma=1) to
    the input tensor.
    q(x): [0, 1] -> [-inf, inf]

    The quantile function is the inverse of the CDF.

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        out (Tensor): The normal quantile-transformed tensor.
    """
    x =  shrink(x, eps)
    return T.special.ndtri(x)


def normal_cdf(x: T.Tensor, eps=1e-6):
    """
    Applies the CDF of the normal distribution (mu=0, sigma=1) to the input tensor.
    cdf(x): [-inf, inf] -> [0, 1] 

    The CDF is the inverse of the quantile function.

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        out (Tensor): The normal CDF-transformed tensor.
    """
    x = T.special.ndtr(x)
    return expand(x, eps)
