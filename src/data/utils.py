from typing import Tuple, List
import numpy as np
import torch as T


# Types conversion
def structured_array_to_tensor(x: np.ndarray) -> T.Tensor:
    """Converts a structured Numpy array to a PyTorch float tensor"""
    return T.tensor(x.view((np.float32, len(x.dtype.names))))


# Generic pre-processing functions
def log_squash(x: T.Tensor) -> T.Tensor:
    """Apply the log-squashing function to a PyTorch tensor"""
    return T.sign(x) * T.log(T.abs(x) + 1)


def inv_log_squash(x: T.Tensor) -> T.Tensor:
    """Apply the inverse log-squashing function to a PyTorch tensor"""
    return T.sign(x) * (T.exp(T.abs(x)) - 1)


def pad_with_zeros(
    data: T.Tensor,
    max_n: int,
    dim: int = 1,
    before: bool = False,
) -> T.Tensor | T.BoolTensor:
    """
    Pads the data with zeros to the given number of objects.

    Args:
        data:
            The data to pad.
        max_n:
            The number of objects to pad to.

    Kwargs:
        dim:
            The dimension to pad along. (default: 1)
        before:
            Whether to pad before the data. (default: False)

    Returns:
        The padded data.
    """

    pad_shape = list(data.shape)
    pad_shape[dim] = max_n - data.shape[dim]
    pad_tensor = T.zeros(pad_shape, dtype=data.dtype, device=data.device)

    if before:
        return T.cat([pad_tensor, data], dim=1)
    else:
        return T.cat([data, pad_tensor], dim=1)


def cat_with_options(
    data_list: List[T.Tensor],
    equal_length: bool = False,
    mixed_shapes: bool = False,
) -> T.Tensor:
    """
    Concatenates the given data list.
    Optionally equalises lengths (dim=0) and pads objects (dim=1).

    Args:
        data_list:
            The list of data to concatenate.

    Kwargs:
        equal_length:
            Whether to equalise the lengths of the data. (default: False)
        mixed_shapes:
            Whether to pad the objects to the same shape. (default: False)

    Returns:
        The concatenated data.
    """

    if equal_length:
        min_len = min([data.shape[0] for data in data_list])
        data_list = [data[:min_len] for data in data_list]

    if mixed_shapes:
        max_n = max([data.shape[1] for data in data_list])
        data_list = [pad_with_zeros(data, max_n) for data in data_list]

    return T.cat(data_list, dim=0)


# Data specific pre-processing functions
def matchability_to_presence(matchability: T.ByteTensor) -> T.BoolTensor:
    """
    Converts a matchability tensor to a presence tensor.

    matchability:
        shape: (batch_size,)
        range: [0, 63]

    presence:
        shape: (batch_size, 6)
        range: {True, False}
    """
    exponents = T.arange(5, -1, -1, dtype=T.uint8, device=matchability.device)
    presence = T.floor_divide(
        matchability.view(-1, 1),
        2**exponents.view(1, -1),
    )
    return T.fmod(presence, 2).bool()


def presence_to_matchability(presence: T.BoolTensor) -> T.ByteTensor:
    """
    Converts a presence tensor to a matchability tensor.

    presence:
        shape: (batch_size, 6)
        range: {True, False}

    matchability:
        shape: (batch_size,)
        range: [0, 63]
    """
    exponents = T.arange(5, -1, -1, dtype=T.uint8, device=presence.device)
    matchability = T.where(
        presence.view(-1, 6),
        2**exponents.view(1, -1),
        0,
    )
    return T.sum(matchability, dim=1, dtype=T.uint8)


# Data specific utility functions
def get_particle_pos(
    channel: T.ByteTensor,
    level: str = "reco",
    max_n: List[int] = [2, 1, 16],
    dataset_name: str = "inclusive",
) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    """
    Get the positions of the particles of a given type for a given decay channel.

    Args:
        channel:
            The decay channel of the events. (0b00, 0b01, 0b10, 0b11)

    Kwargs:
        level:
            The level of the particles to consider. ('part', 'reco')
        max_n:
            The maximum number of reco-level particles of each type.
            Only useful if level is 'reco'.
        dataset_name:
            The name of the dataset. ('inclusive')
            Only useful if level is 'reco'.

    Returns:
        pos_lep:
            The positions of the leptons.
        pos_met:
            The positions of the METs.
        pos_jet:
            The positions of the jets.
    """

    # Get the batch size and the number of particles
    bs = len(channel)
    ps = 6 if level == "part" else sum(max_n)

    # Initialise the positions as False
    pos_lep = T.zeros(size=(bs, ps), dtype=T.bool, device=channel.device)
    pos_met = T.zeros(size=(bs, ps), dtype=T.bool, device=channel.device)
    pos_jet = T.zeros(size=(bs, ps), dtype=T.bool, device=channel.device)

    if level == "part":
        # Channel 0: fully hadronic
        pos_jet[channel == 0b00, :] = True

        # Channel 1: semi leptonic (top)
        pos_lep[channel == 0b01, 1] = True
        pos_met[channel == 0b01, 2] = True
        pos_jet[channel == 0b01, [[0], [3], [4], [5]]] = True

        # Channel 2: semi leptonic (anti-top)
        pos_lep[channel == 0b10, 4] = True
        pos_met[channel == 0b10, 5] = True
        pos_jet[channel == 0b10, [[0], [1], [2], [3]]] = True

        # Channel 3: fully leptonic
        pos_lep[channel == 0b11, [[1], [4]]] = True
        pos_met[channel == 0b11, [[2], [5]]] = True
        pos_jet[channel == 0b11, [[0], [3]]] = True

    elif level == "reco":
        if dataset_name == "inclusive":
            pos_lep[:, :2] = True
            pos_met[:, 2] = True
            pos_jet[:, 3:] = True
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    return pos_lep, pos_met, pos_jet


def get_particle_masks(
    mask: T.BoolTensor,
    channel: T.ByteTensor,
    level: str = "reco",
    max_n: List[int] = [2, 1, 16],
    dataset_name: str = "inclusive",
) -> Tuple[T.BoolTensor, T.BoolTensor, T.BoolTensor]:
    """
    Get the masks for the particles of a given type for a given decay channel.

    Args:
        mask:
            The mask for the whole point cloud.
        channel:
            The channel of the events. (0b00, 0b01, 0b10, 0b11)

    Kwargs:
        level:
            The level of the particles to consider. ('reco', 'part')
        max_n:
            The maximum number of reco-level particles of each type.
            Only useful if level is 'reco'.
        dataset_name:
            The name of the dataset. ('inclusive')
            Only useful if level is 'reco'.

    Returns:
        mask_lep:
            The mask for the leptons.
        mask_met:
            The mask for the METs.
        mask_jet:
            The mask for the jets.
    """

    pos_lep, pos_met, pos_jet = get_particle_pos(
        channel=channel,
        level=level,
        max_n=max_n,
        dataset_name=dataset_name,
    )

    mask_lep = pos_lep & mask
    mask_met = pos_met & mask
    mask_jet = pos_jet & mask

    return mask_lep, mask_met, mask_jet


def get_particle_mult(
    mask: T.BoolTensor,
    channel: T.ByteTensor,
    level: str = "reco",
    max_n: List[int] = [2, 1, 16],
    dataset_name: str = "inclusive",
) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    """
    Get the multiplicity for all particle types.

    Args:
        mask:
            The mask for the whole point cloud.
        channel:
            The channel of the events. (0b00, 0b01, 0b10, 0b11)

    Kwargs:
        level:
            The level of the particles to consider. ('reco', 'part')
        max_n:
            The maximum number of reco-level particles of each type.
            Only useful if level is 'reco'.
        dataset_name:
            The name of the dataset. ('inclusive')
            Only useful if level is 'reco'.

    Returns:
        n_lep:
            The number of leptons.
        n_met:
            The number of METs.
        n_jet:
            The number of jets.
    """

    masks = get_particle_masks(
        mask=mask,
        channel=channel,
        level=level,
        max_n=max_n,
        dataset_name=dataset_name,
    )

    mask_lep, mask_met, mask_jet = masks

    n_lep = mask_lep.sum(dim=-1, keepdim=True).float()
    n_met = mask_met.sum(dim=-1, keepdim=True).float()
    n_jet = mask_jet.sum(dim=-1, keepdim=True).float()

    return n_lep, n_met, n_jet


def get_mask_turbo(
    masks: T.BoolTensor | List[T.BoolTensor],
    channel: T.ByteTensor,
    n_max: int | None = None,
):
    """
    Get a mask that mimic the Turbo-Sim (OTUS) dataset by:
    - Keeping only semi-leptonic events
    - Keeping only events with at least 1 lepton and 4 jets
    - Optionally keeping only the first 'n_max' allowed events

    Args:
        masks:
            The mask for the whole point cloud.
        channel:
            The channel of the events. (0b00, 0b01, 0b10, 0b11)

    Kwargs:
        n_max:
            The maximum number of events to keep. (default: None)

    Returns:
        mask_turbo:
            The restrained mask for the dataset.
    """

    if not isinstance(masks, list):
        masks = [masks]

    # Only keep semi-leptonic events
    mask_chan = (channel == 0b01) | (channel == 0b10)

    # Only keep events with 1 lepton, 1 MET and 4 jets
    mask_mult = T.ones_like(mask_chan, dtype=T.bool)
    for mask in masks:
        n_lep, n_met, n_jet = get_particle_mult(mask, channel)
        mask_mult &= ((n_lep >= 1) & (n_jet >= 4)).squeeze()

    # Combine the masks
    mask_turbo = mask_chan & mask_mult

    # Only keep the first n_max True events
    if n_max is not None:
        mask_turbo[T.cumsum(mask_turbo, dim=0) > n_max] = False

    return mask_turbo
