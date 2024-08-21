from typing import Tuple
from functools import cached_property

import torch as T

from src.data.utils import presence_to_matchability


class Part2RecoMatching():
    """Container class for the mapping from partons to reconstructed objects."""

    def __init__(
        self,
        part: T.Tensor,
        reco: T.Tensor,
        mask_part: T.BoolTensor | None = None,
        mask_reco: T.BoolTensor | None = None,
        R: float = 0.4,
    ):
        """
        Args:
            part:
                Partons point cloud, with shape (batch_size, n_part, 4)
            reco:
                Reconstructed objects point cloud, with shape (batch_size, n_reco, 4)

        Kwargs:
            mask_part:
                Mask for the real partons, with shape (batch_size, n_part)
            mask_reco:
                Mask for the real reconstructed objetcs, with shape (batch_size, n_reco)
            R:
                Matching radius in the dR space
        """

        self.part = part
        self.reco = reco
        self.mask_part = mask_part
        self.mask_reco = mask_reco
        self.R = R

    @cached_property
    def dR(self) -> T.Tensor:
        # Extract the eta and phi coordinates
        eta_part = self.part[..., 1]
        phi_part = self.part[..., 2]
        eta_reco = self.reco[..., 1]
        phi_reco = self.reco[..., 2]

        # Compute the dR between all reconstructed objects and all partons
        dEta = eta_part.unsqueeze(-1) - eta_reco.unsqueeze(-2)
        dPhi = (phi_part.unsqueeze(-1) - phi_reco.unsqueeze(-2)) % (2 * T.pi)
        dPhi = T.minimum(dPhi, 2 * T.pi - dPhi)
        dR = T.sqrt(dEta * dEta + dPhi * dPhi)

        # Discard the dR comming from unreal partons and reconstructed objects
        if self.mask_part is not None:
            dR = dR.masked_fill(
                self.mask_part.unsqueeze(-1) == False,
                float("inf"),
            )
        if self.mask_reco is not None:
            dR = dR.masked_fill(
                self.mask_reco.unsqueeze(-2) == False,
                float("inf"),
            )

        return dR

    @cached_property
    def mapping(self) -> T.LongTensor:
        # Find the closest reconstructed object for each parton
        # /!\ Some reconstructed objects may be matched to multiple partons
        # and vice-versa
        # The 'is_multi_matched' property can be used to check for this
        mapping = self.dR.argmin(dim=-1)

        # Assign -1 if no reconstructed object is found with dR < R
        mapping = mapping.masked_fill(self.dR.min(dim=-1).values >= self.R, -1)

        return mapping

    @cached_property
    def has_multi_part(self) -> T.BoolTensor:
        # Find the dR less than the given R threshold
        dR_mask = self.dR < self.R

        # Check if there are multiple partons matched to a single reco. object
        return (dR_mask.sum(dim=1) > 1).any(dim=-1)

    @cached_property
    def has_multi_reco(self) -> T.BoolTensor:
        # Find the dR less than the given R threshold
        dR_mask = self.dR < self.R

        # Check if there are multiple reco. objects matched to a single parton
        return (dR_mask.sum(dim=2) > 1).any(dim=-1)

    @cached_property
    def is_multi_matched(self) -> T.BoolTensor:
        return self.has_multi_part | self.has_multi_reco

    @cached_property
    def is_matched(self) -> T.BoolTensor:
        return (self.dR < self.R).any(dim=2)

    @cached_property
    def is_all_matched(self) -> T.BoolTensor:
        return self.is_matched.all(dim=-1)

    @cached_property
    def matchability(self) -> T.ByteTensor:
        presence = (self.mapping != -1).unsqueeze(-1)
        matchability = presence_to_matchability(presence)
        matchability[self.is_multi_matched] = 0
        return matchability


def compute_observables(
    pc: T.Tensor,
    mask: T.BoolTensor,
    pc_ref: T.Tensor | None = None,
    mask_ref: T.BoolTensor | None = None,
    channel: T.ByteTensor | None = None,
    is_parton: bool = False,
    matching_R: float = 0.4,
    accept_multi_part: bool = False,
    accept_multi_reco: bool = False,
) -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.BoolTensor, T.Tensor]:
    """
    Computes several observables for several underlying systems in a given
    particle cloud.

    Observables:
        - Invariant masses
        - Transverse momenta
        - Pseudo-rapidities

    The order of the mapping must be the following:
        - 0: b-quark #1
        - 1: particle #1 from W-boson #1
        - 2: particle #2 from W-boson #1
        - 3: b-quark #2
        - 4: particle #1 from W-boson #2
        - 5: particle #2 from W-boson #2

    The order of the output masses is the following:
        - 0: W-boson #1
        - 1: W-boson #2
        - 2: Top-quark #1
        - 3: Top-quark #2
        - 4: Top-pair

    Args:
        pc:
            Input point cloud, with shape (batch_size, n_obj, 4)
        mask:
            Mask for the real particles, with shape (batch_size, n_obj)

    Kwargs:
        pc_ref:
            Reference point cloud, with shape (batch_size, n_part, 4)
        mask_ref:
            Mask for the real particles in the reference point cloud, with shape (batch_size, n_part)
        channel:
            Decay channel of the event, with shape (batch_size,)
        is_parton:
            Whether the input point cloud 'pc' contains partons or reconstructed objects
        matching_R:
            Matching radius in the dR space. Default is 0.4.
        accept_multi_part:
            Whether to accept events with multiple partons matched to the same
            reconstructed object. The closest match in dR space is kept.
            Default is False.
        accept_multi_reco:
            Whether to accept events with multiple reconstructed objects
            matched to the same parton. The closest match in dR space is kept.
            Default is False.

    Returns:
        masses:
            Invariant masses of the underlying particles, with shape (batch_size, 5)
        momenta:
            Transverse momenta of the underlying particles, with shape (batch_size, 5)
        rapidities:
            Pseudo-rapidities of the underlying particles, with shape (batch_size, 5)
        check:
            Mask for the events for which partons are uniquely matched to a reconstructed objects, with shape (batch_size,)
        percent:
            Percentage of reconstructed objects that are uniquely matched to a parton
        match:
            Matchability of the events, with shape (batch_size,)
    """

    if not is_parton:
        # Sanity check
        if pc_ref is None or mask_ref is None or channel is None:
            raise ValueError("'pc_ref', 'mask_ref' and 'channel' must be given if 'is_parton' is False")

        # Add regressed MET to the point cloud for the matching to work
        pc, mask = get_fake_neutrinos(
            pc=pc,
            mask=mask,
            channel=channel,
            pc_ref=pc_ref,
            mask_ref=mask_ref,
        )

        part2reco = Part2RecoMatching(
            part=pc_ref,
            reco=pc,
            mask_part=mask_ref,
            mask_reco=mask,
            R=matching_R,
        )
        mapping = part2reco.mapping
        if accept_multi_part and accept_multi_reco:
            check = part2reco.is_all_matched
        elif accept_multi_part:
            check = part2reco.is_all_matched & ~part2reco.has_multi_reco
        elif accept_multi_reco:
            check = part2reco.is_all_matched & ~part2reco.has_multi_part
        else:
            check = part2reco.is_all_matched & ~part2reco.is_multi_matched
        percent = check.float().mean().unsqueeze(0)
        match = part2reco.matchability

        # Replace -1 values with 0 in the mapping to avoid errors
        # Those events can be discarded later anyway, using the 'check' mask
        # returned by the 'match_reco2part' function above
        mapping = mapping.masked_fill(mapping == -1, 0)

        # Reorder the point cloud to match the mapping
        pc = pc.gather(
            dim=1,
            index=mapping.unsqueeze(-1).expand(*mapping.shape, pc.shape[-1]),
        )

    else:
        check = None
        percent = None
        match = None

    # Convert to Cartesian coordinates
    pc = to_cartesian(pc, has_mass=is_parton)

    # Extract the four-momenta of the desired underlying particles
    p4W1 = pc[:, 1, :] + pc[:, 2, :]  # W-boson #1
    p4W2 = pc[:, 4, :] + pc[:, 5, :]  # W-boson #2
    p4t1 = pc[:, 0, :] + p4W1         # Top-quark #1
    p4t2 = pc[:, 3, :] + p4W2         # Top-quark #2
    p4tt = p4t1 + p4t2                # Top-pair

    # Compute the invariant masses of all systems
    minkowski = T.tensor([-1, -1, -1, 1], dtype=T.float32, device=pc.device)
    minkowski = minkowski.repeat(pc.shape[0], 1)
    mW1 = (minkowski * p4W1**2).sum(dim=-1, keepdim=True).sqrt()
    mW2 = (minkowski * p4W2**2).sum(dim=-1, keepdim=True).sqrt()
    mt1 = (minkowski * p4t1**2).sum(dim=-1, keepdim=True).sqrt()
    mt2 = (minkowski * p4t2**2).sum(dim=-1, keepdim=True).sqrt()
    mtt = (minkowski * p4tt**2).sum(dim=-1, keepdim=True).sqrt()

    # Compute the transverse momenta of all systems
    pTW1 = T.linalg.norm(p4W1[..., :2], dim=-1, keepdim=True)
    pTW2 = T.linalg.norm(p4W2[..., :2], dim=-1, keepdim=True)
    pTt1 = T.linalg.norm(p4t1[..., :2], dim=-1, keepdim=True)
    pTt2 = T.linalg.norm(p4t2[..., :2], dim=-1, keepdim=True)
    pTtt = T.linalg.norm(p4tt[..., :2], dim=-1, keepdim=True)

    # Convert back to pseudo-rapidities
    etaW1 = T.atanh(p4W1[..., 2].unsqueeze(-1) / T.linalg.norm(p4W1[..., :3], dim=-1, keepdim=True))
    etaW2 = T.atanh(p4W2[..., 2].unsqueeze(-1) / T.linalg.norm(p4W2[..., :3], dim=-1, keepdim=True))
    etat1 = T.atanh(p4t1[..., 2].unsqueeze(-1) / T.linalg.norm(p4t1[..., :3], dim=-1, keepdim=True))
    etat2 = T.atanh(p4t2[..., 2].unsqueeze(-1) / T.linalg.norm(p4t2[..., :3], dim=-1, keepdim=True))
    etatt = T.atanh(p4tt[..., 2].unsqueeze(-1) / T.linalg.norm(p4tt[..., :3], dim=-1, keepdim=True))

    # Combine the observables into single tensors
    masses = T.cat([mW1, mW2, mt1, mt2, mtt], dim=-1)
    momenta = T.cat([pTW1, pTW2, pTt1, pTt2, pTtt], dim=-1)
    rapidities = T.cat([etaW1, etaW2, etat1, etat2, etatt], dim=-1)

    return masses, momenta, rapidities, check, percent, match


def get_fake_neutrinos(
    pc: T.Tensor,
    mask: T.BoolTensor,
    pc_ref: T.Tensor,
    mask_ref: T.BoolTensor,
    channel: T.ByteTensor,
) -> Tuple[T.Tensor, T.BoolTensor]:
    """
    Add fake neutrinos to the input point cloud based on MET and true neutrtinos.

    Args:
        pc:
            Input point cloud, with shape (batch_size, n_obj, 4)
        mask:
            Mask for the real particles, with shape (batch_size, n_obj)
        pc_ref:
            Reference point cloud, with shape (batch_size, n_part, 4)
        mask_ref:
            Mask for the real particles in the reference point cloud, with shape (batch_size, n_part)
        channel:
            Decay channel of the event, with shape (batch_size,)

    Returns:
        pc:
            Input point cloud with 2 fake neutrinos added, with shape (batch_size, n_obj + 2, 4)
        mask:
            Mask with 2 fake neutrinos added, with shape (batch_size, n_obj + 2)
    """

    if pc_ref is None or mask_ref is None:
        raise ValueError("'pc_ref' and 'mask_ref' must be given")

    bs = pc.shape[0]
    fs = pc.shape[-1]
    new_part = T.zeros(size=(bs, 2, fs), dtype=pc.dtype, device=pc.device)
    new_mask = T.zeros(size=(bs, 2), dtype=mask.dtype, device=mask.device)

    # Compute phi and energy of the MET in pc given the neutrinos in pc_ref
    # The method is different for each channel
    # Channel 0b01 & 0b10: semi leptonic (1 neutrino)
    new_part[channel == 0b01, 0] = regress_met(
        met=pc[channel == 0b01, 2],
        neutrinos=pc_ref[channel == 0b01, 2],
        n_neutrinos=1,
    )
    new_part[channel == 0b10, 0] = regress_met(
        met=pc[channel == 0b10, 2],
        neutrinos=pc_ref[channel == 0b10, 5],
        n_neutrinos=1,
    )
    new_mask[channel == 0b01, 0] = mask[channel == 0b01, 2]
    new_mask[channel == 0b10, 0] = mask[channel == 0b10, 2]

    # Channel 0b11: fully leptonic (2 neutrinos)
    new_part[channel == 0b11, 0], new_part[channel == 0b11, 1] = regress_met(
        met=pc[channel == 0b11, 2],
        neutrinos=pc_ref[channel == 0b11][:, [2, 5]],
        n_neutrinos=2,
    )
    new_mask[channel == 0b11, 0] = mask[channel == 0b11, 2]
    new_mask[channel == 0b11, 1] = mask[channel == 0b11, 2]

    # Mask the old MET not to create duplicates
    mask = mask.clone()
    mask[channel == 0b01, 2] = False
    mask[channel == 0b10, 2] = False
    mask[channel == 0b11, 2] = False

    # Concatenate the new particles and mask to the point cloud and mask
    pc = T.cat([pc, new_part], dim=1)
    mask = T.cat([mask, new_mask], dim=1)

    return pc, mask


def regress_met(
    met: T.Tensor,
    neutrinos: T.Tensor,
    n_neutrinos: int,
) -> T.Tensor | Tuple[T.Tensor, T.Tensor]:
    """
    Compute the fake neutrinos kinematics from the MET and the true neutrinos.
    """

    # Do not modify the input tensors
    met = met.clone()

    # If 1 neutrino, the fake particle is:
    # MET with parton-level eta and energy computed from pT and eta
    if n_neutrinos == 1:

        # Transform the MET with parton-level eta and corresponding energy
        met[..., 1] = neutrinos[..., 1]  # parton-level eta
        met[..., 3] = met[..., 0] * T.cosh(met[..., 1])  # = |p| ~= energy (no mass)

        return met

    # If 2 neutrinos, the fake particles are:
    # Highest pT neutrino and remaining MET (transformed as in 1 neutrino case)
    elif n_neutrinos == 2:

        # Order the neutrinos and convert to Cartesian coordinates
        neutrinos = order_by_pt(neutrinos)
        neutrinos = to_cartesian(neutrinos, has_mass=True)
        met = to_cartesian(met)

        # Transform the MET with parton-level eta and corresponding energy
        met_true = neutrinos[:, 0] + neutrinos[:, 1]
        met[..., 1] = met_true[..., 1]  # parton-level eta
        met[..., 3] = met[..., 0] * T.cosh(met[..., 1])  # = |p| ~= energy (no mass)

        # Get the remaining MET after removing the highest pT neutrino
        met = met - neutrinos[:, 0]

        return to_polar(neutrinos[:, 0]), to_polar(met)


def order_by_pt(
    pc: T.Tensor,
    mask: T.BoolTensor | None = None,
    is_cartesian: bool = False,
    descending: bool = True,
) -> T.Tensor:
    """
    Orders the particles in the given point cloud by decreasing pT.

    i.e., Sort the input Tensor along the second dimension, using the values of
    the first column of the third dimension.

    /!\ The mask is not updated. The output will contain unphysical particles
    of 0 pT.

    Args:
        pc:
            Input point cloud, with shape (batch_size, n_particles, 4)
        mask:
            Mask for the real particles, with shape (batch_size, n_particles)

    Kwargs:
        is_cartesian:
            Whether the point cloud is in Cartesian coordinates (px, py, pz, energy)
            or in polar coordinates (pT, eta, phi, mass|energy)
        descending:
            Whether to sort in descending order (default) or ascending order

    Returns:
        pc_ordered:
            Ordered point cloud, with shape (batch_size, n_particles, 4)
    """

    # Extract the pT values
    if is_cartesian:
        px = pc[..., 0]
        py = pc[..., 1]
        pt = T.sqrt(px**2 + py**2)
    else:
        pt = pc[..., 0]

    if mask is not None:
        pt = pt * mask

    # Sort the particles by pT values
    _, indices = pt.sort(dim=-1, descending=descending)

    # Reorder the point cloud
    pc_ordered = pc.gather(
        dim=1,
        index=indices.unsqueeze(-1).expand(*indices.shape, pc.shape[-1]),
    )

    return pc_ordered


def to_cartesian(p4: T.Tensor, has_mass: bool = False) -> T.Tensor:
    """
    Convert four-momenta from polar to Cartesian coordinates

    Args:
        p4: The four-momenta in polar coordinates (pT, eta, phi, mass|energy)

    Kwargs:
        has_mass: Whether the four-momenta contain masses (otherwise energies)

    Returns:
        The four-momenta in Cartesian coordinates (px, py, pz, energy)
    """

    # Extract the pT, eta and phi coordinates
    pt = p4[..., 0]
    eta = p4[..., 1]
    phi = p4[..., 2]

    # Convert to cartesian coordinates
    px = pt * T.cos(phi)
    py = pt * T.sin(phi)
    pz = pt * T.sinh(eta)

    # Extract or calculate the energy
    if has_mass:
        mass = p4[..., 3]
        energy = T.sqrt(px**2 + py**2 + pz**2 + mass**2)  # Or (pT*cosh(eta))**2 + mass**2
    else:
        energy = p4[..., 3]

    # Gather in a new point cloud
    return T.stack((px, py, pz, energy), dim=-1)


def to_polar(p4: T.Tensor, return_mass: bool = False) -> T.Tensor:
    """
    Convert four-momenta from Cartesian to polar coordinates

    Args:
        p4: The four-momenta in Cartesian coordinates (px, py, pz, energy)

    Kwargs:
        return_mass: Whether to return the masses (otherwise energies)

    Returns:
        The four-momenta in polar coordinates (pT, eta, phi, mass|energy)
    """

    # Extract the px, py, pz and energy coordinates
    px = p4[..., 0]
    py = p4[..., 1]
    pz = p4[..., 2]
    energy = p4[..., 3]

    # Calculate the pT, eta and phi coordinates
    pt = T.sqrt(px**2 + py**2)
    eta = T.asinh(pz / (pt + 1e-15))  # = T.atanh(pz / (px**2 + py**2 + pz**2))
    phi = T.atan2(py, px)

    # Gather in a new point cloud
    if return_mass:
        mass = T.sqrt(energy**2 - (px**2 + py**2 + pz**2))
        return T.stack((pt, eta, phi, mass), dim=-1)
    else:
        return T.stack((pt, eta, phi, energy), dim=-1)
