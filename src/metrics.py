from typing import Tuple
import os
from pathlib import Path
import yaml

import numpy as np
import torch as T

from scipy.stats import ks_2samp

from src.utils import no_inf
from src.physics import order_by_pt, to_cartesian

from mattstools.mattstools.torch_utils import to_np


def compute_ks(
    pc_x: T.Tensor,
    mask_x_part: Tuple[T.BoolTensor],
    masses_x: T.Tensor,
    momenta_x: T.Tensor,
    rapidities_x: T.Tensor,
    pc_y: T.Tensor,
    mask_y_part: Tuple[T.BoolTensor],
    masses_y: T.Tensor,
    momenta_y: T.Tensor,
    rapidities_y: T.Tensor,
    only_leading: bool = False,
    cartesian: bool = False,
) -> T.Tensor:
    r"""
    Compute the Kolmogorov-Smirnov distance for several features of two
    point clouds as well as invariant masses, transverse momenta and
    pseudo-rapidities of several underlying systems.
    Point clouds are flattened before computing the distance, so every jets
    are treated equally.

    Features of the point clouds:
        - Lepton/MET/Jet `p_T`
        - Lepton/MET/Jet `\eta`
        - Lepton/MET/Jet `\phi`
        - Lepton/MET/Jet `E`

    If 'cartesian' is 'True', they will be converted to:
        - Lepton/MET/Jet `p_x`
        - Lepton/MET/Jet `p_y`
        - Lepton/MET/Jet `p_z`
        - Lepton/MET/Jet `E`

    Masses of the underlying systems:
        - First `W` mass
        - Second `W` mass
        - First Top mass
        - Second Top mass
        - `t\bar{t}` mass

    Transverse momenta of the underlying systems:
        - First `W` `p_T`
        - Second `W` `p_T`
        - First Top `p_T`
        - Second Top `p_T`
        - `t\bar{t}` `p_T`

    Pseudo-rapidities of the underlying systems:
        - First `W` `\eta`
        - Second `W` `\eta`
        - First Top `\eta`
        - Second Top `\eta`
        - `t\bar{t}` `\eta`

    Args:
        pc_x: Point cloud tensor of the first set of events.
        mask_x_part: Tuple of boolean masks for the lepton, MET and jet particles.
        masses_x: Tensor of the masses of the underlying systems of the first set.
        momenta_x: Tensor of the transverse momenta of the underlying systems of the first set.
        rapidities_x: Tensor of the pseudo-rapidities of the underlying systems of the first set.
        pc_y: Point cloud tensor of the second set of events.
        mask_y_part: Tuple of boolean masks for the lepton, MET and jet particles.
        masses_y: Tensor of the masses of the underlying systems of the second set.
        momenta_y: Tensor of the transverse momenta of the underlying systems of the second set.
        rapidities_y: Tensor of the pseudo-rapidities of the underlying systems of the second set.

    Kwargs:
        only_leading: Whether to consider only the leading particles.
        cartesian: Whether to convert the features to Cartesian coordinates.

    Returns:
        Tensor of the Kolmogorov-Smirnov distances for each feature.
    """
    # Convert to Cartesian coordinates if specified
    if cartesian:
        pc_x = to_cartesian(pc_x)
        pc_y = to_cartesian(pc_y)

    # Split features per particle types and convert to Numpy arrays
    if only_leading:
        lep_x = to_np(order_by_pt(pc_x, mask_x_part[0], is_cartesian=cartesian)[:, 0])
        met_x = to_np(order_by_pt(pc_x, mask_x_part[1], is_cartesian=cartesian)[:, 0])
        jet_x = to_np(order_by_pt(pc_x, mask_x_part[2], is_cartesian=cartesian)[:, 0])

        lep_y = to_np(order_by_pt(pc_y, mask_y_part[0], is_cartesian=cartesian)[:, 0])
        met_y = to_np(order_by_pt(pc_y, mask_y_part[1], is_cartesian=cartesian)[:, 0])
        jet_y = to_np(order_by_pt(pc_y, mask_y_part[2], is_cartesian=cartesian)[:, 0])
    else:
        lep_x = to_np(pc_x[mask_x_part[0]])  # Point clouds are flattened here
        met_x = to_np(pc_x[mask_x_part[1]])
        jet_x = to_np(pc_x[mask_x_part[2]])

        lep_y = to_np(pc_y[mask_y_part[0]])
        met_y = to_np(pc_y[mask_y_part[1]])
        jet_y = to_np(pc_y[mask_y_part[2]])

    masses_x = to_np(masses_x)
    masses_y = to_np(masses_y)
    momenta_x = to_np(momenta_x)
    momenta_y = to_np(momenta_y)
    rapidities_x = to_np(rapidities_x)
    rapidities_y = to_np(rapidities_y)

    # Extract the masses of the underlying systems and remove the infinities
    m_w1_x = no_inf(masses_x[:, 0])
    m_w2_x = no_inf(masses_x[:, 1])
    m_t1_x = no_inf(masses_x[:, 2])
    m_t2_x = no_inf(masses_x[:, 3])
    m_tt_x = no_inf(masses_x[:, 4])

    m_w1_y = no_inf(masses_y[:, 0])
    m_w2_y = no_inf(masses_y[:, 1])
    m_t1_y = no_inf(masses_y[:, 2])
    m_t2_y = no_inf(masses_y[:, 3])
    m_tt_y = no_inf(masses_y[:, 4])

    # Compute the Kolmogorov-Smirnov distance for each feature of the leptons
    ks_lep = T.stack(
        [
            T.tensor(try_ks(ks_2samp, lep_x[:, 0], lep_y[:, 0])),  # pT or px
            T.tensor(try_ks(ks_2samp, lep_x[:, 1], lep_y[:, 1])),  # Eta or py
            T.tensor(try_ks(ks_2samp, lep_x[:, 2], lep_y[:, 2])),  # Phi or pz
            T.tensor(try_ks(ks_2samp, lep_x[:, 3], lep_y[:, 3])),  # Energy
            T.tensor(T.inf),  # No KS for the multiplicity (discrete data)
        ],
        dim=-1,
    )

    # Compute the Kolmogorov-Smirnov distance for each feature of the MET
    ks_met = T.stack(
        [
            T.tensor(try_ks(ks_2samp, met_x[:, 0], met_y[:, 0])),  # pT or px
            T.tensor(try_ks(ks_2samp, met_x[:, 1], met_y[:, 1])),  # Eta or py
            T.tensor(try_ks(ks_2samp, met_x[:, 2], met_y[:, 2])),  # Phi or pz
            T.tensor(try_ks(ks_2samp, met_x[:, 3], met_y[:, 3])),  # Energy
            T.tensor(T.inf),  # No KS for the multiplicity (discrete data)
        ],
        dim=-1,
    )

    # Compute the Kolmogorov-Smirnov distance for each feature of the jets
    ks_jet = T.stack(
        [
            T.tensor(try_ks(ks_2samp, jet_x[:, 0], jet_y[:, 0])),  # pT or px
            T.tensor(try_ks(ks_2samp, jet_x[:, 1], jet_y[:, 1])),  # Eta or py
            T.tensor(try_ks(ks_2samp, jet_x[:, 2], jet_y[:, 2])),  # Phi or pz
            T.tensor(try_ks(ks_2samp, jet_x[:, 3], jet_y[:, 3])),  # Energy
            T.tensor(T.inf),  # No KS for the multiplicity (discrete data)
        ],
        dim=-1,
    )

    # Compute the Kolmogorov-Smirnov distance for each mass
    ks_masses = T.stack(
        [
            T.tensor(try_ks(ks_2samp, m_w1_x, m_w1_y)),
            T.tensor(try_ks(ks_2samp, m_w2_x, m_w2_y)),
            T.tensor(try_ks(ks_2samp, m_t1_x, m_t1_y)),
            T.tensor(try_ks(ks_2samp, m_t2_x, m_t2_y)),
            T.tensor(try_ks(ks_2samp, m_tt_x, m_tt_y)),
        ],
        dim=-1,
    )

    # Compute the Kolmogorov-Smirnov distance for each transverse momentum
    ks_momenta = T.stack(
        [
            T.tensor(try_ks(ks_2samp, momenta_x[:, 0], momenta_y[:, 0])),  # W1
            T.tensor(try_ks(ks_2samp, momenta_x[:, 1], momenta_y[:, 1])),  # W2
            T.tensor(try_ks(ks_2samp, momenta_x[:, 2], momenta_y[:, 2])),  # t1
            T.tensor(try_ks(ks_2samp, momenta_x[:, 3], momenta_y[:, 3])),  # t2
            T.tensor(try_ks(ks_2samp, momenta_x[:, 4], momenta_y[:, 4])),  # tt
        ],
        dim=-1,
    )

    # Compute the Kolmogorov-Smirnov distance for each pseudo-rapidity
    ks_rapidities = T.stack(
        [
            T.tensor(try_ks(ks_2samp, rapidities_x[:, 0], rapidities_y[:, 0])),  # W1
            T.tensor(try_ks(ks_2samp, rapidities_x[:, 1], rapidities_y[:, 1])),  # W2
            T.tensor(try_ks(ks_2samp, rapidities_x[:, 2], rapidities_y[:, 2])),  # t1
            T.tensor(try_ks(ks_2samp, rapidities_x[:, 3], rapidities_y[:, 3])),  # t2
            T.tensor(try_ks(ks_2samp, rapidities_x[:, 4], rapidities_y[:, 4])),  # tt
        ],
        dim=-1,
    )

    return T.stack(
        [ks_lep, ks_met, ks_jet, ks_masses, ks_momenta, ks_rapidities],
        dim=-1,
    )


def save_ks(
    ks,
    only_leading=False,
    cartesian=False,
    path=".",
    filename="ks.yaml",
):
    """
    Helper function to save the KS distances to a YAML file.

    Args:
        ks: Tensor of the KS distances to save.

    Kwargs:
        only_leading: Whether only the leading particles were considered.
        cartesian: Whether the features were converted to Cartesian coordinates.
        path: Path to save the YAML file.
        filename: Name of the YAML file.
    """
    # Suffix for only_leading metrics
    s = "_lead" if only_leading else ""

    # Set the momentum keys
    if cartesian:
        p = ["px", "py", "pz", "energy"]
    else:
        p = ["pt", "eta", "phi", "energy"]

    # Extract the metrics
    ks_lep = ks[..., 0]
    ks_met = ks[..., 1]
    ks_jet = ks[..., 2]
    ks_masses = ks[..., 3]
    ks_momenta = ks[..., 4]
    ks_rapidities = ks[..., 5]

    # Create a dictionary with the KS distances for each feature
    ks_dict = {
        "lep_" + p[0] + s: ks_lep[0].item(),
        "lep_" + p[1] + s: ks_lep[1].item(),
        "lep_" + p[2] + s: ks_lep[2].item(),
        "lep_" + p[3] + s: ks_lep[3].item(),
        "lep_n": ks_lep[4].item(),
        "met_" + p[0] + s: ks_met[0].item(),
        "met_" + p[1] + s: ks_met[1].item(),
        "met_" + p[2] + s: ks_met[2].item(),
        "met_" + p[3] + s: ks_met[3].item(),
        "met_n": ks_met[4].item(),
        "jet_" + p[0] + s: ks_jet[0].item(),
        "jet_" + p[1] + s: ks_jet[1].item(),
        "jet_" + p[2] + s: ks_jet[2].item(),
        "jet_" + p[3] + s: ks_jet[3].item(),
        "jet_n": ks_jet[4].item(),
        "m_w1": ks_masses[0].item(),
        "m_w2": ks_masses[1].item(),
        "m_t1": ks_masses[2].item(),
        "m_t2": ks_masses[3].item(),
        "m_tt": ks_masses[4].item(),
        "pt_w1": ks_momenta[0].item(),
        "pt_w2": ks_momenta[1].item(),
        "pt_t1": ks_momenta[2].item(),
        "pt_t2": ks_momenta[3].item(),
        "pt_tt": ks_momenta[4].item(),
        "eta_w1": ks_rapidities[0].item(),
        "eta_w2": ks_rapidities[1].item(),
        "eta_t1": ks_rapidities[2].item(),
        "eta_t2": ks_rapidities[3].item(),
        "eta_tt": ks_rapidities[4].item(),
    }

    # Save the dictionary to a YAML file
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    with open(path/filename, "w") as f:
        yaml.dump(ks_dict, f)


# Error handling (for Kolmogorov-Smirnov tests only)
def try_ks(f, *args, **kwargs):
    """"
    Try to compute a function and extract f().statistic.
    Suitable for Kolmogorov-Smirnov distance.
    Return infinity if it fails.
    """
    try:
        return f(*args, **kwargs).statistic
    except:
        return np.inf
