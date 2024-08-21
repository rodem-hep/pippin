from typing import List, Tuple
import PIL.Image
from pathlib import Path
import numpy as np
import torch as T
from src.plotting.base import plot_color_pointcloud, plot_hist, plot_hist2d
from src.data.utils import presence_to_matchability
from src.physics import order_by_pt, to_cartesian
from src.utils import no_inf


MATCHABILITY_PATTERNS_INTERMEDIATE = {
    0b111111: r"$t\bar{t}$",  # Both Top quarks
    0b011011: r"$W^{\pm}$",   # Both W bosons
    0b100100: r"$b\bar{b}$",  # Both b-quarks
    0b111000: r"$t$",         # First Top quark
    0b011000: r"$W^+$",       # First W boson
    0b100000: r"$b$",         # First b-quark
    0b000111: r"$\bar{t}$",   # Second Top quark
    0b000011: r"$W^-$",       # Second W boson
    0b000100: r"$\bar{b}$",   # Second b-quark
}

MATCHABILITY_PATTERNS_FINAL = {
    0b100000: r"$b$",                   # b-quark from Top
    0b010000: r"$q/\ell$",              # First light-quark/lepton from Top
    0b001000: r"$\bar{q}/\bar{\nu}$",   # Second light-quark/neutrino from Top
    0b000100: r"$\bar{b}$",             # b-quark from anti-Top
    0b000010: r"$\bar{q}/\bar{\ell}$",  # First light-quark/lepton from anti-Top
    0b000001: r"$q/\nu$",               # Second light-quark/neutrino from anti-Top
}

DEFAULT_MATCHABILITY_PATTERNS = MATCHABILITY_PATTERNS_INTERMEDIATE


def plot_pointclouds_for_batch(
    pc_in: T.Tensor,
    mask_in: T.BoolTensor,
    pc_out: T.Tensor,
    mask_out: T.BoolTensor,
    pc_target: T.Tensor,
    mask_target: T.BoolTensor,
    path: Path | str = ".",
) -> List[Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]]:
    """
    Creates a series of scatter plots for a batch of data

    Args:
        pc_in: The input point clouds
        mask_in: The input masks
        pc_out: The output point clouds
        mask_out: The output masks
        pc_target: The target point clouds
        mask_target: The target masks

    Kwargs:
        path: The path to save the plots. Defaults to "."

    Returns:
        plot_list: A list of the created plots
    """

    labels = {
        "x_label": r"Jet $\eta$",
        "y_label": r"Jet $\phi$",
        "z_label": r"Jet $p_\mathrm{{T}}$ (normalised)",
    }

    # Zip together all the inputs so we can cycle neatly
    zipped = zip(pc_in, mask_in, pc_out, mask_out, pc_target, mask_target)

    # Cycle through the batch and create a scatter for each event
    plot_list = []
    for idx, (pi, mi, po, mo, pt, mt) in enumerate(zipped):
        (
            plot_color_pointcloud(
                pi,
                mi,
                save=True,
                path=path,
                filename=f"pointcloud_{idx}_input",
                **labels,
            ),
            plot_color_pointcloud(
                po,
                mo,
                save=True,
                path=path,
                filename=f"pointcloud_{idx}_output",
                **labels,
            ),
            plot_color_pointcloud(
                pt,
                mt,
                save=True,
                path=path,
                filename=f"pointcloud_{idx}_target",
                **labels,
            ),
        )

    return plot_list


def plot_marginals(
    pc_out: T.Tensor | List[T.Tensor],
    mult_out: Tuple[T.Tensor] | List[Tuple[T.Tensor]],
    mask_out: T.BoolTensor,
    mask_out_part: Tuple[T.BoolTensor] | List[Tuple[T.BoolTensor]],
    pc_target: T.Tensor | List[T.Tensor],
    mult_target: Tuple[T.Tensor] | List[Tuple[T.Tensor]],
    mask_target: T.BoolTensor,
    mask_target_part: Tuple[T.BoolTensor] | List[Tuple[T.BoolTensor]],
    channel: T.ByteTensor,
    pred_p: T.BoolTensor | List[T.BoolTensor] | None,
    true_p: T.BoolTensor | List[T.BoolTensor],
    match_out: T.ByteTensor | List[T.ByteTensor] | None = None,
    match_target: T.ByteTensor | List[T.ByteTensor] | None = None,
    pc_alt: T.Tensor | List[T.Tensor] | None = None,
    mult_alt: Tuple[T.Tensor] | List[Tuple[T.Tensor]] | None = None,
    mask_alt_part: Tuple[T.BoolTensor] | List[Tuple[T.BoolTensor]] | None = None,
    only_leading: bool = False,
    cartesian: bool = False,
    model: str = "pippin",
    model_alt: str | None = None,
    suffix: str = "",
    path: Path | str = ".",
) -> List[PIL.Image.Image]:
    """
    Creates a series of marginal plots for a batch of data

    Args:
        pc_out: The output point clouds.
        mult_out: The output multiplicities.
        mask_out: The output masks (not used).
        mask_out_part: The output masks for each particle.
        pc_target: The target point clouds.
        mult_target: The target multiplicities.
        mask_target: The target masks (not used).
        mask_target_part: The target masks for each particle.
        channel: The channel of the event.
        pred_p: The predicted presence of particles.
        true_p: The true presence of particles.

    Kwargs:
        match_out: The matchability of the output data. Defaults to None.
        match_target: The matchability of the target data. Defaults to None.
        pc_alt: The alternative output point clouds. Defaults to None.
        mult_alt: The alternative output multiplicities. Defaults to None.
        mask_alt_part: The alternative output masks for each particle. Defaults to None.
        only_leading: Whether to plot only the leading particle in pT. Defaults to False.
        cartesian: Whether to plot the point clouds in cartesian coordinates (as opposed to polar). Defaults to False.
        model: The model used to generate the output data. Defaults to "pippin".
        model_alt: The model used to generate the alternative output data. Defaults to None.
        suffix: The suffix to append to the filename. Defaults to "".
        path: The path to save the plots. Defaults to ".".

    Returns:
        marginals: A list of the created plots.
    """

    # Create lists of single tensors if single tensors are provided
    if not isinstance(pc_out, list):
        pc_out = [pc_out]
    if not isinstance(mult_out, list):
        mult_out = [mult_out]
    if not isinstance(mask_out_part, list):
        mask_out_part = [mask_out_part]
    if not isinstance(pc_target, list):
        pc_target = [pc_target]
    if not isinstance(mult_target, list):
        mult_target = [mult_target]
    if not isinstance(mask_target_part, list):
        mask_target_part = [mask_target_part]
    if pred_p is not None and not isinstance(pred_p, list):
        pred_p = [pred_p]
    if true_p is not None and not isinstance(true_p, list):
        true_p = [true_p]
    if match_out is not None and not isinstance(match_out, list):
        match_out = [match_out]
    if match_target is not None and not isinstance(match_target, list):
        match_target = [match_target]
    if pc_alt is not None and not isinstance(pc_alt, list):
        pc_alt = [pc_alt]
    if mult_alt is not None and not isinstance(mult_alt, list):
        mult_alt = [mult_alt]
    if mask_alt_part is not None and not isinstance(mask_alt_part, list):
        mask_alt_part = [mask_alt_part]

    # Convert the point clouds to cartesian coordinates if specified
    if cartesian:
        pc_out = [to_cartesian(pc) for pc in pc_out]
        pc_target = [to_cartesian(pc) for pc in pc_target]
        if pc_alt is not None:
            pc_alt = [to_cartesian(pc) for pc in pc_alt]

    # Define names, ranges, styles, etc. for the marginal plots
    names_dict = {
        0: "Leading Lepton" if only_leading else "Lepton",
        1: "MET",
        2: "Leading Jet" if only_leading else "Jet",
    }
    if cartesian:
        features_dict = {
            0: ["lep_px", "lep_py", "lep_pz", "lep_energy"],
            1: ["met_px", "met_py", "met_pz", "met_energy"],
            2: ["jet_px", "jet_py", "jet_pz", "jet_energy"],
        }
        x_ranges = {
            0: [(-600, 600), (-600, 600), (-2000, 2000), (10, 2000)],
            1: [(-600, 600), (-600, 600), (-10000, 10000), (0, 50000)],
            2: [(-1000, 1000), (-1000, 1000), (-3000, 3000), (20, 5000)],
        }
        labels = [r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]", r"$E$ [GeV]"]
        x_scales = ["linear", "linear", "linear", "log"]
        y_scales = ["log", "log", "log", "log"]
    else:
        features_dict = {
            0: ["lep_pt", "lep_eta", "lep_phi", "lep_energy"],
            1: ["met_pt", "met_eta", "met_phi", "met_energy"],
            2: ["jet_pt", "jet_eta", "jet_phi", "jet_energy"],
        }
        x_ranges = {
            0: [(10, 700), (-3, 3), (-np.pi, np.pi), (10, 2000)],
            1: [(0, 1000), (-8, 8), (-np.pi, np.pi), (0, 50000)],
            2: [(20, 2000), (-3, 3), (-np.pi, np.pi), (20, 5000)],
        }
        labels = [r"$p_\mathrm{{T}}$ [GeV]", r"$\eta$", r"$\phi$", r"$E$ [GeV]"]
        x_scales = ["log", "linear", "linear", "log"]
        y_scales = ["log", "log", "log", "log"]

    if only_leading:
        features_dict[0] = [f + "_lead" for f in features_dict[0]]
        features_dict[2] = [f + "_lead" for f in features_dict[2]]

    if model == "pippin":
        legends = ["MC", "PIPPIN"]
    elif model == "turbosim":
        legends = ["MC", "Turbo-Sim"]
    else:
        legends = ["MC", "Unknown"]

    if model_alt == "pippin":
        legends.append("PIPPIN")
    elif model_alt == "turbosim":
        legends.append("Turbo-Sim (new)")
        if model == "pippin": legends[1] += " (res)"
    elif model_alt is not None:
        legends.append("Unknown")

    if model_alt is not None:
        l_styles = ["-", "-", ":"]
        colors = ["grey", "C1", "C4"]
    else:
        l_styles = None
        colors = None

    # The list to return
    marginals = []

    # Create the marginals for pT, eta, phi and energy
    for i_part in [0, 1, 2]:

        name = names_dict[i_part]
        iterable = zip(features_dict[i_part], labels, x_scales, y_scales)
        for i_feat, (feat, label, x_scale, y_scale) in enumerate(iterable):

            if only_leading:
                data_list = [
                    [order_by_pt(p, m[i_part], is_cartesian=cartesian)[:, 0, i_feat] for p, m in zip(pc_target, mask_target_part)],
                    [order_by_pt(p, m[i_part], is_cartesian=cartesian)[:, 0, i_feat] for p, m in zip(pc_out, mask_out_part)],
                ]
                if model_alt is not None:
                    data_list.append(
                        [order_by_pt(p, m[i_part], is_cartesian=cartesian)[:, 0, i_feat] for p, m in zip(pc_alt, mask_alt_part)],
                    )
            else:
                data_list = [
                    [p[m[i_part]][:, i_feat] for p, m in zip(pc_target, mask_target_part)],  # The point clouds are flattened here
                    [p[m[i_part]][:, i_feat] for p, m in zip(pc_out, mask_out_part)],
                ]
                if model_alt is not None:
                    data_list.append(
                        [p[m[i_part]][:, i_feat] for p, m in zip(pc_alt, mask_alt_part)],
                    )

            if any([len(samples) == 0 for data in data_list for samples in data]):
                marginals.append(None)
            else:
                marginals.append(
                    plot_hist(
                        data_list,
                        x_range=x_ranges[i_part][i_feat],
                        legends=legends,
                        l_styles=l_styles,
                        colors=colors,
                        x_label=f"{name} {label}",
                        x_scale=x_scale,
                        y_scale=y_scale,
                        save=True,
                        path=path,
                        filename=f"marginal_{feat}" + suffix,
                    ),
                )

    # Create the marginals for the multiplicity
    features = ["lep_n", "met_n", "jet_n"]
    names = [r"Lepton $N$", r"MET $N$", r"Jet $N$"]

    for  i, feature in enumerate(features):

        data_list = [
            [m[i].flatten() for m in mult_target],
            [m[i].flatten() for m in mult_out],
        ]

        marginals += [
            plot_hist(
                data_list,
                legends=legends,
                x_label=names[i],
                x_scale="linear",
                y_scale="log",
                integers=True,
                save=True,
                path=path,
                filename=f"marginal_{feature}" + suffix,
            )
        ]

    # Create the marginals for the matchability
    if true_p is not None:
        true_match = [presence_to_matchability(p) for p in true_p]
    if pred_p is not None:
        pred_match = [presence_to_matchability(p) for p in pred_p]
    elif true_p is not None:
        pred_match = [T.full_like(m, 0) for m in true_match]

    for stage in ["inter", "final"]:
        if stage == "inter":
            patterns_dict = MATCHABILITY_PATTERNS_INTERMEDIATE
        elif stage == "final":
            patterns_dict = MATCHABILITY_PATTERNS_FINAL

        patterns = list(patterns_dict.keys())
        xticks = list(patterns_dict.values())

        marginals += [
            plot_matchablity(
                true_match,
                pred_match,
                match_out,
                match_target,
                patterns=patterns,
                xticks=xticks,
                model=model,
                suffix=f"_{stage}" + suffix,
                path=path,
            ),
        ] if true_p is not None else [None]

    return marginals


def plot_marginals_2D(
    pc_out: T.Tensor,
    mult_out: T.Tensor,
    mask_out: T.BoolTensor,
    mask_out_part: T.BoolTensor,
    pc_target: T.Tensor,
    mult_target: T.Tensor,
    mask_target: T.BoolTensor,
    mask_target_part: T.BoolTensor,
    channel: T.ByteTensor,
    pred_p: T.BoolTensor,
    true_p: T.BoolTensor,
    model: str = "pippin",
    suffix: str = "",
    path: Path | str = ".",
) -> List[PIL.Image.Image]:
    """
    Creates a series of correlation plots for a batch of data.

    Args:
        pc_out: The output point clouds.
        mult_out: The output multiplicities.
        mask_out: The output masks.
        mask_out_part: The output masks for each particle.
        pc_target: The target point clouds.
        mult_target: The target multiplicities.
        mask_target: The target masks.
        mask_target_part: The target masks for each particle.
        channel: The channel of the event.
        pred_p: The predicted presence of particles.
        true_p: The true presence of particles.

    Kwargs:
        model: The model used to generate the output data. Defaults to "pippin".
        suffix: The suffix to append to the filename. Defaults to "".
        path: The path to save the plots. Defaults to ".".

    Returns:
        marginals_2D: A list of the created plots.
    """

    # Names, labels, scales, etc. for the 2D marginal plots
    features_dict = {
        0: ["lep_pt", "lep_eta", "lep_phi", "lep_energy"],
        1: ["met_pt", "met_eta", "met_phi", "met_energy"],
        2: ["jet_pt", "jet_eta", "jet_phi", "jet_energy"],
    }
    names_dict = {
        0: "Leading Lepton",
        1: "MET",
        2: "Leading Jet",
    }
    labels = [r"$p_\mathrm{{T}}$ [GeV]", r"$\eta$", r"$\phi$", r"$E$ [GeV]"]
    xy_scales = ["log", "linear", "linear", "log"]
    c_scales = ["log", "log", "log", "log"]
    formats = ["sci", None, None, "sci"]

    if model == "pippin":
        y_label = "PIPPIN"
    elif model == "turbosim":
        y_label = "Turbo-Sim"
    else:
        y_label = "Unknown"

    marginals_2D = []

    # Create the 2D marginals for pT, eta, phi and energy
    for i_part in features_dict.keys():

        name = names_dict[i_part]
        for i_feat in range(len(features_dict[i_part])):
            feat = features_dict[i_part][i_feat]
            label = labels[i_feat]
            xy_scale = xy_scales[i_feat]
            c_scale = c_scales[i_feat]
            format = formats[i_feat]

            data_list = [
                order_by_pt(pc_target, mask_target_part[i_part])[:, 0, i_feat],
                order_by_pt(pc_out, mask_out_part[i_part])[:, 0, i_feat],
            ]

            # Remove if both points are zero (likely no real particle)
            mask = T.stack([data != 0 for data in data_list], dim=1).all(dim=1)
            data_list = [data[mask] for data in data_list]

            if any([len(data) == 0 for data in data_list]):
                marginals_2D.append(None)
            else:
                marginals_2D.append(
                    plot_hist2d(
                        data_list,
                        title=f"{name} {label}",
                        x_label="MC",
                        y_label=y_label,
                        xy_format=format,
                        xy_scale=xy_scale,
                        c_scale=c_scale,
                        save=True,
                        path=path,
                        filename=f"marginal2D_{feat}" + suffix,
                    ),
                )

    # Create the 2D marginals for the multiplicity
    features = ["lep_n", "met_n", "jet_n"]
    names = [r"Lepton $N$", r"MET $N$", r"Jet $N$"]
    marginals_2D += [
        plot_hist2d(
            [n_target.flatten(), n_out.flatten()],
            title=names[i],
            x_label="MC",
            y_label=y_label,
            xy_scale="linear",
            c_scale="log",
            integers=True,
            save=True,
            path=path,
            filename=f"marginal2D_{feature}" + suffix,
        )
        for i, (n_target, n_out, feature) in enumerate(
            zip(mult_target, mult_out, features)
        )
    ]

    # Create the marginals for the matchability
    if true_p is not None:
        true_match = presence_to_matchability(true_p)
    if pred_p is not None:
        pred_match = presence_to_matchability(pred_p)
    elif true_p is not None:
        pred_match = T.full_like(true_match, 0)
    marginals_2D += [
        plot_matchablity(
            true_match,
            pred_match,
            patterns=True,
            xticks=True,
            is_2D=True,
            model=model,
            suffix=suffix,
            path=path,
        ),
    ] if true_p is not None else [None]

    return marginals_2D


def plot_masses(
    masses_in: T.Tensor | List[T.Tensor] | None,
    masses_out: T.Tensor | List[T.Tensor],
    masses_target: T.Tensor | List[T.Tensor],
    masses_alt: T.Tensor | List[T.Tensor] | None = None,
    percent_out: T.Tensor | None = None,
    percent_target: T.Tensor | None = None,
    percent_alt: T.Tensor | None = None,
    model: str = "pippin",
    model_alt: str | None = None,
    suffix: str = "",
    path: Path | str = ".",
) -> List[PIL.Image.Image]:
    """
    Creates a series of mass plots for a batch of data.

    Args:
        masses_in: The input masses.
        masses_out: The output masses.
        masses_target: The target masses.

    Kwargs:
        masses_alt: The alternative output masses. Defaults to None.
        percent_out: The percentage of considered output masses. Defaults to None.
        percent_target: The percentage of considered target masses. Defaults to None.
        percent_alt: The percentage of considered alternative output masses. Defaults to None.
        model: The model used to generate the output data. Defaults to "pippin".
        model_alt: The model used to generate the alternative output data. Defaults to None.
        suffix: The suffix to append to the filename. Defaults to "".
        path: The path to save the plots. Defaults to ".".

    Returns:
        images: A list of the created plots.
    """

    # Create lists of single tensors if single tensors are provided
    if masses_in is not None and not isinstance(masses_in, list):
        masses_in = [masses_in]
    if not isinstance(masses_out, list):
        masses_out = [masses_out]
    if not isinstance(masses_target, list):
        masses_target = [masses_target]

    # Define several parameters for the histograms
    names = [r"First $W$", r"Second $W$", r"First Top", r"Second Top", r"$t\bar{t}$ pair"]
    names_simple = ["w1", "w2", "top1", "top2", "ttbar"]

    if model == "pippin":
        legends = ["MC", "PIPPIN"]
    elif model == "turbosim":
        legends = ["MC", "Turbo-Sim"]
    else:
        legends = ["MC", "Unknown"]

    if model_alt == "pippin":
        legends.append("PIPPIN")
    elif model_alt == "turbosim":
        legends.append("Turbo-Sim (new)")
        if model == "pippin": legends[1] += " (res)"
    elif model_alt is not None:
        legends.append("Unknown")

    if percent_target is not None:
        legends[0] += f" ({percent_target.mean().item():.1%})"
    if percent_out is not None:
        legends[1] += f" ({percent_out.mean().item():.1%})"
    if percent_alt is not None:
        legends[2] += f" ({percent_alt.mean().item():.1%})"

    if masses_in is not None:
        legends.append("MC (partons)")

    l_styles = ["-", "-"]
    colors = ["grey", "C1"]
    if model_alt is not None:
        l_styles.append(":")
        colors.append("C4")
    if masses_in is not None:
        l_styles.append("--")
        colors.append("C2")

    # Specific parameters for the masses
    x_max = [400, 400, 600, 600, 2000]

    # Plot the masses for each underlying system
    images = []
    for i in range(masses_out[0].shape[-1]):
        data_list = [
            [m[:, i] for m in masses_target],
            [m[:, i] for m in masses_out],
        ]
        if masses_alt is not None:
            data_list.append([m[:, i] for m in masses_alt])
        if masses_in is not None:
            data_list.append([m[:, i] for m in masses_in])

        images.append(
            plot_hist(
                data_list,
                x_range=(0, x_max[i]),
                legends=legends,
                l_styles=l_styles,
                colors=colors,
                x_label=f"{names[i]} mass [GeV]",
                x_scale="linear",
                y_label="Normalised counts",
                y_scale="log",
                # show_error=[True, True, False],
                normalise=True,
                save=True,
                path=path,
                filename=f"mass_{names_simple[i]}" + suffix,
            )
        )

    return images


def plot_momenta(
    momenta_in: T.Tensor | List[T.Tensor],
    momenta_out: T.Tensor | List[T.Tensor],
    momenta_target: T.Tensor | List[T.Tensor],
    percent_out: T.Tensor,
    percent_target: T.Tensor,
    model: str = "pippin",
    suffix: str = "",
    path: Path | str = ".",
) -> List[PIL.Image.Image]:
    """
    Creates a series of momentum plots for a batch of data.

    Args:
        momenta_in: The input momenta.
        momenta_out: The output momenta.
        momenta_target: The target momenta.
        percent_out: The percentage of considered output momenta.
        percent_target: The percentage of considered target momenta.

    Kwargs:
        model: The model used to generate the output data. Defaults to "pippin".
        suffix: The suffix to append to the filename. Defaults to "".
        path: The path to save the plots. Defaults to ".".

    Returns:
        images: A list of the created plots.
    """

    # Create lists of single tensors if single tensors are provided
    if not isinstance(momenta_in, list):
        momenta_in = [momenta_in]
    if not isinstance(momenta_out, list):
        momenta_out = [momenta_out]
    if not isinstance(momenta_target, list):
        momenta_target = [momenta_target]

    # Define several parameters for the histograms
    names = [r"First $W$", r"Second $W$", r"First Top", r"Second Top", r"$t\bar{t}$ pair"]
    names_simple = ["w1", "w2", "top1", "top2", "ttbar"]
    if model == "pippin":
        legends = ["MC", "PIPPIN", "MC (partons)"]
    elif model == "turbosim":
        legends = ["MC", "Turbo-Sim", "MC (partons)"]
    else:
        legends = ["MC", "Unknown", "MC (partons)"]
    legends[0] += f" ({percent_target.mean().item():.1%})"
    legends[1] += f" ({percent_out.mean().item():.1%})"

    # Specific parameters for the momenta
    x_max = [600, 600, 700, 700, 900]

    # Plot the momenta for each underlying system
    images = []
    for i in range(momenta_out[0].shape[-1]):
        data_list = [
            [p[:, i] for p in momenta_target],
            [p[:, i] for p in momenta_out],
            [p[:, i] for p in momenta_in],
        ]
        images.append(
            plot_hist(
                data_list,
                x_range=(0, x_max[i]),
                legends=legends,
                l_styles=["-", "-", "--"],
                x_label=f"{names[i]} $p_\mathrm{{T}}$ [GeV]",
                x_scale="linear",
                y_label="Normalised counts",
                y_scale="log",
                show_error=[True, True, False],
                normalise=True,
                ratio_idx=[0, 1, 2],
                save=True,
                path=path,
                filename=f"pt_{names_simple[i]}" + suffix,
            )
        )

    return images


def plot_rapidities(
    rapidities_in: T.Tensor | List[T.Tensor],
    rapidities_out: T.Tensor | List[T.Tensor],
    rapidities_target: T.Tensor | List[T.Tensor],
    percent_out: T.Tensor,
    percent_target: T.Tensor,
    model: str = "pippin",
    suffix: str = "",
    path: Path | str = ".",
) -> List[PIL.Image.Image]:
    """
    Creates a series of pseudo rapidity plots for a batch of data.

    Args:
        rapidities_in: The input pseudo rapidities.
        rapidities_out: The output pseudo rapidities.
        rapidities_target: The target pseudo rapidities.
        percent_out: The percentage of considered output rapidities.
        percent_target: The percentage of considered target rapidities.

    Kwargs:
        model: The model used to generate the output data. Defaults to "pippin".
        suffix: The suffix to append to the filename. Defaults to "".
        path: The path to save the plots. Defaults to ".".

    Returns:
        images: A list of the created plots.
    """

    # Create lists of single tensors if single tensors are provided
    if not isinstance(rapidities_in, list):
        rapidities_in = [rapidities_in]
    if not isinstance(rapidities_out, list):
        rapidities_out = [rapidities_out]
    if not isinstance(rapidities_target, list):
        rapidities_target = [rapidities_target]

    # Define several parameters for the histograms
    names = [r"First $W$", r"Second $W$", r"First Top", r"Second Top", r"$t\bar{t}$ pair"]
    names_simple = ["w1", "w2", "top1", "top2", "ttbar"]
    if model == "pippin":
        legends = ["MC", "PIPPIN", "MC (partons)"]
    elif model == "turbosim":
        legends = ["MC", "Turbo-Sim", "MC (partons)"]
    else:
        legends = ["MC", "Unknown", "MC (partons)"]
    legends[0] += f" ({percent_target.mean().item():.1%})"
    legends[1] += f" ({percent_out.mean().item():.1%})"

    # Specific parameters for the pseudo-rapidities
    x_lim = [5, 5, 5, 5, 7]

    # Plot the rapidities for each underlying system
    images = []
    for i in range(rapidities_out[0].shape[-1]):
        data_list = [
            [no_inf(eta[:, i]) for eta in rapidities_target],
            [no_inf(eta[:, i]) for eta in rapidities_out],
            [no_inf(eta[:, i]) for eta in rapidities_in],
        ]
        images.append(
            plot_hist(
                data_list,
                x_range=(-x_lim[i], x_lim[i]),
                legends=legends,
                l_styles=["-", "-", "--"],
                x_label=fr"{names[i]} $\eta$",
                y_label="Normalised counts",
                show_error=[True, True, False],
                normalise=True,
                ratio_idx=[0, 1, 2],
                save=True,
                path=path,
                filename=f"rapidity_{names_simple[i]}" + suffix,
            )
        )

    return images


# TODO: 2D matchability is still WIP
def plot_matchablity(
    true_match: T.ByteTensor | List[T.ByteTensor],
    pred_match: T.ByteTensor | List[T.ByteTensor],
    match_out: T.ByteTensor | List[T.ByteTensor] | None = None,
    match_target: T.ByteTensor | List[T.ByteTensor] | None = None,
    patterns: List[int] | bool | None = None,
    xticks: List[str] | bool | None = None,
    is_2D: bool = False,
    model: str = "pippin",
    suffix: str = "",
    path: Path | str = ".",
) -> PIL.Image.Image:
    """
    Creates multiple plots for the matchability of the events

    Args:
        true_match: The true matchability of the events
        pred_match: The predicted matchability of the events

    Kwargs:
        match_out: The computed matchability of the output data. Defaults to None.
        match_target: The computed matchability of the target data. Defaults to None.
        patterns: The matchability patterns to plot.
            If True, uses the default patterns. If None, plots the raw matchability. Defaults to None.
        xticks: The xticks for the patterns.
            If True, uses the default xticks. If None, uses the raw matchability. Defaults to None.
        is_2D: Whether to plot the matchability in 2D (WIP). Defaults to False.
        model: The model used to generate the output data. Defaults to "pippin".
        suffix: The suffix to append to the filename. Defaults to "".
        path: The path to save the plots. Defaults to ".".

    Returns:
        The matchability plot.
    """

    if model == "pippin":
        legends = ["MC", "PIPPIN"]
        y_label = "PIPPIN"
    elif model == "turbosim":
        legends = ["MC", "Turbo-Sim"]
        y_label = "Turbo-Sim"
    else:
        legends = ["MC", "Unknown"]
        y_label = "Unknown"

    if match_out is not None:
        legends.append(legends[1] + " (out)")
        legends[1] += " (pred)"
    if match_target is not None:
        legends.append(legends[0] + " (target)")
        legends[0] += " (true)"

    l_styles = ["-", "-"]
    colors = ["grey", "C1"]
    if match_out is not None:
        l_styles.append("-.")
        colors.append("C3")
    if match_target is not None:
        l_styles.append(":")
        colors.append("C5")

    # If no patterns are provided, plot the plain matchability of the events
    if not patterns:
        if is_2D:
            return plot_hist2d(
                [true_match, pred_match],
                title="Event matchability",
                x_label="MC",
                y_label=y_label,
                c_scale="log",
                integers=True,
                save=True,
                path=path,
                filename=f"marginal2D_match" + suffix,
            )
        else:
            return plot_hist(
                [true_match, pred_match],
                legends=legends,
                x_label="Event matchability",
                y_scale="log",
                integers=True,
                save=True,
                path=path,
                filename=f"marginal_match" + suffix,
            )

    # If patterns are provided, decompose the plot into the different patterns
    else:

        # Get the default patterns and xticks if not provided
        if not isinstance(patterns, list):
            patterns = list(DEFAULT_MATCHABILITY_PATTERNS.keys())
        if not isinstance(xticks, list) and xticks:
            xticks = list(DEFAULT_MATCHABILITY_PATTERNS.values())
        elif xticks is None:
            xticks = [f"{p:#08b}" for p in patterns]

        # Plot the matchability patterns in 1D or 2D
        if is_2D:
            # Compare the matchabilities to the patterns
            patterns = T.tensor(patterns, dtype=T.uint8, device=true_match.device)
            true_match = true_match.unsqueeze(1) & patterns == patterns
            pred_match = pred_match.unsqueeze(1) & patterns == patterns

            # Compute the counts matrix for each pairs of patterns
            counts = true_match.unsqueeze(-1) & pred_match.unsqueeze(-2)
            counts = counts.float().sum(dim=0)

            return plot_hist2d(
                counts,
                title="Event matchability",
                x_label="MC",
                y_label=y_label,
                c_scale="log",
                # integers=True,
                is_categorical=True,
                categories=xticks,
                save=True,
                path=path,
                filename=f"marginal2D_match" + suffix,
            )

        else:
            # Store the average total number of events
            n_events = np.mean([len(m) for m in true_match])

            # Compare the matchabilities to the patterns
            patterns = T.tensor(patterns, dtype=T.uint8, device=true_match[0].device)
            true_match = [m.unsqueeze(1) & patterns == patterns for m in true_match]
            pred_match = [m.unsqueeze(1) & patterns == patterns for m in pred_match]
            if match_out is not None:
                match_out = [m.unsqueeze(1) & patterns == patterns for m in match_out]
            if match_target is not None:
                match_target = [m.unsqueeze(1) & patterns == patterns for m in match_target]

            # Compute the counts for each pattern
            true_match = [m.sum(dim=0) for m in true_match]
            pred_match = [m.sum(dim=0) for m in pred_match]
            if match_out is not None:
                match_out = [m.sum(dim=0) for m in match_out]
            if match_target is not None:
                match_target = [m.sum(dim=0) for m in match_target]

            # Gather the counts in a list
            data_list = [true_match, pred_match]
            if match_out is not None:
                data_list.append(match_out)
            if match_target is not None:
                data_list.append(match_target)

            return plot_hist(
                data_list,
                legends=legends,
                l_styles=l_styles,
                colors=colors,
                x_label="Event matchability",
                y_label="Proportion of events",
                proportion=True,
                proportion_total=n_events,
                is_categorical=True,
                categories=xticks,
                save=True,
                path=path,
                filename=f"marginal_match" + suffix,
            )
