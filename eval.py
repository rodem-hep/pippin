import logging

import os
from pathlib import Path
import argparse

import h5py
import yaml

import numpy as np
import torch as T

from src.metrics import compute_ks, save_ks
from src.data.utils import (
    get_particle_masks,
    get_particle_mult,
    get_mask_turbo
)
from src.plotting.physics import (
    plot_marginals,
    plot_marginals_2D,
    plot_masses,
    plot_momenta,
    plot_rapidities
)
from src.physics import to_polar, compute_observables

log = logging.getLogger(__name__)


# Define the main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate the PIPPIN model")
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="network",
        help="Network name to evaluate",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        action="store_true",
        help="Compute the metrics",
    )
    parser.add_argument(
        "--plots",
        "-p",
        action="store_true",
        help="Make the plots",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Make the comparison plots",
    )
    parser.add_argument(
        "--inclusive",
        "-i",
        action="store_true",
        help="Inclusive dataset",
    )
    parser.add_argument(
        "--leading",
        "-l",
        action="store_true",
        help="Leading particles only",
    )
    parser.add_argument(
        "--turbolike",
        "-t",
        action="store_true",
        help="Turbo-like outputs",
    )
    parser.add_argument(
        "--turbosim",
        "-T",
        action="store_true",
        help="Turbo-Sim outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Hardcoded variables
    path_plots = Path(f"./outputs/PIPPIN/ttbar/{args.name}/plots/test")
    path_metrics = Path(f"./outputs/PIPPIN/ttbar/{args.name}/metrics/test")
    path_hdf5 = Path(f"./outputs/PIPPIN/ttbar/{args.name}/hdf5/test")
    path_turbo = Path(f"./data/turbosim_outputs.h5")
    filename = Path("outputs.h5")
    max_n = [2, 1, 16]
    dataset_name = "inclusive"
    runs = [0]
    n_test_turbosim = 160_000

    if args.inclusive or args.leading or args.turbolike:
        # Initialize the lists
        pc_in = []
        pc_out = []
        pc_target = []

        pred_p = []
        true_p = []

        check_out = []
        check_target = []

        percent_out = []
        percent_target = []

        match_out = []
        match_target = []

        masses_in = []
        masses_out = []
        masses_target = []

        momenta_in = []
        momenta_out = []
        momenta_target = []

        rapidities_in = []
        rapidities_out = []
        rapidities_target = []

        mult_out = []
        mult_target = []
        mask_out_part = []
        mask_target_part = []

        mask_turbo = []

        # Loop over the runs to fill the lists
        log.info("Start loading the data")
        for i, run in enumerate(runs):

            # Load the outputs from the HDF5 file
            path = path_hdf5/f"{run}"/filename
            with h5py.File(path, "r") as f:
                pippin = f["pippin"]
                pointclouds = T.tensor(pippin["pointclouds"][...])
                masks = T.tensor(pippin["masks"][...])
                channel = T.tensor(pippin["channel"][...])
                presences = T.tensor(pippin["presences"][...])
                # checks = T.tensor(pippin["checks"][...])
                # percents = T.tensor(pippin["percents"][...])
                # matches = T.tensor(pippin["matchabilities"][...])
                # masses = T.tensor(pippin["masses"][...])
                # momenta = T.tensor(pippin["momenta"][...])
                # rapidities = T.tensor(pippin["rapidities"][...])

            # Extract the output and target point clouds
            pc_in.append(pointclouds[..., 0])
            pc_out.append(pointclouds[..., 1])
            pc_target.append(pointclouds[..., 2])

            # Extract the output and target masks
            mask_in = masks[..., 0]
            mask_out = masks[..., 1]
            mask_target = masks[..., 2]

            # Remove the zero padding in the input point cloud and mask
            pc_in[i] = pc_in[i][:, :6]
            mask_in = mask_in[:, :6]

            # Extract the predicted and true presences
            pred_p.append(presences[..., 0])
            true_p.append(presences[..., 1])

            # Recompute the observables
            # obs = (masses, momenta, rapidities, check, percent, match)
            obs_in = compute_observables(
                pc=pc_in[i],
                mask=mask_in,
                is_parton=True,
            )
            obs_out = compute_observables(
                pc=pc_out[i],
                mask=mask_out,
                pc_ref=pc_in[i],
                mask_ref=mask_in,
                channel=channel,
            )
            obs_target = compute_observables(
                pc=pc_target[i],
                mask=mask_target,
                pc_ref=pc_in[i],
                mask_ref=mask_in,
                channel=channel,
            )

            # Extract the output and target checks
            check_out.append(obs_out[3])
            check_target.append(obs_target[3])

            # Extract the output and target percentages
            percent_out.append(obs_out[4])
            percent_target.append(obs_target[4])

            # Extract the output and target matchabilities
            match_out.append(obs_out[5])
            match_target.append(obs_target[5])

            # Extract the input, output and target invariant masses
            masses_in.append(obs_in[0])
            masses_out.append(obs_out[0][check_out[i]])
            masses_target.append(obs_target[0][check_target[i]])

            # Extract the input, output and target transverse momenta
            momenta_in.append(obs_in[1])
            momenta_out.append(obs_out[1][check_out[i]])
            momenta_target.append(obs_target[1][check_target[i]])

            # Extract the input, output and target pseudo-rapidities
            rapidities_in.append(obs_in[2])
            rapidities_out.append(obs_out[2][check_out[i]])
            rapidities_target.append(obs_target[2][check_target[i]])

            # Compute additional quantities
            mult_out.append(get_particle_mult(
                mask=mask_out,
                channel=channel,
                level="reco",
                max_n=max_n,
                dataset_name=dataset_name,
            ))
            mult_target.append(get_particle_mult(
                mask=mask_target,
                channel=channel,
                level="reco",
                max_n=max_n,
                dataset_name=dataset_name,
            ))
            mask_out_part.append(get_particle_masks(
                mask=mask_out,
                channel=channel,
                level="reco",
                max_n=max_n,
                dataset_name=dataset_name,
            ))
            mask_target_part.append(get_particle_masks(
                mask=mask_target,
                channel=channel,
                level="reco",
                max_n=max_n,
                dataset_name=dataset_name,
            ))

            # Get Turbo-Sim like mask
            mask_turbo.append(get_mask_turbo(
                masks=[mask_target],
                channel=channel,
                n_max=n_test_turbosim,
            ))

            log.info(f"Data loaded for run {run}")


            if args.metrics:
                path = path_metrics/f"{run}"

                if args.inclusive:
                    log.info("Start computing KS distances: 'inclusive'")
                    ks_incl = compute_ks(
                        pc_x=pc_out[i],
                        mask_x_part=mask_out_part[i],
                        masses_x=masses_out[i],
                        momenta_x=momenta_out[i],
                        rapidities_x=rapidities_out[i],
                        pc_y=pc_target[i],
                        mask_y_part=mask_target_part[i],
                        masses_y=masses_target[i],
                        momenta_y=momenta_target[i],
                        rapidities_y=rapidities_target[i],
                    )
                    save_ks(
                        ks_incl,
                        path=path,
                        filename="ks_incl.yaml",
                    )

                if args.leading:
                    log.info("Start computing KS distances: 'leading'")
                    ks_lead = compute_ks(
                        pc_x=pc_out[i],
                        mask_x_part=mask_out_part[i],
                        masses_x=masses_out[i],
                        momenta_x=momenta_out[i],
                        rapidities_x=rapidities_out[i],
                        pc_y=pc_target[i],
                        mask_y_part=mask_target_part[i],
                        masses_y=masses_target[i],
                        momenta_y=momenta_target[i],
                        rapidities_y=rapidities_target[i],
                        only_leading=True,
                        cartesian=True,
                    )
                    save_ks(
                        ks_lead,
                        only_leading=True,
                        cartesian=True,
                        path=path,
                        filename="ks_lead.yaml",
                    )

                if args.turbolike:
                    log.info("Start computing KS distances: 'turbo-like'")
                    ks_turbo = compute_ks(
                        pc_x=pc_out[i][mask_turbo[i]],
                        mask_x_part=tuple([m[mask_turbo[i]] for m in mask_out_part[i]]),
                        masses_x=masses_out[i][mask_turbo[i][check_out[i]]],
                        momenta_x=momenta_out[i][mask_turbo[i][check_out[i]]],
                        rapidities_x=rapidities_out[i][mask_turbo[i][check_out[i]]],
                        pc_y=pc_target[i][mask_turbo[i]],
                        mask_y_part=tuple([m[mask_turbo[i]] for m in mask_target_part[i]]),
                        masses_y=masses_target[i][mask_turbo[i][check_target[i]]],
                        momenta_y=momenta_target[i][mask_turbo[i][check_target[i]]],
                        rapidities_y=rapidities_target[i][mask_turbo[i][check_target[i]]],
                        only_leading=True,
                        cartesian=True,
                    )
                    save_ks(
                        ks_turbo,
                        only_leading=True,
                        cartesian=True,
                        path=path,
                        filename="ks_turbolike.yaml",
                    )


    if args.turbosim:
        # Load Turbo-Sim outputs from OTUS-like dataset
        log.info("Start loading the Turbo-Sim data")
        with h5py.File(path_turbo, "r") as f:
            xi = T.tensor(f["xi"][...]).view(-1, 6, 4)
            zi = T.tensor(f["zi"][...]).view(-1, 6, 4)
            xt = T.tensor(f["xt"][...]).view(-1, 6, 4)
            # zt = T.tensor(f["zt"][...])
            # xh = T.tensor(f["xh"][...])
            # zh = T.tensor(f["zh"][...])

        # Reorder partons
        zi = zi[:, [2, 0, 1, 3, 4, 5]]

        # Mimic PIPPIN outputs
        pc_in_ts = to_polar(zi, return_mass=True)
        pc_out_ts = to_polar(xt)
        pc_target_ts = to_polar(xi)

        mask_in_ts = T.ones(size=pc_in_ts.shape[:-1], dtype=T.bool)
        mask_out_ts = T.ones(size=pc_out_ts.shape[:-1], dtype=T.bool)
        mask_target_ts = T.ones(size=pc_target_ts.shape[:-1], dtype=T.bool)

        channel_ts = T.tensor([0b01] * len(pc_in_ts))


        masses_in_ts, momenta_in_ts, rapidities_in_ts, _, _, _ = compute_observables(
            pc=pc_in_ts,
            mask=mask_in_ts,
            is_parton=True,
        )
        masses_out_ts, momenta_out_ts, rapidities_out_ts, check_out_ts, percent_out_ts, match_out_ts = compute_observables(
            pc=pc_out_ts,
            mask=mask_out_ts,
            pc_ref=pc_in_ts,
            mask_ref=mask_in_ts,
            channel=channel_ts,
            matching_R=0.8,
            accept_multi_part=True,
            accept_multi_reco=True,
        )
        masses_target_ts, momenta_target_ts, rapidities_target_ts, check_target_ts, percent_target_ts, match_target_ts = compute_observables(
            pc=pc_target_ts,
            mask=mask_target_ts,
            pc_ref=pc_in_ts,
            mask_ref=mask_in_ts,
            channel=channel_ts,
        )

        shape = pc_out_ts.shape
        mask_lep = T.cat([
            T.ones(size=(shape[0], 1), dtype=T.bool),
            T.zeros(size=(shape[0], shape[1]-1), dtype=T.bool),
        ],
        dim=-1,
        )
        mask_met = T.cat([
            T.zeros(size=(shape[0], 1), dtype=T.bool),
            T.ones(size=(shape[0], 1), dtype=T.bool),
            T.zeros(size=(shape[0], shape[1]-2), dtype=T.bool),
        ],
        dim=-1,
        )
        mask_jet = T.cat([
            T.zeros(size=(shape[0], 2), dtype=T.bool),
            T.ones(size=(shape[0], shape[1]-2), dtype=T.bool),
        ],
        dim=-1,
        )
        mask_out_part_ts = (mask_lep, mask_met, mask_jet)
        mask_target_part_ts = (mask_lep, mask_met, mask_jet)

        mult_lep = T.sum(mask_lep, dim=-1, keepdim=True).float()
        mult_met = T.sum(mask_met, dim=-1, keepdim=True).float()
        mult_jet = T.sum(mask_jet, dim=-1, keepdim=True).float()
        mult_out_ts = (mult_lep, mult_met, mult_jet)
        mult_target_ts = (mult_lep, mult_met, mult_jet)

        if args.metrics:
            log.info("Start computing KS distances: 'Turbo-Sim'")
            ks_ts = compute_ks(
                pc_x=pc_out_ts,
                mask_x_part=mask_out_part_ts,
                masses_x=masses_out_ts[check_out_ts],
                momenta_x=momenta_out_ts[check_out_ts],
                rapidities_x=rapidities_out_ts[check_out_ts],
                pc_y=pc_target_ts,
                mask_y_part=mask_target_part_ts,
                masses_y=masses_target_ts[check_target_ts],
                momenta_y=momenta_target_ts[check_target_ts],
                rapidities_y=rapidities_target_ts[check_target_ts],
                only_leading=True,
                cartesian=True,
            )

            save_ks(
                ks_ts,
                only_leading=True,
                cartesian=True,
                path=path_metrics,
                filename="ks_turbosim.yaml",
            )

    log.info("All data loaded and processed")


    if args.metrics:
        modes = []
        if args.inclusive: modes.append("incl")
        if args.leading: modes.append("lead")
        if args.turbolike: modes.append("turbolike")

        if len(modes) > 0:
            log.info(f"Start combining the KS distances: {modes}")
            # Reaload saved KS distances and compute average for the combined runs
            for mode in modes:
                ks_combined = {}
                for run in runs:
                    path = path_metrics/f"{run}/ks_{mode}.yaml"
                    with open(path, "r") as f:
                        ks = yaml.load(f, Loader=yaml.FullLoader)
                    ks_combined = {k: ks_combined.get(k, []) + [v] for k, v in ks.items()}
                ks_avg = {k: np.mean(v).item() for k, v in ks_combined.items()}
                ks_std = {k: np.std(v).item() for k, v in ks_combined.items()}

                # Save the combined KS distances
                path = path_metrics/"combined"/mode
                os.makedirs(path, exist_ok=True)
                with open(path/"ks_avg.yaml", "w") as f:
                    yaml.dump(ks_avg, f)
                with open(path/"ks_std.yaml", "w") as f:
                    yaml.dump(ks_std, f)

            log.info("All KS distances combined and saved")


    # Make the combined plots
    if args.plots:

        if args.inclusive:
            log.info("Start making the combined plots: 'inclusive'")
            make_plots(
                pc_out=pc_out,
                mult_out=mult_out,
                mask_out_part=mask_out_part,
                pc_target=pc_target,
                mult_target=mult_target,
                mask_target_part=mask_target_part,
                pred_p=pred_p,
                true_p=true_p,
                match_out=match_out,
                match_target=match_target,
                masses_in=masses_in,
                masses_out=masses_out,
                masses_target=masses_target,
                momenta_in=momenta_in,
                momenta_out=momenta_out,
                momenta_target=momenta_target,
                rapidities_in=rapidities_in,
                rapidities_out=rapidities_out,
                rapidities_target=rapidities_target,
                percent_out=T.cat(percent_out).mean().unsqueeze(0),
                percent_target=T.cat(percent_target).mean().unsqueeze(0),
                path=path_plots/"combined/incl",
            )

        if args.leading:
            log.info("Start making the combined plots: 'leading'")
            make_plots(
                pc_out=pc_out,
                mult_out=mult_out,
                mask_out_part=mask_out_part,
                pc_target=pc_target,
                mult_target=mult_target,
                mask_target_part=mask_target_part,
                pred_p=pred_p,
                true_p=true_p,
                match_out=match_out,
                match_target=match_target,
                masses_in=masses_in,
                masses_out=masses_out,
                masses_target=masses_target,
                momenta_in=momenta_in,
                momenta_out=momenta_out,
                momenta_target=momenta_target,
                rapidities_in=rapidities_in,
                rapidities_out=rapidities_out,
                rapidities_target=rapidities_target,
                percent_out=T.cat(percent_out).mean().unsqueeze(0),
                percent_target=T.cat(percent_target).mean().unsqueeze(0),
                only_leading=True,
                cartesian=True,
                path=path_plots/"combined/lead",
            )

        if args.turbolike:
            log.info("Start making the combined plots: 'turbo-like'")
            # Make the Turbo-Sim like plots
            make_plots(
                pc_out=[pc_out[i][mask_turbo[i]] for i in range(len(runs))],
                mult_out=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mult_out[i]]),
                mask_out_part=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mask_out_part[i]]),
                pc_target=[pc_target[i][mask_turbo[i]] for i in range(len(runs))],
                mult_target=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mult_target[i]]),
                mask_target_part=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mask_target_part[i]]),
                pred_p=[pred_p[i][mask_turbo[i]] for i in range(len(runs))],
                true_p=[true_p[i][mask_turbo[i]] for i in range(len(runs))],
                match_out=[match_out[i][mask_turbo[i]] for i in range(len(runs))],
                match_target=[match_target[i][mask_turbo[i]] for i in range(len(runs))],
                masses_in=[masses_in[i][mask_turbo[i]] for i in range(len(runs))],
                masses_out=[masses_out[i][mask_turbo[i][check_out[i]]] for i in range(len(runs))],
                masses_target=[masses_target[i][mask_turbo[i][check_target[i]]] for i in range(len(runs))],
                momenta_in=[momenta_in[i][mask_turbo[i]] for i in range(len(runs))],
                momenta_out=[momenta_out[i][mask_turbo[i][check_out[i]]] for i in range(len(runs))],
                momenta_target=[momenta_target[i][mask_turbo[i][check_target[i]]] for i in range(len(runs))],
                rapidities_in=[rapidities_in[i][mask_turbo[i]] for i in range(len(runs))],
                rapidities_out=[rapidities_out[i][mask_turbo[i][check_out[i]]] for i in range(len(runs))],
                rapidities_target=[rapidities_target[i][mask_turbo[i][check_target[i]]] for i in range(len(runs))],
                percent_out=T.tensor([c[m].float().mean() for c, m in zip(check_out, mask_turbo)]).mean().unsqueeze(0),
                percent_target=T.tensor([c[m].float().mean() for c, m in zip(check_target, mask_turbo)]).mean().unsqueeze(0),
                only_leading=True,
                cartesian=True,
                path=path_plots/"combined/turbolike",
            )

        if args.turbosim:
            log.info("Start making the plots: 'Turbo-Sim'")
            make_plots(
                pc_out=[pc_out_ts],
                mult_out=[mult_out_ts],
                mask_out_part=[mask_out_part_ts],
                pc_target=[pc_target_ts],
                mult_target=[mult_target_ts],
                mask_target_part=[mask_target_part_ts],
                pred_p=None,
                true_p=None,
                match_out=[match_out_ts],
                match_target=[match_target_ts],
                masses_in=[masses_in_ts],
                masses_out=[masses_out_ts[check_out_ts]],
                masses_target=[masses_target_ts[check_target_ts]],
                momenta_in=[momenta_in_ts],
                momenta_out=[momenta_out_ts[check_out_ts]],
                momenta_target=[momenta_target_ts[check_target_ts]],
                rapidities_in=[rapidities_in_ts],
                rapidities_out=[rapidities_out_ts[check_out_ts]],
                rapidities_target=[rapidities_target_ts[check_target_ts]],
                percent_out=percent_out_ts,
                percent_target=percent_target_ts,
                only_leading=True,
                cartesian=True,
                model="turbosim",
                path=path_plots/"turbosim",
            )

    if args.compare and args.turbolike and args.turbosim:
        log.info("Start making the comparison plots: 'turbo-like' & 'Turbo-Sim'")
        make_comparison_plots(
            pc_out=[pc_out[i][mask_turbo[i]] for i in range(len(runs))],
            mult_out=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mult_out[i]]),
            mask_out_part=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mask_out_part[i]]),
            masses_out=[masses_out[i][mask_turbo[i][check_out[i]]] for i in range(len(runs))],
            pc_target=[pc_target[i][mask_turbo[i]] for i in range(len(runs))],
            mult_target=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mult_target[i]]),
            mask_target_part=tuple([m[mask_turbo[i]] for i in range(len(runs)) for m in mask_target_part[i]]),
            masses_target=[masses_target[i][mask_turbo[i][check_target[i]]] for i in range(len(runs))],
            pc_alt=[pc_out_ts],
            mult_alt=[mult_out_ts],
            mask_alt_part=[mask_out_part_ts],
            masses_alt=[masses_out_ts[check_out_ts]],
            only_leading=True,
            cartesian=True,
            model="pippin",
            model_alt="turbosim",
            path=path_plots/"comparison",
        )


# Define the function to make all plots
def make_plots(
    pc_out,
    mult_out,
    mask_out_part,
    pc_target,
    mult_target,
    mask_target_part,
    pred_p,
    true_p,
    match_out,
    match_target,
    masses_in,
    masses_out,
    masses_target,
    momenta_in,
    momenta_out,
    momenta_target,
    rapidities_in,
    rapidities_out,
    rapidities_target,
    percent_out,
    percent_target,
    only_leading=False,
    cartesian=False,
    model="pippin",
    path='.',
):

    # Record the reconstructions using marginal histograms
    _ = plot_marginals(
        pc_out=pc_out,
        mult_out=mult_out,
        mask_out=None,
        mask_out_part=mask_out_part,
        pc_target=pc_target,
        mult_target=mult_target,
        mask_target=None,
        mask_target_part=mask_target_part,
        channel=None,
        pred_p=pred_p,
        true_p=true_p,
        match_out=match_out,
        match_target=match_target,
        only_leading=only_leading,
        cartesian=cartesian,
        model=model,
        path=path/"marginals",
    )

    # Record the reconstructions using 2D marginal histograms
    _ = plot_marginals_2D(
        pc_out=pc_out[-1],
        mult_out=mult_out[-1],
        mask_out=None,
        mask_out_part=mask_out_part[-1],
        pc_target=pc_target[-1],
        mult_target=mult_target[-1],
        mask_target=None,
        mask_target_part=mask_target_part[-1],
        channel=None,
        pred_p=pred_p[-1] if pred_p is not None else None,
        true_p=true_p[-1] if true_p is not None else None,
        model=model,
        path=path/"marginals_2D",
    )

    # Record the invariant masses using histograms
    _ = plot_masses(
        masses_in=masses_in,
        masses_out=masses_out,
        masses_target=masses_target,
        percent_out=percent_out,
        percent_target=percent_target,
        model=model,
        path=path/"masses",
    )

    # Record the transverse momenta using histograms
    _ = plot_momenta(
        momenta_in=momenta_in,
        momenta_out=momenta_out,
        momenta_target=momenta_target,
        percent_out=percent_out,
        percent_target=percent_target,
        model=model,
        path=path/"momenta",
    )

    # Record the pseudo-rapidities using histograms
    _ = plot_rapidities(
        rapidities_in=rapidities_in,
        rapidities_out=rapidities_out,
        rapidities_target=rapidities_target,
        percent_out=percent_out,
        percent_target=percent_target,
        model=model,
        path=path/"rapidities",
    )


def make_comparison_plots(
    pc_out,
    mult_out,
    mask_out_part,
    masses_out,
    pc_target,
    mult_target,
    mask_target_part,
    masses_target,
    pc_alt,
    mult_alt,
    mask_alt_part,
    masses_alt,
    only_leading=False,
    cartesian=False,
    model="pippin",
    model_alt="turbosim",
    path='.',
):

    # Compare the reconstructions using marginal histograms
    _ = plot_marginals(
        pc_out=pc_out,
        mult_out=mult_out,
        mask_out=None,
        mask_out_part=mask_out_part,
        pc_target=pc_target,
        mult_target=mult_target,
        mask_target=None,
        mask_target_part=mask_target_part,
        channel=None,
        pred_p=None,
        true_p=None,
        match_out=None,
        match_target=None,
        pc_alt=pc_alt,
        mult_alt=mult_alt,
        mask_alt_part=mask_alt_part,
        only_leading=only_leading,
        cartesian=cartesian,
        model=model,
        model_alt=model_alt,
        path=path/"marginals",
    )

    # Compare the invariant masses using histograms
    _ = plot_masses(
        masses_in=None,
        masses_out=masses_out,
        masses_target=masses_target,
        masses_alt=masses_alt,
        model=model,
        model_alt=model_alt,
        path=path/"masses",
    )


if __name__ == "__main__":
    main()
