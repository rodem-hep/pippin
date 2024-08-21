import os
from pathlib import Path
from typing import List, Tuple

import PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import torch as T

from src.utils import no_inf

from mattstools.mattstools.torch_utils import to_np

# Flag to lighten the number of plots for debugging
DEBUG = False


# Reset the matplotlib rcParams to the default values
plt.rcdefaults()

# Set some of the matplotlib rcParams to new values
plt.rcParams["xaxis.labellocation"] = "right"
plt.rcParams["yaxis.labellocation"] = "top"
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cmr10"  # "Computer Modern Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["mathtext.fontset"] = "cm"  # For math mode, i.e. $...$
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["patch.linewidth"] = 2

plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["xtick.minor.width"] = 4/3
plt.rcParams["xtick.minor.size"] = 4
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["ytick.minor.width"] = 4/3
plt.rcParams["ytick.minor.size"] = 4
plt.rcParams["ytick.minor.visible"] = True


DEFAULT_COLORS = ["grey"] + [f"C{i}" for i in range(1, 10)]


def plot_color_pointcloud(
    pc: np.ndarray,
    mask: np.ndarray,
    xrange: Tuple | None = None,
    yrange: Tuple | None = None,
    zrange: Tuple | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    z_label: str | None = None,
    show: bool = False,
    save: bool = False,
    save_png: bool = True,
    save_pdf: bool = False if DEBUG else True,
    path: str | None = None,
    filename: str | None = None,
) -> PIL.Image.Image:
    """
    Creates a scatter plot of a 3D cloud using the first dimension as the colour

    Args:
        pc: A 3D point cloud
        mask: A boolean array showing which nodes are real vs padded

    Kwargs:
        xrange: The range for the x-axis
        yrange: The range for the y-axis
        zrange: The range for the z-axis (not used)
        x_label: The label for the x-axis
        y_label: The label for the y-axis
        z_label: The label for the z-axis
        show: If the plot should be shown
        save: If the plot should be saved
        save_png: If the plot should be saved as a PNG
        save_pdf: If the plot should be saved as a PDF
        path: The path to save the plot to
        filename: The name of the file to save the plot to

    Returns:
        img: The image of the plot
    """

    # Make sure we are dealing with numpy arrays
    if isinstance(pc, T.Tensor):
        pc = to_np(pc)
    if isinstance(mask, T.Tensor):
        mask = to_np(mask)

    # Convert strings to Path objects
    path = Path(path) if path is not None else None
    filename = Path(filename) if filename is not None else None

    # Trim to only 3 dimensions
    pc = pc[..., :3]

    # Create the figure and plot the data
    fig, axis = plt.subplots(1, 1, figsize=(4, 4))

    z, x, y = pc[mask].T  # The mask should be unidimensional here
    if len(z) != 0: z = (z - z.min()) / (z.max() - z.min() + 1e-15)

    p = axis.scatter(x, y, c=z, cmap="plasma")

    # Set the labels, colourbar and layout
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    cbar = plt.colorbar(p)
    cbar.set_label(z_label)

    axis.set_xlim(xrange)
    axis.set_ylim(yrange)

    fig.tight_layout()

    # Show if specified
    if show: plt.show()

    # Save if specified and create the folder if it doesnt exist
    if save:
        os.makedirs(path, exist_ok=True)
        if save_png: fig.savefig(f"{path/filename}.png", dpi=300)
        if save_pdf: fig.savefig(f"{path/filename}.pdf")

    # Convert to an image to be returned
    fig.canvas.draw()
    img = PIL.Image.frombytes(
        "RGB",
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb(),
    )

    # Close the figure
    plt.close(fig)

    return img


def plot_hist(
    data_list: List[List[np.ndarray | T.Tensor]],
    n_bins: int = 30,
    x_range: Tuple[float, float] | None = None,
    y_range: Tuple[float, float] | None = None,
    r_range: Tuple[float, float] | None = None,
    legends: List[str] | None = None,
    l_styles: List[str] | None = None,
    colors: List[str] | None = None,
    x_label: str | None = None,
    x_scale: str | None = None,
    y_label: str | None = None,
    y_label_ratio: str | None = None,
    y_scale: str = "linear",
    integers: bool = False,
    normalise: bool = False,
    proportion: bool = False,
    proportion_total: int = 1,
    is_categorical: bool = False,
    categories: List[str] | None = None,
    ratio_idx: List[int] | None = None,
    show_error: bool | List[bool] = True,
    poisson_idx: List[int] | None = None,
    show: bool = False,
    save: bool = False,
    save_png: bool = True,
    save_pdf: bool = False if DEBUG else True,
    path: str | None = None,
    filename: str | None = None,
) -> PIL.Image.Image:
    """
    Plots a histogram of the provided data

    Args:
        data_list: A list of data to be plotted or counts (if is_categorical is True)

    Kwargs:
        n_bins: The number of bins to use
        x_range: The range for the x-axis. If not specified it will be calculated from the data
        y_range: The range for the y-axis. If not specified it will be calculated from the data
        r_range: The range for the ratio plot. Default to (0.5, 1.5)
        legends: A list of legends for each data set
        l_styles: A list of line styles for each data set
        colors: A list of colours for each data set
        x_label: The label for the x-axis
        x_scale: The scale for the x-axis. Default to y_scale.
        y_label: The label for the y-axis
        y_label_ratio: The label for the y-axis of the ratio plot
        y_scale: The scale for the y-axis
        integers: If the data is integers (automatically set to True if is_categorical is True)
        normalise: If the data should be normalised to 1
        proportion: If the data should be shown as a proportion of total counts
        proportion_total: The total number of counts to use for the proportion
        is_categorical: If the data is categorical
        categories: The categories (labels) for the categorical data
        ratio_idx: The indices of the data to show in the ratio plot
        show_error: If the error bands should be shown
        poisson_idx: The indices of the data for which error is Poissonian
        show: If the plot should be shown
        save: If the plot should be saved
        save_png: If the plot should be saved as a PNG
        save_pdf: If the plot should be saved as a PDF
        path: The path to save the plot to
        filename: The name of the file to save the plot to

    Returns:
        img: The image of the plot
    """

    # Check arguments combinations
    if normalise and proportion:
        raise ValueError("Cannot use 'normalise' and 'proportion' at the same time.")

    # Remove infinities and NaNs from the data
    data_list = [[no_inf(samples) for samples in data] for data in data_list]

    # Make sure we are dealing with numpy arrays
    data_list = to_np(data_list)

    # Set the defaut values for the unspecified arguments
    r_range = r_range or (0.5, 1.5)
    l_styles = l_styles or ["-"] * len(data_list)
    colors = colors or DEFAULT_COLORS
    x_scale = x_scale or y_scale
    ratio_idx = ratio_idx or list(range(len(data_list)))
    if isinstance(show_error, bool): show_error = [show_error] * len(data_list)
    poisson_idx = poisson_idx or [0]
    path = Path(path) if path is not None else None
    filename = Path(filename) if filename is not None else None

    # Set integers to True if the data is categorical
    if is_categorical: integers = True

    # Create the figure
    fig, axes = plt.subplots(
        2, 1,
        figsize=(4, 5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Check min/max values to define the range
    if not is_categorical:
        if x_range is None:
            x_min = np.inf
            x_max = -np.inf
            for data in data_list:
                for samples in data:
                    if len(samples) > 0:
                        x_min = min(samples.min(), x_min)
                        x_max = max(samples.max(), x_max)

            if integers:
                x_max += 1
                n_bins = int(x_max - x_min)
        else:
            x_min, x_max = x_range

        if x_scale == "log":
            bins = np.logspace(
                np.log10(max(x_min, 0.1)),
                np.log10(x_max),
                n_bins+1,
            )
        else:
            bins = n_bins

    # Plot the data
    # - normalise to 1 if specified
    # - with error bands if specified
    for i, data in enumerate(data_list):

        # Compute the histogram and the error
        if is_categorical:
            counts_list = data
            edges = np.arange(len(counts_list[0]) + 1)
        else:
            counts_list = []
            for samples in data:
                counts, edges = np.histogram(
                    samples,
                    bins=bins,
                    range=(x_min, x_max),
                )
                counts_list.append(counts)

        # Compute the mean and error of the counts
        counts = np.mean(counts_list, axis=0).round()
        if i in poisson_idx:
            yerr = np.sqrt(counts) if show_error else None
        else:
            yerr = np.std(counts_list, axis=0) if show_error else None

        # Save the reference counts for the ratio plot and compute the ratios
        counts_max = counts.max()
        if i == 0:
            counts_ref = counts
            counts_ref_max = counts_max
        ratios = counts / (counts_ref + 1e-15)
        yerr = yerr / (counts_ref + 1e-15)

        # Normalise the values to if specified
        if normalise:
            ratios = ratios / (counts_max + 1e-15) * counts_ref_max
            yerr = yerr / (counts_max + 1e-15) * counts_ref_max
            counts = counts / (counts_max + 1e-15)
        elif proportion:
            counts = counts / proportion_total

        # Clip the ratios to be at most between the 'r_range'
        # if the error bands are not shown
        if not show_error[i]:
            offset = 0.01*(r_range[1] - r_range[0])
            ratios = np.clip(ratios, r_range[0]+offset, r_range[1]-offset)

        # Do not plot if the reference counts are zero
        ratios[counts_ref == 0] = np.nan
        yerr[counts_ref == 0] = np.nan

        # Do not plot if the counts are all zero (i.e. no data)
        if counts_max == 0:
            ratios[:] = np.nan
            yerr[:] = np.nan

        # Shift the edges if integers in order to center the bars
        if integers:
            edges = edges - 0.5

        # Plot the histogram
        h = axes[0].stairs(
            values=counts,
            edges=edges,
            color=colors[i],
            linestyle=l_styles[i],
            fill=True if i == 0 else False,
            alpha=0.5 if i == 0 else 0.8,
            label=legends[i] if legends is not None else None,
        )

        if i in ratio_idx:
            # Plot the ratios
            r = axes[1].stairs(
                values=ratios,
                edges=edges,
                baseline=1,
                color=colors[i],
                linestyle=l_styles[i],
                alpha=0.5 if i == 0 else 0.8,
            )

            # Plot the error bands if specified
            if show_error[i]:
                axes[1].stairs(
                    values=ratios + yerr,
                    edges=edges,
                    baseline=ratios - yerr,
                    fill=True,
                    hatch="///" if i == 0 else None,
                    alpha=0.2,
                    color=r._edgecolor,
                )

    # Set the labels, legend and layout
    if legends is not None:
        axes[0].legend()
    if not is_categorical:
        axes[0].set_xscale(x_scale)
    axes[0].set_yscale(y_scale)
    if y_label is not None:
        axes[0].set_ylabel(y_label)
    elif normalise:
        axes[0].set_ylabel("Normalised")
    elif proportion:
        axes[0].set_ylabel("Proportion")
    else:
        axes[0].set_ylabel("Counts")

    if not is_categorical:
        axes[1].set_xscale(x_scale)
    axes[1].set_xlabel(x_label)
    if y_label_ratio is not None:
        axes[1].set_ylabel(y_label_ratio)
    else:
        axes[1].set_ylabel("Ratio")

    if is_categorical:
        axes[1].set_xticks(
            np.arange(len(categories)),
            categories,
            rotation=90,
            # ha="right",
        )

    # Give each legend item 15% of the y-axis
    plot_space = 1-0.15*len(legends) if legends is not None else 1
    y_min, y_max = y_range or axes[0].get_ylim()
    if y_scale == "log":
        # Ensure that there is at least a factor of 10 between x_min and x_max
        if y_max/y_min < 10:
            y_min = y_max/10
        y_max = np.exp(np.log(y_min)+(np.log(y_max)-np.log(y_min))/plot_space)
    else:
        y_max = y_min + (y_max-y_min)/plot_space
    axes[0].set_ylim(y_min, y_max)

    # Set the y-axis to be formatted as scientific notation
    if y_scale != "log":
        axes[0].ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )

    # Clip the ratio plot to be symmetrical and at most between 0.5 and 1.5
    axes[1].set_ylim(r_range)

    # Set ticks parameters
    axes[0].tick_params(axis="y", which="minor", left=False, right=False)
    axes[1].tick_params(axis="y", which="minor", left=False, right=False)
    if integers or is_categorical:
        axes[0].tick_params(axis="x", which="minor", bottom=False, top=False)
        axes[1].tick_params(axis="x", which="minor", bottom=False, top=False)
    else:
        axes[0].tick_params(axis="x", which="minor", bottom=True, top=True)
        axes[1].tick_params(axis="x", which="minor", bottom=True, top=True)
    if integers:
        axes[0].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=1))
        axes[1].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=1))

    # Prettify the plot
    fig.align_ylabels()
    fig.tight_layout()

    # Show if specified
    if show: plt.show()

    # Save if specified and create the folder if it doesnt exist
    if save:
        os.makedirs(path, exist_ok=True)
        if save_png: fig.savefig(f"{path/filename}.png", dpi=300)
        if save_pdf: fig.savefig(f"{path/filename}.pdf")

    # Convert to an image and return
    fig.canvas.draw()
    img = PIL.Image.frombytes(
        "RGB",
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb(),
    )

    # Close the figure
    plt.close(fig)

    return img


def plot_hist2d(
    data: List[np.ndarray | T.Tensor] | np.ndarray | T.Tensor,
    n_bins: int = 50,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    xy_format: str = None,
    xy_scale: str | None = None,
    c_scale: str = "linear",
    integers: bool = False,
    is_categorical: bool = False,
    categories: List[str] | None = None,
    show: bool = False,
    save: bool = False,
    save_png: bool = True,
    save_pdf: bool = False if DEBUG else True,
    path: str = None,
    filename: str = None,
) -> PIL.Image.Image:
    """
    Plots a 2D histogram of the provided data

    Args:
        data: The data to be plotted or counts (if is_categorical is True)

    Kwargs:
        n_bins: The number of bins to use
        x_range: The range for the x-axis. If not specified it will be calculated from the data
        y_range: The range for the y-axis. If not specified it will be calculated from the data
        title: The title of the plot
        x_label: The label for the x-axis
        y_label: The label for the y-axis
        xy_format: The format for the x and y axes
        xy_scale: The scale for the x and y axes. Default to c_scale.
        c_scale: The scale for the colour map
        integers: If the data is integers
        is_categorical: If the data is categorical
        categories: The categories (labels) for the categorical data
        show: If the plot should be shown
        save: If the plot should be saved
        save_png: If the plot should be saved as a PNG
        save_pdf: If the plot should be saved as a PDF
        path: The path to save the plot to
        filename: The name of the file to save the plot to

    Returns:
        img: The image of the plot
    """

    # Unpack data and make sure we are dealing with numpy arrays
    if isinstance(data, list) and len(data) == 2:
        data_x, data_y = to_np(data)
    else:
        data = to_np(data)

    # Set the xy_scale to c_scale if not specified
    xy_scale = xy_scale or c_scale

    # Convert strings to Path objects
    path = Path(path) if path is not None else None
    filename = Path(filename) if filename is not None else None

    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    if is_categorical:
        counts = data
        xedges = np.arange(len(counts) + 1)
        yedges = xedges

    else:
        # Check min/max values to define the ranges
        x_min, x_max = data_x.min(), data_x.max() if x_range is None else x_range
        y_min, y_max = data_y.min(), data_y.max() if y_range is None else y_range

        # Use the same range for both axes to make the plot square
        xy_min = min(x_min, y_min)
        xy_max = max(x_max, y_max)

        # Tweak the range and bins if data are integers
        if integers:
            n_bins = int(xy_max - xy_min) + 1
            xy_min -= 0.5
            xy_max += 0.5

        # Tweak the bins if the x and y axes are logarithmic
        if xy_scale == "log":
            bins = np.logspace(
                np.log10(max(xy_min, 0.1)),
                np.log10(xy_max),
                n_bins+1,
            )
        else:
            bins = n_bins

        # Compute the 2D histogram
        counts, xedges, yedges = np.histogram2d(
            data_x,
            data_y,
            bins=(bins, bins),
            range=((xy_min, xy_max), (xy_min, xy_max)),
        )

    # Convert scale to match pcolormesh arguments requirements
    c_scale = None if c_scale == "linear" else c_scale
    counts[counts == 0.] = np.nan

    # Safeguard against empty plot
    if np.isnan(counts).all():
        counts[0, 0] = 1

    # Plot the 2D histogram
    im = ax.pcolormesh(
        xedges,
        yedges,
        counts.T,
        cmap="plasma",
        norm=c_scale,
    )

    # Plot the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)

    # cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Counts")
    if c_scale != "log":
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)

    # Set the labels and layout
    ax.set_xlabel(x_label, loc="center")
    ax.set_ylabel(y_label, loc="center")

    # Set the x and y axes to be formatted as scientific notation if specified
    if xy_format == "sci":
        ax.ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )

    # Set the x and y axes to be logarithmic if specified
    if not is_categorical:
        ax.set_xscale(xy_scale)
        ax.set_yscale(xy_scale)

    # Set the ticks parameters
    if integers or is_categorical or xy_scale == "log":
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.tick_params(axis="y", which="minor", left=False, right=False)
    else:
        ax.tick_params(axis="x", which="minor", bottom=True, top=True)
        ax.tick_params(axis="y", which="minor", left=True, right=True)
    if integers:
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=1))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=1))

    # Set the title
    fig.suptitle(title)

    # Set the figure aspect
    ax.set_aspect("equal")
    fig.tight_layout()

    # Show if specified
    if show: plt.show()

    # Save if specified and create the folder if it doesnt exist
    if save:
        os.makedirs(path, exist_ok=True)
        if save_png: fig.savefig(f"{path/filename}.png", dpi=300)
        if save_pdf: fig.savefig(f"{path/filename}.pdf")

    # Convert to an image and return
    fig.canvas.draw()
    img = PIL.Image.frombytes(
        "RGB",
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb(),
    )

    # Close the figure
    plt.close(fig)

    return img
