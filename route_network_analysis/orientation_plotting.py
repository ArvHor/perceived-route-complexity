import osmnx as ox
import networkx as nx
import numpy as np

from . import map_plotting as mp

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
from matplotlib import colors
from matplotlib.axes._axes import Axes  # noqa: TC002
from matplotlib.figure import Figure  # noqa: TC002
from matplotlib.projections.polar import PolarAxes  # noqa: TC002

from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

def plot_all_city_routes_html(od_pair_data,city_name):
    
    od_pair_data = od_pair_data[od_pair_data['city_name'] == city_name]
    unique_origins  = od_pair_data['origin_point'].unique()

    all_city_routes = []
    for unique_origin in unique_origins:
        shortest_path_nodes = od_pair_data[od_pair_data['origin_point'] == unique_origin]['shortest_path_nodes'].values[0]
        graph_path = od_pair_data[od_pair_data['origin_point'] == unique_origin]['graph_path'].values[0]
        graph = ox.load_graphml(graph_path)
        route_gdf = ox.routing.route_to_gdf(graph, shortest_path_nodes, weight='length')
        all_city_routes.append(route_gdf)

    mp.plot_all_routes(all_city_routes,"demonstration/barcelona.html",unique_origins)



def plot_orientation(
    bin_counts,
    *,
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (5, 5),
    area: bool = True,
    color: str = "#004991",
    edgecolor: str = "k",
    linewidth: float = 0.5,
    alpha: float = 0.7,
    title: str | None = None,
    title_y: float = 1.05,
    title_font: dict[str] | None = None,
    xtick_font: dict[str] | None = None,
) -> tuple[Figure, PolarAxes]:
    """
    Plot a polar histogram of a spatial network's edge bearings.
    """

    if title_font is None:
        title_font = {"family": "DejaVu Sans", "size": 24, "weight": "bold"}
    if xtick_font is None:
        xtick_font = {
            "family": "Courier New",
            "size": 10,
            "weight": "bold",
            "alpha": 1.0,
            "zorder": 3,
        }
    num_bins = len(bin_counts)
    bin_centers = np.arange(0, 180, 180 / num_bins)  # Ensure this matches the number of labels

    positions = np.radians(bin_centers)

    # width: make bars fill the circumference without gaps or overlaps
    width = 2 * np.pi / 36  # Each bin is 10 degrees wide

    # radius: how long to make each bar. set bar length so either the bar area
    # (ie, via sqrt) or the bar height is proportional to the bin's frequency
    bin_frequency = bin_counts / bin_counts.sum()
    radius = np.sqrt(bin_frequency) if area else bin_frequency

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)  # Set the direction to counter-clockwise
    ax.set_ylim(top=radius.max())

    # Set the theta limits to display only the upper half of the plot
    ax.set_thetamin(0)
    ax.set_thetamax(175)

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, radius.max(), 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels for 10 degree increments
    xtick_degrees = np.arange(0, 181, 10)
    xticklabels = [str(deg) for deg in xtick_degrees]
    ax.set_xticks(np.radians(xtick_degrees))
    ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.bar(
        positions,
        height=radius,
        width=width,
        align="center",
        bottom=0,
        zorder=2,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=1.0,  # Ensure this is not transparent
    )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)
    fig.tight_layout()
    return fig, ax


def _get_fig_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
    bgcolor: str | None,
    polar: bool,  # noqa: FBT001
) -> tuple[Figure, Axes | PolarAxes]:
    """
    Generate a matplotlib Figure and (Polar)Axes or return existing ones.

    Parameters
    ----------
    ax
        If not None, plot on this pre-existing axes instance.
    figsize
        If `ax` is None, create new figure with size `(width, height)`.
    bgcolor
        Background color of figure.
    polar
        If True, generate a `PolarAxes` instead of an `Axes` instance.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        if polar:
            # make PolarAxes
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        else:
            # make regular Axes
            fig, ax = plt.subplots(figsize=figsize, facecolor=bgcolor, frameon=False)
            ax.set_facecolor(bgcolor)
    else:
        fig = ax.figure  # type: ignore[assignment]

    return fig, ax


# if polar = False, return Axes
@overload
def _get_fig_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
    bgcolor: str | None,
    polar: Literal[False],
) -> tuple[Figure, Axes]: ...


# if polar = True, return PolarAxes
@overload
def _get_fig_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
    bgcolor: str | None,
    polar: Literal[True],
) -> tuple[Figure, PolarAxes]: ...


def _get_fig_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
    bgcolor: str | None,
    polar: bool,  # noqa: FBT001
) -> tuple[Figure, Axes | PolarAxes]:
    """
    Generate a matplotlib Figure and (Polar)Axes or return existing ones.

    Parameters
    ----------
    ax
        If not None, plot on this pre-existing axes instance.
    figsize
        If `ax` is None, create new figure with size `(width, height)`.
    bgcolor
        Background color of figure.
    polar
        If True, generate a `PolarAxes` instead of an `Axes` instance.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        if polar:
            # make PolarAxes
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        else:
            # make regular Axes
            fig, ax = plt.subplots(figsize=figsize, facecolor=bgcolor, frameon=False)
            ax.set_facecolor(bgcolor)
    else:
        fig = ax.figure  # type: ignore[assignment]

    return fig, ax