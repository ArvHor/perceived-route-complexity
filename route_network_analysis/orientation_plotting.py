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

def create_orientation_plot(self, filepath):
    fig, ax = ox.plot_orientation(self.undirected_subgraph, weight="length", min_length=10)
    r_dist = self.route_direction_bearing_dist

    self._plot_overlaid_distribution(ax, r_dist, num_bins=36, area=True)
    fig.savefig(filepath)

  


def plot_orientation(  # noqa: PLR0913
    bin_frequency,
    *,
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (10, 10),
    area: bool = True,
    color: str = "#d3d3d3",
    edgecolor: str = "k",
    linewidth: float = 0.5,
    alpha: float = 1,
    title: str | None = None,
    title_y: float = 1.05,
    title_font: dict[str, Any] | None = None,
    xtick_font: dict[str, Any] | None = None,
) -> tuple[Figure, PolarAxes]:
    """
    Plot a polar histogram of a spatial network's edge bearings.

    Ignores self-loop edges as their bearings are undefined. If `G` is a
    MultiGraph, all edge bearings will be bidirectional (ie, two reciprocal
    bearings per undirected edge). If `G` is a MultiDiGraph, all edge bearings
    will be directional (ie, one bearing per directed edge). See also the
    `bearings` module.

    For more info see: Boeing, G. 2019. "Urban Spatial Order: Street Network
    Orientation, Configuration, and Entropy." Applied Network Science, 4 (1),
    67. https://doi.org/10.1007/s41109-019-0189-1

    Parameters
    ----------
    G
        Unprojected graph with `bearing` attributes on each edge.
    num_bins
        Number of bins. For example, if `num_bins=36` is provided, then each
        bin will represent 10 degrees around the compass.
    min_length
        Ignore edges with "length" attribute values less than `min_length`.
    weight
        If not None, weight the edges' bearings by this (non-null) edge
        attribute.
    ax
        If not None, plot on this pre-existing axes instance (must have
        projection=polar).
    figsize
        If `ax` is None, create new figure with size `(width, height)`.
    area
        If True, set bar length so area is proportional to frequency.
        Otherwise, set bar length so height is proportional to frequency.
    color
        Color of the histogram bars.
    edgecolor
        Color of the histogram bar edges.
    linewidth
        Width of the histogram bar edges.
    alpha
        Opacity of the histogram bars.
    title
        The figure's title.
    title_y
        The y position to place `title`.
    title_font
        The title's `fontdict` to pass to matplotlib.
    xtick_font
        The xtick labels' `fontdict` to pass to matplotlib.

    Returns
    -------
    fig, ax
    """

    if title_font is None:
        title_font = {"family": "monospace","name":"Courier","size": 24, "weight": "bold"}
    if xtick_font is None:
        xtick_font = {
            "family": "monospace",
            "name":"Courier",
            "size": 10,
            "weight": "bold",
            "alpha": 1.0,
            "zorder": 3,
        }

    # get the bearing distribution's bin counts and center values in degrees

    num_bins = len(bin_frequency)
    bin_centers = np.arange(0, 360, 360 / num_bins)

    positions = np.radians(bin_centers)

    # width: make bars fill the circumference without gaps or overlaps
    width = 2 * np.pi / num_bins

    # radius: how long to make each bar. set bar length so either the bar area
    # (ie, via sqrt) or the bar height is proportional to the bin's frequency
    radius = np.sqrt(bin_frequency) if area else bin_frequency
    # create PolarAxes (if not passed-in) then set N at top and go clockwise
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize, bgcolor=None, polar=True)
    ax.set_theta_zero_location("N")  # Set 0 degrees to the right (east)
    ax.set_theta_direction("clockwise")
    
    ax.bar(
        positions,
        height=radius,
        width=width,
        align="center",
        bottom=0,
        zorder=1,
        color=color,
        alpha=1,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.set_ylim(top=radius.max()+(radius.max()*0.1))
    # Set the theta limits to display only from 355 degrees to 175 degrees
    #ax.set_thetamin(0)
    #ax.set_thetamax(175)

    # configure the y-ticks and remove their labels
    yticks = np.linspace(0, 0.5, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels="")

    # Add markers on the y-axis to represent the radius of the bars
    for r in yticks[1:]:  # skip the center (r=0)
        ax.plot(0, r, marker="o", color="k", markersize=4, zorder=10)
        # Add a label slightly offset from the marker
        ax.text(
            np.radians(-8),  # a small angle to the left of the y-axis (adjust as needed)
            r,
            f"{r:.1f}",
            va="center",
            ha="right",
            fontsize=16,
            color="k",
            zorder=11,
        )


    # configure the x-ticks and their labels
    xtick_angles = np.radians(np.arange(0, 361, 10))
    # Create labels for every 10 degrees, with cardinal directions at 0, 90, 180, 270, 360
    xticklabels = []
    for deg in range(0, 361, 10):
        if deg == 0 or deg == 360:
            xticklabels.append("N")
        elif deg == 90:
            xticklabels.append("E")
        elif deg == 180:
            xticklabels.append("S")
        elif deg == 270:
            xticklabels.append("W")
        else:
            xticklabels.append("")
    ax.set_xticks(xtick_angles)
    ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars


    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)
    fig.tight_layout()
    return fig, ax

def plot_alignment_orientation(
    bin_counts,
    *,
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (10, 6),
    area: bool = True,
    color: str = "#d3d3d3",
    edgecolor: str = "k",
    linewidth: float = 0.5,
    alpha: float = 1,
    title: str | None = None,
    title_y: float = 1.05,
    title_font: dict[str] | None = None,
    xtick_font: dict[str] | None = None,
) -> tuple[Figure, PolarAxes]:
    """
    Plot a polar histogram of a spatial network's edge bearings.
    """

    if title_font is None:
        title_font = {"family": "monospace","name":"Courier","size": 24, "weight": "bold"}
    if xtick_font is None:
        xtick_font = {
            "family": "monospace",
            "name":"Courier",
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
   
    radius = np.sqrt(bin_counts) if area else bin_counts

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)  # Set the direction to counter-clockwise
    ax.set_ylim(top=radius.max()+(radius.max()*0.1))  # Add some space above the max radius

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
        alpha=1,
        edgecolor=edgecolor,
        linewidth=linewidth,
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