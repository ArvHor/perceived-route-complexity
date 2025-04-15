import math
import osmnx as ox
import map_plotting as mp
from matplotlib.projections.polar import PolarAxes
import numpy as np

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


def _plot_overlaid_distribution(self,
                                ax: PolarAxes,
                                new_distribution: np.ndarray,
                                num_bins: int,
                                area: bool
                                ) -> None:
    print(new_distribution)
    bin_centers = 360 / num_bins * np.arange(num_bins)
    positions = np.radians(bin_centers)
    width = 2 * np.pi / num_bins

    # Normalize the new distribution to calculate height/area
    new_bin_frequency = new_distribution / new_distribution.sum()
    new_radius = np.sqrt(new_bin_frequency) if area else new_bin_frequency
    print(new_distribution)
    ax.bar(
        positions,
        height=new_radius,
        width=width,
        align="center",
        bottom=0,
        zorder=3,  # Ensure red bars are on top
        color="red",
        edgecolor="k",
        linewidth=0.5,
        alpha=0.5,
    )


