import osmnx as ox
import weighting_algorithms as wa
import pyproj


def get_azimuth(G, node_a, node_b, return_all=False):
    """Calculate the azimuth and distance between two points.
    
    Pyproj uses an equidistant azimuthal projection with the north pole as the center of a flat circle.
    This means that the azimuth
    ----
    Parameters:
    point_a: The coordinates of the first point.
    point_b: The coordinates of the second point.
    
    ----
    Returns:
    fwd_azimuth: The forward azimuth from point_a to point_b.
    back_azimuth: The back azimuth from point_b to point_a.
    distance: The distance between the two points.
    
    """
    point_a = [G.nodes[node_a]['y'], G.nodes[node_a]['x']]
    point_b = [G.nodes[node_b]['y'], G.nodes[node_b]['x']]
    if ox.projection.is_projected(G.graph['crs']):
        lat1, lat2, long1, long2 = point_a[0], point_b[0], point_a[1], point_b[1]
        transformer = pyproj.Transformer.from_crs(G.graph['crs'], "EPSG:4326", always_xy=True)
        lon1, lat1 = transformer.transform(long1, lat1)
        lon2, lat2 = transformer.transform(long2, lat2)
        geodesic = pyproj.Geod(ellps='WGS84')
        fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
        if return_all:
            return fwd_azimuth, back_azimuth, distance
        else:
            return fwd_azimuth
    else:
        lat1, lat2, long1, long2 = point_a[0], point_b[0], point_a[1], point_b[1]
        geodesic = pyproj.Geod(ellps='WGS84')
        fwd_azimuth, back_azimuth, distance = geodesic.inv(long1, lat1, long2, lat2)
        if return_all:
            return fwd_azimuth, back_azimuth, distance
        else:
            return fwd_azimuth

    
# home location 63.86162247171633, 20.339043153757505
home = (63.86162247171633, 20.339043153757505)

graph = ox.graph.graph_from_point(center_point=home, dist=600,dist_type="bbox", network_type='drive')

#ox.plot_graph(graph)
import matplotlib.pyplot as plt

fig, ax = ox.plot_graph(graph, show=False, close=False,dpi=300,bgcolor='white',edge_color='black',edge_linewidth=0.5,node_size=0,node_color='black')
for node, data in graph.nodes(data=True):
    x, y = data['x'], data['y']
    ax.text(x, y, str(node), fontsize=8, ha='center', va='center', color='blue')
plt.savefig('graph_plot.png')
plt.close()

# test 1
u = 156639763
v = 444493179
w = 156639764
print(get_azimuth(graph, u, v))
print(wa.is_t_turn(graph, u, v))

# test 2
u = 6302228883
v = 2217097983
print(get_azimuth(graph, u, v))
print(wa.is_t_turn(graph, u, v))

# test 3
u = 301123231
v = 156639548
print(get_azimuth(graph, u, v))
print(wa.is_t_turn(graph, u, v))