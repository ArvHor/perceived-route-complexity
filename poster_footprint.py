
import osmnx as ox
from IPython.display import Image
ox.__version__
def get_highres_footprint():
    visby = [57.64056058217987, 18.2961212994695]
    coords = [40.75042885900174, -73.98315714296373]
    endpoint = [40.73573228521026, -73.98532436781716]
    startpoint = [40.75044511461143, -73.99474428575762]
    oslo = [59.920507031897664, 10.742931082961324]
    enkoping = [59.641437415133694, 17.083529369664802]
    #G = ox.graph_from_point(coords, dist=3000, network_type='drive',truncate_by_edge=True)
    #startnode = ox.distance.nearest_nodes(G, startpoint[1], startpoint[0])
    #endnode = ox.distance.nearest_nodes(G, endpoint[1], endpoint[0])

    #route = ox.routing.shortest_path(G, startnode, endnode, weight='length')
    #route_gdf = ox.routing.route_to_gdf(G, route,weight='length')
    #ox.plot.plot_graph_route(G, route, route_linewidth=6, node_size=2, bgcolor='w', show=False, close=False,save=True,filepath='demonstration/route.png',dpi=300)

    
    #fig, ax = ox.plot.plot_figure_ground(G, dist=1000, default_width=3, bgcolor='w',color='black', show=False,close=False)
    #fig.savefig('demonstration/highres_footprint.png', dpi=300)



    img_folder = "demonstration"
    extension = "svg"
    fp = f"./{img_folder}/enkoping.{extension}"
    fp2 = f"./{img_folder}/enkoping.{extension}"
    visby_tags = {"building": True,"barrier": True,"geological":True}
    tags={"building": True,"highway":True}
    gdf = ox.features.features_from_point(center_point=coords,tags=tags,dist=10000)
    gdf_proj = ox.projection.project_gdf(gdf)
    fig, ax = ox.plot.plot_footprints(gdf_proj, filepath=fp, dpi=300, save=True, show=False, close=True, bgcolor='white',color='black',figsize=(3508/30, 2480/30))
    Image(fp2, height=1000, width=1000)

get_highres_footprint()