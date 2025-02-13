
import folium
from pyproj import CRS, Transformer, Geod
import pyproj
import geo_utilities
import osmnx as ox
from od_pair import od_pair
from origin_graph import origin_graph
def get_route_bbox(crs, origin_point, destination_point,padding=0.25):

    geod = Geod(crs=crs)
    lon1, lat1 = origin_point
    lon2, lat2 = destination_point
    fwd_azimuth, back_azimuth, distance = geod.inv(lon1, lat1, lon2, lat2)

    extension = distance * padding
    ext_lon1, ext_lat1, _ = geod.fwd(lon1, lat1, back_azimuth, extension)
    ext_lon2, ext_lat2, _ = geod.fwd(lon2, lat2, fwd_azimuth, extension)



def get_azimuth(point_a,point_b, return_all=False):
    lat1, lat2, long1, long2 = point_a[0], point_b[0], point_a[1], point_b[1]
    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth, back_azimuth, distance = geodesic.inv(long1, lat1, long2, lat2)
    if return_all:
        return fwd_azimuth, back_azimuth, distance
    else:
        return fwd_azimuth
    

def get_turn_type(point_a,point_b,_point_c):
    bearing_origin_to_intermediate = get_azimuth(point_a, point_b)
    bearing_intermediate_to_destination = get_azimuth(point_b, _point_c)
    bearing_difference = (bearing_origin_to_intermediate - bearing_intermediate_to_destination)
    print(f"bearing difference {bearing_difference}")
    bearing_difference = bearing_difference % 360 # Normalize to -180 to 180
    print(f"bearing difference {bearing_difference}")
    if bearing_difference > 180:
        bearing_difference -= 360  
    print(f"bearing difference {bearing_difference}")
    if -10 < bearing_difference < 10:
        return 'straight'
    elif 45 <= bearing_difference:
        if 10 < bearing_difference:
            return "slight_right_turn"
        return 'right_turn'
    elif bearing_difference <= -45:
        if bearing_difference < -10:
            return "slight_left_turn"
        return 'left_turn'

#Northeast of MIThuset
ne = 20.3086550,63.8206501

#Northwest of MIThuset
nw = 20.3068467,63.8207038

#Southwest of MIThuset
sw = 20.3068858,63.8201906

#Southeast of MIThuset
se = 20.3086355,63.8201866

#center of MIThuset
center = 20.3077636,63.8204552

#center of my home
home = 20.3370856,63.8627840

print("Azimuths:")
print("center to ne:")
print(get_azimuth(center,ne)) #45

print("center to nw:")
print(get_azimuth(center,nw)) #-45

print("center to sw:")
print(get_azimuth(center,sw)) #-135

print("center to se:")
print(get_azimuth(center,se)) #135



turn_type = get_turn_type(se,center,ne)
print("Turn type from se to center to ne:")
print(turn_type) #right_turn

turn_type = get_turn_type(ne,center,se)
print("Turn type from ne to center to se:")
print(turn_type) #left_turn


turn_type = get_turn_type(se,center,nw)
print("Turn type from se to center to nw:")
print(turn_type) #left_turn

shape_dict = geo_utilities.get_od_pair_polygon((home[1],home[0]),(center[1],center[0]))

poly = shape_dict["polygon"]
bbox =shape_dict["bbox"]
#print(poly)
#print(bbox)
m = folium.Map(location=[63.8204552,20.3077636], zoom_start=15, tiles="CartoDB positron")

# Add the bounding box to the map
bbox_coords = [
    [bbox[1], bbox[0]],  # bottom-left
    [bbox[1], bbox[2]],  # bottom-right
    [bbox[3], bbox[2]],  # top-right
    [bbox[3], bbox[0]],  # top-left
    [bbox[1], bbox[0]]   # close the polygon
]
folium.PolyLine(locations=bbox_coords, color='red').add_to(m)
# Add the polygon to the map
folium.GeoJson(poly).add_to(m)
# Add the se and nw points to the map
folium.Marker(location=[center[1], center[0]], popup='work').add_to(m)
folium.Marker(location=[home[1], home[0]], popup='home').add_to(m)
folium.PolyLine(locations=[(center[1], center[0]), (home[1], home[0])], color='blue').add_to(m)
# Save the map to an HTML file
m.save('/home/arvidh/Documents/GitHub/routing_graph/azimuth_test_map.html')


#o_graph = origin_graph(origin_point=[home[1],home[0]],origin_info = {"city_name":"Ume","station_name":"Home","region_name":"Ume","country_name":"Sweden","network_type":"drive"},network_type="drive",remove_parallel=True,distance_from_point=5000)
#o_graph.add_weights(weightstring="decision_complexity")
#o_graph.remove_infinite_edges()
#o_graph.save_graph("umetest/ume_graph.graphml")

o_graph = origin_graph(origin_point=[home[1],home[0]],origin_info = {"city_name":"Ume","station_name":"work","region_name":"Ume","country_name":"Sweden","network_type":"drive"},
                       network_type="drive",remove_parallel=True,distance_from_point=5000,load_graphml="umetest/ume_graph.graphml")

home_node = ox.distance.nearest_nodes(o_graph.graph, home[0], home[1])
work_node = ox.distance.nearest_nodes(o_graph.graph, center[0], center[1])

print(o_graph.graph.nodes[home_node])


#ox.plot_graph(o_graph.graph)
home_work_pair = od_pair(o_graph.graph, home_node, work_node)

print(home_work_pair.get_EMD_alignment())




#manhattan aligned
manhattan_aligned_A = -73.967053,40.754002
manhattan_aligned_B = -73.996057,40.762327

#m1_o_graph = origin_graph(origin_point=[manhattan_aligned_A[1],manhattan_aligned_A[0]],
#                          origin_info = {"city_name":"manhattan1","station_name":"Home","region_name":"manhattan1","country_name":"Sweden","network_type":"drive"},network_type="drive",remove_parallel=True,distance_from_point=5000)
#m1_o_graph.add_weights(weightstring="decision_complexity")
#m1_o_graph.remove_infinite_edges()
#m1_o_graph.save_graph("umetest/manhattan1_graph.graphml")

m1_o_graph = origin_graph(origin_point=[manhattan_aligned_A[1],manhattan_aligned_A[0]],
                          origin_info = {"city_name":"manhattan1","station_name":"Home","region_name":"manhattan1","country_name":"Sweden","network_type":"drive"},network_type="drive",remove_parallel=True,distance_from_point=5000,load_graphml="umetest/manhattan1_graph.graphml")



manhattan_aligned_A_node = ox.distance.nearest_nodes(m1_o_graph.graph, manhattan_aligned_A[0], manhattan_aligned_A[1])
manhattan_aligned_B_node = ox.distance.nearest_nodes(m1_o_graph.graph, manhattan_aligned_B[0], manhattan_aligned_B[1])

manhattan_aligned_pair = od_pair(m1_o_graph.graph, manhattan_aligned_A_node, manhattan_aligned_B_node)

print("VERY IMPORTANT MANHATTAN ALIGNED:")
print(manhattan_aligned_pair.get_EMD_alignment())
_,lag = manhattan_aligned_pair.get_crosscorrelation_alignment()
print(lag)


#manhattan not aligned
manhattan_not_aligned_A = -73.978645,40.746824
manhattan_not_aligned_B = -73.985259,40.769613

#m2_o_graph = origin_graph(origin_point=[manhattan_not_aligned_A[1],manhattan_not_aligned_A[0]],
#                      origin_info = {"city_name":"manhattan2","station_name":"Home","region_name":"manhattan2","country_name":"Sweden","network_type":"drive"},network_type="drive",remove_parallel=True,distance_from_point=5000)
#o_graph.add_weights(weightstring="decision_complexity")
#o_graph.remove_infinite_edges()
#o_graph.save_graph("umetest/manhattan2_graph.graphml")

m2_o_graph = origin_graph(origin_point=[manhattan_not_aligned_A[1],manhattan_not_aligned_A[0]],
                          origin_info = {"city_name":"manhattan1","station_name":"Home","region_name":"manhattan1","country_name":"Sweden","network_type":"drive"},
                          network_type="drive",remove_parallel=True,distance_from_point=5000,load_graphml="umetest/manhattan2_graph.graphml")




manhattan_not_aligned_A_node = ox.distance.nearest_nodes(m2_o_graph.graph, manhattan_not_aligned_A[0], manhattan_not_aligned_A[1])
manhattan_not_aligned_B_node = ox.distance.nearest_nodes(m2_o_graph.graph, manhattan_not_aligned_B[0], manhattan_not_aligned_B[1])

manhattan_not_aligned_pair = od_pair(m2_o_graph.graph, manhattan_not_aligned_A_node, manhattan_not_aligned_B_node)

print("VERY IMPORTANT MANHATTAN NOT ALIGNED:")
print(manhattan_not_aligned_pair.get_EMD_alignment())
_,lag = manhattan_not_aligned_pair.get_crosscorrelation_alignment()
print(lag)
