import os
import folium.vector_layers
import matplotlib as plt
from matplotlib.colors import Normalize
import osmnx as ox
import time
import pandas as pd
import pyproj
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
import selenium.webdriver as webdriver
from selenium.webdriver import Firefox, FirefoxOptions
from folium.features import DivIcon
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from folium.elements import *
import folium
import math

def calculate_bounding_box(center_lat, center_lng, zoom=16, width_pixels=1600, height_pixels=1200):
    """
    Calculates the bounding box coordinates for a Leaflet map with UTM projection.
    https://wiki.openstreetmap.org/wiki/Zoom_levels
    Args:
      center_lat: Latitude of the map's center point.
      center_lng: Longitude of the map's center point.
      zoom: Zoom level of the map (default 16).
      width_pixels: Width of the map in pixels (default 1600).
      height_pixels: Height of the map in pixels (default 1200).

    Returns:
      A tuple containing the (north, south, east, west) coordinates of the 
      bounding box in WGS84 (latitude, longitude).
    """

    meters_per_pixel = 156543.03 * math.cos(math.radians(center_lat)) / (2 ** zoom)
    
    width_meters = width_pixels * meters_per_pixel
    height_meters = height_pixels * meters_per_pixel
    
    
    # --- UTM Projection ---
    wgs84 = CRS.from_epsg(4326)
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=center_lng,
            south_lat_degree=center_lat,
            east_lon_degree=center_lng,
            north_lat_degree=center_lat,
        ),
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)

    wgs84_to_utm = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    utm_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    center_x_utm, center_y_utm = wgs84_to_utm.transform(center_lng, center_lat)
    # --- End UTM Projection ---

    # Calculate corner coordinates in UTM
    north_utm = center_y_utm + (height_meters / 2)
    south_utm = center_y_utm - (height_meters / 2)
    east_utm = center_x_utm + (width_meters / 2)
    west_utm = center_x_utm - (width_meters / 2)

    # Convert UTM corners back to WGS84 (lat, lng)
    north_lng, north_lat = utm_to_wgs84.transform(east_utm, north_utm)
    south_lng, south_lat = utm_to_wgs84.transform(east_utm, south_utm)  
    east_lng, east_lat = utm_to_wgs84.transform(east_utm, north_utm)  
    west_lng, west_lat = utm_to_wgs84.transform(west_utm, north_utm)  
    
    bbox = (west_lng,south_lat,east_lng,north_lat)

    return bbox

def plot_route_gdf(G, route_gdf,start_node,end_node,info_text="null",imgpath="route_on_map.png",file_path="route_on_map.png",map_tiles="CartoDB.VoyagerNoLabels",return_bbox=False, flip=False):
    #print(map_tiles)
    #apikey = '54NexSXPLjyL0FsLdsoy'
    geom = route_gdf['geometry'].unary_union
    route_gdf['geometry'] = merge_and_simplify_geometry(geom, 0.0001)
    start_location = (G.nodes[start_node]['y'], G.nodes[start_node]['x'])
    end_location = (G.nodes[end_node]['y'], G.nodes[end_node]['x'])
    
    midpoint = ((start_location[0] + end_location[0]) / 2, (start_location[1] + end_location[1]) / 2)

    #m = route_gdf.explore(tiles="CartoDB.VoyagerNoLabels",color='blue', control_scale=False,location=midpoint, style_kwds={"weight": 5,"opacity":1},width="2100px",height='1400px',zoom_snap=0.25,zoom_start=16,set_zoom=16,legend=False,zoom_control=False)

    m = route_gdf.explore(tiles=map_tiles,color='blue',
                        control_scale=False,
                        zoom_control=False,
                        location=midpoint,style_kwds={
                            "weight": 7,"opacity":0.7,
                            'dashArray':'1,20'},
                        height="100%",
                        zoom_start=16,
                        min_zoom=16,
                        max_zoom=16,
                        legend=False,
                        attr=None)
    
    
    
    midpoint = ((start_location[0] + end_location[0]) / 2, (start_location[1] + end_location[1]) / 2)
    bbox = calculate_bounding_box(midpoint[0], midpoint[1], width_pixels=1600, height_pixels=1200, zoom=16)
    if info_text != "null":
            folium.map.Marker(
            [midpoint[0], midpoint[1]],
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 10pt">{info_text}</div>',
                )
            ).add_to(m)

    if flip==True:
        m = flip_map(m,end_location,start_location)
    else:
        folium.Marker(
            location=start_location,
            icon=folium.Icon(color='green', icon='fa-map-marker', prefix='fa-solid'),  # green map pin icon without dot
        ).add_to(m)

        # Add destination marker (end location)
        folium.Marker(
            location=end_location,
            icon=folium.Icon(color='black', icon='fa-flag-checkered', prefix='fa'),  # red map pin icon with dot
        ).add_to(m)
    m.save(file_path)
    full_path = os.path.abspath(file_path)	
    screenshot_map(full_path, imgpath)
    if return_bbox:
        return bbox
    

def get_routegdf_bbox(G, route_nodes, start_node,end_node):
    route_gdf = ox.routing.route_to_gdf(G,route_nodes)
    geom = route_gdf['geometry'].unary_union
    route_gdf['geometry'] = merge_and_simplify_geometry(geom, 0.0001)
    start_location = (G.nodes[start_node]['y'], G.nodes[start_node]['x'])
    end_location = (G.nodes[end_node]['y'], G.nodes[end_node]['x'])
    
    midpoint = ((start_location[0] + end_location[0]) / 2, (start_location[1] + end_location[1]) / 2)
    bbox = calculate_bounding_box(midpoint[0], midpoint[1], width_pixels=1600, height_pixels=1200, zoom=16)

    return bbox

def flip_map(m,end_location,start_location):
    
    css = None
    end_icon_html = """
    <div class="awesome-marker-icon-black awesome-marker leaflet-zoom-animated leaflet-interactive" tabindex="0" role="button" 
    style="width: 35px; height: 45px; transform: scale(-1, -1);z-index: 681; outline: none;">
        <i class="fa-rotate-0 fa fa-flag-checkered  icon-white"></i>
    </div>
    """
    
    folium.Marker(
        location=end_location,
        icon=DivIcon(
            icon_size=(60, 60),
            icon_anchor=(-15, -40),  # Center of the icon
            html=end_icon_html
        )
    ).add_to(m)
    start_icon_html = """
    <div class="awesome-marker-icon-green awesome-marker leaflet-zoom-animated leaflet-interactive" tabindex="0" role="button" 
    style="width: 35px; height: 45px; transform: scale(-1, -1);z-index: 704; outline: none;">
        
    </div>
    """
    
    folium.Marker(
        location=start_location,
        icon=DivIcon(
            icon_size=(60, 60),
            icon_anchor=(-15, -40),  # Center of the icon
            html=start_icon_html
        )
    ).add_to(m)
    css = """
            <style>
                .folium-map {
                    transform: scale(-1, -1);  /* This flips both horizontally and vertically */
                    transform-origin: center center;
                }
                /* Hide the attribution/credits */
                .leaflet-control-attribution {
                    display: none !important;
                }
            </style>
            """
    m.get_root().header.add_child(folium.Element(css))
    return m

def add_rotation_to_map(m, rotation_angle):
    # Add extra padding to hide white background during rotation
    css = f"""
            <style>
                .folium-map {{
                    transform: rotate({rotation_angle}deg) scale(1.5);  /* Scale up by 50% */
                    transform-origin: center center;
                }}
                /* Ensure the container fills the viewport */
                #map {{
                    width: 100vw !important;
                    height: 100vh !important;
                    position: fixed !important;
                    top: 0;
                    left: 0;
                    margin: 0;
                    padding: 0;
                }}
            </style>
            """
    m.get_root().header.add_child(folium.Element(css))
    
    # Add JavaScript to adjust the map container size to prevent cutoff
    js = """
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                var map = document.querySelector('.folium-map');
                var container = map.parentElement;
                var angle = Math.abs(%s);
                
                // Calculate new dimensions based on rotation with extra padding
                var rad = angle * Math.PI / 180;
                var sin = Math.abs(Math.sin(rad));
                var cos = Math.abs(Math.cos(rad));
                var originalWidth = window.innerWidth * 1.5;  // 50%% larger than viewport
                var originalHeight = window.innerHeight * 1.5;
                
                // New dimensions to ensure no content is cut off
                var newWidth = Math.ceil(originalWidth * cos + originalHeight * sin);
                var newHeight = Math.ceil(originalHeight * cos + originalWidth * sin);
                
                // Set new dimensions and center the map
                container.style.width = newWidth + 'px';
                container.style.height = newHeight + 'px';
                container.style.overflow = 'hidden';
                map.style.position = 'absolute';
                map.style.width = originalWidth + 'px';
                map.style.height = originalHeight + 'px';
                map.style.left = ((newWidth - originalWidth) / 2) + 'px';
                map.style.top = ((newHeight - originalHeight) / 2) + 'px';
                
                // Force map to redraw at new size
                window.dispatchEvent(new Event('resize'));
                
                // Add background color to container
                document.body.style.backgroundColor = '#000';  // or any color that matches your map background
                document.body.style.margin = '0';
                document.body.style.overflow = 'hidden';
            });
            </script>
            """ % rotation_angle
    
    m.get_root().html.add_child(folium.Element(js))
    return m

    
def merge_and_simplify_geometry(geometry,tolerance):
    # Merge the MultiLineString into a single LineString
    line = linemerge(geometry)
    # Simplify the merged LineString
    simplified = line.simplify(tolerance, preserve_topology=True)

    return simplified  


def plot_all_routes_complexity(G, routes, map_path,startnode):
    """Plot all routes in a single map"""
    startnode = G.nodes[startnode]
    route_gdfs = []
    origin = (startnode['y'], startnode['x'])

    routes_df = pd.DataFrame(routes)
    #routes['nodes'] = routes['nodes'].apply(ast.literal_eval)
    route_nodes = routes['nodes']

    m = folium.Map(location=origin, zoom_start=14.4, tiles="CartoDB.VoyagerNoLabels")


    complexity_values = routes['sum_decision_complexity'].values
    norm = Normalize(vmin=min(complexity_values), vmax=max(complexity_values))
    cmap = plt.cm.get_cmap('plasma')
    #print(f'routes: {routes_df}')


    for index, route in routes_df.iterrows():
        route_nodes = route['nodes']

        #print(f'route nodes: {route_nodes}')
        complexity_weightsum = float(route['sum_decision_complexity'])
        route_gdf = ox.routing.route_to_gdf(G, route_nodes, weight='length')
        geom = route_gdf['geometry'].unary_union
        #route_gdf['geometry'] = merge_and_simplify_geometry(geom, 0.0001)
        route_gdf['sum_decision_complexity'] = complexity_weightsum
        route_gdfs.append(route_gdf)

    for route_gdf in route_gdfs:
        
        color = cmap(norm(route_gdf['sum_decision_complexity'].values[0]))
        
        color = plt.colors.to_hex(color)
        
        route_linestring = route_gdf['geometry'].unary_union

        if route_linestring.geom_type == 'LineString':
            route_linestring = merge_and_simplify_geometry(route_linestring,0.000001)
            route_linestring = [[coord[1], coord[0]] for coord in list(route_linestring.coords)]
        elif route_linestring.geom_type == 'MultiLineString':
            route_linestring = merge_and_simplify_geometry(route_gdf.geometry.explode().unary_union,0.000001)
            if route_linestring.geom_type == 'LineString':
                route_linestring = [[coord[1], coord[0]] for coord in list(route_linestring.coords)]
            else:
                continue
            
       

        folium.PolyLine(route_linestring, color=color, weight=4, opacity=0.3).add_to(m)

    m.save(map_path)
    map_path = os.path.abspath(map_path)
    imgpath = map_path.replace('.html','.png')
    screenshot_map(map_path, imgpath)
    #m.save(f'maps/interactive maps/{name}.html')   


def plot_all_routes_complexity(routes, map_path):
    """Plot all routes in a single map"""
    route_gdfs = []

    routes_df = pd.DataFrame(routes)

    m = folium.Map(tiles="OpenStreetMap.Mapnik")


    complexity_values = routes['sum_decision_complexity'].values
    norm = Normalize(vmin=min(complexity_values), vmax=max(complexity_values))
    cmap = plt.cm.get_cmap('plasma')
    #print(f'routes: {routes_df}')


    for index, route in routes_df.iterrows():
        route_nodes = route['nodes']

        #print(f'route nodes: {route_nodes}')
        complexity_weightsum = float(route['sum_decision_complexity'])
        route_gdf = ox.routing.route_to_gdf(G, route_nodes, weight='length')
        geom = route_gdf['geometry'].unary_union
        #route_gdf['geometry'] = merge_and_simplify_geometry(geom, 0.0001)
        route_gdf['sum_decision_complexity'] = complexity_weightsum
        route_gdfs.append(route_gdf)

    for route_gdf in route_gdfs:
        
        color = cmap(norm(route_gdf['sum_decision_complexity'].values[0]))
        
        color = plt.colors.to_hex(color)
        
        route_linestring = route_gdf['geometry'].unary_union

        if route_linestring.geom_type == 'LineString':
            route_linestring = merge_and_simplify_geometry(route_linestring,0.000001)
            route_linestring = [[coord[1], coord[0]] for coord in list(route_linestring.coords)]
        elif route_linestring.geom_type == 'MultiLineString':
            route_linestring = merge_and_simplify_geometry(route_gdf.geometry.explode().unary_union,0.000001)
            if route_linestring.geom_type == 'LineString':
                route_linestring = [[coord[1], coord[0]] for coord in list(route_linestring.coords)]
            else:
                continue
            
       

        folium.PolyLine(route_linestring, color=color, weight=4, opacity=0.3).add_to(m)
    m.fit_bounds()
    m.save(map_path)
    map_path = os.path.abspath(map_path)
    imgpath = map_path.replace('.html','.png')
    screenshot_map(map_path, imgpath)
    #m.save(f'maps/interactive maps/{name}.html')   

def plot_all_routes(route_gdfs, map_path,point_list):
    """Plot all routes in a single map"""

    m = folium.Map(tiles="OpenStreetMap.Mapnik")

    all_bounds = [] 
    for route_gdf in route_gdfs:
        
        color = "blue"
        color = plt.colors.to_hex(color)
        
        route_linestring = route_gdf['geometry'].unary_union

        if route_linestring.geom_type == 'LineString':
            route_linestring = merge_and_simplify_geometry(route_linestring,0.000001)
            route_linestring = [[coord[1], coord[0]] for coord in list(route_linestring.coords)]
        elif route_linestring.geom_type == 'MultiLineString':
            route_linestring = merge_and_simplify_geometry(route_gdf.geometry.explode().unary_union,0.000001)
            if route_linestring.geom_type == 'LineString':
                route_linestring = [[coord[1], coord[0]] for coord in list(route_linestring.coords)]
            else:
                continue
        # Calculate bounds for the current route and add them to the list
        bounds = route_gdf.total_bounds  # Get bounds as [minx, miny, maxx, maxy]
        
        # Convert geographic coordinates to Folium's expected format (lat, lon)
        all_bounds.append([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        folium.PolyLine(route_linestring, color=color, weight=4, opacity=0.3).add_to(m)
    for point in point_list:
        
            folium.Marker(
                location=point,
                icon=folium.Icon(color='green', icon='fa-map-marker', prefix='fa-solid'),  # green map pin icon without dot
            ).add_to(m)
    # Calculate the overall bounds to fit all routes
    if all_bounds:
        # Find the minimum and maximum coordinates for latitude and longitude
        min_lat = min(b[0][0] for b in all_bounds)
        min_lon = min(b[0][1] for b in all_bounds)
        max_lat = max(b[1][0] for b in all_bounds)
        max_lon = max(b[1][1] for b in all_bounds)

        overall_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        m.fit_bounds(overall_bounds)
    else:
        # Handle the case where no valid bounds were found
        print("Warning: No valid bounds found to fit the map.")
        # Optionally, set a default view or leave the map as is
       

        folium.PolyLine(route_linestring, color=color, weight=4, opacity=0.3).add_to(m)

    m.save(map_path)
    map_path = os.path.abspath(map_path)
    imgpath = map_path.replace('.html','.png')
    screenshot_map(map_path, imgpath)


def add_padding_to_bbox(bbox, padding_meters):
    # Create a geodetic transformer
    max_y, min_y, max_x, min_x = bbox

    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=-min_x,
            south_lat_degree=min_y,
            east_lon_degree=-max_x,
            north_lat_degree=max_y,
        ),
    )

    utm_crs = CRS.from_epsg(utm_crs_list[0].code)

    transformer = pyproj.Transformer.from_crs(crs_from="EPSG:4326", crs_to=utm_crs, always_xy=True)

    # Convert to Web Mercator
    min_x, min_y = transformer.transform(min_x, min_y)
    max_x, max_y = transformer.transform(max_x, max_y)
    print(f"min_x: {min_x}")
    # Add padding
    min_x += padding_meters+200
    min_y += padding_meters
    max_x -= padding_meters+200
    max_y -= padding_meters

    # Convert back to WGS84
    inverse_transformer = pyproj.Transformer.from_crs(crs_from=utm_crs, crs_to="EPSG:4326", always_xy=True)
    min_lon, min_lat = inverse_transformer.transform(min_x, min_y)
    max_lon, max_lat = inverse_transformer.transform(max_x, max_y)
    print(f"Padding added to bbox")
    return max_lat, min_lat, max_lon, min_lon




def get_route_bearing(route_gdf):
    origin = route_gdf['geometry'].iloc[0].coords[0]
    destination = route_gdf['geometry'].iloc[-1].coords[-1]
    bearing = ox.bearing.calculate_bearing(origin[1],origin[0], destination[1], destination[0])
    return bearing

def screenshot_map(full_path,imgpath):
    opts = webdriver.FirefoxOptions()
    opts.add_argument("--width=1600")
    opts.add_argument("--height=1286")
    opts.add_argument("--headless")
    opts.add_argument("--window-size=1600,1286")
    driver = Firefox(options=opts) 
    driver.set_page_load_timeout(60)
    
    try:
        #driver.set_window_size(1600, 1200)
        driver.get(f'file://{full_path}')	
        driver.set_window_size(1600, 1286)
        time.sleep(2)
        driver.save_screenshot(imgpath)
    except Exception as e:
        print("Error loading map")
        print(e)
    
    
    driver.quit()