import osmnx as ox
import networkx as nx
from pyproj import CRS

import geo_util as geo_util
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

def get_map_clutter(G,map_bbox):
    undirected_G = ox.convert.to_undirected(G)
    undirected_G = ox.truncate.truncate_graph_bbox(undirected_G,map_bbox)
    length = sum(data[2]["length"] for data in undirected_G.edges(data=True))
    intersection_count = len(undirected_G.nodes())
    street_count = len(undirected_G.edges())

    return length, intersection_count, street_count


def get_routegdf_bbox(G, route_nodes, buffer_percentage=0.1):

    # Convert route to GeoDataFrame
    route_gdf = ox.routing.route_to_gdf(G, route_nodes)

    # Get the route's geometry
    if len(route_gdf) > 1:
        # Combine all geometries in the route
        geom = route_gdf['geometry'].unary_union
    else:
        geom = route_gdf['geometry'].iloc[0]

    # Get the bounds of the geometry (minx, miny, maxx, maxy)
    bounds = geom.bounds

    # Calculate the dimensions of the bounds
    width = bounds[2] - bounds[0]  # east - west
    height = bounds[3] - bounds[1]  # north - south

    # Add buffer around the bounds
    buffer_x = width * buffer_percentage
    buffer_y = height * buffer_percentage

    # Create the final bounding box (north, south, east, west)
    bbox = (
        bounds[3] + buffer_y,  # north
        bounds[1] - buffer_y,  # south
        bounds[2] + buffer_x,  # east
        bounds[0] - buffer_x  # west
    )

    return bbox


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
