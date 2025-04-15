import pyproj
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import osmnx as ox
from shapely.ops import transform
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
import numpy as np

def find_utm_zone(lat, lon):
    """
    Find the UTM zone for a given latitude and longitude.
    
    Input:
    - lat: latitude
    - lon: longitude
    """
    wgs84 = pyproj.crs.from_epsg(4326)
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    utm_crs = pyproj.crs.from_epsg(utm_crs_list[0].code)

    wgs84_to_utm = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    utm_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    center_x_utm, center_y_utm = wgs84_to_utm.transform(lon, lat)
    return utm_crs, (center_x_utm, center_y_utm)

def get_od_pair_polygon(origin_point, destination_point,padding=0.25):
    geod = pyproj.Geod(ellps='WGS84')

    lat1,lon1 = origin_point
    lat2,lon2 = destination_point
    fwd_azimuth, bwd_azimuth, distance = geod.inv(lon1, lat1, lon2, lat2)

    extension = distance * padding
    ext_lon1, ext_lat1, _ = geod.fwd(lon1, lat1, bwd_azimuth, extension)
    ext_lon2, ext_lat2, _ = geod.fwd(lon2, lat2, fwd_azimuth, extension)

    ext_point1 = (ext_lon1,ext_lat1)
    ext_point2 = (ext_lon2, ext_lat2)

    _, _, extended_distance = geod.inv(ext_lon1, ext_lat1, ext_lon2, ext_lat2)

    intermediate_points = geod.npts(lon1, lat1, lon2, lat2, 1)
    #print(intermediate_points[0])
    mid_lon, mid_lat = intermediate_points[0]
    perp_fwd = fwd_azimuth + 90
    perp_bwd = fwd_azimuth - 90
    perp_distance = extended_distance / 2
    perp_lon1, perp_lat1, _ = geod.fwd(mid_lon, mid_lat, perp_fwd, perp_distance)
    perp_lon2, perp_lat2, _ = geod.fwd(mid_lon, mid_lat, perp_bwd, perp_distance)
    perp_point1 = (perp_lon1, perp_lat1)
    perp_point2 = (perp_lon2, perp_lat2)
    
    #print(perp_point1)
    #print(perp_point2)
    polygon = Polygon([ext_point1, perp_point1, ext_point2, perp_point2])
    polygon = orient(polygon)
    bbox = polygon.bounds

    bbox_coords = [
    [bbox[1], bbox[0]],  # bottom-left
    [bbox[1], bbox[2]],  # bottom-right
    [bbox[3], bbox[2]],  # top-right
    [bbox[3], bbox[0]],  # top-left
    [bbox[1], bbox[0]]   # close the polygon
    ]

    minx, miny, maxx, maxy = bbox
    bbox_tuple = (miny, maxy, minx, maxx)

    bbox_polygon = Polygon(bbox_coords)


    fwd_bearing = ox.bearing.calculate_bearing(lat1, lon1, lat2, lon2)
    bwd_bearing = ox.bearing.calculate_bearing(lat2, lon2, lat1, lon1)
    print(f"fwd: {fwd_bearing}")
    print(f"bwd: {bwd_bearing}")
    perpendicular_fwd_bearing = ox.bearing.calculate_bearing(perp_point1[1], perp_point1[0], perp_point2[1], perp_point2[0])
    perpendicular_bwd_bearing = ox.bearing.calculate_bearing(perp_point2[1], perp_point2[0], perp_point1[1], perp_point1[0])

    bearings = [fwd_bearing, bwd_bearing, perpendicular_fwd_bearing, perpendicular_bwd_bearing]
    bearings = [b if b >= 0 else b + 360 for b in bearings]
    shape_dict = {
        "fwd_bearing": bearings[0],
        "bwd_bearing": bearings[1],
        "perpendicular_fwd_bearing": bearings[2],
        "perpendicular_bwd_bearing": bearings[3],
        "polygon": polygon,
        "bbox" : bbox,
        "bbox_polygon": bbox_polygon,
        "osmnx_bbox": bbox_tuple
    }

    print(f"fwd: {shape_dict['fwd_bearing']}")
    print(f"bwd: {shape_dict['bwd_bearing']}")
    print(f"perp fwd: {shape_dict['perpendicular_fwd_bearing']}")
    print(f"perp bwd: {shape_dict['perpendicular_bwd_bearing']}")
    return shape_dict


def calculate_area_with_utm(polygon):
    """Calculates the area of a polygon using the appropriate UTM projection.

    Args:
        polygon: A Shapely Polygon object with coordinates in latitude and longitude (EPSG:4326).

    Returns:
        The area of the polygon in square meters.
    """
    # Calculate the centroid of the polygon
    centroid = polygon.centroid

    # Determine the UTM zone based on the centroid's longitude
    utm_zone = int((centroid.x + 180) / 6) + 1

    # Define the source (geographic) and target (projected) coordinate systems
    geographic_crs = pyproj.CRS("EPSG:4326")  # WGS 84
    projected_crs = pyproj.CRS(f"EPSG:326{utm_zone}")  # UTM Zone N (North)
    if centroid.y < 0:
        projected_crs = pyproj.CRS(f"EPSG:327{utm_zone}")  # UTM Zone S (South)

    # Create a transformer to convert between the coordinate systems
    project = pyproj.Transformer.from_crs(geographic_crs, projected_crs, always_xy=True).transform

    # Project the polygon to the UTM coordinate system
    projected_polygon = transform(project, polygon)

    # Calculate the area of the projected polygon in square meters
    return projected_polygon.area


def get_azimuth(G, point_a, point_b, return_all=False):
    """Calculate the azimuth and distance between two points.

    Pyproj uses an equidistant azimuthal projection with the north pole as the center of a flat circle.
    This means that the azimuth
    ----
    Parameters:
    point_a: The coordinates of the first point.
    point_b: The coordinates of the second point.
    return_all: If True, return the forward azimuth, back azimuth, and distance. If False, return only the forward azimuth.

    ----
    Returns:
    fwd_azimuth: The forward azimuth from point_a to point_b.
    back_azimuth: The back azimuth from point_b to point_a.
    distance: The distance between the two points.

    """
    if isinstance(point_a,int) and isinstance(point_b,int):
        point_a = (G.nodes[point_a]['y'], G.nodes[point_a]['x'])
        point_b = (G.nodes[point_a]['y'], G.nodes[point_a]['x'])

    if not isinstance(point_a,tuple) and not isinstance(point_b,tuple):
        msg = "point_a and point_b must either both be node_ids or coordinates."
        raise ValueError(msg)

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


def merge_and_simplify_geometry(geometry,tolerance):
    # Merge the MultiLineString into a single LineString
    line = linemerge(geometry)
    # Simplify the merged LineString
    simplified = line.simplify(tolerance, preserve_topology=True)

    return simplified

def perpendicular_distance(point, line_start, line_end):

    p = Point(point)
    line = LineString([line_start, line_end])

    nearest_line_point = line.interpolate(line.project(p))

    distance = ox.distance.great_circle(lat1=point[1], lon1=point[0], lat2=nearest_line_point.y, lon2=nearest_line_point.x)

    return distance

def douglas_peucker(route_linestring, thold):
    dmax = 0.0
    index = 0
    first,last = route_linestring[0],route_linestring[-1]

    for i in range(1, last):
        d = perpendicular_distance(route_linestring[i - 1], route_linestring[i], route_linestring[end])
        if d > dmax:
            index = i
            dmax = d

    results = []

    if (dmax >= thold):
        recResults1 = douglas_peucker(route_linestring[:index + 1],thold)
        recResults2 = douglas_peucker(route_linestring[index:],thold)

        results = recResults1[:-1] + recResults2
    else:
        results = route_linestring[:index + 1]

    return results

def get_bearing_difference(G, origin, intermediate, destination):
    if isinstance(origin, int) and isinstance(destination, int) and isinstance(intermediate, int):
        origin = [G.nodes[origin]['y'], G.nodes[origin]['x']]
        destination = [G.nodes[destination]['y'], G.nodes[destination]['x']]
        intermediate = [G.nodes[intermediate]['y'], G.nodes[intermediate]['x']]

    bearing_origin_to_intermediate = get_azimuth(G, origin, intermediate)
    bearing_intermediate_to_destination = get_azimuth(G, intermediate, destination)

    bearing_difference = (bearing_intermediate_to_destination - bearing_origin_to_intermediate)

    bearing_difference = bearing_difference % 360

    if bearing_difference > 180:
        bearing_difference -= 360
    elif bearing_difference < -180:
        bearing_difference += 360
    return bearing_difference