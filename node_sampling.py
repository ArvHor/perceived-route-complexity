import ast
import time
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.distance import geodesic
import overpy
import osmnx as ox
import random
# Initialize geolocator


api = overpy.Overpass()

def get_train_station_coordinates(city):
  """
  Retrieves the coordinates of a train station in a given city using Overpass API.

  Args:
    city: The name of the city.

  Returns:
    A tuple containing the latitude and longitude of the train station as floats, 
    or None if no train station is found.
  """
  geolocator = Nominatim(user_agent="train_station_finder")
  api = overpy.Overpass()
  try:
    location = geolocator.geocode(city)
    if not location:
      print(f"Could not find the city: {city}")
      return None
    city_center_coords = (location.latitude, location.longitude)
    lat, lon = location.latitude, location.longitude
    result = api.query(f"""
      [out:json];
      node(around:10000,{lat},{lon})["railway"="station"];
      out center;
    """)
    if result.nodes:
      closest_station = min(result.nodes,key=lambda node: geodesic(city_center_coords, (float(node.lat), float(node.lon))).km)
      # Get the first train station found
      train_station_name = closest_station.tags.get("name:en")
      if not train_station_name:
        train_station_name = closest_station.tags.get("name", "Unknown")

      city_location = geolocator.reverse((lat, lon), exactly_one=True)
      address = city_location.raw.get('address', {})
      city_country = address.get('country', 'Unknown')
      #city_continent = address.get('continent', 'Unknown')
      #print(f"Train station found: {train_station_name}")
      return float(closest_station.lat), float(closest_station.lon), train_station_name, city_country,#city_continent  # Explicitly convert to floats
    else:
      print(f"No train station found in {city}")
      return None
  except GeocoderTimedOut:
    print(f"Geocoding timed out for city: {city}")
    return None
  except overpy.exception.OverpassTooManyRequests:
    print(f"Too many requests to Overpass API. Try again later.")
    return None
  except overpy.exception.OverpassGatewayTimeout:
    print(f"Timeout error occurred while querying Overpass API for {city}")
    return None

def get_coord_info(lat,lon):
  geolocator = Nominatim(user_agent="train_station_finder")
  try:
    location = geolocator.reverse((lat, lon), exactly_one=True)
    address = location.raw.get('address', {})
    city = address.get('city', '')
    country = address.get('country', '')
    return city, country
  except GeocoderTimedOut:
    print(f"Geocoding timed out for coordinates: {lat}, {lon}")
    return None, None

def create_train_station_csv(origin_locations):
  new_locations = []

  for index, row in origin_locations.iterrows():
    city_name = row['city_name']
    train_station_coords = get_train_station_coordinates(city_name)
    if train_station_coords:
      location = {
        'country': train_station_coords[3],
        'city_name': city_name,
        'train_station_name': train_station_coords[2],
        'train_station_lat': train_station_coords[0],
        'train_station_lon': train_station_coords[1]
      }
      new_locations.append(location)
      print(f"Train station coordinates for {city_name}: {train_station_coords}")
    time.sleep(1)  # Sleep for 1 second to avoid making too many requests in a short period of time

  # Save the new DataFrame to a new CSV file
  new_locations = pd.DataFrame(new_locations)
  new_locations.to_csv('./parameter_data/boeing_locations_with_stations.csv', index=False)


def download_street_network_and_select_random_nodes(city_name,min_distance_km=3):
  """
  Downloads the street network graph of a city and selects 3 random nodes that are at least 4 kilometers away from each other.

  Args:
    city_name: The name of the city.

  Returns:
    A list containing 3 random nodes from the city's street network graph.
  """
  try:
    # Download the street network graph for the city
    graph = ox.graph_from_place(city_name, network_type='drive')
    
    # Get all the nodes in the graph
    nodes = list(graph.nodes)
    
    # Function to check if nodes are at least 4 kilometers apart
    def nodes_far_enough(node_list, new_node, min_distance_km=min_distance_km):
      for node in node_list:
        dist = geodesic((graph.nodes[node]['y'], graph.nodes[node]['x']), 
                        (graph.nodes[new_node]['y'], graph.nodes[new_node]['x'])).km
        if dist < min_distance_km:
          return False
      return True
    
    # Select 3 random nodes that are at least 4 kilometers apart
    random_nodes = []
    while len(random_nodes) < 3:
      candidate_node = random.choice(nodes)
      if nodes_far_enough(random_nodes, candidate_node):
        random_nodes.append(candidate_node)
    
    print(f"Random nodes for {city_name} that are at least 4 km apart: {random_nodes}")
    return graph, random_nodes
  
  except Exception as e:
    print(f"An error occurred while processing {city_name}: {e}")
    return None

def get_random_nodes_for_all_cities(origin_locations, min_distance_km=3):
  # Example usage
  locations = []
  for index, row in origin_locations.iterrows():
    city_name = row['city_name']
    try:
      graph, random_nodes = download_street_network_and_select_random_nodes(city_name)
    except Exception as e:
      print(f"An error occurred while processing {city_name}: {e}")
      continue
    if random_nodes:
      print(f"Random nodes for {city_name}: {random_nodes}")
    #time.sleep(1)  # Sleep for 1 second to avoid making too many requests in a short period of time
    if random_nodes:
      city = row['city_name']
      #country = location_info[location_info['city_name'] == city]['country']
      #region = location_info[location_info['city_name'] == city]['region']
      node_samples_latlon = []
      node_samples_ids = []
      for index, node in enumerate(random_nodes):
        print(f"Node: {node}")
        
        node_lat = graph.nodes[node]['y']
        node_lon = graph.nodes[node]['x']
        node_latlon = (node_lat, node_lon)
        print(f"Random node coordinates for {city_name}: {node_lat}, {node_lon}")
        node_samples_latlon.append(node_latlon)
        node_samples_ids.append(node)

    location = {
      'city_name': city,
      'country': row['country'],
      'region': row['region'],
      "network_type":"drive",
      "node1_id": node_samples_ids[0],
      'node1_latlon': node_samples_latlon[0],
      'node2_id': node_samples_ids[1],
      'node2_latlon': node_samples_latlon[1],
      'node3_id': node_samples_ids[2],
      'node3_latlon': node_samples_latlon[2]
    }
    locations.append(location)
    # Save the random nodes DataFrame to a new CSV file
  node_samples_df = pd.DataFrame(locations)
  return node_samples_df 


def get_coord_info(lat, lon, max_retries=3, retry_delay=2):
    """
    Retrieves address information (city, region, country) in English for a given latitude and longitude.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        max_retries: Maximum number of retries if the request times out or fails.
        retry_delay: Delay (in seconds) between retries.

    Returns:
        A dictionary containing the city, region, country, and full address in English, 
        or None if the request fails after multiple retries.
    """

    geolocator = Nominatim(user_agent="train_station_finder")

    for attempt in range(max_retries):
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
            address = location.raw.get('address', {})

            # Get city, considering alternatives if 'city' is missing
            city = address.get('city', 
                               address.get('town', 
                                           address.get('village', 
                                                       address.get('hamlet', ''))))

            # Get the region (state, county, or other administrative division)
            region = address.get('state', 
                                 address.get('county', 
                                             address.get('region', '')))

            country = address.get('country', '')
            full_address = location.address

            return {
                'city': city,
                'region': region,
                'country': country,
                'address': full_address
            }

        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Geocoding failed after multiple retries for coordinates: {lat}, {lon}")
                return None

def fix_unknown(origin_locations):
  for index, row in origin_locations.iterrows():
    city = row['city_name']
    if row['country'] == 'Unknown':
      print(f"City: {city}")
      lat,lon = ast.literal_eval(row['node1_latlon'])
      coord_info = get_coord_info(lat, lon)
      if coord_info:
        print(f"City: {city}, Country: {coord_info['country']}")
        origin_locations.at[index, 'country'] = coord_info['country']
        origin_locations.at[index, 'region'] = coord_info['region']
  origin_locations.to_csv('./parameter_data/boeing_locations_3node_sample_B.csv', index=False)
